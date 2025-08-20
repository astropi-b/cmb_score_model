"""
sde_utils.py
-----------------

Helper classes and functions for score‑based generative models built around
stochastic differential equations (SDEs).  Two SDEs are supported:

* Variance–Exploding (VE) SDE, where the forward process has no drift and
  a diffusion term that grows exponentially.  This is the SDE used in the
  original implementation, with parameters `sigma_min` and `sigma_max`.

* Variance–Preserving (VP) SDE, also known as the Ornstein–Uhlenbeck
  process.  It has a drift term that contracts the state and a diffusion
  coefficient controlled by a `beta(t)` schedule.  A linear schedule is
  implemented, with parameters `beta_min` and `beta_max`.

Each class implements the functions required for training and sampling:

* `marginal_prob(t)`: returns `(mean_coeff, std)` so that a point drawn
  from the forward SDE at time `t` can be sampled via

    x_t = mean_coeff * x_0 + std * z

  where `z ~ N(0, I)`.

* `diffusion_coeff(t)`: returns the diffusion coefficient `g(t)` used in
  the reverse SDE.

* `drift(x, t)`: returns the drift term `f(x,t)` of the forward SDE.

The code is written to be used with PyTorch tensors.  All time arguments
should be `torch.Tensor` of shape `(batch,)`.  Devices and dtypes are
propagated automatically from the input.
"""

from __future__ import annotations

import math
import torch


class VESDE:
    """Variance–exploding SDE.

    The forward SDE has no drift and a diffusion coefficient that scales
    multiplicatively with time:

        dX = g(t) dW_t

    with

        g(t) = sigma_min * (sigma_max / sigma_min) ** t * sqrt(2 * log(sigma_max / sigma_min)).

    The marginal distribution at time `t` is

        X_t = X_0 + sigma(t) Z,

    where Z ~ N(0, I) and

        sigma(t) = sigma_min * sqrt((sigma_max / sigma_min) ** (2t) - 1).

    Parameters
    ----------
    sigma_min : float
        Lower bound of the noise scale.
    sigma_max : float
        Upper bound of the noise scale.
    """

    def __init__(self, sigma_min: float = 0.1, sigma_max: float = 10.0) -> None:
        if sigma_min <= 0 or sigma_max <= 0:
            raise ValueError("sigma_min and sigma_max must be positive")
        if sigma_max <= sigma_min:
            raise ValueError("sigma_max must be greater than sigma_min")
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self._log_ratio = math.log(self.sigma_max / self.sigma_min)

    def marginal_prob(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the mean coefficient and standard deviation at time `t`.

        For the VE SDE, the mean coefficient is unity and the standard
        deviation follows

            sigma(t) = sigma_min * sqrt((sigma_max / sigma_min) ** (2t) - 1).

        Parameters
        ----------
        t : torch.Tensor
            A batch of times in [0, 1].

        Returns
        -------
        mean_coeff : torch.Tensor
            A tensor of shape `(batch,)` filled with ones.
        std : torch.Tensor
            A tensor of shape `(batch,)` with the standard deviations.
        """
        # Ensure the tensor is on the correct device/dtype
        t = t.to(dtype=torch.float32)
        ratio = self.sigma_max / self.sigma_min
        exp_term = (ratio ** (2.0 * t)) - 1.0
        std = self.sigma_min * torch.sqrt(exp_term)
        mean_coeff = torch.ones_like(std)
        return mean_coeff, std

    def diffusion_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """Return the diffusion coefficient g(t) for the forward SDE.

        g(t) = sigma_min * (sigma_max / sigma_min)**t * sqrt(2 * log(sigma_max / sigma_min)).

        Parameters
        ----------
        t : torch.Tensor
            A batch of times in [0, 1].

        Returns
        -------
        g : torch.Tensor
            Diffusion coefficient of shape `(batch,)`.
        """
        t = t.to(dtype=torch.float32)
        ratio = self.sigma_max / self.sigma_min
        sigma_t = self.sigma_min * (ratio ** t)
        g = sigma_t * math.sqrt(2.0 * self._log_ratio)
        return g

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward drift term f(x,t).

        The VE SDE has zero drift.
        """
        return torch.zeros_like(x)


class VPSDE:
    """Variance–preserving SDE.

    The forward SDE is

        dX = -0.5 * beta(t) * X dt + sqrt(beta(t)) dW_t,

    with a linear beta schedule

        beta(t) = beta_min + (beta_max - beta_min) * t.

    The marginal distribution at time `t` given X_0 has mean

        mean_coeff(t) = exp(-0.5 * integral_beta(t)),

    and standard deviation

        std(t) = sqrt(1 - exp(-integral_beta(t))),

    where

        integral_beta(t) = beta_min * t + 0.5 * (beta_max - beta_min) * t**2.

    Parameters
    ----------
    beta_min : float
        Initial value of the beta schedule.
    beta_max : float
        Final value of the beta schedule.
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        if beta_min <= 0 or beta_max <= 0:
            raise ValueError("beta_min and beta_max must be positive")
        if beta_max <= beta_min:
            raise ValueError("beta_max must be greater than beta_min")
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        """Return beta(t) for a linear schedule."""
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def _integrated_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Compute ∫_0^t beta(s) ds for the linear schedule.

        integral_beta(t) = beta_min * t + 0.5 * (beta_max - beta_min) * t**2
        """
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t * t

    def marginal_prob(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean_coeff, std) at time t.

        mean_coeff = exp(-0.5 * integral_beta(t))
        std       = sqrt(1 - exp(-integral_beta(t)))

        Parameters
        ----------
        t : torch.Tensor
            A batch of times in [0, 1].

        Returns
        -------
        mean_coeff : torch.Tensor
            Mean scaling factor of shape `(batch,)`.
        std : torch.Tensor
            Standard deviation of shape `(batch,)`.
        """
        t = t.to(dtype=torch.float32)
        int_beta = self._integrated_beta(t)
        mean_coeff = torch.exp(-0.5 * int_beta)
        std = torch.sqrt(1.0 - torch.exp(-int_beta))
        return mean_coeff, std

    def diffusion_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """Return g(t) = sqrt(beta(t))."""
        t = t.to(dtype=torch.float32)
        return torch.sqrt(self._beta(t))

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward drift term f(x,t) = -0.5 * beta(t) * x."""
        beta_t = self._beta(t).unsqueeze(-1)
        return -0.5 * beta_t * x


def get_sde(sde_type: str, **params) -> object:
    """Factory function to construct an SDE object.

    Parameters
    ----------
    sde_type : str
        Either "VE" or "VP" (case insensitive).
    **params : dict
        Parameters passed to the SDE constructor.  For VE this includes
        `sigma_min` and `sigma_max`; for VP it includes `beta_min` and
        `beta_max`.

    Returns
    -------
    sde : VESDE or VPSDE
        An instance of the selected SDE.
    """
    sde_type = sde_type.upper()
    if sde_type == "VE":
        return VESDE(**params)
    elif sde_type == "VP":
        return VPSDE(**params)
    else:
        raise ValueError(f"Unknown sde_type {sde_type}. Choose 'VE' or 'VP'.")