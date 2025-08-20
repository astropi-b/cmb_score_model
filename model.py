"""
model.py
---------

Definition of the time‑dependent score model used throughout this repository.
The core component is `ScoreNet1D`, an MLP that takes as input a 1‑D vector
representing a single power spectrum sample and a scalar time step, and
outputs the score function (gradient of the log probability density) for that
sample at that time.  A random Fourier feature embedding of time is used to
encourage the network to learn smooth time dependencies.

The model exposes two key functions:

* The constructor `ScoreNet1D` which instantiates the MLP with
  configurable dimensions and activation functions.

* A helper function `create_score_model` that returns a model already
  initialised on the desired device with optional parameter overrides.

The implementation follows the architecture from the provided baseline code.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class GaussianFourierProjection(nn.Module):
    """Project scalar time `t` into a high‑dimensional embedding using random Fourier features.

    Each input `t` is mapped to a vector of dimension `embed_dim` via sinusoidal
    functions with random frequencies.  The weights are initialised once and
    remain fixed during training, effectively encoding the continuous time
    information as periodic patterns.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the time embedding.  Must be even.
    scale : float, optional
        Scale factor applied to the random frequencies.  Larger scales lead
        to higher frequency oscillations in the time embedding.
    """

    def __init__(self, embed_dim: int, scale: float = 30.0) -> None:
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("Time embedding dimension must be even.")
        self.embed_dim = embed_dim
        # Random frequencies drawn from N(0, scale^2)
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the Fourier features for a batch of time steps.

        Parameters
        ----------
        t : torch.Tensor
            A tensor of shape `(batch,)` containing time values in [0, 1].

        Returns
        -------
        features : torch.Tensor
            A tensor of shape `(batch, embed_dim)` containing the concatenated
            sine and cosine features.
        """
        # Expand t to shape (batch, embed_dim//2) for broadcasting
        x_proj = t[:, None] * self.W[None, :] * 2.0 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ScoreNet1D(nn.Module):
    """Time‑dependent score network for 1D data.

    The network takes an input vector `x` of dimension `input_dim` and a
    scalar time `t`, projects `t` into a high‑dimensional embedding, concatenates
    this with `x`, and feeds the result through a multi‑layer perceptron with
    Swish activations.  The output has the same dimensionality as `x` and
    approximates the score function \(\nabla_x \log p_t(x)\) divided by the
    standard deviation of the forward process at time `t`.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input data (e.g. 30 for NSIDE=16).
    hidden_dim : int
        Number of units in each hidden layer.
    embed_dim : int
        Size of the time embedding.  Must be even.
    num_hidden_layers : int
        Total number of linear layers in the MLP, including the output
        layer.  There will be `num_hidden_layers-1` hidden layers.
    marginal_prob_std : callable
        A function taking a tensor of times and returning `(mean, std)` for
        the forward SDE.  This is used to normalise the output by the
        standard deviation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embed_dim: int,
        num_hidden_layers: int,
        marginal_prob_std: callable,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_hidden_layers = num_hidden_layers
        self.marginal_prob_std = marginal_prob_std

        # Time embedding network
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # Input layer: concatenated data and time embedding
        self.input_layer = nn.Linear(input_dim + embed_dim, hidden_dim)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers - 2):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)

        # Activation: swish
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the network on a batch of data.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(batch, input_dim)`.
        t : torch.Tensor
            Time tensor of shape `(batch,)` with values in [0, 1].

        Returns
        -------
        score : torch.Tensor
            Output tensor of shape `(batch, input_dim)` approximating
            \(\nabla_x \log p_t(x)\).  The output is divided by the
            standard deviation of the forward process at time `t`.
        """
        # Compute time embedding
        temb = self.time_embed(t)
        temb = self.act(temb)

        # Concatenate input and time embedding
        h = torch.cat([x, temb], dim=1)

        # Hidden layers with swish activation
        h = self.act(self.input_layer(h))
        for layer in self.hidden_layers:
            h = self.act(layer(h))

        # Output layer
        h = self.output_layer(h)

        # Divide by standard deviation to ensure scale invariance
        # marginal_prob_std returns (mean, std)
        _, std = self.marginal_prob_std(t)
        h = h / std.unsqueeze(-1)
        return h


def create_score_model(
    input_dim: int,
    hidden_dim: int = 256,
    embed_dim: int = 128,
    num_hidden_layers: int = 5,
    marginal_prob_std: callable | None = None,
    device: str | torch.device = "cpu",
) -> ScoreNet1D:
    """Instantiate a ScoreNet1D on the specified device.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the data.
    hidden_dim : int, optional
        Size of hidden layers.
    embed_dim : int, optional
        Size of the time embedding.  Must be even.
    num_hidden_layers : int, optional
        Number of layers in the MLP.
    marginal_prob_std : callable, optional
        Function returning `(mean, std)` for the forward SDE.  This must be
        provided; see `sde_utils`.
    device : str or torch.device, optional
        Device on which to allocate the network parameters.

    Returns
    -------
    model : ScoreNet1D
        A score model ready for training or inference.
    """
    if marginal_prob_std is None:
        raise ValueError("marginal_prob_std must be provided to create the score model")
    model = ScoreNet1D(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        num_hidden_layers=num_hidden_layers,
        marginal_prob_std=marginal_prob_std,
    )
    return model.to(device)