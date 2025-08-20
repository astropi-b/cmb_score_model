"""
sample.py
---------

Perform denoising and sampling using a trained score model.  The script
supports three solvers:

* **sde** – stochastic reverse process solved with the Euler–Maruyama method;
* **ode** – deterministic probability–flow ODE;
* **pc** – predictor–corrector sampler, combining an Euler step with Langevin
  corrections.

Given a noisy dataset and a trained model checkpoint, the script applies
the chosen solver to each spectrum, writes the denoised result to disk and
optionally plots example spectra and trajectory visualisations.  When the
ground truth is provided, basic error metrics are reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from sde_utils import get_sde
from model import create_score_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Denoise noisy B-mode spectra using a trained score model")
    parser.add_argument(
        "--data-file-noisy",
        type=str,
        required=True,
        help="Path to .npy file containing the noisy D_ℓ spectra to denoise",
    )
    parser.add_argument(
        "--data-file-gt",
        type=str,
        default=None,
        help="Optional path to .npy file with ground truth primordial spectra (for evaluation)",
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        required=True,
        help="Path to .npz file with mean and std used during training (stats.npz)",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (model.pth)",
    )
    parser.add_argument(
        "--sde-type",
        type=str,
        choices=["VE", "VP", "ve", "vp"],
        default="VE",
        help="SDE type used during training (VE or VP)",
    )
    parser.add_argument(
        "--sigma-min",
        type=float,
        default=0.1,
        help="Sigma_min for VE SDE (ignored for VP)",
    )
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=10.0,
        help="Sigma_max for VE SDE (ignored for VP)",
    )
    parser.add_argument(
        "--beta-min",
        type=float,
        default=0.1,
        help="Beta_min for VP SDE (ignored for VE)",
    )
    parser.add_argument(
        "--beta-max",
        type=float,
        default=20.0,
        help="Beta_max for VP SDE (ignored for VE)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["sde", "ode", "pc"],
        default="sde",
        help="Solver to use: sde (Euler-Maruyama), ode (probability flow ODE), or pc (predictor-corrector)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1000,
        help="Number of discretisation steps for the solver (default: 1000)",
    )
    parser.add_argument(
        "--corrector-steps",
        type=int,
        default=1,
        help="Number of Langevin correction steps per solver step when using PC (default: 1)",
    )
    parser.add_argument(
        "--corrector-step-size",
        type=float,
        default=1e-3,
        help="Step size for Langevin corrections in PC sampling (default: 1e-3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to perform sampling on (e.g. 'cpu', 'cuda', 'mps')",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="denoised.npy",
        help="File name to save the denoised spectra (default: denoised.npy)",
    )
    parser.add_argument(
        "--plot-examples",
        action="store_true",
        help="If set, plot random example spectra before and after denoising",
    )
    parser.add_argument(
        "--plot-trajectories",
        action="store_true",
        help="If set, record intermediate states and plot sample trajectories",
    )
    return parser.parse_args()


def load_data(data_file: str) -> np.ndarray:
    data = np.load(data_file)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.shape}")
    return data


def load_stats(stats_file: str) -> Tuple[float, float]:
    stats = np.load(stats_file)
    if 'mean' not in stats or 'std' not in stats:
        raise ValueError("stats.npz must contain 'mean' and 'std'")
    return float(stats['mean']), float(stats['std'])


def load_model(
    model_checkpoint: str,
    sde_type: str,
    sde_params: dict,
    input_dim: int,
    hidden_dim: int,
    embed_dim: int,
    num_layers: int,
    device: str | torch.device,
) -> Tuple[object, object]:
    """Load a trained ScoreNet and its associated SDE.

    Parameters
    ----------
    model_checkpoint : str
        Path to the saved PyTorch model state dictionary.
    sde_type : str
        'VE' or 'VP' matching the training.
    sde_params : dict
        Parameters for the SDE constructor.
    input_dim : int
        Dimensionality of the data.
    hidden_dim : int
        Hidden layer width used during training.
    embed_dim : int
        Time embedding dimension used during training.
    num_layers : int
        Number of MLP layers.
    device : str or torch.device
        Device to allocate the model.

    Returns
    -------
    model : torch.nn.Module
        Loaded score model.
    sde : object
        SDE instance used for sampling.
    """
    sde = get_sde(sde_type, **sde_params)

    # marginal_prob wrapper for the model
    def marginal_prob_std_fn(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return sde.marginal_prob(t)

    model = create_score_model(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        num_hidden_layers=num_layers,
        marginal_prob_std=marginal_prob_std_fn,
        device=device,
    )
    # Load state dict
    state_dict = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, sde


def euler_maruyama_denoiser(
    model: torch.nn.Module,
    x_init: torch.Tensor,
    sde: object,
    num_steps: int,
    device: torch.device,
    eps: float = 1e-5,
    return_trajectory: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Denoise data using the Euler–Maruyama solver on the reverse SDE.

    Parameters
    ----------
    model : torch.nn.Module
        Trained score model.
    x_init : torch.Tensor
        Normalised log‑data tensor of shape `(batch, input_dim)`.
    sde : object
        SDE instance exposing `drift`, `diffusion_coeff` and `marginal_prob`.
    num_steps : int
        Number of time steps to discretise the interval [0,1].
    device : torch.device
        Device on which to run the solver.
    eps : float, optional
        Small positive value; integration is performed from t=1 down to t=eps.
    return_trajectory : bool, optional
        If True, return the intermediate states for each sample (used for plotting trajectories).

    Returns
    -------
    x_final : torch.Tensor
        Approximation to the clean log‑data at t=0.
    trajectory : torch.Tensor or None
        If `return_trajectory` is True, returns a tensor of shape `(num_steps, batch, input_dim)`
        storing x at each time step; otherwise returns None.
    """
    x = x_init.clone().to(device)
    batch_size = x.shape[0]
    time_steps = torch.linspace(1.0, eps, num_steps, device=device)
    dt = time_steps[0] - time_steps[1]  # negative
    if return_trajectory:
        traj = torch.zeros((num_steps, batch_size, x.shape[1]), device=device)
    for i, t in enumerate(time_steps):
        t_batch = torch.ones(batch_size, device=device) * t
        f = sde.drift(x, t_batch)  # shape (batch, input_dim)
        g = sde.diffusion_coeff(t_batch)  # shape (batch,)
        # Score prediction
        score = model(x, t_batch)
        # Reverse SDE drift: f(x,t) - g(t)^2 * score
        drift = f - (g.unsqueeze(-1)**2) * score
        x_mean = x + drift * dt
        # Stochasticity
        noise = torch.randn_like(x)
        x = x_mean + g.unsqueeze(-1) * torch.sqrt(torch.abs(dt)) * noise
        if return_trajectory:
            traj[i] = x.detach()
    return (x_mean, traj) if return_trajectory else (x_mean, None)


def probability_flow_ode_denoiser(
    model: torch.nn.Module,
    x_init: torch.Tensor,
    sde: object,
    num_steps: int,
    device: torch.device,
    eps: float = 1e-5,
    return_trajectory: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Denoise data using the probability–flow ODE (deterministic sampler).

    Parameters are analogous to `euler_maruyama_denoiser`.  The ODE updates
    
        dx = [f(x,t) - 0.5 g(t)^2 * score(x,t)] dt.
    """
    x = x_init.clone().to(device)
    batch_size = x.shape[0]
    time_steps = torch.linspace(1.0, eps, num_steps, device=device)
    dt = time_steps[0] - time_steps[1]
    if return_trajectory:
        traj = torch.zeros((num_steps, batch_size, x.shape[1]), device=device)
    for i, t in enumerate(time_steps):
        t_batch = torch.ones(batch_size, device=device) * t
        f = sde.drift(x, t_batch)
        g = sde.diffusion_coeff(t_batch)
        score = model(x, t_batch)
        drift = f - 0.5 * (g.unsqueeze(-1)**2) * score
        x = x + drift * dt
        if return_trajectory:
            traj[i] = x.detach()
    return (x, traj) if return_trajectory else (x, None)


def predictor_corrector_denoiser(
    model: torch.nn.Module,
    x_init: torch.Tensor,
    sde: object,
    num_steps: int,
    corrector_steps: int,
    corrector_step_size: float,
    device: torch.device,
    eps: float = 1e-5,
    return_trajectory: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Denoise data using the predictor–corrector sampler.

    At each time step, a predictor step (Euler–Maruyama) is followed by
    `corrector_steps` Langevin iterations on the current marginal distribution.
    The Langevin step updates are simple gradient ascent on the score field
    with Gaussian noise:

        x <- x + step_size * score(x,t) + sqrt(2 * step_size) * z.

    The step size can be tuned via `corrector_step_size`.
    """
    x = x_init.clone().to(device)
    batch_size = x.shape[0]
    time_steps = torch.linspace(1.0, eps, num_steps, device=device)
    dt = time_steps[0] - time_steps[1]
    if return_trajectory:
        traj = torch.zeros((num_steps, batch_size, x.shape[1]), device=device)
    for i, t in enumerate(time_steps):
        t_batch = torch.ones(batch_size, device=device) * t
        f = sde.drift(x, t_batch)
        g = sde.diffusion_coeff(t_batch)
        score = model(x, t_batch)
        # Predictor (Euler–Maruyama)
        drift = f - (g.unsqueeze(-1)**2) * score
        x_mean = x + drift * dt
        noise_pred = torch.randn_like(x)
        x = x_mean + g.unsqueeze(-1) * torch.sqrt(torch.abs(dt)) * noise_pred
        # Corrector (Langevin dynamics)
        for _ in range(corrector_steps):
            # Resample noise each corrector step
            z = torch.randn_like(x)
            score_corr = model(x, t_batch)
            x = x + corrector_step_size * score_corr + torch.sqrt(2.0 * corrector_step_size) * z
        if return_trajectory:
            traj[i] = x.detach()
    return (x_mean, traj) if return_trajectory else (x_mean, None)


def denoise_dataset(
    model: torch.nn.Module,
    sde: object,
    noisy_data: np.ndarray,
    mean: float,
    std: float,
    method: str,
    num_steps: int,
    corrector_steps: int,
    corrector_step_size: float,
    device: torch.device,
    record_trajectories: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Denoise an entire dataset using the specified solver.

    The noisy data are log‑transformed and normalised using the provided
    `mean` and `std`, passed through the reverse process, then exponentiated
    back to the original scale.  Optionally record the trajectories of one
    sample per batch.

    Returns
    -------
    denoised : np.ndarray
        Denoised data of shape `(num_samples, input_dim)`.
    traj_out : np.ndarray or None
        If `record_trajectories` is True, returns a tensor of shape
        `(num_steps, input_dim)` recording the trajectory of the first sample in
        the dataset; otherwise returns None.
    """
    eps = 1e-12
    # Log-transform and normalise
    noisy_log = np.log(noisy_data + eps)
    noisy_norm = (noisy_log - mean) / std
    x_tensor = torch.tensor(noisy_norm, dtype=torch.float32, device=device)
    num_samples, input_dim = x_tensor.shape
    denoised_list = []
    traj_list = []
    # We process in mini-batches to avoid memory issues
    batch_size = 256  # internal batch size for denoising
    solver_func = None
    if method == 'sde':
        solver_func = euler_maruyama_denoiser
    elif method == 'ode':
        solver_func = probability_flow_ode_denoiser
    elif method == 'pc':
        solver_func = predictor_corrector_denoiser
    else:
        raise ValueError(f"Unknown method {method}")
    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            x_batch = x_tensor[start:end]
            if method == 'sde':
                x_out, traj = solver_func(
                    model, x_batch, sde, num_steps=num_steps, device=device, return_trajectory=record_trajectories
                )
            elif method == 'ode':
                x_out, traj = solver_func(
                    model, x_batch, sde, num_steps=num_steps, device=device, return_trajectory=record_trajectories
                )
            elif method == 'pc':
                x_out, traj = solver_func(
                    model, x_batch, sde,
                    num_steps=num_steps,
                    corrector_steps=corrector_steps,
                    corrector_step_size=corrector_step_size,
                    device=device,
                    return_trajectory=record_trajectories
                )
            # Un-normalise and exp to get D_ell
            x_out = x_out * std + mean
            denoised_batch = torch.exp(x_out).cpu().numpy()
            denoised_list.append(denoised_batch)
            if record_trajectories and traj is not None:
                # Record only the first element of the batch
                traj_list.append(traj[:, 0].cpu().numpy())
    denoised = np.vstack(denoised_list)
    # For trajectories, average across batches (should only record one per batch)
    traj_out = None
    if record_trajectories and traj_list:
        # traj_list is a list of arrays of shape (num_steps, input_dim)
        # Concatenate along a new axis representing different batches and average
        traj_arr = np.stack(traj_list, axis=1)  # shape (num_steps, num_batches, input_dim)
        traj_out = traj_arr[:, 0, :]  # pick the first recorded trajectory
    return denoised, traj_out


def main() -> None:
    args = parse_args()
    sde_type = args.sde_type.upper()
    # Build SDE parameters
    if sde_type == 'VE':
        sde_params = {"sigma_min": args.sigma_min, "sigma_max": args.sigma_max}
    else:
        sde_params = {"beta_min": args.beta_min, "beta_max": args.beta_max}

    # Load data and stats
    noisy_data = load_data(args.data_file_noisy)
    mean, std = load_stats(args.stats_file)
    input_dim = noisy_data.shape[1]

    # If ground truth provided, load
    gt_data = None
    if args.data_file_gt is not None and Path(args.data_file_gt).exists():
        gt_data = load_data(args.data_file_gt)
        if gt_data.shape != noisy_data.shape:
            raise ValueError("Ground truth and noisy data shapes do not match")

    # For loading the model, we need the architecture parameters.  These are
    # stored in a JSON file next to the model checkpoint if available.  If
    # absent, default to common values and hope the user matches them.
    model_dir = Path(args.model_checkpoint).resolve().parent
    arch_file = model_dir / "architecture.json"
    if arch_file.exists():
        with open(arch_file, "r") as f:
            arch = json.load(f)
        hidden_dim = arch.get("hidden_dim", 256)
        embed_dim = arch.get("embed_dim", 128)
        num_layers = arch.get("num_layers", 5)
    else:
        # Heuristics: embed_dim must be even; assume typical configuration
        hidden_dim = 256
        embed_dim = 128
        num_layers = 5
        print("Warning: architecture.json not found. Using default model hyperparameters")

    device = torch.device(args.device)
    model, sde = load_model(
        model_checkpoint=args.model_checkpoint,
        sde_type=sde_type,
        sde_params=sde_params,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        device=device,
    )

    # Denoise dataset
    print(f"Starting denoising on {args.device} with method {args.method}...")
    record_traj = args.plot_trajectories
    denoised, traj = denoise_dataset(
        model=model,
        sde=sde,
        noisy_data=noisy_data,
        mean=mean,
        std=std,
        method=args.method,
        num_steps=args.num_steps,
        corrector_steps=args.corrector_steps,
        corrector_step_size=args.corrector_step_size,
        device=device,
        record_trajectories=record_traj,
    )
    # Save output
    out_file = Path(args.output_file)
    np.save(out_file, denoised)
    print(f"Saved denoised spectra to {out_file}")

    # If GT provided, evaluate MSE
    if gt_data is not None:
        mse = np.mean((denoised - gt_data)**2)
        print(f"Mean squared error: {mse:.6e}")

    # Plot examples if requested
    if args.plot_examples:
        import matplotlib.pyplot as plt
        num_examples = min(9, denoised.shape[0])
        indices = np.random.choice(denoised.shape[0], size=num_examples, replace=False)
        ncols = 3
        nrows = int(np.ceil(num_examples / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows), sharey=True)
        axes = axes.flat if hasattr(axes, 'flat') else [axes]
        for ax, idx in zip(axes, indices):
            ax.plot(noisy_data[idx], label='Noisy', linestyle='-', marker='o', markersize=3, alpha=0.5)
            ax.plot(denoised[idx], label='Denoised', linestyle='-', marker='^', markersize=3, alpha=0.7)
            if gt_data is not None:
                ax.plot(gt_data[idx], label='Ground Truth', linestyle='-', marker='s', markersize=3, alpha=0.7)
            ax.set_title(f"Sample {idx}")
            ax.set_xlabel('Multipole bin index')
        axes[0].legend()
        plt.tight_layout()
        plt.yscale('log')
        plt.show()

    # Plot trajectories if requested
    if record_traj and traj is not None:
        import matplotlib.pyplot as plt
        # traj has shape (num_steps, input_dim)
        fig, ax = plt.subplots(figsize=(10, 6))
        time_axis = np.linspace(1.0, 0.0, args.num_steps)
        for d in range(traj.shape[1]):
            ax.plot(time_axis, traj[:, d], alpha=0.3)
        ax.set_title('Evolution of sample dimensions over time')
        ax.set_xlabel('Normalised time step (t from 1 to 0)')
        ax.set_ylabel('Normalised log-amplitude')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()