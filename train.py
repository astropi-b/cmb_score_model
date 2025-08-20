"""
train.py
--------

Train a score‑based model on primordial B‑mode spectra using either the
variance–exploding (VE) or variance–preserving (VP) SDE.  The script reads
training data from a `.npy` file, applies a logarithmic transform, normalises
the log data to zero mean and unit variance, and trains a `ScoreNet1D`
model using denoising score matching.  The model and training statistics
are saved to disk for use during sampling.

Example usage:

```bash
python train.py \
  --data-file data/dl_train_r0.001.npy \
  --sde-type VE \
  --sigma-min 0.1 --sigma-max 10 \
  --batch-size 128 --epochs 150 --lr 7e-4 \
  --hidden-dim 256 --embed-dim 128 --num-layers 5 \
  --device cuda
```

This trains a model on the specified dataset and writes `model.pth`,
`training_loss.npy`, and `stats.npz` containing the mean and standard
deviation of the log data.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from typing import List

from sde_utils import get_sde, VESDE, VPSDE
from model import create_score_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a score-based model on primordial B-mode spectra")
    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Path to the .npy file containing training data (D_ℓ).",
    )
    parser.add_argument(
        "--sde-type",
        type=str,
        choices=["VE", "VP", "ve", "vp"],
        default="VE",
        help="Type of SDE to use: VE (variance exploding) or VP (variance preserving).",
    )
    parser.add_argument(
        "--sigma-min",
        type=float,
        default=0.1,
        help="Minimum sigma for VE SDE (ignored for VP).",
    )
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=10.0,
        help="Maximum sigma for VE SDE (ignored for VP).",
    )
    parser.add_argument(
        "--beta-min",
        type=float,
        default=0.1,
        help="Minimum beta for VP SDE (ignored for VE).",
    )
    parser.add_argument(
        "--beta-max",
        type=float,
        default=20.0,
        help="Maximum beta for VP SDE (ignored for VE).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of training epochs (default: 150)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=7e-4,
        help="Learning rate for Adam optimizer (default: 7e-4)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Number of units in hidden layers (default: 256)",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=128,
        help="Dimensionality of the time embedding (must be even, default: 128)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=5,
        help="Number of MLP layers including output (default: 5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run training on (e.g. 'cpu', 'cuda', 'mps')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="trained_models",
        help="Directory where model and logs will be saved (default: trained_models)",
    )
    return parser.parse_args()


def train_model(
    data_file: str,
    sde_type: str,
    sde_params: dict,
    batch_size: int,
    epochs: int,
    lr: float,
    hidden_dim: int,
    embed_dim: int,
    num_layers: int,
    device: str | torch.device,
    output_dir: str | Path,
) -> None:
    """Main training routine.

    Parameters
    ----------
    data_file : str
        Path to a numpy file containing the training data (D_ℓ values).  Shape is
        `(num_samples, input_dim)`.
    sde_type : str
        SDE type ("VE" or "VP").
    sde_params : dict
        Parameters required to instantiate the SDE.
    batch_size : int
        Number of samples per training batch.
    epochs : int
        Number of epochs to train for.
    lr : float
        Learning rate for Adam.
    hidden_dim : int
        Hidden layer width.
    embed_dim : int
        Time embedding dimension.
    num_layers : int
        Number of MLP layers.
    device : str or torch.device
        Device on which to perform training.
    output_dir : str or Path
        Directory to save model and logs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading training data from {data_file}...")
    data = np.load(data_file)
    if data.ndim != 2:
        raise ValueError(f"Expected data of shape (num_samples, input_dim), got {data.shape}")
    num_samples, input_dim = data.shape
    print(f"Loaded {num_samples} samples with input dimension {input_dim}")

    # Log-transform the data; add a small epsilon to avoid log(0)
    epsilon = 1e-12
    data_log = np.log(data + epsilon)
    mean = float(data_log.mean())
    std = float(data_log.std())
    data_log_norm = (data_log - mean) / std

    # Save statistics for later use
    stats_file = output_dir / "stats.npz"
    np.savez(stats_file, mean=mean, std=std)
    print(f"Saved normalisation statistics to {stats_file}")

    # Prepare DataLoader
    data_tensor = torch.tensor(data_log_norm, dtype=torch.float32)
    dataset = TensorDataset(data_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate SDE and score model
    sde = get_sde(sde_type, **sde_params)

    # Create wrapper for marginal_prob function to use inside the model
    def marginal_prob_std_fn(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return sde.marginal_prob(t)

    device_torch = torch.device(device)
    model = create_score_model(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        num_hidden_layers=num_layers,
        marginal_prob_std=marginal_prob_std_fn,
        device=device_torch,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Keep track of losses
    epoch_losses: List[float] = []
    eps = 1e-5

    print(f"Starting training for {epochs} epochs on {device_torch}...")
    for epoch in range(epochs):
        total_loss = 0.0
        total_items = 0
        for (x_batch,) in data_loader:
            x_batch = x_batch.to(device_torch)
            batch_size_actual = x_batch.shape[0]
            # Sample random t in (eps, 1)
            random_t = torch.rand(batch_size_actual, device=device_torch) * (1.0 - eps) + eps
            # Standard normal noise
            z = torch.randn_like(x_batch)
            # Perturb data
            mean_coeff, std = sde.marginal_prob(random_t)
            mean_coeff = mean_coeff.to(device_torch)
            std = std.to(device_torch)
            perturbed_x = mean_coeff.unsqueeze(-1) * x_batch + z * std.unsqueeze(-1)
            # Compute score
            score = model(perturbed_x, random_t)
            # Loss: weighted denoising score matching
            loss = torch.mean(torch.sum((score * std.unsqueeze(-1) + z)**2, dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size_actual
            total_items += batch_size_actual
        avg_loss = total_loss / total_items
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, loss = {avg_loss:.6f}")

    # Save the model
    model_file = output_dir / "model.pth"
    torch.save(model.state_dict(), model_file)
    print(f"Saved trained model to {model_file}")

    # Save training loss history
    loss_file = output_dir / "training_loss.npy"
    np.save(loss_file, np.array(epoch_losses))
    print(f"Saved training losses to {loss_file}")

    # Save architecture configuration for sampling
    arch_file = output_dir / "architecture.json"
    arch_config = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "embed_dim": embed_dim,
        "num_layers": num_layers,
        "sde_type": sde_type,
        # Save SDE parameters as well
        "sde_params": sde_params,
    }
    with open(arch_file, "w") as f:
        json.dump(arch_config, f)
    print(f"Saved architecture configuration to {arch_file}")


def main() -> None:
    args = parse_args()
    sde_type = args.sde_type.upper()
    # Collect SDE parameters
    sde_params = {}
    if sde_type == "VE":
        sde_params = {"sigma_min": args.sigma_min, "sigma_max": args.sigma_max}
    elif sde_type == "VP":
        sde_params = {"beta_min": args.beta_min, "beta_max": args.beta_max}
    else:
        raise ValueError(f"Unknown SDE type {args.sde_type}")
    train_model(
        data_file=args.data_file,
        sde_type=sde_type,
        sde_params=sde_params,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()