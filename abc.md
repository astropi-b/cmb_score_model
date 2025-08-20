# CMB B‑mode Denoising with Score‑based Diffusion Models

This repository contains a modular implementation of score‑based diffusion
models tailored for Cosmic Microwave Background (CMB) B‑mode analysis. The
code takes inspiration from the user‑provided scripts and extends them in
several directions:

* **Flexible SDEs:** Both variance exploding (VE) and variance preserving (VP)
  stochastic differential equations are supported. This allows experiments with
  different noise schedules such as those used in NCSN++/DDPM models.
* **Multiple sampling schemes:** In addition to the Euler–Maruyama sampler
  provided in the original code, we implement the probability flow ODE and
  predictor–corrector (PC) samplers for deterministic and hybrid denoising.
* **Configurable network architecture:** The hidden layer width, number of
  hidden layers and Fourier embedding dimension are exposed via command line
  flags.
* **Data generation for arbitrary `r` values:** You can generate training and
  test datasets for any tensor‑to‑scalar ratio `r` and choose different
  foreground models through the PySM3 preset strings.
* **Trajectory visualisation:** Optionally record and save the entire
  denoising trajectory to inspect how spectra evolve as the diffusion
  process is reversed.

The overall workflow consists of three stages: data generation, model
training and inference/denoising. Each stage is encapsulated in its own
command line script under the `cmb_sde_repo` package and described below.

## 1. Generating simulated datasets

Before training a model you need to generate training, noisy and ground
truth D_ℓ spectra. These can be produced with the `data_generation.py`
script. The output consists of three NumPy arrays saved to disk:

* `dl_train_r{r}.npy` – purely primordial B‑mode spectra smoothed by a
  Gaussian beam, used for training.
* `dl_noisy_r{r}.npy` – total B‑mode spectra including lensing, noise and
  foregrounds.
* `dl_gt_r{r}.npy` – purely primordial B‑mode spectra matching the noisy
  dataset, used as ground truth during evaluation.

You can also produce diagnostic plots showing a few random spectra and the
mean spectrum of each dataset. The simulation relies on [CAMB](https://camb.readthedocs.io/),
[healpy](https://healpy.readthedocs.io/) and [PySM3](https://pysm3.readthedocs.io/), which must be available in your Python environment.

### Example

```bash
# Generate 10k training samples and 2k test samples for r=0.001 using the
# default foreground model (s3, d1, a2) and NSIDE=16. Save outputs under data/.
python -m cmb_sde_repo.data_generation \
  --r 0.001 --nside 16 --num_train 10000 --num_test 2000 \
  --noise_rms 11.9 --fwhm 24.8 --pysm s3 d1 a2 \
  --outdir data --plots

# This produces:
#  data/dl_train_r0.001.npy  (shape [10000, 30])
#  data/dl_noisy_r0.001.npy  (shape [ 2000, 30])
#  data/dl_gt_r0.001.npy     (shape [ 2000, 30])
#  data/random_spectra_r0.001.png  (optional random spectra plot)
#  data/mean_spectra_r0.001.png    (optional mean spectra plot)
#  data/metadata_r0.001.json       (simulation parameters)
```

To explore other tensor‑to‑scalar ratios simply change `--r`. Different
foreground models can be specified via `--pysm` followed by three preset
strings (e.g. `s0 d0 a0`, `s3 d5 a2`) as documented in PySM3.

## 2. Training a score network

Once the training dataset is ready, train a score network using the
`train.py` script. The network operates on the logarithm of the D_ℓ
spectra and uses the denoising score matching objective. All
normalisation statistics and hyperparameters are stored with the model.

### Example

```bash
# Train a VE SDE model on the r=0.001 dataset.
python -m cmb_sde_repo.train \
  --data data/dl_train_r0.001.npy --sde VE \
  --sigma_min 0.1 --sigma_max 10 \
  --hidden_dim 256 --embed_dim 128 --num_hidden_layers 5 \
  --batch_size 128 --lr 7e-4 --epochs 150 \
  --outdir models --device cuda

# The script prints the loss after each epoch and saves:
#  models/score_model_VE.pth          (model weights and metadata)
#  models/loss_history_VE.json        (list of epoch losses)
```

To train a VP SDE model instead, set `--sde VP` and optionally adjust
`--beta_min` and `--beta_max`. You can also experiment with wider or
deeper networks by modifying `--hidden_dim` and `--num_hidden_layers`.

## 3. Denoising and evaluation

With a trained model you can recover clean spectra from the noisy test
set using the `inference.py` script. Three sampling strategies are
available:

* **euler:** stochastic Euler–Maruyama integration of the reverse SDE,
  identical to the method in the original user code.
* **ode:** deterministic integration of the probability flow ODE. This
  typically yields smoother results at the cost of more computation.
* **pc:** predictor–corrector sampler combining stochastic and
  deterministic steps, controlled by the `--snr` parameter.

All methods run backwards in time from `t=1` to a small `t=eps`. You can
adjust the number of time steps with `--num_steps` (larger values improve
accuracy). The script normalises the input data using the same mean and
standard deviation learned during training.

### Example

```bash
# Denoise the noisy test set using the trained VE model and Euler sampler.
python -m cmb_sde_repo.inference \
  --model models/score_model_VE.pth \
  --noisy data/dl_noisy_r0.001.npy \
  --gt data/dl_gt_r0.001.npy \
  --method euler --batch_size 256 --num_steps 1000 \
  --outdir results --plots

# This produces:
#  results/denoised_euler.npy          (denoised spectra)
#  results/random_comparison_euler.png (random spectra plot)
#  results/mean_comparison_euler.png   (mean spectra plot)
#  results/trajectory/step_*.npy       (if --trajectory is specified)
```

Setting `--trajectory` will record all intermediate spectra for the first
sample in the batch. This can be useful for visualising how the model
gradually removes noise. The trajectory is saved as a series of `.npy`
arrays in the `results/trajectory` directory.

## Extending to other `r` values

All scripts accept `--r` as a flag during data generation, so you can
generate and train on any tensor‑to‑scalar ratio. For each new `r`
value, you should generate a separate dataset, train a dedicated model
and run inference accordingly. The modular design makes it straightforward
to loop over multiple values of `r` and compare performance.

## Requirements

The simulation code relies on several external packages that are **not
installed** in this environment. To run the full pipeline you need:

* `healpy` for Healpix map synthesis and analysis
* `camb` for computing theoretical CMB power spectra
* `pysm3` for Galactic foreground simulations
* `numpy`, `matplotlib` and `torch` (PyTorch)

Install these via `pip` in your own Python environment. GPU support is
optional but recommended for training and sampling on larger datasets.

## Repository structure

```
cmb_sde_repo/
  __init__.py          Package initialisation
  data_generation.py   Data simulation script and helper
  model.py             Score network and SDE utilities
  sampling.py          Sampling/denoising algorithms
  train.py             Training script
  inference.py         Denoising and evaluation script
  utils.py             Plotting and spectrum utilities

data/                  Example directory for generated arrays and plots
models/                Place to store trained models
results/               Place to store denoised outputs and figures
``` 

Feel free to adapt and extend the modules—for example by adding new
foreground models, experimenting with sub‑VP SDEs or implementing more
advanced score network architectures. The scripts are designed to be
self‑contained and easy to modify.
