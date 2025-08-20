# CMB Score-Based Generative Model

This repository provides a complete workflow for generating Cosmic Microwave Background (CMB) B‑mode power spectra, training score‑based generative models on those spectra and performing denoising via stochastic differential equations (SDEs) and their deterministic counterparts.  The focus is on low‑resolution B‑mode spectra computed for different values of the primordial tensor–to–scalar ratio `r`.  Two classes of SDEs are implemented: the **variance–exploding (VE) SDE** used in the initial prototype, and the **variance–preserving (VP) SDE**.  Sample generation can be performed with a straightforward Euler–Maruyama solver, a probability–flow ODE solver, and a predictor–corrector (PC) method that refines trajectories using Langevin dynamics.

The codebase is modular.  Training and sampling scripts are separate from data generation, and SDE parameters can be configured from the command line.  The model architecture is a time‑dependent multi‑layer perceptron (MLP) with random Fourier feature time embeddings, as in the original implementation.

## Directory layout

```
cmb_score_model/
│
├── data_generation.py   # Scripts to generate training/testing datasets for arbitrary r
├── model.py             # ScoreNet1D architecture definition
├── sde_utils.py         # SDE helper classes for VE and VP processes
├── train.py             # Train a score model on a specified dataset
├── sample.py            # Perform denoising with SDE, ODE or predictor–corrector samplers
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Requirements

The scripts require Python 3.8+ and the following packages:

* `numpy`
* `torch` (PyTorch ≥1.12)
* `matplotlib`
* `healpy` – for spherical harmonic transforms on the sphere
* `camb` – to compute the CMB angular power spectra
* `pysm3` – to synthesise Galactic foregrounds
* `jupyter` (optional) – for interactive exploration

Install the dependencies with pip:

```bash
pip install -r requirements.txt
```

Note that `healpy`, `camb` and `pysm3` rely on compiled libraries; installation may take a few minutes and may require a C compiler.  If you encounter issues building these packages, consult their respective documentation.

## Dataset generation

Training and testing data are generated from simulated B‑mode angular power spectra.  The training set consists of primordial B‑mode power spectra filtered by an instrument beam with full–width at half–maximum (FWHM) of 24.8 arcminutes.  The test set contains “observational” spectra built from the sum of primordial B‑modes, gravitational lensing, Galactic foregrounds and pixel noise.  The data live in the multipole range ℓ ∈ [2, 2 NSIDE] and are expressed as `D_ℓ = ℓ(ℓ+1) C_ℓ/(2π)`; each sample is therefore a 1D array of length `2⋅NSIDE−2`.

To generate data for a particular tensor–to–scalar ratio `r`, run

```bash
python data_generation.py --r 0.001 --nside 16 --num-samples 10000 --out-dir data
```

This writes three NumPy files into `data/`:

* `dl_train_r0.001.npy` – a `(num_samples, 2*nside-2)` array of primordial `D_ℓ` values;
* `dl_noisy_r0.001.npy` – the corresponding noisy observational spectra;
* `dl_gt_r0.001.npy` – the ground–truth primordial spectra (after beam convolution) used for evaluation.

You can pass multiple `--r` values at once (e.g. `--r 0.001 0.005 0.01`) to generate several datasets in one call.

### Caveats

* Generating large datasets is computationally expensive.  The number of samples (`num-samples`) should be chosen based on available CPU time and memory.  A few thousand samples are sufficient for testing the modelling pipeline.
* `camb` uses the **tensor–to–scalar ratio** parameter `r` at pivot scale `k_pivot=0.05/Mpc`.  The arguments in `data_generation.py` are passed directly to CAMB.
* Foreground simulations come from [PySM3](https://github.com/pysm-project/pysm) presets.  You can modify the preset strings or the observing frequency via command line arguments.

## Training a score model

Once data are available, a score model can be trained.  For example, to train on the `r=0.001` dataset using a VE SDE:

```bash
python train.py \
  --data-file data/dl_train_r0.001.npy \
  --sde-type VE \
  --sigma-min 0.1 --sigma-max 10 \
  --batch-size 128 \
  --epochs 150 \
  --lr 7e-4 \
  --hidden-dim 256 --embed-dim 128 --num-layers 5 \
  --device cpu
```

The script computes a log transform of the `D_ℓ` values, normalises them to zero mean and unit variance, and trains a `ScoreNet1D` on those normalised logs using denoising score matching.  The model weights are saved to `model.pth`, and a log of training losses is written to `training_loss.npy`.  You can adjust the MLP width (`--hidden-dim`), the number of time embedding dimensions (`--embed-dim`) and the number of hidden layers (`--num-layers`).

For a VP SDE, specify `--sde-type VP` and set the β–schedule parameters, e.g. `--beta-min 0.1 --beta-max 20`.  The training loop automatically adapts the noise perturbation to the chosen SDE.

## Denoising and sampling

After training, denoise observational data or generate new samples using `sample.py`.  The script supports three solver types:

* **sde** – stochastic reverse process solved with the Euler–Maruyama method;
* **ode** – deterministic probability–flow ODE;
* **pc** – predictor–corrector sampler: an Euler–Maruyama step followed by one or more Langevin corrector steps.

To denoise a noisy dataset, run for example

```bash
python sample.py \
  --data-file-noisy data/dl_noisy_r0.001.npy \
  --data-file-gt data/dl_gt_r0.001.npy \
  --model-checkpoint model.pth \
  --sde-type VE \
  --sigma-min 0.1 --sigma-max 10 \
  --method sde \
  --num-steps 1000 \
  --device cpu
```

This loads the trained model and the noisy and ground‑truth datasets, normalises the noisy data using the same statistics as were computed during training, runs the reverse process to recover `D_ℓ` values at `t=0`, exponentiates to undo the log transform and writes `denoised.npy`.  The script also prints basic metrics (mean squared error) and plots a few example spectra.

For ODE sampling, set `--method ode`.  For predictor–corrector sampling, set `--method pc` and optionally adjust `--corrector-steps` and `--corrector-step-size`.

## Plotting trajectories

`sample.py` can also visualise the reverse trajectories of individual spectra.  With `--plot-trajectories`, the script stores the intermediate states of the reverse process and generates a plot showing how each dimension of a chosen spectrum evolves from noise back to the denoised state.  This is helpful for understanding the qualitative behaviour of different solvers and SDE types.

## Testing different r values

To explore the impact of different tensor–to–scalar ratios, repeat the data generation and training steps for each `r` value of interest.  Since the data dimensionality (`2⋅NSIDE−2`) is fixed, the same model architecture can be reused.  The sampling scripts accept the same SDE parameters but should be run with the corresponding model checkpoint and datasets.

## Notes on Theory

**Variance–exploding SDE (VE):**

$$
\mathrm{d}\mathbf{x} = g(t)\,\mathrm{d}W_t\,,
$$

where the diffusion coefficient 

$$
g(t) = \sigma_{\min}\, \left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^t \sqrt{2\,\log\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)}
$$

The marginal distribution at time $t \in [0,1]$ is

$$
\mathbf{x}_t = \mathbf{x}_0 + \sigma(t)\,\mathbf{z}
$$

with

$$
\sigma(t) = \sigma_{\min} \sqrt{\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^{2t} - 1}
$$

and $\mathbf{z} \sim \mathcal{N}(0,I)$.

In the reverse process, the drift term is

$$
g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})
$$

The corresponding probability–flow ODE replaces this drift with

$$
-\frac{1}{2}g(t)^2 \nabla_{\mathbf{x}}\log p_t(\mathbf{x})
$$

---

**Variance–preserving SDE (VP):**

$$
\mathrm{d}\mathbf{x} = -\frac{1}{2}\beta(t)\,\mathbf{x}\,\mathrm{d}t + \sqrt{\beta(t)}\,\mathrm{d}W_t\,,
$$

with a scalar noise schedule $\beta(t)$. A common choice is a linear schedule:

$$
\beta(t)=\beta_{\min} + (\beta_{\max}-\beta_{\min})t
$$

The marginal distribution of $\mathbf{x}_t$ given $\mathbf{x}_0$ has mean

$$
\exp\left(-\frac{1}{2}\int_0^t \beta(s)\,\mathrm{d}s\right)\,\mathbf{x}_0
$$

and standard deviation

$$
\sqrt{1 - \exp\left(-\int_0^t \beta(s)\,\mathrm{d}s\right)}
$$

The reverse SDE includes both the negative drift $-\frac{1}{2}\beta(t)\mathbf{x}$ and a score–based term $-\beta(t)\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$. The probability–flow ODE uses

$$
-\frac{1}{2}\beta(t)\left[\mathbf{x} + \nabla_{\mathbf{x}}\log p_t(\mathbf{x})\right]
$$

for its drift.

---

These formulations follow the general framework of **score–based generative modelling through stochastic differential equations**. The provided implementations keep the mathematical details transparent so that users can adapt them for other datasets or noise schedules.

## Contact

If you have questions or encounter issues when using this code, please open an issue in the repository.  Contributions and improvements are welcome!
