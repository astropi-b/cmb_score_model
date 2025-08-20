"""
data_generation.py
------------------

Generate training and testing datasets of B‑mode power spectra for score‑based
generative modelling.  This script wraps the physics code provided in the
original prototype into convenient functions and a command‑line interface.

Two types of datasets are produced:

* **Primordial**: purely tensor B‑mode spectra filtered by an instrument
  beam.  These are used for training the score model.
* **Observational**: total B‑mode spectra including primordial modes,
  gravitational lensing, Galactic foregrounds and white noise.  These
  constitute the noisy data that the score model must denoise.  The
  corresponding beam‑convolved primordial spectra are also saved to allow
  evaluation of the denoising performance.

The function `generate_datasets` can be called from Python, while the
command‑line interface is convenient for batch generation of multiple r
values.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

try:
    import camb
except ImportError as e:
    raise ImportError(
        "CAMB is required for dataset generation. Please install it via `pip install camb`.")

try:
    import healpy as hp
except ImportError as e:
    raise ImportError(
        "healpy is required for dataset generation. Please install it via `pip install healpy`.")

try:
    import pysm3
    import pysm3.units as u
except ImportError as e:
    raise ImportError(
        "PySM3 is required for dataset generation. Please install it via `pip install pysm3`.")


def cl2dl(cl: np.ndarray, ell: np.ndarray) -> np.ndarray:
    """Convert C_ℓ to D_ℓ = ℓ(ℓ+1) C_ℓ/(2π).

    Parameters
    ----------
    cl : np.ndarray
        Array of C_ℓ values starting from ℓ=2.
    ell : np.ndarray
        Corresponding multipole values.

    Returns
    -------
    d_ell : np.ndarray
        The D_ℓ values.
    """
    return cl * ell * (ell + 1) / (2.0 * np.pi)


def generate_primordial_dataset(
    r: float,
    nside: int,
    num_samples: int,
    fwhm_arcmin: float,
    seed_offset: int = 1,
    ) -> np.ndarray:
    """Simulate primordial B‑mode spectra for a given tensor–to–scalar ratio.

    The pipeline follows the steps in the provided prototype:
    1. Compute the primordial tensor C_ℓ^BB using CAMB for the specified r.
    2. Build a Gaussian beam transfer function with FWHM `fwhm_arcmin`.
    3. Draw `num_samples` random realisations of the B‑mode map with `healpy.synfast`.
    4. Deconvolve the map to obtain C_ℓ estimates and apply the beam filter.
    5. Convert to D_ℓ and return the resulting array of shape `(num_samples, 2*nside-2)`.

    Parameters
    ----------
    r : float
        Tensor–to–scalar ratio used in CAMB.
    nside : int
        HEALPix NSIDE parameter.  The maximum multipole is 2*nside-1.
    num_samples : int
        Number of realisations to generate.
    fwhm_arcmin : float
        Full width at half maximum of the beam in arcminutes.
    seed_offset : int, optional
        Offset applied to the random seed of each sample to ensure reproducibility.

    Returns
    -------
    dl_train : np.ndarray
        Array of shape `(num_samples, 2*nside-2)` containing D_ℓ spectra.
    """
    lmax = 2 * nside - 1

    # Set up CAMB parameters for tensor modes
    pars = camb.CAMBparams(WantTensors=True)
    pars.set_cosmology(H0=67.32, ombh2=0.0224, omch2=0.1201, Alens=1)
    pars.InitPower.set_params(ns=0.966, r=r)
    reion_model = camb.reionization.TanhReionization()
    reion_model.set_tau(0.05431)
    pars.Reion = reion_model
    pars.Reion.use_optical_depth = True
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
    cl_bb_tensor = powers['tensor'][:, 2]

    # Beam transfer function
    fwhm_rad = (fwhm_arcmin / 60.0) * np.pi / 180.0
    sigma_b = fwhm_rad / np.sqrt(8.0 * np.log(2.0))
    ell = np.arange(lmax + 1)
    b_ell = np.exp(-ell * (ell + 1) * sigma_b**2 / 2.0)
    bl2 = b_ell**2

    # Generate realisations
    dl_train: List[np.ndarray] = []
    for i in range(num_samples):
        seed = seed_offset + i
        np.random.seed(seed)
        # Generate primordial map
        m = hp.synfast(cl_bb_tensor, nside=nside, pol=False)
        # Compute power spectrum and apply beam
        cl_ob = hp.anafast(m, lmax=lmax)
        cl_beam = cl_ob * bl2
        # Convert to D_ell (skip ell=0,1)
        d_ell = cl2dl(cl_beam[2:], ell[2:])
        dl_train.append(d_ell)
    return np.stack(dl_train, axis=0)


def generate_observational_dataset(
    r: float,
    nside: int,
    num_samples: int,
    fwhm_arcmin: float,
    noise_rms_uK: float,
    foreground_presets: Iterable[str] = ("s3", "d1", "a2"),
    observing_frequency_ghz: float = 28.0,
    seed_offset: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate noisy B‑mode spectra including lensing, foregrounds and noise.

    For each realisation:

    1. Draw a random CMB B‑mode map from the **total** (lensed + primordial + scalar) BB power spectrum.
    2. Convolve with the instrument beam.
    3. Draw Gaussian Q/U noise maps with RMS `noise_rms_uK` per arcminute and convert to B‑mode.
    4. Draw foreground Q/U maps from PySM and convert to B‑mode.
    5. Sum the three contributions at the map level.
    6. Compute the resulting BB spectrum and convert to D_ℓ.
    7. Additionally compute the purely primordial spectrum (tensor only) convolved with the beam for evaluation.

    Parameters
    ----------
    r : float
        Tensor–to–scalar ratio used in CAMB.
    nside : int
        HEALPix NSIDE parameter.
    num_samples : int
        Number of realisations to generate.
    fwhm_arcmin : float
        Full width at half maximum of the beam in arcminutes.
    noise_rms_uK : float
        RMS of the white noise in microkelvin per arcminute.
    foreground_presets : Iterable[str], optional
        PySM3 preset strings for synchrotron, dust and AME components.
    observing_frequency_ghz : float, optional
        Central observing frequency of the experiment in GHz.
    seed_offset : int, optional
        Seed offset for random number generation.

    Returns
    -------
    dl_noisy : np.ndarray
        Array of shape `(num_samples, 2*nside-2)` containing noisy D_ℓ spectra.
    dl_gt : np.ndarray
        Array of shape `(num_samples, 2*nside-2)` containing the ground truth
        primordial D_ℓ spectra after beam convolution.
    """
    lmax = 2 * nside - 1
    npix = hp.nside2npix(nside)

    # Set up CAMB for total and tensor B‑mode power spectra
    pars = camb.CAMBparams(WantTensors=True)
    pars.set_cosmology(H0=67.32, ombh2=0.0224, omch2=0.1201, Alens=1)
    pars.InitPower.set_params(ns=0.966, r=r)
    reion_model = camb.reionization.TanhReionization()
    reion_model.set_tau(0.05431)
    pars.Reion = reion_model
    pars.Reion.use_optical_depth = True
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
    cl_bb_total = powers['total'][:, 2]   # lensed + tensor + scalar
    cl_bb_tensor = powers['tensor'][:, 2] # purely primordial

    # Beam transfer function
    fwhm_rad = (fwhm_arcmin / 60.0) * np.pi / 180.0
    sigma_b = fwhm_rad / np.sqrt(8.0 * np.log(2.0))
    ell = np.arange(lmax + 1)
    b_ell = np.exp(-ell * (ell + 1) * sigma_b**2 / 2.0)
    bl2 = b_ell**2

    # Convert noise RMS (in uK·arcmin) to noise standard deviation per pixel
    noise_rms = noise_rms_uK * u.uK
    sigma = (((noise_rms * np.pi)/(60.0 * 180.0))**2 * (npix/(4.0 * np.pi)))**0.5

    # Foreground: generate a single B‑mode map and reuse for all realisations
    sky = pysm3.Sky(nside=nside, preset_strings=list(foreground_presets))
    emission = sky.get_emission(observing_frequency_ghz * u.GHz)[1:3]  # Q, U
    map_q_fg, map_u_fg = [x.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(observing_frequency_ghz * u.GHz)) for x in emission]
    map_t_fg = np.zeros_like(map_q_fg)
    alm_T_fg, alm_E_fg, alm_B_fg = hp.map2alm(
        [map_t_fg, map_q_fg, map_u_fg], lmax=lmax, pol=True
    )
    alm_B_fg_beam = hp.almxfl(alm_B_fg, b_ell)
    map_f_beam = hp.alm2map(alm_B_fg_beam, nside=nside, lmax=lmax, pol=False)

    # Storage
    dl_noisy_list: List[np.ndarray] = []
    dl_gt_list: List[np.ndarray] = []

    for i in range(num_samples):
        # Seed noise differently for reproducibility
        seed_cmb = seed_offset + i * 2
        seed_noise_q = seed_offset * 9783 + i
        seed_noise_u = seed_offset * 24214 + i

        # CMB total B‑mode map
        np.random.seed(seed_cmb)
        map_cmb = hp.synfast(cl_bb_total, nside=nside, pol=False)
        alm_cmb = hp.map2alm(map_cmb, lmax=lmax, pol=False)
        alm_cmb_beam = hp.almxfl(alm_cmb, b_ell)
        map_cmb_beam = hp.alm2map(alm_cmb_beam, nside=nside, lmax=lmax, pol=False)

        # Noise maps
        np.random.seed(seed_noise_q)
        noise_q = sigma * np.random.normal(0.0, 1.0, npix)
        np.random.seed(seed_noise_u)
        noise_u = sigma * np.random.normal(0.0, 1.0, npix)
        noise_t = np.zeros_like(noise_q)
        alm_Tn, alm_En, alm_Bn = hp.map2alm(
            [noise_t, noise_q, noise_u], lmax=lmax, pol=True
        )
        alm_Bn_beam = hp.almxfl(alm_Bn, b_ell)
        noise_b_beam = hp.alm2map(alm_Bn_beam, nside=nside, lmax=lmax, pol=False)

        # Sum contributions in map space
        map_total = map_cmb_beam + map_f_beam + noise_b_beam

        # Compute spectrum and convert to D_ell
        cl_tot = hp.anafast(map_total, lmax=lmax)
        d_tot = cl2dl(cl_tot[2:], ell[2:])
        dl_noisy_list.append(d_tot)

        # Ground truth primordial map for evaluation
        np.random.seed(seed_cmb)
        map_p = hp.synfast(cl_bb_tensor, nside=nside, pol=False)
        cl_p = hp.anafast(map_p, lmax=lmax)
        cl_p_beam = cl_p * bl2
        d_gt = cl2dl(cl_p_beam[2:], ell[2:])
        dl_gt_list.append(d_gt)

    return np.stack(dl_noisy_list, axis=0), np.stack(dl_gt_list, axis=0)


def generate_datasets(
    r_values: Iterable[float],
    nside: int,
    num_samples: int,
    fwhm_arcmin: float,
    noise_rms_uK: float,
    foreground_presets: Iterable[str],
    observing_frequency_ghz: float,
    out_dir: str | Path,
    ) -> None:
    """Generate datasets for a list of r values and write them to disk.

    Parameters
    ----------
    r_values : iterable of float
        The tensor–to–scalar ratios to simulate.
    nside : int
        HEALPix NSIDE parameter.
    num_samples : int
        Number of realisations per r value.
    fwhm_arcmin : float
        Beam full width at half maximum in arcminutes.
    noise_rms_uK : float
        RMS of the white noise in microkelvin per arcminute.
    foreground_presets : iterable of str
        List of PySM preset strings.
    observing_frequency_ghz : float
        Observation frequency in GHz for the foregrounds.
    out_dir : str or Path
        Directory in which to save the output `.npy` files.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for r in r_values:
        print(f"Generating datasets for r={r}...")
        dl_train = generate_primordial_dataset(
            r=r,
            nside=nside,
            num_samples=num_samples,
            fwhm_arcmin=fwhm_arcmin,
        )
        dl_noisy, dl_gt = generate_observational_dataset(
            r=r,
            nside=nside,
            num_samples=num_samples,
            fwhm_arcmin=fwhm_arcmin,
            noise_rms_uK=noise_rms_uK,
            foreground_presets=foreground_presets,
            observing_frequency_ghz=observing_frequency_ghz,
        )
        # File names embed the value of r to avoid conflicts
        fname_train = out_path / f"dl_train_r{r}.npy"
        fname_noisy = out_path / f"dl_noisy_r{r}.npy"
        fname_gt    = out_path / f"dl_gt_r{r}.npy"
        np.save(fname_train, dl_train)
        np.save(fname_noisy, dl_noisy)
        np.save(fname_gt, dl_gt)
        print(f"Saved training data to {fname_train}")
        print(f"Saved noisy data to   {fname_noisy}")
        print(f"Saved ground truth to {fname_gt}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate CMB B-mode datasets for score-based modelling")
    parser.add_argument(
        "--r",
        type=float,
        nargs='+',
        required=True,
        help="One or more tensor-to-scalar ratios for which to generate data",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=16,
        help="HEALPix NSIDE parameter (default: 16)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of realisations per r value (default: 10000)",
    )
    parser.add_argument(
        "--fwhm",
        type=float,
        default=24.8,
        help="Beam full-width at half-maximum in arcminutes (default: 24.8)",
    )
    parser.add_argument(
        "--noise-rms",
        type=float,
        default=11.9,
        help="Pixel noise RMS in microkelvin per arcminute (default: 11.9 uK arcmin)",
    )
    parser.add_argument(
        "--foreground-presets",
        type=str,
        default="s3,d1,a2",
        help="Comma-separated PySM preset strings (default: 's3,d1,a2')",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=28.0,
        help="Observation frequency in GHz used for foregrounds (default: 28 GHz)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data",
        help="Directory to save the generated datasets (default: 'data')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Parse foreground presets
    presets = [p.strip() for p in args.foreground_presets.split(',') if p.strip()]
    generate_datasets(
        r_values=args.r,
        nside=args.nside,
        num_samples=args.num_samples,
        fwhm_arcmin=args.fwhm,
        noise_rms_uK=args.noise_rms,
        foreground_presets=presets,
        observing_frequency_ghz=args.frequency,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()