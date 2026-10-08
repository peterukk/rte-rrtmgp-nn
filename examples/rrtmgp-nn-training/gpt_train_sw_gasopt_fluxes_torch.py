#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train new shortwave gas-optics MLPs through the differentiable PyTorch
shortwave radiation model, using band-wise or broadband fluxes as the target.
The user specifies the bands 

The training target from load_rrtmgp(..., predictand="sw_gpt_fluxes") is assumed
to be a tuple:
    y_ref = (rsu_gpt, rsd_gpt, rsd_dir_gpt)
with each array shaped (nbatch, nlev, ng_ref).

The model prediction is reduced with my_reduction(). It is possible to train on
broadband fluxes, i.e. a sum over the spectral/g-point dimension, or by optimizing fluxes
in specific bands
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Tuple
from random import randrange
import numpy as np
import torch
import torch.nn as nn

try:
    import wandb
except ImportError:
    wandb = None
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchinfo import summary

from gpt_ml_load_save_preproc import (
    load_RFMIP_data,
    load_rrtmgp,
    prepare_RFMIP_data,
)
import gpt_torch_models_rad as radlib
from gpt_torch_models_rad import (
    SW_rad_torch,
    load_gas_optics_from_file,
    mlp_gasopt_inlined_processing,
)
from coefficients import xmin_sw, xmax_sw, xmin_lw, xmax_lw #, WAVENUM_SPLITS

RFMIP_EXPERIMENTS = {
    0: "Present day (PD)",
    1: "Pre-industrial (PI) greenhouse gas concentrations",
    2: "4xCO2",
    3: "future",
    4: "0.5xCO2",
    5: "2xCO2",
    6: "3xCO2",
    7: "8xCO2",
    8: "PI CO2",
    9: "PI CH4",
    10: "PI N2O",
    11: "PI O3",
    12: "PI HCs",
    13: "+4K",
    14: "+4K, const. RH",
    15: "PI all",
    16: "future-all",
    17: "LGM",
}

# Experiment pairs are (perturbed, baseline), matching the requested
# forcing-error convention: (true2 - true1) - (pred2 - pred1).
RFMIP_IRF_PAIRS = {
    "future_minus_pi": (3, 1),
    "ch4_pd_minus_pi": (0, 9),
    "co2_8x_minus_0.5x": (7,4),

}

RFMIP_WANDB_METRICS_SW = (
    "rfmip/mae_heating_rate_all",
    "rfmip/mae_heating_rate_present_day",
    "rfmip/mae_heating_rate_preindustrial",
    "rfmip/mae_heating_rate_future_all",
    "rfmip/bias_surface_downwelling_flux",
    "rfmip/bias_toa_irf_future_minus_pi",
    "rfmip/bias_surface_irf_future_minus_pi",
    "rfmip/bias_surface_irf_ch4_pd_minus_pi",
)

# train_on_bands=True 

def make_band_bounds_sw(
    *,
    rrtmgp_splits: list[int],
    ng_per_band: list[int],
    ng_ref: int = 112,
) -> tuple[list[int], list[int], int]:
    """
    Build band bounds for:
      1. reference/RRTMGP training data, usually ng_ref=112
      2. learned model g-points, where ng_per_band controls allocation

    rrtmgp_splits are Python-style zero-based exclusive upper bounds.

    Example:
        RRTMGP Fortran band limits:
            1-10, 11-18, 19-29, 30-37, ...
        valid Python split points:
            10, 18, 29, 37, ...

        rrtmgp_splits = [29, 80, 89, 102]
        ng_per_band   = [4, 7, 2, 2, 1]

    Returns:
        rrtmgp_bounds: [0] + rrtmgp_splits + [ng_ref]
        model_bounds:  cumulative bounds from ng_per_band
        ng:            total learned g-points, sum(ng_per_band)
    """

    # RRTMGP SW bnd_limits_gpt, originally 1-based inclusive:
    #   1-10, 11-18, 19-29, 30-37, 38-46, 47-56, 57-67,
    #   68-71, 72-80, 81-89, 90-96, 97-102, 103-109, 110-112
    #
    # Converted to Python-style zero-based exclusive upper bounds:
    #   0:10, 10:18, 18:29, 29:37, ..., 109:112
    rrtmgp_band_bounds = [
        0, 10, 18, 29,
        37, 46, 56, 67,
        71, 80, 89, 96,
        102, 109, 112,
    ]

    if ng_ref != 112:
        raise ValueError(
            "This RRTMGP band-boundary check is hardcoded for ng_ref=112; "
            f"got ng_ref={ng_ref}"
        )

    rrtmgp_splits = [int(x) for x in rrtmgp_splits]
    ng_per_band = [int(x) for x in ng_per_band]

    if len(ng_per_band) != len(rrtmgp_splits) + 1:
        raise ValueError(
            "len(ng_per_band) must equal len(rrtmgp_splits) + 1; "
            f"got len(ng_per_band)={len(ng_per_band)}, "
            f"len(rrtmgp_splits)={len(rrtmgp_splits)}"
        )

    if any(n <= 0 for n in ng_per_band):
        raise ValueError(f"All ng_per_band entries must be positive; got {ng_per_band}")

    if any(s <= 0 or s >= ng_ref for s in rrtmgp_splits):
        raise ValueError(
            f"All rrtmgp_splits must be between 1 and ng_ref-1={ng_ref - 1}; "
            f"got {rrtmgp_splits}"
        )

    if any(b <= a for a, b in zip(rrtmgp_splits[:-1], rrtmgp_splits[1:])):
        raise ValueError(f"rrtmgp_splits must be strictly increasing; got {rrtmgp_splits}")

    valid_split_points = set(rrtmgp_band_bounds[1:-1])
    invalid_splits = [s for s in rrtmgp_splits if s not in valid_split_points]
    if invalid_splits:
        raise ValueError(
            "rrtmgp_splits must fall exactly on RRTMGP SW band boundaries. "
            f"Invalid split(s): {invalid_splits}. "
            f"Valid split points are: {sorted(valid_split_points)}"
        )

    rrtmgp_bounds = [0] + rrtmgp_splits + [ng_ref]

    model_bounds = [0]
    for n in ng_per_band:
        model_bounds.append(model_bounds[-1] + n)

    ng = model_bounds[-1]

    return rrtmgp_bounds, model_bounds, ng

def my_reduction(
    flux_tuple: Tuple[torch.Tensor, ...],
    bounds: list[int] | None = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Reduce spectral fluxes using explicit Python-style g-point bounds.

    Parameters
    ----------
    flux_tuple
        Tuple of spectral flux tensors, each shaped (..., ng).

    bounds
        Python-style exclusive bounds, e.g.
            [0, 29, 80, 89, 102, 112]

        If None, reduce to broadband by summing over all g-points.

    Returns
    -------
    Tuple of reduced flux tensors. With explicit bounds, the final dimension
    is the number of requested bands. With bounds=None, the spectral
    dimension is removed.
    """
    if bounds is None:
        return tuple(f.sum(dim=-1) for f in flux_tuple)

    def band_sum(x):
        return torch.cat(
            [
                x[..., bounds[i]:bounds[i + 1]].sum(dim=-1, keepdim=True)
                for i in range(len(bounds) - 1)
            ],
            dim=-1,
        )

    return tuple(band_sum(f) for f in flux_tuple)

def flatten_flux_tuple(fluxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Concatenate rsu, rsd, rsd_dir after reduction for equal-weighted loss."""
    parts = []
    for f in fluxes:
        if f.ndim == 2:
            # (batch, nlev) -> (batch, nlev)
            parts.append(f)
        elif f.ndim == 3:
            # (batch, nlev, nband) -> (batch, nlev*nband)
            parts.append(f.reshape(f.shape[0], -1))
        else:
            raise ValueError(f"Expected reduced flux with 2 or 3 dims, got shape {tuple(f.shape)}")
    return torch.cat(parts, dim=-1)

def compute_band_flux_weights(
    y_ref: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ref_band_bounds: list[int],
    device: torch.device,
    eps: float = 1.0,
) -> torch.Tensor:
    """
    Compute per-band loss weights as the inverse variance of band fluxes across
    the training set. Weights are normalised so their mean equals 1, keeping
    the overall loss magnitude comparable to the unweighted case.

    Args:
        y_ref: tuple of (rsu, rsd, rsd_dir) numpy arrays containing the full
               RRTMGP g-point fluxes before reduction.
        ref_band_bounds: explicit Python-style bounds used to reduce the
                         reference RRTMGP spectrum to the user-selected bands.
        device: target torch device.
        eps: floor added to variance before inversion to avoid division by
             near-zero variance.

    Returns:
        weights: float32 tensor of shape (nband,) on `device`.
    """
    flux_t = tuple(
        torch.as_tensor(np.asarray(y), dtype=torch.float32)
        for y in y_ref
    )

    rsu_b, rsd_b, rsd_dir_b = my_reduction(
        flux_t,
        bounds=ref_band_bounds,
    )

    # Variance across all (nobs * nlev) samples for each band, averaged over
    # the three flux components so a single weight applies to all three.
    def _band_var(x):
        return x.reshape(-1, x.shape[-1]).var(dim=0)

    var = (
        _band_var(rsu_b)
        + _band_var(rsd_b)
        + _band_var(rsd_dir_b)
    ) / 3.0

    weights = 1.0 / (var + eps)
    weights = weights / weights.mean()

    print("Band flux variances:", var.tolist())
    print("Band loss weights:  ", weights.tolist())

    return weights.to(device)

def flux_loss(
    pred_tuple,
    true_tuple,
    pred_band_bounds: list[int] | None = None,
    true_band_bounds: list[int] | None = None,
    band_weights: torch.Tensor | None = None,
    alpha: float = 0.0,
    pres: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """
    MSE loss over explicitly reduced spectral fluxes, optionally weighted per
    band. If both band-bound arguments are None, the loss is broadband.

    The prediction and reference use different g-point index spaces, so their
    reduction bounds are passed separately:
      - pred_band_bounds: learned-model g-point bounds
      - true_band_bounds: reference/RRTMGP g-point bounds

    If alpha > 0, adds a broadband heating-rate MSE term:
        L = (1 - alpha) * flux_loss + alpha * hr_loss
    """
    rsu_p, rsd_p, rsd_dir_p = my_reduction(
        pred_tuple,
        bounds=pred_band_bounds,
    )
    rsu_t, rsd_t, rsd_dir_t = my_reduction(
        true_tuple,
        bounds=true_band_bounds,
    )

    def _mse(p, t):
        sq = (p - t) ** 2
        if band_weights is not None and sq.ndim == 3:
            sq = sq * band_weights
        return sq.mean()

    loss_rsu = _mse(rsu_p, rsu_t)
    loss_rsd = _mse(rsd_p, rsd_t)
    loss_dir = _mse(rsd_dir_p, rsd_dir_t)
    loss_flux = (loss_rsu + loss_rsd + loss_dir) / 3.0

    if alpha == 0.0:
        return loss_flux

    # Heating-rate loss is always broadband, independent of the training-band
    # reduction used for the flux loss.
    assert pres is not None, "pres must be provided when alpha > 0"
    rsu_p_bb = pred_tuple[0].sum(dim=-1)
    rsd_p_bb = pred_tuple[1].sum(dim=-1)
    rsu_t_bb = true_tuple[0].sum(dim=-1)
    rsd_t_bb = true_tuple[1].sum(dim=-1)

    hr_p = calc_heatingrate_torch(rsu_p_bb, rsd_p_bb, pres)
    hr_t = calc_heatingrate_torch(rsu_t_bb, rsd_t_bb, pres)
    loss_hr = torch.mean((hr_p - hr_t) ** 2)

    return (1.0 - alpha) * loss_flux + alpha * loss_hr

# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------

def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def build_trainable_radiation_model(
    *,
    device, #: torch.device,
    xmin,#: np.ndarray,
    xmax,#: np.ndarray,
    ng,#: int,
    nh,#: int,
    do_norm,#: bool,
    band_bounds=None,
    rrtmgp_band_bounds=None,
) -> SW_rad_torch:
    """Construct two new gas-optics MLPs and wrap them in SW_rad_torch."""

    shortwave=True

    gas_abs = mlp_gasopt_inlined_processing(
        device=device,
        xmin=xmin,
        xmax=xmax,
        ymean=None,
        ystd=None,
        nn_w1=None,
        nn_w2=None,
        nn_w3=None,
        nn_b1=None,
        nn_b2=None,
        nn_b3=None,
        is_longwave=not shortwave,
        solar_source=None,
        rrtmgp_bounds_in=rrtmgp_band_bounds,
        band_bounds=band_bounds,  
        lock_weights=False,
        ng=ng,
        nh=nh,
        do_norm=do_norm,
    )
    gas_ray = mlp_gasopt_inlined_processing(
        device=device,
        xmin=xmin,
        xmax=xmax,
        ymean=None,
        ystd=None,
        nn_w1=None,
        nn_w2=None,
        nn_w3=None,
        nn_b1=None,
        nn_b2=None,
        nn_b3=None,
        is_longwave=not shortwave,
        solar_source=None,
        rrtmgp_bounds_in=rrtmgp_band_bounds,
        band_bounds=band_bounds,
        lock_weights=False,
        ng=ng,
        nh=nh,
        do_norm=do_norm,
    )
    infostr = summary(gas_abs)
    infostr = summary(gas_ray)

    print("INPUT NG", ng, "GAS ABS NG", gas_abs.ng)

    model = SW_rad_torch(
        device=device,
        gas_optics_model_sw_abs=gas_abs,
        gas_optics_model_sw_ray=gas_ray,
        ng_sw=ng,
        return_gpt_fluxes=True,
    ).to(device)
    return model

def forward_fluxes(
    model: SW_rad_torch,
    x: torch.Tensor,
    col_dry: torch.Tensor,
    mu0: torch.Tensor,
    toa: torch.Tensor,
    pres: torch.Tensor,
    albedo: torch.Tensor,
    printdebug=False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, rsu_pred, rsd_pred, rsd_dir_pred = model(
        x,
        col_dry,
        mu0,
        toa,
        pres,
        albedo,
        albedo,
        printdebug,
    )
    batch_size = x.shape[0]
    nlev = pres.shape[1]
    return rsu_pred, rsd_pred, rsd_dir_pred


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

# def load_input_norm_coeffs(
#     gasopt_abs_file: str,
#     device: torch.device,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Load input normalisation coefficients from an existing gas optics file."""
#     gas_abs_dummy = load_gas_optics_from_file(device, gasopt_abs_file)
#     coeffs = (
#         gas_abs_dummy.xmin.detach().cpu().numpy(),
#         gas_abs_dummy.xmax.detach().cpu().numpy(),
#     )
#     del gas_abs_dummy
#     if device.type == "cuda":
#         torch.cuda.empty_cache()
#     return coeffs

def load_training_data(
    *,
    data_file: str,
    input_norm_coeffs: Tuple[np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, Dict[str, np.ndarray], list[str], str]:
    predictand = "sw_gpt_fluxes"
    x, y_ref, col_dry, input_names, kdist_str, aux = load_rrtmgp(
        data_file,
        predictand=predictand,
        load_fluxes=True,
        input_norm_coefficients=input_norm_coeffs,
    )

    if not isinstance(y_ref, tuple) or len(y_ref) != 3:
        raise TypeError(
            "Expected load_rrtmgp(..., predictand='sw_gpt_fluxes') to return "
            "y_ref as a tuple (rsu, rsd, rsd_dir)."
        )

    return x, y_ref, col_dry, aux, input_names, kdist_str

def make_dataloaders(
    *,
    x: np.ndarray,
    y_ref: Tuple[np.ndarray, np.ndarray, np.ndarray],
    col_dry: np.ndarray,
    aux: Dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    val_fraction: float,
    preload_gpu: bool,
    seed: int,
) -> Tuple[DataLoader, DataLoader | None]:
    target_device = device if preload_gpu else torch.device("cpu")

    tensors = [
        torch.as_tensor(np.asarray(x), dtype=torch.float32, device=target_device),
        torch.as_tensor(np.asarray(col_dry), dtype=torch.float32, device=target_device),
        torch.as_tensor(np.asarray(aux["mu0"]), dtype=torch.float32, device=target_device),
        torch.as_tensor(np.asarray(aux["total_solar_irradiance"]), dtype=torch.float32, device=target_device),
        torch.as_tensor(np.asarray(aux["pres_level"]), dtype=torch.float32, device=target_device),
        torch.as_tensor(np.asarray(aux["surface_albedo"]), dtype=torch.float32, device=target_device),
        torch.as_tensor(np.asarray(y_ref[0]), dtype=torch.float32, device=target_device),
        torch.as_tensor(np.asarray(y_ref[1]), dtype=torch.float32, device=target_device),
        torch.as_tensor(np.asarray(y_ref[2]), dtype=torch.float32, device=target_device),
    ]

    dataset = TensorDataset(*tensors)
    if val_fraction > 0.0:
        n_total = len(dataset)
        n_val = max(1, int(round(n_total * val_fraction)))
        n_train = n_total - n_val
        train_ds, val_ds = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        train_ds = dataset
        val_ds = None

    pin_memory = (device.type == "cuda" and not preload_gpu)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=pin_memory,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=pin_memory,
        )
    return train_loader, val_loader


# -----------------------------------------------------------------------------
# Train / eval loops
# -----------------------------------------------------------------------------

def unpack_batch(batch, device: torch.device):
    batch = [b.to(device, non_blocking=True) if b.device != device else b for b in batch]
    x, col_dry, mu0, toa, pres, albedo, rsu, rsd, rsd_dir = batch
    return x, col_dry, mu0, toa, pres, albedo, (rsu, rsd, rsd_dir)


def calc_heatingrate_torch(fluxup: torch.Tensor, fluxdn: torch.Tensor, pres_level: torch.Tensor) -> torch.Tensor:
    """Heating rate in K/day from up/down broadband fluxes and pressure levels."""
    F = fluxdn - fluxup
    dF = F[:, 1:] - F[:, :-1]
    dp = pres_level[:, 1:] - pres_level[:, :-1]
    dFdp = dF / dp
    g = 9.81
    cp = 1004
    dTdt = -(g / cp) * dFdp
    return (24 * 3600) * dTdt


def load_rfmip_validation_data(
    input_norm_coeffs: Tuple[np.ndarray, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Load and preprocess the default RFMIP validation files once."""
    (
        x_raw,
        pres_level,
        flux_up_true,
        flux_dn_true,
        expt_labels,
        rfmip_aux,
    ) = load_RFMIP_data(return_aux=True)

    if x_raw.shape[0] < len(RFMIP_EXPERIMENTS):
        raise ValueError(
            f"Expected at least {len(RFMIP_EXPERIMENTS)} RFMIP experiments, "
            f"got {x_raw.shape[0]}"
        )

    prepared = prepare_RFMIP_data(
        x_raw,
        flux_up_true,
        flux_dn_true,
        # exp_index_pairs=list(RFMIP_IRF_PAIRS.values()),
        input_norm_coefficients=input_norm_coeffs,
        pres_level=pres_level,
        aux=rfmip_aux,
        # return_full=True,
    )
    prepared["expt_labels"] = expt_labels

    print("Loaded RFMIP validation data")
    print(f"  experiments: {x_raw.shape[0]}, sites: {x_raw.shape[1]}, layers: {x_raw.shape[2]}")
    print(f"  x shape: {prepared['x'].shape}")
    print(f"  reference flux shape: {prepared['flux_up_true'].shape}")
    return prepared


def run_rfmip_validation(
    *,
    model: SW_rad_torch,
    rfmip_data: Dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    """Evaluate the requested RFMIP heating-rate, flux-bias and IRF metrics."""
    model.eval()

    x = np.asarray(rfmip_data["x"])
    nexpt, nsite, nlay, nx = x.shape
    nprof = nexpt * nsite
    nlev = nlay + 1

    def _flat(name, trailing_shape):
        arr = np.asarray(rfmip_data[name])
        return arr.reshape((nprof,) + trailing_shape)

    x_flat = x.reshape(nprof, nlay, nx)
    col_dry_flat = _flat("col_dry", (nlay,))
    mu0_flat = _flat("mu0", ())
    toa_flat = _flat("total_solar_irradiance", ())
    pres_flat = _flat("pres_level", (nlev,))
    albedo_flat = _flat("surface_albedo", ())

    rsu_pred_parts = []
    rsd_pred_parts = []
    eval_batch_size = max(1, int(batch_size))

    with torch.no_grad():
        for start in range(0, nprof, eval_batch_size):
            stop = min(start + eval_batch_size, nprof)
            xb = torch.as_tensor(x_flat[start:stop], dtype=torch.float32, device=device)
            colb = torch.as_tensor(col_dry_flat[start:stop], dtype=torch.float32, device=device)
            mu0b = torch.as_tensor(mu0_flat[start:stop], dtype=torch.float32, device=device)
            toab = torch.as_tensor(toa_flat[start:stop], dtype=torch.float32, device=device)
            presb = torch.as_tensor(pres_flat[start:stop], dtype=torch.float32, device=device)
            albedob = torch.as_tensor(albedo_flat[start:stop], dtype=torch.float32, device=device)

            rsu_gpt, rsd_gpt, _ = forward_fluxes(
                model, xb, colb, mu0b, toab, presb, albedob, printdebug=False
            )
            # RFMIP has no reference rsd_dir. The model still returns it, but it
            # is intentionally ignored here. Metrics use broadband rsu and rsd.
            rsu_pred_parts.append(rsu_gpt.sum(dim=-1).cpu())
            rsd_pred_parts.append(rsd_gpt.sum(dim=-1).cpu())

    rsu_pred = torch.cat(rsu_pred_parts, dim=0).reshape(nexpt, nsite, nlev)
    rsd_pred = torch.cat(rsd_pred_parts, dim=0).reshape(nexpt, nsite, nlev)
    rsu_true = torch.as_tensor(rfmip_data["flux_up_true"], dtype=torch.float32)
    rsd_true = torch.as_tensor(rfmip_data["flux_dn_true"], dtype=torch.float32)
    pres = torch.as_tensor(rfmip_data["pres_level"], dtype=torch.float32)

    if rsu_true.shape != (nexpt, nsite, nlev) or rsd_true.shape != (nexpt, nsite, nlev):
        raise ValueError(
            "Expected RFMIP reference fluxes to have shape "
            f"({nexpt}, {nsite}, {nlev}); got rsu={tuple(rsu_true.shape)}, "
            f"rsd={tuple(rsd_true.shape)}"
        )

    hr_pred = calc_heatingrate_torch(
        rsu_pred.reshape(nprof, nlev),
        rsd_pred.reshape(nprof, nlev),
        pres.reshape(nprof, nlev),
    ).reshape(nexpt, nsite, nlay)
    hr_true = calc_heatingrate_torch(
        rsu_true.reshape(nprof, nlev),
        rsd_true.reshape(nprof, nlev),
        pres.reshape(nprof, nlev),
    ).reshape(nexpt, nsite, nlay)
    hr_abs_error = torch.abs(hr_true - hr_pred)

    # Determine surface/TOA from pressure rather than assuming level ordering.
    # p_first = float(pres[..., 0].mean())
    # p_last = float(pres[..., -1].mean())
    # if p_first <= p_last:
    toa_index, surface_index = 0, -1
    # else:
    #     toa_index, surface_index = -1, 0


    surface_dn_bias = torch.mean(rsd_true[..., surface_index]) - torch.mean(rsd_pred[..., surface_index])

    net_true = rsd_true - rsu_true
    net_pred = rsd_pred - rsu_pred

    def _irf_bias(iexp2: int, iexp1: int, ilev: int) -> torch.Tensor:
        true_irf = net_true[iexp2, :, ilev] - net_true[iexp1, :, ilev]
        pred_irf = net_pred[iexp2, :, ilev] - net_pred[iexp1, :, ilev]
        return torch.mean(true_irf - pred_irf)

    future, pi = RFMIP_IRF_PAIRS["future_minus_pi"]
    pd, pi_ch4 = RFMIP_IRF_PAIRS["ch4_pd_minus_pi"]

    metrics = {
        "rfmip/mae_heating_rate_all": float(hr_abs_error.mean()),
        "rfmip/mae_heating_rate_present_day": float(hr_abs_error[0].mean()),
        "rfmip/mae_heating_rate_preindustrial": float(hr_abs_error[1].mean()),
        "rfmip/mae_heating_rate_future_all": float(hr_abs_error[16].mean()),
        "rfmip/bias_surface_downwelling_flux": float(surface_dn_bias),
        "rfmip/bias_toa_irf_future_minus_pi": float(_irf_bias(future, pi, toa_index)),
        "rfmip/bias_surface_irf_future_minus_pi": float(_irf_bias(future, pi, surface_index)),
        "rfmip/bias_surface_irf_ch4_pd_minus_pi": float(_irf_bias(pd, pi_ch4, surface_index)),
    }
    return metrics


def run_epoch(
    *,
    model: SW_rad_torch,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float | None,
    band_weights: torch.Tensor | None = None,
    alpha: float = 0.0,
    pred_band_bounds: list[int] | None = None,
    true_band_bounds: list[int] | None = None,
) -> Dict[str, float]:

    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_rmse = 0.0
    total_hr_sse = 0.0
    total_hr_n = 0
    nsamp = 0

    pearson_stats = {
        "rsu": {"n": 0, "sum_p": 0.0, "sum_t": 0.0, "sum_p2": 0.0, "sum_t2": 0.0, "sum_pt": 0.0},
        "rsd": {"n": 0, "sum_p": 0.0, "sum_t": 0.0, "sum_p2": 0.0, "sum_t2": 0.0, "sum_pt": 0.0},
    }

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            x, col_dry, mu0, toa, pres, albedo, y_true_fluxes = unpack_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            y_pred_fluxes = forward_fluxes(model, x, col_dry, mu0, toa, pres, albedo, printdebug=False)
            # loss = flux_mse_loss(y_pred_fluxes, y_true_fluxes)

            # if band_weights is not None:
            #   loss = flux_loss(y_pred_fluxes, y_true_fluxes, band_weights=band_weights)
            # else:
            #   loss = flux_loss(y_pred_fluxes, y_true_fluxes)
            loss = flux_loss(
                y_pred_fluxes,
                y_true_fluxes,
                pred_band_bounds=pred_band_bounds,
                true_band_bounds=true_band_bounds,
                band_weights=band_weights,
                alpha=alpha,
                pres=pres.detach() if alpha > 0.0 else None,
            )

            if training:
                loss.backward()
                if grad_clip is not None and grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            rsu_p = y_pred_fluxes[0].detach().sum(dim=-1)
            rsd_p = y_pred_fluxes[1].detach().sum(dim=-1)

            rsu_t = y_true_fluxes[0].detach().sum(dim=-1)
            rsd_t = y_true_fluxes[1].detach().sum(dim=-1)

            surface_dn_bias = torch.mean(rsd_t[..., -1]) - torch.mean(rsd_p[..., -1])
            # print("rsd_sfc t ",rsd_t[..., -1].mean().item(), "p",  rsd_p[..., -1].mean().item())

            hr_p = calc_heatingrate_torch(rsu_p, rsd_p, pres.detach())
            hr_t = calc_heatingrate_torch(rsu_t, rsd_t, pres.detach())
            hr_diff = hr_p - hr_t
            total_hr_sse += float(torch.sum(hr_diff * hr_diff))
            total_hr_n += hr_diff.numel()

            for name, pred, true in (
                ("rsu", rsu_p, rsu_t),
                ("rsd", rsd_p, rsd_t),
            ):
                pred = pred.reshape(-1).double()
                true = true.reshape(-1).double()
                pearson_stats[name]["n"] += pred.numel()
                pearson_stats[name]["sum_p"] += pred.sum().item()
                pearson_stats[name]["sum_t"] += true.sum().item()
                pearson_stats[name]["sum_p2"] += torch.sum(pred * pred).item()
                pearson_stats[name]["sum_t2"] += torch.sum(true * true).item()
                pearson_stats[name]["sum_pt"] += torch.sum(pred * true).item()

            bs = x.shape[0]
            # one CPU sync per batch for logging only
            total_loss += float(loss.detach()) * bs
            total_rmse += float(torch.sqrt(loss.detach())) * bs
            nsamp += bs

    def _corr(stats: Dict[str, float]) -> float:
        n = float(stats["n"])
        if n == 0.0:
            return float("nan")
        cov = stats["sum_pt"] - (stats["sum_p"] * stats["sum_t"] / n)
        var_p = stats["sum_p2"] - (stats["sum_p"] ** 2 / n)
        var_t = stats["sum_t2"] - (stats["sum_t"] ** 2 / n)
        denom = np.sqrt(max(var_p, 0.0) * max(var_t, 0.0))
        if denom == 0.0:
            return float("nan")
        return cov / denom

    return {
        "mse": total_loss / max(nsamp, 1),
        "rmse": total_rmse / max(nsamp, 1),
        "hr_rmse": np.sqrt(total_hr_sse / max(total_hr_n, 1)),
        "rsu_pearson": _corr(pearson_stats["rsu"]),
        "rsd_pearson": _corr(pearson_stats["rsd"]),
        "surface_dn_bias": float(surface_dn_bias),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SW gas-optics models through SW_rad_torch using spectral flux targets."
    )
    parser.add_argument("--data-file", required=True, help="NetCDF file containing sw_gpt_fluxes training data")
    # parser.add_argument("--gasopt-abs-file", required=True, help="Existing SW absorption gas-optics NetCDF model; used for input normalization only")
    # parser.add_argument("--gasopt-ray-file", required=True, help="Existing SW Rayleigh gas-optics NetCDF model; used for input normalization only")
    parser.add_argument("--output", default=None, help="Output PyTorch checkpoint (default: auto-generated from hyperparameters)")
    parser.add_argument("--device", default=None, help="Device string, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--ng", type=int, default=16, help="Number of learned spectral/g-point channels in the new gas-optics models")
    parser.add_argument("--nh", type=int, default=32, help="Hidden neurons in each gas-optics MLP hidden layer")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--optimizer",
        choices=["adam", "soap"],
        default="soap",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "plateau", "onecycle"],
        default="none",
    )

    parser.add_argument(
        "--scheduler-max-lr",
        type=float,
        default=0.0015, #1.0e-3,
    )

    parser.add_argument(
        "--scheduler-min-lr",
        type=float,
        default=3e-7, #1.0e-6,
    )

    parser.add_argument(
        "--scheduler-peak-epoch",
        type=int,
        default=20, # 4?
    )

    parser.add_argument(
        "--scheduler-end-epoch",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--scheduler-annealing",
        choices=["linear", "cos"],
        default="cos",
    )

    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--validate-on-rfmip",
        action="store_true",
        help=(
            "At the end of every epoch, evaluate the model on the default RFMIP "
            "dataset and log heating-rate, flux-bias and radiative-forcing metrics"
        ),
    )
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preload-gpu", action="store_true", help="Move the full dataset to GPU before training")
    parser.add_argument("--train-solar-weights", action="store_true", help="Allow gas_abs.sw_solar_weights to be optimized")
    parser.add_argument("--alpha", type=float, default=0.0,
        help="Weight of heating-rate loss term: L = (1-alpha)*flux_loss + alpha*hr_loss. Default 0 (flux only).")
    parser.add_argument("--do-norm", action="store_true", help="use learned output normalisation coefficients")
    parser.add_argument("--compile", action="store_true", help="Try torch.compile(model) before training")
    parser.add_argument(
        "--rrtmgp-splits",
        type=_parse_int_list,
        default=None, #"29,80,89,102",
        help="Comma-separated RRTMGP g-point split indices, e.g. 29,80,89,102",
    )

    parser.add_argument(
        "--ng-per-band",
        type=_parse_int_list,
        default=None, #"4,7,2,2,1",
        help="Comma-separated learned g-points per band, e.g. 4,7,2,2,1",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="sw-gasopt-flux-training", help="Weights & Biases project name")

    return parser.parse_args()


def main() -> None:

    shortwave=True # add support for longwave later
    if shortwave:
      prefix = "sw"
      ng_ref = 112
    else:
      prefix = "lw"
      ng_ref = 128

    model_num = randrange(10,99999)

    args = parse_args()
    if args.rrtmgp_splits is None or args.ng_per_band is None:
        print("band training OFF since rrtmgp_splits and ng_per_band were not provided")
        train_on_bands = False
        ng = args.ng
    else:
        train_on_bands = True
        ng = sum(args.ng_per_band)
        print("Number of g-points: ", ng)

    if args.output is None:
        if train_on_bands:
            split_str = "-".join(str(x) for x in args.rrtmgp_splits)
            ng_band_str = "-".join(str(x) for x in args.ng_per_band)
            args.output = (
                f"trained_models/{prefix}_gasopt"
                f"_bnd{split_str}"
                f"_ng{ng_band_str}"
                f"_nh{args.nh}"
                f"_alpha{args.alpha:.2f}_num{model_num}.pt"
            )
        else:
            args.output = f"trained_models/{prefix}_gasopt_ng{ng}_nh{args.nh}_alpha{args.alpha:.2f}_num{model_num}.pt"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise ImportError("Weights & Biases logging requested with --wandb, but wandb is not installed")
        wandb_run = wandb.init(
            project=args.wandb_project,
            config=vars(args),
        )
        if args.validate_on_rfmip:
            wandb.define_metric("epoch")
            if shortwave:
              RFMIP_WANDB_METRICS = RFMIP_WANDB_METRICS_SW
            else:
              RFMIP_WANDB_METRICS = RFMIP_WANDB_METRICS_LW
            for metric_name in RFMIP_WANDB_METRICS:
                wandb.define_metric(metric_name, step_metric="epoch")

    def _as_data_file_list(data_file_arg):
        """
        Supports either:
          --data-file file1.nc
          --data-file file1.nc,file2.nc,file3.nc
        and also works if parse_args() was later changed to return a list.
        """
        if isinstance(data_file_arg, (list, tuple)):
            return [str(f) for f in data_file_arg]
        return [f.strip() for f in str(data_file_arg).split(",") if f.strip()]

    def _concat_aux_dicts(aux_list):
        out = {}
        keys = aux_list[0].keys()
        for key in keys:
            vals = [a[key] for a in aux_list]
            if all(isinstance(v, np.ndarray) for v in vals):
                try:
                    out[key] = np.concatenate(vals, axis=0)
                except ValueError:
                    # For non-sample arrays, keep the first value.
                    out[key] = vals[0]
            else:
                out[key] = vals[0]
        return out

    data_files = _as_data_file_list(args.data_file)

    # The first input file defines the primary vertical grid. Files with the same
    # number of layers are concatenated exactly as before and share the existing
    # train/validation split. Files with a different nlay get their own training-
    # only DataLoader. This avoids padding/interpolation solely for batching.
    primary_nlay = None
    primary_x_all = []
    primary_y_ref_all = [[], [], []]
    primary_col_dry_all = []
    primary_aux_all = []
    primary_data_files = []
    extra_training_sets = []

    input_names = None
    kdist_str = None
    # input_norm_coeffs = None

    # input_norm_coeffs = load_input_norm_coeffs(args.gasopt_abs_file, device)
    if shortwave:
      input_norm_coeffs = (xmin_sw, xmax_sw)
    else:
      input_norm_coeffs = (xmin_lw, xmax_lw)

    rfmip_data = None
    if args.validate_on_rfmip:
        rfmip_data = load_rfmip_validation_data(input_norm_coeffs)

    for i, data_file in enumerate(data_files):
        x_i, y_ref_i, col_dry_i, aux_i, input_names_i, kdist_str_i = load_training_data(
            data_file=data_file,
            input_norm_coeffs=input_norm_coeffs,
        )
        if i == 0:
            input_names = input_names_i
            kdist_str = kdist_str_i
            # input_norm_coeffs = input_norm_coeffs_i
        else:
            if kdist_str_i != kdist_str:
                print(f"WARNING: k-dist differs for {data_file}: {kdist_str_i} != {kdist_str}")
            if list(input_names_i) != list(input_names):
                raise ValueError(
                    f"Input names differ for {data_file}; refusing to train on incompatible inputs."
                )

        x_i = np.asarray(x_i)
        col_dry_i = np.asarray(col_dry_i)
        y_ref_i = tuple(np.asarray(y) for y in y_ref_i)
        nlay_i = int(x_i.shape[1])

        if primary_nlay is None:
            primary_nlay = nlay_i

        if nlay_i == primary_nlay:
            primary_x_all.append(x_i)
            primary_col_dry_all.append(col_dry_i)
            primary_aux_all.append(aux_i)
            for j in range(3):
                primary_y_ref_all[j].append(y_ref_i[j])
            primary_data_files.append(data_file)
            dataset_role = "primary grid"
        else:
            extra_training_sets.append(
                {
                    "data_file": data_file,
                    "x": x_i,
                    "y_ref": y_ref_i,
                    "col_dry": col_dry_i,
                    "aux": aux_i,
                    "nlay": nlay_i,
                }
            )
            dataset_role = "separate training-only grid"

        print(f"Loaded dataset {i + 1}/{len(data_files)}: {Path(data_file).name}")
        print(f"  role: {dataset_role}; nlay={nlay_i}")
        print(f"  x shape: {x_i.shape}")
        print(f"  col_dry shape: {col_dry_i.shape}")
        print(
            "  target flux shapes: "
            f"rsu={y_ref_i[0].shape}, "
            f"rsd={y_ref_i[1].shape}, "
            f"rsd_dir={y_ref_i[2].shape}"
        )

    if not primary_x_all:
        raise RuntimeError("No datasets were assigned to the primary vertical grid")

    # Concatenate only datasets compatible with the primary vertical grid.
    x = np.concatenate(primary_x_all, axis=0)
    col_dry = np.concatenate(primary_col_dry_all, axis=0)
    y_ref = (
        np.concatenate(primary_y_ref_all[0], axis=0),
        np.concatenate(primary_y_ref_all[1], axis=0),
        np.concatenate(primary_y_ref_all[2], axis=0),
    )
    aux = _concat_aux_dicts(primary_aux_all)

    xmin, xmax = input_norm_coeffs

    print(f"Loaded {len(data_files)} dataset(s) across {1 + len(extra_training_sets)} loader group(s)")
    print(f"k-dist: {kdist_str}")
    print(f"primary nlay: {primary_nlay}")
    print(f"primary files: {[Path(f).name for f in primary_data_files]}")
    print(f"primary combined x shape: {x.shape}")
    print(f"primary combined col_dry shape: {col_dry.shape}")
    print(
        "primary combined target flux shapes: "
        f"rsu={y_ref[0].shape}, "
        f"rsd={y_ref[1].shape}, "
        f"rsd_dir={y_ref[2].shape}"
    )

    # Only the primary grid gets a validation split.
    train_loader, val_loader = make_dataloaders(
        x=x,
        y_ref=y_ref,
        col_dry=col_dry,
        aux=aux,
        device=device,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        preload_gpu=args.preload_gpu,
        seed=args.seed,
    )

    train_loader_specs = [
        {
            "name": "primary",
            "wandb_prefix": "train",
            "loader": train_loader,
            "nlay": primary_nlay,
            "data_files": list(primary_data_files),
        }
    ]

    for iextra, extra in enumerate(extra_training_sets, start=1):
        extra_loader, extra_val_loader = make_dataloaders(
            x=extra["x"],
            y_ref=extra["y_ref"],
            col_dry=extra["col_dry"],
            aux=extra["aux"],
            device=device,
            batch_size=args.batch_size,
            val_fraction=0.0,
            preload_gpu=args.preload_gpu,
            seed=args.seed + iextra,
        )
        assert extra_val_loader is None
        name = f"extra{iextra}_nlay{extra['nlay']}"
        train_loader_specs.append(
            {
                "name": name,
                "wandb_prefix": f"train_{name}",
                "loader": extra_loader,
                "nlay": extra["nlay"],
                "data_files": [extra["data_file"]],
            }
        )
        print(
            f"Created training-only loader {name}: "
            f"{Path(extra['data_file']).name}, {len(extra_loader.dataset)} profiles"
        )

    if train_on_bands:
        rrtmgp_bounds, band_bounds, ng = make_band_bounds_sw(
            rrtmgp_splits=args.rrtmgp_splits,
            ng_per_band=args.ng_per_band,
            ng_ref=ng_ref,
        )

        print(f"RRTMGP reference band bounds: {rrtmgp_bounds}")
        print(f"Learned model band bounds:    {band_bounds}")
        print(f"Learned model ng:             {ng}")

        print("Computing per-band flux weights from training data...")
        band_weights = compute_band_flux_weights(
            y_ref,
            ref_band_bounds=rrtmgp_bounds,
            device=device,
        )

    else:
        band_weights = None
        rrtmgp_bounds = None
        band_bounds = None
        ng = args.ng

    if wandb_run is not None:
        wandb_run.config.update(
            {
                "train_on_bands": train_on_bands,
                "resolved_ng": ng,
                "band_bounds": band_bounds,
                "rrtmgp_bounds": rrtmgp_bounds,
                "data_files": data_files,
                "primary_data_files": primary_data_files,
                "primary_nlay": primary_nlay,
                "extra_training_files": [d["data_file"] for d in extra_training_sets],
                "extra_training_nlay": [d["nlay"] for d in extra_training_sets],
                "kdist_str": kdist_str,
                "model_num": model_num,
            },
            allow_val_change=True,
        )

    model = build_trainable_radiation_model(
        device=device,
        xmin=np.asarray(xmin, dtype=np.float32),
        xmax=np.asarray(xmax, dtype=np.float32),
        ng=ng,
        nh=args.nh,
        band_bounds=band_bounds,
        rrtmgp_band_bounds=rrtmgp_bounds,
        do_norm=args.do_norm,
        
    )

    if args.compile:
        print("Compiling model with torch.compile(...)")
        model = torch.compile(model)

    if args.optimizer == "soap":
        from soap import SOAP
        optimizer = SOAP(model.parameters(), lr = args.lr, betas=(.95, .95), weight_decay=.01, precondition_frequency=2)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = None

    if args.lr_scheduler == "onecycle":

        scheduler_end_epoch = (
            args.scheduler_end_epoch
            if args.scheduler_end_epoch is not None
            else args.epochs
        )

        if args.scheduler_peak_epoch >= scheduler_end_epoch:
            raise ValueError(
                "--scheduler-peak-epoch must be smaller than "
                "--scheduler-end-epoch"
            )

        max_lr = args.scheduler_max_lr
        min_lr = args.scheduler_min_lr

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,

            # Makes initial scheduler LR equal to --lr.
            div_factor=max_lr / args.lr,

            # Makes final scheduler LR equal to --scheduler-min-lr.
            final_div_factor=args.lr / min_lr,

            # One scheduler.step() per epoch.
            total_steps=scheduler_end_epoch,

            pct_start=(
                args.scheduler_peak_epoch
                / scheduler_end_epoch
            ),

            anneal_strategy=args.scheduler_annealing,
        )
        print("Initial optimizer LR:", optimizer.param_groups[0]["lr"])

    elif args.lr_scheduler == "plateau":
        raise NotImplementedError()
        # if not (0.0 < args.lr_scheduler_factor < 1.0):
        #     raise ValueError("--lr-scheduler-factor must be between 0 and 1")
        # if args.lr_scheduler_patience < 0:
        #     raise ValueError("--lr-scheduler-patience must be >= 0")
        # if args.min_lr < 0.0:
        #     raise ValueError("--min-lr must be >= 0")
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="min",
        #     factor=args.lr_scheduler_factor,
        #     patience=args.lr_scheduler_patience,
        #     min_lr=args.min_lr,
        # )
        # print(
        #     "Enabled ReduceLROnPlateau: "
        #     f"factor={args.lr_scheduler_factor}, "
        #     f"patience={args.lr_scheduler_patience}, min_lr={args.min_lr}"
        # )
    if args.lr > args.scheduler_max_lr:
        raise ValueError(
            "--lr must be <= --scheduler-max-lr for OneCycleLR"
        )

    if args.scheduler_min_lr >= args.lr:
        raise ValueError(
            "--scheduler-min-lr must be < --lr"
        )

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    wait = 0

    print("Beginning training, saving model to {} when new validation loss is reached".format(args.output))

    for epoch in range(1, args.epochs + 1):
        # Alternate loader order each epoch so an incompatible-grid dataset is
        # not always the final source of optimizer updates. With two loaders:
        # odd epochs are primary -> extra, even epochs are extra -> primary.
        if epoch % 2 == 1:
            epoch_loader_specs = train_loader_specs
        else:
            epoch_loader_specs = list(reversed(train_loader_specs))
        epoch_lr = optimizer.param_groups[0]["lr"]

        train_metrics_by_name = {}
        for spec in epoch_loader_specs:
            train_metrics_by_name[spec["name"]] = run_epoch(
                model=model,
                loader=spec["loader"],
                device=device,
                optimizer=optimizer,
                grad_clip=args.grad_clip,
                band_weights=band_weights,
                alpha=args.alpha,
                pred_band_bounds=band_bounds if train_on_bands else None,
                true_band_bounds=rrtmgp_bounds if train_on_bands else None,
            )

        # Keep the existing train/* metrics tied to the primary-grid data.
        # Additional loaders are logged separately below.
        train_metrics = train_metrics_by_name["primary"]

        if val_loader is not None:
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
                grad_clip=None,
                band_weights=band_weights,
                alpha=args.alpha,
                pred_band_bounds=band_bounds if train_on_bands else None,
                true_band_bounds=rrtmgp_bounds if train_on_bands else None,
            )
            monitor = val_metrics["rmse"]
            val_str = (
                f" - val_rmse: {val_metrics['rmse']:.2f}"
                # f" - val_mse: {val_metrics['mse']:.2f}"
                f" - val_hr_rmse: {val_metrics['hr_rmse']:.2f}"
                f" - val_rsu_r: {val_metrics['rsu_pearson']:.5f}"
                f" - val_rsd_r: {val_metrics['rsd_pearson']:.5f}"
            )
        else:
            # If primary validation is disabled, retain the original fallback:
            # checkpointing/scheduling use primary-grid training RMSE only.
            monitor = train_metrics["rmse"]
            val_str = ""

        rfmip_metrics = None
        if rfmip_data is not None:
            rfmip_metrics = run_rfmip_validation(
                model=model,
                rfmip_data=rfmip_data,
                device=device,
                batch_size=args.batch_size,
            )

        lr_before_scheduler = float(optimizer.param_groups[0]["lr"])

        if args.lr_scheduler == "plateau":
            lr_scheduler.step(monitor)

        elif args.lr_scheduler == "onecycle":
            if epoch <= scheduler_end_epoch:
                lr_scheduler.step()
        lr_after_scheduler = float(optimizer.param_groups[0]["lr"])

        # loader_order_str = " -> ".join(spec["name"] for spec in epoch_loader_specs)
        print(
            f"Epoch {epoch:04d}/{args.epochs}"
            # f" - loader_order: {loader_order_str}"
            f" - lr: {lr_before_scheduler:.3e}"
            f" - train_rmse: {train_metrics['rmse']:.2f}"
            f" - train_mse: {train_metrics['mse']:.2f}"
            f" - train_hr_rmse: {train_metrics['hr_rmse']:.2f}"
            f" - train_rsu_r: {train_metrics['rsu_pearson']:.5f}"
            f" - train_rsd_r: {train_metrics['rsd_pearson']:.5f}"
            f"{val_str}"
        )

        for spec in train_loader_specs[1:]:
            metrics = train_metrics_by_name[spec["name"]]
            print(
                f"  {spec['name']}"
                f" - rmse: {metrics['rmse']:.2f}"
                f" - hr_rmse: {metrics['hr_rmse']:.2f}"
                f" - rsu_r: {metrics['rsu_pearson']:.5f}"
                f" - rsd_r: {metrics['rsd_pearson']:.5f}"
            )


        if rfmip_metrics is not None:
            print(
                "  RFMIP"
                f" - HR MAE all: {rfmip_metrics['rfmip/mae_heating_rate_all']:.4f}"
                f" - PD: {rfmip_metrics['rfmip/mae_heating_rate_present_day']:.4f}"
                f" - PI: {rfmip_metrics['rfmip/mae_heating_rate_preindustrial']:.4f}"
                f" - future-all: {rfmip_metrics['rfmip/mae_heating_rate_future_all']:.4f}"
                f" - sfc dn bias: {rfmip_metrics['rfmip/bias_surface_downwelling_flux']:.4f}"
                f" - TOA IRF future-PI bias: {rfmip_metrics['rfmip/bias_toa_irf_future_minus_pi']:.4f}"
                f" - sfc IRF future-PI bias: {rfmip_metrics['rfmip/bias_surface_irf_future_minus_pi']:.4f}"
                f" - sfc IRF CH4 PD-PI bias: {rfmip_metrics['rfmip/bias_surface_irf_ch4_pd_minus_pi']:.4f}"
            )

        if wandb_run is not None:
            wandb_metrics = {
                "epoch": epoch,
                "train/mse": train_metrics["mse"],
                "train/rmse": train_metrics["rmse"],
                "train/hr_rmse": train_metrics["hr_rmse"],
                "train/rsu_pearson": train_metrics["rsu_pearson"],
                "train/rsd_pearson": train_metrics["rsd_pearson"],
                "optimizer/lr": lr_after_scheduler,
                "monitor/rmse": monitor,
                "best/rmse": best_val,
                "early_stop/wait": wait,
            }

            for spec in train_loader_specs[1:]:
                metrics = train_metrics_by_name[spec["name"]]
                prefix = spec["wandb_prefix"]
                wandb_metrics.update(
                    {
                        f"{prefix}/mse": metrics["mse"],
                        f"{prefix}/rmse": metrics["rmse"],
                        f"{prefix}/hr_rmse": metrics["hr_rmse"],
                        f"{prefix}/rsu_pearson": metrics["rsu_pearson"],
                        f"{prefix}/rsd_pearson": metrics["rsd_pearson"],
                    }
                )

            if val_loader is not None:
                wandb_metrics.update(
                    {
                        "val/mse": val_metrics["mse"],
                        "val/rmse": val_metrics["rmse"],
                        "val/hr_rmse": val_metrics["hr_rmse"],
                        "val/rsu_pearson": val_metrics["rsu_pearson"],
                        "val/rsd_pearson": val_metrics["rsd_pearson"],
                        "val/bias_sfc_dn_flux": val_metrics["surface_dn_bias"],
                    }
                )
            if rfmip_metrics is not None:
                wandb_metrics.update(rfmip_metrics)
            wandb_run.log(wandb_metrics, step=epoch)

        if monitor < best_val:
            best_val = monitor
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            torch.save(
                {
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                    "epoch": epoch,
                    "best_metric_rmse": best_val,
                    "args": vars(args),
                    "data_files": data_files,
                    "primary_data_files": primary_data_files,
                    "primary_nlay": primary_nlay,
                    "extra_training_files": [d["data_file"] for d in extra_training_sets],
                    "extra_training_nlay": [d["nlay"] for d in extra_training_sets],
                    "input_names": input_names,
                    "kdist_str": kdist_str,
                    "xmin": xmin,
                    "xmax": xmax,
                    # --- everything needed to reconstruct mlp_gasopt_inlined_processing ---
                    "do_norm": args.do_norm,
                    "rrtmgp_band_bounds": rrtmgp_bounds,          # Explicit reference/RRTMGP bounds
                    "band_bounds": band_bounds,                    # List[int] or None
                    "train_on_bands": train_on_bands,
                },
                args.output,
            )
         
            if shortwave:
              abs_path = args.output.replace(".pt", "_abs.pt")
              ray_path = args.output.replace(".pt", "_ray.pt")
            else:
              raise NotImplementedError()

            torch.save({
                "model_state_dict": {
                    k.removeprefix("gas_optics_model_sw_abs."): v
                    for k, v in best_state.items()
                    if k.startswith("gas_optics_model_sw_abs.")
                },
                "do_norm":     args.do_norm,
                "band_bounds": band_bounds,
            }, abs_path)
            torch.save({
                "model_state_dict": {
                    k.removeprefix("gas_optics_model_sw_ray."): v
                    for k, v in best_state.items()
                    if k.startswith("gas_optics_model_sw_ray.")
                },
                "do_norm":     args.do_norm,
                "band_bounds": band_bounds,
            }, ray_path)
            # print(f"Saved absorption sub-model to {abs_path}")
            # print(f"Saved Rayleigh sub-model to {ray_path}")
            if wandb_run is not None:
                wandb_run.summary["best_metric_rmse"] = best_val
                wandb_run.summary["best_epoch"] = epoch
                wandb_run.summary["checkpoint_path"] = args.output

        else:
            wait += 1
            if args.patience > 0 and wait >= args.patience:
                print(f"Early stopping after {epoch} epochs; best RMSE = {best_val:.6f}")
                break

    model.load_state_dict(best_state)
    print(f"Saved best checkpoint to {args.output}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()