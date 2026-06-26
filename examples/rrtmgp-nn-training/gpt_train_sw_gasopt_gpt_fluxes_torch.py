#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train new shortwave gas-optics MLPs through the differentiable PyTorch
shortwave radiation model, using spectral/g-point fluxes as the target.

This script intentionally does not modify gpt_ml_load_save_preproc.py or
 gpt_torch_models_rad.py. It uses their existing load_rrtmgp(),
load_gas_optics_from_file(), mlp_gasopt_inlined_processing, and SW_rad_torch.

The training target from load_rrtmgp(..., predictand="sw_gpt_fluxes") is assumed
to be a tuple:
    y_ref = (rsu_gpt, rsd_gpt, rsd_dir_gpt)
with each array shaped (nbatch, nlev, ng_ref).

The model prediction is reduced with my_reduction(). For now this trains on
broadband fluxes, i.e. a sum over the spectral/g-point dimension. The commented
section in my_reduction() shows where to replace this with custom band sums.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import wandb
except ImportError:
    wandb = None
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchinfo import summary

from gpt_ml_load_save_preproc import load_rrtmgp
import gpt_torch_models_rad as radlib
from gpt_torch_models_rad import (
    SW_rad_torch,
    load_gas_optics_from_file,
    mlp_gasopt_inlined_processing,
)
from coefficients import RRTMGP_SPLITS#, WAVENUM_SPLITS

# train_on_bands=True 

def make_band_bounds(
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
    rsu_gpt: torch.Tensor,      # (batch, nlev, 112) or (batch, nlev, ng)
    rsd_gpt: torch.Tensor,
    rsd_dir_gpt: torch.Tensor, train_on_bands=True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reduce g-point fluxes to the 5-band hybrid spectral bands before computing loss.
    Works for both RRTMGP true fluxes (ng=112) and ML predicted fluxes (ng=8,12,16,32).

    The 5 bands (in RRTMGP g-point space, 1-indexed) are:
      Band 1:  250–4200  cm-1  (Slingo band 4, thermal NIR)         → g-pts  1–29   (RRTMGP bands 1–3)
      Band 2:  4200–14500 cm-1 (Slingo bands 2+3, solar NIR / H2O)  → g-pts 30–80   (RRTMGP bands 4–9)
      Band 3:  14500–16000 cm-1 (PAR boundary / ~0.7 µm)            → g-pts 81–89   (RRTMGP band 10)
      Band 4:  16000–22000 cm-1 (Chappuis O3, visible)              → g-pts 90–102  (RRTMGP bands 11–12)
      Band 5:  22000–50000 cm-1 (UV, Hartley/Huggins O3)            → g-pts 103–112 (RRTMGP bands 13–14)

    For ng=112: boundaries are exact (derived from RRTMGP bnd_limits_gpt).
    For ng<112:  boundaries are scaled proportionally, matching the approach
                 used in slingo_liq_cloud_optics_sw().

    Returns:
        rsu, rsd, rsd_dir: each shaped (batch, nlev, 5)
    """
    if not train_on_bands:
      rsu = rsu_gpt.sum(dim=-1)
      rsd = rsd_gpt.sum(dim=-1)
      rsd_dir = rsd_dir_gpt.sum(dim=-1)
      return rsu, rsd, rsd_dir
    else:
      ng = rsu_gpt.shape[-1]

      # -------------------------------------------------------------------------
      # Band boundary g-point indices for RRTMGP (ng=112), 0-indexed, exclusive
      # upper bounds (i.e. Python slice notation: band_k = gpt[..., lb:ub])
      # Derived from bnd_limits_gpt in RRTMGP:
      #   Band 1:  g-pts  1–29   → 0:29
      #   Band 2:  g-pts 30–80   → 29:80
      #   Band 3:  g-pts 81–89   → 80:89
      #   Band 4:  g-pts 90–102  → 89:102
      #   Band 5:  g-pts 103–112 → 102:112
      # To adjust band boundaries, edit these four split points (in ng=112 space):
      # RRTMGP_SPLITS = [29, 80, 89, 102]  # 4 interior boundaries → 5 bands
      # RRTMGP_SPLITS now loaded from coefficients

      NG_RRTMGP = 112
      # -------------------------------------------------------------------------

      if ng == NG_RRTMGP:
          splits = RRTMGP_SPLITS
      else:
          # Scale boundaries proportionally, matching slingo_liq_cloud_optics_sw()
          splits = [int(round((s / NG_RRTMGP) * ng)) for s in RRTMGP_SPLITS]

      # Build slice boundaries: [0] + splits + [ng]
      bounds = [0] + splits + [ng]
      # print("splits", splits, "bounds", bounds)
          
      def band_sum(x):
          # x: (batch, nlev, ng) → (batch, nlev, 5)
          return torch.cat(
              [x[..., bounds[i]:bounds[i+1]].sum(dim=-1, keepdim=True)
              for i in range(len(bounds) - 1)],
              dim=-1,
          )

      return band_sum(rsu_gpt), band_sum(rsd_gpt), band_sum(rsd_dir_gpt)

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
    device: torch.device,
    eps: float = 1.0,
) -> torch.Tensor:
    """
    Compute per-band loss weights as the inverse variance of band fluxes across
    the training set.  Weights are normalised so their mean equals 1, keeping
    the overall loss magnitude comparable to the unweighted case.

    Args:
        y_ref:  tuple of (rsu, rsd, rsd_dir) numpy arrays, each (nobs, nlev, 112).
                These are the full RRTMGP g-point fluxes before any reduction.
        device: target torch device.
        eps:    floor added to variance before inversion to avoid division by
                near-zero variance (e.g. UV band at night-time).

    Returns:
        weights: float32 tensor of shape (nband,) = (5,) on `device`.
    """
    # Reduce the full training set to band fluxes (CPU, done once)
    rsu_t   = torch.as_tensor(np.asarray(y_ref[0]),   dtype=torch.float32)
    rsd_t   = torch.as_tensor(np.asarray(y_ref[1]),   dtype=torch.float32)
    rsd_dir_t = torch.as_tensor(np.asarray(y_ref[2]), dtype=torch.float32)

    rsu_b, rsd_b, rsd_dir_b = my_reduction(rsu_t, rsd_t, rsd_dir_t, train_on_bands=True)  # each (nobs, nlev, nband)

    # Variance across all (nobs * nlev) samples for each band, averaged over the
    # three flux components so a single weight applies to all three.
    def _band_var(x):
        # x: (nobs, nlev, nband) → var over obs+lev dims → (nband,)
        return x.reshape(-1, x.shape[-1]).var(dim=0)

    var = (_band_var(rsu_b) + _band_var(rsd_b) + _band_var(rsd_dir_b)) / 3.0  # (nband,)

    weights = 1.0 / (var + eps)
    weights = weights / weights.mean()   # normalise so mean weight == 1

    print("Band flux variances:", var.tolist())
    print("Band loss weights:  ", weights.tolist())

    return weights.to(device)

def flux_loss(
    pred_tuple,
    true_tuple,
    band_weights: torch.Tensor | None = None,
    alpha: float = 0.0,
    pres: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """
    MSE loss over band-reduced fluxes, optionally weighted per band.
    If alpha > 0, adds a broadband heating-rate MSE term:
        L = (1 - alpha) * flux_loss + alpha * hr_loss

    band_weights: (nband,) tensor; if None, all bands are weighted equally.
    alpha:        weight of heating-rate loss component (0 = flux only).
    pres:         pressure levels (batch, nlev+1), required when alpha > 0.
    """
    rsu_p, rsd_p, rsd_dir_p = my_reduction(*pred_tuple)
    rsu_t, rsd_t, rsd_dir_t = my_reduction(*true_tuple)

    # print("Mean of pred rsd_bands across batch,lev ", rsd_p.mean(dim=(0,1)))
    # print("Mean of pred rsd_dir_bands across batch,lev ", rsd_dir_p.mean(dim=(0,1)))

    def _mse(p, t):
        sq = (p - t) ** 2                          # (batch, nlev, nband) or (batch, nlev)
        if band_weights is not None and sq.ndim == 3:
            sq = sq * band_weights                 # broadcast over batch and nlev
        return sq.mean()

    loss_rsu = _mse(rsu_p, rsu_t)
    loss_rsd = _mse(rsd_p, rsd_t)
    loss_dir = _mse(rsd_dir_p, rsd_dir_t)
    loss_flux = (loss_rsu + loss_rsd + loss_dir) / 3.0

    if alpha == 0.0:
        return loss_flux

    # Heating-rate loss on broadband fluxes (sum over all g-points / bands)
    assert pres is not None, "pres must be provided when alpha > 0"
    rsu_p_bb = pred_tuple[0].sum(dim=-1)   # (batch, nlev)
    rsd_p_bb = pred_tuple[1].sum(dim=-1)
    rsu_t_bb = true_tuple[0].sum(dim=-1)
    rsd_t_bb = true_tuple[1].sum(dim=-1)

    hr_p = calc_heatingrate_torch(rsu_p_bb, rsd_p_bb, pres)   # (batch, nlev-1)
    hr_t = calc_heatingrate_torch(rsu_t_bb, rsd_t_bb, pres)
    loss_hr = torch.mean((hr_p - hr_t) ** 2)
    # print("flux term mean", torch.mean((1.0 - alpha) * loss_flux ), "hr", torch.mean(alpha * loss_hr) )
    return (1.0 - alpha) * loss_flux + alpha * loss_hr

# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------

# def _set_uniform_solar_weights(model: nn.Module, ng: int, trainable: bool) -> None:
#     """
#     SW_rad_torch currently reads gas_abs.sw_solar_weights directly when gas
#     optics modules are supplied. For newly initialized gas-optics models, make
#     sure those weights are non-zero before training.
#     """
#     value = torch.full((1, ng), 1.0 / float(ng), device=next(model.parameters()).device)
#     if hasattr(model, "sw_solar_weights") and isinstance(model.sw_solar_weights, nn.Parameter):
#         with torch.no_grad():
#             model.sw_solar_weights.copy_(value)
#         model.sw_solar_weights.requires_grad = bool(trainable)
#     else:
#         model.sw_solar_weights = nn.Parameter(value, requires_grad=bool(trainable))

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
        solar_source=None,
        rrtmgp_bounds_in=rrtmgp_band_bounds,
        band_bounds=band_bounds,  
        lock_weights=False,
        ny=ng,
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
        solar_source=None,
        rrtmgp_bounds_in=rrtmgp_band_bounds,
        band_bounds=band_bounds,
        lock_weights=False,
        ny=ng,
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, rsu_pred, rsd_pred, rsd_dir_pred = model(
        x,
        col_dry,
        mu0,
        toa,
        pres,
        albedo,
        albedo,
    )
    batch_size = x.shape[0]
    nlev = pres.shape[1]
    return rsu_pred, rsd_pred, rsd_dir_pred


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_input_norm_coeffs(
    gasopt_abs_file: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load input normalisation coefficients from an existing gas optics file."""
    gas_abs_dummy = load_gas_optics_from_file(device, gasopt_abs_file)
    coeffs = (
        gas_abs_dummy.xmin.detach().cpu().numpy(),
        gas_abs_dummy.xmax.detach().cpu().numpy(),
    )
    del gas_abs_dummy
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return coeffs

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


def run_epoch(
    *,
    model: SW_rad_torch,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float | None,
    band_weights: torch.Tensor | None = None,
    alpha: float = 0.0,
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

            y_pred_fluxes = forward_fluxes(model, x, col_dry, mu0, toa, pres, albedo)
            # loss = flux_mse_loss(y_pred_fluxes, y_true_fluxes)

            # if band_weights is not None:
            #   loss = flux_loss(y_pred_fluxes, y_true_fluxes, band_weights=band_weights)
            # else:
            #   loss = flux_loss(y_pred_fluxes, y_true_fluxes)
            loss = flux_loss(
                y_pred_fluxes,
                y_true_fluxes,
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
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SW gas-optics models through SW_rad_torch using spectral flux targets."
    )
    parser.add_argument("--data-file", required=True, help="NetCDF file containing sw_gpt_fluxes training data")
    parser.add_argument("--gasopt-abs-file", required=True, help="Existing SW absorption gas-optics NetCDF model; used for input normalization only")
    # parser.add_argument("--gasopt-ray-file", required=True, help="Existing SW Rayleigh gas-optics NetCDF model; used for input normalization only")
    parser.add_argument("--output", default=None, help="Output PyTorch checkpoint (default: auto-generated from hyperparameters)")
    parser.add_argument("--device", default=None, help="Device string, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--ng", type=int, default=16, help="Number of learned spectral/g-point channels in the new gas-optics models")
    parser.add_argument("--nh", type=int, default=32, help="Hidden neurons in each gas-optics MLP hidden layer")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
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
                f"sw_gasopt"
                f"_bnd{split_str}"
                f"_ng{ng_band_str}"
                f"_nh{args.nh}"
                f"_alpha{args.alpha:.2f}.pt"
            )
        else:
            args.output = f"sw_gasopt_ng{ng}_nh{args.nh}_alpha{args.alpha:.2f}.pt"

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

    x_all = []
    y_ref_all = [[], [], []]
    col_dry_all = []
    aux_all = []

    input_names = None
    kdist_str = None
    # input_norm_coeffs = None

    input_norm_coeffs = load_input_norm_coeffs(args.gasopt_abs_file, device)
    # xmin, xmax = input_norm_coeffs

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
                raise ValueError(f"Input names differ for {data_file}; refusing to concatenate datasets.")

        x_all.append(np.asarray(x_i))
        col_dry_all.append(np.asarray(col_dry_i))
        aux_all.append(aux_i)

        y_ref_all[0].append(np.asarray(y_ref_i[0]))
        y_ref_all[1].append(np.asarray(y_ref_i[1]))
        y_ref_all[2].append(np.asarray(y_ref_i[2]))

        print(f"Loaded dataset {i + 1}/{len(data_files)}: {Path(data_file).name}")
        print(f"  x shape: {np.asarray(x_i).shape}")
        print(f"  col_dry shape: {np.asarray(col_dry_i).shape}")
        print(
            "  target flux shapes: "
            f"rsu={np.asarray(y_ref_i[0]).shape}, "
            f"rsd={np.asarray(y_ref_i[1]).shape}, "
            f"rsd_dir={np.asarray(y_ref_i[2]).shape}"
        )

    x = np.concatenate(x_all, axis=0)
    col_dry = np.concatenate(col_dry_all, axis=0)
    y_ref = (
        np.concatenate(y_ref_all[0], axis=0),
        np.concatenate(y_ref_all[1], axis=0),
        np.concatenate(y_ref_all[2], axis=0),
    )
    aux = _concat_aux_dicts(aux_all)

    xmin, xmax = input_norm_coeffs

    print(f"Loaded {len(data_files)} dataset(s)")
    print(f"k-dist: {kdist_str}")
    print(f"combined x shape: {np.asarray(x).shape}")
    print(f"combined col_dry shape: {np.asarray(col_dry).shape}")
    print(
        "combined target flux shapes: "
        f"rsu={np.asarray(y_ref[0]).shape}, "
        f"rsd={np.asarray(y_ref[1]).shape}, "
        f"rsd_dir={np.asarray(y_ref[2]).shape}"
    )

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

    # if train_on_bands:
    #   print("Computing per-band flux weights from training data...")
    #   band_weights = compute_band_flux_weights(y_ref, device=device)
    #   ng = args.ng
    #   splits = [int(round((s / 112) * ng)) for s in RRTMGP_SPLITS]
    #   # Build slice boundaries: [0] + splits + [ng]
    #   band_bounds = [0] + splits + [ng]
    # else:
    #   band_weights = None 
    #   band_bounds = None

    if train_on_bands:
        print("Computing per-band flux weights from training data...")
        band_weights = compute_band_flux_weights(y_ref, device=device)

        rrtmgp_bounds, band_bounds, ng = make_band_bounds(
            rrtmgp_splits=args.rrtmgp_splits,
            ng_per_band=args.ng_per_band,
            ng_ref=112,
        )

        print(f"RRTMGP reference band bounds: {rrtmgp_bounds}")
        print(f"Learned model band bounds:    {band_bounds}")
        print(f"Learned model ng:             {ng}")

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
                "kdist_str": kdist_str,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    wait = 0

    print("Beginning training, saving model to {} when new validation loss is reached".format(args.output))

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            band_weights=band_weights,
            alpha=args.alpha,
        )

        if val_loader is not None:
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
                grad_clip=None,
                band_weights=band_weights,
                alpha=args.alpha,
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
            monitor = train_metrics["rmse"]
            val_str = ""

        print(
            f"Epoch {epoch:04d}/{args.epochs}"
            f" - train_rmse: {train_metrics['rmse']:.2f}"
            f" - train_mse: {train_metrics['mse']:.2f}"
            f" - train_hr_rmse: {train_metrics['hr_rmse']:.2f}"
            f" - train_rsu_r: {train_metrics['rsu_pearson']:.5f}"
            f" - train_rsd_r: {train_metrics['rsd_pearson']:.5f}"
            f"{val_str}"
        )

        if wandb_run is not None:
            wandb_metrics = {
                "epoch": epoch,
                "train/mse": train_metrics["mse"],
                "train/rmse": train_metrics["rmse"],
                "train/hr_rmse": train_metrics["hr_rmse"],
                "train/rsu_pearson": train_metrics["rsu_pearson"],
                "train/rsd_pearson": train_metrics["rsd_pearson"],
                "monitor/rmse": monitor,
                "best/rmse": best_val,
                "early_stop/wait": wait,
            }
            if val_loader is not None:
                wandb_metrics.update(
                    {
                        "val/mse": val_metrics["mse"],
                        "val/rmse": val_metrics["rmse"],
                        "val/hr_rmse": val_metrics["hr_rmse"],
                        "val/rsu_pearson": val_metrics["rsu_pearson"],
                        "val/rsd_pearson": val_metrics["rsd_pearson"],
                    }
                )
            wandb_run.log(wandb_metrics, step=epoch)

        if monitor < best_val:
            best_val = monitor
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            torch.save(
                {
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_metric_rmse": best_val,
                    "args": vars(args),
                    "data_files": data_files,
                    "input_names": input_names,
                    "kdist_str": kdist_str,
                    "xmin": xmin,
                    "xmax": xmax,
                    # --- everything needed to reconstruct mlp_gasopt_inlined_processing ---
                    "do_norm": args.do_norm,
                    #"rrtmgp_splits": rrtmgp_splits,               # List[int] from coefficients.py
                    "rrtmgp_band_bounds":rrtmgp_bounds, #  [0] + RRTMGP_SPLITS + [112], # List[int], derived but save explicitly
                    "band_bounds": band_bounds,                    # List[int] or None
                    "train_on_bands": train_on_bands,
                },
                args.output,
            )
         
            abs_path = args.output.replace(".pt", "_abs.pt")
            ray_path = args.output.replace(".pt", "_ray.pt")
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