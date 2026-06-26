#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the PyTorch shortwave radiation model against reference fluxes.

This script loads:
  - a NetCDF evaluation dataset containing shortwave gas-optics inputs,
    auxiliary radiation inputs, and reference fluxes
  - existing gas-optics NetCDF models for shortwave absorption and Rayleigh
  - the PyTorch shortwave radiation model SW_rad_torch

It then runs inference and compares the predicted fluxes against reference
shortwave fluxes loaded from the same file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
from gpt_ml_load_save_preproc import load_rrtmgp
from gpt_torch_models_rad import SW_rad_torch, load_gas_optics_from_file
from torchinfo import summary

def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    diff = predictions - targets
    return float(np.mean(np.abs(diff)))


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(((predictions - targets) ** 2).mean()))


def bias(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(predictions - targets))


def maxabs(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.max(np.abs(predictions - targets)))


def calc_heatingrate(fluxup: np.ndarray, fluxdn: np.ndarray, pres_level: np.ndarray) -> np.ndarray:
    """Heating rate in K/day from up/down fluxes and pressure levels."""
    F = fluxdn - fluxup
    dF = F[:, 1:] - F[:, :-1]
    dp = pres_level[:, 1:] - pres_level[:, :-1]
    dFdp = dF / dp
    g = 9.81
    cp = 1004
    dTdt = -(g / cp) * dFdp
    return (24 * 3600) * dTdt


def _fmt(name: str, mae_v: float, rmse_v: float, bias_v: float, max_v: float) -> str:
    return (
        f"{name:>12s} | MAE {mae_v:.2f} | RMSE {rmse_v:.2f} | "
        f"Bias {bias_v:.2f} | MaxAbs {max_v:.2f}"
    )


def compare_fluxes(
    rsu_pred: np.ndarray,
    rsd_pred: np.ndarray,
    rsd_dir_pred: np.ndarray,
    rsu_ref: np.ndarray,
    rsd_ref: np.ndarray,
    rsd_dir_ref: np.ndarray,
    pres_level: np.ndarray,
) -> None:
    """Print a compact validation summary."""
    print("\n================ SHORTWAVE FLUX VALIDATION ================")
    print(f"Samples: {rsu_ref.shape[0]:d}, levels: {rsu_ref.shape[1]:d}")

    flux_sw_pred = rsd_pred - rsu_pred
    flux_sw_ref = rsd_ref - rsu_ref

    print(" Flux_up mean ref {:.2f} pred {:.2f}.".format(rsu_ref.mean(), rsu_pred.mean()))
    print(" Flux_dn mean ref {:.2f} pred {:.2f}.".format(rsd_ref.mean(), rsd_pred.mean()))
    print(" Flux_dn_dir mean ref {:.2f} pred {:.2f}.".format(rsd_dir_ref.mean(), rsd_dir_pred.mean()))

    print("\nFlux errors")
    print(_fmt("rsu", mae(rsu_pred, rsu_ref), rmse(rsu_pred, rsu_ref), bias(rsu_pred, rsu_ref), maxabs(rsu_pred, rsu_ref)))
    print(_fmt("rsd", mae(rsd_pred, rsd_ref), rmse(rsd_pred, rsd_ref), bias(rsd_pred, rsd_ref), maxabs(rsd_pred, rsd_ref)))
    print(_fmt("rsd_dir", mae(rsd_dir_pred, rsd_dir_ref), rmse(rsd_dir_pred, rsd_dir_ref), bias(rsd_dir_pred, rsd_dir_ref), maxabs(rsd_dir_pred, rsd_dir_ref)))
    print(_fmt("net_flux", mae(flux_sw_pred, flux_sw_ref), rmse(flux_sw_pred, flux_sw_ref), bias(flux_sw_pred, flux_sw_ref), maxabs(flux_sw_pred, flux_sw_ref)))
    print(_fmt("surf_rsd", mae(rsd_pred[:, -1:], rsd_ref[:, -1:]), rmse(rsd_pred[:, -1:], rsd_ref[:, -1:]), bias(rsd_pred[:, -1:], rsd_ref[:, -1:]), maxabs(rsd_pred[:, -1:], rsd_ref[:, -1:])))
    

    print("\nHeating-rate errors")
    dTdt_pred = calc_heatingrate(rsu_pred, rsd_pred, pres_level)
    dTdt_ref = calc_heatingrate(rsu_ref, rsd_ref, pres_level)
    print(_fmt("hr", mae(dTdt_pred, dTdt_ref), rmse(dTdt_pred, dTdt_ref), bias(dTdt_pred, dTdt_ref), maxabs(dTdt_pred, dTdt_ref)))

    pres_lay = 0.5 * (pres_level[:, 1:] + pres_level[:, :-1])
    inds_trop = pres_lay > 10000.0
    inds_strat = pres_lay < 10000.0
    print(
        f"Troposphere HR bias: {bias(dTdt_pred[inds_trop], dTdt_ref[inds_trop]): .4e} | "
        f"MAE: {mae(dTdt_pred[inds_trop], dTdt_ref[inds_trop]): .4e}"
    )
    print(
        f"Stratosphere HR bias: {bias(dTdt_pred[inds_strat], dTdt_ref[inds_strat]): .4e} | "
        f"MAE: {mae(dTdt_pred[inds_strat], dTdt_ref[inds_strat]): .4e}"
    )


def run_evaluation(
    data_file: str,
    gasopt_abs_file: str,
    gasopt_ray_file: str,
    device_str: str | None = None,
) -> Dict[str, np.ndarray]:


    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # # CHANGE ORDER: load_gas_optics_from_file first so we can get the input norm coeffs, 
    # # change those to numpy, and provide those to load_rrtmgp
    # x, y_ref, col_dry, input_names, kdist_str, aux = load_rrtmgp(
    #     data_file,
    #     predictand="sw_fluxes",
    #     load_fluxes=True,
    # )
    gas_abs, gas_ray = load_gas_optics_from_file(device, gasopt_abs_file, gasopt_ray_file)
    gas_abs.eval()
    gas_ray.eval()

    input_norm_coeffs = (
        gas_abs.xmin.detach().cpu().numpy(),
        gas_abs.xmax.detach().cpu().numpy(),
    )

    x, y_ref, col_dry, input_names, kdist_str, aux = load_rrtmgp(
        data_file,
        predictand="sw_fluxes",
        load_fluxes=True,
        input_norm_coefficients=input_norm_coeffs,
    )

    print(f"Loaded dataset: {Path(data_file).name}")
    print(f"k-dist: {kdist_str}")
    # print(f"x shape: {x.shape}, col_dry shape: {col_dry.shape}, y_ref shape: {y_ref.shape}")

    rad_model = SW_rad_torch(
        device=device,
        gas_optics_model_sw_abs=gas_abs,
        gas_optics_model_sw_ray=gas_ray,
    ).to(device)

    print(rad_model)
    infostr = summary(rad_model)
    rad_model.eval()

    x_t = torch.from_numpy(np.asarray(x)).float().to(device)
    col_dry_t = torch.from_numpy(np.asarray(col_dry)).float().to(device)
    mu0_t = torch.from_numpy(np.asarray(aux["mu0"])).float().to(device)
    toa_t = torch.from_numpy(np.asarray(aux["total_solar_irradiance"])).float().to(device)
    pres_t = torch.from_numpy(np.asarray(aux["pres_level"])).float().to(device)
    albedo_t = torch.from_numpy(np.asarray(aux["surface_albedo"])).float().to(device)

    with torch.no_grad():
        dT_pred_t, rsu_pred_t, rsd_pred_t, rsd_dir_pred_t = rad_model(
            x_t,
            col_dry_t,
            mu0_t,
            toa_t,
            pres_t,
            albedo_t,
            albedo_t,
        )

    rsu_pred = rsu_pred_t.detach().cpu().numpy()
    rsd_pred = rsd_pred_t.detach().cpu().numpy()
    rsd_dir_pred = rsd_dir_pred_t.detach().cpu().numpy()

    # rsu_ref = y_ref[:, :, 0]
    # rsd_ref = y_ref[:, :, 1]
    # rsd_dir_ref = y_ref[:, :, 2]
    rsu_ref, rsd_ref, rsd_dir_ref = y_ref

    pres_level = np.asarray(aux["pres_level"])

    compare_fluxes(
        rsu_pred=rsu_pred,
        rsd_pred=rsd_pred,
        rsd_dir_pred=rsd_dir_pred,
        rsu_ref=rsu_ref,
        rsd_ref=rsd_ref,
        rsd_dir_ref=rsd_dir_ref,
        pres_level=pres_level,
    )

    # mu0 = aux["mu0"]
    # rsd_dir_ref_sfc = rsd_dir_ref[:,-1]
    # rsd_dir_pred_sfc = rsd_dir_pred[:,-1]
    # inds_pos = rsd_dir_ref_sfc > 1.0
    # err1 = (rsd_dir_pred_sfc[inds_pos] - rsd_dir_ref_sfc[inds_pos]) / rsd_dir_ref_sfc[inds_pos]
    # mu0 = mu0[inds_pos]
    # print("corrcoef mu0, sfc rsd_dir diff: {}".format(np.corrcoef(mu0,err1)[0,1]))

    return {
        "rsu_pred": rsu_pred,
        "rsd_pred": rsd_pred,
        "rsd_dir_pred": rsd_dir_pred,
        "rsu_ref": rsu_ref,
        "rsd_ref": rsd_ref,
        "rsd_dir_ref": rsd_dir_ref,
        "pres_level": pres_level,
        "dT_pred": dT_pred_t.detach().cpu().numpy(),
        "x": np.asarray(x),
        "y_ref": np.asarray(y_ref),
        "input_names": np.asarray(input_names, dtype=object),
        "kdist_str": kdist_str,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SW_rad_torch against reference fluxes.")
    parser.add_argument("--data-file", required=True, help="NetCDF file with inputs and reference fluxes")
    parser.add_argument("--gasopt-abs", required=True, help="NetCDF shortwave absorption gas optics model")
    parser.add_argument("--gasopt-ray", required=True, help="NetCDF shortwave Rayleigh gas optics model")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda, cuda:0, or cpu")
    args = parser.parse_args()

    run_evaluation(
        data_file=args.data_file,
        gasopt_abs_file=args.gasopt_abs,
        gasopt_ray_file=args.gasopt_ray,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
