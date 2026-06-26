#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch version of the old RRTMGP gas-optics training script.

This keeps the original workflow as closely as possible:
- same input/output preprocessing
- same optional hybrid loss
- same custom radiation callback idea (external Fortran program)
- same NetCDF model format for saving and loading

The main change is that the model is now a native torch.nn.Module.
"""

import os
import sys
import copy
import shlex
from subprocess import Popen, PIPE
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary

from ml_load_save_preproc import (
    save_model_netcdf as _legacy_save_model_netcdf,
    load_rrtmgp,
    scale_outputs_wrapper,
    preproc_pow_standardization_reverse,
    preproc_tau_to_crossection,
    preproc_minmax_inputs_rrtmgp,
)
from ml_scaling_coefficients import xcoeffs_all, input_names_all


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def get_stdout(cmd):
    """Execute an external command and return stdout, stderr."""
    args = shlex.split(cmd)
    proc = Popen(args, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    return out.decode("utf-8"), err.decode("utf-8", errors="replace")


def expdiff(y_true, y_pred):
    """Mean absolute difference of adjacent experiment pairs."""
    diff_pred = y_pred[1::2, :] - y_pred[0::2, :]
    diff_true = y_true[1::2, :] - y_true[0::2, :]
    return torch.mean(torch.abs(diff_pred - diff_true))


def hybrid_loss_wrapper(alpha):
    def loss_expdiff(y_true, y_pred):
        err_tot = torch.mean((y_pred - y_true) ** 2)
        err_diff = expdiff(y_true, y_pred)
        return alpha * err_diff + (1.0 - alpha) * err_tot

    return loss_expdiff


def _activation_from_name(name):
    name = str(name).lower()
    if name == "linear":
        return lambda x: x
    if name == "relu":
        return torch.relu
    if name == "softsign":
        return torch.nn.functional.softsign
    if name == "tanh":
        return torch.tanh
    if name == "sigmoid":
        return torch.sigmoid
    if name == "gelu":
        return torch.nn.functional.gelu
    if name == "elu":
        return torch.nn.functional.elu
    if name == "softplus":
        return torch.nn.functional.softplus
    raise ValueError(f"Unsupported activation: {name}")


def _init_linear(linear_layer, kernel_init):
    kernel_init = str(kernel_init).lower()
    if kernel_init in {"glorot_uniform", "xavier_uniform"}:
        nn.init.xavier_uniform_(linear_layer.weight)
    elif kernel_init in {"he_uniform", "kaiming_uniform"}:
        nn.init.kaiming_uniform_(linear_layer.weight, a=np.sqrt(5))
    elif kernel_init == "lecun_uniform":
        nn.init.kaiming_uniform_(linear_layer.weight, a=np.sqrt(3))
    else:
        # Reasonable default.
        nn.init.xavier_uniform_(linear_layer.weight)
    if linear_layer.bias is not None:
        nn.init.zeros_(linear_layer.bias)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

from torch_models_ml import GasOpticsMLP, GasOpticsSwBothMLP
# -----------------------------------------------------------------------------
# NetCDF saving / loading
# -----------------------------------------------------------------------------

def _get_weighted_layers(model):
    """Return the torch Linear layers in order."""
    if hasattr(model, "linear_layers"):
        layers = list(model.linear_layers)
        if layers and all(isinstance(l, nn.Linear) for l in layers):
            return layers
    # fallback: walk modules in definition order
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def save_model_netcdf(
    fpath_netcdf,
    model,
    activation_names,
    input_names,
    emulator_target,
    xmin,
    xmax,
    ymean=None,
    ystd=None,
    y_scaling_comment=None,
    x_scaling_comment=None,
    data_comment=None,
    model_comment=None,
):
    """Save a torch model in the legacy NetCDF format used by RRTMGP."""
    from netCDF4 import Dataset, stringtochar

    weighted_layers = _get_weighted_layers(model)
    if not weighted_layers:
        raise ValueError("No weighted layers found in model")

    # PyTorch Linear weight layout is (out_features, in_features).
    # NetCDF / legacy format expects (in_features, out_features).
    first_weight = weighted_layers[0].weight.detach().cpu().numpy().T
    nx = first_weight.shape[0]
    nlay = len(weighted_layers)

    if len(activation_names) != nlay:
        raise ValueError(
            f"activation_names has length {len(activation_names)}, but model has {nlay} weighted layers"
        )

    input_names = [str(x) for x in input_names]
    if len(input_names) != nx:
        raise ValueError(
            f"input_names has length {len(input_names)}, but model input dimension is {nx}"
        )

    dat_new = Dataset(fpath_netcdf, "w")

    dat_new.createDimension("nn_layers", nlay)
    str_dim_prev = "nn_dim_input"
    dat_new.createDimension(str_dim_prev, nx)

    nc_dimsize = dat_new.createVariable("nn_dimsize", "i4", ("nn_layers",))
    nc_dimsize.long_name = "Dimension of each layer, not including the input layer"
    nc_activ = dat_new.createVariable("nn_activation", "str", ("nn_layers",))
    nc_input = dat_new.createVariable("nn_inputs", "str", (str_dim_prev,))
    nc_input.long_name = "Specifies the inputs in their correct order"
    nc_input_coeffs_max = dat_new.createVariable("nn_input_coeffs_max", "f4", (str_dim_prev,))
    nc_input_coeffs_min = dat_new.createVariable("nn_input_coeffs_min", "f4", (str_dim_prev,))
    nc_input_coeffs_max.long_name = "xmax, see global attribute input_scaling_info"
    nc_input_coeffs_min.long_name = "xmin, see global attribute input_scaling_info"

    dat_new.emulator_target = emulator_target
    if data_comment is not None:
        dat_new.data_info = data_comment
    if model_comment is not None:
        dat_new.model_info = model_comment

    dat_new.createDimension("string_len", 32)
    nc_activ_char = dat_new.createVariable("nn_activation_char", "S1", ("nn_layers", "string_len"))
    nc_input_char = dat_new.createVariable("nn_inputs_char", "S1", (str_dim_prev, "string_len"))
    nc_activ_char[:, :] = " "
    nc_input_char[:, :] = " "

    nc_input_coeffs_max[:] = np.asarray(xmax)
    nc_input_coeffs_min[:] = np.asarray(xmin)

    for i, layer in enumerate(weighted_layers):
        j = i + 1
        weight = layer.weight.detach().cpu().numpy().T
        bias = layer.bias.detach().cpu().numpy()
        dimsize = weight.shape[1]

        if i < nlay - 1:
            str_dim_this = f"nn_dim_hidden{j}"
        else:
            str_dim_this = "nn_dim_outp"
        dat_new.createDimension(str_dim_this, dimsize)

        str_weight = f"nn_weights_{j}"
        str_bias = f"nn_bias_{j}"
        nc_weight = dat_new.createVariable(str_weight, "f4", (str_dim_prev, str_dim_this))
        nc_bias = dat_new.createVariable(str_bias, "f4", (str_dim_this,))

        nc_dimsize[i] = dimsize
        nc_weight[:] = weight
        nc_bias[:] = bias

        activ_str = str(activation_names[i])
        nc_activ[i] = activ_str
        charfmt = f"S{len(activ_str)}"
        activ_chars = stringtochar(np.array(activ_str, charfmt))
        nc_activ_char[i, 0 : len(activ_str)] = activ_chars

        str_dim_prev = str_dim_this

    if ymean is not None and ystd is not None:
        nc_output_coeffs_mean = dat_new.createVariable("nn_output_coeffs_mean", "f4", ("nn_dim_outp",))
        nc_output_coeffs_std = dat_new.createVariable("nn_output_coeffs_std", "f4", ("nn_dim_outp",))
        ymean = np.asarray(ymean)
        ystd = np.asarray(ystd)
        nyy = ymean.size
        nc_output_coeffs_mean[0:nyy] = ymean
        nc_output_coeffs_std[0:nyy] = ystd
        nc_output_coeffs_mean.long_name = "ymean(igpt) = mean(y_cross(igpt)**(1/8))"
        nc_output_coeffs_std.long_name = "ystd(igpt) = std(y_cross(igpt)**(1/8))"

    if y_scaling_comment is not None:
        dat_new.output_scaling_info = y_scaling_comment

    for i in range(nx):
        input_str = input_names[i]
        nc_input[i] = input_str
        charfmt = f"S{len(input_str)}"
        input_chars = stringtochar(np.array(input_str, charfmt))
        nc_input_char[i, 0 : len(input_str)] = input_chars

    if x_scaling_comment is not None:
        dat_new.input_scaling_info = x_scaling_comment

    dat_new.close()


# Backward-compatible alias in case older code imports this name.
save_model_netcdf_torch = save_model_netcdf




def load_gas_optics_model(gasopt_file, device, lock_weights=False):
    """Load a legacy NetCDF gas-optics model into a native torch model."""
    import xarray as xr

    with xr.open_dataset(gasopt_file) as ds:
        input_names = [str(x) for x in np.atleast_1d(ds["nn_inputs"].values)]
        xmin = ds["nn_input_coeffs_min"].values.astype(np.float32)
        xmax = ds["nn_input_coeffs_max"].values.astype(np.float32)

        nn_w1 = ds["nn_weights_1"].values.astype(np.float32)
        nn_w2 = ds["nn_weights_2"].values.astype(np.float32)
        nn_w3 = ds["nn_weights_3"].values.astype(np.float32)
        nn_b1 = ds["nn_bias_1"].values.astype(np.float32)
        nn_b2 = ds["nn_bias_2"].values.astype(np.float32)
        nn_b3 = ds["nn_bias_3"].values.astype(np.float32)

        ymean = ds["nn_output_coeffs_mean"].values.astype(np.float32) if "nn_output_coeffs_mean" in ds else None
        ystd = ds["nn_output_coeffs_std"].values.astype(np.float32) if "nn_output_coeffs_std" in ds else None

        activations = [str(a) for a in np.atleast_1d(ds["nn_activation"].values)]

    if len(activations) != 3:
        raise ValueError(
            f"This loader expects a 3-layer Dense MLP netCDF file, but found {len(activations)} weighted layers"
        )

    nx = nn_w1.shape[0]
    nh1 = nn_w1.shape[1]
    nh2 = nn_w2.shape[1]
    ny = nn_w3.shape[1]

    model = GasOpticsMLP(nx=nx, ny=ny, neurons=[nh1, nh2], activ=activations, kernel_init="glorot_uniform")

    # Load weights. NetCDF stores Keras-style kernels as (in, out), but torch uses (out, in).
    with torch.no_grad():
        model.linear_layers[0].weight.copy_(torch.from_numpy(nn_w1.T))
        model.linear_layers[1].weight.copy_(torch.from_numpy(nn_w2.T))
        model.linear_layers[2].weight.copy_(torch.from_numpy(nn_w3.T))
        model.linear_layers[0].bias.copy_(torch.from_numpy(nn_b1))
        model.linear_layers[1].bias.copy_(torch.from_numpy(nn_b2))
        model.linear_layers[2].bias.copy_(torch.from_numpy(nn_b3))

    if lock_weights:
        for p in model.parameters():
            p.requires_grad = False

    model.to(device)
    model.eval()
    return model, xmin, xmax, ymean, ystd, input_names, activations


def load_sw_both_model(abs_file, ray_file, device, lock_weights=False):
    """Load two legacy SW models and wrap them for joint training."""
    abs_model, xmin_a, xmax_a, ymean_a, ystd_a, input_names_a, activations_a = load_gas_optics_model(
        abs_file, device=device, lock_weights=lock_weights
    )
    ray_model, xmin_r, xmax_r, ymean_r, ystd_r, input_names_r, activations_r = load_gas_optics_model(
        ray_file, device=device, lock_weights=lock_weights
    )

    if input_names_a != input_names_r:
        raise ValueError("SW absorption and Rayleigh models use different input_names.")
    if not np.allclose(xmin_a, xmin_r) or not np.allclose(xmax_a, xmax_r):
        raise ValueError("SW absorption and Rayleigh models use different input scaling coefficients.")

    model = GasOpticsSwBothMLP(abs_model, ray_model).to(device)
    return {
        "model": model,
        "abs_model": abs_model,
        "ray_model": ray_model,
        "xmin": xmin_a,
        "xmax": xmax_a,
        "input_names": input_names_a,
        "ymean_abs": ymean_a,
        "ystd_abs": ystd_a,
        "ymean_ray": ymean_r,
        "ystd_ray": ystd_r,
        "activations_abs": activations_a,
        "activations_ray": activations_r,
    }
# -----------------------------------------------------------------------------
# Radiation callback (external Fortran program)
# -----------------------------------------------------------------------------

class RunRadiationScheme:
    def __init__(self, cmd, modelpath, modelsaver, patience=5, interval=1):
        self.interval = interval
        self.cmd = cmd
        self.modelpath = modelpath
        self.modelsaver = modelsaver
        self.patience = patience
        self.best_weights = None
        self.err_metrics_rrtmgp = None
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.inf
        self.best_epoch = 0

    def on_train_begin(self):
        print(
            "Using RunRadiationScheme earlystopper, fluxes are validated against Line-By-Line benchmark (RFMIP),\n"
            "and training stopped when a weighted mean of the metrics printed by the radiation program have\n"
            f"not improved for {self.patience} epochs"
        )
        print(f"The temporary model is saved to {self.modelpath}")

        cmd_ref = self.cmd[0:75]
        out, err = get_stdout(cmd_ref)
        outstr = out.split("--------")
        if len(outstr) < 3:
            raise RuntimeError(
                "Could not parse reference metrics from the radiation code output. "
                "Check cmd_ref and the external executable output format."
            )
        err_metrics_str = outstr[2].strip("\n")
        err_metrics_str = err_metrics_str.split(",")
        self.err_metrics_rrtmgp = np.float32(err_metrics_str)

    def on_epoch_end(self, epoch, model, logs):
        if epoch % self.interval != 0:
            return False

        self.modelsaver(self.modelpath, model)

        out, err = get_stdout(self.cmd)
        outstr = out.split("--------")
        if len(outstr) < 3:
            raise RuntimeError(
                "Could not parse radiation metrics. Check the external executable output format."
            )
        metric_names = outstr[1].strip("\n").split(",")
        metric_names = [m.lstrip().rstrip() for m in metric_names]
        err_metrics_str = outstr[2].strip("\n").split(",")
        err_metrics = np.float32(err_metrics_str)
        err_metrics = err_metrics / self.err_metrics_rrtmgp

        indices = [i for i, elem in enumerate(metric_names) if "HR" in elem]
        ind_forc = indices[-1] + 1

        hr_err = np.sqrt(np.mean(np.square(err_metrics[0:ind_forc])))
        forcing_err = np.sqrt(np.mean(np.square(err_metrics[ind_forc:])))
        score = np.sqrt(np.mean(np.square(err_metrics)))

        logs["mean_relative_heating_rate_error"] = float(err_metrics[0])
        logs["mean_relative_forcing_error"] = float(forcing_err)
        logs["radiation_score"] = float(score)

        print(
            "The RFMIP accuracy relative to RRTGMP was:   {:.2f}   (HR {:.2f}, FLUXES/FORCINGS {:.2f})".format(
                score, hr_err, forcing_err
            )
        )
        for i in range(len(err_metrics)):
            if i == len(err_metrics) - 1:
                print(f"{metric_names[i]}: {err_metrics[i]:.2f} \n", end=" ")
            else:
                print(f"{metric_names[i]}: {err_metrics[i]:.2f}, ", end=" ")

        current = logs.get("radiation_score")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            self.best_weights = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(
                    "Early stopping, the best radiation score (comprised of LBL heating rate and forcing errors "
                    f"normalized by RRTGMP values) was {self.best:.2f}"
                )
                self.stopped_epoch = epoch
                print(f"Restoring model weights from the end of the best epoch ({self.best_epoch + 1})")
                model.load_state_dict(self.best_weights)
                return True

        return False

    def on_train_end(self):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


# -----------------------------------------------------------------------------
# Data helpers and plotting
# -----------------------------------------------------------------------------


def add_dataset(fpath, predictand, expfirst, x, y, col_dry, input_names, kdist, data_str):
    x_new, y_new, col_dry_new, input_names_new, kdist_new = load_rrtmgp(fpath, predictand, expfirst=expfirst)
    if not (kdist == kdist_new):
        print("Kdist does not match previous dataset!")
        return None
    if not (input_names == input_names_new):
        print("Input_names does not match previous dataset!")
        return None
    ns = x.shape[0]
    x = np.concatenate((x, x_new), axis=0)
    y = np.concatenate((y, y_new), axis=0)
    col_dry = np.concatenate((col_dry, col_dry_new), axis=0)
    print(f"{ns:.2e} samples previously, {x.shape[0]:.2e} after adding data from: {fpath.split('/')[-1]}")
    data_str = data_str + " , " + fpath.split('/')[-1]
    return x, y, col_dry, data_str


def plot_performance(history, hybrid_loss_expdiffs):
    import matplotlib.pyplot as plt

    fs = 12
    y0 = np.array(history["loss"])
    if hybrid_loss_expdiffs:
        y0e = np.array(history["expdiff"])
    y1 = np.array(history["radiation_score"])
    y2 = np.array(history["mean_relative_heating_rate_error"])
    x1 = np.arange(1, y1.size + 1)

    losslabel = "Loss (MSE + expdiff)" if hybrid_loss_expdiffs else "Loss (MSE)"
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    c1 = "r"
    c2 = "mediumblue"
    c3 = "dodgerblue"

    p1, = ax1.plot(x1, y0, c1, label=losslabel)
    if hybrid_loss_expdiffs:
        p1e, = ax1.plot(x1, y0e, c1, label="Loss (expdiff)", linestyle="dashed")
    p2, = ax2.plot(x1, y1, color=c2, label="Radiation error (heating rate + forcing)")
    p3, = ax2.plot(x1, y2, color=c3, label="Heating rate error", linestyle="dashed", linewidth=1.7)

    ax1.set_xlabel("Epochs", fontsize=fs)
    ax1.set_ylabel("Training loss", fontsize=fs)
    ax2.set_ylabel("Normalized errors (w.r.t. LBL)", fontsize=fs)
    ax1.set_yscale("log")
    _, ymax = ax2.get_ylim()
    ax2.set_ylim(0.8, ymax)
    ax2.grid()
    ax1.yaxis.label.set_color(p1.get_color())
    ax2.yaxis.label.set_color(p2.get_color())
    ax2.axhline(y=1.0, color="k", linestyle="--", linewidth=1)
    ax2.annotate("= RRTMGP", ha="left", fontsize=11, xy=(1.025, 0.025), xycoords="axes fraction", color="blue")

    tkw = dict(size=4, width=1.5)
    ax1.tick_params(axis="y", colors=p1.get_color(), **tkw)
    ax2.tick_params(axis="y", colors=p2.get_color(), **tkw)
    ax1.tick_params(axis="x", **tkw)

    if hybrid_loss_expdiffs:
        ax1.legend(handles=[p1, p1e, p2, p3])
    else:
        ax1.legend(handles=[p1, p2, p3])


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

datadir = "/data/rad/RRTMGP_input_output/"

fpath = datadir + "ml_training_lw_g128_Garand_BIG.nc"
fpath2 = datadir + "ml_training_lw_g128_AMON_ssp245_ssp585_2054_2100.nc"
fpath3 = datadir + "ml_training_lw_g128_CAMS_new_CKDMIPstyle.nc"
fpath4 = datadir + "ml_training_lw_g128_CKDMIP-MMM-Big.nc"
fpaths = [fpath, fpath2, fpath3, fpath4]
fpaths = [fpath] 

predictand = "sw_absorption"
predictand = "sw_both"
# predictand = 'sw_rayleigh'
# predictand = 'lw_absorption'
# predictand = 'lw_planck_frac'
# predictand = 'lw_both'

if predictand in ("sw_absorption", "sw_rayleigh", "sw_both"):
    fpaths = [sub.replace("lw_g128", "sw_g112") for sub in fpaths]

scaling_method = "Ukkonen2020"
use_existing_input_scaling_coefficients = True

use_gpu = False
num_cpu_threads = 12

save_new_model = False
use_existing_model = False
existing_gasopt_file = ""
existing_gasopt_file_sw_abs = "../../neural/data/sw-g112-210809_absorption_BEST.nc"
existing_gasopt_file_sw_ray = "../../neural/data/sw-g112-210809_rayleigh_BEST.nc"

early_stop_on_rfmip_fluxes = True
patience = 70
if early_stop_on_rfmip_fluxes:
    epochs = 800
else:
    epochs = 200

hybrid_loss_expdiffs = False
if hybrid_loss_expdiffs:
    if predictand in ("sw_absorption", "sw_rayleigh", "sw_both"):
        alpha = 0.2
    else:
        alpha = 0.6
    lossfunc = hybrid_loss_wrapper(alpha=alpha)
    expfirst = True
else:
    lossfunc = lambda y_true, y_pred: torch.mean((y_pred - y_true) ** 2)
    expfirst = False

lr = 0.001
batch_size = 2048

if predictand == "lw_absorption":
    neurons = [64, 64]
elif predictand == "lw_planck_frac":
    neurons = [24, 24]
elif predictand == "lw_both":
    neurons = [72, 72]
else:
    neurons = [32, 32]

activ = ["softsign", "softsign", "linear"]
initializer = "glorot_uniform"



# -----------------------------------------------------------------------------
# Data loading and preprocessing
# -----------------------------------------------------------------------------

if predictand == "sw_both":
    x_tr_abs_raw, y_tr_abs_raw, col_dry_tr_abs, input_names_abs, kdist_abs = load_rrtmgp(
        fpaths[0], "sw_absorption", expfirst=expfirst
    )
    x_tr_ray_raw, y_tr_ray_raw, col_dry_tr_ray, input_names_ray, kdist_ray = load_rrtmgp(
        fpaths[0], "sw_rayleigh", expfirst=expfirst
    )

    if input_names_abs != input_names_ray:
        raise ValueError("SW absorption and Rayleigh datasets use different input_names.")
    if kdist_abs != kdist_ray:
        raise ValueError("SW absorption and Rayleigh datasets use different kdist values.")
    if not np.allclose(x_tr_abs_raw, x_tr_ray_raw):
        raise ValueError("SW absorption and Rayleigh datasets use different x inputs.")
    if not np.allclose(col_dry_tr_abs, col_dry_tr_ray):
        raise ValueError("SW absorption and Rayleigh datasets use different col_dry values.")

    data_str = fpaths[0].split("/")[-1]
    data_str_abs = data_str
    data_str_ray = data_str

    for fpath_i in fpaths[1:]:
        x_tr_abs_raw, y_tr_abs_raw, col_dry_tr_abs, data_str_abs = add_dataset(
            fpath_i, "sw_absorption", expfirst, x_tr_abs_raw, y_tr_abs_raw, col_dry_tr_abs, input_names_abs, kdist_abs, data_str_abs
        )
        x_tr_ray_raw, y_tr_ray_raw, col_dry_tr_ray, data_str_ray = add_dataset(
            fpath_i, "sw_rayleigh", expfirst, x_tr_ray_raw, y_tr_ray_raw, col_dry_tr_ray, input_names_ray, kdist_ray, data_str_ray
        )
        if not np.allclose(x_tr_abs_raw, x_tr_ray_raw):
            raise ValueError("Merged SW absorption and Rayleigh datasets use different x inputs.")
        if not np.allclose(col_dry_tr_abs, col_dry_tr_ray):
            raise ValueError("Merged SW absorption and Rayleigh datasets use different col_dry values.")

    x_tr_raw = x_tr_abs_raw
    y_tr_abs, ymean_abs, ystd_abs = scale_outputs_wrapper(y_tr_abs_raw, col_dry_tr_abs, "sw_absorption")
    y_tr_ray, ymean_ray, ystd_ray = scale_outputs_wrapper(y_tr_ray_raw, col_dry_tr_abs, "sw_rayleigh")
    y_tr = np.concatenate((y_tr_abs, y_tr_ray), axis=1)
    y_tr_raw = np.concatenate((y_tr_abs_raw, y_tr_ray_raw), axis=1)
    col_dry_tr = col_dry_tr_abs
    input_names = input_names_abs
    kdist = kdist_abs
    ymean = np.concatenate((ymean_abs, ymean_ray))
    ystd = np.concatenate((ystd_abs, ystd_ray))
    ny_abs = y_tr_abs.shape[1]
    ny_ray = y_tr_ray.shape[1]
    ny = ny_abs + ny_ray
    data_str = data_str_abs + " , " + data_str_ray
else:
    x_tr_raw, y_tr_raw, col_dry_tr, input_names, kdist = load_rrtmgp(fpaths[0], predictand, expfirst=expfirst)
    data_str = fpaths[0].split("/")[-1]

    for fpath_i in fpaths[1:]:
        x_tr_raw, y_tr_raw, col_dry_tr, data_str = add_dataset(
            fpath_i, predictand, expfirst, x_tr_raw, y_tr_raw, col_dry_tr, input_names, kdist, data_str
        )

    ny = y_tr_raw.shape[1]

# In case of hybrid loss measuring diffs between experiments,
# manually shuffle data in pairs (keeping adjacent experiments)
if hybrid_loss_expdiffs:
    ns = x_tr_raw.shape[0]
    inds_all = np.arange(ns)
    inds_all = inds_all.reshape(int(ns / 2), 2)
    np.random.shuffle(inds_all)
    inds_all = inds_all.reshape(ns)
    x_tr_raw = x_tr_raw[inds_all, :]
    y_tr_raw = y_tr_raw[inds_all, :]
    col_dry_tr = col_dry_tr[inds_all]
    shuffle = False
else:
    shuffle = True

nx = x_tr_raw.shape[1]

# -----------------------------------------------------
# -------- Input and output scaling ------------------
# -----------------------------------------------------
if scaling_method != "Ukkonen2020":
    raise RuntimeError("Only one type of pre-processing currently supported!")

# Input scaling - min-max
if use_existing_input_scaling_coefficients:
    if xcoeffs_all is None:
        sys.exit("Input scaling coefficients (xcoeffs) missing!")
    xmin_all, xmax_all = xcoeffs_all
    a = np.array(input_names_all)
    b = np.array(input_names)
    indices = np.where(b[:, None] == a[None, :])[1]
    xmin = xmin_all[indices]
    xmax = xmax_all[indices]
    x_tr = preproc_minmax_inputs_rrtmgp(x_tr_raw, (xmin, xmax))
else:
    x_tr, xmin, xmax = preproc_minmax_inputs_rrtmgp(x_tr_raw)

# Output scaling
if predictand != "sw_both":
    y_tr, ymean, ystd = scale_outputs_wrapper(y_tr_raw, col_dry_tr, predictand)

x_scaling_str = (
    "To get the required NN inputs, do the following: "
    "x(i) = log(x(i)) for i=pressure; "
    "x(i) = x(i)**(1/4) for i=H2O and O3; "
    "x(i) = (x(i) - xmin(i)) / (xmax(i) - xmin(i)) for all inputs"
)
if predictand == "lw_planck_frac":
    y_scaling_str = "Model predicts the square root of Planck fraction."
elif predictand == "sw_both":
    y_scaling_str = (
        "Model predicts scaled cross-sections for both SW absorption and Rayleigh. "
        "Given the raw NN outputs y, do the following to obtain optical depth: "
        "y(igpt,j) = ystd(igpt)*y(igpt,j) + ymean(igpt); y(igpt,j) "
        "= y(igpt,j)**8; y(igpt,j) = y(igpt,j) * layer_dry_air_molecules(j)"
    )
else:
    y_scaling_str = (
        "Model predicts scaled cross-sections. Given the raw NN output y,"
        " do the following to obtain optical depth: "
        "y(igpt,j) = ystd(igpt)*y(igpt,j) + ymean(igpt); y(igpt,j) "
        "= y(igpt,j)**8; y(igpt,j) = y(igpt,j) * layer_dry_air_molecules(j)"
    )

if predictand == "sw_absorption":
    model_str = "Shortwave model predicting ABSORPTION CROSS-SECTION"
elif predictand == "sw_rayleigh":
    model_str = "Shortwave model predicting RAYLEIGH CROSS-SECTION"
elif predictand == "sw_both":
    model_str_abs = "Shortwave model predicting ABSORPTION CROSS-SECTION"
    model_str_ray = "Shortwave model predicting RAYLEIGH CROSS-SECTION"

elif predictand == "lw_absorption":
    model_str = "Longwave model predicting ABSORPTION CROSS-SECTION"
elif predictand == "lw_planck_frac":
    model_str = "Longwave model predicting PLANCK FRACTION"
else:
    model_str = ""
# -----------------------------------------------------------------------------
# Torch setup
# -----------------------------------------------------------------------------

if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    torch.set_num_threads(num_cpu_threads)

train_ds = TensorDataset(
    torch.from_numpy(x_tr).float().to(device),
    torch.from_numpy(y_tr).float().to(device),
)
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=hybrid_loss_expdiffs,
)

# In this codebase there is no separate validation set in the pasted script.
# Keep the interface for future extension, but train/early-stop via the radiation callback.
val_loader = None

# -----------------------------------------------------------------------------
# Model creation / loading
# -----------------------------------------------------------------------------

if predictand == "sw_both":
    if use_existing_model:
        print(f"Loading pre-existing shortwave !ABSORPTION! gas optics model from {existing_gasopt_file_sw_abs}")
        abs_model, xmin_abs_file, xmax_abs_file, ymean_abs_file, ystd_abs_file, input_names_abs_file, activations_abs_file = load_gas_optics_model(
            existing_gasopt_file_sw_abs, device=device, lock_weights=False
        )
        print(f"Loading pre-existing shortwave !RAYLEIGH! gas optics model from {existing_gasopt_file_sw_ray}")
        ray_model, xmin_ray_file, xmax_ray_file, ymean_ray_file, ystd_ray_file, input_names_ray_file, activations_ray_file = load_gas_optics_model(
            existing_gasopt_file_sw_ray, device=device, lock_weights=False
        )
        model = GasOpticsSwBothMLP(abs_model, ray_model).to(device)
    else:
        abs_model = GasOpticsMLP(nx=nx, ny=ny_abs, neurons=neurons, activ=activ, kernel_init=initializer)
        ray_model = GasOpticsMLP(nx=nx, ny=ny_ray, neurons=neurons, activ=activ, kernel_init=initializer)
        model = GasOpticsSwBothMLP(abs_model, ray_model).to(device)
else:
    if use_existing_model:
        print(f"Loading pre-existing gas optics model from {existing_gasopt_file}")
        model, xmin_file, xmax_file, ymean_file, ystd_file, input_names_file, activations_file = load_gas_optics_model(
            existing_gasopt_file, device=device, lock_weights=False
        )
    else:
        model = GasOpticsMLP(nx=nx, ny=ny, neurons=neurons, activ=activ, kernel_init=initializer).to(device)
        activations_file = activ

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print(model)
infostr = summary(model)


# -----------------------------------------------------------------------------
# Optional radiation monitor
# -----------------------------------------------------------------------------

callbacks = []
if early_stop_on_rfmip_fluxes:
    if predictand == "sw_both":
        fpath_save_tmp_abs = "../../neural/data/tmp_model_sw_abs.nc"
        fpath_save_tmp_ray = "../../neural/data/tmp_model_sw_ray.nc"
        modelinput = f"{fpath_save_tmp_abs} {fpath_save_tmp_ray}"
        modelpath = (fpath_save_tmp_abs, fpath_save_tmp_ray)
    else:
        fpath_save_tmp = "../../neural/data/tmp_model.nc"
        modelpath = fpath_save_tmp
        if predictand == "lw_absorption":
            modelinput = f"{fpath_save_tmp} ../../neural/data/lw-g128-210809_planck_frac_BEST.nc"
        elif predictand == "lw_planck_frac":
            modelinput = f"../../neural/data/lw-g128-210809_absorption_BEST.nc {fpath_save_tmp}"
        elif predictand == "lw_both":
            modelinput = f"{fpath_save_tmp}"
        elif predictand == "sw_absorption":
            modelinput = f"{fpath_save_tmp} ../../neural/data/sw-g112-210809_rayleigh_BEST.nc"
        elif predictand == "sw_rayleigh":
            modelinput = f"../../neural/data/sw-g112-210809_absorption_BEST.nc {fpath_save_tmp}"
        else:
            raise ValueError(f"Unknown predictand: {predictand}")

    def model_saver(model_path, model_obj):
        if predictand == "sw_both":
            abs_path, ray_path = model_path
            save_model_netcdf(
                abs_path,
                model_obj.abs_model,
                model_obj.abs_model.activation_names,
                input_names,
                kdist,
                xmin,
                xmax,
                ymean=ymean_abs,
                ystd=ystd_abs,
                y_scaling_comment=y_scaling_str,
                x_scaling_comment=x_scaling_str,
                data_comment=data_str,
                model_comment=model_str_abs,
            )
            save_model_netcdf(
                ray_path,
                model_obj.ray_model,
                model_obj.ray_model.activation_names,
                input_names,
                kdist,
                xmin,
                xmax,
                ymean=ymean_ray,
                ystd=ystd_ray,
                y_scaling_comment=y_scaling_str,
                x_scaling_comment=x_scaling_str,
                data_comment=data_str,
                model_comment=model_str_ray,
            )
        else:
            save_model_netcdf(
                model_path,
                model_obj,
                model_obj.activation_names if hasattr(model_obj, "activation_names") else activations_file,
                input_names,
                kdist,
                xmin,
                xmax,
                ymean=ymean,
                ystd=ystd,
                y_scaling_comment=y_scaling_str,
                x_scaling_comment=x_scaling_str,
                data_comment=data_str,
                model_comment=model_str,
            )

    if predictand in ["lw_absorption", "lw_planck_frac", "lw_both"]:
        cmd = f"./rrtmgp_lw_eval_nn_rfmip 8 ../../rrtmgp/data/{kdist} 1 1 {modelinput}"
    else:
        cmd = f"./rrtmgp_sw_eval_nn_rfmip 8 ../../rrtmgp/data/{kdist} 1 {modelinput}"
    print("cmd ", cmd)
    callbacks = [RunRadiationScheme(cmd, modelpath=modelpath, modelsaver=model_saver, patience=patience)]
    callbacks[0].on_train_begin()
# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

history = {
    "loss": [],
    "radiation_score": [],
    "mean_relative_heating_rate_error": [],
    "mean_relative_forcing_error": [],
}
if hybrid_loss_expdiffs:
    history["expdiff"] = []

best_state = copy.deepcopy(model.state_dict())
best_metric = np.inf
best_epoch = 0
patience_wait = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    nsamp = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = lossfunc(yb, pred)
        loss.backward()
        optimizer.step()

        bs = xb.shape[0]
        running_loss += float(loss.detach().cpu()) * bs
        nsamp += bs

    train_loss = running_loss / max(nsamp, 1)
    history["loss"].append(train_loss)

    logs = {"loss": train_loss}
    stop_now = False

    if early_stop_on_rfmip_fluxes:
        stop_now = callbacks[0].on_epoch_end(epoch, model, logs)
        history["radiation_score"].append(logs["radiation_score"])
        history["mean_relative_heating_rate_error"].append(logs["mean_relative_heating_rate_error"])
        history["mean_relative_forcing_error"].append(logs["mean_relative_forcing_error"])
        if hybrid_loss_expdiffs:
            history["expdiff"].append(float(expdiff(yb, pred).detach().cpu()))

        metric = logs["radiation_score"]
    else:
        metric = train_loss

    if metric < best_metric:
        best_metric = metric
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        patience_wait = 0
    else:
        patience_wait += 1

    print(
        f"Epoch {epoch + 1:04d}/{epochs} - loss: {train_loss:.6f}"
        + (f" - radiation_score: {logs['radiation_score']:.4f}" if early_stop_on_rfmip_fluxes else "")
    )

    if stop_now or (not early_stop_on_rfmip_fluxes and patience_wait >= patience):
        print(f"Early stopping triggered at epoch {epoch + 1}.")
        break


model.load_state_dict(best_state)
if early_stop_on_rfmip_fluxes:
    callbacks[0].on_train_end()
    plot_performance(history, hybrid_loss_expdiffs)



# -----------------------------------------------------------------------------
# Save final model
# -----------------------------------------------------------------------------

model.train(False)

def save_model():
    neurons_arr = np.array([layer.out_features for layer in model.abs_model.linear_layers[:-1]]) if predictand == "sw_both" else np.array([layer.out_features for layer in model.linear_layers[:-1]])
    neurons_str = np.array2string(neurons_arr).strip("[]").replace(" ", "_")
    source = kdist[12:].strip(".nc")

    if predictand == "sw_both":
        if early_stop_on_rfmip_fluxes:
            ind = int(np.array(history["radiation_score"]).argmin())
            hr_err_final = float(np.array(history["mean_relative_heating_rate_error"])[ind])
            forcing_err_final = float(np.array(history["mean_relative_forcing_error"])[ind])
            fpath_netcdf_abs = (
                "../../neural/data/"
                + source
                + "_sw_absorption_"
                + neurons_str
                + f"_HR_{hr_err_final:.2e}_FRC_{forcing_err_final:.2e}.nc"
            )
            fpath_netcdf_ray = (
                "../../neural/data/"
                + source
                + "_sw_rayleigh_"
                + neurons_str
                + f"_HR_{hr_err_final:.2e}_FRC_{forcing_err_final:.2e}.nc"
            )
        else:
            fpath_netcdf_abs = (
                "../../neural/data/"
                + source
                + "_sw_absorption_"
                + neurons_str
                + ".nc"
            )
            fpath_netcdf_ray = (
                "../../neural/data/"
                + source
                + "_sw_rayleigh_"
                + neurons_str
                + ".nc"
            )

        print(f"Saving SW-both absorption model to {fpath_netcdf_abs}")
        save_model_netcdf(
            fpath_netcdf_abs,
            model.abs_model,
            model.abs_model.activation_names,
            input_names,
            kdist,
            xmin,
            xmax,
            ymean=ymean_abs,
            ystd=ystd_abs,
            y_scaling_comment=y_scaling_str,
            x_scaling_comment=x_scaling_str,
            data_comment=data_str,
            model_comment=model_str_abs,
        )
        print(f"Saving SW-both Rayleigh model to {fpath_netcdf_ray}")
        save_model_netcdf(
            fpath_netcdf_ray,
            model.ray_model,
            model.ray_model.activation_names,
            input_names,
            kdist,
            xmin,
            xmax,
            ymean=ymean_ray,
            ystd=ystd_ray,
            y_scaling_comment=y_scaling_str,
            x_scaling_comment=x_scaling_str,
            data_comment=data_str,
            model_comment=model_str_ray,
        )
        return

    if early_stop_on_rfmip_fluxes:
        ind = int(np.array(history["radiation_score"]).argmin())
        hr_err_final = float(np.array(history["mean_relative_heating_rate_error"])[ind])
        forcing_err_final = float(np.array(history["mean_relative_forcing_error"])[ind])
        fpath_netcdf = (
            "../../neural/data/"
            + source
            + "_"
            + predictand[3:]
            + "_"
            + neurons_str
            + f"_HR_{hr_err_final:.2e}_FRC_{forcing_err_final:.2e}.nc"
        )
    else:
        fpath_netcdf = (
            "../../neural/data/"
            + source
            + "_"
            + predictand[3:]
            + "_"
            + neurons_str
            + ".nc"
        )

    print(f"Saving model to {fpath_netcdf}")
    save_model_netcdf(
        fpath_netcdf,
        model,
        model.activation_names,
        input_names,
        kdist,
        xmin,
        xmax,
        ymean=ymean,
        ystd=ystd,
        y_scaling_comment=y_scaling_str,
        x_scaling_comment=x_scaling_str,
        data_comment=data_str,
        model_comment=model_str,
    )

if save_new_model:
    save_model()
