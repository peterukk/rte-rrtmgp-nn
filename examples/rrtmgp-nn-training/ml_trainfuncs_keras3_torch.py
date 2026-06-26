#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python framework for developing neural network emulators of
RRTMGP gas optics scheme

Modernized for Keras 3 with the PyTorch backend.

This module keeps the original helper structure but removes TensorFlow-specific
imports and uses backend-agnostic Keras ops where needed.

@author: Peter Ukkonen
"""

from __future__ import annotations

import shlex
from subprocess import PIPE, Popen
from typing import Any, Callable

import h5py
import keras
from keras import losses, optimizers, ops
from keras.callbacks import Callback
from keras.layers import Dense
from keras.models import Sequential
import numpy as np

# Optional legacy imports retained for compatibility with old downstream code
# that may still expect these module-level names to exist.
# They are not required by the functions below.


# err_metrics_rrtmgp_lw = np.array([...])
# Heating rate (all exps), Heating rate (present), SFC forcing (pre-industrial to present),
# SFC forcing (present to future), TOA forcing (present to future),
# TOA forcing CO2 (pre-industrial to 8x), SFC forcing N2O (pre-industrial to present)


def hybrid_loss_wrapper(alpha: float) -> Callable[[Any, Any], Any]:
    """Return a hybrid loss combining total error and adjacent-experiment diffs."""

    def loss_expdiff(y_true, y_pred):
        err_tot = ops.mean(ops.square(y_pred - y_true))
        err_diff = expdiff(y_true, y_pred)
        return alpha * err_diff + (1.0 - alpha) * err_tot

    return loss_expdiff


def expdiff(y_true, y_pred):
    """Mean absolute difference between adjacent experiment pairs."""

    diff_pred = y_pred[1::2, :] - y_pred[0::2, :]
    diff_true = y_true[1::2, :] - y_true[0::2, :]
    return ops.mean(ops.abs(diff_pred - diff_true))


def get_stdout(cmd):
    """Execute the external command and get its exitcode, stdout and stderr."""

    args = shlex.split(cmd)
    proc = Popen(args, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    out = out.decode("utf-8")
    return out, err


class RunRadiationScheme(Callback):
    """Custom callback that runs the radiation solver during training.

    This is kept close to the original implementation so existing training
    scripts continue to work, but it now subclasses keras.callbacks.Callback.
    """

    def __init__(self, cmd, modelpath, modelsaver, patience=5, interval=1):
        super().__init__()
        self.interval = interval
        self.cmd = cmd
        self.modelpath = modelpath
        self.modelsaver = modelsaver
        self.patience = patience
        self.best_weights = None
        self.err_metrics_rrtmgp = None

    def on_train_begin(self, logs=None):
        logs = logs or {}
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.inf
        self.best_epoch = 0

        print(
            "Using RunRadiationScheme earlystopper, fluxes are validated "
            "against Line-By-Line benchmark (RFMIP),\nand training stopped when a "
            "weighted mean of the metrics printed by the radiation program have\n"
            "not improved for {} epochs".format(self.patience)
        )
        print("The temporary model is saved to {}".format(self.modelpath))

        # First run the RRTMGP code without NNs to get the reference errors.
        cmd_ref = self.cmd[0:75]
        print("cmd ref", cmd_ref)
        out, err = get_stdout(cmd_ref)
        outstr = out.split("--------")
        err_metrics_str = outstr[2].strip("\n")
        err_metrics_str = err_metrics_str.split(",")
        self.err_metrics_rrtmgp = np.float32(err_metrics_str)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if epoch % self.interval != 0:
            return

        # Save model, run radiation code, parse output metrics.
        self.modelsaver(self.modelpath, self.model)
        # print("cmd: ", self.cmd)
        out, err = get_stdout(self.cmd)
        outstr = out.split("--------")
        metric_names = outstr[1].strip("\n")
        metric_names = metric_names.split(",")
        for i in range(len(metric_names)):
            metric_names[i] = metric_names[i].lstrip().rstrip()

        err_metrics_str = outstr[2].strip("\n")
        err_metrics_str = err_metrics_str.split(",")
        err_metrics = np.float32(err_metrics_str)
        err_metrics = err_metrics / self.err_metrics_rrtmgp

        indices = [i for i, elem in enumerate(metric_names) if "HR" in elem]
        ind_forc = indices[-1] + 1

        logs["mean_relative_heating_rate_error"] = err_metrics[0]
        forcing_err = np.sqrt(np.mean(np.square(err_metrics[ind_forc:])))
        logs["mean_relative_forcing_error"] = forcing_err
        hr_err = np.sqrt(np.mean(np.square(err_metrics[0:ind_forc])))

        score = np.sqrt(np.mean(np.square(err_metrics)))
        logs["radiation_score"] = score

        print(
            " The RFMIP accuracy relative to RRTGMP was:   {:.2f}   (HR {:.2f}, FLUXES/FORCINGS {:.2f})".format(
                score, hr_err, forcing_err
            )
        )
        for i in range(len(err_metrics)):
            if i == len(err_metrics) - 1:
                print("{}: {:.2f} \n".format(metric_names[i], err_metrics[i]), end=" ")
            else:
                print("{}: {:.2f}, ".format(metric_names[i], err_metrics[i]), end=" ")

        current = logs.get("radiation_score")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(
                    "Early stopping, the best radiation score (comprised of LBL heating rate"
                    " and forcing errors normalized by RRTGMP values) was {:.2f}".format(
                        self.best
                    )
                )
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print(
                    "Restoring model weights from the end of the best epoch ({})".format(
                        self.best_epoch + 1
                    )
                )
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


# 1. Define an objective function to be maximized.
def create_model_hyperopt(trial, nx, ny):
    """Build a Sequential MLP for Optuna-style hyperparameter search."""

    model = Sequential()

    n_layers = trial.suggest_int("n_layers", 1, 3)
    activ0 = trial.suggest_categorical("activation", ["relu", "softsign"])
    num_hidden0 = trial.suggest_int("n_neurons_l0_l", 64, 256)
    model.add(Dense(num_hidden0, input_dim=nx, activation=activ0))

    for i in range(1, n_layers):
        num_hidden = trial.suggest_int(f"n_neurons_l{i}", 64, 256)
        activ = trial.suggest_categorical(f"activation_l{i}", ["relu", "softsign"])
        model.add(Dense(num_hidden, activation=activ))

    model.add(Dense(ny, activation="linear"))

    # Newer Optuna API prefers suggest_float(..., log=True).
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    model.compile(
        loss=losses.mean_squared_error,
        optimizer=optimizers.Adam(learning_rate=lr),
        metrics=["mean_absolute_error"],
    )
    return model



def create_model_mlp(nx, ny, neurons=[40, 40], activ=["softsign", "softsign", "linear"], kernel_init="he_uniform"):
    """Build a plain dense MLP using Keras 3 layers."""

    model = Sequential()
    model.add(keras.Input(shape=(nx,)))
    model.add(Dense(neurons[0], kernel_initializer=kernel_init, activation=activ[0]))
    for i in range(1, np.size(neurons)):
        model.add(Dense(neurons[i], activation=activ[i], kernel_initializer=kernel_init))
    model.add(Dense(ny, activation=activ[-1], kernel_initializer=kernel_init))
    return model



def savemodel(kerasfile, model):
    """Save a Keras model and write the legacy txt export."""

    model.summary()
    newfile = kerasfile[:-3] + ".txt"
    try:
        model.save(kerasfile)
    except Exception:
        # Keep the old behavior of failing softly here.
        pass
    print("saving to {}".format(newfile))
    h5_to_txt(kerasfile, newfile)



def get_available_layers(model_layers, available_model_layers=[b"dense"]):
    parsed_model_layers = []
    for l in model_layers:
        for g in available_model_layers:
            if g in l:
                parsed_model_layers.append(l)
    return parsed_model_layers


# KERAS HDF5 NEURAL NETWORK MODEL FILE TO NEURAL-FORTRAN ASCII MODEL FILE

def h5_to_txt(weights_file_name, output_file_name=""):
    """Convert a legacy Keras HDF5 file to the text format used downstream."""

    with h5py.File(weights_file_name, "r") as weights_file:
        weights_group_key = list(weights_file.keys())[0]

        model_config = weights_file.attrs["model_config"]
        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")
        model_config = model_config.replace("true", "True")
        model_config = model_config.replace("false", "False")
        model_config = model_config.replace("null", "None")
        model_config = eval(model_config)

        model_layers = list(weights_file["model_weights"].attrs["layer_names"])
        print("names of layers in h5 file: %s \n" % model_layers)

        num_model_layers = len(model_layers) + 1

        dimensions = []
        bias = {}
        weights = {}
        activations = []

        print("Processing the following {} layers: \n{}\n".format(len(model_layers), model_layers))
        if "Input" in model_config["config"]["layers"][0]["class_name"]:
            model_config = model_config["config"]["layers"][1:]
        else:
            model_config = model_config["config"]["layers"]

        for num, l in enumerate(model_layers):
            layer_info_keys = list(weights_file[weights_group_key][l][l].keys())
            for key in layer_info_keys:
                if "bias" in key:
                    bias.update({num: np.array(weights_file[weights_group_key][l][l][key])})
                elif "kernel" in key:
                    weights.update({num: np.array(weights_file[weights_group_key][l][l][key])})
                    if num == 0:
                        dimensions.append(str(np.array(weights_file[weights_group_key][l][l][key]).shape[0]))
                        dimensions.append(str(np.array(weights_file[weights_group_key][l][l][key]).shape[1]))
                    else:
                        dimensions.append(str(np.array(weights_file[weights_group_key][l][l][key]).shape[1]))

            if "Dense" in model_config[num]["class_name"]:
                activations.append(model_config[num]["config"]["activation"])
            else:
                print("Skipping bad layer: '{}'\n".format(model_config[num]["class_name"]))

    if not output_file_name:
        output_file_name = weights_file_name.replace(".h5", ".txt")

    with open(output_file_name, "w") as output_file:
        output_file.write(str(num_model_layers) + "\n")
        output_file.write("\t".join(dimensions) + "\n")
        if bias:
            for x in range(len(model_layers)):
                bias_str = "\t".join(list(map(str, bias[x].tolist())))
                output_file.write(bias_str + "\n")
        if weights:
            for x in range(len(model_layers)):
                weights_str = "\t".join(list(map(str, weights[x].T.flatten())))
                output_file.write(weights_str + "\n")
        if activations:
            for a in activations:
                if a == "softmax":
                    print("WARNING: Softmax activation not allowed... Replacing with Linear activation")
                    a = "linear"
                output_file.write(a + "\n")
