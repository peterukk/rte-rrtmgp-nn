#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python code for developing neural networks to replace RRTMGP look up tables
    
This program takes existing input-output data generated with RRTMGP and
user-specified hyperparameters such as the number of neurons, 
scales the data, and trains a neural network

Currently supported predictands are g-point vectors containing: 
    - (LW) Planck fraction, absorption cross-section, or both
    - (SW) absorption cross-section or Rayleigh cross-section

-------------
Optional: monitor heating rate and flux/forcing errors (with respect to LBL)
during training, and early stop when those metrics have stopped improving. 
This works by running RTE+RRTNGP-NN on RFMIP data at the end of every epoch,
using the NN model that is being trained. The RFMIP Fortran program prints
some custom metrics to stdout that is then read by Python
If not predicting combined LW Planck frac + absorption vectors, the variable
that is not predicted by the currently trained model needs to be predicted using
an existing model (by default loaded from ../../neural/data/$othervar_BEST). 

To enable this, set early_stop_on_rfmip_fluxes as True and run the script from 
rte-rrtmgp-nn/examples/rrtmgp-nn-training

Models are saved to ../../neural/data with a file name containing the custom
radiation scores.
---------------

---------------
Scaling method: currently fixed so that inputs are min-max scaled to (0..1) and 
outputs are standardized to ~zero mean, ~unit variance
However, power transformations are first used to reduce dynamical range 
and make the distributions more Gaussian (can improve model convergence)

Input preprocessing:
  1. x(i) = log(x(i)) for input feature i = pressure
  2. x(i) = x(i)**(1/4) for i = H2O and O3
  3. x(i) = x(i) - max(i) / (max(x(i) - min(x(i)))) for all i

Output preprocessing:_
  1. Normalize optical depths (g-point vectors) by layer number of molecules
     y(ig,isample) = y_raw(ig,isample) / N (isample) 
  2. y = y**(1/8)
  3. ynorm = (y - ymean) / ystd, where ymeans are means for individual
  g-points, but ystd is the standard deviation across all g-points 
  (preserves relationships between outputs)
----------------
  
This script has only been tested interactively  (from Spyder) and has 
several options defined in the beginning of the program which should be changed 
to fit user needs.

Contributions to e.g. add more options, new datasets, or clean up the code 
are very welcome!

@author: Peter Ukkonen
"""
import os
import sys
import numpy as np

# Keras 3 backend selection must happen before importing keras.
os.environ.setdefault('KERAS_BACKEND', 'torch')

import keras
from keras import losses, optimizers, layers, Input, Model
from keras.callbacks import EarlyStopping
from keras import ops

from ml_load_save_preproc import save_model_netcdf, \
    load_rrtmgp, scale_outputs_wrapper, \
    preproc_pow_standardization_reverse,\
    preproc_tau_to_crossection, preproc_minmax_inputs_rrtmgp
from ml_scaling_coefficients import xcoeffs_all, input_names_all
from ml_trainfuncs_keras3_torch import expdiff, hybrid_loss_wrapper, RunRadiationScheme


def create_model_mlp(nx, ny, neurons, activ, kernel_init='glorot_uniform'):
    """Keras 3 / backend-agnostic MLP used for this training script."""
    if len(activ) != len(neurons) + 1:
        raise ValueError('Number of activations must be number of hidden layers + 1!')

    inputs = Input(shape=(nx,), name='inputs')
    x = inputs
    for i, units in enumerate(neurons):
        x = layers.Dense(
            units,
            activation=activ[i],
            kernel_initializer=kernel_init,
            name=f'dense_{i+1}',
        )(x)
    outputs = layers.Dense(
        ny,
        activation=activ[-1],
        kernel_initializer=kernel_init,
        name='dense_output',
    )(x)
    return Model(inputs=inputs, outputs=outputs, name='mlp_rrtmgp')


def add_dataset(fpath, predictand, expfirst, x, y, col_dry, input_names, kdist, data_str):
    # Concatenate existing dataset (containing raw inputs and outputs) with another
    x_new, y_new, col_dry_new, input_names_new, kdist_new     = load_rrtmgp(fpath, predictand, expfirst=expfirst) 
    if not (kdist==kdist_new):
        print("Kdist does not match previous dataset!")
        return None
    if not (input_names==input_names_new):
        print("Input_names does not match previous dataset!")
        return None
    ns = x.shape[0]
    x = np.concatenate((x,x_new),axis=0)
    y = np.concatenate((y,y_new),axis=0)
    col_dry = np.concatenate((col_dry,col_dry_new),axis=0)
    print("{:.2e} samples previously, {:.2e} after adding data from: {}".format(ns, x.shape[0],fpath.split('/')[-1]))
    data_str = data_str + " , " + fpath.split('/')[-1]
    return x, y, col_dry, data_str

def plot_performance(history, hybrid_loss_expdiffs):
    # Plot the loss and radiation metrics (Figure 2 in paper)
    fs = 12
    import matplotlib.pyplot as plt
    y0 = np.array(history['loss'])
    if hybrid_loss_expdiffs:
        y0e = np.array(history['expdiff'])
        # y0m = np.array(history['mean_squared_error'])
    y1 = np.array(history['radiation_score'])
    y2 = np.array(history['mean_relative_heating_rate_error'])
    # y3 = np.array(history['mean_relative_forcing_error'])
    x1 = np.arange(1,y1.size+1)
    
    if hybrid_loss_expdiffs:
        losslabel = 'Loss (MSE + expdiff)'
    else:
        losslabel = 'Loss (MSE)'
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    label2 = 'Radiation error (heating rate + forcing)'
    # label2 = 'Radiation score (heating rate + forcing errors)'
    label3 = 'Heating rate error'
    c1 = 'r'
    c2 = 'b'
    c2 = 'mediumblue'
    # c3 = 'b'
    c3 = 'dodgerblue'
    
    p1, = ax1.plot(x1, y0, c1, label=losslabel)
    lw = 1.7
    if hybrid_loss_expdiffs:
        p1e, = ax1.plot(x1, y0e, c1, label='Loss (expdiff)', linestyle='dashed')
        # p1m, = ax1.plot(x1, y0m, 'k-.',label=losslabel)
    p2, = ax2.plot(x1, y1, color=c2, label=label2)#, linestyle='dashed')
    # p3_2, = ax2.plot(x1, y3, color='blue',label='Forcing errors w.r.t. LBL', linestyle='dashed')
    p3, = ax2.plot(x1, y2, color=c3,label=label3, linestyle='dashed',linewidth=lw)
    
    
    ax1.set_xlabel('Epochs',fontsize=fs)
    ax1.set_ylabel('Training loss',fontsize=fs)
    ax2.set_ylabel('Normalized errors (w.r.t. LBL)',fontsize=fs)
    
    ax1.set_yscale('log')
    
    _,ymax = ax2.get_ylim(); print(ymax)
    ax2.set_ylim(0.8,ymax)
    # ax2.set_ylim(0.8, 6.172778531908989)
    ax2.grid()
    
    # import mpl_axes_aligner
    # mpl_axes_aligner.align.yaxes(ax2, 1.0, ax3, 1.0, 0.1)
    # lim = ax2.get_ylim()
    # ax2.set_yticks(np.concatenate((ax2.get_yticks(),np.array([1]))))
    # ax2.set_ylim(lim)
        
    ax1.yaxis.label.set_color(p1.get_color())
    ax2.yaxis.label.set_color(p2.get_color())
    
    ax2.axhline(y=1.0, color='k', linestyle='--',linewidth=1)
    xoffset = 1.025
    yoffset = 0.025
    ax2.annotate('= RRTMGP', ha='left',fontsize=11, xy=(xoffset, yoffset), 
                 xycoords='axes fraction',color='blue')

    tkw = dict(size=4, width=1.5)
    ax1.tick_params(axis='y', colors=p1.get_color(), **tkw)
    ax2.tick_params(axis='y', colors=p2.get_color(), **tkw)
    ax1.tick_params(axis='x', **tkw)
    
    if hybrid_loss_expdiffs:
        ax1.legend(handles=[p1, p1e, p2, p3])
    else:
        ax1.legend(handles=[p1, p2, p3])


# ----------------------------------------------------------------------------
# ----------------- Provide data containing inputs and outputs ---------------
# ----------------------------------------------------------------------------
datadir = "/data/rad/RRTMGP_input_output/"

fpath   = datadir+"ml_training_lw_g128_Garand_BIG.nc"
fpath2  = datadir+"ml_training_lw_g128_AMON_ssp245_ssp585_2054_2100.nc"
fpath3   = datadir+"ml_training_lw_g128_CAMS_new_CKDMIPstyle.nc"
fpath4  = datadir+"ml_training_lw_g128_CKDMIP-MMM-Big.nc"

# Let's use the (expanded!) Garand profiles, GCM data (AMON_...), CAMS data,
# and extended CKDMIP-Mean-Maximum-Minimum profiles
# RFMIP ised used for validation
fpaths = [fpath,fpath2,fpath3,fpath4]

fpaths = [fpath]

# ----------------------------------------------------------------------------
# --------------- CONFIGURE: predictand, NN complexity etc -------------------
# ----------------------------------------------------------------------------

# Choose one of the following predictands (target output)
# 'lw_absorption', 'lw_planck_frac', 'sw_absorption', 'sw_rayleigh'

predictand = 'sw_absorption'
# predictand = 'sw_rayleigh'
# predictand = 'lw_absorption'
# predictand = 'lw_planck_frac'
# predictand = 'lw_both' # old

if (predictand=='sw_absorption' or predictand=='sw_rayleigh'):
    fpaths = [sub.replace('lw_g128', 'sw_g112') for sub in fpaths]

scaling_method = 'Ukkonen2020' # only option currently

use_existing_input_scaling_coefficients = True 
# ^True is generally a safe choice, min max coefficients have been computed
# using a large dataset spanning both LGM (Last Glacial Maximum) and high 
# future emissions scenarios. However, check that your scaled inputs 
# fall somewhere in the 0-1 range. Negative values in particular might
# cause problems

# Model training: use CPU or GPU?
use_gpu = False
num_cpu_threads = 12

# Save model to NN model directory (../../neural/data) after training?
# File name includes loss values, so shouldn't override anything
save_new_model = False


# --- Loss function, metrics and early stopping

# --- It would be great if we could directly optimize for fluxes and heating 
# --- rates. For now, let's just monitor them.

# --- Monitor flux and heating rate errors w.r.t. LBL RFMIP data by running the 
# --- radiation scheme with the model as it's being trained, and early stop 
# --- when some custom metrics (printed by the modified RFMIP programs) 
# --- have not improved for a certain number of epochs ("patience")
early_stop_on_rfmip_fluxes = True
# patience    = 30
patience    = 70

if early_stop_on_rfmip_fluxes:
    epochs = 800  # set a high number with early stopping
    # epochs = 221
else:
    epochs = 200

# --- Forcing errors: We can try to reduce radiative forcing errors
# --- by using a hybrid loss function which measures the difference in y 
# --- between adjacent experiments (for instance two experiments where the 
# --- concentration of a single gas is varied from present-day to future)
# --- requires bespoke data but can help minimize TOA / surface forcing errors
hybrid_loss_expdiffs = False


if hybrid_loss_expdiffs:
    if (predictand=='sw_absorption' or predictand=='sw_rayleigh'):
        alpha = 0.2
    else:
        alpha = 0.6
        # alpha = 0.7
        # alpha = 0.75

    loss_expdiff = hybrid_loss_wrapper(alpha=alpha)

    lossfunc = loss_expdiff
    mymetrics   = ['mean_squared_error', expdiff]
    expfirst = True
else:
    lossfunc    = losses.mean_squared_error
    mymetrics   = ['mean_absolute_error']
    expfirst = False

# --- batch size and learning rate 
lr          = 0.001 
# batch_size  = 1024
batch_size  = 2048

# batch_size  = 3*batch_size
# lr          = 2 * lr


# ----NN HYPERPARAMETERS 
# --- Number of neurons in each hidden layer
if predictand == 'lw_absorption':
    # neurons     = [80,80]
    # neurons     = [72,72]
    neurons     = [64,64]
    # neurons     = [58,58]
elif predictand == 'lw_planck_frac':
    neurons     = [24,24]
elif predictand == 'lw_both':
    # neurons     = [80,80]
    neurons     = [72,72]
    # neurons     = [64,64]
    # neurons     = [56,56]
else:
    # neurons     = [16,16] 
    # neurons     = [24,24] 
    neurons     = [32,32] 
    # 16 in two hidden layers seems enough for all but the LW absorption model

# ---  Activation functions used after each layer: first the input layer, and 
# ---  then the hidden layers 
activ = ['softsign', 'softsign','linear']
# activ = ['relu', 'relu','linear']

if np.size(activ) != np.size(neurons)+1:
    print("Number of activations must be number of hidden layers + 1!")
    # exit

# --- Weight initializer: the default is probably an OK choice  (glorot)
initializer = 'glorot_uniform'   
# initializer = 'lecun_uniform'


# ----------------------------------------------------------------------------


# -----------------------------------------------------
# --------- Load data ---------------------------------
# -----------------------------------------------------
# Load training data 
x_tr_raw, y_tr_raw, col_dry_tr, input_names, kdist   = load_rrtmgp(fpaths[0], predictand, expfirst=expfirst) 

data_str = fpath.split('/')[-1]

# We can have different datasets that we merge
for fpath in fpaths[1:]:
    x_tr_raw, y_tr_raw, col_dry_tr, data_str = add_dataset(fpath, predictand, expfirst, x_tr_raw, 
                                    y_tr_raw, col_dry_tr, input_names, kdist, data_str)

nx = x_tr_raw.shape[1] #  temperature + pressure + gases
ny = y_tr_raw.shape[1] #  number of g-points


# In case of hybrid loss measuring diffs between experiments, 
# manually shuffle data in pairs (keeping adjacent experiments)
if hybrid_loss_expdiffs:
    ns = x_tr_raw.shape[0]
    inds_all = np.arange(ns)
    inds_all = inds_all.reshape(int(ns/2),2)
    np.random.shuffle(inds_all)
    inds_all = inds_all.reshape(ns)
    x_tr_raw = x_tr_raw[inds_all,:]
    y_tr_raw = y_tr_raw[inds_all,:]
    col_dry_tr = col_dry_tr[inds_all]
    shuffle = False
else:
    shuffle = True

# -----------------------------------------------------
# -------- Input and output scaling ------------------
# -----------------------------------------------------
if scaling_method != 'Ukkonen2020':
    print ("Only one type of pre-processing currently supported!")
else:
    # Input scaling - min-max
    if use_existing_input_scaling_coefficients:
        if xcoeffs_all == None:
            sys.exit("Input scaling coefficients (xcoeffs) missing!")
        (xmin_all,xmax_all) = xcoeffs_all
        # input_names loaded from file, describes inputs in order of x_tr_raw
        # input_names_all corresponds to xmin_all and xmax_all
        # Order of inputs may be different than in the existing coefficients,
        # account for that by indexing
        a = np.array(input_names_all)
        b = np.array(input_names)
        indices = np.where(b[:, None] == a[None, :])[1]
        xmin = xmin_all[indices]; xmax = xmax_all[indices]
        
        x_tr = preproc_minmax_inputs_rrtmgp(x_tr_raw, (xmin,xmax))
    else:
        x_tr,xmin,xmax  = preproc_minmax_inputs_rrtmgp(x_tr_raw)

    # Output scaling
    # first, do y = y / N if y is optical depth, to get cross-sections
    # then, square root scaling y: y=y**(1/nfac); cheaper and weaker version of 
    # log scaling. nfac = 8 for cross-sections, 2 for Planck fraction
    # After this, use standard-scaling (not for Planck fraction)
    # print("y tr raw max", y_tr_raw.max())
    y_tr, ymean, ystd = scale_outputs_wrapper(y_tr_raw, col_dry_tr, predictand)

# ---------------------------------------------------------------------------

# I/O: RRTMGP-NN models are saved as NetCDF files which contain metadata
# describing how to obtain the physical outputs, as well as the training data
x_scaling_str = "To get the required NN inputs, do the following: "\
        "x(i) = log(x(i)) for i=pressure; "\
        "x(i) = x(i)**(1/4) for i=H2O and O3; "\
        "x(i) = (x(i) - xmin(i)) / (xmax(i) - xmin(i)) for all inputs"
if predictand == 'lw_planck_frac':
    y_scaling_str = "Model predicts the square root of Planck fraction."        
else:
    y_scaling_str = "Model predicts scaled cross-sections. Given the raw NN output y,"\
            " do the following to obtain optical depth: "\
            "y(igpt,j) = ystd(igpt)*y(igpt,j) + ymean(igpt); y(igpt,j) "\
            "= y(igpt,j)**8; y(igpt,j) = y(igpt,j) * layer_dry_air_molecules(j)"
        
# data_str = "Extensive training data set comprising of reanalysis, climate model,"\
#     " and idealized profiles, which has then been augmented using statistical"\
#     " methods (Hypercube sampling). See https://doi.org/10.1029/2020MS002226"

if (predictand == 'sw_absorption'):
    model_str = "Shortwave model predicting ABSORPTION CROSS-SECTION"
elif (predictand == 'sw_rayleigh'):
    model_str = "Shortwave model predicting RAYLEIGH CROSS-SECTION"
elif (predictand == 'lw_absorption'):
    model_str = "Longwave model predicting ABSORPTION CROSS-SECTION"
elif (predictand == 'lw_planck_frac'):
    model_str = "Longwave model predicting PLANCK FRACTION"
else: 
    model_str = ""


# ---------------------------------------------------------------------------
# TENSORFLOW-KERAS TRAINING
#
# ------------------------------------------------------
# --- Setup CPU or GPU training  ----
# ------------------------------------------------------
if use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    # Keep CPU execution predictable when running the PyTorch backend.
    os.environ['OMP_NUM_THREADS'] = str(num_cpu_threads)
    os.environ['KMP_BLOCKTIME'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# optim = optimizers.Adam(lr=lr,rescale_grad=1/batch_size) 
optim = optimizers.Adam(learning_rate=lr)

# Create and compile model
model = create_model_mlp(nx=nx,ny=ny,neurons=neurons,activ=activ,
                         kernel_init=initializer)
model.compile(loss=lossfunc, optimizer=optim, metrics=mymetrics)
model.summary()

# Create earlystopper and possibly other callbacks
if early_stop_on_rfmip_fluxes:

    fpath_save_tmp = '../../neural/data/tmp_model.nc'
    
    if predictand == 'lw_absorption':
        modelinput = '{} ../../neural/data/lw-g128-210809_planck_frac_BEST.nc'.format(fpath_save_tmp)
    elif predictand == 'lw_planck_frac':
        modelinput = '../../neural/data/lw-g128-210809_absorption_BEST.nc {}'.format(fpath_save_tmp)
    elif predictand == 'lw_both':
        modelinput = '{}'.format(fpath_save_tmp)
    elif predictand == 'sw_absorption':
        modelinput = '{} ../../neural/data/sw-g112-210809_rayleigh_BEST.nc'.format(fpath_save_tmp)
    elif predictand == 'sw_rayleigh':
        modelinput = '../../neural/data/sw-g112-210809_absorption_BEST.nc {}'.format(fpath_save_tmp)

    def model_saver(fpath_save_tmp, model):
        save_model_netcdf(fpath_save_tmp, model, activ, input_names, kdist,
                               xmin, xmax, ymean, ystd, y_scaling_comment=y_scaling_str, 
                               x_scaling_comment=x_scaling_str,
                               data_comment=data_str, model_comment=model_str)

    if predictand in ['lw_absorption', 'lw_planck_frac','lw_both']:
        cmd = './rrtmgp_lw_eval_nn_rfmip 8 ../../rrtmgp/data/{}'.format(kdist) + ' 1 1 ' + modelinput
    else:
        cmd = './rrtmgp_sw_eval_nn_rfmip 8 ../../rrtmgp/data/{}'.format(kdist) + ' 1 ' + modelinput

    # out,err = get_stdout(cmd)
    callbacks = [RunRadiationScheme(cmd, modelpath=fpath_save_tmp, 
                                        modelsaver=model_saver,
                                        patience=patience)]
else:
    callbacks = []


# ------------------------------------------------------
# --- Start training -----------------------------------
# ------------------------------------------------------
history = model.fit(x_tr, y_tr, epochs=epochs, batch_size=batch_size,
                    shuffle=shuffle, verbose=1, callbacks=callbacks)
history = history.history

if early_stop_on_rfmip_fluxes:
    plot_performance(history, hybrid_loss_expdiffs)

# ------------------------------------------------------
# --- Save model?  -------------------------------------
# ------------------------------------------------------
model.summary()

def save_model():
    # Get a descriptive filename for the model

    neurons_str = np.array2string(np.array(neurons)).strip('[]').replace(' ','_')
    
    source = kdist[12:].strip('.nc')
    if early_stop_on_rfmip_fluxes:
        ind = np.array(history['radiation_score']).argmin()
        hr_err_final = np.array(history['mean_relative_heating_rate_error'])[ind]
        forcing_err_final = np.array(history['mean_relative_forcing_error'])[ind]
        fpath_keras = "../../neural/data/" + source + "_" + predictand[3:] + "_" + \
          neurons_str + "_HR_{:.2e}_FRC_{:.2e}.keras".format(hr_err_final, forcing_err_final)
    else:
        fpath_keras = "../../neural/data/" + source + "_" + predictand[3:] + "_" + \
            neurons_str + ".keras"
    model.save(fpath_keras)
    
    fpath_netcdf = fpath_keras[:-6]+".nc"
    
    print("Saving model from best epoch in both netCDF and HDF5 format to {}".format(fpath_netcdf))
    save_model_netcdf(fpath_netcdf, model, activ, input_names, kdist,
                           xmin, xmax, ymean, ystd, y_scaling_comment=y_scaling_str, 
                           x_scaling_comment=x_scaling_str,
                           data_comment=data_str, model_comment=model_str)

if save_new_model:
    save_model()

# neurons_str = np.array2string(np.array(neurons)).strip('[]').replace(' ','_')
# source = kdist[12:].strip('.nc')
# ind = np.array(history['radiation_score']).argmin()
# hr_err_final = np.array(history['mean_relative_heating_rate_error'])[ind]
# forcing_err_final = np.array(history['mean_relative_forcing_error'])[ind]
# fp =  source + "_" + predictand[3:] + "_" + \
#   neurons_str + "_HR_{:.2e}_FRC_{:.2e}_history.npy".format(hr_err_final, forcing_err_final)
# np.save('/media/peter/samlinux/gdrive/phd/results/paper3_IFS_RRTMGP/'+fp,history)

# fp2 = '/media/peter/samlinux/gdrive/phd/results/paper3_IFS_RRTMGP/lw-g128-210809_absorption_72_72_HR_1.14e+00_FRC_5.19e-01_history.npy'
# history2 = np.load(fp2,allow_pickle=True)
# history2 = history2.tolist()
# plot_performance(history2, True)


# # ------------------------------------------------------
# # --- Evaluate on another dataset  ---------------------
# # ------------------------------------------------------
# def eval_valdata():
#     y_pred       = model.predict(x_val);  
#     y_pred       = preproc_pow_standardization_reverse(y_pred, nfac, ymean, ystd)

#     if predictand not in ['planck_frac', 'ssa_sw']:
#         y_pred = y_pred * (np.repeat(col_dry_val[:,np.newaxis],ny,axis=1))

#     plot_hist2d(y_val_raw,  y_pred,20,True)   # Optical depth 
#     plot_hist2d_T(y_val_raw,y_pred,20,True)   # Transmittance  

# eval_valdata()
