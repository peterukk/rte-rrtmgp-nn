#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python framework for developing neural network emulators of RRTMGP gas optics
scheme, the RTE radiative transfer solver, or their combination RTE+RRTMGP (a 
radiative transfer scheme).

This file provides functions for loading and preprocessing data so that it
may be used for training e.g. a neural network version of RRTMGP

@author: Peter Ukkonen
"""
import torch
import torch.nn as nn
import torch.nn.parameter as Parameter
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Final, Optional
import sys
import numpy as np
from numba import jit, njit, prange
from netCDF4 import Dataset


class gasopt_mlp(nn.Module):
    is_longwave: Final[bool]
    lock_weights: Final[bool]
    def __init__(self, device,
                xmin, xmax, ymean, ystd,
                nn_w1, nn_w2, nn_w3,
                nn_b1, nn_b2, nn_b3, 
                lock_weights=True,
                is_longwave=True):
        super(gasopt_mlp, self).__init__()
        self.nx = xmin.shape[0]
        self.ny = ymean.shape[0]
        self.is_longwave=is_longwave
        if self.is_longwave:
            self.ng = self.ny//2
        else:
            self.ng = self.ny
        self.change_last_layer = False
        self.nh = nn_w1.shape[1]
        xmin  = torch.from_numpy(xmin)
        xmax  = torch.from_numpy(xmax)
        xdiv = xmax - xmin
        self.register_buffer('xmin', xmin)
        self.register_buffer('xmax', xmax)
        self.register_buffer('xdiv', xdiv)
        ymean = torch.from_numpy(ymean[0:self.ng])
        ystd  = torch.from_numpy(ystd[0:self.ng])
        self.register_buffer('ymean', ymean)
        self.register_buffer('ystd', ystd)
        self.softsign =  nn.Softsign()
        self.mlp1 = nn.Linear(self.nx, self.nh)
        self.mlp2 = nn.Linear(self.nh, self.nh)
        self.mlp3 = nn.Linear(self.nh, self.ny)
        self.lock_weights=lock_weights 
        print("gasopt_mlp_lw number of g-points: {}, hidden neurons: {}, inputs: {}".format(self.ng, self.nh, self.nx)) 

        self.mlp1.weight = torch.nn.Parameter(torch.from_numpy(nn_w1.T))
        self.mlp2.weight = torch.nn.Parameter(torch.from_numpy(nn_w2.T))
        self.mlp1.bias = torch.nn.Parameter(torch.from_numpy(nn_b1.T))
        self.mlp2.bias = torch.nn.Parameter(torch.from_numpy(nn_b2.T))
        self.mlp3.weight = torch.nn.Parameter(torch.from_numpy(nn_w3.T))
        self.mlp3.bias = torch.nn.Parameter(torch.from_numpy(nn_b3.T))
        if self.lock_weights:
          self.mlp1.weight.requires_grad = False; self.mlp1.bias.requires_grad = False
          self.mlp2.weight.requires_grad = False; self.mlp2.bias.requires_grad = False
          if not self.change_last_layer:
            self.mlp3.weight.requires_grad = False; self.mlp3.bias.requires_grad = False

        self.to(device)

    def forward(self, x, col_dry):
        x = self.mlp1(x)
        x = self.softsign(x)
        x = self.mlp2(x)
        x = self.softsign(x)
        x = self.mlp3(x)

        if self.is_longwave:
            tau, pfrac = x.chunk(2,-1)
            pfrac = torch.square(pfrac)
            if self.change_last_layer:
                pfrac = self.softmax(pfrac)
        else:
            tau = x 

        # if col_dry is not None:
        # print("shape coldry", col_dry.shape, "tau", tau.shape, "ystd", self.ystd.shape)
        tau = col_dry * torch.pow(self.ystd*tau + self.ymean,8)
        # if self.change_last_layer:
        #     tau = 1e-19*tau 
    #    ! Postprocess absorption output: reverse standard scaling and square root scaling
    #    tau(igpt,ilay,icol) = (ystd(igpt) * outp_both(igpt,ilay,icol) + ymeans(igpt))**8
    #    ! Optical depth from cross-sections
    #    tau(igpt,ilay,icol) = tau(igpt,ilay,icol)*col_dry_wk(ilay,icol)
        if self.is_longwave:
            return tau, pfrac
        else:
            return tau
        
def load_gas_optics_model(gasopt_file, device, num_outputs_desired):#, lock_weights):
  import xarray as xr
  ds = xr.open_dataset(gasopt_file) # Open netCDF file with saved weights and normalization coefficients
  input_str = ds.nn_inputs
  if 'cfc11' in input_str.values:
      is_longwave=True 
  else: 
      is_longwave=False 

  nn_w1 = ds['nn_weights_1'][:].values
  nn_w2 = ds['nn_weights_2'][:].values
  nn_w3 = ds['nn_weights_3'][:].values

  nn_b1 = ds['nn_bias_1'][:].values
  nn_b2 = ds['nn_bias_2'][:].values
  nn_b3 = ds['nn_bias_3'][:].values

  ynorm_lw_mean = ds['nn_output_coeffs_mean'][:].values 
  ynorm_lw_std =  ds['nn_output_coeffs_std'][:].values 

  xnorm_lw_max = ds['nn_input_coeffs_max'][:].values 
  xnorm_lw_min =  ds['nn_input_coeffs_min'][:].values 
  # ng = 32
  nn = gasopt_mlp(device, xnorm_lw_min, xnorm_lw_max, 
                      ynorm_lw_mean, ynorm_lw_std,
                      nn_w1, nn_w2, nn_w3,
                      nn_b1, nn_b2, nn_b3, num_outputs_desired=num_outputs_desired, is_longwave=is_longwave)#, lock_weights=lock_weights)
  infostr = summary(nn)
  return nn 


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
    """
    Save a Dense Keras model to the legacy RRTMGP NetCDF format.

    Works with Keras 3 by ignoring non-weighted layers (e.g. InputLayer)
    and by reading weights from the actual Dense layers only.
    """
    from netCDF4 import Dataset, stringtochar
    import numpy as np

    def _layer_weights(layer):
        """Return (kernel, bias) as NumPy arrays, or (None, None) if absent."""
        weights = layer.get_weights()
        if len(weights) < 2:
            return None, None
        return np.asarray(weights[0]), np.asarray(weights[1])

    # Keep only layers that really have kernel/bias weights
    weighted_layers = []
    for layer in model.layers:
        kernel, bias = _layer_weights(layer)
        if kernel is not None and bias is not None:
            weighted_layers.append(layer)

    if not weighted_layers:
        raise ValueError(
            "No weighted layers found. Make sure the model is built and contains Dense layers."
        )

    # Infer input dimension from the first weighted layer
    first_kernel, _ = _layer_weights(weighted_layers[0])
    nx = first_kernel.shape[0]
    nlay = len(weighted_layers)

    # Basic validation
    if len(activation_names) != nlay:
        raise ValueError(
            f"activation_names has length {len(activation_names)}, but model has {nlay} weighted layers."
        )
    if len(input_names) != nx:
        raise ValueError(
            f"input_names has length {len(input_names)}, but model input dimension is {nx}."
        )

    with Dataset(fpath_netcdf, "w", format="NETCDF4") as dat_new:
        # Dimensions
        dat_new.createDimension("nn_layers", nlay)
        str_dim_prev = "nn_dim_input"
        dat_new.createDimension(str_dim_prev, nx)

        # Variables
        nc_dimsize = dat_new.createVariable("nn_dimsize", "i4", ("nn_layers",))
        nc_dimsize.long_name = "Dimension of each layer, not including the input layer"

        nc_activ = dat_new.createVariable("nn_activation", str, ("nn_layers",))
        nc_input = dat_new.createVariable("nn_inputs", str, (str_dim_prev,))
        nc_input.long_name = "Specifies the inputs in their correct order"

        nc_input_coeffs_max = dat_new.createVariable(
            "nn_input_coeffs_max", "f4", (str_dim_prev,)
        )
        nc_input_coeffs_min = dat_new.createVariable(
            "nn_input_coeffs_min", "f4", (str_dim_prev,)
        )
        nc_input_coeffs_max.long_name = "xmax, see global attribute input_scaling_info"
        nc_input_coeffs_min.long_name = "xmin, see global attribute input_scaling_info"

        dat_new.emulator_target = emulator_target
        if data_comment is not None:
            dat_new.data_info = data_comment
        if model_comment is not None:
            dat_new.model_info = model_comment

        # NetCDF Fortran-friendly string storage
        dat_new.createDimension("string_len", 32)
        nc_activ_char = dat_new.createVariable(
            "nn_activation_char", "S1", ("nn_layers", "string_len")
        )
        nc_input_char = dat_new.createVariable(
            "nn_inputs_char", "S1", (str_dim_prev, "string_len")
        )

        nc_activ_char[:, :] = " "
        nc_input_char[:, :] = " "

        nc_input_coeffs_max[:] = np.asarray(xmax)
        nc_input_coeffs_min[:] = np.asarray(xmin)

        # Write layer weights
        for i, layer in enumerate(weighted_layers):
            j = i + 1
            weight, bias = _layer_weights(layer)

            dimsize = weight.shape[1]

            if i < nlay - 1:
                str_dim_this = f"nn_dim_hidden{j}"
            else:
                str_dim_this = "nn_dim_outp"
            dat_new.createDimension(str_dim_this, dimsize)

            str_weight = f"nn_weights_{j}"
            str_bias = f"nn_bias_{j}"
            nc_weight = dat_new.createVariable(
                str_weight, "f4", (str_dim_prev, str_dim_this)
            )
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

        # Output scaling coefficients, if present
        if ymean is not None and ystd is not None:
            nc_output_coeffs_mean = dat_new.createVariable(
                "nn_output_coeffs_mean", "f4", ("nn_dim_outp",)
            )
            nc_output_coeffs_std = dat_new.createVariable(
                "nn_output_coeffs_std", "f4", ("nn_dim_outp",)
            )
            ymean = np.asarray(ymean)
            ystd = np.asarray(ystd)
            nyy = ymean.size
            nc_output_coeffs_mean[0:nyy] = ymean
            nc_output_coeffs_std[0:nyy] = ystd
            nc_output_coeffs_mean.long_name = (
                "ymean(igpt) = mean(y_cross(igpt)**(1/8))"
            )
            nc_output_coeffs_std.long_name = "ystd(igpt) = std(y_cross(igpt)**(1/8))"

        if y_scaling_comment is not None:
            dat_new.output_scaling_info = y_scaling_comment

        for i in range(nx):
            input_str = str(input_names[i])
            nc_input[i] = input_str

            charfmt = f"S{len(input_str)}"
            input_chars = stringtochar(np.array(input_str, charfmt))
            nc_input_char[i, 0 : len(input_str)] = input_chars

        if x_scaling_comment is not None:
            dat_new.input_scaling_info = x_scaling_comment

def load_rrtmgp(fname,predictand, dcol=1, skip_lastlev=False, skip_firstlev=False,expfirst=False):
    # Load data for training a GAS OPTICS (RRTMGP) emulator,
    # where inputs are layer-wise atmospheric conditions (T,p, gas concentrations)
    # and outputs are vectors of optical properties across g-points (e.g. optical depth)
    dat = Dataset(fname)
    
    if predictand not in ['lw_absorption', 'lw_planck_frac', 'sw_absorption', 
                          'sw_rayleigh', 'lw_both']: 
        sys.exit("Second drgument to load_rrtmgp (predictand) " \
        "must be either lw_absorption, lw_planck_frac, sw_absorption, or sw_rayleigh")
            
    # k-distribution info
    try: 
        kdist_str = dat.comment
        kdist_str = kdist_str.split(' ')
        kdist_str = kdist_str[2]
        kdist_str = kdist_str.split('/')
        kdist_str = kdist_str[4]
    except:
        kdist_str = None
        
    # inputs
    if predictand in ["lw_absorption", "lw_planck_frac",'lw_both']: # Longwave
        xname = 'rrtmgp_lw_input'
    else: # Shortwave
        xname = 'rrtmgp_sw_input'
        
    x = dat.variables[xname][:].data
    
    try:
        input_names = dat.variables[xname].comment
        print('input_names found in file')
        input_names = input_names.split(' ')
        if (input_names[0]=='Features:'):
            input_names = input_names[1:]
    except:
        print("input_names not found in file") 
        input_names = None
        
    nx = x.shape[-1]
    
    # outputs
    if (predictand=='sw_rayleigh'):
        ssa = dat.variables['ssa_sw_gas'][:].data
        tau = dat.variables['tau_sw_gas'][:].data
        y = tau * ssa # tau_sw_rayleigh = tau_tot * single scattering albedo
        del tau, ssa
    elif (predictand=='sw_absorption'):
        ssa = dat.variables['ssa_sw_gas'][:].data
        tau = dat.variables['tau_sw_gas'][:].data
        tau_sw_rayleigh = tau * ssa
        y = tau - tau_sw_rayleigh # tay_sw_abs = tau_tot - tau_ray
        del tau, ssa, tau_sw_rayleigh
    elif (predictand=='lw_absorption'):
        y = dat.variables['tau_lw_gas'][:].data
    elif (predictand=='lw_planck_frac'):
        y = dat.variables['planck_fraction'][:].data
    elif (predictand=='lw_both'):
        y = dat.variables['tau_lw_gas'][:].data
        y2 = dat.variables['planck_fraction'][:].data   
        y = np.concatenate((y,y2),axis=-1)
    else:
        y  = dat.variables[predictand][:].data
        
    # if predictand in ['lw_absorption','tau_sw', 'ssa_sw']:
    col_dry = dat.variables['col_dry'][:].data
    
    if col_dry[0,0,0] == 0.0:
        skip_firstlev = True
        
    if np.size(y.shape) == 4:
        (nexp,ncol,nlay,ngpt) = y.shape
    elif np.size(y.shape) == 3:
        (ncol,nlay,ngpt) = y.shape
        nexp = 1
        y = np.reshape(y,(nexp,ncol,nlay,ngpt))
        x = np.reshape(y,(nexp,ncol,nlay,nx))
        col_dry = np.reshape(col_dry,(nexp,ncol,nlay))
    else:
        sys.exit("Invalid array shapes, RRTMGP output should have at least 3 dimensions")
        
    if skip_lastlev:
        x = x[:,:,0:-1,:]; y = y[:,:,0:-1,:]
        col_dry = col_dry[:,:,0:-1]
        nlay = nlay -1
        
    if skip_firstlev:
        x = x[:,:,1:,:]; y = y[:,:,1:,:]
        col_dry = col_dry[:,:,1:]
        nlay = nlay - 1 
        
    if dcol>1:
        y  = y[:,::dcol,:,:]; x  = x[:,::dcol,:,:]
        col_dry = col_dry[:,::dcol,:]

    nobs = nexp*ncol*nlay
    print( "there are {} profiles in this dataset ({} experiments, {} columns)".format(nexp*ncol,nexp,ncol))

    if expfirst:
        print("Reshaping so that adjacent samples are from different experiments")
        x = np.rollaxis(x,0,3)     
        y = np.rollaxis(y,0,3) 
        col_dry = np.rollaxis(col_dry,0,3)
        
    y = np.reshape(y, (nobs,ngpt)); x = np.reshape(x, (nobs,nx))
    col_dry = np.reshape(col_dry,(nobs))
    
    return x,y,col_dry,input_names,kdist_str


def get_col_dry(vmr_h2o, plev):
    grav = 9.80665
    m_dry = 0.028964
    m_h2o =  0.018016
    avogad = 6.02214076e23
    delta_plev = plev[:,1:] - plev[:,0:-1]
    # Get average mass of moist air per mole of moist air
    fact = 1.0 / (1. + vmr_h2o)
    m_air = (m_dry + m_h2o * vmr_h2o) * fact
    col_dry = 10.0 * np.float64(delta_plev) * avogad * np.float64(fact) / (1000.0 * m_air * 100.0 * grav)
    return np.float32(col_dry)

@njit(parallel=True)
def preproc_tau_to_crossection(tau, col_dry):
    y = np.zeros(tau.shape,dtype=np.float32)
    for iobs in range(tau.shape[0]):
        for igpt in range(tau.shape[1]):
            y[iobs,igpt]  = tau[iobs,igpt] / col_dry[iobs]
    return y

@njit(parallel=True)
def preproc_pow_standardization(y, nfac, means,sigma):
    # scale y to y', where y is a data matrix of shape (nsamples, ng) consisting
    # consisting of ng outputs; e.g. g-point vector of absorption cross-sections,
    # and y' has been scaled for more effective neural network training
    # y is first power-scaled by y = y**(1/nfac) and then normalized by
    # y'g =  (y_g - mean(y_g)) / std(y_g), 
    # when training correlated-k gas optics models, g is a single g-point.
    # using means of individual g-points but sigma across g-points 
    # is recommended as it preserves correlations but scales to a common range
    # the means and sigma(s) are input arguments as they need to be fixed 
    # for production
    (nobs,ngpt) = y.shape
    y_scaled = np.zeros(y.shape,dtype=np.float32)

    nfacc = 1/nfac
    for iobs in prange(nobs):
       for igpt in prange(ngpt):
           y_scaled[iobs,igpt] = np.power(y[iobs,igpt],nfacc)
           y_scaled[iobs,igpt] = (y_scaled[iobs,igpt] - means[igpt]) / sigma[igpt]
                
    return y_scaled



@njit(parallel=True)
def preproc_pow_standardization_reverse(y_scaled, nfac, means,sigma):
    # y has shape (nobs,gpts)
    y = np.zeros(y_scaled.shape)

    (nobs,ngpt) = y_scaled.shape
    for iobs in prange(nobs):
        for igpt in prange(ngpt):
            y[iobs,igpt] = (y_scaled[iobs,igpt] * sigma[igpt]) + means[igpt]
            y[iobs,igpt] = np.power(y[iobs,igpt],nfac)

    return y

@njit(parallel=True)
def preproc_standardization(y, means,sigma):
    (nobs,ngpt) = y.shape
    y_scaled = np.copy(y)
    for iobs in prange(nobs):
       for igpt in prange(ngpt):
           y_scaled[iobs,igpt] = (y_scaled[iobs,igpt] - means[igpt]) / sigma[igpt]
                
    return y_scaled

@njit(parallel=True)
def preproc_standardization_reverse(y_scaled, means,sigma):
    # y has shape (nobs,gpts)
    y = np.zeros(y_scaled.shape)
    (nobs,ngpt) = y_scaled.shape
    for iobs in prange(nobs):
        for igpt in prange(ngpt):
            y[iobs,igpt] = (y_scaled[iobs,igpt] * sigma[igpt]) + means[igpt]
            
    return y

def preproc_minmax_inputs(x, xcoeffs=None):
        x_scaled = np.copy(x)
        if xcoeffs is None:
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler()  
            scaler.fit(x_scaled)
            x_scaled = scaler.transform(x_scaled)  
            return x_scaled, scaler.data_min_, scaler.data_max_
        else:
            (xmin,xmax) = xcoeffs
            for i in range(x.shape[1]):
                if (xmax[i] - xmin[i]) == 0.0:
                    x_scaled[:,i] = 0.0
                else:
                    x_scaled[:,i] =  (x_scaled[:,i] - xmin[i]) / (xmax[i] - xmin[i] )
            return x_scaled

def preproc_divbymax(x,xmax=None):
    x_scaled = np.copy(x)
    if xmax is None:
        xmax = np.zeros(x.shape[-1])
        
        if np.size(x.shape)==3:
            for i in range(x.shape[-1]):
                xmax[i] =  np.max(x[:,:,i])
                x_scaled[:,:,i] =  x_scaled[:,:,i] / xmax[i]
        else:
            for i in range(x.shape[-1]):
                xmax[i] =  np.max(x[:,i])
                x_scaled[:,i] =  x_scaled[:,i] / xmax[i]
        return x_scaled, xmax
    else:
        if np.size(x.shape)==3:
            for i in range(x.shape[-1]):
                x_scaled[:,:,i] =  x_scaled[:,:,i] / xmax[i]
        else:
            for i in range(x.shape[-1]):
                x_scaled[:,i] =  x_scaled[:,i] / xmax[i]
                
        return x_scaled

def preproc_minmax_reverse(x_scaled, xcoeffs):
        x = np.copy(x_scaled)

        (xmin,xmax) = xcoeffs
        for i in range(x.shape[1]):
            if (xmax[i] - xmin[i]) == 0.0:
                x[:,i] = 0.0
            else:
                x[:,i] =  (x[:,i] + xmin[i]) * (xmax[i] - xmin[i] )
        return x

# Preprocess RRTMGP inputs (p,T, gas concs)
def preproc_minmax_inputs_rrtmgp(x, xcoeffs=None): #, datamin, datamax):
        x_scaled = np.copy(x)
        # Log-scale pressure, power-scale H2O and O3
        x_scaled[:,1] = np.log(x_scaled[:,1])
        x_scaled[:,2] = x_scaled[:,2]**(1.0/4) 
        x_scaled[:,3] = x_scaled[:,3]**(1.0/4) 
        # x = minmaxscale(x,data_min_,data_max_)
        if xcoeffs==None:
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler()  
            scaler.fit(x_scaled)
            x_scaled = scaler.transform(x_scaled) 
            return x_scaled, scaler.data_min_, scaler.data_max_

        else:
            (xmin,xmax) = xcoeffs
            for i in range(x.shape[1]):
                x_scaled[:,i] =  (x_scaled[:,i] - xmin[i]) / (xmax[i] - xmin[i] )
            return x_scaled

# A wrapping function to scale inputs and/or outputs   
def scale_gasopt(x_raw, y_raw, col_dry, scale_inputs=False, scale_outputs=False, 
                 nfac=1, y_mean=None, y_sigma=None, xcoeffs=None):

    if scale_inputs:
        if xcoeffs is None:
            x,xmin,xmax = preproc_minmax_inputs_rrtmgp(x_raw)
        else:
            x = preproc_minmax_inputs_rrtmgp(x_raw,xcoeffs )
    else:
        x = x_raw
        
    if scale_outputs:
        # Standardization coefficients loaded from file
        #  y_mean = ymeans_sw_abs; y_sigma = ysigma_sw_abs
        # Set power scaling coefficient (y == y**(1/nfac))
        # nfac = 8 
        if np.any(y_sigma)==None:
            y_sigma = np.repeat(np.float32(1),y_raw.shape[1])

        if np.any(y_mean)==None:
            y_mean = np.repeat(np.float32(0),y_raw.shape[1])
        
        # Scale by layer number of molecules to obtain absorption cross section
        y   = preproc_tau_to_crossection(y_raw, col_dry)
        # Scale using power-scaling followed by standard-scaling
        y   = preproc_pow_standardization(y, nfac, y_mean, y_sigma)
    else:
        y = y_raw
        
    if xcoeffs is None:
        return x, y, xmin, xmax
    else: return x,y
    
def scale_outputs_wrapper(y_raw, col_dry, predictand, ymean=None, ystd=None):
    print("y raw shape", y_raw.shape)
    ny = y_raw.shape[1]
    if (predictand == 'lw_planck_frac'):
        nfac = 2
        y    = scale_outputs(y_raw, None, nfac, None, None)
        
    elif (predictand == 'lw_both'): 
        # I tested just having a unified LW model, didn't seem very promising
        # nfac = 4
        # nyy = int(ny/2)
        # y = y_raw.copy()
        # y[:,0:nyy] = preproc_tau_to_crossection(y[:,0:nyy], col_dry)

        # if np.any(ymean)==None:
        #     ymean = np.zeros(ny); ystd = np.zeros(ny)
        #     for i in range(ny):
        #         ymean[i] = np.mean(y[:,i]**(1/nfac))
        #         ystd[i]  = np.std(y[:,i]**(1/nfac))
        
        # # Scale data
        # y[:,0:nyy]   = scale_outputs(y_raw[:,0:nyy], col_dry, nfac, ymean[0:nyy], ystd[0:nyy])
        # y[:,nyy:]    = scale_outputs(y_raw[:,nyy:], None, nfac, ymean[nyy:], ystd[nyy:])
        
        nfac = 8
        nfac2 = 2
        nyy = int(ny/2)
        y = y_raw.copy()
        y[:,0:nyy] = preproc_tau_to_crossection(y[:,0:nyy], col_dry)

        if np.any(ymean)==None:
            ymean = np.zeros(nyy); ystd = np.zeros(nyy)
            for i in range(nyy):
                ymean[i] = np.mean(y[:,i]**(1/nfac))
                ystd[i]  = np.std(y[:,i]**(1/nfac))
        
        # Scale data
        y[:,0:nyy]   = scale_outputs(y_raw[:,0:nyy], col_dry, nfac, ymean[0:nyy], ystd[0:nyy])
        y[:,nyy:]    = scale_outputs(y_raw[:,nyy:], None, nfac2, None, None)
    else:  # For scaling optical depths
        nfac = 8

        y   = preproc_tau_to_crossection(y_raw, col_dry)
        # if np.any(ymean)==None:
        ymean = np.zeros(ny); ystd = np.zeros(ny)
        for i in range(ny):
            ymean[i] = np.mean(y[:,i]**(1/nfac))
            # ystd[i]  = np.std(y[:,i]**(1/nfac))
        ystd = np.repeat(np.std(y**(1/nfac)),ny)
        # print("ymean", ymean)
        # Scale data
        y    = scale_outputs(y_raw, col_dry, nfac, ymean, ystd)
    return y, ymean, ystd
        
def scale_outputs(y_raw, col_dry=None, nfac=1, 
                 y_mean=None, y_sigma=None):
    # Y_mean and y_sigma are optional outputs: if missing, skip standard-scaling
    if np.any(y_sigma)==None:
        y_sigma = np.repeat(np.float32(1), y_raw.shape[1])

    if np.any(y_mean)==None:
        y_mean = np.repeat(np.float32(0), y_raw.shape[1])
    
    # Scale by layer number of molecules to obtain absorption cross section
    if np.any(col_dry)==None:
        y = np.copy(y_raw)
    else:
        y = preproc_tau_to_crossection(y_raw, col_dry)

    # print("y mean ", y_mean, "sig", y_sigma, "shape y", y.shape, "max", y.max())
    # Scale using power-scaling followed by standard-scaling
    y   = preproc_pow_standardization(y, nfac, y_mean, y_sigma)
    return y