#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytorch implementation of a two-stream SW radiation scheme using a two-stream approximation
and neural network gas optics models, plus helper functions for loading existing gas optics models
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parameter as Parameter
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Final, Optional
import xarray as xr
from torchinfo import summary

rrtmgp_sw_solar_source = np.array([6.12496230e+00, 1.93416359e+00, 1.54202784e+00, 1.27604859e+00,
       1.40585104e+00, 1.16409128e+00, 7.08588401e-01, 2.38161909e-01,
       2.80633452e-02, 1.17647195e-02, 4.77236041e+00, 1.41260578e+00,
       1.27267340e+00, 1.10027782e+00, 9.01671021e-01, 6.75989641e-01,
       5.17114084e-01, 1.23600030e-01, 2.81340293e+00, 3.00515086e+00,
       5.66785860e+00, 2.44188210e+00, 2.09266117e+00, 1.71121207e+00,
       1.28693293e+00, 8.72636422e-01, 2.37112760e-01, 7.54697157e-02,
       1.25168160e-02, 1.40388409e+01, 2.73778552e+00, 2.34644879e+00,
       1.90880448e+00, 1.42339088e+00, 9.49046209e-01, 1.84368958e-01,
       1.53579391e-01, 1.25842097e+01, 2.71192095e+00, 2.37346215e+00,
       1.95650839e+00, 1.47565524e+00, 1.00127937e+00, 2.00756405e-01,
       1.24308531e-01, 4.75310979e-02, 2.67146438e+01, 7.31047641e+00,
       6.03680402e+00, 5.62099956e+00, 4.40638678e+00, 3.24467384e+00,
       2.19979011e+00, 5.95967386e-01, 1.87873000e-01, 3.18016967e-02,
       3.11399060e+01, 1.48087207e+01, 1.36782529e+01, 1.23425661e+01,
       1.07046771e+01, 8.77752262e+00, 6.60841895e+00, 4.48445158e+00,
       1.21470779e+00, 3.84841438e-01, 6.44010175e-02, 2.31922031e+01,
       1.04592715e+00, 3.37244765e-01, 4.99117831e-02, 1.98145677e+02,
       4.06918810e+01, 3.51424680e+01, 2.87645888e+01, 2.16718947e+01,
       1.47965596e+01, 2.95515997e+00, 1.83591468e+00, 7.06192584e-01,
       6.58608873e+01, 1.23497277e+02, 1.37219660e+01, 9.39462675e+00,
       1.87920792e+00, 6.74776664e-01, 4.97130627e-01, 3.21763938e-01,
       1.38897267e-01, 6.36205305e+01, 5.73208949e+01, 5.15562123e+01,
       4.37227132e+01, 6.95342200e+01, 2.47825111e+01, 3.38733953e+01,
       3.04849196e+01, 2.82836381e+01, 2.06528161e+01, 2.52860801e+01,
       9.46283897e+00, 1.57151062e+01, 1.47029312e+01, 1.06884199e+01,
       7.67680730e+00, 5.14239730e+00, 3.29607519e+00, 1.44944362e+00,
       2.80638276e+00, 6.16410042e-01, 6.90493136e-01, 1.48562384e+00], dtype=np.float32)


def load_gas_optics_from_file(existing_gasopt_file_sw_abs, existing_gasopt_file_sw_ray, device):
  print("Loading pre-existing shortwave !ABSORPTION! gas optics model from".format(existing_gasopt_file_sw_abs))
  mlp_gasopt_model_sw_abs = load_gas_optics_model(existing_gasopt_file_sw_abs, device)
  print("Loading pre-existing shortwave !RAYLEIGH! gas optics model from {}".format(existing_gasopt_file_sw_ray))
  mlp_gasopt_model_sw_ray = load_gas_optics_model(existing_gasopt_file_sw_ray, device)
  return mlp_gasoptics_model_sw_abs, mlp_gasopt_model_sw_ray


def load_gas_optics_model(gasopt_file, device, lock_weights=False):
  # Load model from NetCDF file
  ds = xr.open_dataset(gasopt_file)
  input_str = ds.nn_inputs

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

  nn = existing_gasopt_inference_mlp(device, xnorm_lw_min, xnorm_lw_max, 
                      ynorm_lw_mean, ynorm_lw_std,
                      nn_w1, nn_w2, nn_w3,
                      nn_b1, nn_b2, nn_b3, lock_weights=lock_weights)
  infostr = summary(nn)
  return nn 

class existing_gasopt_inference_mlp(nn.Module):
    """
    Gas optics neural networks: differs from GasOpticsMLP in that the post-processing is inlined
    Carries the normalisation coefficients (also for pre-processing inputs) 
    """
    lock_weights: Final[bool]
    def __init__(self, device,
                xmin, xmax, ymean, ystd,
                nn_w1, nn_w2, nn_w3,
                nn_b1, nn_b2, nn_b3, lock_weights = True):
        super(existing_gasopt_inference_mlp, self).__init__()
        self.nx = xmin.shape[0]
        self.ny = ymean.shape[0]
        self.ng = self.ny
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
          self.mlp3.weight.requires_grad = False; self.mlp3.bias.requires_grad = False

        self.to(device)

    def forward(self, x, col_dry):
        x = self.mlp1(x)
        x = self.softsign(x)
        x = self.mlp2(x)
        x = self.softsign(x)
        x = self.mlp3(x)
        tau = x 
        # Postprocessing inlined: reverse standard scaling and square root scaling, multiply with number of dry air molecules
        tau = col_dry * torch.pow(self.ystd*tau + self.ymean,8)
        return tau


class SW_rad_torch(nn.Module):
    use_existing_gas_optics_sw: Final[bool] # Use existing gas optics model (RRTMGP-SW emulator)
    return_band_fluxes: Final[bool]
    
    def __init__(self, 
                device: torch.device,
                gas_optics_model_sw_abs:existing_gasopt_inference_mlp,
                gas_optics_model_sw_ray:existing_gasopt_inference_mlp,
                ng_sw : int=112, # default value of 112 corresponds to RRTMGP-NN,
                # nlev=61, # number of levels - can be determined from inputs
                return_band_fluxes:bool=False, # Later we may want to look at specific fluxes, for now not used
                ):
        super().__init__()
        self.return_band_fluxes = return_band_fluxes
        # self.nlev = nlev 
        if gas_optics_model_sw_abs is not None:
          self.gas_optics_model_sw_abs = gas_optics_model_sw_abs # Absorption cross-section model
          self.gas_optics_model_sw_ray = gas_optics_model_sw_ray # Rayleigh cross-section model
          self.use_existing_gas_optics_sw = True 
          print("Existing shortwave gas optics (absorption) model LOADED! Number of g-points: {}".format(self.gas_optics_model_sw_abs.ng))
          print("Existing shortwave gas optics (Rayleigh) model LOADED! Number of g-points: {}".format(self.gas_optics_model_sw_ray.ng))
          self.ng_sw = self.gas_optics_model_sw_abs.ng 
          rrtmgp_sw_solar_source = rrtmgp_sw_solar_source/np.sum(rrtmgp_sw_solar_source)
          sw_solar_weights = torch.tensor(rrtmgp_sw_solar_source, device=device).unsqueeze(0)
          self.register_buffer('sw_solar_weights', sw_solar_weights)
        else:
          raise NotImplementedError("Training new gas optics schemes on the fly not yet implemented (how do we handle missing normalisation coefficients?)")
          # self.gas_optics_model_sw_abs =  GasOpticsMLP
          # self.gas_optics_model_sw_ray =  GasOpticsMLP
          # self.use_existing_gas_optics_sw = False 
          # self.sw_solar_weights = nn.Parameter(torch.zeros(1, self.ng_sw)) 

    def forward(self, x_gas, # inputs to gas optics NN model, already normalised
                col_dry, mu0, incoming_toa, pres_lev, # unnormalised variables used in radiative transfer computations 
                albedo_surf_dir_sw, # If albedo_surf_diff_sw is None, this is general albedo used for both dir and diff
                albedo_surf_diff_sw=None):

      """
      Two-stream clear-sky shortwave radiative transfer with correlated-k (or other spectral discretization)
      Computations assume dimensions (level, column, spectral) where we can collapse the column (batch) and spectral dims
      For shortwave computations accounting only for gases we need to first compute optical properties from:
        - x_gas: [temperature, perssure, volume mixing ratios of H2O, O3, N2O, CH4]
      Then for flux computations we also need: 
        - solar zenith angle (mu0), incoming flux at top-of-atmosphere, albedo to diffuse and direct radiation
         (usually we only have one albedo value for each column, which we for both diffuse an direct computation)
      Finally to compute heating rate we also need the pressure differences between adjacent levels
      Returns:
        - dT_rad (ncol, nlay) : shortwave heating rate
        - flux_sw_up (ncol, nlay+1)     : upwelling shortwave flux 
        - flux_sw_dn (ncol, nlay+1)     : downwelling total (direct+diffuse) shortwave flux 
        - flux_dn_direct (ncol, nlay+1) : downwelling direct shortwave flux
      """

      batch_size, nlay, nx = x_gas.shape 
      nlev = nlay + 1 
      device = x_gas.device

      # Transpose arrays, because the RTE is faster with levels outermost 
      x_gas = torch.transpose(x_gas,0,1)

      if albedo_surf_diff_sw==None:
         albedo_surf_diff_sw = torch.clone(albedo_surf_diff_sw)

      # -------------------------- SHORTWAVE -----------------------------
      # 

      # GAS OPTICAL PROPERTIES IN EACH LAYER
      # x_gas = torch.cat((temp, pres, vmr_h2o, o3, co2, n2o, ch4), dim=2)
      # x_gas = (x_gas - self.gas_optics_model_sw1.xmin) / self.gas_optics_model_sw1.div
      tau_sw        = self.gas_optics_model_sw1(x_gas, col_dry)
      tau_sw_scat   = self.gas_optics_model_sw2(x_gas, col_dry)

      ssa_sw  = tau_sw_scat / tau_sw
      g_sw    = torch.zeros_like(ssa_sw) # asymmetry factor is zero for gases

      # Here we set the cosine of solar zenith angle to a minimum value, later we set the fluxes in night-time columns to zero
      min_mu = 1e-6 #1e-3
      mu0_comp = torch.clamp(mu0, min=min_mu) 

      # expand mu0 (cosine of solar zenith angle) from (ncol) -> (nlay,ncol,ng_sw)
      mu0_rep = mu0_comp.reshape((1,-1,1)).expand(nlay, -1, self.ng_sw)

      # SW REFLECTANCE-TRANSMITTANCE COMPUTATIONS FOR EACH LAYER
      ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir = calc_ref_trans_sw(mu0_rep, tau_sw, ssa_sw, g_sw)

      ref_diff            = ref_diff.view(nlay, -1)
      trans_diff          = trans_diff.view(nlay, -1)
      ref_dir             = ref_dir.view(nlay, -1)
      trans_dir_diff      = trans_dir_diff.view(nlay, -1)
      trans_dir_dir       = trans_dir_dir.view(nlay, -1)
      del tau_sw, ssa_sw, g_sw, mu0_rep

      if (self.use_existing_gas_optics_sw):
        toa_spectral = self.sw_solar_weights 
      else:
        # Here we apply softmax to ensure the solar weights sum to 1 (and are positive)
        toa_spectral = self.softmax_dim1(self.sw_solar_weights)

      incoming_toa = incoming_toa*toa_spectral
      incoming_toa = incoming_toa.view(-1)

      albedo_surf_dir_sw    = albedo_surf_dir_sw.view(-1)
      albedo_surf_diff_sw   = albedo_surf_diff_sw.view(-1)

      # --------- SW RADIATIVE TRANSFER USING ADDING METHOD -----------
    
      flux_sw_up, flux_sw_dn_diffuse, flux_sw_dn_direct = adding_ica_sw(
                  incoming_toa, albedo_surf_diff_sw, albedo_surf_dir_sw, 
                  ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir)
          
      del ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir

      flux_sw_up = torch.reshape(flux_sw_up, (nlev, batch_size, self.ng_sw))
      flux_sw_dn_diffuse = torch.reshape(flux_sw_dn_diffuse, (nlev, batch_size, self.ng_sw))
      flux_sw_dn_direct = torch.reshape(flux_sw_dn_direct, (nlev, batch_size, self.ng_sw))

      if self.return_band_fluxes: # To be properly implemented later
        # RRTMGP bands and g-points:
        # bnd_limits_gpt
        # 1,10     | 11,18    | 19,29    | 30,37   | 38,46     | 47,56     | 57,67     | 68,71      | 72,80      |  81,89    | 90, 96    | 97, 102   | 103, 109 | 110, 112 
        # bnd_limits_wavenumber
        # 820,2680 | 2680,3250 | 3250,4k | 4k,4650 | 4650,5150 | 5150,6150 | 6150,7700 | 7700,8050  | 8050,12850 | 12850,16k | 16k,22650 | 22650,29k | 29k,38k  | 38k,50k 
        # in micrometers 
        # 12.2,3.73| 3.73,3.08 | 3.08,2.5| 2.5,2.15| 2.15,1.94 | 1.94,1.63 | 1.63,1.3  | 1.3,1.24   | 1.24,0.78  | 0.78,0.62 | 0.62,0.44 | 0.44,0.34 | 0.34,0.26| 0.26  0.2 ]
        # band 1        2         3           4           5         6             7         8              9           10         11          12         13         14
        # E3SM wants:
        # ! sols(pcols)      Direct solar rad on surface (< 0.7 micrometer)
        # ! soll(pcols)      Direct solar rad on surface (>= 0.7 micrometer)
        iend_ir = int(round((80/112)*self.ng_sw)) # RRTMGP bands 1-9 (g-points 1-80) encompass 820-12850 cm-1 (near-ir), see data/rrtmgp-data-sw-g112-210809.nc
        iend_mix= int(round((89/112)*self.ng_sw)) # RRTMGP band 10 is in between UV/visible and near-IR, and bands 11-14 (89-112) are fully in visible range (> 14286 ! cm^-1)

        # Sum over whole / parts of spectral dimension to get broadband / band-wise fluxes
        sw_dir_dn_mixband = torch.sum(flux_sw_dn_direct[-1,:,iend_ir:iend_mix],dim=1,keepdim=True)
        SOLL = torch.sum(flux_sw_dn_direct[-1,:,0:iend_ir],dim=1,keepdim=True) + 0.5*sw_dir_dn_mixband
        SOLS = torch.sum(flux_sw_dn_direct[-1,:,iend_mix:],dim=1,keepdim=True) + 0.5*sw_dir_dn_mixband

        sw_diff_dn_mixband = torch.sum(flux_sw_dn_diffuse[-1,:,iend_ir:iend_mix],dim=1,keepdim=True)
        SOLLD = torch.sum(flux_sw_dn_diffuse[-1,:,0:iend_ir],dim=1,keepdim=True) + 0.5*sw_diff_dn_mixband
        SOLSD = torch.sum(flux_sw_dn_diffuse[-1,:,iend_mix:],dim=1,keepdim=True) + 0.5*sw_diff_dn_mixband

      flux_sw_up          = torch.sum(flux_sw_up,dim=2)
      flux_sw_dn_diffuse  = torch.sum(flux_sw_dn_diffuse,dim=2)
      flux_sw_dn_direct   = torch.sum(flux_sw_dn_direct,dim=2)
      
      flux_sw_dn          = flux_sw_dn_diffuse + flux_sw_dn_direct
      flux_sw_dn_sfc      = flux_sw_dn[-1,:].unsqueeze(1)             # NETSW  

      flux_sw_net = flux_sw_dn - flux_sw_up
      # We did SW computations in all columns, so now we need to zero out nighttime columns
      inds_zero = mu0 < min_mu
      flux_sw_net[:,inds_zero] = 0.0
      flux_sw_dn_sfc[inds_zero] = 0.0

      # COMPUTE HEATING RATES
      flux_diff       = flux_sw_net[1:] - flux_sw_net[0:-1]
      pres_diff       = pres_lev[1:] - pres_lev[0:-1]
      dT_rad          = -(flux_diff / pres_diff.squeeze()) * 0.009761357302 # * g/cp = 9.80665 / 1004.64
      # dT_rad          = dT_rad * self.yscale_lev[:,0:1] # scaling (physical computations gave us unscaled tendency, but target outputs are scaled)      
      # out_sfc_rad = out_sfc_rad * self.yscale_sca_rad

      # Transpose back to (batch, lev)
      dT_rad      = torch.transpose(dT_rad,0,1)
      flux_sw_up  = torch.transpose(flux_sw_up,0,1)
      flux_sw_dn_direct  = torch.transpose(flux_sw_dn_direct,0,1)

      if self.return_band_fluxes:
        SOLL[inds_zero] = 0.0
        SOLS[inds_zero] = 0.0
        SOLLD[inds_zero] = 0.0
        SOLSD[inds_zero] = 0.0
        return dT_rad, flux_sw_up, flux_sw_dn, flux_sw_dn_direct, SOLS, SOLL, SOLSD, SOLLD
      else:
        return dT_rad, flux_sw_up, flux_sw_dn, flux_sw_dn_direct


# -------------------------------------------- SHORTWAVE FUNCTIONS --------------------------------------------

@torch.compile(dynamic=False)
def calc_ref_trans_sw(mu0, od, ssa, asymmetry):
    """
    Two-stream shortwave reflectance and transmittance calculation.
    Implements Meador & Weaver (1980) equations.

    Args:   (variable dimensions don't matter because all operations are element wise)
        mu0:       Cosine of solar zenith angle (nlev,ncol,ng) (expanded view)
        od:        Optical depth, shape (nlev,ncol,ng) 
        ssa:       Single scattering albedo, shape (nlev,ncol,ng) 
        asymmetry: Asymmetry factor, shape (nlev,ncol,ng) 

    Returns:
        ref_diff:       Diffuse reflectance.
        trans_diff:     Diffuse transmittance.
        ref_dir:        Direct reflectance.
        trans_dir_diff: Direct-to-diffuse transmittance.
        trans_dir_dir:  Direct unscattered transmittance.
    """
    # eps = torch.finfo(od.dtype).eps
    eps = 1.0e-7

    # ------------------------------------------------------------------ #
    # Unscattered direct transmittance
    # ------------------------------------------------------------------ #
    trans_dir_dir = torch.exp(-od / mu0)

    # ------------------------------------------------------------------ #
    # Two-stream gamma coefficients
    # ------------------------------------------------------------------ #
    gamma1 = (8 - ssa*(5 + 3*asymmetry)) * 0.25
    gamma2 = 3*(ssa*(1 - asymmetry)) * 0.25
    gamma3 = (2 - 3*mu0*asymmetry) * 0.25
    gamma4  = 1.0  - gamma3

    # alpha1 / alpha2  (Eqs. 16-17)
    alpha1 = gamma1 * gamma4 + gamma2 * gamma3
    alpha2 = gamma1 * gamma3 + gamma2 * gamma4

    # ------------------------------------------------------------------ #
    # Diffuse reflectance / transmittance  (Eqs. 25-26)
    # ------------------------------------------------------------------ #
    # k_exponent  (Eq. 18) — clamped for numerical safety
    k = torch.sqrt(torch.clamp((gamma1 - gamma2) * (gamma1 + gamma2), min=1.0e-4)) # 1e-4 TUNED FOR SINGLE PRECISION!

    exponential   = torch.exp(-k * od)
    exponential2  = exponential ** 2
    k_2_exp       = 2.0 * k * exponential

    reftrans_factor = 1.0 / (k + gamma1 + (k - gamma1) * exponential2)

    ref_diff   = gamma2 * (1.0 - exponential2) * reftrans_factor

    zeros=torch.zeros_like(ref_diff)
    trans_diff = torch.clamp(
        k_2_exp * reftrans_factor,
        min=zeros,
        max=1.0 - ref_diff,          # never exceeds 1 − ref_diff
    )
    trans_diff = torch.clamp(trans_diff, min=0.0)

    # ------------------------------------------------------------------ #
    # Direct reflectance / transmittance  (Eqs. 14-15)
    # ------------------------------------------------------------------ #
    k_mu0              = k * mu0
    one_minus_kmu0_sqr = 1.0 - k_mu0 ** 2
    k_gamma3           = k * gamma3
    k_gamma4           = k * gamma4

    # Guard against one_minus_kmu0_sqr ≈ 0 (mirrors Fortran's merge/epsilon)
    safe_denom = torch.where(
        one_minus_kmu0_sqr.abs() > eps,
        one_minus_kmu0_sqr,
        torch.full_like(one_minus_kmu0_sqr, eps),
    )
    # safe_denom = one_minus_kmu0_sqr.abs().clamp(min=eps) * one_minus_kmu0_sqr.sign()

    # reftrans_factor = mu0 * ssa * reftrans_factor / safe_denom
    reftrans_factor = ssa * reftrans_factor / safe_denom

    # Eq. 14
    ref_dir = reftrans_factor * (
            (1.0 - k_mu0) * (alpha2 + k_gamma3)
        - (1.0 + k_mu0) * (alpha2 - k_gamma3) * exponential2
        - k_2_exp * (gamma3 - alpha2 * mu0) * trans_dir_dir
    )

    # Eq. 15 (minus the direct unscattered term)
    trans_dir_diff = reftrans_factor * (
            k_2_exp * (gamma4 + alpha1 * mu0)
        - trans_dir_dir * (
                (1.0 + k_mu0) * (alpha1 + k_gamma4)
            - (1.0 - k_mu0) * (alpha1 - k_gamma4) * exponential2
            )
    )

    # ------------------------------------------------------------------ #
    # Final clipping so that ref_dir + trans_dir_diff ≤ mu0*(1−T_dir_dir)
    # ------------------------------------------------------------------ #
    # max_direct = mu0 * (1.0 - trans_dir_dir)
    max_direct = (1.0 - trans_dir_dir)

    ref_dir        = torch.clamp(ref_dir,        min=zeros, max=max_direct)
    trans_dir_diff = torch.clamp(trans_dir_diff, min=zeros, max=max_direct - ref_dir)

    return ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir


# @torch.jit.script
@torch.compile(dynamic=False)
def adding_ica_sw_batchlast_opt(incoming_toa, albedo_surf_diffuse, albedo_surf_direct,
                R, T, ref_dir, T_dir_diff, T_dir_dir):
        """
        Adding method for shortwave radiation. 
        Adapted from ecRad-TripleClouds to use just two vertical loops instead of three as in RTE. 
        Args are torch.Tensors:
          incoming_toa[nbatch], albedo_surf_diffuse[nbatch], albedo_surf_direct[nbatch],
          reflectance[nlev,nbatch], transmittance[nlev,nbatch], ref_dir[nlev,nbatch], trans_dir_diff[nlev,nbatch],
          trans_dir_dir[nlev,nbatch]
        Here nlev is the contiguous dimension in memory and nbatch (=ng*ncol) is a batch dimension without loop dependencies, 
            where spectral (ng) and column (ncol) dimensions should have been collapsed before calling this function. 
        This should be optimal for GPU; for CPU it *may* be better to have (ng,nlev,ncol) (in Python row major notation)
            where SIMD vectorization is used for the innermost ng and multithreading is used for outermost ncol. 
        Returns:
            tuple: (flux_up, flux_dn_diffuse, flux_dn_direct) each of shape [nbatch, nlev+1]
        """
        
        nlev, nbatch = R.shape
        device = R.device
        
        # Set surface albedo
        albedo = torch.jit.annotate(List[Tensor], [])
        albedo0 = albedo_surf_diffuse
        albedo += [albedo0]

        albedodir = torch.jit.annotate(List[Tensor], [])
        albedodir0 = albedo_surf_direct
        albedodir += [albedodir0]

        # Work up through the atmosphere and compute the albedo of the entire earth/atmosphere system below that half-level
        for jlev in range(nlev-1, -1, -1):  # nlev down to 1 in Fortran indexing

            # comparing ecRad Tripleclouds code to the McICA code, "source" variable  is like fluxdndir*albedodir
            # If we use albedodir instead (like in TripleClouds), we dont need to precompute fluxdndir in a separate loop, so just two vertical loops
            # Adapted from https://github.com/ecmwf-ifs/ecrad/blob/master/radiation/radiation_tripleclouds_sw.F90
            inv_denom = 1.0/(1.0 - albedo0 * R[jlev])
            albedodir0 = ref_dir[jlev] + (T_dir_dir[jlev]*albedodir0 + T_dir_diff[jlev]*albedo0) *T[jlev]* inv_denom
            albedodir  += [albedodir0]  
            
            albedo0 = R[jlev] + torch.square(T[jlev]) * albedo0  * inv_denom #/ (1.0 - albedo0 * R[jlev])
            albedo  += [albedo0]

        # Reverse arrays because next loop will go from top-of-atmosphere to surface
        albedo.reverse(); albedodir.reverse()
        
        # At top-of-atmosphere, all upwelling radiation is due to scattering by the direct beam below that level
        fluxup = incoming_toa*albedodir[0]
        flux_up = torch.jit.annotate(List[Tensor], [])
        flux_up += [fluxup]

        fluxdndir = incoming_toa
        flux_dn_direct = torch.jit.annotate(List[Tensor], [])
        flux_dn_direct += [fluxdndir]

        # At top-of-atmosphere there is no diffuse downwelling radiation
        fluxdndiff = torch.zeros_like(incoming_toa)
        flux_dn_diffuse = torch.jit.annotate(List[Tensor], [])
        flux_dn_diffuse += [fluxdndiff]

        # Work back down through the atmosphere computing the fluxes at each half-level
        for jlev in range(nlev):  # 1 to nlev in Fortran indexing

            fluxdndiff = (T[jlev]*fluxdndiff 
                + fluxdndir * (T[jlev]*albedodir[jlev+1]*R[jlev] + T_dir_diff[jlev])) / (1.0 - R[jlev]*albedo[jlev+1])

            fluxdndir =  fluxdndir * T_dir_dir[jlev,:]
            # Apply cosine correction to direct flux..NOT HERE, already done to TOA incoming flux
            # flux_dn_direct = fluxdndir * cos_sza
            flux_dn_direct  += [fluxdndir]
            flux_dn_diffuse += [fluxdndiff]

            fluxup = fluxdndir*albedodir[jlev+1] + fluxdndiff* albedo[jlev + 1]
            flux_up += [fluxup]       
        
        flux_dn_direct  = torch.stack(flux_dn_direct)
        flux_dn_diffuse = torch.stack(flux_dn_diffuse)
        flux_up = torch.stack(flux_up)

        return flux_up, flux_dn_diffuse, flux_dn_direct

# @torch.jit.script
@torch.compile(dynamic=False)
def adding_ica_sw_inference(
    incoming_toa: Tensor,
    albedo_surf_diffuse: Tensor,
    albedo_surf_direct: Tensor,
    reflectance: Tensor,
    transmittance: Tensor,
    ref_dir: Tensor,
    trans_dir_diff: Tensor,
    trans_dir_dir: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:

    nlev, nbatch = reflectance.shape

    # --- Upward sweep ---
    # Only keep current-level scalars; no list accumulation needed.
    # BUT: the downward sweep still needs albedo[jlev+1] at every level,
    # so we cannot avoid storing the full arrays.
    # We CAN avoid the Python list + reverse + stack overhead by writing
    # directly into pre-allocated tensors — safe here because autograd
    # is not running, so no version counter issues.
    albedo    = torch.empty(nlev + 1, nbatch, dtype=reflectance.dtype,
                             device=reflectance.device)
    albedodir = torch.empty(nlev + 1, nbatch, dtype=reflectance.dtype,
                             device=reflectance.device)

    albedo[0]    = albedo_surf_diffuse
    albedodir[0] = albedo_surf_direct

    alb0  = albedo_surf_diffuse
    adir0 = albedo_surf_direct

    for k in range(nlev):
        jlev = nlev - 1 - k
        R  = reflectance[jlev]
        T  = transmittance[jlev]
        inv_denom = 1.0 / (1.0 - alb0 * R)
        adir0 = ref_dir[jlev] + (trans_dir_dir[jlev] * adir0 + trans_dir_diff[jlev] * alb0) * T * inv_denom
        alb0  = R + T * T * alb0 * inv_denom
        albedo   [k + 1] = alb0
        albedodir[k + 1] = adir0

    # --- Downward sweep ---
    flux_up         = torch.empty(nlev + 1, nbatch, dtype=reflectance.dtype,
                                   device=reflectance.device)
    flux_dn_diffuse = torch.empty(nlev + 1, nbatch, dtype=reflectance.dtype,
                                   device=reflectance.device)
    flux_dn_direct  = torch.empty(nlev + 1, nbatch, dtype=reflectance.dtype,
                                   device=reflectance.device)

    fluxdndir  = incoming_toa
    fluxdndiff = torch.zeros(nbatch, dtype=reflectance.dtype,
                              device=reflectance.device)

    flux_up        [0] = incoming_toa * albedodir[nlev]   # TOA = slot nlev
    flux_dn_direct [0] = fluxdndir
    flux_dn_diffuse[0] = fluxdndiff

    for jlev in range(nlev):
        R     = reflectance[jlev]
        T     = transmittance[jlev]
        below = nlev - (jlev + 1)
        alb1  = albedo   [below]
        adir1 = albedodir[below]
        fluxdndiff = ((T * fluxdndiff  + fluxdndir * (T * adir1 * R + trans_dir_diff[jlev])) / (1.0 - R * alb1))
        fluxdndir  = fluxdndir * trans_dir_dir[jlev]
        flux_dn_direct [jlev + 1] = fluxdndir
        flux_dn_diffuse[jlev + 1] = fluxdndiff
        flux_up        [jlev + 1] = fluxdndir * adir1 + fluxdndiff * alb1

    return flux_up, flux_dn_diffuse, flux_dn_direct

def adding_ica_sw(
    incoming_toa, albedo_surf_diffuse, albedo_surf_direct,
    reflectance, transmittance, ref_dir, trans_dir_diff, trans_dir_dir
):
    if torch.is_grad_enabled():
        # print("calling normal adding!")
        return adding_ica_sw_batchlast_opt(
            incoming_toa, albedo_surf_diffuse, albedo_surf_direct,
            reflectance, transmittance, ref_dir, trans_dir_diff, trans_dir_dir
        )
    else:
        # print("calling inference adding!")
        return adding_ica_sw_inference(
            incoming_toa, albedo_surf_diffuse, albedo_surf_direct,
            reflectance, transmittance, ref_dir, trans_dir_diff, trans_dir_dir
        )
