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
from coefficients import rrtmgp_sw_solar_source
from coefficients import RRTMGP_SPLITS, WAVENUM_SPLITS  # band limits and corresponding wavenumbers
# Build slice boundaries: [0] + splits + [ng]
RRTMGP_BOUNDS = [0] + RRTMGP_SPLITS + [112]

def load_gas_optics_from_file(device, existing_gasopt_file_sw_abs, existing_gasopt_file_sw_ray=None):
  print(f"Loading pre-existing shortwave !ABSORPTION! gas optics model from {existing_gasopt_file_sw_abs}")
  mlp_gasopt_model_sw_abs = load_gas_optics_model(existing_gasopt_file_sw_abs, device)
  if existing_gasopt_file_sw_ray is not None:
    print("Loading pre-existing shortwave !RAYLEIGH! gas optics model from {}".format(existing_gasopt_file_sw_ray))
    mlp_gasopt_model_sw_ray = load_gas_optics_model(existing_gasopt_file_sw_ray, device)
    return mlp_gasopt_model_sw_abs, mlp_gasopt_model_sw_ray
  else:
    return mlp_gasopt_model_sw_abs 

def load_gas_optics_model(gasopt_file, device, lock_weights=False, is_shortwave=True):
  # Load model from NetCDF file
  ds = xr.open_dataset(gasopt_file)
  input_str = ds.nn_inputs

  nn_w1 = ds['nn_weights_1'][:].values
  nn_w2 = ds['nn_weights_2'][:].values
  nn_w3 = ds['nn_weights_3'][:].values

  nn_b1 = ds['nn_bias_1'][:].values
  nn_b2 = ds['nn_bias_2'][:].values
  nn_b3 = ds['nn_bias_3'][:].values

  ynorm_mean = ds['nn_output_coeffs_mean'][:].values 
  ynorm_std =  ds['nn_output_coeffs_std'][:].values 

  xnorm_max = ds['nn_input_coeffs_max'][:].values 
  xnorm_min =  ds['nn_input_coeffs_min'][:].values 

  if is_shortwave:
    from coefficients import rrtmgp_sw_solar_source
    rrtmgp_sw_solar_source = rrtmgp_sw_solar_source/np.sum(rrtmgp_sw_solar_source)

  nn = mlp_gasopt_inlined_processing(device=device, 
                      xmin=xnorm_min, xmax=xnorm_max, 
                      ymean=ynorm_mean, ystd=ynorm_std,
                      nn_w1=nn_w1, nn_w2=nn_w2, nn_w3=nn_w3,
                      nn_b1=nn_b1, nn_b2=nn_b2, nn_b3=nn_b3, 
                      solar_source=rrtmgp_sw_solar_source,
                      lock_weights=lock_weights)
  nn.eval()
  infostr = summary(nn)
  return nn 

class mlp_gasopt_inlined_processing(nn.Module):
    """
    Gas optics neural networks: differs from GasOpticsMLP in that the post-processing is inlined.
    Can be used for a pre-trained gas optics model, in which case weights (nn_w1,..) and 
    solar_source must be provided.
    Contains the normalisation coefficients for pre-processing the inputs (xmin, xmax), 
    and if using pre-trained models, also output scaling coefficients ymean, ystd. 
    If we are training a new model from scratch, ny (number of g-points) and nh (hidden neurons) are hyperparameters.
    """
    lock_weights: Final[bool]
    do_norm: Final[bool]
    is_rrtmgp: Final[bool]
    def __init__(self, device, 
                xmin, xmax, ymean=None, ystd=None,
                nn_w1=None, nn_w2=None, nn_w3=None,
                nn_b1=None, nn_b2=None, nn_b3=None, 
                solar_source=None,
                lock_weights = True,
                do_norm=False,
                ny=16, nh=32,
                band_bounds=None,
                rrtmgp_bounds_in=None, 
                wavenum_splits_in=None):
        super(mlp_gasopt_inlined_processing, self).__init__()
        self.nx = xmin.shape[0]
        self.do_norm = False
        if ymean is not None:
          self.ny = ymean.shape[0]
          self.ng = self.ny
          ymean = torch.from_numpy(ymean[0:self.ny])
          ystd  = torch.from_numpy(ystd[0:self.ny])
          self.register_buffer('ymean', ymean)
          self.register_buffer('ystd',  ystd)
          print("Loaded existing y normalisation coefficients")
          self.do_norm = True
        else:
          self.ny = ny 
          self.ng = self.ny  
          self.do_norm = do_norm
          if self.do_norm:
            self.ymean = 0 # # nn.Parameter(torch.zeros(self.ng)) 
            self.ystd = 1# 0.00060 # nn.Parameter(torch.zeros(1)) 
            print("Using learnable y normalisation coefficients (may not work)")
        if self.ng==112:
          self.is_rrtmgp=True #
        else:
          self.is_rrtmgp=False
        if band_bounds is not None:
          print("mlp_gasopt_inlined_processing band bounds:", band_bounds)
          print("mlp_gasopt_inlined_processing RRTMGP bounds:",rrtmgp_bounds_in )
          self.band_bounds = band_bounds
          if rrtmgp_bounds_in is not None:
            self.rrtmgp_bounds = rrtmgp_bounds_in
            self.wavenum_splits = wavenum_splits_in  
          else:
            self.rrtmgp_bounds = RRTMGP_BOUNDS
            self.wavenum_splits = WAVENUM_SPLITS

          self.num_bands = len(band_bounds) - 1 
          print("Number of bands: {}".format(self.num_bands))
          # self.register_buffer("band_bounds", band_bounds)
        else:
          self.num_bands = 1

        if band_bounds is not None:
            # RRTMGP band wavenumber limits, in the same g-point order as RRTMGP_BOUNDS
            # Band 1: 820-2680, Band 2: 2680-3250, ..., Band 14: 38000-50000
            # Plus a leading entry for the start of band 1 (820 cm-1) and 
            # the SW lower limit (250 cm-1 is used conventionally but RRTMGP starts at 820)
            RRTMGP_WAVENUM_LOW  = [820, 2680, 3250, 4000, 4650, 5150, 6150, 7700, 8050, 12850, 16000, 22650, 29000, 38000]
            RRTMGP_WAVENUM_HIGH = [2680, 3250, 4000, 4650, 5150, 6150, 7700, 8050, 12850, 16000, 22650, 29000, 38000, 50000]
            # Map from g-point index to RRTMGP band index (0-based)
            # rrtmgp_bounds = [0, 29, 80, 89, 102, 112] means:
            #   g-pts 0:29   → RRTMGP bands 0-2  (820–4000   cm-1)
            #   g-pts 29:80  → RRTMGP bands 3-8  (4000–12850 cm-1)
            #   g-pts 80:89  → RRTMGP band  9    (12850–16000 cm-1)
            #   g-pts 89:102 → RRTMGP bands 10-11 (16000–29000 cm-1)
            #   g-pts 102:112→ RRTMGP bands 12-13 (29000–50000 cm-1)
            # Build a lookup: for each RRTMGP g-point boundary, what is the wavenumber?
            # RRTMGP g-point band assignments (cumulative, 0-based band index):
            RRTMGP_GPT_BOUNDS = [0, 10, 18, 29, 37, 46, 56, 67, 71, 80, 89, 96, 102, 109, 112]
            # RRTMGP_GPT_BOUNDS[b]:RRTMGP_GPT_BOUNDS[b+1] are g-points of RRTMGP band b

            def gpt_to_wavenum_low(gpt: int) -> int:
                """Return the lower wavenumber of the RRTMGP band containing g-point gpt."""
                for b in range(14):
                    if gpt <= RRTMGP_GPT_BOUNDS[b + 1]:
                        return RRTMGP_WAVENUM_LOW[b]
                return RRTMGP_WAVENUM_LOW[13]

            def gpt_to_wavenum_high(gpt: int) -> int:
                """Return the upper wavenumber of the RRTMGP band containing g-point gpt."""
                for b in range(14):
                    if gpt <= RRTMGP_GPT_BOUNDS[b + 1]:
                        return RRTMGP_WAVENUM_HIGH[b]
                return RRTMGP_WAVENUM_HIGH[13]

            # Derive wavenumber bounds for each custom band from rrtmgp_bounds
            # rrtmgp_bounds[0]=0 → lower wavenumber of the band containing g-pt 0 = 820
            # rrtmgp_bounds[1]=29 → g-pt 29 is the first g-pt of the next band, so upper wavenum
            #                        of custom band 1 = lower wavenum of the band containing g-pt 29
            # rrtmgp_bounds[-1]=112 → upper wavenumber = 50000
            rb = self.rrtmgp_bounds  # e.g. [0, 29, 80, 89, 102, 112]
            wavenum_bounds: List[int] = []
            wavenum_bounds.append(RRTMGP_WAVENUM_LOW[0])   # start of first band = 820 cm-1
            for i in range(1, len(rb) - 1):
                # rb[i] is the first g-point of the next custom band
                # so the boundary wavenumber is the lower limit of the RRTMGP band it starts in
                wavenum_bounds.append(gpt_to_wavenum_low(rb[i]))
            wavenum_bounds.append(RRTMGP_WAVENUM_HIGH[13])  # end of last band = 50000 cm-1

            self.wavenum_bounds: List[int] = wavenum_bounds
            print(f"Derived wavenumber bounds: {self.wavenum_bounds}")

            # Precompute NIR/visible split for surface flux computation
            # E3SM NIR/visible boundary: 0.7 µm = 14286 cm-1
            NIR_VIS_BOUNDARY: int = 14286
            self.wavenum_nir_vis_boundary: int = NIR_VIS_BOUNDARY
            self.i_gpt_nir_end: int = 0
            self.i_gpt_vis_start: int = self.ng
            self.vis_transition_fraction: float = 0.0

            for b in range(self.num_bands):
                wlo = self.wavenum_bounds[b]
                whi = self.wavenum_bounds[b + 1]
                if whi <= NIR_VIS_BOUNDARY:
                    self.i_gpt_nir_end = band_bounds[b + 1]
                elif wlo >= NIR_VIS_BOUNDARY:
                    if self.i_gpt_vis_start == self.ng:  # first fully visible band
                        self.i_gpt_vis_start = band_bounds[b]
                else:
                    # Transition band straddles the boundary
                    self.i_gpt_nir_end = band_bounds[b]
                    self.i_gpt_vis_start = band_bounds[b + 1]
                    self.vis_transition_fraction = float(whi - NIR_VIS_BOUNDARY) / float(whi - wlo)

            print(f"NIR g-points: 0:{self.i_gpt_nir_end}, "
                  f"transition: {self.i_gpt_nir_end}:{self.i_gpt_vis_start} "
                  f"(vis fraction={self.vis_transition_fraction:.3f}), "
                  f"visible: {self.i_gpt_vis_start}:{self.ng}")
    
        print("do norm", do_norm)
        if nn_w1 is not None:
          self.nh = nn_w1.shape[1]
        else:
          self.nh = nh
        xmin  = torch.from_numpy(xmin)
        xmax  = torch.from_numpy(xmax)
        xdiv = xmax - xmin
        self.register_buffer('xmin', xmin)
        self.register_buffer('xmax', xmax)
        self.register_buffer('xdiv', xdiv)
        self.softsign =  nn.Softsign()
        self.softmax = nn.Softmax(dim=-1)
        self.mlp1 = nn.Linear(self.nx, self.nh)
        self.mlp2 = nn.Linear(self.nh, self.nh)
        self.mlp3 = nn.Linear(self.nh, self.ny)
        self.lock_weights=lock_weights
        print("gasopt_mlp number of g-points: {}, hidden neurons: {}, inputs: {}".format(self.ng, self.nh, self.nx)) 
        if solar_source is not None:
          sw_solar_weights = torch.tensor(solar_source, device=device).unsqueeze(0)
          self.register_buffer('sw_solar_weights', sw_solar_weights)
        else:
          self.sw_solar_weights = nn.Parameter(torch.zeros(1, self.ng)) 
          self.softmax_dim1 = nn.Softmax(dim=1)
          rrtmgp_sw_solar_weights = torch.tensor(rrtmgp_sw_solar_source, device=device).unsqueeze(0)
          self.register_buffer('rrtmgp_sw_solar_weights', rrtmgp_sw_solar_weights)

        if nn_w1 is not None:
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
        # print("x shape", x.shape, "coldry", col_dry.shape, "ystd",self.ystd.shape)
        # Postprocessing inlined: reverse standard scaling and square root scaling, multiply with number of dry air molecules
        if self.do_norm:
          tau = col_dry * torch.pow(self.ystd*tau + self.ymean,8)
        else:
          tau = col_dry * torch.pow(tau,8)
          # print("mean tau after coldry, pow8", tau.mean().item())

        coeff=1e-17
        return tau*coeff

    def get_solar_weights(self):
        if self.is_rrtmgp:
          return self.sw_solar_weights
        else:
          if self.num_bands==1:
            solar_weights = torch.softmax(self.sw_solar_weights,dim=-1)
          else:
            # Compute target band fractions from RRTMGP solar source
            # self.rrtmgp_sw_solar_source: (1, 112) or (112,)
            rrtmgp_src = self.rrtmgp_sw_solar_weights.reshape(-1)  # (112,)
            total = rrtmgp_src.sum()
            p_b = torch.stack([
                rrtmgp_src[self.rrtmgp_bounds[b]:self.rrtmgp_bounds[b+1]].sum() / total
                for b in range(self.num_bands)
            ])  # (nband,) — target fraction of total flux for each band

            # For each band: softmax over the raw learned weights within that band,
            # then scale so the band sums to its RRTMGP target fraction p_b
            raw = self.sw_solar_weights.reshape(-1)  # (ng,)
            band_weights = torch.cat([
                torch.softmax(raw[self.band_bounds[b]:self.band_bounds[b+1]], dim=0) * p_b[b]
                for b in range(self.num_bands)
            ], dim=0)  # (ng,) — sums to 1.0 overall
            # print("shape band weights", band_weights.shape)
            return band_weights.unsqueeze(0)  # (1, ng) matching RRTMGP format
        return solar_weights



class SW_rad_torch(nn.Module):
    """
    Two-stream clear-sky shortwave radiative transfer with correlated-k (or other spectral discretization)
    Computations assume dimensions (level, column, spectral) where we can collapse the column (batch) and spectral dims

    For shortwave computations accounting only for gases we need to first compute optical properties from:
        - x_gas: [temperature, perssure, volume mixing ratios of H2O, O3, N2O, CH4]
    For the gas optics computations, this radiation module uses the custom class mlp_gasopt_inlined_processing,
    where two shortwave gas optics models are provided as arguments. 
    Note these can be pre-trained gas optics models, or new models that we train them from scratch.
    To train new gas optics models, you probably want spectral fluxes as output: set return_gpt_fluxes to True. 
    These spectral fluxes can then be averaged over specific indices to get fluxes in specific bands, 
    it best to do this user-configured reduction inside a function custom_reduction() 
    
    Then for flux computations we also need: 
        - solar zenith angle (mu0), incoming flux at top-of-atmosphere, albedo to diffuse and direct radiation
        (usually we only have one albedo value for each column, which we for both diffuse an direct computation)
    To compute heating rates we also need the pressure at level interfaces. 
    If return_gpt_fluxes=False, returns:
        - dT_rad (ncol, nlay) : shortwave heating rate
        - flux_sw_up (ncol, nlay+1)     : upwelling shortwave flux 
        - flux_sw_dn (ncol, nlay+1)     : downwelling total (direct+diffuse) shortwave flux 
        - flux_dn_direct (ncol, nlay+1) : downwelling direct shortwave flux
    If return_gpt_fluxes=True, returns the same but the flux variables have dimensions (ncol, nlay+1, ng=112)
    """
    use_existing_gas_optics_sw: Final[bool] # Use existing gas optics model (RRTMGP-SW emulator)
    return_gpt_fluxes: Final[bool]
    is_rrtmgp: Final[bool]
    def __init__(self, 
                device: torch.device,
                gas_optics_model_sw_abs:mlp_gasopt_inlined_processing,
                gas_optics_model_sw_ray:mlp_gasopt_inlined_processing,
                ng_sw : int=112, # default value of 112 corresponds to RRTMGP-NN, for training new gas optics set to e.g. 16
                # nlev=61, # number of levels - can be determined from inputs
                return_gpt_fluxes:bool=False, # Set to true to return spectral fluxes (useful for training)
                ):
        super().__init__()
        self.return_gpt_fluxes = return_gpt_fluxes
        # self.nlev = nlev 
        self.gas_optics_model_sw_abs = gas_optics_model_sw_abs # Absorption cross-section model
        self.gas_optics_model_sw_ray = gas_optics_model_sw_ray # Rayleigh cross-section model
        # self.use_existing_gas_optics_sw = True 
        self.ng_sw = self.gas_optics_model_sw_abs.ng 
          # self.gas_optics_model_sw_abs =  GasOpticsMLP
          # self.gas_optics_model_sw_ray =  GasOpticsMLP
          # self.use_existing_gas_optics_sw = False 
          # self.sw_solar_weights = nn.Parameter(torch.zeros(1, self.ng_sw)) 
        if self.ng_sw==112:
          self.is_rrtmgp=True
          print("Using existing gas optics models (RRTMGP-NN)")
        else:
          self.is_rrtmgp=False 
          print("Training new gas optics schemes on the fly!!")

    def forward(self, x_gas, # inputs to gas optics NN model, already normalised
                col_dry, mu0, incoming_toa, pres_lev, # unnormalised variables used in radiative transfer computations 
                albedo_surf_dir_sw, # If albedo_surf_diff_sw is None, this is general albedo used for both dir and diff
                albedo_surf_diff_sw=None):

        printdebug = False 

        batch_size, nlay, nx = x_gas.shape 
        nlev = nlay + 1 
        device = x_gas.device

        # Transpose arrays, because the RTE is faster with levels outermost 
        x_gas = torch.transpose(x_gas,0,1).contiguous()
        col_dry = torch.transpose(col_dry,0,1).unsqueeze(-1).contiguous()
        pres_lev = torch.transpose(pres_lev,0,1).contiguous()
        #   print("shape gas", x_gas.shape, "coldry", col_dry.shape, "pres", pres_lev.shape)

        if albedo_surf_diff_sw is None:
            albedo_surf_diff_sw = torch.clone(albedo_surf_dir_sw)

        # -------------------------- SHORTWAVE -----------------------------
        # 

        # GAS OPTICAL PROPERTIES IN EACH LAYER
        # x_gas = torch.cat((temp, pres, vmr_h2o, o3, co2, n2o, ch4), dim=2)
        # x_gas = (x_gas - self.gas_optics_model_sw_abs.xmin) / self.gas_optics_model_sw_abs.div

        # print("col_dry mean", col_dry.mean().item())
        # for i in range(x_gas.shape[-1]):
        #     print("x_gas {} min {} max {} mean {}".format(i,torch.min(x_gas[:,:,i]).item(), torch.max(x_gas[:,:,i]).item(),  torch.mean(x_gas[:,:,i]).item()))

        if self.is_rrtmgp:
          tau_sw      = self.gas_optics_model_sw_abs(x_gas, col_dry)
          tau_sw_scat = self.gas_optics_model_sw_ray(x_gas, col_dry)
        else:
          # coeff=1e-17
          # coeff=1e-21
          # coeff=1e-16
          tau_sw      = self.gas_optics_model_sw_abs(x_gas, col_dry) #*coeff
          tau_sw      = torch.clamp(tau_sw,min=1e-9)
          tau_sw_scat = self.gas_optics_model_sw_ray(x_gas, col_dry) #*coeff

          # print("tau sw mean", tau_sw.mean().item(), "max", tau_sw.max().item())
          # print("tau sw scat mean", tau_sw_scat.mean().item(), "max", tau_sw_scat.max().item())

        tau_sw      = tau_sw + tau_sw_scat 

        ssa_sw  = tau_sw_scat / tau_sw
        g_sw    = torch.zeros_like(ssa_sw) # asymmetry factor is zero for gases

        # Here we set the cosine of solar zenith angle to a minimum value, later we set the fluxes in night-time columns to zero
        min_mu = 1e-6 #1e-3
        mu0_comp = torch.clamp(mu0, min=min_mu) 

        # expand mu0 (cosine of solar zenith angle) from (ncol) -> (nlay,ncol,ng_sw)
        mu0_rep = mu0_comp.reshape((1,-1,1)).expand(nlay, -1, self.ng_sw).contiguous()

        # SW REFLECTANCE-TRANSMITTANCE COMPUTATIONS FOR EACH LAYER
        if printdebug: 
            print("tau_sw min max mean", tau_sw.min().item(), tau_sw.max().item(), tau_sw.mean().item())
            print("tau_sw_scat min max mean", tau_sw_scat.min().item(), tau_sw_scat.max().item(), tau_sw_scat.mean().item())

            # print("tau sw lev29 col0", tau_sw[29,0])
            # print("tau sw scat lev29 col0", tau_sw_scat[29,0])


            print("ssa_sw min max mean", ssa_sw.min().item(), ssa_sw.max().item(), ssa_sw.mean().item()) 
            print("g_sw min max mean", g_sw.min().item(), g_sw.max().item(), g_sw.mean().item()) 
            
        ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir = calc_ref_trans_sw(mu0_rep, tau_sw, ssa_sw, g_sw)

        # print("ref diff shape", ref_diff.shape, "mu0", mu0_rep.shape, "tau", tau_sw.shape, "nlay", nlay)
        # print("tau_sw 29,0 :", tau_sw[29,0,:], "mean", tau_sw.mean().item())

        # print("ref diff 29,0 :", ref_diff[29,0,:], "mean", ref_diff.mean().item())
        # print("tra diff 29,0 :", trans_diff[29,0,:], "mean", trans_diff.mean().item())
        # print("ref dir 29,0 :", ref_dir[29,0,:], "mean", ref_dir.mean().item())
        # print("trans_dir_diff 29,0 :", trans_dir_diff[29,0,:], "mean", trans_dir_diff.mean().item())
        # print("tnoscat 29,0 :", trans_dir_dir[29,0,:], "mean", trans_dir_dir.mean().item())

        ref_diff            = ref_diff.view(nlay, -1)
        trans_diff          = trans_diff.view(nlay, -1)
        ref_dir             = ref_dir.view(nlay, -1)
        trans_dir_diff      = trans_dir_diff.view(nlay, -1)
        trans_dir_dir       = trans_dir_dir.view(nlay, -1)
        del tau_sw, ssa_sw, g_sw#, mu0_rep

        if (self.is_rrtmgp):
            toa_spectral = self.gas_optics_model_sw_abs.sw_solar_weights 
        else:
            # Here we apply softmax to ensure the solar weights sum to 1 (and are positive)
            toa_spectral = self.gas_optics_model_sw_abs.get_solar_weights()
            # toa_spectral = torch.softmax(self.gas_optics_model_sw_abs.sw_solar_weights,dim=-1)
        
        # print("shape inc toa", incoming_toa.shape, "toa spectral", toa_spectral.shape, "mu", mu0_comp.shape)        
        # print("sum spectral", toa_spectral.sum().item(), "inc toa 1", incoming_toa[0])
        # print("sum band 0", torch.sum(toa_spectral[:,self.gas_optics_model_sw_abs.band_bounds[0]:self.gas_optics_model_sw_abs.band_bounds[1]]))
        # print("sum band rrtmgp", torch.sum(self.gas_optics_model_sw_abs.rrtmgp_sw_solar_weights[:,RRTMGP_BOUNDS[0]:RRTMGP_BOUNDS[1]]))

        incoming_toa = incoming_toa.unsqueeze(1)*toa_spectral*mu0_comp.unsqueeze(1)
        if printdebug:
          print("inc toa 1 sum", incoming_toa[2].sum(), "mu0 1", mu0_comp[2])
          print("max min sum inc toa -1 ", incoming_toa.sum(dim=-1).max().item(),  incoming_toa.sum(dim=-1).min().item())
        incoming_toa = incoming_toa.view(-1)

 
        albedo_surf_dir_sw    = albedo_surf_dir_sw.unsqueeze(1).expand(-1,self.ng_sw).contiguous().view(-1)
        albedo_surf_diff_sw   = albedo_surf_diff_sw.unsqueeze(1).expand(-1,self.ng_sw).contiguous().view(-1)

        # --------- SW RADIATIVE TRANSFER USING ADDING METHOD -----------
        # print("shape inc toa", incoming_toa.shape, "alb", albedo_surf_diff_sw.shape, "ref", ref_diff.shape)

        if printdebug: 
            print("incoming_toa min max mean", incoming_toa.min().item(), incoming_toa.max().item(), incoming_toa.mean().item())
            print("albedo_surf_diff_sw min max", albedo_surf_diff_sw.min().item(), albedo_surf_diff_sw.max().item())
            print("ref_diff min max mean", ref_diff.min().item(), ref_diff.max().item(), ref_diff.mean().item()) 
            
        flux_sw_up_gpt, flux_sw_dn_diffuse_gpt, flux_sw_dn_direct_gpt = adding_ica_sw(
                    incoming_toa, albedo_surf_diff_sw, albedo_surf_dir_sw, 
                    ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir, mu0_rep[0].view(-1))

        del ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir

        flux_sw_up_gpt = torch.reshape(flux_sw_up_gpt, (nlev, batch_size, self.ng_sw))
        flux_sw_dn_diffuse_gpt = torch.reshape(flux_sw_dn_diffuse_gpt, (nlev, batch_size, self.ng_sw))
        flux_sw_dn_direct_gpt = torch.reshape(flux_sw_dn_direct_gpt, (nlev, batch_size, self.ng_sw))
        
        # if self.return_gpt_fluxes: # To be properly implemented later
        #     # RRTMGP bands and g-points:
        #     # bnd_limits_gpt
        #     # 1,10     | 11,18    | 19,29    | 30,37   | 38,46     | 47,56     | 57,67     | 68,71      | 72,80      |  81,89    | 90, 96    | 97, 102   | 103, 109 | 110, 112 
        #     # bnd_limits_wavenumber
        #     # 820,2680 | 2680,3250 | 3250,4k | 4k,4650 | 4650,5150 | 5150,6150 | 6150,7700 | 7700,8050  | 8050,12850 | 12850,16k | 16k,22650 | 22650,29k | 29k,38k  | 38k,50k 
        #     # in micrometers 
        #     # 12.2,3.73| 3.73,3.08 | 3.08,2.5| 2.5,2.15| 2.15,1.94 | 1.94,1.63 | 1.63,1.3  | 1.3,1.24   | 1.24,0.78  | 0.78,0.62 | 0.62,0.44 | 0.44,0.34 | 0.34,0.26| 0.26  0.2 ]
        #     # band 1        2         3           4           5         6             7         8              9           10         11          12         13         14
        #     # E3SM wants:
        #     # ! sols(pcols)      Direct solar rad on surface (< 0.7 micrometer)
        #     # ! soll(pcols)      Direct solar rad on surface (>= 0.7 micrometer)
        #     iend_ir = int(round((80/112)*self.ng_sw)) # RRTMGP bands 1-9 (g-points 1-80) encompass 820-12850 cm-1 (near-ir), see data/rrtmgp-data-sw-g112-210809.nc
        #     iend_mix= int(round((89/112)*self.ng_sw)) # RRTMGP band 10 is in between UV/visible and near-IR, and bands 11-14 (89-112) are fully in visible range (> 14286 ! cm^-1)

        #     # Sum over whole / parts of spectral dimension to get broadband / band-wise fluxes
        #     sw_dir_dn_mixband = torch.sum(flux_sw_dn_direct[-1,:,iend_ir:iend_mix],dim=1,keepdim=True)
        #     SOLL = torch.sum(flux_sw_dn_direct[-1,:,0:iend_ir],dim=1,keepdim=True) + 0.5*sw_dir_dn_mixband
        #     SOLS = torch.sum(flux_sw_dn_direct[-1,:,iend_mix:],dim=1,keepdim=True) + 0.5*sw_dir_dn_mixband

        #     sw_diff_dn_mixband = torch.sum(flux_sw_dn_diffuse[-1,:,iend_ir:iend_mix],dim=1,keepdim=True)
        #     SOLLD = torch.sum(flux_sw_dn_diffuse[-1,:,0:iend_ir],dim=1,keepdim=True) + 0.5*sw_diff_dn_mixband
        #     SOLSD = torch.sum(flux_sw_dn_diffuse[-1,:,iend_mix:],dim=1,keepdim=True) + 0.5*sw_diff_dn_mixband

        flux_sw_up          = torch.sum(flux_sw_up_gpt,dim=2)
        flux_sw_dn_diffuse  = torch.sum(flux_sw_dn_diffuse_gpt,dim=2)
        flux_sw_dn_direct   = torch.sum(flux_sw_dn_direct_gpt,dim=2)
        if self.return_gpt_fluxes:
          flux_sw_dn_gpt = flux_sw_dn_diffuse_gpt + flux_sw_dn_direct_gpt
          flux_sw_dn_gpt = torch.transpose(flux_sw_dn_gpt,0,1)
          flux_sw_dn_direct_gpt = torch.transpose(flux_sw_dn_direct_gpt,0,1)
          flux_sw_up_gpt = torch.transpose(flux_sw_up_gpt,0,1)

        else:
          del flux_sw_up_gpt, flux_sw_dn_diffuse_gpt, flux_sw_dn_direct_gpt

        flux_sw_dn          = flux_sw_dn_diffuse + flux_sw_dn_direct
        flux_sw_dn_sfc      = flux_sw_dn[-1,:].unsqueeze(1)             # NETSW  

        if printdebug:
            print("flux dn mean", flux_sw_dn.mean().item())
            # print("flux sw dn col 1 ", flux_sw_dn[:,0])
            # print("flux sw dn dir col 1 ", flux_sw_dn_direct[:,0])
            # print("flux sw up col 1 ", flux_sw_up[:,0])

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
        # print("shape flux sw up 0", flux_sw_up.shape)
        dT_rad            = torch.transpose(dT_rad,0,1)
        flux_sw_up        = torch.transpose(flux_sw_up,0,1)
        flux_sw_dn_direct = torch.transpose(flux_sw_dn_direct,0,1)
        flux_sw_dn        = torch.transpose(flux_sw_dn,0,1)

        # print("shape flux sw up 1", flux_sw_up.shape)

        if self.return_gpt_fluxes:
            # SOLL[inds_zero] = 0.0
            # SOLS[inds_zero] = 0.0
            # SOLLD[inds_zero] = 0.0
            # SOLSD[inds_zero] = 0.0
            # return dT_rad, flux_sw_up, flux_sw_dn, flux_sw_dn_direct, SOLS, SOLL, SOLSD, SOLLD
            return dT_rad, flux_sw_up_gpt, flux_sw_dn_gpt, flux_sw_dn_direct_gpt
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
# @torch.compile(dynamic=False)
def adding_ica_sw_batchlast_opt(incoming_toa, albedo_surf_diffuse, albedo_surf_direct,
                R, T, ref_dir, T_dir_diff, T_dir_dir, cos_sza):
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

        # print("albedo toa", albedo[0].view(-1,112)[0,:])
        # print(" albedo mean",  torch.mean(albedo[0].view(-1,112)[0,:]))
        # print("albedo sfc", albedo[-1].view(-1,112)[0,:])
        # print("albedo sfc -1", albedo[-2].view(-1,112)[0,:])
        # print(" albedo mean",  torch.mean(albedo[-2].view(-1,112)[0,:]))

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
            # fluxdndir = fluxdndir * cos_sza
            flux_dn_direct  += [fluxdndir]
            flux_dn_diffuse += [fluxdndiff]

            fluxup = fluxdndir*albedodir[jlev+1] + fluxdndiff* albedo[jlev + 1]
            flux_up += [fluxup]       
        
        flux_dn_direct  = torch.stack(flux_dn_direct)
        flux_dn_diffuse = torch.stack(flux_dn_diffuse)
        flux_up = torch.stack(flux_up)

        return flux_up, flux_dn_diffuse, flux_dn_direct

# @torch.jit.script
# @torch.compile(dynamic=False)
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
    reflectance, transmittance, ref_dir, trans_dir_diff, trans_dir_dir, mu0
):
    # if torch.is_grad_enabled():
        # print("calling normal adding!")
    return adding_ica_sw_batchlast_opt(
        incoming_toa, albedo_surf_diffuse, albedo_surf_direct,
        reflectance, transmittance, ref_dir, trans_dir_diff, trans_dir_dir, mu0
    )
    # else:
    #     # print("calling inference adding!")
    #     return adding_ica_sw_inference(
    #         incoming_toa, albedo_surf_diffuse, albedo_surf_direct,
    #         reflectance, transmittance, ref_dir, trans_dir_diff, trans_dir_dir
    #     )
