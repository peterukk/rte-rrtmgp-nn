#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytorch implementation of a two-stream SW radiation scheme using neural network gas optics models, plus helper functions for loading existing gas optics models
Differentiable radiative transfer equations (RTEs) allows training new gas optics models on fluxes (in specific bands or broadband fluxes), in which case 
the spectral decomposition (in each band) becomes fully machine-learned (and may not use the correlated-k assumption)
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
from coefficients import rrtmgp_sw_solar_source, RRTMGP_GPT_BOUNDS_SW, RRTMGP_WAVENUM_LOW_SW, RRTMGP_WAVENUM_HIGH_SW

# from coefficients import RRTMGP_SPLITS, WAVENUM_SPLITS  # band limits and corresponding wavenumbers
# Build slice boundaries: [0] + splits + [ng]
# RRTMGP_BOUNDS = [0] + RRTMGP_SPLITS + [112]

def load_gas_optics_from_file(device, existing_gasopt_file, existing_gasopt_file2=None):
  if 'sw' in existing_gasopt_file:
    shortwave=True
    print(f"Loading pre-existing shortwave !ABSORPTION! gas optics model from {existing_gasopt_file}")
  elif 'lw' in existing_gasopt_file:
    print(f"Loading pre-existing longwave !COMBINED! (Planck fraction + optical depth) gas optics model from {existing_gasopt_file}")
    shortwave=False
    if existing_gasopt_file2 is not None:
      raise NotImplementedError("Only combined LW files supported, please only provide 1 file")
  else:
    raise NotImplementedError("file name not recognized - should have identifier sw or lw")

  mlp_gasopt_model= load_gas_optics_model(existing_gasopt_file, device)
  if shortwave: #existing_gasopt_file_sw_ray is not None:
    print("Loading pre-existing shortwave !RAYLEIGH! gas optics model from {}".format(existing_gasopt_file2))
    mlp_gasopt_model_sw_ray = load_gas_optics_model(existing_gasopt_file2, device)
    return mlp_gasopt_model, mlp_gasopt_model_sw_ray
  else:
    return mlp_gasopt_model 

def load_gas_optics_model(gasopt_file, device, lock_weights=False):
  # Load model from NetCDF file
  ds = xr.open_dataset(gasopt_file)
  shortwave = True if "sw" in gasopt_file else False 
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

  if shortwave:
    from coefficients import rrtmgp_sw_solar_source
    rrtmgp_sw_solar_source = rrtmgp_sw_solar_source/np.sum(rrtmgp_sw_solar_source)

    nn = mlp_gasopt_inlined_processing(device=device, 
                        xmin=xnorm_min, xmax=xnorm_max, 
                        ymean=ynorm_mean, ystd=ynorm_std,
                        nn_w1=nn_w1, nn_w2=nn_w2, nn_w3=nn_w3,
                        nn_b1=nn_b1, nn_b2=nn_b2, nn_b3=nn_b3, 
                        solar_source=rrtmgp_sw_solar_source,
                        lock_weights=lock_weights)
  else:
    nn = mlp_gasopt_inlined_processing(device=device, 
                        xmin=xnorm_min, xmax=xnorm_max, 
                        ymean=ynorm_mean, ystd=ynorm_std,
                        nn_w1=nn_w1, nn_w2=nn_w2, nn_w3=nn_w3,
                        nn_b1=nn_b1, nn_b2=nn_b2, nn_b3=nn_b3, 
                        lock_weights=lock_weights)    
  nn.eval()
  infostr = summary(nn)
  return nn 

def rrtmgp_bounds_to_wavenum_bounds(rrtmgp_band_bounds):
    """
    Convert custom band boundaries expressed in RRTMGP g-point space
    (e.g. [0, 29, 80, 89, 102, 112]) to wavenumber boundaries (cm-1),
    using the actual RRTMGP band edges — not nominal design targets.

    Each boundary g must coincide with an RRTMGP band edge (i.e. g must
    appear in RRTMGP_GPT_BOUNDS_SW); raises if not, since a non-aligned
    boundary cannot be represented exactly in RRTMGP g-point space.
    """
    wavenum_bounds = []
    for g in rrtmgp_band_bounds:
        if g == 0:
            wavenum_bounds.append(RRTMGP_WAVENUM_LOW_SW[0])      # 820
        elif g == 112:
            wavenum_bounds.append(RRTMGP_WAVENUM_HIGH_SW[-1])    # 50000
        else:
            assert g in RRTMGP_GPT_BOUNDS_SW, (
                f"g-point boundary {g} does not align with any RRTMGP band edge "
                f"{RRTMGP_GPT_BOUNDS_SW}. Custom band boundaries must coincide with "
                f"RRTMGP band edges."
            )
            band_idx = RRTMGP_GPT_BOUNDS_SW.index(g)
            # g is the END of band (band_idx - 1) and START of band band_idx
            # Use the wavenumber at that shared edge
            wavenum_bounds.append(RRTMGP_WAVENUM_LOW_SW[band_idx])
    return wavenum_bounds

def make_band_coordinate_vectors(band_bounds, ng, device=None, dtype=torch.float32):
    """
    Returns:
        band_index: shape (ng,), integer band index for each g-point
        band_coord: shape (ng,), coordinate from 0 to 1 within each band
    """
    band_index = torch.empty(ng, device=device, dtype=torch.long)
    band_coord = torch.empty(ng, device=device, dtype=dtype)

    for iband in range(len(band_bounds) - 1):
        i0 = int(band_bounds[iband])
        i1 = int(band_bounds[iband + 1])
        n = i1 - i0

        if n <= 0:
            raise ValueError(f"Bad band_bounds: {band_bounds}")

        band_index[i0:i1] = iband

        if n == 1:
            band_coord[i0:i1] = 1.0
        else:
            band_coord[i0:i1] = torch.linspace(
                0.0,
                1.0,
                n,
                device=device,
                dtype=dtype,
            )

    return band_index, band_coord

class mlp_gasopt_inlined_processing(nn.Module):
    """
    Gas optics neural networks: differs from GasOpticsMLP in that the post-processing is inlined.
    This version if meant for training new gas optics in which case we don't have output normalisation coefficients (do_norm=false),
    but it can also used with pre-trained gas optics model, in which case output scaling coefficients, weights (nn_w1,..) and 
    solar_source must be provided.
    If we are training a new model from scratch, ny (number of g-points) and nh (hidden neurons) are hyperparameters.
    """
    lock_weights: Final[bool]
    do_norm: Final[bool]
    is_rrtmgp: Final[bool]
    monotonic_prior: Final[bool]
    is_longwave: Final[bool]
    # extra_layer: Final[bool]
    def __init__(self, device, 
                xmin, xmax, ymean=None, ystd=None,
                nn_w1=None, nn_w2=None, nn_w3=None,
                nn_b1=None, nn_b2=None, nn_b3=None, 
                solar_source=None,
                lock_weights = True,
                do_norm=False,
                ng=16, nh=32,
                band_bounds=None,
                rrtmgp_bounds_in=None, 
                wavenum_splits_in=None,
                is_longwave=None):
        super(mlp_gasopt_inlined_processing, self).__init__()
        self.nx = xmin.shape[0]
        self.do_norm = False
        self.monotonic_prior=False
        if is_longwave is not None:
          self.is_longwave=is_longwave 
        else:
          if solar_source is None:
            self.is_longwave=True
            print("neither is_longwave nor solar_source provided, setting is_longwave to True")
          else:
            self.is_longwave=False
        if ymean is not None:
          self.ny = ymean.shape[0]
          # self.ng = self.ny
          if self.is_longwave:
              self.ng = self.ny//2
          else:
              self.ng = self.ny
          ymean = torch.from_numpy(ymean[0:self.ng])
          ystd  = torch.from_numpy(ystd[0:self.ng])
          self.register_buffer('ymean', ymean)
          self.register_buffer('ystd',  ystd)
          print("Loaded existing y normalisation coefficients")
          self.do_norm = True
        else:
          if self.is_longwave:
            self.ny = 2*ng 
          else:
            self.ny = ng 
          self.ng = ng
          self.do_norm = do_norm
          if self.do_norm:
            self.ymean = 0 # # nn.Parameter(torch.zeros(self.ng)) 
            self.ystd = 1# 0.00060 # nn.Parameter(torch.zeros(1)) 
            print("Using learnable y normalisation coefficients (may not work)")
        print("Is_longwave", self.is_longwave, "Ng:", self.ng, "Ny:", self.ny)
        if self.ng in [112,128]:
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
            # self.rrtmgp_bounds = RRTMGP_BOUNDS
            # self.wavenum_splits = WAVENUM_SPLITS
            raise ValueError(
                            "rrtmgp_bounds_in must be provided")
          self.num_bands = len(band_bounds) - 1 
          print("Number of bands: {}".format(self.num_bands))
          # self.register_buffer("band_bounds", band_bounds)
        else:
          self.num_bands = 1

        if band_bounds is not None:
            print("band bounds is {} and rrtmgp_bounds is {}, now deriving wavenumber bounds from these".format(band_bounds, self.rrtmgp_bounds))
           
            # self.wavenum_bounds: List[int] = wavenum_bounds
            self.wavenum_bounds: List[int] = rrtmgp_bounds_to_wavenum_bounds(self.rrtmgp_bounds)
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
    
            if self.monotonic_prior:
                band_index, band_coord = make_band_coordinate_vectors(
                    self.band_bounds,
                    ng=self.ng,
                    device=device,
                    dtype=torch.float32,
                )

                self.register_buffer("band_index", band_index)
                self.register_buffer("band_coord", band_coord.reshape(1, 1, self.ng))

                # One learned steepness per band.
                # raw parameter is unconstrained; softplus makes steepness positive.
                self.band_log_slope = nn.Parameter(torch.zeros(self.num_bands))

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
        if self.monotonic_prior:
          x = x * self.get_monotonic_shape_factor()
        if self.is_longwave:
          tau, pfrac = x.chunk(2,-1)
          pfrac = torch.square(pfrac)
          if not self.is_rrtmgp: 
            pfrac = torch.softmax(pfrac,dim=-1)
        else:
          tau = x 
        # Postprocessing inlined: reverse power root scaling, multiply with number of dry air molecules
        if self.do_norm:
          tau = col_dry * torch.pow(self.ystd*tau + self.ymean,8)
        else:
          tau = col_dry * torch.pow(tau,8)
          # print("mean tau after coldry, pow8", tau.mean().item())
        # coeff=1e-17
        coeff=1e-16
        if not self.is_rrtmgp:
          tau = tau*coeff
        if self.is_longwave:
          return tau, pfrac 
        else:
          return tau

    def get_monotonic_shape_factor(self):
        """
        Returns shape factor of shape (1, 1, ng), increasing within each band.

        Each band has its own learned exponential steepness.
        The factor is normalized so the mean within each band is ~1, avoiding
        a strong change in overall optical-depth scale.
        """
        if self.band_bounds is None:
            return 1.0

        # Positive steepness per band
        slope_per_band = torch.nn.functional.softplus(self.band_log_slope)

        # Map each g-point to its band's slope
        slope_g = slope_per_band[self.band_index].reshape(1, 1, self.ng)

        # Exponential ramp within each band
        shape = torch.exp(slope_g * self.band_coord)

        # Normalize each band so mean factor in that band is 1.
        pieces = []
        for iband in range(self.num_bands):
            i0 = self.band_bounds[iband]
            i1 = self.band_bounds[iband + 1]

            s = shape[..., i0:i1]
            s = s / s.mean(dim=-1, keepdim=True)
            pieces.append(s)

        return torch.cat(pieces, dim=-1)

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
                albedo_surf_diff_sw=None,
                printdebug=False):

        # printdebug = False 

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
        if printdebug:
          for ix in range(nx):
            print("gas i", ix, "min", x_gas[:,:,ix].min().item(), "max", x_gas[:,:,ix].max().item())

        if self.is_rrtmgp:
          tau_sw      = self.gas_optics_model_sw_abs(x_gas, col_dry)
          tau_sw_scat = self.gas_optics_model_sw_ray(x_gas, col_dry)
        else:
          tau_sw      = self.gas_optics_model_sw_abs(x_gas, col_dry) 
          tau_sw      = torch.clamp(tau_sw,min=1e-9)
          tau_sw_scat = self.gas_optics_model_sw_ray(x_gas, col_dry) 
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
            
        flux_sw_up_gpt, flux_sw_dn_diffuse_gpt, flux_sw_dn_direct_gpt = adding_ica_sw_batchlast_opt(
                    incoming_toa, albedo_surf_diff_sw, albedo_surf_dir_sw, 
                    ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir, mu0_rep[0].view(-1))

        del ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir

        flux_sw_up_gpt = torch.reshape(flux_sw_up_gpt, (nlev, batch_size, self.ng_sw))
        flux_sw_dn_diffuse_gpt = torch.reshape(flux_sw_dn_diffuse_gpt, (nlev, batch_size, self.ng_sw))
        flux_sw_dn_direct_gpt = torch.reshape(flux_sw_dn_direct_gpt, (nlev, batch_size, self.ng_sw))
        
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
            return dT_rad, flux_sw_up_gpt, flux_sw_dn_gpt, flux_sw_dn_direct_gpt
        else:
            return dT_rad, flux_sw_up, flux_sw_dn, flux_sw_dn_direct



class LW_rad_torch(nn.Module):
    """
    If return_gpt_fluxes=False, returns:
        - dT_rad (ncol, nlay) : longwave heating rate
        - flux_lw_up (ncol, nlay+1)     : upwelling longwave flux 
        - flux_sw_dn (ncol, nlay+1)     : downwelling longwave flux 
    If return_gpt_fluxes=True, returns the same but fluxes are spectral: (ncol, nlay+1, ng)
    """
    use_existing_gas_optics_lw: Final[bool] # Use existing gas optics model (RRTMGP-SW emulator)
    return_gpt_fluxes: Final[bool]
    is_rrtmgp: Final[bool]
    
    def __init__(
        self,
        device: torch.device,
        gas_optics_model_lw: mlp_gasopt_inlined_processing,
        ng_lw: int = 128,
        return_gpt_fluxes: bool = False,
        rrtmgp_coeff_file_lw: str | None = None,
    ):
        super().__init__()

        self.return_gpt_fluxes = return_gpt_fluxes
        self.gas_optics_model_lw = gas_optics_model_lw
        self.ng_lw = self.gas_optics_model_lw.ng

        if self.ng_lw == 128:
            self.is_rrtmgp = True
            print("Using existing gas optics models (RRTMGP-NN)")
        else:
            self.is_rrtmgp = False
            print("Training new gas optics schemes on the fly!!")

        # RRTMGP Planck lookup information is only needed for the original
        # 128-g-point RRTMGP(-NN) spectral representation.
        if self.is_rrtmgp:
            if rrtmgp_coeff_file_lw is None:
                raise ValueError(
                    "rrtmgp_coeff_file_lw must be provided when using "
                    "the 128-g-point RRTMGP-NN longwave model"
                )

            with xr.open_dataset(rrtmgp_coeff_file_lw) as ds:
                totplnk = np.asarray(ds["totplnk"].values, dtype=np.float32)
                bnd_limits_gpt = np.asarray(
                    ds["bnd_limits_gpt"].values, dtype=np.int64
                )
                temp_ref = np.asarray(ds["temp_ref"].values, dtype=np.float32)

            # totplnk shape: (nband=16, ntemp=196)
            self.nbnd_lw = totplnk.shape[0]
            ntemp_planck = totplnk.shape[1]

            self.register_buffer(
                "totplnk",
                torch.from_numpy(totplnk),
            )

            # The Planck table spans temp_ref_min .. temp_ref_max with
            # ntemp_planck uniformly spaced points.
            temp_ref_min = float(temp_ref[0])
            temp_ref_max = float(temp_ref[-1])

            self.temp_ref_min_planck = temp_ref_min
            self.temp_ref_max_planck = temp_ref_max
            self.totplnk_delta = (
                (temp_ref_max - temp_ref_min) / float(ntemp_planck - 1)
            )

            print(
                f"LW Planck lookup: Tmin={self.temp_ref_min_planck}, "
                f"Tmax={self.temp_ref_max_planck}, "
                f"dT={self.totplnk_delta}, "
                f"shape={totplnk.shape}"
            )

            # bnd_limits_gpt is (16, 2), 1-based inclusive in the RRTMGP file.
            # Build zero-based g-point -> band mapping.
            gpt_to_band = np.empty(self.ng_lw, dtype=np.int64)

            for ibnd, (gpt_start, gpt_end) in enumerate(bnd_limits_gpt):
                i0 = int(gpt_start) - 1
                i1 = int(gpt_end)       # Python upper-exclusive
                gpt_to_band[i0:i1] = ibnd

            self.register_buffer(
                "gpt_to_band",
                torch.from_numpy(gpt_to_band),
            )

    def interpolate_totplnk(self, temperature: torch.Tensor) -> torch.Tensor:
        """
        Interpolate RRTMGP total Planck irradiance by band.

        Parameters
        ----------
        temperature
            Any shape (...), in K.

        Returns
        -------
        planck_band
            Shape (..., nbnd_lw), in the same units as totplnk.
        """

        # Match the lookup-table range.
        T = torch.clamp(
            temperature,
            min=self.temp_ref_min_planck,
            max=self.temp_ref_max_planck,
        )

        # Floating-point location in the 196-point table.
        fidx = (
            (T - self.temp_ref_min_planck)
            / self.totplnk_delta
        )

        idx0 = torch.floor(fidx).long()
        idx0 = torch.clamp(idx0, 0, self.totplnk.shape[1] - 2)

        idx1 = idx0 + 1
        frac = fidx - idx0.to(fidx.dtype)

        # self.totplnk: (nband, ntemp)
        #
        # Indexing by idx0 gives:
        #     (nband, ...)
        p0 = self.totplnk[:, idx0]
        p1 = self.totplnk[:, idx1]

        frac = frac.unsqueeze(0)

        planck = p0 + frac * (p1 - p0)
        # RRTMGP totplnk contains band-integrated Planck radiance.
        # Convert radiance to hemispheric irradiance/flux:
        #     F_band = pi * B_band
        planck = torch.pi * planck
        # Move band dimension from first to last:
        #     (nband, ...) -> (..., nband)
        dims = list(range(1, planck.ndim)) + [0]
        return planck.permute(*dims)

    def forward(self, 
                x_gas, # inputs to gas optics NN model, already normalised
                col_dry, temp_lev, pres_lev, temp_sfc, emis_sfc, # unnormalised variables
                printdebug=False):

        # printdebug = False 

        batch_size, nlay, nx = x_gas.shape 
        nlev = nlay + 1 
        device = x_gas.device

        # Transpose arrays, because the RTE is faster with levels/layers outermost (columns being contiguous)
        x_gas = torch.transpose(x_gas,0,1).contiguous()
        col_dry = torch.transpose(col_dry,0,1).unsqueeze(-1).contiguous()
        pres_lev = torch.transpose(pres_lev,0,1).contiguous()
        temp_lev = torch.transpose(temp_lev,0,1).contiguous()
        if printdebug:
          for ix in range(nx):
            print("gas i", ix, "min", x_gas[:,:,ix].min().item(), "max", x_gas[:,:,ix].max().item())

        # Call NN gas optics to compute optical depth and Planck fractions, which sum to 1 along the spectral dim
        tau_lw, pfrac      = self.gas_optics_model_lw(x_gas, col_dry)
        if self.is_rrtmgp:
            # ---------------------------------------------------------------
            # Original RRTMGP spectral representation:
            #
            # pfrac sums to ~1 WITHIN EACH BAND, hence ~16 over all 128 gpts.
            # totplnk supplies the blackbody irradiance for each of the 16 bands.
            # ---------------------------------------------------------------

            planck_band_lev = self.interpolate_totplnk(temp_lev)
            # (nlev, batch, 16)
            pp = planck_band_lev.sum(dim=-1)

            planck_gpt_lev = planck_band_lev[..., self.gpt_to_band]
            # (nlev, batch, 128)

            planck_band_sfc = self.interpolate_totplnk(temp_sfc)
            # (batch, 16)

            planck_gpt_sfc = planck_band_sfc[..., self.gpt_to_band]
            # (batch, 128)

            # IMPORTANT:
            # pfrac belongs to a LAYER.
            #
            # The same layer pfrac is used with the Planck function at both
            # bounding levels of that layer, matching the Fortran implementation.
            planck_top = pfrac * planck_gpt_lev[:-1]
            planck_bot = pfrac * planck_gpt_lev[1:]

            # RRTMGP uses the Planck fractions of the lowest atmospheric layer
            # for the surface source.
            source_sfc = pfrac[-1] * planck_gpt_sfc

        else:
            tau_lw      = torch.clamp(tau_lw,min=1e-9)
            # ---------------------------------------------------------------
            # New learned spectral representation:
            #
            # pfrac is normalized across the complete learned spectral dimension,
            # so broadband sigma*T^4 can be distributed directly using pfrac.
            # ---------------------------------------------------------------

            planck_lev = outgoing_lw(temp_lev)
            # (nlev, batch)

            planck_top = pfrac * planck_lev[:-1].unsqueeze(-1)
            planck_bot = pfrac * planck_lev[1:].unsqueeze(-1)

            planck_sfc = outgoing_lw(temp_sfc)
            source_sfc = pfrac[-1] * planck_sfc.unsqueeze(-1)

        # if printdebug:
        #   print("tau lw min max mean", tau_lw.min().item(), tau_lw.max().item(), tau_lw.mean().item())
        #   print("pfrac min max", pfrac.min().item(), pfrac.max().item(), "sum max", pfrac.sum(dim=-1).max().item())
        # olr_lev    = torch.unsqueeze(outgoing_lw(temp_lev),2) # (nlev, nb, 1)
        # if printdebug:
        #   print("olr_lev max", olr_lev.max().item(), "mean", olr_lev.mean().item())
        # source_lev  = torch.zeros(nlev, batch_size, self.ng_lw, device=device)
        # source_lev[-1,:,:] = pfrac[-1,:,:] * olr_lev[-1,:,:]
        # source_lev[0:-1,:,:] = pfrac[:,:,:]  * olr_lev[0:-1,:,:]

        # olr_sfc    = torch.unsqueeze(outgoing_lw(temp_sfc),1) # (nb, 1)
        # source_sfc  = pfrac[-1,:,:]*olr_sfc # (nb, ng_lw)

        # # Computation of layer-wise LW transmittances and source terms 
        # planck_top = source_lev[0:-1,:,:]
        # planck_bot = source_lev[1:,:,:]
        if printdebug:
          print("planck_bot mean per lev", planck_bot.mean(dim=(1,2)))
        source_up, source_dn, trans_lw = reftrans_lw(planck_top.view(-1),planck_bot.view(-1), tau_lw.view(-1))
        if printdebug:
          print("trans min max mean", trans_lw.min().item(), trans_lw.max().item(), trans_lw.mean())
          print("source_dn sum(dim=-1) mean per lev", source_dn.view(-1,self.ng_lw).sum(dim=-1).view(nlay, -1).mean(dim=-1))

        del tau_lw, planck_top, planck_bot#, source_lev

        # Provided emissivity is broadband, expand to spectral
        emissivity_surf = torch.repeat_interleave(emis_sfc.unsqueeze(1),self.ng_lw,dim=1)

        # call LW solver to predict downward and upward spectral fluxes at each half-level. LW scattering is ignored here
        flux_lw_dn_gpt, flux_lw_up_gpt = lw_solver_noscat_batchlast(trans_lw.view(nlay, -1), source_dn.view(nlay, -1), source_up.view(nlay, -1), 
                                                            source_sfc.view(-1), emissivity_surf.view(-1))

        flux_lw_up_gpt = flux_lw_up_gpt.view(nlev,batch_size, self.ng_lw)
        flux_lw_dn_gpt = flux_lw_dn_gpt.view(nlev,batch_size, self.ng_lw)

        flux_lw_up  = torch.sum(flux_lw_up_gpt,dim=2)
        flux_lw_dn  = torch.sum(flux_lw_dn_gpt,dim=2)

        flux_lw_net = flux_lw_dn - flux_lw_up

        # COMPUTE HEATING RATES
        flux_diff       = flux_lw_net[1:] - flux_lw_net[0:-1]
        pres_diff       = pres_lev[1:] - pres_lev[0:-1]
        dT_rad          = -(flux_diff / pres_diff.squeeze()) * 0.009761357302 # * g/cp = 9.80665 / 1004.64

        # Transpose back to (batch, lev)
        dT_rad            = torch.transpose(dT_rad,0,1)
        flux_lw_up        = torch.transpose(flux_lw_up,0,1)
        flux_lw_dn        = torch.transpose(flux_lw_dn,0,1)

        if self.return_gpt_fluxes:
            flux_lw_dn_gpt = torch.transpose(flux_lw_dn_gpt,0,1)
            flux_lw_up_gpt = torch.transpose(flux_lw_up_gpt,0,1)
            return dT_rad, flux_lw_up_gpt, flux_lw_dn_gpt
        else:
            return dT_rad, flux_lw_up, flux_lw_dn

# -------------------------------------------- RTE KERNELS --------------------------------------------

def interpolate_tlev_batchlast(tlay, play, plev):
    nlay, ncol = tlay.shape
    device = tlay.device
    dtype = tlay.dtype
    # Initialize output arrays
    tlev = torch.zeros(nlay + 1, ncol, dtype=dtype, device=device)
    
    tlev[0] = tlay[0] + (plev[0]-play[0])*(tlay[1]-tlay[0]) / (play[1]-play[0])
    for ilay in range(1, nlay):
      tlev[ilay] = (play[ilay-1]*tlay[ilay-1]*(plev[ilay]-play[ilay]) \
            + play[ilay]*tlay[ilay]*(play[ilay-1]-plev[ilay])) /  (plev[ilay]*(play[ilay-1] - play[ilay]))
                              
    tlev[nlay] = tlay[nlay-1] + (plev[nlay]-play[nlay-1])*(tlay[nlay-1]-tlay[nlay-2])  \
            / (play[nlay-1]-play[nlay-2])
                              
    return tlev

def outgoing_lw(temp):
    # Stefan-Boltzmann constant (W/m²/K⁴)
    # sigma = 5.670374419e-8
    
    # Assuming emissivity = 1 (blackbody approximation)
    olr_exact = 5.670374419e-8 * torch.pow(temp,4)
    return olr_exact

@torch.compile(dynamic=False)
def reftrans_lw(planck_top, planck_bot, od):
    """
    Calculate longwave transmittance and source terms using Padé approximant method.
    
    This function implements the alternative source computation using a Padé approximant
    for the linear-in-tau solution, following Clough et al. (1992), doi:10.1029/92JD01419, Eq 15.
    This method requires no conditional statements but introduces some approximation error.
    
    Args:
        planck_top (torch.Tensor): Planck function at layer top
        planck_bot (torch.Tensor): Planck function at layer bottom
        od (torch.Tensor): Optical depth
        LwDiffusivity (float): Longwave diffusivity factor (default 1.66)
    
    Returns:
        tuple: (transmittance, source_up, source_dn)
            - source_up (torch.Tensor): Upward emission at layer top 
            - source_dn (torch.Tensor): Downward emission at layer bottom 
            - transmittance (torch.Tensor): Diffuse transmittance
    """
    LwDiffusivity=1.66
    od = LwDiffusivity * od
    trans_lw = torch.exp(-od)
    # Calculate coefficient for Padé approximant (vectorized)
    coeff = 0.2 * od
    # Calculate mean Planck function (vectorized)
    planck_fl = 0.5 * (planck_top + planck_bot)
    # Calculate source terms using Padé approximant (vectorized)
    # one_minus_trans = 1.0 - trans_lw
    # one_plus_coeff = 1.0 + coeff
    source_dn = (1.0 - trans_lw) * (planck_fl + coeff * planck_bot) / (1.0 + coeff)
    source_up = (1.0 - trans_lw) * (planck_fl + coeff * planck_top) / (1.0 + coeff)
    return source_up, source_dn, trans_lw


@torch.compile(dynamic=False)
def lw_solver_noscat_batchlast(trans_lw, source_dn, source_up, source_sfc, emissivity_surf):
    
    nlev = trans_lw.shape[0]
    
    # At top-of-atmosphere there is no diffuse downwelling radiation
    flux_lw_dn0 = torch.zeros_like(emissivity_surf)
    flux_lw_dn = torch.jit.annotate(List[Tensor], [])
    flux_lw_dn += [flux_lw_dn0]

    # Work down through the atmosphere computing the downward fluxes
    # at each half-level (vectorized over columns)
    for jlev in range(nlev):
        # flux_lw_dn[jlev + 1] = (trans_lw[jlev] * flux_lw_dn[jlev].clone()  + 
        #                        source_dn[jlev])
        flux_lw_dn0 = (trans_lw[jlev] * flux_lw_dn0 + source_dn[jlev])
        flux_lw_dn += [flux_lw_dn0]

    # flux_lw_up[nlev] = source_sfc + albedo_surf * flux_lw_dn[nlev]
    #                                              albedo
    flux_lw_up0   = emissivity_surf*source_sfc +  (1-emissivity_surf) * flux_lw_dn[nlev]
    flux_lw_up    = torch.jit.annotate(List[Tensor], [])
    flux_lw_up    += [flux_lw_up0]

    flux_lw_dn = torch.stack(flux_lw_dn)

    # Work back up through the atmosphere computing the upward fluxes
    # at each half-level (vectorized over columns)
    for jlev in range(nlev - 1, -1, -1):
        flux_lw_up0 = (trans_lw[jlev] * flux_lw_up0  + source_up[jlev])    
        flux_lw_up += [flux_lw_up0]

    flux_lw_up.reverse()
    flux_lw_up  = torch.stack(flux_lw_up)
    return flux_lw_dn, flux_lw_up

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