! This code is part of RRTM for GCM Applications - Parallel (RRTMGP)
!
! Contacts: Robert Pincus and Eli Mlawer
! email:  rrtmgp@aer.com
!
! Copyright 2015-2018,  Atmospheric and Environmental Research and
! Regents of the University of Colorado.  All right reserved.
!
! Use and duplication is permitted under the terms of the
!    BSD 3-clause license, see http://opensource.org/licenses/BSD-3-Clause
! -------------------------------------------------------------------------------------------------
!
! Example program to demonstrate the calculation of shortwave radiative fluxes in clear, aerosol-free skies.
!   The example files come from the Radiative Forcing MIP (https://www.earthsystemcog.org/projects/rfmip/)
!   The large problem (1800 profiles) is divided into blocks
!
! Program is invoked as rrtmgp_rfmip_sw [block_size input_file  coefficient_file upflux_file downflux_file]
!   All arguments are optional but need to be specified in order.
!
! -------------------------------------------------------------------------------------------------
!
! Error checking: Procedures in rte+rrtmgp return strings which are empty if no errors occured
!   Check the incoming string, print it out and stop execution if non-empty
!
subroutine stop_on_err(error_msg)
  use iso_fortran_env, only : error_unit
  character(len=*), intent(in) :: error_msg

  if(error_msg /= "") then
    write (error_unit,*) trim(error_msg)
    write (error_unit,*) "rrtmgp_rfmip_sw stopping"
    stop
  end if
end subroutine stop_on_err
! -------------------------------------------------------------------------------------------------
!
! Main program
!
! -------------------------------------------------------------------------------------------------
program rrtmgp_rfmip_sw
  ! --------------------------------------------------
  !
  ! Modules for working with rte and rrtmgp
  !
#ifdef USE_OPENMP
  use omp_lib
#endif
  ! Working precision for real variables
  !
  use mo_rte_kind,           only: wp, sp
  !
  ! Array utilities
  !
  use mo_rte_util_array,     only: zero_array
  !
  ! Optical properties of the atmosphere as array of values
  !   In the longwave we include only absorption optical depth (_1scl)
  !   Shortwave calculations use optical depth, single-scattering albedo, asymmetry parameter (_2str)
  !
  ! use mo_optical_props,      only: ty_optical_props_arry, make_2str, make_1scl
  use mo_optical_props,      only: ty_optical_props_2str
  !
  ! Gas optics: maps physical state of the atmosphere to optical properties
  !
  use mo_gas_optics_rrtmgp,  only: ty_gas_optics_rrtmgp
  !
  ! Gas optics uses a derived type to represent gas concentrations compactly
  !
  use mo_gas_concentrations, only: ty_gas_concs
  !
  ! RTE shortwave driver
  !
  use mo_rte_sw,             only: rte_sw
  !
  ! RTE driver uses a derived type to reduce spectral fluxes to whatever the user wants
  !   Here we're just reporting broadband fluxes
  !
  use mo_fluxes,             only: ty_fluxes_broadband, ty_fluxes_flexible
  ! --------------------------------------------------
  !
  ! modules for reading and writing files
  !
  ! RRTMGP's gas optics class needs to be initialized with data read from a netCDF files
  !
  use mo_load_coefficients,  only: load_and_init
  use mo_rfmip_io,           only: read_size, read_and_block_pt, read_and_block_gases_ty, unblock_and_write, &
                                   unblock, read_and_block_sw_bc, determine_gas_names                        
  use mo_simple_netcdf,      only: read_field, write_field, get_dim_size
  use netcdf
  use mod_network_rrtmgp   
#ifdef USE_OPENACC  
  use cublas
  use openacc
#endif          
#ifdef USE_TIMING
  !
  ! Timing library
  !
  use gptl,                  only: gptlstart, gptlstop, gptlinitialize, gptlpr_file, gptlfinalize, gptlsetoption, &
                                   gptlpercent, gptloverhead
#endif

  implicit none

#ifdef USE_PAPI  
#include "f90papi.h"
#endif  
  ! --------------------------------------------------
  !
  ! Local variables
  !
  character(len=132)  ::  rfmip_file = 'multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc', &
                          kdist_file = 'coefficients_sw.nc'
  ! Neural networks for gas optics (optional) - netCDF files which describe the models and pre-processing coefficients
  ! (The two models predict SW absorption and Rayleigh scattering, respectively)
  character(len=80)   ::  modelfile_tau= "../../neural/data/BEST_tau-sw-abs-7-16-16-mae_2.nc", &
                          modelfile_ray= "../../neural/data/BEST_tau-sw-ray-7-16-16_2.nc"
  character(len=132)  ::  flx_file, flx_file_ref, flx_file_lbl, timing_file
  integer             ::  nargs, ncol, nlay, nbnd, ngpt, nexp, nblocks, block_size, forcing_index
  logical 	      ::  top_at_1
  integer             ::  b, icol, ilay,ibnd, igpt, igas, ncid, ninputs, count_rate, iTime1, iTime2, iTime3, ret, i, istat
  character(len=4)    ::  block_size_char, forcing_index_char = '1'
  character(len=32 ), dimension(:),  allocatable   ::  kdist_gas_names, rfmip_gas_names
  real(wp), dimension(:,:,:),         allocatable :: p_lay, p_lev, t_lay, t_lev ! block_size, nlay, nblocks
  real(wp), dimension(:,:,:), target, allocatable :: flux_up, flux_dn, flux_dn_dir
  real(wp), dimension(:,:,:,:), target, allocatable :: gpt_flux_up, gpt_flux_dn, gpt_flux_dn_dir
  real(wp), dimension(:,:,:),         allocatable :: rsu_ref, rsd_ref, rsu_nn, rsd_nn, rsu_lbl, rsd_lbl, rsdu_ref, rsdu_nn, rsdu_lbl
  real(wp), dimension(:,:  ),         allocatable :: surface_albedo, total_solar_irradiance, solar_zenith_angle
                                                     ! block_size, nblocks
  real(wp), dimension(:,:  ),         allocatable :: sfc_alb_spec ! nbnd, block_size; spectrally-resolved surface albedo
  real(wp), dimension(:),             allocatable :: temparray
  real(wp), parameter :: deg_to_rad = acos(-1._wp)/180._wp
  real(wp) :: def_tsi_s

  logical 		                        :: use_rrtmgp_nn, do_gpt_flux, compare_flux, save_flux
  !
  ! Classes used by rte+rrtmgp
  !
  type(ty_gas_optics_rrtmgp)                    :: k_dist
  type(ty_optical_props_2str)                   :: optical_props
  type(ty_fluxes_flexible)                      :: fluxes
  type(rrtmgp_network_type), dimension(2)        :: neural_nets ! First model is absorption, second is Rayleigh

  real(wp), dimension(:,:), allocatable         :: toa_flux ! block_size, ngpt
  real(wp), dimension(:  ), allocatable         :: def_tsi, mu0    ! block_size
  logical , dimension(:,:), allocatable         :: usecol ! block_size, nblocks
  !
  ! ty_gas_concentration holds multiple columns; we make an array of these objects to
  !   leverage what we know about the input file
  !
  type(ty_gas_concs), dimension(:), allocatable  :: gas_conc_array

  ! Initialize GPU kernel
#ifdef USE_OPENACC  
  type(cublasHandle) :: h
  istat = cublasCreate(h) 
  ! istat = cublasSetStream(h, acc_get_cuda_stream(acc_async_sync))
#endif

  ! -------------------------------------------------------------------------------------------------
  !
  ! Code starts
  !   all arguments are optional
  !
  !  ------------ I/O and settings -----------------
  ! Use neural networks for gas optics?  if NN models provided, set to true, but can also be overriden
  use_rrtmgp_nn      = .false.
  ! Save fluxes
  save_flux    = .true.
  ! compare fluxes to reference code as well as line-by-line (RFMIP only)
  compare_flux = .false.
  ! Compute fluxes per g-point?
  do_gpt_flux = .true.

  print *, "Usage: rrtmgp_rfmip_sw [block_size] [rfmip_file] [k-distribution_file] [forcing_index (1,2,3)]"
  print *, "OR:  rrtmgp_rfmip_sw [block_size] [rfmip_file] [k-distribution_file] [forcing_index] [NN_sw_abs_file] [NN_sw_ray_file]"

  nargs = command_argument_count()

  call get_command_argument(1, block_size_char)
  read(block_size_char, '(i4)') block_size
  if(nargs >= 2) call get_command_argument(2, rfmip_file)
  if(nargs >= 3) call get_command_argument(3, kdist_file)
  if(nargs >= 4) call get_command_argument(4, forcing_index_char)

  if(nargs == 5) stop "provide 1-4 or 6 arguments"
  if(nargs >= 6) then
    use_rrtmgp_nn = .true.
    call get_command_argument(5, modelfile_tau)
    call get_command_argument(6, modelfile_ray)
  end if
  ! How big is the problem? Does it fit into blocks of the size we've specified?
  !
  call read_size(rfmip_file, ncol, nlay, nexp)

  print *, "input file:", rfmip_file
  print *, "nexp:", nexp, "ncol:", ncol, "nlay:", nlay
  if (nexp==18) compare_flux=.true.

  if(mod(ncol*nexp, block_size) /= 0 ) call stop_on_err("rrtmgp_rfmip_lw: number of columns doesn't fit evenly into blocks.")
  nblocks = (ncol*nexp)/block_size
  print *, "Doing ",  nblocks, "blocks of size ", block_size

  read(forcing_index_char, '(i4)') forcing_index
  if(forcing_index < 1 .or. forcing_index > 4) &
    stop "Forcing index is invalid (must be 1,2 or 3)"

  ! Save upwelling and downwelling fluxes in the same file
  if (use_rrtmgp_nn) then
    flx_file = 'output_fluxes/rsud_Efx_RTE-RRTMGP-NN-181204_rad-irf_r1i1p1f' // trim(forcing_index_char) // '_gn.nc'
  else
    flx_file = 'output_fluxes/rsud_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f' // trim(forcing_index_char) // '_gn.nc'
  end if
  !
  ! Identify the set of gases used in the calculation based on the forcing index
  !   A gas might have a different name in the k-distribution than in the files
  !   provided by RFMIP (e.g. 'co2' and 'carbon_dioxide')
  !
  call determine_gas_names(rfmip_file, kdist_file, forcing_index, kdist_gas_names, rfmip_gas_names)
  ! print *, "Calculation uses RFMIP gases: ", (trim(rfmip_gas_names(b)) // " ", b = 1, size(rfmip_gas_names))
  print *, "Calculation uses RFMIP gases: ", (trim(kdist_gas_names(b)) // " ", b = 1, size(kdist_gas_names))

  ! Load Neural Network models
  if (use_rrtmgp_nn) then
	  print *, 'loading shortwave absorption model from ', modelfile_tau
    call neural_nets(1) % load_netcdf(modelfile_tau)
    print *, 'loading rayleigh model from ', modelfile_ray
    call neural_nets(2) % load_netcdf(modelfile_ray)
    ninputs = size(neural_nets(1) % layers(1) % w_transposed, 2)
    print *, "NN supports gases: ", &
    (trim(neural_nets(1)%input_names(b)) // " ", b = 3, size(neural_nets(1)%input_names))
  end if  
  ! --------------------------------------------------
  !
  ! Prepare data for use in rte+rrtmgp
  !
  !
  ! Allocation on assignment within reading routines
  !
  call read_and_block_pt(rfmip_file, block_size, p_lay, p_lev, t_lay, t_lev)
  !
  ! Are the arrays ordered in the vertical with 1 at the top or the bottom of the domain?
  !
  top_at_1 = p_lay(1, 1, 1) < p_lay(nlay, 1, 1)

  !
  ! Read the gas concentrations and surface properties
  !
  call read_and_block_gases_ty(rfmip_file, block_size, kdist_gas_names, rfmip_gas_names, gas_conc_array)
  ! do b = 1, size(gas_conc_array(1)%concs)
  !   print *, "max of gas ", gas_conc_array(1)%gas_name(b), ":", maxval(gas_conc_array(1)%concs(b)%conc)
  ! end do

  call read_and_block_sw_bc(rfmip_file, block_size, surface_albedo, total_solar_irradiance, solar_zenith_angle)
  !
  ! Read k-distribution information. load_and_init() reads data from netCDF and calls
  !   k_dist%init(); users might want to use their own reading methods
  !
  call load_and_init(k_dist, trim(kdist_file), gas_conc_array(1))
  
  if(.not. k_dist%source_is_external()) &
    stop "rrtmgp_rfmip_sw: k-distribution file isn't SW"
  nbnd = k_dist%get_nband()
  ngpt = k_dist%get_ngpt()

  allocate(toa_flux(k_dist%get_ngpt(), block_size), &
           def_tsi(block_size), usecol(block_size,nblocks))
  !
  ! RRTMGP won't run with pressure less than its minimum. The top level in the RFMIP file
  !   is set to 10^-3 Pa. Here we pretend the layer is just a bit less deep.
  !   This introduces an error but shows input sanitizing.
  !
  if(top_at_1) then
    p_lev(1,:,:) = k_dist%get_press_min() + epsilon(k_dist%get_press_min())
  else
    p_lev(nlay+1,:,:) &
                 = k_dist%get_press_min() + epsilon(k_dist%get_press_min())
  end if

  !
  ! RTE will fail if passed solar zenith angles greater than 90 degree. We replace any with
  !   nighttime columns with a default solar zenith angle. We'll mask these out later, of
  !   course, but this gives us more work and so a better measure of timing.
  !
  do b = 1, nblocks
    usecol(1:block_size,b)  = solar_zenith_angle(1:block_size,b) < 90._wp - 2._wp * spacing(90._wp)
  end do

  !
  ! Allocate space for output fluxes (accessed via pointers in ty_fluxes_broadband),
  !   gas optical properties, and source functions. The %alloc() routines carry along
  !   the spectral discretization from the k-distribution.
  !
  allocate(flux_up(    	nlay+1, block_size, nblocks), &
           flux_dn(    	nlay+1, block_size, nblocks))
  allocate(flux_dn_dir(    	nlay+1, block_size, nblocks))

  ! Allocate g-point fluxes if desired
  if (do_gpt_flux) then
    allocate(gpt_flux_up(ngpt, nlay+1, block_size, nblocks), &
    gpt_flux_dn(ngpt, nlay+1, block_size, nblocks))
    allocate(gpt_flux_dn_dir(ngpt, nlay+1, block_size, nblocks))
  end if

  ! allocate(mu0(block_size), sfc_alb_spec(nbnd,block_size))
  allocate(mu0(block_size), sfc_alb_spec(ngpt,block_size))

  ! Alloc is generic and calls appropriate allocation routine for the specific type of optical_props
  ! call stop_on_err(optical_props%alloc(block_size, nlay, k_dist))
  call stop_on_err(optical_props%alloc_2str(block_size, nlay, k_dist))
  
  !$acc enter data create (toa_flux, def_tsi)
  !$acc enter data create (sfc_alb_spec, mu0) 
  !$acc enter data copyin(total_solar_irradiance, surface_albedo, usecol, solar_zenith_angle)

  !custom TSI
  call stop_on_err(k_dist%set_tsi(1361.0_wp))


#ifdef USE_TIMING
  !
  ! Initialize timers
  !
  ret = gptlsetoption (gptlpercent, 1)        ! Turn on "% of" print
  ret = gptlsetoption (gptloverhead, 0)       ! Turn off overhead estimate
#ifdef USE_PAPI  
#ifdef DOUBLE_PRECISION
  ret = GPTLsetoption (PAPI_DP_OPS, 1);         ! Turn on FLOPS estimate (DP)
#else
  ret = GPTLsetoption (PAPI_SP_OPS, 1);         ! Turn on FLOPS estimate (SP)
#endif
#endif  
  ret = gptlinitialize()
#endif

  ! --------------------------------------------------

  if (use_rrtmgp_nn) then
    print *, "starting clear-sky SW computations, using neural networks as RRTMGP kernel"
  else
    print *, "starting clear-sky SW computations, using lookup-table as RRTMGP kernel"
  end if
  call system_clock(count_rate=count_rate)
  call system_clock(iTime1)
  !
  ! Loop over blocks
  !
#ifdef USE_TIMING
  ret =  gptlstart('clear_sky_total (SW)')
! do i = 1, 32
#endif
#ifdef USE_OPENMP
  !$OMP PARALLEL firstprivate(def_tsi,toa_flux,sfc_alb_spec,mu0,fluxes,optical_props) default(shared)
  !$OMP DO 
#endif
  do b = 1, nblocks

    fluxes%flux_up => flux_up(:,:,b)
    fluxes%flux_dn => flux_dn(:,:,b)
    fluxes%flux_dn_dir => flux_dn_dir(:,:,b)
    if (do_gpt_flux) then
      fluxes%gpt_flux_up => gpt_flux_up(:,:,:,b)
      fluxes%gpt_flux_dn => gpt_flux_dn(:,:,:,b)
      fluxes%gpt_flux_dn_dir => gpt_flux_dn_dir(:,:,:,b)
    end if
    !
    ! Compute the optical properties of the atmosphere and the Planck source functions
    !    from pressures, temperatures, and gas concentrations...
    !
#ifdef USE_TIMING
    ret =  gptlstart('gas_optics (SW)')
#endif
    
    if (use_rrtmgp_nn) then
      call stop_on_err(k_dist%gas_optics(p_lay(:,:,b), &
                                        p_lev(:,:,b),       &
                                        t_lay(:,:,b),       &
                                        gas_conc_array(b),  &
                                        optical_props,      &
                                        toa_flux, neural_nets=neural_nets))
    else
      call stop_on_err(k_dist%gas_optics(p_lay(:,:,b), &
                                        p_lev(:,:,b),       &
                                        t_lay(:,:,b),       &
                                        gas_conc_array(b),  &
                                        optical_props,      &
                                        toa_flux))
    end if
! #ifdef USE_OPENMP
!     PRINT *, "process: ", OMP_GET_THREAD_NUM(), "b: ",b," mean tau ", mean_3d(optical_props%tau)
! #endif
    ! !$acc update host(optical_props%tau, optical_props%ssa, optical_props%g)
    ! print *," max, min (tau)",   maxval(optical_props%tau), minval(optical_props%tau)
    ! print *," max, min (ssa)",   maxval(optical_props%ssa), minval(optical_props%ssa)
    ! print *," max, min (g)",   maxval(optical_props%g), minval(optical_props%g)
    ! print *, "sum toa flux", sum(toa_flux(:,1))

    if (nblocks==1) call system_clock(iTime2)

#ifdef USE_TIMING
    ret =  gptlstop('gas_optics (SW)')
#endif
    !
    ! Boundary conditions
    !
    ! What's the total solar irradiance assumed by RRTMGP?
    ! 
    !$acc parallel loop gang default(present)
    do icol = 1, block_size
      def_tsi_s = 0.0_wp
      !$acc loop vector reduction(+:def_tsi_s)
      do igpt = 1, ngpt
        def_tsi_s = def_tsi_s + toa_flux(igpt, icol)
      end do
      def_tsi(icol) = def_tsi_s
    end do

    !$acc parallel default(present)    
    !$acc loop collapse(2)
    do icol = 1, block_size
      do igpt = 1, ngpt
        ! Normalize incoming solar flux to match RFMIP specification
        toa_flux(igpt,icol) = toa_flux(igpt,icol) * total_solar_irradiance(icol,b)/def_tsi(icol)
        ! Expand the spectrally-constant surface albedo to a per-g-point albedo for each column
        sfc_alb_spec(igpt,icol) = surface_albedo(icol,b)
      end do
    end do
    !
    ! Cosine of the solar zenith angle
    !
    !$acc loop
    do icol = 1, block_size
      mu0(icol) = merge(cos(solar_zenith_angle(icol,b)*deg_to_rad), 1._wp, usecol(icol,b))
    end do
    !$acc end parallel

    !
    ! ... and compute the spectrally-resolved fluxes, providing reduced values
    !    via ty_fluxes_broadband
    !
#ifdef USE_TIMING
    ret =  gptlstart('rte_sw')
#endif

    call stop_on_err(rte_sw(optical_props,   &
                            top_at_1,        &
                            mu0,             &
                            toa_flux,        &
                            sfc_alb_spec,    &
                            sfc_alb_spec,    &
                            fluxes))!,          &
#ifdef USE_TIMING
    ret =  gptlstop('rte_sw')
#endif
    !
    ! Zero out fluxes for which the original solar zenith angle is > 90 degrees.
    !
    do icol = 1, block_size
      if(.not. usecol(icol,b)) then
        flux_up(:,icol,b)  = 0._wp
        flux_dn(:,icol,b)  = 0._wp
      end if
    end do
    
  end do !blocks
#ifdef USE_OPENMP
  !$OMP END DO
  !$OMP END PARALLEL
#endif
  !
  ! End timers
  !
#ifdef USE_TIMING
!  end do
  ret =  gptlstop('clear_sky_total (SW)')
  timing_file = "timing.sw-" // adjustl(trim(block_size_char))
  ret = gptlpr_file(trim(timing_file))
  ret = gptlfinalize()
#endif

  !$acc exit data delete(total_solar_irradiance, surface_albedo, usecol, solar_zenith_angle)
  !$acc exit data delete(sfc_alb_spec, mu0)
  !$acc exit data delete(toa_flux, def_tsi)
  call optical_props%finalize() ! Also deallocates arrays on device

#ifdef USE_OPENACC  
  istat = cublasDestroy(h) 
#endif

  if (nblocks==1) then
    call system_clock(iTime3)
    print *, "-----------------------------------------------------------------------------------------"
    print '(a,f11.4,/,a,f11.4,/,a,f11.4,a)', ' Time elapsed in gas optics:',real(iTime2-iTime1)/real(count_rate), &
    ' Time elapsed in solver:    ', real(iTime3-iTime2)/real(count_rate), ' Time elapsed in total:     ', &
    real(iTime3-iTime1)/real(count_rate)
    print *, "-----------------------------------------------------------------------------------------"
  else 
    call system_clock(iTime3)
    print *,'Elapsed time on everything ',real(iTime3-iTime1)/real(count_rate)
  end if

  print *, "mean of flux_down is:", mean_3d(flux_dn)  !  mean of flux_down is:   103.2458
  print *, "mean of flux_up is:", mean_3d(flux_up)
  ! mean of flux_down is:   292.71945410963957     
  ! mean of flux_up is:   41.835381782065106 

  ! --------------------------------------------------

  ! Save fluxes ?
  if (save_flux) then
    print *, "Attempting to save fluxes to ", flx_file
    call unblock_and_write(trim(flx_file), 'rsu', flux_up)
    call unblock_and_write(trim(flx_file), 'rsd', flux_dn)
    print *, "Fluxes saved to ", flx_file
  end if 

  ! Compare fluxes to benchmark line-by-line results, alongside reference RTE+RRTMGP computations?
  if (compare_flux) then
    print *, "-----------------------------------------------------------------------------------------------------"
    print *, "-----COMPARING ERRORS (W.R.T. LINE-BY-LINE) OF NEW RESULTS AND RRTMGP-224  -------"
    print *, "-----------------------------------------------------------------------------------------------------"

    allocate(rsd_ref( nlay+1, ncol, nexp))
    allocate(rsu_ref( nlay+1, ncol, nexp))  
    allocate(rsdu_ref( nlay+1, ncol, nexp))  
    allocate(rsd_nn( nlay+1, ncol, nexp))
    allocate(rsu_nn( nlay+1, ncol, nexp))
    allocate(rsdu_nn( nlay+1, ncol, nexp))
    allocate(rsd_lbl( nlay+1, ncol, nexp))
    allocate(rsu_lbl( nlay+1, ncol, nexp))
    allocate(rsdu_lbl( nlay+1, ncol, nexp))

    flx_file_ref = 'output_fluxes/rsud_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn_REF-DP.nc'
    flx_file_lbl = 'output_fluxes/rsud_Efx_LBLRTM-12-8_rad-irf_r1i1p1f1_gn.nc'

    call unblock(flux_up, rsu_nn)
    call unblock(flux_dn, rsd_nn)

    rsdu_nn = rsd_nn - rsu_nn

    if(nf90_open(trim(flx_file_ref), NF90_NOWRITE, ncid) /= NF90_NOERR) &
      call stop_on_err("read_and_block_gases_ty: can't find file " // trim(flx_file_ref))

    rsu_ref = read_field(ncid, "rsu", nlay+1, ncol, nexp)
    rsd_ref = read_field(ncid, "rsd", nlay+1, ncol, nexp)
    rsdu_ref = rsd_ref - rsu_ref

    if(nf90_open(trim(flx_file_lbl), NF90_NOWRITE, ncid) /= NF90_NOERR) &
    call stop_on_err("read_and_block_gases_ty: can't find file " // trim(flx_file_lbl))

    rsu_lbl = read_field(ncid, "rsu", nlay+1, ncol, nexp)
    rsd_lbl = read_field(ncid, "rsd", nlay+1, ncol, nexp)
    rsdu_lbl = rsd_lbl - rsu_lbl

    print *, "------------- UPWELLING -------------- "

    print *, "MAE in upwelling fluxes of new result and RRTMGP-224, present-day:            ", &
     mae(reshape(rsu_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rsu_nn(:,:,1), shape = [1*ncol*(nlay+1)])),&
     mae(reshape(rsu_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rsu_ref(:,:,1), shape = [1*ncol*(nlay+1)]))

    ! print *, "MAE in upwelling fluxes of new result and RRTMGP-224, future:                 ", &
    !  mae(reshape(rsu_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rsu_nn(:,:,4), shape = [1*ncol*(nlay+1)])),&
    !  mae(reshape(rsu_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rsu_ref(:,:,4), shape = [1*ncol*(nlay+1)]))

    print *, "bias in upwelling flux of new result and RRTMGP-224, present-day, top-of-atm.:", &
      bias(reshape(rsu_lbl(1,:,1), shape = [1*ncol]),    reshape(rsu_nn(1,:,1), shape = [1*ncol])), &
      bias(reshape(rsu_lbl(1,:,1), shape = [1*ncol]),    reshape(rsu_ref(1,:,1), shape = [1*ncol])) 

    print *, "bias in upwelling flux of new result and RRTMGP-224, future, top-of-atm.:     ", &
      bias(reshape(rsu_lbl(1,:,4), shape = [1*ncol]),    reshape(rsu_nn(1,:,4), shape = [1*ncol])), &
      bias(reshape(rsu_lbl(1,:,4), shape = [1*ncol]),    reshape(rsu_ref(1,:,4), shape = [1*ncol])) 

    ! print *, "bias in upwelling flux of new result and RRTMGP-224, future-all, top-of-atm.: ", &
    !   bias(reshape(rsu_lbl(1,:,17), shape = [1*ncol]),    reshape(rsu_nn(1,:,17), shape = [1*ncol])), &
    !   bias(reshape(rsu_lbl(1,:,17), shape = [1*ncol]),    reshape(rsu_ref(1,:,17), shape = [1*ncol])) 

    print *, "bias in upwelling flux of new result and RRTMGP-224, ALL EXPS, top-of-atm.:   ", &
      bias(reshape(rsu_lbl(1,:,:), shape = [nexp*ncol]),    reshape(rsu_nn(1,:,:), shape = [nexp*ncol])), &
      bias(reshape(rsu_lbl(1,:,:), shape = [nexp*ncol]),    reshape(rsu_ref(1,:,:), shape = [nexp*ncol])) 


    print *, "-------------- DOWNWELLING --------------"

    print *, "MAE in downwelling fluxes of new result and RRTMGP-224, present-day:          ", &
     mae(reshape(rsd_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rsd_nn(:,:,1), shape = [1*ncol*(nlay+1)])),&
     mae(reshape(rsd_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rsd_ref(:,:,1), shape = [1*ncol*(nlay+1)]))

    print *, "MAE in downwelling fluxes of new result and RRTMGP-224, future:               ", &
     mae(reshape(rsd_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rsd_nn(:,:,4), shape = [1*ncol*(nlay+1)])),&
    mae(reshape(rsd_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rsd_ref(:,:,4), shape = [1*ncol*(nlay+1)]))

    print *, "-------------- NET FLUX --------------"

     print *, "Max-vertical-error in net fluxes of new result and RRTMGP-224, pres.day:  ", &
     maxval(abs(rsdu_lbl(:,:,1)-rsdu_nn(:,:,1))), maxval(abs(rsdu_lbl(:,:,1)-rsdu_ref(:,:,1)))

    !  print *, "Max-vertical-error in net fluxes of new result and RRTMGP-224, future:    ", &
    !  maxval(abs(rsdu_lbl(:,:,4)-rsdu_nn(:,:,4))), maxval(abs(rsdu_lbl(:,:,4)-rsdu_ref(:,:,4)))

     print *, "Max-vertical-error in net fluxes of new result and RRTMGP-224, future-all:", &
     maxval(abs(rsdu_lbl(:,:,17)-rsdu_nn(:,:,17))), maxval(abs(rsdu_lbl(:,:,17)-rsdu_ref(:,:,17)))

     print *, "---------"

     print *, "MAE in net fluxes of new result and RRTMGP-224, present-day:               ", &
     mae(reshape(rsdu_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rsdu_nn(:,:,1), shape = [1*ncol*(nlay+1)])), &
     mae(reshape(rsdu_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rsdu_ref(:,:,1), shape = [1*ncol*(nlay+1)])) 

    ! print *, "MAE in net fluxes of new result and RRTMGP-224, future:                    ", &
    !  mae(reshape(rsdu_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rsdu_nn(:,:,4), shape = [1*ncol*(nlay+1)])), &
    !  mae(reshape(rsdu_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rsdu_ref(:,:,4), shape = [1*ncol*(nlay+1)]))

    print *, "MAE in net fluxes of new result and RRTMGP-224, future-all:                ", &
     mae(reshape(rsdu_lbl(:,:,17), shape = [1*ncol*(nlay+1)]), reshape(rsdu_nn(:,:,17), shape = [1*ncol*(nlay+1)])),&
     mae(reshape(rsdu_lbl(:,:,17), shape = [1*ncol*(nlay+1)]), reshape(rsdu_ref(:,:,17), shape = [1*ncol*(nlay+1)]))

     print *, "MAE in net fluxes of new result and RRTMGP-224, ALL EXPS:                  ", &
     mae(reshape(rsdu_lbl(:,:,:), shape = [nexp*ncol*(nlay+1)]),    reshape(rsdu_nn(:,:,:), shape = [nexp*ncol*(nlay+1)])), &
     mae(reshape(rsdu_lbl(:,:,:), shape = [nexp*ncol*(nlay+1)]),    reshape(rsdu_ref(:,:,:), shape = [nexp*ncol*(nlay+1)])) 

    ! print *, "---------"

    ! print *, "MAE in net fluxes at TOA of new result and RRTMGP-224, future:             ", &
    !  mae(reshape(rsdu_lbl(1,:,4), shape = [1*ncol*(1)]), reshape(rsdu_nn(1,:,4), shape = [1*ncol*(1)])), &
    !  mae(reshape(rsdu_lbl(1,:,4), shape = [1*ncol*(1)]), reshape(rsdu_ref(1,:,4), shape = [1*ncol*(1)]))

    !  print *, "MAE in net fluxes at TOA of new result and RRTMGP-224, present-day:        ", &
    !  mae(reshape(rsdu_lbl(1,:,1), shape = [1*ncol*(1)]), reshape(rsdu_nn(1,:,1), shape = [1*ncol*(1)])), &
    !  mae(reshape(rsdu_lbl(1,:,1), shape = [1*ncol*(1)]), reshape(rsdu_ref(1,:,1), shape = [1*ncol*(1)]))

     print *, "---------"

    ! print *, "RMSE in net fluxes of new result and RRTMGP-224, present-day, PBL:         ", &
    !  rmse(reshape(rsdu_lbl(33:nlay+1,:,1), shape = [1*ncol*(nlay+1-33)]),    reshape(rsdu_nn(33:nlay+1,:,1), shape = [1*ncol*(nlay+1-33)])), &
    !  rmse(reshape(rsdu_lbl(33:nlay+1,:,1), shape = [1*ncol*(nlay+1-33)]),    reshape(rsdu_ref(33:nlay+1,:,1), shape = [1*ncol*(nlay+1-33)]))

    print *, "RMSE in net fluxes of new result and RRTMGP-224, present-day, SURFACE:     ", &
     rmse(reshape(rsdu_lbl(nlay+1,:,1), shape = [1*ncol]),    reshape(rsdu_nn(nlay+1,:,1), shape = [1*ncol])), &
     rmse(reshape(rsdu_lbl(nlay+1,:,1), shape = [1*ncol]),    reshape(rsdu_ref(nlay+1,:,1), shape = [1*ncol]))

    print *, "RMSE in net fluxes of new result and RRTMGP-224, future-all, SURFACE:     ", &
     rmse(reshape(rsdu_lbl(nlay+1,:,17), shape = [1*ncol]),    reshape(rsdu_nn(nlay+1,:,17), shape = [1*ncol])), &
     rmse(reshape(rsdu_lbl(nlay+1,:,17), shape = [1*ncol]),    reshape(rsdu_ref(nlay+1,:,17), shape = [1*ncol]))

    print *, "RMSE in net fluxes of new result and RRTMGP-224, pre-industrial, SURFACE: ", &
     rmse(reshape(rsdu_lbl(nlay+1,:,2), shape = [1*ncol]),    reshape(rsdu_nn(nlay+1,:,2), shape = [1*ncol])), &
     rmse(reshape(rsdu_lbl(nlay+1,:,2), shape = [1*ncol]),    reshape(rsdu_ref(nlay+1,:,2), shape = [1*ncol]))

    print *, "---------"

    ! print *, "bias in net fluxes of new result and RRTMGP-224, present-day:              ", &
    !  bias(reshape(rsdu_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rsdu_nn(:,:,1), shape = [1*ncol*(nlay+1)])), &
    !  bias(reshape(rsdu_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rsdu_ref(:,:,1), shape = [1*ncol*(nlay+1)])) 

    print *, "bias in net fluxes of new result and RRTMGP-224, present-day, SURFACE:     ", &
     bias(reshape(rsdu_lbl(nlay+1,:,1), shape = [1*ncol]),    reshape(rsdu_nn(nlay+1,:,1), shape = [1*ncol])), &
     bias(reshape(rsdu_lbl(nlay+1,:,1), shape = [1*ncol]),    reshape(rsdu_ref(nlay+1,:,1), shape = [1*ncol])) 

    ! print *, "bias in net fluxes of new result and RRTMGP-224, future:                   ", &
    !  bias(reshape(rsdu_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rsdu_nn(:,:,4), shape = [1*ncol*(nlay+1)])), &
    !  bias(reshape(rsdu_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rsdu_ref(:,:,4), shape = [1*ncol*(nlay+1)]))

    ! print *, "bias in net fluxes of new result and RRTMGP-224, future-all:               ", &
    !  bias(reshape(rsdu_lbl(:,:,17), shape = [1*ncol*(nlay+1)]), reshape(rsdu_nn(:,:,17), shape = [1*ncol*(nlay+1)])), &
    !  bias(reshape(rsdu_lbl(:,:,17), shape = [1*ncol*(nlay+1)]), reshape(rsdu_ref(:,:,17), shape = [1*ncol*(nlay+1)]))

    print *, "bias in net fluxes of new result and RRTMGP-224, future-all, SURFACE:      ", &
     bias(reshape(rsdu_lbl(nlay+1,:,17), shape = [1*ncol]),    reshape(rsdu_nn(nlay+1,:,17), shape = [1*ncol])), &
     bias(reshape(rsdu_lbl(nlay+1,:,17), shape = [1*ncol]),    reshape(rsdu_ref(nlay+1,:,17), shape = [1*ncol])) 

    print *, "---------"

    ! print *, "MAE in upwelling fluxes of new result w.r.t RRTMGP-224, present-day:       ", &
    !  mae(reshape(rsu_ref(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rsu_nn(:,:,1), shape = [1*ncol*(nlay+1)]))

    ! print *, "MAE in downwelling fluxes of new result w.r.t RRTMGP-224, present-day:     ", &
    !  mae(reshape(rsd_ref(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rsd_nn(:,:,1), shape = [1*ncol*(nlay+1)]))

    print *, "MAE in net flux w.r.t. RRTMGP-224      ", &
    mae(reshape(rsdu_ref(:,:,:), shape = [nexp*ncol*(nlay+1)]),    reshape(rsdu_nn(:,:,:), shape = [nexp*ncol*(nlay+1)]))

    print *, "Max-diff in d.w. flux w.r.t. RRTMGP-224", &
     maxval(abs(rsd_ref(:,:,:)-rsd_nn(:,:,:)))
 
    print *, "Max-diff in u.w. flux w.r.t. RRTMGP-224", &
     maxval(abs(rsu_ref(:,:,:)-rsu_nn(:,:,:)))

    print *, "Max-diff in net flux w.r.t. RRTMGP-224 ", &
     maxval(abs(rsdu_ref(:,:,:)-rsdu_nn(:,:,:))) 

    deallocate(rsd_ref,rsu_ref,rsd_nn,rsu_nn,rsd_lbl,rsu_lbl,rsdu_ref,rsdu_nn,rsdu_lbl)

  end if

  deallocate(flux_up, flux_dn)
  print *, "SUCCESS!"

  contains

  function rmse(x1,x2) result(res)
    implicit none 
    real(wp), dimension(:), intent(in) :: x1,x2
    real(wp) :: res
    real(wp), dimension(size(x1)) :: diff 
    
    diff = x1 - x2
    res = sqrt( sum(diff**2)/size(diff) )
  end function rmse

  function mae(x1,x2) result(res)
    implicit none 
    real(wp), dimension(:), intent(in) :: x1,x2
    real(wp) :: res
    real(wp), dimension(size(x1)) :: diff 
    
    diff = abs(x1 - x2)
    res = sum(diff, dim=1)/size(diff, dim=1)
  end function mae

  function bias(x1,x2) result(res)
    implicit none 
    real(wp), dimension(:), intent(in) :: x1,x2
    real(wp) :: mean1,mean2, res
    
    mean1 = sum(x1, dim=1)/size(x1, dim=1)
    mean2 = sum(x2, dim=1)/size(x2, dim=1)
    res = mean1 - mean2

  end function bias

  function mean(x) result(mean1)
    implicit none 
    real(wp), dimension(:), intent(in) :: x
    real(wp) :: mean1
    mean1 = sum(x) / size(x)
  end function mean

  function mean_2d(x) result(mean2)
    implicit none 
    real(wp), dimension(:,:), intent(in) :: x
    real(wp) :: mean2
    mean2 = sum(x) / size(x)
  end function mean_2d

  function mean_3d(x) result(mean3)
    implicit none 
    real(wp), dimension(:,:,:), intent(in) :: x
    real(wp) :: mean3
    mean3 = sum(x) / size(x)
  end function mean_3d

end program rrtmgp_rfmip_sw
