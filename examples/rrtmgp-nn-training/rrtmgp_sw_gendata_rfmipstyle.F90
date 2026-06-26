! This program is for generating training data for neural network emulators of RRTMGP-SW

! Data is loaded from a RFMIP-style netCDF files containing atmospheric profiles, RTE+RRTMGP is run,
! and finally RRTMGP input and output is saved to a netCDF file which can be used for ML training
!
! The RRTMGP inputs, used as NN inputs, are layer-wise RRTMGP input variables (T, p, gas concentrations)
! The fina lRRTMGP outputs are optical depth tau and single-scattering albedo ssa. These could be used as NN outputs, 
! but in order to support the methodology in Ukkonen (2020), where two separate NNs predict absorption or
! Rayleigh scattering cross-sections, for convenience the layer number of dry air molecules (N) are also saved
! 
! y_abs      = (tau - tau_ray) / N = (tau - tau*ssa) / N
! y_rayleigh = (tau*ssa) / N 
! Ukkonen et al. (2020): https://doi.org/10.1029/2020MS002226
!
! Fortran program arguments:
! rrtmgp_sw_gendata_rfmipstyle [block_size] [input file] [k-distribution file] [file to save NN inputs/outputs]"
!
! Developed by Peter Ukkonen (built on RFMIP example, and existing RTE+RRTMGP code by Robert Pincus)
!
! -------------------------------------------------------------------------------------------------
!
! Error checking: Procedures in rte+rrtmgp return strings which are empty if no errors occured
!   Check the incoming string, print it out and stop execution if non-empty
!
subroutine stop_on_err(error_msg)
  use iso_fortran_env, only : error_unit
  use iso_c_binding
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
  ! Optical properties of the atmosphere as array of values
  !   Shortwave calculations use optical depth, single-scattering albedo, asymmetry parameter (_2str)
  !
  use mo_optical_props,      only: ty_optical_props_2str 
  !
  ! Gas optics: maps physical state of the atmosphere to optical properties
  !
  use mo_gas_optics_rrtmgp,  only: ty_gas_optics_rrtmgp, compute_nn_inputs, get_col_dry
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
  !   Here we're using a flexible type which saves broadband fluxes and optionally g-point fluxes
  !
  use mo_fluxes,             only: ty_fluxes_flexible
  ! --------------------------------------------------
  !
  ! modules for reading and writing files
  !
  ! RRTMGP's gas optics class needs to be initialized with data read from a netCDF files
  !
  use mo_load_coefficients,  only: load_and_init
  use mo_io_rfmipstyle_generic, only: read_size, read_and_block_pt, read_and_block_gases_ty, unblock_and_write, &
                                   read_and_block_sw_bc, determine_gas_names                             
  use netcdf
  use easy_netcdf    
  use mod_network_rrtmgp   

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
  character(len=132)  ::  input_file = 'multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc', &
                          kdist_file = '../../rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc'
  character(len=132)  ::  flx_file, timing_file, nndev_file='', nn_input_str, cmt
  integer             ::  nargs, ncol, nlay, nbnd, ngpt, nexp, nblocks, block_size
  logical             ::  top_at_1, use_rrtmgp_nn
  integer             ::  b, icol, ilay, igpt, ngas, ninputs, num_gases, ret, i, istat
  character(len=4)    ::  block_size_char
  character(len=6)    ::  emulated_component
  character(len=32 ), dimension(:),     allocatable :: kdist_gas_names, input_file_gas_names, gasopt_input_names
  ! Output fluxes
  real(wp), dimension(:,:,:),   target, allocatable :: flux_up, flux_dn, flux_dn_dir
  real(wp), dimension(:,:,:,:), target, allocatable :: gpt_flux_up, gpt_flux_dn, gpt_flux_dn_dir
  ! Thermodynamic and other variables
  real(wp), dimension(:,:,:),           allocatable :: p_lay, p_lev, t_lay, t_lev ! nlay, block_size, nblocks
  real(wp), dimension(:,:  ),           allocatable :: surface_albedo, total_solar_irradiance, solar_zenith_angle
                                                     ! block_size, nblocks
  real(wp), dimension(:,:  ),           allocatable :: sfc_alb_spec ! nbnd, block_size; spectrally-resolved surface albedo
  ! Optical properties and column dry amounts for NN development
  real(sp), dimension(:,:,:,:),         allocatable :: tau_sw, ssa_sw 
  real(wp), dimension(:,:,:),           allocatable :: col_dry, vmr_h2o
  ! RRTMGP inputs for NN development
  real(sp), dimension(:,:,:,:),         allocatable :: nn_gasopt_input ! (nfeatures,nlay,block_size,nblocks)
  ! RTE inputs for NN development
  real(sp),                             allocatable :: toa_flux_save(:,:,:), sfc_alb_spec_save(:,:,:), mu0_save(:,:)
  
  type(rrtmgp_network_type), dimension(2)        :: neural_nets ! First model is absorption, second is Rayleigh

  !
  ! logicals to control program
  logical ::  do_gpt_flux, save_optical_properties, save_gpt_flux
  !
  ! Derived types from the RTE and RRTMGP libraries
  !
  type(ty_gas_optics_rrtmgp)                    :: k_dist
  type(ty_optical_props_2str)                   :: atmos
  type(ty_fluxes_flexible)                      :: fluxes

  real(wp), dimension(:,:), allocatable          :: toa_flux ! ngpt, block_size
  real(wp), dimension(:  ), allocatable          :: def_tsi, mu0    ! block_size
  logical , dimension(:,:), allocatable          :: usecol ! block_size, nblocks
  !
  ! ty_gas_concentration holds multiple columns; we make an array of these objects to
  !   leverage what we know about the input file
  !
  type(ty_gas_concs), dimension(:), allocatable  :: gas_conc_array
  real(wp), parameter :: deg_to_rad = acos(-1._wp)/180._wp
  real(wp) :: def_tsi_s
  ! ecRAD type netCDF IO
  type(netcdf_file)  :: nndev_file_netcdf

  ! -------------------------------------------------------------------------------------------------
  !
  ! Code starts
  !   all arguments are optional
  !
  !  ------------ I/O and settings -----------------
  ! Compute fluxes per g-point?
  do_gpt_flux   = .true.
  save_gpt_flux = .true.
  save_optical_properties = .false.
  use_rrtmgp_nn = .true.

  !
  ! Identify the set of gases used in the calculation 
  ! A gas might have a different name in the k-distribution than in the files
  ! provided (e.g. 'co2' and 'carbon_dioxide'), user needs to provide the correct ones
  !
  kdist_gas_names = ["h2o  ","o3   ","co2  ","n2o  ", "ch4  ","o2   ", "n2   "] !,"no2  "]
  input_file_gas_names =  ['water_vapor   ', &
                    'ozone         ', &
                    'carbon_dioxide', &
                    'nitrous_oxide ', &
                    'methane       ', &
                    'oxygen        ', &
                    'nitrogen      ']!,'no2           ']  
  num_gases = size(kdist_gas_names)

  print *, "Usage   : ./rrtmgp_sw_gendata_rfmipstyle [block_size] [input file] [k-distribution file] [input-output file]"

  nargs = command_argument_count()
  if (nargs <  4) call stop_on_err("Need to provide four arguments!")

  call get_command_argument(1, block_size_char)
  read(block_size_char, '(i4)') block_size
  call get_command_argument(2, input_file)
  call get_command_argument(3, kdist_file)
  call get_command_argument(4, nndev_file)

  ! How big is the problem? Does it fit into blocks of the size we've specifed?
  !
  call read_size(input_file, ncol, nlay, nexp)
  print *, "input file", input_file
  print *, "ncol:", ncol, "nexp:", nexp, "nlay:", nlay

  if(mod(ncol*nexp, block_size) /= 0 ) call stop_on_err("Stopping: number of columns doesn't fit evenly into blocks.")
  nblocks = (ncol*nexp)/block_size
  print *, "Doing ",  nblocks, "blocks of size ", block_size

  print *, "Calculation uses gases: ", (trim(input_file_gas_names(b)) // " ", b = 1, size(input_file_gas_names))

  ! How many neural network input features does this correspond to?
  ninputs = 2 ! RRTMGP-NN inputs consist of temperature, pressure and..
  do b = 1, num_gases
    ! ..mixing ratios of all selected gases except N2 and O2 (constants)
    if (trim(kdist_gas_names(b))=='o2' .or. trim(kdist_gas_names(b))=='n2') cycle
    ninputs = ninputs + 1 
  end do

  ! Load Neural Network models
  if (use_rrtmgp_nn) then
	  print *, 'loading shortwave absorption model'
    call neural_nets(1) % load_netcdf("../../neural/data/sw-g112-210809_absorption_BEST.nc")
    call neural_nets(2) % load_netcdf("../../neural/data/sw-g112-210809_rayleigh_BEST.nc")
    ninputs = size(neural_nets(1) % layers(1) % w_transposed, 2)
    ! print *, "NN supports gases: ", (trim(neural_nets(1)%input_names(b)) // " ", b = 3, size(neural_nets(1)%input_names))
  end if  

  ! --------------------------------------------------
  !
  ! Prepare data for use in rte+rrtmgp
  !
  !
  ! Allocation on assignment within reading routines
  !
  call read_and_block_pt(input_file, block_size, p_lay, p_lev, t_lay, t_lev)
  !
  ! Are the arrays ordered in the vertical with 1 at the top or the bottom of the domain?
  !
  top_at_1 = p_lay(1, 1, 1) < p_lay(nlay, 1, 1)

  !
  ! Read the gas concentrations and surface properties
  !
  call read_and_block_gases_ty(input_file, block_size, kdist_gas_names, input_file_gas_names, gas_conc_array)
  ! do b = 1, size(gas_conc_array(1)%concs)
  !   print *, "max of gas ", gas_conc_array(1)%gas_name(b), ":", maxval(gas_conc_array(1)%concs(b)%conc)
  ! end do

   call read_and_block_sw_bc(input_file, block_size, surface_albedo, total_solar_irradiance, solar_zenith_angle)

  !
  ! Read k-distribution information. load_and_init() reads data from netCDF and calls
  !   k_dist%init(); users might want to use their own reading methods
  !
  call load_and_init(k_dist, trim(kdist_file), gas_conc_array(1))
  if(.not. k_dist%source_is_external()) stop "rrtmgp_rfmip_sw: k-distribution file isn't SW"
  nbnd = k_dist%get_nband()
  ngpt = k_dist%get_ngpt()

  allocate(toa_flux(k_dist%get_ngpt(), block_size), &
           def_tsi(block_size), usecol(block_size,nblocks))

  !
  ! RRTMGP won't run with pressure less than its minimum. The top level in the RFMIP file
  !   is set to 10^-3 Pa. Here we pretend the layer is just a bit less deep.
  !   This introduces an error but shows input sanitizing.
  !
  print *, "min of play", minval(p_lay), "k_dist%get_press_min()", k_dist%get_press_min() 

  ! where(p_lay < k_dist%get_press_min()) p_lay = k_dist%get_press_min() + spacing (k_dist%get_press_min())
  ! where(p_lev < k_dist%get_press_min()) p_lev = k_dist%get_press_min() + spacing (k_dist%get_press_min())

  if(top_at_1) then
    p_lay(1,:,:)    = k_dist%get_press_min() + epsilon(k_dist%get_press_min())
  else
    p_lay(nlay,:,:) = k_dist%get_press_min() + epsilon(k_dist%get_press_min())
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

  !
  ! Allocate g-point fluxes if desired
  !
  if (do_gpt_flux) then
    allocate(gpt_flux_up(ngpt, nlay+1, block_size, nblocks), &
             gpt_flux_dn(ngpt, nlay+1, block_size, nblocks))
    allocate(gpt_flux_dn_dir(ngpt, nlay+1, block_size, nblocks))
  end if

  ! allocate(mu0(block_size), sfc_alb_spec(nbnd,block_size))
  allocate(mu0_save(block_size, nblocks), sfc_alb_spec(ngpt,block_size))


  ! Allocate derived types - optical properties of gaseous atmosphere
  call stop_on_err(atmos%alloc_2str(block_size, nlay, k_dist))

  ! RRTMGP inputs
  allocate(gasopt_input_names(ninputs)) ! temperature + pressure + gases

  ! if (save_optical_properties) then
  allocate(nn_gasopt_input(   ninputs, nlay, block_size, nblocks))
  ! number of dry air molecules
  allocate(col_dry(nlay, block_size, nblocks), vmr_h2o(nlay, block_size, nblocks)) 
  print *, "allocated"
  ! end if

  ! RRTMGP outputs
  allocate(tau_sw(ngpt, nlay, block_size, nblocks), ssa_sw(ngpt, nlay, block_size, nblocks))


#ifdef USE_TIMING
  !
  ! Initialize timers
  !
  ret = gptlsetoption (gptlpercent, 1)        ! Turn on "% of" print
  ret = gptlsetoption (gptloverhead, 0)       ! Turn off overhead estimate 
  ret =  gptlinitialize()
#endif

  print *, "-------------------------------------------------------------------------"
  print *, "starting clear-sky shortwave computations, save_input_vec:", save_optical_properties

  !
  ! Loop over blocks
  !

#ifdef USE_OPENMP
  !$OMP PARALLEL shared(k_dist) firstprivate(def_tsi,toa_flux,sfc_alb_spec,mu0_save,fluxes,atmos)
  !$OMP DO 
#endif
  do b = 1, nblocks

    fluxes%flux_up => flux_up(:,:,b)
    fluxes%flux_dn => flux_dn(:,:,b)
    fluxes%flux_dn_dir => flux_dn_dir(:,:,b)
    if (do_gpt_flux) then
      ! If g-point fluxes are allocated, the RTE kernels write to 3D arrays, otherwise broadband
      ! computation is inlined
      fluxes%gpt_flux_up => gpt_flux_up(:,:,:,b)
      fluxes%gpt_flux_dn => gpt_flux_dn(:,:,:,b)
      fluxes%gpt_flux_dn_dir => gpt_flux_dn_dir(:,:,:,b)
    end if
    !
    ! Compute the optical properties of the atmosphere and the Planck source functions
    !    from pressures, temperatures, and gas concentrations...
    !


    if (use_rrtmgp_nn) then
      call stop_on_err(k_dist%gas_optics(p_lay(:,:,b), &
                                          p_lev(:,:,b),       &
                                          t_lay(:,:,b),       &
                                          gas_conc_array(b),  &
                                          atmos,      &
                                          toa_flux, neural_nets=neural_nets))
    else
      call stop_on_err(k_dist%gas_optics(p_lay(:,:,b), &
                                          p_lev(:,:,b),       &
                                          t_lay(:,:,b),       &
                                          gas_conc_array(b),  &
                                          atmos,      &
                                          toa_flux))
    end if                              

    ! Save RRTMGP outputs for NN training   
    tau_sw(:,:,:,b)     = atmos%tau
    ssa_sw(:,:,:,b)     = atmos%ssa 

    ! print *, "b", b, "mean ssa", mean_3d(ssa_sw(:,:,:,b))
    if (b==1) then 
      print *, "tau sw lev 30 col1", tau_sw(:,30,1,b)
      print *, "ssa sw lev 30 col1", ssa_sw(:,30,1,b)

    end if 


    ! NN inputs: this is just a 3D array (ninputs,nlay,ncol),
    ! where the inner dimension vector consists of (tlay, play, vmr_h2o, vmr_o3, vmr_co2...)
    ! Since these inputs are already provided in the original profiles, they are only saved if
    ! save_optical_properties = true; otherwise only the string gasopt_input_names is saved in the output file
    call stop_on_err(get_gasopt_nn_inputs(                  &
                  block_size, nlay, ninputs,                        &
                  p_lay(:,:,b), t_lay(:,:,b), gas_conc_array(b),      &
                  nn_gasopt_input(:,:,:,b), gasopt_input_names))
  
    ! if (save_optical_properties) then
    ! column dry amount, needed to normalize outputs (could also be computed within Python)
    call stop_on_err(gas_conc_array(b)%get_vmr('h2o', vmr_h2o(:,:,b)))
    call get_col_dry(vmr_h2o(:,:,b), p_lev(:,:,b), col_dry(:,:,b))
    ! end if

    ! print *, "mean tau after gas optics", mean_3d(atmos%tau)

    !
    ! Boundary conditions
    !
    ! What's the total solar irradiance assumed by RRTMGP?
    ! 
    do icol = 1, block_size
      def_tsi_s = 0.0_wp
      do igpt = 1, ngpt
        def_tsi_s = def_tsi_s + toa_flux(igpt, icol)
      end do
      def_tsi(icol) = def_tsi_s
    end do

      
    ! if (b==1) then 
    !   print *, "irrad", total_solar_irradiance(1,b)
    !   print *, "toa flux :, 1", toa_flux(:,1)
    ! end if 

    do icol = 1, block_size
      do igpt = 1, ngpt
        ! Normalize incoming solar flux to match RFMIP specification
        toa_flux(igpt,icol) = toa_flux(igpt,icol) * total_solar_irradiance(icol,b)/def_tsi(icol)
        ! Expand the spectrally-constant surface albedo to a per-g-point albedo for each column
        sfc_alb_spec(igpt,icol) = surface_albedo(icol,b)
      end do
    end do
      
    ! if (b==1) then 
    !   print *, " 2 toa flux :, 1", toa_flux(:,1)
    ! end if 
    !
    ! Cosine of the solar zenith angle
    !
    do icol = 1, block_size
      mu0_save(icol, b) = merge(cos(solar_zenith_angle(icol,b)*deg_to_rad), 1._wp, usecol(icol,b))
    end do


    !
    ! ... and compute the spectrally-resolved fluxes, providing reduced values
    !    via ty_fluxes_broadband
    !
    call stop_on_err(rte_sw(atmos,   &
                          top_at_1,        &
                          mu0_save(:,b),             &
                          toa_flux,        &
                          sfc_alb_spec, sfc_alb_spec,  &
                          fluxes))
 
      
    if (b==1) then 
      print *, "flux dn 1", flux_dn(:,1,b)
      print *, "flux dndir 1", flux_dn_dir(:,1,b)

      print *, "flux up 1", flux_up(:,1,b)
    end if 
                    
    ! print *, "mu0_save", mu0_save(1), "toa flux", sum(toa_flux(:,1)), "usecol", usecol(1, b)


  end do !blocks
#ifdef USE_OPENMP
  !$OMP END DO
  !$OMP END PARALLEL
  !$OMP barrier
#endif
  !
  ! End timers
  !
#ifdef USE_TIMING
  timing_file = "timing.sw-" // adjustl(trim(block_size_char))
  ret = gptlpr_file(trim(timing_file))
  ret = gptlfinalize()
#endif

  call atmos%finalize()

  print *, "Finished with computations!"
!
  ! Zero out fluxes for which the original solar zenith angle is > 90 degrees.
  !
  ! do b = 1, nblocks
  !   do icol = 1, block_size
  !     if(.not. usecol(icol,b)) then
  !       flux_up(:,icol,b)  = 0._wp
  !       flux_dn(:,icol,b)  = 0._wp
  !     end if
  !   end do
  ! end do

  print *, "mean of tau", mean_4d(tau_sw), "mean of ssa", mean_4d(ssa_sw), "max of ssa", maxval(ssa_sw)
  print *, "mean of flux_down is:", mean_3d(flux_dn)  ! mean of flux_down is:   292.71945410963957     
  print *, "mean of flux_up is:", mean_3d(flux_up)    ! mean of flux_up is:   41.835381782065106 
  !  if(do_gpt_flux) print *, "mean of gpt_flux_up for gpt=1 is:", mean_3d(gpt_flux_up(1,:,:,:))

  print *, "-------------------------------------------------------------------------"

  print *, "Attempting to save RRTMGP input/output to ", nndev_file

  ! Create file
  call nndev_file_netcdf%create(trim(nndev_file),is_hdf5_file=.true.)

  ! Put global attributes

  cmt = "generated with " // kdist_file 
  call nndev_file_netcdf%put_global_attributes( &
          &   title_str="Input - output from computations with RRTMGP-SW gas optics scheme", &
          &   input_str=input_file, comment_str = trim(cmt))

  ! Define dimensions
  call nndev_file_netcdf%define_dimension("expt", nexp)
  call nndev_file_netcdf%define_dimension("site", ncol)
  call nndev_file_netcdf%define_dimension("layer", nlay)
  call nndev_file_netcdf%define_dimension("level", nlay+1)
  call nndev_file_netcdf%define_dimension("feature", ninputs)
  call nndev_file_netcdf%define_dimension("gpt", ngpt)
  call nndev_file_netcdf%define_dimension("bnd", nbnd)

  ! RTE inputs and outputs (broadband fluxes), always saved (may not be needed, but take up little space)
  call nndev_file_netcdf%define_variable("rsu", &
  &   dim3_name="expt", dim2_name="site", dim1_name="level", &
  &   long_name="upwelling shortwave flux")

  call nndev_file_netcdf%define_variable("rsd", &
  &   dim3_name="expt", dim2_name="site", dim1_name="level", &
  &   long_name="downwelling shortwave flux")

  call nndev_file_netcdf%define_variable("rsd_dir", &
  &   dim3_name="expt", dim2_name="site", dim1_name="level", &
  &   long_name="direct downwelling shortwave flux")

  call nndev_file_netcdf%define_variable("pres_level", &
  &   dim3_name="expt", dim2_name="site", dim1_name="level", &
  &   long_name="pressure at half-level")

  ! RRTMGP inputs
  cmt = "inputs for RRTMGP shortwave gas optics"
  ! nn_input_str = 'Features:'
  ! do b  = 1, size(gasopt_input_names)
  !     nn_input_str = trim(nn_input_str) // " " // trim(gasopt_input_names(b)) 
  ! end do
  nn_input_str = trim(gasopt_input_names(1)) 
  do b  = 2, size(gasopt_input_names)
      nn_input_str = trim(nn_input_str) // " " // trim(gasopt_input_names(b)) 
  end do
  call nndev_file_netcdf%define_variable("rrtmgp_sw_input", &
  &   dim4_name="expt", dim3_name="site", dim2_name="layer", dim1_name="feature", &
  &   long_name =cmt, comment_str=nn_input_str, &
  &   data_type_name="float")

  if (save_optical_properties) then
    call nndev_file_netcdf%define_variable("tau_sw_gas", &
    &   dim4_name="expt", dim3_name="site", dim2_name="layer", dim1_name="gpt", &
    &   long_name="gas optical depth", data_type_name="float")
    call nndev_file_netcdf%define_variable("ssa_sw_gas", &
    &   dim4_name="expt", dim3_name="site", dim2_name="layer", dim1_name="gpt", &
    &   long_name="gas single scattering albedo", data_type_name="float")
  end if
  call nndev_file_netcdf%define_variable("col_dry", &
  &   dim3_name="expt", dim2_name="site", dim1_name="layer", &
  &   long_name="layer number of dry air molecules")

  call nndev_file_netcdf%define_variable("surface_albedo", &
    &   dim2_name="expt", dim1_name="site", &
    &   long_name="surface albedo (broadband)")


  call nndev_file_netcdf%define_variable("mu0", &
    &   dim2_name="expt", dim1_name="site", &
    &   long_name="cosine of solar zenith angle")

  call nndev_file_netcdf%define_variable("total_solar_irradiance", &
    &   dim2_name="expt", dim1_name="site", &
    &   long_name="incoming solar irradiance at top-of-atmosphere")

  if (save_gpt_flux) then 

    call nndev_file_netcdf%define_variable("gpt_flux_up", &
    &   dim4_name="expt", dim3_name="site", dim2_name="level", dim1_name="gpt", &
    &   long_name="spectral upwelling shortwave flux", data_type_name="float")

    call nndev_file_netcdf%define_variable("gpt_flux_dn", &
    &   dim4_name="expt", dim3_name="site", dim2_name="level", dim1_name="gpt", &
    &   long_name="spectral downwelling shortwave flux", data_type_name="float")

    call nndev_file_netcdf%define_variable("gpt_flux_dn_dir", &
    &   dim4_name="expt", dim3_name="site", dim2_name="level", dim1_name="gpt", &
    &   long_name="spectral direct downwelling shortwave flux", data_type_name="float")
  end if 


  call nndev_file_netcdf%end_define_mode()

  call unblock_and_write(trim(nndev_file), 'pres_level', p_lev)

  call unblock_and_write(trim(nndev_file), 'rsu', flux_up)
  call unblock_and_write(trim(nndev_file), 'rsd', flux_dn)
  call unblock_and_write(trim(nndev_file), 'rsd_dir', flux_dn_dir)

  call unblock_and_write(trim(nndev_file), 'rrtmgp_sw_input',nn_gasopt_input)
  deallocate(nn_gasopt_input)
  print *, "RRTMGP inputs (gas concs + T + p) were successfully saved"

  ! print *," min max col dry", minval(col_dry), maxval(col_dry)
  call unblock_and_write(trim(nndev_file), 'col_dry', col_dry)
  deallocate(col_dry)
  if (save_optical_properties) then
    call unblock_and_write(trim(nndev_file), 'tau_sw_gas', tau_sw)
    call unblock_and_write(trim(nndev_file), 'ssa_sw_gas', ssa_sw)
    deallocate(tau_sw,ssa_sw)
  end if
  if (save_gpt_flux) then
    call unblock_and_write(trim(nndev_file), 'gpt_flux_up', gpt_flux_up)
    call unblock_and_write(trim(nndev_file), 'gpt_flux_dn', gpt_flux_dn)
    call unblock_and_write(trim(nndev_file), 'gpt_flux_dn_dir', gpt_flux_dn_dir)
  end if


  call unblock_and_write(trim(nndev_file), 'surface_albedo', surface_albedo)
  call unblock_and_write(trim(nndev_file), 'mu0', mu0_save)

  call unblock_and_write(trim(nndev_file), 'total_solar_irradiance', total_solar_irradiance)

  print *, "Optical properties (RRTMGP output) were successfully saved. All done!"

  if (do_gpt_flux) then
      deallocate(gpt_flux_up, gpt_flux_dn, gpt_flux_dn_dir)
  end if 

  call nndev_file_netcdf%close()

  deallocate(flux_up, flux_dn, flux_dn_dir)

  print *, "-------------------------------------------------------------------------"

  contains

  ! -------------------------------------------------------------------------------------------------
  ! Routine for preparing neural network inputs from the gas concentrations, temperature and pressure
  ! Having the NN inputs in the same file as the outputs, and in the same shape, may be convenient
  ! but the input can also be constructed from the profiles. In this case, the string gasopt_input_names
  ! is still needed to know which gases were used in the computation
  function get_gasopt_nn_inputs(ncol, nlay, ninputs, &
                              play, tlay, gas_desc,           &
                              nn_inputs, gasopt_input_names) result(error_msg)

    integer,                                  intent(in   ) ::  ncol, nlay, ninputs
    real(wp), dimension(nlay,ncol),           intent(in   ) ::  play, &   ! layer pressures [Pa, mb]; (nlay,ncol)
                                                                tlay
    type(ty_gas_concs),                       intent(in   ) ::  gas_desc  ! Gas volume mixing ratios  
    real(sp), dimension(ninputs, nlay, ncol),  intent(inout) ::  nn_inputs !
    character(len=32 ), dimension(ninputs),    intent(inout) ::  gasopt_input_names 
    character(len=128)                                  :: error_msg
    ! ----------------------------------------------------------
    ! Local variables
    integer :: igas, ilay, icol, ndims, idx_h2o, idx_o3, idx_gas, i
    character(len=32)                           :: gas_name    
    real(wp),       dimension(nlay,ncol)        :: vmr

    ! First lets write temperature, pressure, water vapor and ozone into the inputs
    ! These are assumed to always be present!
    error_msg = gas_desc%get_conc_dims_and_igas('h2o', ndims, idx_h2o)
    error_msg = gas_desc%get_conc_dims_and_igas('o3',  ndims, idx_o3)
    if(error_msg  /= '') return

    gasopt_input_names(1) = 'tlay'
    gasopt_input_names(2) = 'play'
    gasopt_input_names(3) = 'h2o'
    gasopt_input_names(4) = 'o3'
    do icol = 1, ncol
      do ilay = 1, nlay
          nn_inputs(1,ilay,icol)    =  tlay(ilay,icol)
          nn_inputs(2,ilay,icol)    =  play(ilay,icol)
          nn_inputs(3,ilay,icol)    =  gas_desc%concs(idx_h2o)%conc(ilay,icol)
          nn_inputs(4,ilay,icol)    =  gas_desc%concs(idx_o3) %conc(ilay,icol)
      end do
    end do

    ! Write the remaining gases
    ! The scaling coefficients are tied to a string specifying the gas names, these are all loaded from rrtmgp_constants.F90
    ! Lets find the indices which map the available gases to the scaling coefficients of each gas, 
    ! and also the dimensions of the concentration array
    i = 5
    do igas = 1, size(gas_desc%gas_name)
    
      gas_name = gas_desc%gas_name(igas)
      if(gas_name=='h2o' .or. gas_name=='o3' .or. gas_name=='o2' .or. gas_name=='n2') cycle

      ! Save gas name
      gasopt_input_names(i) = gas_name

      ! Fill 2D (lay,col) array with gas concentration
      error_msg = gas_desc%get_vmr(gas_name, vmr(:,:))

      ! Write to nn_input non-contiguously
      nn_inputs(i,:,:) = vmr(:,:)
      i = i + 1
    end do

  end function get_gasopt_nn_inputs


  function rmse(x1,x2) result(res)
    implicit none 
    real(wp), dimension(:), intent(in) :: x1,x2
    real(wp) :: res
    real(wp), dimension(size(x1)) :: diff 
    
    diff = x1 - x2
    res = sqrt( sum(diff**2)/size(diff) )
  end function rmse

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
  
  function mean_4d(x) result(mean4)
    implicit none 
    real(wp), dimension(:,:,:,:), intent(in) :: x
    real(wp) :: mean4
    mean4 = sum(x) / size(x)
  end function mean_4d

end program rrtmgp_rfmip_sw
