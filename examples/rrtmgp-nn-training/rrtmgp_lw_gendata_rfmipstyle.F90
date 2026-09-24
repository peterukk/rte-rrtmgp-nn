! This program is for generating training data for neural network emulators of RRTMGP-LW
!
! The RRTMGP inputs, used as NN inputs, are layer-wise RRTMGP input variables (T, p, gas concentrations)
! The fina lRRTMGP outputs are optical depth tau and single-scattering albedo ssa. These could be used as NN outputs, 
! but in order to support the methodology in Ukkonen (2020), where two separate NNs predict absorption cross-section
! and Planck fractions, for convenience the layer number of dry air molecules (N) are also saved
! 
! y_abs      = (tau) / N 
! Ukkonen et al. (2020): https://doi.org/10.1029/2020MS002226
!
! Fortran program arguments:
! rrtmgp_lw_gendata_rfmipstyle [block_size] [input file] [k-distribution file] [file to save NN inputs/outputs]"
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
    write (error_unit,*) "rrtmgp_rfmip_lw stopping"
    stop
  end if
end subroutine stop_on_err
! -------------------------------------------------------------------------------------------------
!
! Main program
!
! -------------------------------------------------------------------------------------------------
program rrtmgp_rfmip_lw
  ! --------------------------------------------------
  !
  ! Modules for working with rte and rrtmgp
  !
#ifdef USE_OPENMP
  use omp_lib
#endif
  ! Working precision for real variables
  !
  use mo_rte_kind,           only: wp, sp, wl, i4
  !
  ! Optical properties of the atmosphere as array of values
  !   In the longwave we include only absorption optical depth (_1scl)
  !
  use mo_optical_props,      only: ty_optical_props_1scl
  !
  ! Gas optics: maps physical state of the atmosphere to optical properties
  !
  use mo_gas_optics_rrtmgp,  only: ty_gas_optics_rrtmgp, compute_nn_inputs, get_col_dry
  !
  ! Gas optics uses a derived type to represent gas concentrations compactly...
  !
  use mo_gas_concentrations, only: ty_gas_concs
  !
  ! ... and another type to encapsulate the longwave source functions.
  !
  use mo_source_functions,   only: ty_source_func_lw
  !
  ! RTE longwave driver
  !
  use mo_rte_lw,             only: rte_lw
  !
  ! RTE driver uses a derived type to reduce spectral fluxes to whatever the user wants
  !   Here we're using a flexible type which saves broadband fluxes and optionally g-point fluxes
  !
  use mo_fluxes,             only: ty_fluxes_broadband, ty_fluxes_flexible
  ! --------------------------------------------------
  !
  ! modules for reading and writing files
  !
  ! RRTMGP's gas optics class needs to be initialized with data read from a netCDF files
  !
  use mo_load_coefficients,  only: load_and_init
  use mo_io_rfmipstyle_generic, only: read_size, read_and_block_pt, read_and_block_gases_ty, unblock_and_write, &
                                   read_and_block_lw_bc, determine_gas_names                             
  use netcdf
  use easy_netcdf        
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
  character(len=132) :: input_file = 'multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc', &
                        kdist_file = 'coefficients_lw.nc'
  character(len=132) :: flx_file, timing_file, nndev_file='', nn_input_str, cmt
  integer            :: nargs, ncol, nlay, nbnd, ngas, ngpt, nexp, nblocks, block_size, forcing_index, physics_index, n_quad_angles = 1
  logical            :: top_at_1
  integer            :: b, icol, ilay, ibnd, igpt, ninputs, num_gases, istat, igas, ret, i
  character(len=5)   :: block_size_char, forcing_index_char = '1', physics_index_char = '1'
  character(len=32 ), dimension(:),     allocatable :: kdist_gas_names, input_file_gas_names, gasopt_input_names
  real(wp), dimension(:,:,:),         allocatable :: p_lay, p_lev, t_lay, t_lev ! block_size, nlay, nblocks
  real(wp), dimension(:,:,:), target, allocatable :: flux_up, flux_dn
  real(wp), dimension(:,:,:,:), target, allocatable :: gpt_flux_up, gpt_flux_dn
  real(wp), dimension(:,:  ),         allocatable :: sfc_emis, sfc_t  ! block_size, nblocks (emissivity is spectrally constant)
  real(wp), dimension(:,:  ),         allocatable :: sfc_emis_spec    ! nbands, block_size (spectrally-resolved emissivity)
  ! Optical properties and column dry amounts for NN development
  real(sp), dimension(:,:,:,:),         allocatable :: tau_lw, planck_frac 
  real(wp), dimension(:,:,:),           allocatable :: col_dry, vmr_h2o
  ! RRTMGP inputs for NN development
  real(sp), dimension(:,:,:,:),         allocatable :: nn_gasopt_input ! (nfeatures,nlay,block_size,nblocks)
  ! logicals to control program
  logical 	:: do_gpt_flux, save_gpt_flux, save_input_vectors = .true.
  !
  ! Derived types from the RTE and RRTMGP libraries
  !
  type(ty_gas_optics_rrtmgp)  :: k_dist
  type(ty_source_func_lw)     :: source
  type(ty_optical_props_1scl) :: atmos
  type(ty_fluxes_flexible)   :: fluxes
  !
  ! ty_gas_concentration holds multiple columns; we make an array of these objects to
  !   leverage what we know about the input file
  !
  type(ty_gas_concs), dimension(:), allocatable  :: gas_conc_array
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
  save_input_vectors = .true.
  
  ! Identify the set of gases used in the calculation 
  ! The data file might have gases we're not interested in, so the gas names should be provided
  ! by the user somehow, checked that they're in the file, and then mapped to the k-distribution
  ! since a gas might have a different name in the k-distribution file
  ! A lot of this could be done within Python
  
  ! CKDMIP gases
  ! kdist_gas_names = ["h2o  ","o3   ","co2  ", "ch4  ", "n2o  ", "o2   ", "n2   ", "cfc11", "cfc12"]!, "ccl4 " ] 
  ! input_file_gas_names = ['water_vapor          ', &
  !                         'ozone                ', &            
  !                         'carbon_dioxide       ', &
  !                         'methane              ', &
  !                         'nitrous_oxide        ', &
  !                         'oxygen               ', &
  !                         'nitrogen             ', &
  !                         'cfc11                ', &
  !                         'cfc12                ']!, &
  !                         !'carbon_tetrachloride ']

  ! All LW RRTMGP gases
  kdist_gas_names = ["h2o    ","o3     ","co2    ", "ch4    ", "n2o    ", "o2     ", "n2     ", "cfc11  ", "cfc12  ", & 
                     "co     ","ccl4   ","cfc22  ", "hfc143a", "hfc125 ", "hfc23  ", "hfc32  ", "hfc134a", "cf4    "]! no2
  input_file_gas_names =  [ 'water_vapor         ', &
                            'ozone               ', &            
                            'carbon_dioxide      ', &
                            'methane             ', &
                            'nitrous_oxide       ', &
                            'oxygen              ', &
                            'nitrogen            ', &
                            'cfc11               ', &
                            'cfc12               ', &
                            'carbon_monoxide     ', &
                            'carbon_tetrachloride', &
                            'hcfc22              ', &
                            'hfc143a             ', &
                            'hfc125              ', &
                            'hfc23               ', &
                            'hfc32               ', &
                            'hfc134a             ', &
                            'cf4                 ' ]


  num_gases = size(kdist_gas_names)

  print *, "Usage: rrtmgp_rfmip_lw [block_size] [rfmip_file] [k-distribution_file] input_output file]"

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

  call read_and_block_lw_bc(input_file, block_size, sfc_emis, sfc_t)
  
  !
  ! Read k-distribution information. load_and_init() reads data from netCDF and calls
  !   k_dist%init(); users might want to use their own reading methods
  !
  call load_and_init(k_dist, trim(kdist_file), gas_conc_array(1))

  if(.not. k_dist%source_is_internal()) &
    stop "rrtmgp_rfmip_lw: k-distribution file isn't LW"

  nbnd = k_dist%get_nband()
  ngpt = k_dist%get_ngpt()

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
  ! Allocate space for output fluxes (accessed via pointers in ty_fluxes_broadband),
  !   gas optical properties, and source functions. The %alloc() routines carry along
  !   the spectral discretization from the k-distribution.
  !
  allocate(flux_up(    	nlay+1, block_size, nblocks), &
           flux_dn(    	nlay+1, block_size, nblocks))
  !
  ! Allocate g-point fluxes if desired
  !
  if (do_gpt_flux) then
    allocate(gpt_flux_up(ngpt, nlay+1, block_size, nblocks), &
             gpt_flux_dn(ngpt, nlay+1, block_size, nblocks))
  end if

  allocate(sfc_emis_spec(nbnd, block_size))

  call stop_on_err(source%alloc    (block_size, nlay, k_dist, save_pfrac = .true.))   
  call stop_on_err(atmos%alloc_1scl(block_size, nlay, k_dist))

  ! RRTMGP inputs
  allocate(gasopt_input_names(ninputs)) ! temperature + pressure + gases

  if (save_input_vectors) then
    allocate(nn_gasopt_input(   ninputs, nlay, block_size, nblocks))
    ! number of dry air molecules
    allocate(col_dry(nlay, block_size, nblocks), vmr_h2o(nlay, block_size, nblocks)) 
  end if

  ! RRTMGP outputs
  allocate(tau_lw(ngpt, nlay, block_size, nblocks), planck_frac(ngpt, nlay, block_size, nblocks))


#ifdef USE_TIMING
  !
  ! Initialize timers
  !
  ret = gptlsetoption (gptlpercent, 1)        ! Turn on "% of" print
  ret = gptlsetoption (gptloverhead, 0)       ! Turn off overhead estimate 
  ret =  gptlinitialize()
#endif

  print *, "-------------------------------------------------------------------------"
  print *, "starting clear-sky longwave computations"

  !
  ! Loop over blocks
  !

#ifdef USE_OPENMP
  !$OMP PARALLEL shared(k_dist) firstprivate(sfc_emis_spec,fluxes,atmos,source)
  !$OMP DO 
#endif
  do b = 1, nblocks
    
    fluxes%flux_up => flux_up(:,:,b)
    fluxes%flux_dn => flux_dn(:,:,b)    
    if (do_gpt_flux) then
      fluxes%gpt_flux_up => gpt_flux_up(:,:,:,b)
      fluxes%gpt_flux_dn => gpt_flux_dn(:,:,:,b)
    end if
    !
    ! Expand the spectrally-constant surface emissivity to a per-band emissivity for each column
    !   (This is partly to show how to keep work on GPUs using OpenACC)
    !
    do icol = 1, block_size
      do ibnd = 1, nbnd
        sfc_emis_spec(ibnd,icol) = sfc_emis(icol,b)
      end do
    end do

    !
    ! Compute the optical properties of the atmosphere and the Planck source functions
    !    from pressures, temperatures, and gas concentrations...
    !
    call stop_on_err(k_dist%gas_optics(p_lay(:,:,b),      &
                                      p_lev(:,:,b),       &
                                      t_lay(:,:,b),       &
                                      sfc_t(:  ,b),       &
                                      gas_conc_array(b),  &
                                      atmos,      &
                                      source,            &
                                      tlev = t_lev(:,:,b) ))

    ! Save RRTMGP outputs for NN training   
    tau_lw(:,:,:,b)       = atmos%tau
    planck_frac(:,:,:,b)  = source%planck_frac 
    ! NN inputs: this is just a 3D array (ninputs,nlay,ncol),
    ! where the inner dimension vector consists of (tlay, play, vmr_h2o, vmr_o3, vmr_co2...)
    ! Since these inputs are already provided in the original profiles, they are only saved if
    ! save_input_vectors = true; otherwise only the string gasopt_input_names is saved in the output file
    call stop_on_err(get_gasopt_nn_inputs(                  &
                  block_size, nlay, ninputs,                        &
                  p_lay(:,:,b), t_lay(:,:,b), gas_conc_array(b),      &
                  nn_gasopt_input(:,:,:,b), gasopt_input_names))
    if (save_input_vectors) then
      ! column dry amount, needed to normalize outputs (could also be computed within Python)
      call stop_on_err(gas_conc_array(b)%get_vmr('h2o', vmr_h2o(:,:,b)))
      call get_col_dry(vmr_h2o(:,:,b), p_lev(:,:,b), col_dry(:,:,b))
    end if

    ! print *, "mean of pfrac is:", mean_3d(planck_frac(:,:,:,b))   
    ! print *, "mean of tau is:", mean_3d(atmos%tau)
    ! print *, "mean of lay_source is:", mean_3d(source%lay_source)
    ! print *, "mean of lev_source is:", mean_3d(source%lev_source)

    !
    ! ... and compute the spectrally-resolved fluxes, providing reduced values
    !    via ty_fluxes_broadband
    !
    call stop_on_err(rte_lw(atmos,   &
                            top_at_1,        &
                            source,          &
                            sfc_emis_spec,   &
                            fluxes,          &
                            n_gauss_angles = n_quad_angles, use_2stream = .false.) )

  end do ! blocks

#ifdef USE_OPENMP
  !$OMP END DO
  !$OMP END PARALLEL
  !$OMP barrier
#endif

#ifdef USE_TIMING
  !   End timers
  timing_file = "timing.lw-" // adjustl(trim(block_size_char))
  ret = gptlpr_file(trim(timing_file))
  ret = gptlfinalize()
#endif

  call atmos%finalize()
  call source%finalize()

  print *, "Finished with computations!"

  print *, "mean of flux_down is:", mean_3d(flux_dn)  !  mean of flux_down is:   103.2458
  print *, "mean of flux_up is:", mean_3d(flux_up)

  print *, "-------------------------------------------------------------------------"
  print *, "Attempting to save RRTMGP input/output to ", nndev_file

  ! Create file
  call nndev_file_netcdf%create(trim(nndev_file),is_hdf5_file=.true.)

  ! Put global attributes

  cmt = "generated with " // kdist_file 
  call nndev_file_netcdf%put_global_attributes( &
          &   title_str="Input - output from computations with RRTMGP-LW gas optics scheme", &
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
  &   long_name="upwelling longwave flux")

  call nndev_file_netcdf%define_variable("rsd", &
  &   dim3_name="expt", dim2_name="site", dim1_name="level", &
  &   long_name="downwelling longwave flux")

  call nndev_file_netcdf%define_variable("pres_level", &
  &   dim3_name="expt", dim2_name="site", dim1_name="level", &
  &   long_name="pressure at half-level")

  ! RRTMGP inputs
  cmt = "inputs for RRTMGP longwave gas optics"
  ! nn_input_str = 'Features:'
  ! nn_input_str = ''
  ! do b  = 1, size(gasopt_input_names)
  !     nn_input_str = trim(nn_input_str) // " " // trim(gasopt_input_names(b)) 
  ! end do
  nn_input_str = trim(gasopt_input_names(1)) 
  do b  = 2, size(gasopt_input_names)
      nn_input_str = trim(nn_input_str) // " " // trim(gasopt_input_names(b)) 
  end do
  call nndev_file_netcdf%define_variable("rrtmgp_lw_input", &
  &   dim4_name="expt", dim3_name="site", dim2_name="layer", dim1_name="feature", &
  &   long_name =cmt, comment_str=nn_input_str, &
  &   data_type_name="float")

  if (save_input_vectors) then
    call nndev_file_netcdf%define_variable("tau_lw_gas", &
    &   dim4_name="expt", dim3_name="site", dim2_name="layer", dim1_name="gpt", &
    &   long_name="gas optical depth", data_type_name="float")
    call nndev_file_netcdf%define_variable("planck_fraction", &
    &   dim4_name="expt", dim3_name="site", dim2_name="layer", dim1_name="gpt", &
    &   long_name="Fraction of the Planck function associated with a g-point", data_type_name="float")
  end if
  call nndev_file_netcdf%define_variable("col_dry", &
  &   dim3_name="expt", dim2_name="site", dim1_name="layer", &
  &   long_name="layer number of dry air molecules")

  call nndev_file_netcdf%define_variable("sfc_emis", &
    &   dim2_name="expt", dim1_name="site", &
    &   long_name="surface emissivity (broadband)")

  call nndev_file_netcdf%define_variable("sfc_temperature", &
    &   dim2_name="expt", dim1_name="site", &
    &   long_name="surface temperature")


  if (save_gpt_flux) then 

    call nndev_file_netcdf%define_variable("gpt_flux_up", &
    &   dim4_name="expt", dim3_name="site", dim2_name="level", dim1_name="gpt", &
    &   long_name="spectral upwelling longwave flux", data_type_name="float")

    call nndev_file_netcdf%define_variable("gpt_flux_dn", &
    &   dim4_name="expt", dim3_name="site", dim2_name="level", dim1_name="gpt", &
    &   long_name="spectral downwelling longwave flux", data_type_name="float")

  end if 


  call nndev_file_netcdf%end_define_mode()

  call unblock_and_write(trim(nndev_file), 'pres_level', p_lev)

  call unblock_and_write(trim(nndev_file), 'rsu', flux_up)
  call unblock_and_write(trim(nndev_file), 'rsd', flux_dn)

  call unblock_and_write(trim(nndev_file), 'rrtmgp_lw_input',nn_gasopt_input)
  deallocate(nn_gasopt_input)
  print *, "RRTMGP inputs (gas concs + T + p) were successfully saved"

  ! print *," min max col dry", minval(col_dry), maxval(col_dry)
  call unblock_and_write(trim(nndev_file), 'col_dry', col_dry)
  deallocate(col_dry)
  if (save_input_vectors) then
    call unblock_and_write(trim(nndev_file), 'tau_lw_gas', tau_lw)
    call unblock_and_write(trim(nndev_file), 'planck_fraction', planck_frac)
    deallocate(tau_lw, planck_frac)
  end if
  if (save_gpt_flux) then
    call unblock_and_write(trim(nndev_file), 'gpt_flux_up', gpt_flux_up)
    call unblock_and_write(trim(nndev_file), 'gpt_flux_dn', gpt_flux_dn)
  end if

  call unblock_and_write(trim(nndev_file), 'sfc_emis', sfc_emis)
  call unblock_and_write(trim(nndev_file), 'sfc_temperature', sfc_t)

  print *, "Optical properties (RRTMGP output) were successfully saved. All done!"

  if (do_gpt_flux) then
      deallocate(gpt_flux_up, gpt_flux_dn)
  end if 

  call nndev_file_netcdf%close()

  deallocate(flux_up, flux_dn)

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

end program rrtmgp_rfmip_lw
