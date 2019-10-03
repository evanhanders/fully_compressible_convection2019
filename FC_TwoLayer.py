"""
Dedalus script for fully compressible, two-layer, internally heated convection.

Usage:
    FC_TwoLayer.py [options] 

Options:
    --Rayleigh=<Rayleigh>      Rayleigh number [default: 1e3]
    --Prandtl=<Prandtl>        Prandtl number = nu/kappa [default: 1]
    --n_rho_T=<n>              Number of density scale heights of top [default: 3]
    --n_rho_B=<n>              Number of density scale heights of bot [default: 1]
    --stable_top               If True, setup is top:RZ/bot:CZ, else it's flipped.
    --epsilon=<epsilon>        Superadiabatic excess of polytropic atmosphere [default: 1e-4]
    --aspect=<aspect>          Aspect ratio of problem (Lx = Ly = aspect*Lz) [default: 2]
    --3D                       Run in 3D

    --thermal_BC=<bc>          A short string specifying the thermal boundary conditions:
                                'flux'      : Fixed flux (top & bot)
                                'temp'      : Fixed temp (top & bot)
                                'entropy'   : Fixed entropy (top & bot)
                                'flux_temp' : Fixed flux (bot) and temp (top)
                                'temp_flux' : Fixed temp (bot) and flux (top)
                                [default: flux]
    
    --velocity_BC=<bc>         A short string specifying the horizontal velocity boundary conditions:
                                'stress_free' : dz(horizontal velocity) = 0 (top & bot)
                                'no_slip'     : (horizontal velocity) = 0 (top & bot)
                                [default: stress_free]

    --nz=<nz>                  Vertical coeff resolution   [default: 32]
    --nx=<nx>                  Horizontal coeff resolution [default: 64]
    --ny=<nx>                  Horizontal coeff resolution [default: 64]
    --mesh=<mesh>              Processor mesh if distributing 3D run in 2D 
    
    --restart=<restart_file>   The path to a checkpoint file to restart from
    --seed=<seed>              RNG seed for initial conditoins [default: 42]

    --run_time_wall=<t>        Run time, in wall hours [default: 23.5]
    --run_time_buoy=<t>        Run time, in simulation buoyancy times
    --run_time_therm=<t>       Run time, in simulation thermal times [default: 1]
    --run_time_restarted       If flagged, try to run exactly run_time_buoy or run_time_therm from the current sim time

    --root_dir=<dir>           Root directory for output [default: ./]
    --label=<label>            Optional additional case name label
    --output_dt=<num>          Simulation buoyancy times between outputs [default: 0.2]
    --checkpoint_buoy=<dt>     Simulation buoyancy times between outputs [default: 10]
    --no_join                  If flagged, don't join files at end of run

    --ae                       Accelerated evolution
    --ae_outs                  Output Accelerated evolution comparison tasks
    --ae_start_time=<t>        Buoyancy times to wait before first AE averaging [default: 20]

"""
def name_case(input_dict):
    """
    Creates an informative string of the form:

    FC_poly_Ra{0}_Pr{1}_nRho{2a}-{2b}_{3}_eps{4}_a{5}_{6}D_T{7}_V{8}_{9}{10}{11}

    which is the name of the output directory, and where the numbers here are:
    {0}     - Rayleigh number
    {1}     - Prandtl number
    {2a,b}  - number of density scale heights in (top, bot layer)
    {3}     - StableTop if top layer is stable, else StableBot
    {4}     - epsilon, superadiabatic excess
    {5}     - aspect ratio
    {6}     - 2 or 3 (dimensions of the problem)
    {7}     - Thermal BCs
    {8}     - Velocity BCs
    {9}     - Resolution (nz x nx [2D] or nz x nx x ny [3D])
    {10}     - _AE if doing an AE run, 
    {11}    - _label if label is not None

    Parameters
    ----------
    input_dict   : dict
        A dictionary of strings whose keys are all of the options specified above in the FC_poly.py docstring
    """
    import sys
    # save data in directory named after script
    case_name = sys.argv[0].split('.py')[0]
    case_name += "_Ra{}_Pr{}_nRho{}-{}_eps{}_a{}".format( input_dict['--Rayleigh'], 
                                                    input_dict['--Prandtl'], 
                                                    input_dict['--n_rho_B'],
                                                    input_dict['--n_rho_T'],
                                                    input_dict['--epsilon'],
                                                    input_dict['--aspect'] )
    if input_dict['--stable_top']:
        case_name += '_StableTop'
    else:
        case_name += '_StableBot'
    if input_dict['--3D']:
        case_name += '_3D'
    else:
        case_name += '_2D'
    case_name += '_T{}_V{}'.format( input_dict['--thermal_BC'], input_dict['--velocity_BC'] )
    if input_dict['--3D']:
        case_name += '_{}x{}x{}'.format( input_dict['--nz'], input_dict['--nx'], input_dict['--ny'] )
    else:
        case_name += '_{}x{}'.format( input_dict['--nz'], input_dict['--nx'] )
    if input_dict['--ae']:
        case_name += '_AE'
    if input_dict['--label'] is not None:
        case_name += "_{}".format(input_dict['--label'])
    data_dir = '{:s}/{:s}/'.format(input_dict['--root_dir'], case_name)
    return data_dir, case_name

def FC_TwoLayer_convection(input_dict):
    """
    A driver function for running FC convection in Dedalus.

    Parameters
    ----------
    input_dict   : dict
        A dictionary of strings whose keys are all of the options specified above in the FC_poly.py docstring
    """
    import os
    from collections import OrderedDict
    from dedalus.tools.config import config
    from logic.functions     import mpi_makedirs

    #Get info on data directory, create directories, setup logging
    # (Note: this order of imports is bad form, but gets logs properly outputting)
    data_dir, case_name = name_case(input_dict)
    mpi_makedirs(data_dir)
    logdir = os.path.join(data_dir,'logs')
    mpi_makedirs(logdir)
    file_level = config['logging']['file_level'] = 'debug'
    filename = config['logging']['filename'] = os.path.join(logdir,'dedalus_log')

    from dedalus import public as de
    from dedalus.extras import flow_tools

    from logic.atmospheres   import TwoLayerIH
    from logic.domains       import DedalusDomain
    from logic.experiments   import CompressibleConvection
    from logic.problems      import DedalusIVP
    from logic.equations     import KappaMuFCE
    from logic.ae_tools      import FCAcceleratedEvolutionIVP
    from logic.outputs       import initialize_output, ae_initialize_output
    from logic.checkpointing import Checkpoint
    from logic.field_averager import AveragerFCAE, AveragerFCStructure

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Running polytrope case: {}".format(case_name))
    logger.info("Saving run in: {}".format(data_dir))

    # Read in command line args & process them
    # Atmosphere params
    Ra, Pr, n_rho_T, n_rho_B, eps, aspect = [float(input_dict[k]) for k in ('--Rayleigh', '--Prandtl', '--n_rho_T', '--n_rho_B', '--epsilon', '--aspect')]
    threeD = input_dict['--3D']

    # BCs
    thermal_BC = OrderedDict(( ('flux',      False),
                               ('temp',      False),
                               ('flux_temp', False),
                               ('temp_flux', False) ))
    velocity_BC = OrderedDict(( ('stress_free',      False),
                                ('no_slip',          False) ))
    thermal_BC[input_dict['--thermal_BC']]   = True
    velocity_BC[input_dict['--velocity_BC']] = True

    # Coeff resolutions
    nx, ny, nz = [int(input_dict[n]) for n in ['--nx', '--ny', '--nz']]

    # 3D processor mesh.
    mesh = input_dict['--mesh']
    if mesh is not None:
        mesh = mesh.split(',')
        mesh = [int(mesh[0]), int(mesh[1])]

    # Stop conditions
    run_time_wall, run_time_buoy, run_time_therm = [input_dict[t] for t in ['--run_time_wall', '--run_time_buoy', '--run_time_therm']]
    run_time_wall = float(run_time_wall)
    if run_time_buoy is not None:
        run_time_buoy = float(run_time_buoy)
        run_time_therm = None
        logger.info("Terminating run after {} buoyancy times".format(run_time_buoy))
    else:
        run_time_therm = float(run_time_therm)
        logger.info("Terminating run after {} thermal times".format(run_time_therm))


    # Initialize atmosphere class 
    atmo_kwargs   = OrderedDict([('epsilon',        eps),
                                 ('n_rho_T',        n_rho_T),
                                 ('n_rho_B',        n_rho_B),
                                 ('gamma',          5./3),
                                 ('R',              1),
                                 ('stable_top',     args['--stable_top'])
                                ])
    atmosphere = TwoLayerIH(**atmo_kwargs)

    # Initialize domain class
    resolution = (nz, nx, ny)
    if not threeD: resolution = resolution[:2]
    domain_args   = (atmosphere, resolution, aspect)
    domain_kwargs = OrderedDict( (('mesh',       mesh),) )
    de_domain = DedalusDomain(*domain_args, **domain_kwargs)

    #Set diffusivities
    experiment_args = (de_domain, atmosphere, Ra, Pr)
    experiment_kwargs = {}
    experiment = CompressibleConvection(*experiment_args, **experiment_kwargs)

    #Set up problem and equations
    if args['--ae']:
        problem_type = FCAcceleratedEvolutionIVP
    else:
        problem_type = DedalusIVP
    de_problem = problem_type(de_domain)

    equations = KappaMuFCE(thermal_BC, velocity_BC, atmosphere, de_domain, de_problem)

    atmosphere.save_atmo_file('./', de_domain)

    # Build solver, set stop times
    de_problem.build_solver(de.timesteppers.RK222)

    #Setup checkpointing and initial conditions
    checkpoint = Checkpoint(data_dir)
    dt, mode =     experiment.set_IC(de_problem.solver, eps, checkpoint, restart=args['--restart'], seed=int(args['--seed']), checkpoint_dt=float(args['--checkpoint_buoy'])*atmosphere.atmo_params['t_buoy'])
    if run_time_buoy is None:
        stop_sim_time = run_time_therm*atmosphere.atmo_params['t_therm']
    else:
        stop_sim_time = run_time_buoy*atmosphere.atmo_params['t_buoy']
    if args['--run_time_restarted']:
        stop_sim_time += de_problem.solver.sim_time
        
    de_problem.set_stop_condition(stop_wall_time=run_time_wall*3600, stop_sim_time=stop_sim_time)



    #Set up outputs
    output_dt = float(args['--output_dt'])*atmosphere.atmo_params['t_buoy']
    if args['--ae_outs']:
        analysis_tasks = ae_initialize_output(de_domain, de_problem, data_dir, 
                                       output_dt=output_dt, output_vol_dt=atmosphere.atmo_params['t_buoy'], mode=mode)# volumes_output=True)
    else:
        analysis_tasks = initialize_output(de_domain, de_problem, data_dir, 
                                       output_dt=output_dt, output_vol_dt=atmosphere.atmo_params['t_buoy'], mode=mode)# volumes_output=True)

    # Ensure good initial dt and setup CFL
    max_dt = min((0.2, float(args['--output_dt'])))*atmosphere.atmo_params['t_buoy']
    if dt is None:
        dt = max_dt
    cfl_safety = 0.1
    CFL = flow_tools.CFL(de_problem.solver, initial_dt=dt, cadence=1, safety=cfl_safety*2,
                         max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.1)
    if threeD:
        CFL.add_velocities(('u', 'v', 'w'))
    else:
        CFL.add_velocities(('u', 'w'))

    # Solve the IVP.
    if args['--ae']:
        task_args = (thermal_BC,)
        pre_loop_args = ((AveragerFCAE, AveragerFCStructure), (True, False), data_dir, atmo_kwargs, CompressibleConvection, experiment_args, experiment_kwargs)
        task_kwargs = {}
        pre_loop_kwargs = { 'sim_time_start' : int(args['--ae_start_time'])*atmosphere.atmo_params['t_buoy'], 
                            'min_bvp_time' : 10*atmosphere.atmo_params['t_buoy'], 
                            'between_ae_wait_time' : 20*atmosphere.atmo_params['t_buoy'],
                            'later_bvp_time' : 20*atmosphere.atmo_params['t_buoy'],
                            'ae_convergence' : 1e-2, 
                            'bvp_threshold' : 1e-2,
                            'min_bvp_threshold' : 5e-3
                            }
        solve_args = (dt, CFL, data_dir, analysis_tasks)
        solve_kwargs = {    'task_args' : task_args,
                            'pre_loop_args' : pre_loop_args,
                            'task_kwargs' : task_kwargs,
                            'pre_loop_kwargs' : pre_loop_kwargs,
                            'time_div' : atmosphere.atmo_params['t_buoy'],
                            'track_fields' : ['Pe_rms', 'dissipation'],
                            'threeD' : threeD,
                            'Hermitian_cadence' : 100,
                            'no_join' : args['--no_join'],
                            'mode' : mode
                        }
        de_problem.solve_IVP(*solve_args, **solve_kwargs)
    else:
        de_problem.solve_IVP(dt, CFL, data_dir, analysis_tasks, time_div=atmosphere.atmo_params['t_buoy'], track_fields=['Pe_rms', 'dissipation'], threeD=threeD, Hermitian_cadence=100, no_join=args['--no_join'], mode=mode)

if __name__ == "__main__":
    from docopt import docopt
    args = docopt(__doc__)
    FC_TwoLayer_convection(args)
