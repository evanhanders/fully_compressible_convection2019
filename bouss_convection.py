"""
Dedalus script for Boussinesq convection.

Usage:
    bouss_convection.py [options] 

Options:
    --Rayleigh=<Rayleigh>      Rayleigh number [default: 1e6]
    --Prandtl=<Prandtl>        Prandtl number = nu/kappa [default: 1]
    --nz=<nz>                  Vertical resolution [default: 128]
    --nx=<nx>                  Horizontal resolution; if not set, nx=aspect*nz_cz
    --ny=<nx>                  Horizontal resolution; if not set, nx=aspect*nz_cz
    --aspect=<aspect>          Aspect ratio of problem [default: 4]

    --fixed_f                  Fixed flux boundary conditions top/bottom
    --fixed_t                  Fixed temperature boundary conditions top/bottom
    --fixed_f_fixed_t          Fixed flux (bot) and fixed temp (top) bcs; default is no choice is made
    --fixed_t_fixed_f          Fixed temp (bot) and fixed flux (top) bcs

    --stress_free              Stress free boundary conditions top/bottom
    --no_slip                  no slip boundary conditions top/bottom; default if no choice is made

    --BC_driven                If flagged, study BC driven, not internally heated

    --3D                       Run in 3D
    --mesh=<mesh>              Processor mesh if distributing 3D run in 2D 
    
    --restart=<restart_file>   Restart from checkpoint
    --seed=<seed>              RNG seed for initial conditoins [default: 42]

    --run_time=<run_time>             Run time, in hours [default: 23.5]
    --run_time_buoy=<run_time_bouy>   Run time, in buoyancy times
    --run_time_therm=<run_time_therm> Run time, in thermal times [default: 1]

    --output_dt=<num>          Simulation time between outputs [default: 0.2]
    --overwrite                If flagged, force file mode to overwrite
    --coeffs                   If flagged, output coeffs   
    --volumes                  If flagged, output volumes   

    --label=<label>            Optional additional case name label
    --verbose                  Do verbose output (e.g., sparsity patterns of arrays)
    --no_join                  If flagged, don't join files at end of run
    --root_dir=<dir>           Root directory for output [default: ./]

"""
import logging
logger = logging.getLogger(__name__)

import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post

from logic.domains     import DedalusDomain
from logic.equations   import BoussinesqEquations2D, BoussinesqEquations3D
from logic.experiments import BoussinesqConvection
from logic.functions   import initialize_output
from tools.checkpointing import Checkpoint
checkpoint_min = 30
    
def boussinesq_convection(Rayleigh=1e6, Prandtl=1, nz=64, nx=None, ny=None, aspect=4,
                    fixed_f=False, fixed_t=False, fixed_f_fixed_t = True, fixed_t_fixed_f=False,
                    stress_free=False, no_slip=True,
                    IH=True,
                    restart=None,
                    run_time=23.5, run_time_buoyancy=None, run_time_therm=1,
                    output_dt=0.2,
                    data_dir='./', coeff_output=False, verbose=False, no_join=False,
                    threeD=False, seed=42, mesh=None, overwrite=False, volume_output=False):
    import os
    from dedalus.tools.config import config
    
    config['logging']['filename'] = os.path.join(data_dir,'logs/dedalus_log')
    config['logging']['file_level'] = 'DEBUG'
    
    import mpi4py.MPI
    if mpi4py.MPI.COMM_WORLD.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.makedirs('{:s}/'.format(data_dir))
        logdir = os.path.join(data_dir,'logs')
        if not os.path.exists(logdir):
            os.mkdir(logdir)
    logger = logging.getLogger(__name__)
    logger.info("saving run in: {}".format(data_dir))

    import time
    from dedalus import public as de
    from dedalus.extras import flow_tools
    from dedalus.tools  import post
    
    # input parameters
    logger.info("Ra = {}, Pr = {}".format(Rayleigh, Prandtl))

    # Parameters
    Lz = 1.
    Lx = aspect*Lz
    Ly = aspect*Lz
    if nx is None:  nx = int(nz*aspect)
    if ny is None:  ny = int(nz*aspect)
    if threeD:      dimensions = 3
    else:           dimensions = 2

    de_domain = DedalusDomain(nx, ny, nz, Lx, Ly, Lz, dimensions=dimensions, mesh=mesh)

    bc_dict = { 'fixed_f'              :   None,
                'fixed_t'              :   None,
                'fixed_f_fixed_t'      :   None,
                'fixed_t_fixed_f'      :   None,
                'stress_free'          :   None,
                'no_slip'              :   None }

    if fixed_f_fixed_t:
        bc_dict['fixed_f_fixed_t'] = True
    elif fixed_t_fixed_f:
        bc_dict['fixed_t_fixed_f'] = True
    elif fixed_t:
        bc_dict['fixed_t'] = True
    elif fixed_f:
        bc_dict['fixed_f'] = True
    
    
    if stress_free:
        bc_dict['stress_free'] = True
    elif no_slip:
        bc_dict['no_slip'] = True
    
    if threeD:
        logger.info("resolution: [{}x{}x{}]".format(nx, ny, nz))
        equations = BoussinesqEquations3D(de_domain)
    else:
        logger.info("resolution: [{}x{}]".format(nx, nz))
        equations = BoussinesqEquations2D(de_domain)
    equations.set_IVP()
    convection = BoussinesqConvection(equations, Rayleigh=Rayleigh, Prandtl=Prandtl, IH=IH, fixed_t=fixed_t)
    equations.set_equations()
    equations.set_BC(**bc_dict)

    # Build solver
    ts = de.timesteppers.SBDF2
    cfl_safety = 0.2
    solver = equations.problem.build_solver(ts)
    logger.info('Solver built')

    checkpoint = Checkpoint(data_dir)
    if isinstance(restart, type(None)):
        convection.set_IC(solver, de_domain, seed=seed)
        dt = None
        mode = 'overwrite'
    else:
        logger.info("restarting from {}".format(restart))
        checkpoint.restart(restart, solver)
        if overwrite:
            mode = 'overwrite'
        else:
            mode = 'append'
    checkpoint.set_checkpoint(solver, wall_dt=checkpoint_min*60, mode=mode)
        
    solver.stop_sim_time  = np.inf
    solver.stop_iteration  = np.inf
    solver.stop_wall_time = run_time*3600.
    Hermitian_cadence = 100

    # Analysis
    max_dt    = output_dt
    analysis_tasks = initialize_output(solver, de_domain, data_dir, coeff_output=coeff_output, 
                                       output_dt=output_dt, mode=mode, volumes_output=volume_output)

    # CFL
    CFL = flow_tools.CFL(solver, initial_dt=0.1, cadence=1, safety=cfl_safety,
                         max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.1)
    if threeD:
        CFL.add_velocities(('u', 'v', 'w'))
    else:
        CFL.add_velocities(('u', 'w'))

    # Flow properties
    flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
    flow.add_property("Pe", name='Pe')

    first_step = True
    start_time = time.time()
    # Main loop
    try:
        logger.info('Starting loop')
        Pe_avg = 0
        not_corrected_times = True
        init_time = solver.sim_time
        start_iter = solver.iteration
        while (solver.ok and np.isfinite(Pe_avg)):
            dt = CFL.compute_dt()
            solver.step(dt) #, trim=True)


            # Solve for blow-up over long timescales in 3D due to hermitian-ness
            effective_iter = solver.iteration - start_iter
            if threeD and effective_iter % Hermitian_cadence == 0:
                for field in solver.state.fields:
                    field.require_grid_space()

            Pe_avg = flow.grid_average('Pe')
            log_string =  'Iteration: {:5d}, '.format(solver.iteration)
            log_string += 'Time: {:8.3e} ({:8.3e} therm), dt: {:8.3e}, '.format(solver.sim_time, solver.sim_time/convection.thermal_time,  dt)
            log_string += 'Pe: {:8.3e}/{:8.3e}'.format(Pe_avg, flow.max('Pe'))
            logger.info(log_string)

            if not_corrected_times and Pe_avg > 1:
                if not isinstance(run_time_therm, type(None)):
                    solver.stop_sim_time = run_time_therm*convection.thermal_time + solver.sim_time
                elif not isinstance(run_time_buoyancy, type(None)):
                    solver.stop_sim_time  = run_time_buoyancy + solver.sim_time
                not_corrected_times = False


            
            if first_step:
                if verbose:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    ax = fig.add_subplot(1,1,1)
                    ax.spy(solver.pencils[0].L, markersize=1, markeredgewidth=0.0)
                    fig.savefig(data_dir+"sparsity_pattern.png", dpi=1200)
                    
                    import scipy.sparse.linalg as sla
                    LU = sla.splu(solver.pencils[0].LHS.tocsc(), permc_spec='NATURAL')
                    fig = plt.figure()
                    ax = fig.add_subplot(1,2,1)
                    ax.spy(LU.L.A, markersize=1, markeredgewidth=0.0)
                    ax = fig.add_subplot(1,2,2)
                    ax.spy(LU.U.A, markersize=1, markeredgewidth=0.0)
                    fig.savefig(data_dir+"sparsity_pattern_LU.png", dpi=1200)
                    
                    logger.info("{} nonzero entries in LU".format(LU.nnz))
                    logger.info("{} nonzero entries in LHS".format(solver.pencils[0].LHS.tocsc().nnz))
                    logger.info("{} fill in factor".format(LU.nnz/solver.pencils[0].LHS.tocsc().nnz))
                first_step=False
    except:
        raise
        logger.error('Exception raised, triggering end of main loop.')
    finally:
        end_time = time.time()
        main_loop_time = end_time-start_time
        n_iter_loop = solver.iteration-1
        logger.info('Iterations: {:d}'.format(n_iter_loop))
        logger.info('Sim end time: {:f}'.format(solver.sim_time))
        logger.info('Run time: {:f} sec'.format(main_loop_time))
        logger.info('Run time: {:f} cpu-hr'.format(main_loop_time/60/60*de_domain.domain.dist.comm_cart.size))
        logger.info('iter/sec: {:f} (main loop only)'.format(n_iter_loop/main_loop_time))
        try:
            final_checkpoint = Checkpoint(data_dir, checkpoint_name='final_checkpoint')
            final_checkpoint.set_checkpoint(solver, wall_dt=1, mode=mode)
            solver.step(dt) #clean this up in the future...works for now.
            post.merge_process_files(data_dir+'/final_checkpoint/', cleanup=False)
        except:
            raise
            print('cannot save final checkpoint')
        finally:
            if not no_join:
                logger.info('beginning join operation')
                post.merge_analysis(data_dir+'checkpoints')

                for key, task in analysis_tasks.items():
                    logger.info(task.base_path)
                    post.merge_analysis(task.base_path)

            logger.info(40*"=")
            logger.info('Iterations: {:d}'.format(n_iter_loop))
            logger.info('Sim end time: {:f}'.format(solver.sim_time))
            logger.info('Run time: {:f} sec'.format(main_loop_time))
            logger.info('Run time: {:f} cpu-hr'.format(main_loop_time/60/60*de_domain.domain.dist.comm_cart.size))
            logger.info('iter/sec: {:f} (main loop only)'.format(n_iter_loop/main_loop_time))

if __name__ == "__main__":
    from docopt import docopt
    args = docopt(__doc__)
    import logging
    logger = logging.getLogger(__name__)

    from numpy import inf as np_inf
    
    import sys

    fixed_f = args['--fixed_f']
    fixed_t = args['--fixed_t']
    fixed_f_fixed_t = args['--fixed_f_fixed_t']
    fixed_t_fixed_f = args['--fixed_t_fixed_f']
    if not ((fixed_f or fixed_t) or fixed_t_fixed_f):
        fixed_f_fixed_t = True


    stress_free = args['--stress_free']
    no_slip = args['--no_slip']
    if not stress_free:
        no_slip = True

    # save data in directory named after script
    data_dir = args['--root_dir'] + '/' + sys.argv[0].split('.py')[0]
    if args['--BC_driven']:
        data_dir += '_BCdriven'
    if args['--3D']:
        data_dir += '_3D'
    else:
        data_dir += '_2D'
    if fixed_f:
        data_dir += '_fixedF'
    elif fixed_f_fixed_t:
        data_dir += '_fixedF_fixedT'
    elif fixed_t_fixed_f:
        data_dir += '_fixedT_fixedF'
    else:
        data_dir += '_fixedT'

    if no_slip:
        data_dir += '_noSlip'
    else:
        data_dir += '_stressFree'


    data_dir += "_Ra{}_Pr{}_a{}".format(args['--Rayleigh'], args['--Prandtl'], args['--aspect'])
    if args['--label'] is not None:
        data_dir += "_{}".format(args['--label'])
    data_dir += '/'
    logger.info("saving run in: {}".format(data_dir))

    if args['--nx'] is not None:
        nx = int(args['--nx'])
    else:
        nx = None
    if args['--ny'] is not None:
        ny = int(args['--ny'])
    else:
        ny = None

    run_time_buoy = args['--run_time_buoy']
    if not isinstance(run_time_buoy, type(None)):
        run_time_buoy = float(run_time_buoy)
    run_time_therm = args['--run_time_therm']
    if not isinstance(run_time_therm, type(None)):
        run_time_therm = float(run_time_therm)

    mesh = args['--mesh']
    if mesh is not None:
        mesh = mesh.split(',')
        mesh = [int(mesh[0]), int(mesh[1])]
        
    boussinesq_convection(Rayleigh            = float(args['--Rayleigh']),
                          Prandtl             = float(args['--Prandtl']),
                          nz                  = int(args['--nz']),
                          nx                  = nx,
                          ny                  = ny,
                          aspect              = float(args['--aspect']),
                          fixed_f             = fixed_f, 
                          fixed_t             = fixed_t,
                          fixed_f_fixed_t     = fixed_f_fixed_t,
                          fixed_t_fixed_f     = fixed_t_fixed_f,
                          stress_free         = stress_free,
                          no_slip             = no_slip, 
                          IH                  = not(args['--BC_driven']),
                          threeD              = args['--3D'],
                          mesh                = mesh,
                          restart             = args['--restart'],
                          seed                = int(args['--seed']),
                          run_time            = float(args['--run_time']),
                          run_time_buoyancy   = run_time_buoy,
                          run_time_therm      = run_time_therm,
                          output_dt           = float(args['--output_dt']),
                          overwrite           = args['--overwrite'],
                          coeff_output        = args['--coeffs'],
                          volume_output       = args['--volumes'],
                          verbose             = args['--verbose'],
                          no_join             = args['--no_join'],
                          data_dir            = data_dir
                          )
    

