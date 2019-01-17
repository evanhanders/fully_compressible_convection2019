"""
Dedalus script for Boussinesq convection.

Usage:
    bouss_convection.py [options] 

Options:
    --Rayleigh=<Rayleigh>      Rayleigh number [default: 1e6]
    --Prandtl=<Prandtl>        Prandtl number = nu/kappa [default: 1]
    --nz=<nz>                  Vertical resolution [default: 64]
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
    --z_bot=<z>                If z_bot < 0, there is a stable layer [default: -1]

    --3D                       Run in 3D
    --mesh=<mesh>              Processor mesh if distributing 3D run in 2D 
    
    --restart=<restart_file>   Restart from checkpoint
    --seed=<seed>              RNG seed for initial conditoins [default: 42]

    --run_time=<t>             Run time, in hours [default: 23.5]
    --run_time_buoy=<t>        Run time, in buoyancy times
    --run_time_therm=<t>       Run time, in thermal times [default: 1]

    --output_dt=<num>          Simulation time between outputs [default: 0.2]
    --overwrite                If flagged, force file mode to overwrite
    --coeffs                   If flagged, output coeffs   
    --volumes                  If flagged, output volumes   

    --label=<label>            Optional additional case name label
    --no_join                  If flagged, don't join files at end of run
    --root_dir=<dir>           Root directory for output [default: ./]

"""
import logging
logger = logging.getLogger(__name__)

import numpy as np

from logic.domains       import DedalusDomain
from logic.problems      import DedalusIVP
from logic.equations     import BoussinesqEquations2D, BoussinesqEquations3D
from logic.experiments   import BoussinesqConvection
from logic.outputs       import initialize_output
from logic.functions     import mpi_makedirs
from logic.checkpointing import Checkpoint
checkpoint_min = 30
    
def boussinesq_convection(Rayleigh=1e6, Prandtl=1, nz=64, nx=None, ny=None, aspect=4,
                    fixed_f=False, fixed_t=False, fixed_f_fixed_t = True, fixed_t_fixed_f=False,
                    stress_free=False, no_slip=True,
                    IH=True, z_bot=-1,
                    restart=None,
                    run_time=23.5, run_time_buoyancy=None, run_time_therm=1,
                    output_dt=0.2,
                    data_dir='./', coeff_output=False, no_join=False,
                    threeD=False, seed=42, mesh=None, overwrite=False, volume_output=False):
    import os
    from dedalus.tools.config import config
    
    config['logging']['filename'] = os.path.join(data_dir,'logs/dedalus_log')
    config['logging']['file_level'] = 'DEBUG'

    mpi_makedirs(data_dir)
    logdir = os.path.join(data_dir,'logs')
    mpi_makedirs(logdir)
    
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
   

    de_domain = DedalusDomain(nx, ny, nz, Lx, Ly, Lz, z_bot=z_bot, dimensions=dimensions, mesh=mesh)
    de_problem = DedalusIVP(de_domain)
    if threeD:
        logger.info("resolution: [{}x{}x{}]".format(nx, ny, nz))
        equations = BoussinesqEquations3D(de_domain, de_problem)
    else:
        logger.info("resolution: [{}x{}]".format(nx, nz))
        equations = BoussinesqEquations2D(de_domain, de_problem)
    de_problem.build_problem()
    convection = BoussinesqConvection(de_domain, de_problem, Rayleigh=Rayleigh, Prandtl=Prandtl, IH=IH)
    equations.set_equations()
    equations.set_BC(**bc_dict)

    # Build solver
    de_problem.build_solver(ts = de.timesteppers.SBDF2)


    stop_sim_time = run_time_therm*convection.thermal_time
    if not(run_time_buoyancy is None): stop_sim_time = run_time_buoyancy
    de_problem.set_stop_condition(stop_wall_time=run_time*3600, stop_sim_time=stop_sim_time)

    checkpoint = Checkpoint(data_dir)
    convection.T0.set_scales(de_domain.dealias, keep_data=True)
    noise_scale = convection.T0['g'] * convection.P
    dt, mode =     equations.set_IC(noise_scale, checkpoint, restart=restart, seed=seed, checkpoint_dt=checkpoint_min*60, overwrite=overwrite)
   
    if dt is None:
        dt = max_dt    = output_dt

    # Analysis
    analysis_tasks = initialize_output(de_domain, de_problem, data_dir, coeff_output=coeff_output, 
                                       output_dt=output_dt, mode=mode, volumes_output=volume_output)

    # CFL
    cfl_safety = 0.2
    CFL = flow_tools.CFL(de_problem.solver, initial_dt=dt, cadence=1, safety=cfl_safety,
                         max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.1)
    if threeD:
        CFL.add_velocities(('u', 'v', 'w'))
    else:
        CFL.add_velocities(('u', 'w'))
    
    de_problem.solve_IVP(dt, CFL, data_dir, analysis_tasks, track_fields=['Pe'], threeD=threeD, Hermitian_cadence=100, no_join=no_join, mode=mode)


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
    if args['--3D']:
        data_dir += '_3D'
    else:
        data_dir += '_2D'
    if args['--BC_driven']:
        data_dir += '_BCdriven'
    data_dir += '_zb{:.2g}'.format(float(args['--z_bot']))
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

    print(args)

    run_time_buoy = args['--run_time_buoy']
    if run_time_buoy is not None:
        run_time_buoy = float(run_time_buoy)
    run_time_therm = args['--run_time_therm']
    if run_time_therm is not None:
        run_time_therm = float(run_time_therm)

    logger.info("{}, {}".format(run_time_buoy, run_time_therm))

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
                          z_bot               = float(args['--z_bot']),
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
                          no_join             = args['--no_join'],
                          data_dir            = data_dir
                          )
    

