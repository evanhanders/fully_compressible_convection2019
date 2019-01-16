import numpy as np
from mpi4py import MPI
import scipy.special as scp

from collections import OrderedDict

import time
import logging
logger = logging.getLogger(__name__.split('.')[-1])

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post

try:
    from checkpointing import Checkpoint
except:
    from sys import path
    path.insert(0, './logic')
    from logic.checkpointing import Checkpoint



class DedalusProblem():
    """
    An abstract class that interacts with Dedalus to do some over-arching equation
    setup logic, etc.
    """

    def __init__(self, de_domain, variables=None):
        """Initialize the class

        Inputs:
            de_domain   -   A DedalusDomain object.
        """
        self.de_domain = de_domain
        self.variables = variables
        self.solver    = None
        return

    def build_problem(self):
        pass

    def build_solver(self):
        pass


class DedalusIVP(DedalusProblem):

    def __init__(self, *args, **kwargs):
        super(DedalusIVP, self).__init__(*args, **kwargs)
    
    def build_problem(self, *args, ncc_cutoff=1e-10, **kwargs):
        """
        Constructs and initial value problem of the current object's equation set
        """
        if self.variables is None:
            logger.error("IVP variables must be set before problem is built")
        self.problem_type = 'IVP'
        self.problem = de.IVP(self.de_domain.domain, variables=self.variables, ncc_cutoff=ncc_cutoff)

    def build_solver(self, ts=de.timesteppers.SBDF2):
        # Build solver
        self.solver = self.problem.build_solver(ts)

    def set_stop_condition(self, stop_sim_time=np.inf, stop_iteration=np.inf, stop_wall_time=28800):
        """ Default runs 8 hours (28800 seconds) """
        self.solver.stop_sim_time  = stop_sim_time
        self.solver.stop_iteration = stop_iteration
        self.solver.stop_wall_time = stop_wall_time

    def solve_IVP(self, dt, CFL, data_dir, analysis_tasks, track_fields=['Pe'], threeD=False, Hermitian_cadence=100, no_join=False, mode='append'):
        # Flow properties
        flow = flow_tools.GlobalFlowProperty(self.solver, cadence=1)
        for f in track_fields:
            flow.add_property(f, name=f)

        start_time = time.time()
        # Main loop
        try:
            logger.info('Starting loop')
            flow_avg = 0
            init_time = self.solver.sim_time
            start_iter = self.solver.iteration
            while (self.solver.ok and np.isfinite(flow_avg)):
                dt = CFL.compute_dt()
                self.solver.step(dt) #, trim=True)

                # prevents blow-up over long timescales in 3D due to hermitian-ness
                effective_iter = self.solver.iteration - start_iter
                if threeD and effective_iter % Hermitian_cadence == 0:
                    for field in self.solver.state.fields:
                        field.require_grid_space()

                #reporting string
                flow_avg = flow.grid_average(track_fields[0])
                log_string =  'Iteration: {:5d}, '.format(self.solver.iteration)
                log_string += 'Time: {:8.3e}, dt: {:8.3e}, '.format(self.solver.sim_time, dt)
                for f in track_fields:
                    log_string += '{}: {:8.3e}/{:8.3e} '.format(f, flow.grid_average(f), flow.max(f))
                logger.info(log_string)

        except:
            raise
            logger.error('Exception raised, triggering end of main loop.')
        finally:
            end_time = time.time()
            main_loop_time = end_time-start_time
            n_iter_loop = self.solver.iteration-1
            logger.info('Iterations: {:d}'.format(n_iter_loop))
            logger.info('Sim end time: {:f}'.format(self.solver.sim_time))
            logger.info('Run time: {:f} sec'.format(main_loop_time))
            logger.info('Run time: {:f} cpu-hr'.format(main_loop_time/60/60*self.de_domain.domain.dist.comm_cart.size))
            logger.info('iter/sec: {:f} (main loop only)'.format(n_iter_loop/main_loop_time))
            try:
                final_checkpoint = Checkpoint(data_dir, checkpoint_name='final_checkpoint')
                final_checkpoint.set_checkpoint(self.solver, wall_dt=1, mode=mode)
                self.solver.step(dt) #clean this up in the future...works for now.
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
                logger.info('Sim end time: {:f}'.format(self.solver.sim_time))
                logger.info('Run time: {:f} sec'.format(main_loop_time))
                logger.info('Run time: {:f} cpu-hr'.format(main_loop_time/60/60*self.de_domain.domain.dist.comm_cart.size))
                logger.info('iter/sec: {:f} (main loop only)'.format(n_iter_loop/main_loop_time))




class DedalusEVP(DedalusProblem):

    def __init__(self, *args, **kwargs):
        super(DedalusIVP, self).__init__(*args, **kwargs)

    def build_problem(self, ncc_cutoff=1e-10, tolerance=1e-10):
        """
        Constructs an eigenvalue problem of the current objeect's equation set.
        Note that dt(f) = omega * f, not i * omega * f, so real parts of omega
        are growth / shrinking nodes, imaginary parts are oscillating.
        """
        if self.variables is None:
            logger.error("IVP variables must be set before problem is built")
        self.problem_type = 'EVP'
        self.problem = de.EVP(self.de_domain.domain, variables=self.variables, eigenvalue='omega', ncc_cutoff=ncc_cutoff, tolerance=tolerance)
        self.problem.substitutions['dt(f)'] = "omega*f"

    def build_solver(self):
        """ needs to be implemented """
        pass


