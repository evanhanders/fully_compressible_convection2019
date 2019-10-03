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

from sys import path
path.insert(0, './logic')
from logic.checkpointing import Checkpoint
from logic.domains import DedalusDomain



class DedalusProblem():
    """
    An abstract class that interacts with Dedalus to do some over-arching equation
    setup logic, etc.

    Attributes
    ----------
    de_domain    : A DedalusDomain object
        Contains information about the dedalus domain on which th eproblem is being solved.
    variables    : List of strings
        A list of strings containing the names of dedalus problem variables
    solver       : A solver object, from the Dedalus package.
        The solver for the problem.
    problem : Dedalus Problem object
        The problem that this class controls
    problem_type : string
        Specifies the type of problem being solved ('IVP', 'EVP', etc.)
    """

    def __init__(self, de_domain, variables=None):
        """Initialize the class.  Arguments are defined in class docstring.
        """
        self.de_domain      = de_domain
        self.variables      = variables
        self.problem        = None
        self.problem_type   = None
        self.solver         = None
        return

    def build_solver(self, *args):
        """ Wraps the dedalus solver class and fills the solver attribute of the DedalusProblem class """
        self.solver = self.problem.build_solver(*args)

    def build_problem(self):
        pass

    def set_variables(self, variables, **kwargs):
        """Set the variables, a list of strings, for this dedalus problem."""
        self.variables = variables
        self.build_problem(**kwargs)


class DedalusIVP(DedalusProblem):
    """
    An extension of the DedalusProblem class with some important functionality for 
    initial value problems.

    Attributes:
    -----------

    """

    def __init__(self, *args, **kwargs):
        super(DedalusIVP, self).__init__(*args, **kwargs)
        self.problem_type = 'IVP'
    
    def build_problem(self, ncc_cutoff=1e-10):
        """Constructs and initial value problem of the current object's equation set

        Arguments:
        ----------
        ncc_cutoff  : float
            The largest coefficient magnitude to keep track of when building NCCs
        """
        if self.variables is None:
            logger.error("IVP variables must be set before problem is built")
        self.problem = de.IVP(self.de_domain.domain, variables=self.variables, ncc_cutoff=ncc_cutoff)

    def set_stop_condition(self, stop_sim_time=np.inf, stop_iteration=np.inf, stop_wall_time=28800):
        """Set the conditions for when the solver should stop timestepping
        
        Parameters:
        -----------
        stop_sim_time       : float, optional
            Time at which to stop timestepping (in simulation units)
        stop_iteration      : float, optional
            Iteration number at which to stop timestepping
        stop_wall_time      : float, optional
            Wall time at which to stop timestepping (in seconds). Default: 8 hours.
        """
        self.solver.stop_sim_time  = stop_sim_time
        self.solver.stop_iteration = stop_iteration
        self.solver.stop_wall_time = stop_wall_time

    def solve_IVP(self, dt, CFL, data_dir, analysis_tasks, task_args=(), pre_loop_args=(), task_kwargs={}, pre_loop_kwargs={},
                  time_div=None, track_fields=['Pe'], threeD=False, Hermitian_cadence=100, no_join=False, mode='append'):
        """Logic for a while-loop that solves an initial value problem.

        Parameters
        ----------
        dt                  : float
            The initial timestep of the simulation
        CFL                 : a Dedalus CFL object
            A CFL object that calculates the timestep of the simulation on the fly
        data_dir            : string
            The parent directory of output files
        analysis_tasks      : OrderedDict()
            An OrderedDict of dedalus FileHandler objects
        task_args, task_kwargs : list, dict, optional
            arguments & keyword arguments to the self._special_tasks() function
        pre_loop_args, pre_loop_kwargs: list, dict, optional
            arguments & keyword arguments to the self.pre_loop_setup() function
        time_div            : float, optional
            A siulation time to divide the normal time by for easier output tracking
        threeD              : bool, optional
            If True, occasionally force the solution to grid space to remove Hermitian errors
        Hermitian_cadence   : int, optional
            The number of timesteps between grid space forcings in 3D.
        no_join             : bool, optional
            If True, do not join files at the end of the simulation run.
        mode                : string, optional
            Dedalus output mode for final checkpoint. "append" or "overwrite"
        args, kwargs        : list and dictionary
            Additional arguments and keyword arguments to be passed to the self.special_tasks() function
        """
    
        # Flow properties
        self.flow = flow_tools.GlobalFlowProperty(self.solver, cadence=1)
        for f in track_fields:
            self.flow.add_property(f, name=f)

        self.pre_loop_setup(*pre_loop_args, **pre_loop_kwargs)

        start_time = time.time()
        # Main loop
        count = 0
        try:
            logger.info('Starting loop')
            init_time = self.solver.sim_time
            start_iter = self.solver.iteration
            while (self.solver.ok):
                dt = CFL.compute_dt()
                self.solver.step(dt) #, trim=True)

                # prevents blow-up over long timescales in 3D due to hermitian-ness
                effective_iter = self.solver.iteration - start_iter
                if threeD and effective_iter % Hermitian_cadence == 0:
                    for field in self.solver.state.fields:
                        field.require_grid_space()

                self.special_tasks(*task_args, **task_kwargs)

                #reporting string
                self.iteration_report(dt, track_fields, time_div=time_div)

                if not np.isfinite(self.flow.grid_average(track_fields[0])):
                    break
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
                    post.merge_analysis(data_dir+'checkpoint')

                    for key, task in analysis_tasks.items():
                        logger.info(task.base_path)
                        post.merge_analysis(task.base_path)

                logger.info(40*"=")
                logger.info('Iterations: {:d}'.format(n_iter_loop))
                logger.info('Sim end time: {:f}'.format(self.solver.sim_time))
                logger.info('Run time: {:f} sec'.format(main_loop_time))
                logger.info('Run time: {:f} cpu-hr'.format(main_loop_time/60/60*self.de_domain.domain.dist.comm_cart.size))
                logger.info('iter/sec: {:f} (main loop only)'.format(n_iter_loop/main_loop_time))
    
    def iteration_report(self, dt, track_fields, time_div=None):
        """
        This function is called every iteration of the simulation loop and provides some text output
        to the user to tell them about the current status of the simulation. This function is meant
        to be overwritten in inherited child classes for specific use cases.

        Parameters
        ----------
        dt  : float
            The current timestep of the simulation
        track_fields : List of strings
            The fields being tracked by self.flow
        time_div            : float, optional
            A siulation time to divide the normal time by for easier output tracking
        """
        log_string =  'Iteration: {:5d}, '.format(self.solver.iteration)
        log_string += 'Time: {:8.3e}'.format(self.solver.sim_time)
        if time_div is not None:
            log_string += ' ({:8.3e})'.format(self.solver.sim_time/time_div)
        log_string += ', dt: {:8.3e}, '.format(dt)
        for f in track_fields:
            log_string += '{}: {:8.3e}/{:8.3e} '.format(f, self.flow.grid_average(f), self.flow.max(f))
        logger.info(log_string)

    def special_tasks(self, *args, **kwargs):
        """An abstract function that occurs every iteration of the simulation. Child classes
        should implement case-specific logic here. """
        pass

    def pre_loop_setup(self, *args, **kwargs):
        """An abstract function that occurs before the simulation. Child classes
        should implement case-specific logic here. """
        pass

class DedalusNLBVP(DedalusProblem):
    """
    An extension of the DedalusProblem class with some important functionality for 
    nonlinear boundary value problems.
    """

    def __init__(self, *args, **kwargs):
        super(DedalusNLBVP, self).__init__(*args, **kwargs)
        self.problem_type = 'NLBVP'
    
    def build_problem(self, ncc_cutoff=1e-10):
        """Constructs and initial value problem of the current object's equation set

        Arguments:
        ----------
        ncc_cutoff  : float
            The largest coefficient magnitude to keep track of when building NCCs
        """
        if self.variables is None:
            logger.error("BVP variables must be set before problem is built")
        self.problem = de.NLBVP(self.de_domain.domain, variables=self.variables, ncc_cutoff=ncc_cutoff)

    def solve_BVP(self, tolerance=1e-10):
        """Logic for a while-loop that solves a boundary value problem.

        Parameters
        ----------
        """
        pert = self.solver.perturbations.data
        pert.fill(1+tolerance)
        while np.sum(np.abs(pert)) > tolerance:
            self.solver.newton_iteration()
            logger.info('Perturbation norm: {}'.format(np.sum(np.abs(pert))))
