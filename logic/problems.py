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
    from domains import DedalusDomain
    from equations import AEKappaMuFCE
except:
    from sys import path
    path.insert(0, './logic')
    from logic.checkpointing import Checkpoint
    from logic.domains import DedalusDomain
    from logic.equations import AEKappaMuFCE



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


class AcceleratedEvolutionIVP(DedalusIVP):
    """
    Solves an IVP using BVPs to accelerate the evolution of the IVP, as in Anders, Brown, & Oishi 2018 PRFluids.
    Must be extended for specific equation sets (boussinesq, FC, etc.), but contains some generalized overarching logic.
    """

    def pre_loop_setup(self, averager_classes, convergence_averager, root_dir, atmo_kwargs, experiment_class, experiment_args, experiment_kwargs, 
                             ae_convergence=0.01, sim_time_start=0, min_bvp_time=5, bvp_threshold=1e-2, between_ae_wait_time=None, later_bvp_time=None, min_bvp_threshold=1e-3):
        """
        Sets up AE routines before the main IVP loop.

        Arguments:
        ----------
            averager_classes :  list of FieldAverager classes
                The averagers to keep track of during AE process.
            convergence_averager : list of bools
                For each averager, if True, use that averager to check if AE is converged.
            root_dir : string
                root output directory
            atmo_kwargs : dict
                dictionary of keyword arguments used in creation of atmosphere class
            experiment_class : Experiment class
                The class type of the convection experiment being run
            experiment_args : tuple
                Arguments for experiment creation, aside from domain and atmosphere
            experiment_kwargs : dict
                Keyword arguments for experiment creation
            ae_convergence : float, optional
                Fractional convergence of T1 change at which AE stops
            sim_time_start : float, optional
                Amount of sim time to wait to start taking averages
            min_bvp_time : float, optional
                Minimum sim time over which to take averages
            bvp_threshold : float, optional
                Starting convergence threshold required for BVP profiles. Decreases with each AE iteration.
            min_bvp_threshold : float, optional
                Minimum value to reduce bvp threshold to. Threshold redues by a factor of 2 on each AE solve until it reaches this value.
            between_ae_wait_time : float, optional
                Sim time to wait after completing an AE solve before starting to measure for the next
            later_bvp_time : float, optional
                Like 'min_bvp_time' for all AE solves after the first.
        """
        self.ae_convergence = ae_convergence
        self.bvp_threshold = bvp_threshold
        self.min_bvp_threshold = min_bvp_threshold
        self.min_bvp_time = self.later_bvp_time  = min_bvp_time
        self.sim_time_start = self.sim_time_wait = sim_time_start
        self.num_ae_solves = 0
        if between_ae_wait_time is not None:
            self.sim_time_wait = between_ae_wait_time 
        if later_bvp_time is not None:
            self.later_bvp_time = later_bvp_time 
        self.doing_ae, self.finished_ae, self.Pe_switch = False, False, False
        self.averagers = []
        for conv, cl in zip(convergence_averager, averager_classes):
            self.averagers.append((conv, cl(self.solver, self.de_domain, root_dir)))

        self.scale_sim_to_ae = 1
        self.scale_ae_to_sim = 1
        self.AE_atmo   = type(self.de_domain.atmosphere)(**atmo_kwargs)
        self.AE_domain = DedalusDomain(self.AE_atmo, (self.de_domain.resolution[0],), 0, comm=MPI.COMM_SELF)
        self.AE_experiment = experiment_class(self.AE_domain, self.AE_atmo, *experiment_args[2:], **experiment_kwargs)

        if len(self.de_domain.domain.dist.mesh) == 0:
            self.z_rank = self.de_domain.domain.dist.comm_cart.rank
        else:
            self.z_rank = self.de_domain.domain.dist.comm_cart.rank % self.de_domain.domain.dist.mesh[-1] 


    def check_averager_convergence(self):
        """
        For each averager in self.averagers which is being tracked for convergence criterion,
        check if its fields have converged below the bvp threshold for AE.
        """
        if (self.solver.sim_time  - self.sim_time_start) > self.min_bvp_time:
            for conv, averager in self.averagers:
                if not conv: continue
                maxs = list()
                for k in averager.FIELDS.keys():
                    maxs.append(averager.find_global_max(averager.local_l2[k]))

                logger.info('AE: Max abs L2 norm for convergence: {:.4e} / {:.4e}'.format(np.median(maxs), self.bvp_threshold))
                if np.median(maxs) < self.bvp_threshold:
                    return True
                else:
                    return False
        
    def special_tasks(self, thermal_BC_dict):
        """
        Logic for AE performed every loop iteration

        Inputs:
        -------
        thermal_BC_dict : dictionary, optional
            A dictionary of keywords containing info about the thermal boundary conditions used in the problem.
        """
        # Don't do anything AE related if Pe < 1
        if self.flow.grid_average('Pe_rms') < 1 and not self.Pe_switch:
            return 
        elif not self.Pe_switch:
            self.sim_time_start += self.solver.sim_time
            self.Pe_switch = True

        #If first averaging iteration, reset stuff properly 
        first = False
        if not self.doing_ae and not self.finished_ae and self.solver.sim_time >= self.sim_time_start:
            for conv, averager in self.averagers:
                averager.reset_fields() #set time data properly
            self.doing_ae = True
            first = True


        if self.doing_ae:
            for conv, averager in self.averagers:
                averager.update_avgs()
            if first: return 

            do_AE = self.check_averager_convergence()
            if do_AE:
                #Get averages from global domain
                avg_fields = OrderedDict()
                for conv, averager in self.averagers:
                    for k, prof in averager.avg_profiles.items():
                        avg_fields[k] = averager.local_to_global_average(prof/averager.elapsed_avg_time)
                    averager.save_file()

                #Solve BVP
                if self.de_domain.domain.dist.comm_cart.rank == 0:
                    de_problem = DedalusNLBVP(self.AE_domain)
                    if self.num_ae_solves == 0: first_solve=True
                    else: first_solve=False
                    equations  = AEKappaMuFCE(thermal_BC_dict, avg_fields, self.scale_ae_to_sim, averager.elapsed_avg_time, self.AE_atmo, self.AE_domain, de_problem, first=first_solve)
                    de_problem.build_solver()
                    de_problem.solve_BVP()
                else:
                    de_problem = None

                # Update fields appropriately
                ae_structure = self.local_to_global_ae(de_problem)
                diff = self.update_simulation_fields(ae_structure, avg_fields)
                
                #communicate diff
                if diff < self.ae_convergence: self.finished_ae = True
                logger.info('Diff: {:.4e}, finished_ae? {}'.format(diff, self.finished_ae))
                self.doing_ae = False
                self.sim_time_start = self.solver.sim_time + self.sim_time_wait
                self.min_bvp_time = self.later_bvp_time
                
                if self.bvp_threshold/2 < self.min_bvp_threshold:
                    self.bvp_threshold = self.min_bvp_threshold
                else:
                    self.bvp_threshold /= 2
                self.num_ae_solves += 1

    def update_simulation_fields(self, de_problem):
        """ Updates simulation solver states """
        pass
    
    def local_to_global_ae(self, de_problem):
        """ Communicate results of AE solve from process 0 to all """
        pass


class FCAcceleratedEvolutionIVP(AcceleratedEvolutionIVP):
    """ 
    Extends the AcceleratedEvolutionIVP class to include a specific implementation of AE for
    the fully compressible equations. Theoretically, should work generally regardless of
    atmosphere or diffusivitiy specification.
    """

    def pre_loop_setup(self, *args, **kwargs):
        """ Extends parent class pre loop setup, keeps track of important thermo avgs """
        super(FCAcceleratedEvolutionIVP, self).pre_loop_setup(*args, **kwargs)
        self.flow.add_property("plane_avg(T1)", name='T1_avg')
        self.flow.add_property("plane_avg(T1_z)", name='T1_z_avg')
        self.flow.add_property("plane_avg(ln_rho1)", name='ln_rho1_avg')
        self.flow.add_property("plane_avg(rho_fluc)", name='rho_fluc_avg')
        self.flow.add_property("vol_avg(right(s1)-left(s1))", name='delta_s1')

    def local_to_global_ae(self, de_problem):
        """ Communicates AE solve info from process 0 to all processes """
        ae_profiles = OrderedDict()
        ae_profiles['T1'] = np.zeros(self.de_domain.resolution[0]*1)
        ae_profiles['ln_rho1'] = np.zeros(self.de_domain.resolution[0]*1)
        ae_profiles['Raf'] = np.zeros(1)
        ae_profiles['delta_s1'] = np.zeros(1)

        if self.de_domain.domain.dist.comm_cart.rank == 0:
            T1 = de_problem.solver.state['T1']
            Raf = de_problem.solver.state['Raf']
            ln_rho1 = de_problem.solver.state['ln_rho1']
            delta_s1 = de_problem.solver.state['delta_s1']
            T1.set_scales(self.scale_ae_to_sim, keep_data=True)
            ln_rho1.set_scales(self.scale_ae_to_sim, keep_data=True)
            ae_profiles['T1'] = np.copy(T1['g'])
            ae_profiles['ln_rho1'] = np.copy(ln_rho1['g'])
            ae_profiles['Raf'] = np.mean(Raf.integrate()['g'])/de_problem.problem.parameters['Lz']
            ae_profiles['delta_s1'] = np.mean(delta_s1['g'])

        for profile in ('T1', 'ln_rho1', 'Raf', 'delta_s1'):
            ae_profiles[profile] = self.de_domain.domain.dist.comm_cart.bcast(ae_profiles[profile], root=0)
        return ae_profiles
        
    def update_simulation_fields(self, ae_profiles, avg_fields):
        """ Updates T1, T1_z, ln_rho1 with AE profiles """

        z_slices = self.de_domain.domain.dist.grid_layout.slices(scales=1)[-1]

        u_scaling = ae_profiles['Raf']**(1./2)
        thermo_scaling = u_scaling

        #Calculate instantaneous thermo profiles
        [self.flow.properties[f].set_scales(1, keep_data=True) for f in ('T1_avg', 'ln_rho1_avg', 'rho_fluc_avg', 'delta_s1')]
        T1_prof = self.flow.properties['T1_avg']['g']
        ln_rho_prof = self.flow.properties['ln_rho1_avg']['g']
        rho_fluc_prof = self.flow.properties['rho_fluc_avg']['g']
        old_delta_s1 = np.mean(self.flow.properties['delta_s1']['g'])
        new_delta_s1 = ae_profiles['delta_s1']

        T1 = self.solver.state['T1']
        T1_z = self.solver.state['T1_z']
        ln_rho1 = self.solver.state['ln_rho1']

        for k in ae_profiles.keys():
            print(k, ae_profiles[k])

        #Adjust Temp
        T1.set_scales(1, keep_data=True)
        T1['g'] -= T1_prof
        T1.set_scales(1, keep_data=True)
        T1['g'] *= thermo_scaling
        T1.set_scales(1, keep_data=True)
        T1['g'] += ae_profiles['T1'][z_slices]
        T1.set_scales(1, keep_data=True)
        T1.differentiate('z', out=self.solver.state['T1_z'])

        #Adjust lnrho
        self.AE_atmo.atmo_fields['rho0'].set_scales(1, keep_data=True)
        rho0 = self.AE_atmo.atmo_fields['rho0']['g'][z_slices]
        ln_rho1.set_scales(1, keep_data=True)
        rho_full_flucs = rho0*(np.exp(ln_rho1['g']) - 1) - rho_fluc_prof
        rho_full_flucs *= thermo_scaling
        rho_full_new = rho_full_flucs + rho0*np.exp(ae_profiles['ln_rho1'])[z_slices]
        ln_rho1.set_scales(1, keep_data=True)
        ln_rho1['g']  = np.log(rho_full_new/rho0)
        ln_rho1.set_scales(1, keep_data=True)

        #Adjust velocity
        vel_fields = ['u', 'w']
        if self.de_domain.dimensions == 3:
            vel_fields.append('v')
        for k in vel_fields:
            self.solver.state[k].set_scales(1, keep_data=True)
            self.solver.state[k]['g'] *= u_scaling
            self.solver.state[k].differentiate('z', out=self.solver.state['{:s}_z'.format(k)])
            self.solver.state[k].set_scales(1, keep_data=True)
            self.solver.state['{:s}_z'.format(k)].set_scales(1, keep_data=True)

        #See how much delta S over domain has changed.
        diff = np.mean(np.abs(1 - new_delta_s1/old_delta_s1))
        return diff
        
