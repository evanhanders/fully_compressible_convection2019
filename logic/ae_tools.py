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
from logic.domains import DedalusDomain
from logic.equations import Equations
from logic.problems import DedalusIVP, DedalusNLBVP

class AEKappaMuFCE(Equations):
    """
    Accelerated Evolution equations in a Kappa/Mu formulation.
    """

    def __init__(self, thermal_BC_dict, avg_field_dict, scale_ae_to_sim, elapsed_time, *args, ncc_cutoff=1e-10, first=False, kx=0, ky=0):
        super(AEKappaMuFCE, self).__init__(*args)
        variables = ['T1', 'T1_z', 'ln_rho1', 'M1', 'delta_s1', 'Raf']

        self.de_problem.set_variables(variables, ncc_cutoff=ncc_cutoff)
        self._set_parameters(avg_field_dict, scale_ae_to_sim, elapsed_time, first=first)
        self._set_equations()
        self._set_BCs(thermal_BC_dict)
    
    def _set_equations(self):
        """ 
        Sets the horizontally-averaged, stationary FC equations.
        
        Four equations are used in the FC BVP:
        (1) A definition of the entropy jump across the domain, used for convergence.
        (2) A simple definition of T1_z
        (3) A mass-accounting equation
        (4) Modified hydrostatic equilibrium
        (5) Temperature profile forcing
        (6) Some sort of measure of the ratio of the old effective Ra to the new effective Ra.
        

        In the last equation, we use the FULL energy equation, and only ensure
        that the divergence of the vertical fluxes are balanced (which is to say,
        there's no divergence of vertical flux unless there's an explicit internal
        heating)

        Four BCs must be used with these equations: two on integrated mass (which
        constrain ln_rho1), and two on the temperature (the same as are used
        in the IVP.
        """
        self.de_problem.problem.substitutions['AE_rho_full'] = '(rho0* exp(ln_rho1))'
        self.de_problem.problem.substitutions['AE_rho_fluc'] = '(rho0*(exp(ln_rho1) - 1))'
        self.de_problem.problem.substitutions['s1'] = '(Cv*log(1+T1/T0) - R*ln_rho1)'
        self.de_problem.problem.substitutions['exp_timestep'] = '(right(T1_z_in) - left(T1_z_in))/(left(F_Tz) - right(F_Tz))' #Need to fix this to specialize it for different flux BCs...might already do that OK

        self.de_problem.problem.add_equation("delta_s1 = right(s1) - left(s1)")
        self.de_problem.problem.add_equation("dz(T1) - T1_z = 0")
        self.de_problem.problem.add_equation("dz(M1) = AE_rho_fluc")
        self.de_problem.problem.add_equation("T1_z + T1*dz(ln_rho0) + T0*dz(ln_rho1) = -T1 * dz(ln_rho1) - ( Raf**2 )*udotgradW + Raf*visc_forcing")
        self.de_problem.problem.add_equation("dz(T1_z) = T1_zz_in + F_Tzz*exp_timestep") #perhaps the dt here should be defined as the timestep necessary to make left flux = right flux.
#       self.de_problem.problem.add_equation("exp_timestep*(left(F_Tz) - right(F_Tz)) = right(T1_z_in) - left(T1_z_in)")
        self.de_problem.problem.add_equation("Raf = sqrt(((integ(T1_z + T0_z - T_ad_z)) / (integ(T1_z_in + T0_z - T_ad_z)))**2)")
        
    def _set_BCs(self, thermal_BC_dict):
        """ 
        Sets thermal and mass-conserving boundary conditions for the BVP. By setting
        the integrated mass fluctuation's value to be 0 at the top and bottom, we ensure
        that no mass enters or leaves the domain in the process of the BVP solve.
        """

        if thermal_BC_dict['flux']:
            raise NotImplementedError("BVP method not implemented for fixed flux BCs")
        elif thermal_BC_dict['temp']:
            raise NotImplementedError("BVP method not implemented for fixed temp BCs")
        elif thermal_BC_dict['temp_flux']:
            self.de_problem.problem.add_bc('left(T1) = 0')
            self.de_problem.problem.add_bc('right(T1_z) = 0')
        elif thermal_BC_dict['flux_temp']:
            self.de_problem.problem.add_bc('left(T1_z) = 0')
            self.de_problem.problem.add_bc('right(T1) = 0')
        self.de_problem.problem.add_bc('right(M1) = 0')
        self.de_problem.problem.add_bc('left(M1) = 0')

    def _set_parameters(self, field_dict, scale_ae_to_sim, elapsed_time, f=1./3, first=False):

        dt  = -(np.exp(-f) - 1)*self.atmosphere.atmo_params['t_therm']*np.exp(-elapsed_time/2/self.atmosphere.atmo_params['t_therm']) #f is fraction of thermal time
#        self.de_problem.problem.parameters['exp_timestep'] = dt
        for k in ['T1_in', 'T1_z_in', 'T1_zz_in', 's1_in', 's1_z_in', 'udotgradW', 'visc_forcing', 'F_Tz', 'F_Tzz']:
            this_field = self.de_domain.new_ncc()
            this_field.set_scales(scale_ae_to_sim, keep_data=False)
            this_field['g'] = field_dict[k]
#            if first is False and k == 'F_Tzz': continue
#            if first is True and k == 'F_tzz_full': continue
#            elif k == 'F_Tzz_full': 
#                self.de_problem.problem.parameters['F_Tzz'] = this_field 
#                continue
            self.de_problem.problem.parameters[k] = this_field 

        for k, fd in self.atmosphere.atmo_fields.items():
            self.de_problem.problem.parameters[k] = fd
            fd.set_scales(1, keep_data=True)

        for k, p in self.atmosphere.atmo_params.items():
            self.de_problem.problem.parameters[k] = p





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
        
