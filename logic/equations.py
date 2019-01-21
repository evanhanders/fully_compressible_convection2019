import numpy as np
from mpi4py import MPI
import scipy.special as scp

from collections import OrderedDict

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from dedalus import public as de

try:
    from functions import global_noise
except:
    from sys import path
    path.insert(0, './logic')
    from logic.functions import global_noise



class Equations():
    """
    An abstract class that interacts with Dedalus to do some over-arching equation
    setup logic, etc.
    """

    def __init__(self, de_domain, de_problem):
        """Initialize the class

        Parameters
        ----------
        de_domain   : DedalusDomain object
            Contains info regarding domain on which equations will be solved
        de_problem  : DedalusProblem object
            Contains info regarding problem on which equations will be solved
        """
        self.de_domain  = de_domain
        self.de_problem = de_problem
        return


class FullyCompressibleEquations(Equations):
    """
    An abstract class containing the fully compressible equations which must
    be extended to specify the type of diffusivities
    """
    def __init__(self, *args):
        super(FullyCompressibleEquations, self).__init__(*args)

        variables = ['u','u_z','v', 'v_z', 'w','w_z','T1', 'T1_z', 'ln_rho1']
        if not self.de_domain.dimensions == 3:
            variables.remove('v')
            variables.remove('v_z')
        self.de_problem.set_variables(variables)

    def _set_subs(self, kx = 0):
        if self.de_domain.dimensions == 1:
            self.de_problem.problem.parameters['j'] = 1j
            self.de_problem.problem.substitutions['dx(f)'] = "j*kx*(f)"
            self.de_problem.problem.parameters['kx'] = kx
        if not self.de_domain.dimensions == 3:
            self.de_problem.problem.substitutions['v'] = '0'
            self.de_problem.problem.substitutions['v_z'] = '0'
            self.de_problem.problem.substitutions['dy(A)'] = '0*A'

        self.de_problem.problem.substitutions['vel_rms']  = 'sqrt(u**2 + v**2 + w**2)'

        self.de_problem.problem.substitutions['Lap(f, f_z)'] = "(dx(dx(f)) + dy(dy(f)) + dz(f_z))"
        self.de_problem.problem.substitutions['Div(fx, fy, fz_z)'] = "(dx(fx) + dy(fy) + fz_z)"
        self.de_problem.problem.substitutions['Div_u'] = "Div(u, v, w_z)"
        self.de_problem.problem.substitutions['UdotGrad(f, f_z)'] = "(u*dx(f) + v*dy(f) + w*(f_z))"

        self.de_problem.problem.substitutions["Sig_xx"] = "(2*dx(u) - 2/3*Div_u)"
        self.de_problem.problem.substitutions["Sig_yy"] = "(2*dy(v) - 2/3*Div_u)"
        self.de_problem.problem.substitutions["Sig_zz"] = "(2*w_z   - 2/3*Div_u)"
        self.de_problem.problem.substitutions["Sig_xy"] = "(dx(v) + dy(u))"
        self.de_problem.problem.substitutions["Sig_xz"] = "(dx(w) +  u_z )"
        self.de_problem.problem.substitutions["Sig_yz"] = "(dy(w) +  v_z )"

        self.de_problem.problem.substitutions['Vort_x'] = '(dy(w) - v_z)'
        self.de_problem.problem.substitutions['Vort_y'] = '( u_z  - dx(w))'
        self.de_problem.problem.substitutions['Vort_z'] = '(dx(v) - dy(u))'
        self.de_problem.problem.substitutions['enstrophy']   = '(Vort_x**2 + Vort_y**2 + Vort_z**2)'


    def set_BC(self,
               fixed_f=None, fixed_t=None, fixed_f_fixed_t=None, fixed_t_fixed_f=None,
               stress_free=None, no_slip=None):
        """
        Sets the velocity and thermal boundary conditions at the upper and lower boundaries.  
        Choose one thermal type of BC and one velocity type of BC to set those conditions.  
        See set_thermal_BC() and set_velocity_BC() functions for more info.

        Parameters
        ----------
        fixed_f         : bool, optional
            If flagged, use fixed-flux thermal boundary conditions
        fixed_t         : bool, optional
            If flagged, use fixed-temperature thermal boundary conditions
        fixed_f_fixed_t : bool, optional
            If flagged, use fixed-flux (bottom) and fixed-temperature (top) thermal BCs
        fixed_t_fixed_f : bool, optional
            If flagged, use fixed-temperature (bottom) and fixed-flux (top) thermal BCs
        stress_free     : bool, optional
            If flagged, use stress-free dynamical boundary conditions
        no_slip         : bool, optional
            If flagged, use no-slip dynamical boundary conditions.
        """
        self.dirichlet_set = []
        self.set_thermal_BC(fixed_f=fixed_f, fixed_t=fixed_t,
                            fixed_f_fixed_t=fixed_f_fixed_t, fixed_t_fixed_f=fixed_t_fixed_f)
        self.set_velocity_BC(stress_free=stress_free, no_slip=no_slip)
        for key in self.dirichlet_set:
            self.de_problem.problem.meta[key]['z']['dirichlet'] = True
            
    def set_thermal_BC(self, fixed_f=None, fixed_t=None, fixed_f_fixed_t=None, fixed_t_fixed_f=None):
        """
        Sets the thermal boundary conditions at the top and bottom of the atmosphere.  

        Parameters
        ----------
        fixed_f         : bool, optional
            If flagged, T1_z = 0 at top and bottom
        fixed_t         : bool, optional
            If flagged, T1   = 0 at top and bottom
        fixed_f_fixed_t : bool, optional
            If flagged, T1_z = 0 at bottom and T1 = 0 at top [DEFAULT]
        fixed_t_fixed_f : bool, optional
            If flagged, T1 = 0 at bottom and T1_z = 0 at top
        """
        if not(fixed_f) and not(fixed_t) and not(fixed_t_fixed_f) and not(fixed_f_fixed_t):
            fixed_f_fixed_t = True

        if fixed_f:
            logger.info("Thermal BC: fixed flux (full form)")
            self.de_problem.problem.add_bc("fixed_flux_L_LHS = fixed_L_flux_RHS")
            self.de_problem.problem.add_bc("fixed_flux_R_LHS = fixed_R_flux_RHS")
            self.dirichlet_set.append('T1_z')
        elif fixed_t:
            logger.info("Thermal BC: fixed temperature (T1)")
            self.de_problem.problem.add_bc( "left(T1) = 0")
            self.de_problem.problem.add_bc("right(T1) = 0")
            self.dirichlet_set.append('T1')
        elif fixed_f_fixed_t:
            logger.info("Thermal BC: fixed flux/fixed temperature")
            self.de_problem.problem.add_bc("fixed_flux_L_LHS = fixed_flux_L_RHS")
            self.de_problem.problem.add_bc("right(T1)  = 0")
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('T1')
        elif fixed_t_fixed_f:
            logger.info("Thermal BC: fixed temperature/fixed flux")
            self.de_problem.problem.add_bc("left(T1)    = 0")
            self.de_problem.problem.add_bc("fixed_flux_R_LHS = fixed_flux_R_RHS")
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('T1')
        else:
            logger.error("Incorrect thermal boundary conditions specified")
            raise

    def set_velocity_BC(self, stress_free=None, no_slip=None):
        """
        Sets the velocity boundary conditions at the top and bottom of the atmosphere.  
        Boundaries are additionally impenetrable (w = 0 at top and bottom)

        Parameters
        ----------
        stress_free     : bool, optional
            If flagged, dz(horizontal velocity) is zero at the top and bottom
        no_slip         : bool, optional
            If flagged, velocity is zero at the top and bottom [DEFAULT]
        """

        if not(stress_free) and not(no_slip):
            stress_free = True
            
        # horizontal velocity boundary conditions
        if stress_free:
            logger.info("Horizontal velocity BC: stress free")
            self.de_problem.problem.add_bc("left(u_z) = 0")
            self.de_problem.problem.add_bc("right(u_z) = 0")
            self.dirichlet_set.append('u_z')
            if self.de_domain.dimensions == 3:
                self.de_problem.problem.add_bc("left(v_z) = 0")
                self.de_problem.problem.add_bc("right(v_z) = 0")
                self.dirichlet_set.append('v_z')
        elif no_slip:
            logger.info("Horizontal velocity BC: no slip")
            self.de_problem.problem.add_bc( "left(u) = 0")
            self.de_problem.problem.add_bc("right(u) = 0")
            self.dirichlet_set.append('u')
            if self.de_domain.dimensions == 3:
                self.de_problem.problem.add_bc( "left(v) = 0")
                self.de_problem.problem.add_bc("right(v) = 0")
                self.dirichlet_set.append('v')
        else:
            logger.error("Incorrect horizontal velocity boundary conditions specified")
            raise

        # vertical velocity boundary conditions
        logger.info("Vertical velocity BC: impenetrable")
        self.de_problem.problem.add_bc( "left(w) = 0")
        self.de_problem.problem.add_bc("right(w) = 0")
        self.dirichlet_set.append('w')

    def set_IC(self, noise_scale, checkpoint, A0=1e-6, restart=None, checkpoint_dt=1800, overwrite=True, **kwargs):
        """
        Set initial conditions as random noise in the temperature perturbations, tapered to
        zero at the boundaries.  

        Parameters
        ----------
        noise_scale     : NumPy array, size matches local dealiased z-grid.
            Scales the noise so that it is O(A0) compared to fluctuations. (See A0 below)
        checkpoint      : A Checkpoint object
            The checkpointing object of the current simulations
        A0              : Float, optional
            The size of the perturbation. Generally should be very small.
        restart         : String, optional
            If not None, the path to the checkpoint file to restart the simulation from.
        checkpoint_dt   : Int, optional
            The amount of wall time, in seconds, between checkpoints (default 30 min)
        overwrite       : Bool, optional
            If True, auto-set the file mode to overwrite, even if checkpoint-restarting
        kwargs          : Dict, optional
            Additional keyword arguments for the global_noise() function

        """
        if restart is None:
            # initial conditions
            T_IC = self.de_problem.solver.state['T1']
            T_z_IC = self.de_problem.solver.state['T1_z']
                
            noise = global_noise(self.de_domain, **kwargs)
            noise.set_scales(self.de_domain.dealias, keep_data=True)
            T_IC.set_scales(self.de_domain.dealias, keep_data=True)
            T_IC['g'] = A0*noise_scale*np.sin(np.pi*self.de_domain.z_de/self.de_domain.Lz)*noise['g']
            T_IC.differentiate('z', out=T_z_IC)
            logger.info("Starting with T1 perturbations of amplitude A0 = {:g}".format(A0))
            dt = None
            mode = 'overwrite'
        else:
            logger.info("restarting from {}".format(restart))
            dt = checkpoint.restart(restart, self.de_problem.solver)
            if overwrite:
                mode = 'overwrite'
            else:
                mode = 'append'
        checkpoint.set_checkpoint(self.de_problem.solver, wall_dt=checkpoint_dt, mode=mode)
        return dt, mode
 


    def set_equations(self, atmosphere, *args, kx = 0, **kwargs):
        ''' 
        Sets the fully compressible equations of in a ln_rho / T formulation. These
        equations take the form:
        
        D ln ρ + ∇ · u = 0
        D u = - ∇ T - T∇ ln ρ - gẑ + (1/ρ) * ∇ · Π
        D T + (γ - 1)T∇ · u - (1/[ρ Cv]) ∇ · (- Kap∇ T) = (1/[ρ Cv])(Π ·∇ )·u 
        
        Where

        D = ∂/∂t + (u · ∇ ) 

        and

        Π = - Mu (∂u_i/∂x_j + ∂u_j/∂x_i - (2/3)D_{ij}∇ · u)

        is the viscous stress tensor. The variables are u (vector velocity), T (temp) and 
        ρ (density). Temperature, density, and pressure are related through an ideal gas
        equation of state,

        P = ρT

        Which has already been assumed in the formulation of these equations.
        '''
        self._set_subs(kx = kx)
        self._setup_diffusivities(atmosphere, *args, **kwargs)
        atmosphere.set_output_subs()

        self.de_problem.problem.add_equation(    "dz(u) - u_z = 0")
        if self.de_domain.dimensions == 3:
            self.de_problem.problem.add_equation("dz(v) - v_z = 0")
        self.de_problem.problem.add_equation(    "dz(w) - w_z = 0")
        self.de_problem.problem.add_equation(    "dz(T1) - T1_z = 0")
        self.de_problem.problem.add_equation((    "(scale_c)*( dt(ln_rho1)   + w*ln_rho0_z + Div_u ) = (scale_c)*(-UdotGrad(ln_rho1, dz(ln_rho1)))"))
        self.de_problem.problem.add_equation(    ("(scale_m_z)*( dt(w) + T1_z     + T0*dz(ln_rho1) + T1*ln_rho0_z - L_visc_w) = "
                                       "(scale_m_z)*(- UdotGrad(w, w_z) - T1*dz(ln_rho1) + R_visc_w)"))
        self.de_problem.problem.add_equation(    ("(scale_m)*( dt(u) + dx(T1)   + T0*dx(ln_rho1)                  - L_visc_u) = "
                                       "(scale_m)*(-UdotGrad(u, u_z) - T1*dx(ln_rho1) + R_visc_u)"))
        if self.de_domain.dimensions == 3:
            self.de_problem.problem.add_equation(("(scale_m)*( dt(v) + dy(T1)   + T0*dy(ln_rho1)                  - L_visc_v) = "
                                       "(scale_m)*(-UdotGrad(v, v_z) - T1*dy(ln_rho1) + R_visc_v)"))
        self.de_problem.problem.add_equation((    "(scale_e)*( dt(T1)   + w*T0_z  + (gamma-1)*T0*Div_u -  L_thermal) = "
                                       "(scale_e)*(-UdotGrad(T1, T1_z) - (gamma-1)*T1*Div_u + R_thermal + R_visc_heat + source_terms)"))


    def _set_diffusion_subs(self):
        pass


class KappaMuFCE(FullyCompressibleEquations):
    '''
    An extension of the fully compressible equations where the diffusivities are
    set based on kappa and mu, not chi and nu.
    '''

    def __init__(self, *args, **kwargs):
        super(KappaMuFCE, self).__init__(*args, **kwargs)

    def _setup_diffusivities(self, atmosphere, max_ncc=2):
        self.de_problem.problem.substitutions['kappa_full']   = '(kappa0)'
        self.de_problem.problem.substitutions['kappa_full_z'] = '(0)'
        self.de_problem.problem.substitutions['kappa_fluc']   = '(0)'
        self.de_problem.problem.substitutions['chi_full']     = '(kappa_full/rho_full)'
        self.de_problem.problem.substitutions['chi_fluc']     = '(chi_full - chi0)'
        self.de_problem.problem.substitutions['mu_full']      = '(mu0)'
        self.de_problem.problem.substitutions['mu_full_z']    = '(0)'
        self.de_problem.problem.substitutions['mu_fluc']      = '(0)'
        self.de_problem.problem.substitutions['nu_full']      = '(mu_full/rho_full)'
        self.de_problem.problem.substitutions['nu_fluc']      = '(nu_full - nu0)'

        rho_profile = self.de_domain.generate_vertical_profile(atmosphere.atmo_fields['rho0'])
        z_profile = self.de_domain.generate_vertical_profile(self.de_domain.z)
        fit = np.polyfit(z_profile, rho_profile, max_ncc)
        fit_str = ''
        for i, f in enumerate(fit):
            fit_str += '(({:.8f})*z**{}) + '.format(f, max_ncc-i-1)
        fit_str = '({:s})'.format(fit_str[:-2])
        self.de_problem.problem.substitutions['rho0_fit']   = fit_str

        self.de_problem.problem.substitutions['scale_m_z']  = '(rho0_fit)'
        self.de_problem.problem.substitutions['scale_m']    = '(rho0_fit)'
        self.de_problem.problem.substitutions['scale_e']    = '(rho0_fit)'
        self.de_problem.problem.substitutions['scale_c']    = '(T0)'


        #Viscous subs -- momentum equation     
        self.de_problem.problem.substitutions['visc_u']   = "( (mu_full)*(Lap(u, u_z) + 1/3*Div(dx(u), dx(v), dx(w_z))) + (mu_full_z)*(Sig_xz))"
        self.de_problem.problem.substitutions['visc_v']   = "( (mu_full)*(Lap(v, v_z) + 1/3*Div(dy(u), dy(v), dy(w_z))) + (mu_full_z)*(Sig_yz))"
        self.de_problem.problem.substitutions['visc_w']   = "( (mu_full)*(Lap(w, w_z) + 1/3*Div(  u_z, dz(v), dz(w_z))) + (mu_full_z)*(Sig_zz))"                
        self.de_problem.problem.substitutions['L_visc_u'] = "(visc_u/rho0_fit)"
        self.de_problem.problem.substitutions['L_visc_v'] = "(visc_v/rho0_fit)"
        self.de_problem.problem.substitutions['L_visc_w'] = "(visc_w/rho0_fit)"                
        self.de_problem.problem.substitutions['R_visc_u'] = "(visc_u/rho_full - (L_visc_u))"
        self.de_problem.problem.substitutions['R_visc_v'] = "(visc_v/rho_full - (L_visc_v))"
        self.de_problem.problem.substitutions['R_visc_w'] = "(visc_w/rho_full - (L_visc_w))"

        self.de_problem.problem.substitutions['thermal'] = ('( ((1/Cv))*(kappa_full*Lap(T1, T1_z) + kappa_full_z*T1_z) )')
        self.de_problem.problem.substitutions['L_thermal'] = ('thermal/rho0_fit')
        self.de_problem.problem.substitutions['R_thermal'] = ('( thermal/rho_full - (L_thermal) + ((1/Cv)/(rho_full))*(kappa_full*T0_zz + kappa_full_z*T0_z) )' )
        self.de_problem.problem.substitutions['source_terms'] = '0'
        #Viscous heating
        self.de_problem.problem.substitutions['R_visc_heat']  = " (mu_full/rho_full*(1/Cv))*(dx(u)*Sig_xx + dy(v)*Sig_yy + w_z*Sig_zz + Sig_xy**2 + Sig_xz**2 + Sig_yz**2)"

        #Fixed-flux BC. LHS = RHS at boundary. L = left (lower). R = right (upper)
        self.de_problem.problem.substitutions['fixed_flux_L_LHS'] = "left(T1_z)"
        self.de_problem.problem.substitutions['fixed_flux_L_RHS'] = "(0)"
        self.de_problem.problem.substitutions['fixed_flux_R_LHS'] = "right(T1_z)"
        self.de_problem.problem.substitutions['fixed_flux_R_RHS'] = "(0)"
