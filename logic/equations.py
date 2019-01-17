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
    

class BoussinesqEquations(Equations):
    """
    An extension of the Equations class which contains the full 2D form of the boussinesq
    equations.   
    """
    def __init__(self, *args, **kwargs):
        super(BoussinesqEquations, self).__init__(*args, **kwargs)
       
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
            self.de_problem.problem.add_bc( "left(T1_z) = 0")
            self.de_problem.problem.add_bc("right(T1_z) = 0")
            self.dirichlet_set.append('T1_z')
        elif fixed_t:
            logger.info("Thermal BC: fixed temperature (T1)")
            self.de_problem.problem.add_bc( "left(T1) = 0")
            self.de_problem.problem.add_bc("right(T1) = 0")
            self.dirichlet_set.append('T1')
        elif fixed_f_fixed_t:
            logger.info("Thermal BC: fixed flux/fixed temperature")
            self.de_problem.problem.add_bc("left(T1_z) = 0")
            self.de_problem.problem.add_bc("right(T1)  = 0")
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('T1')
        elif fixed_t_fixed_f:
            logger.info("Thermal BC: fixed temperature/fixed flux")
            logger.info("warning; these are not fully correct fixed flux conditions yet")
            self.de_problem.problem.add_bc("left(T1)    = 0")
            self.de_problem.problem.add_bc("right(T1_z) = 0")
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
            If flagged, Horizontal vorticity is zero at the top and bottom
        no_slip         : bool, optional
            If flagged, velocity is zero at the top and bottom [DEFAULT]
        """

        if not(stress_free) and not(no_slip):
            stress_free = True
            
        # horizontal velocity boundary conditions
        if stress_free:
            logger.info("Horizontal velocity BC: stress free")
            self.de_problem.problem.add_bc("left(Oy) = 0")
            self.de_problem.problem.add_bc("right(Oy) = 0")
            self.dirichlet_set.append('Oy')
            if self.de_domain.dimensions == 3:
                self.de_problem.problem.add_bc("left(Ox) = 0")
                self.de_problem.problem.add_bc("right(Ox) = 0")
                self.dirichlet_set.append('Ox')
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
        if self.de_domain.dimensions == 2:
            self.de_problem.problem.add_bc("right(p) = 0", condition="(nx == 0)")
            self.de_problem.problem.add_bc("right(w) = 0", condition="(nx != 0)")
        elif self.de_domain.dimensions == 3:
            self.de_problem.problem.add_bc("right(p) = 0", condition="(nx == 0) and (ny == 0)")
            self.de_problem.problem.add_bc("right(w) = 0", condition="(nx != 0) or  (ny != 0)")
        else:
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
 
       

class BoussinesqEquations2D(BoussinesqEquations):
    """ 
    Two-dimensional, incompressible Boussinesq equations in the form:

        ∇ · u = 0
        d_t u - u ⨯ ω = - ∇ p + T1 (zhat) - √(Pr/Ra) * ∇ ⨯ ω
        d_t T1 + u · ∇ (T0 + T1) = 1/(√[Pr Ra]) * ∇ ² T1

    See Anders, Brown, & Oishi (2018, PRFluids) for some info on where
    this form comes from compared to a more-traditional writing of the eqns.
    """
    def __init__(self, *args, **kwargs):
        """ 
        Initialize class and set up variables that will be used in eqns:
            T1   - Temperature fluctuations from initial state
            T1_z - z-derivative of T1
            p    - Pressure, magic
            u    - Horizontal velocity
            w    - Vertical velocity
            Oy   - y-vorticity (out of plane)
        """
        super(BoussinesqEquations2D, self).__init__(*args, **kwargs)
        self.de_problem.set_variables(['T1_z','T1','p','u','w','Oy'])

    def _set_subs(self, kx = 0):
        """ Set some important substitutions for the equations
        
        Parameters
        ----------
        kx  : float
            The horizontal wavenumber to use, if using a 1D atmosphere
        """
        if self.de_domain.dimensions == 1:
            self.de_problem.problem.parameters['j'] = 1j
            self.de_problem.problem.substitutions['dx(f)'] = "j*kx*(f)"
            self.de_problem.problem.parameters['kx'] = kx

        self.de_problem.problem.substitutions['UdotGrad(A, A_z)'] = '(u * dx(A) + w * A_z)'
        self.de_problem.problem.substitutions['Lap(A, A_z)'] = '(dx(dx(A)) + dz(A_z))'
       
        self.de_problem.problem.substitutions['v'] = '0'
        self.de_problem.problem.substitutions['dy(A)'] = '0'

        self.de_problem.problem.substitutions['Ox'] = '(dy(w) - dz(v))'
        self.de_problem.problem.substitutions['Oz'] = '(dx(v) - dy(u))'

    def set_equations(self, kx = 0):
        """ Setup the equations in Dedalus
        
        Parameters
        ----------
        kx  : float
            The horizontal wavenumber to use, if using a 1D atmosphere
        """

        self._set_subs(kx=kx)

        logger.debug('Adding Eqn: Incompressibility constraint')
        self.de_problem.problem.add_equation("dx(u) + dz(w) = 0")
        logger.debug('Adding Eqn: T1_z defn')
        self.de_problem.problem.add_equation("T1_z - dz(T1) = 0")
        logger.debug('Adding Eqn: Vorticity defn')
        self.de_problem.problem.add_equation("Oy - dz(u) + dx(w) = 0")
        logger.debug('Adding Eqn: Momentum, x')
        self.de_problem.problem.add_equation("dt(u)  - R*dz(Oy)  + dx(p)              =  v*Oz - w*Oy ")
        logger.debug('Adding Eqn: Momentum, z')
        self.de_problem.problem.add_equation("dt(w)  + R*dx(Oy)  + dz(p)    - T1      =  u*Oy - v*Ox ")
        logger.debug('Adding Eqn: Energy')
        self.de_problem.problem.add_equation("dt(T1) - P*Lap(T1, T1_z) + w*T0_z   = -UdotGrad(T1, T1_z)")

class BoussinesqEquations3D(BoussinesqEquations):
    """ 
    Three-dimensional, incompressible Boussinesq equations in the form:

        ∇ · u = 0
        d_t u - u ⨯ ω = - ∇ p + T1 (zhat) - √(Pr/Ra) * ∇ ⨯ ω
        d_t T1 + u · ∇ (T0 + T1) = 1/(√[Pr Ra]) * ∇ ² T1

    See Anders, Brown, & Oishi (2018, PRFluids) for some info on where
    this form comes from compared to a more-traditional writing of the eqns.
    """

    def __init__(self, *args, **kwargs):
        """ 
        Initialize class and set up variables that will be used in eqns:
            T1   - Temperature fluctuations from static state
            T1_z - z-derivative of T1
            p    - Pressure, magic
            u    - Horizontal velocity (x)
            v    - Horizontal velocity (y)
            w    - Vertical velocity
            Ox   - x-vorticity
            Oy   - y-vorticity
            Oz   - z-vorticity
        """
        super(BoussinesqEquations3D, self).__init__(*args, **kwargs)
        self.de_problem.set_variables(['T1','T1_z','p','u','v', 'w','Ox', 'Oy', 'Oz'])

    def _set_subs(self, kx = 0, ky = 0):
        """ Set some important substitutions for the equations
        
        Parameters
        ----------
        kx  : float
            The x-wavenumber to use, if using a 1D atmosphere
        ky  : float
            The y-wavenumber to use, if using a 1D atmosphere
        """
        if self.dimensions == 1:
            self.de_problem.problem.parameters['j'] = 1j
            self.de_problem.problem.substitutions['dx(f)'] = "j*kx*(f)"
            self.de_problem.problem.parameters['kx'] = kx
            self.de_problem.problem.substitutions['dy(f)'] = "j*ky*(f)"
            self.de_problem.problem.parameters['ky'] = ky
 
        self.de_problem.problem.substitutions['UdotGrad(A, A_z)'] = '(u * dx(A) + v * dy(A) + w * A_z)'
        self.de_problem.problem.substitutions['Lap(A, A_z)'] = '(dx(dx(A)) + dy(dy(A)) + dz(A_z))'

    def set_equations(self, kx = 0, ky = 0):
        """ Setup the equations in Dedalus
        
        Parameters
        ----------
        kx  : float
            The x-wavenumber to use, if using a 1D atmosphere
        ky  : float
            The y-wavenumber to use, if using a 1D atmosphere
        """
        self._set_subs(kx=kx, ky=ky)

        logger.debug('Adding Eqn: Incompressibility constraint')
        self.de_problem.problem.add_equation("dx(u) + dy(v) + dz(w) = 0")
        logger.debug('Adding Eqn: Energy')
        self.de_problem.problem.add_equation("dt(T1) - P*Lap(T1, T1_z) + w*T0_z           = -UdotGrad(T1, T1_z)")
        logger.debug('Adding Eqn: Momentum, x')
        self.de_problem.problem.add_equation("dt(u)  + R*(dy(Oz) - dz(Oy))  + dx(p)       =  v*Oz - w*Oy ")
        logger.debug('Adding Eqn: Momentum, y')
        self.de_problem.problem.add_equation("dt(v)  + R*(dz(Ox) - dx(Oz))  + dy(p)       =  w*Ox - u*Oz ")
        logger.debug('Adding Eqn: Momentum, z')
        self.de_problem.problem.add_equation("dt(w)  + R*(dx(Oy) - dy(Ox))  + dz(p) - T1  =  u*Oy - v*Ox ")
        logger.debug('Adding Eqn: T1_z defn')
        self.de_problem.problem.add_equation("T1_z - dz(T1) = 0")
        logger.debug('Adding Eqn: X Vorticity defn')
        self.de_problem.problem.add_equation("Ox - dy(w) + dz(v) = 0")
        logger.debug('Adding Eqn: Y Vorticity defn')
        self.de_problem.problem.add_equation("Oy - dz(u) + dx(w) = 0")
        logger.debug('Adding Eqn: Z Vorticity defn')
        self.de_problem.problem.add_equation("Oz - dx(v) + dy(u) = 0")

