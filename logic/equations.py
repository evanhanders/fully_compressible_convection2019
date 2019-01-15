import numpy as np
from mpi4py import MPI
import scipy.special as scp

from collections import OrderedDict

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from dedalus import public as de

class Equations():
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
        return
    
    def set_IVP(self, *args, ncc_cutoff=1e-10, **kwargs):
        """
        Constructs and initial value problem of the current object's equation set
        """
        self.problem_type = 'IVP'
        self.problem = de.IVP(self.de_domain.domain, variables=self.variables, ncc_cutoff=ncc_cutoff)

    def set_EVP(self, *args, ncc_cutoff=1e-10, tolerance=1e-10, **kwargs):
        """
        Constructs an eigenvalue problem of the current objeect's equation set.
        Note that dt(f) = omega * f, not i * omega * f, so real parts of omega
        are growth / shrinking nodes, imaginary parts are oscillating.
        """

        self.problem_type = 'EVP'
        self.problem = de.EVP(self.de_domain.domain, variables=self.variables, eigenvalue='omega', ncc_cutoff=ncc_cutoff, tolerance=tolerance)
        self.problem.substitutions['dt(f)'] = "omega*f"

    def set_equations(self, *args, **kwargs):
        """ This function must be implemented in child classes """
        pass

    def _set_subs(self):
        """ This function must be implemented in child classes """
        pass


class BoussinesqEquations(Equations):
    """
    An extension of the Equations class which contains the full 2D form of the boussinesq
    equations.   
    """
    def __init__(self, *args, **kwargs):
        """ Set up class """
        super(BoussinesqEquations, self).__init__(*args, **kwargs)
       
    def set_BC(self,
               fixed_f=None, fixed_t=None, fixed_f_fixed_t=None, fixed_t_fixed_f=None,
               stress_free=None, no_slip=None):
        """
        Sets the velocity and thermal boundary conditions at the upper and lower boundaries.  Choose
        one thermal type of BC and one velocity type of BC to set those conditions.  See
        set_thermal_BC() and set_velocity_BC() functions for default choices and specific formulations.
        """
        self.dirichlet_set = []

        self.set_thermal_BC(fixed_f=fixed_f, fixed_t=fixed_t,
                            fixed_f_fixed_t=fixed_f_fixed_t, fixed_t_fixed_f=fixed_t_fixed_f)
        
        self.set_velocity_BC(stress_free=stress_free, no_slip=no_slip)
        
        for key in self.dirichlet_set:
            self.problem.meta[key]['z']['dirichlet'] = True
            
    def set_thermal_BC(self, fixed_f=None, fixed_t=None, fixed_f_fixed_t=None, fixed_t_fixed_f=None):
        """
        Sets the thermal boundary conditions at the top and bottom of the atmosphere.  If no choice is made, then the
        default BC is fixed flux (bottom), fixed temperature (top).

        Choices:
            fixed_f              - T1_z = 0 at top and bottom
            fixed_t       - T1 = 0 at top and bottom
            fixed_f_fixed_t  - T1_z = 0 at bottom, T1 = 0 at top
            fixed_t_fixed_f  - T1 = 0 at bottom, T1_z = 0 at top.
        """
        if not(fixed_f) and not(fixed_t) and not(fixed_t_fixed_f) and not(fixed_f_fixed_t):
            fixed_f_fixed_t = True

        if fixed_f:
            logger.info("Thermal BC: fixed flux (full form)")
            self.problem.add_bc( "left(T1_z) = 0")
            self.problem.add_bc("right(T1_z) = 0")
            self.dirichlet_set.append('T1_z')
        elif fixed_t:
            logger.info("Thermal BC: fixed temperature (T1)")
            self.problem.add_bc( "left(T1) = 0")
            self.problem.add_bc("right(T1) = 0")
            self.dirichlet_set.append('T1')
        elif fixed_f_fixed_t:
            logger.info("Thermal BC: fixed flux/fixed temperature")
            self.problem.add_bc("left(T1_z) = 0")
            self.problem.add_bc("right(T1)  = 0")
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('T1')
        elif fixed_t_fixed_f:
            logger.info("Thermal BC: fixed temperature/fixed flux")
            logger.info("warning; these are not fully correct fixed flux conditions yet")
            self.problem.add_bc("left(T1)    = 0")
            self.problem.add_bc("right(T1_z) = 0")
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('T1')
        else:
            logger.error("Incorrect thermal boundary conditions specified")
            raise

    def set_velocity_BC(self, stress_free=None, no_slip=None):
        """
        Sets the velocity boundary conditions at the top and bottom of the atmosphere.  If no choice is made, then the
        default BC is no slip (top and bottom)

        Boundaries are, by default, impenetrable (w = 0 at top and bottom)

        Choices:
            stress_free         - Oy = 0 at top and bottom [note: Oy = dz(u) - dx(w). With
                                    impenetrable boundaries at top and bottom, dx(w) = 0, so
                                    really these are dz(u) = 0 boundary conditions]
            no_slip             - u = 0 at top and bottom.
        """

        if not(stress_free) and not(no_slip):
            stress_free = True
            
        # horizontal velocity boundary conditions
        if stress_free:
            logger.info("Horizontal velocity BC: stress free")
            self.problem.add_bc("left(Oy) = 0")
            self.problem.add_bc("right(Oy) = 0")
            self.dirichlet_set.append('Oy')
            if self.de_domain.dimensions == 3:
                self.problem.add_bc("left(Ox) = 0")
                self.problem.add_bc("right(Ox) = 0")
                self.dirichlet_set.append('Ox')
        elif no_slip:
            logger.info("Horizontal velocity BC: no slip")
            self.problem.add_bc( "left(u) = 0")
            self.problem.add_bc("right(u) = 0")
            self.dirichlet_set.append('u')
            if self.de_domain.dimensions == 3:
                self.problem.add_bc( "left(v) = 0")
                self.problem.add_bc("right(v) = 0")
                self.dirichlet_set.append('v')
        else:
            logger.error("Incorrect horizontal velocity boundary conditions specified")
            raise

        # vertical velocity boundary conditions
        logger.info("Vertical velocity BC: impenetrable")
        self.problem.add_bc( "left(w) = 0")
        if self.de_domain.dimensions == 2:
            self.problem.add_bc("right(p) = 0", condition="(nx == 0)")
            self.problem.add_bc("right(w) = 0", condition="(nx != 0)")
        elif self.de_domain.dimensions == 3:
            self.problem.add_bc("right(p) = 0", condition="(nx == 0) and (ny == 0)")
            self.problem.add_bc("right(w) = 0", condition="(nx != 0) or  (ny != 0)")
        else:
            self.problem.add_bc("right(w) = 0")
        self.dirichlet_set.append('w')
        

class BoussinesqEquations2D(BoussinesqEquations):

    def __init__(self, *args, **kwargs):
        """ 
        Initialize class and set up variables that will be used in eqns:
            T1 - Temperature fluctuations from static state
            T1_z - z-derivative of T1
            p    - Pressure, magic
            u    - Horizontal velocity
            w    - Vertical velocity
            Oy   - y-vorticity (out of plane)
        """
        variables = ['T1_z','T1','p','u','w','Oy']
        super(BoussinesqEquations2D, self).__init__(*args, variables=variables, **kwargs)

    def _set_subs(self, kx=0):
        if self.de_domain.dimensions == 1:
            self.problem.parameters['j'] = 1j
            self.problem.substitutions['dx(f)'] = "j*kx*(f)"
            self.problem.parameters['kx'] = kx

        self.problem.substitutions['UdotGrad(A, A_z)'] = '(u * dx(A) + w * A_z)'
        self.problem.substitutions['Lap(A, A_z)'] = '(dx(dx(A)) + dz(A_z))'
       
        self.problem.substitutions['v'] = '0'
        self.problem.substitutions['dy(A)'] = '0'

        self.problem.substitutions['Ox'] = '(dy(w) - dz(v))'
        self.problem.substitutions['Oz'] = '(dx(v) - dy(u))'

    def set_equations(self, kx = 0):
        """
        Set the Boussinesq, Incompressible equations:

            ∇ · u = 0
            d_t u - u ⨯ ω = - ∇ p + T1 (zhat) - √(Pr/Ra) * ∇ ⨯ ω
            d_t T1 + u · ∇ (T0 + T1) = 1/(√[Pr Ra]) * ∇ ² T1

        Here, the form of the momentum equation has been recovered from a more
        familiar form:
            d_t u + u · ∇ u = - ∇ p + T1 (zhat) + √(Pr/Ra) * ∇ ² u,
        where vector operations have been used to express the equation mostly in terms
        of vorticity.  There is a leftover term in
            u · ∇ u = (1/2) ∇ u² - u ⨯ ω,
        but this u² term gets swept into the ∇ p term in boussinesq convection, where p 
        enforces ∇ · u = 0
        """

        self._set_subs(kx=kx)

        # This formulation is numerically faster to run than the standard form.
        # 2D Boussinesq hydrodynamics

        logger.debug('Adding Eqn: Incompressibility constraint')
        self.problem.add_equation("dx(u) + dz(w) = 0")
        logger.debug('Adding Eqn: T1_z defn')
        self.problem.add_equation("T1_z - dz(T1) = 0")
        logger.debug('Adding Eqn: Vorticity defn')
        self.problem.add_equation("Oy - dz(u) + dx(w) = 0")
        logger.debug('Adding Eqn: Momentum, x')
        self.problem.add_equation("dt(u)  - R*dz(Oy)  + dx(p)              =  v*Oz - w*Oy ")
        logger.debug('Adding Eqn: Momentum, z')
        self.problem.add_equation("dt(w)  + R*dx(Oy)  + dz(p)    - T1      =  u*Oy - v*Ox ")
        logger.debug('Adding Eqn: Energy')
        self.problem.add_equation("dt(T1) - P*Lap(T1, T1_z) + w*T0_z   = -UdotGrad(T1, T1_z)")

class BoussinesqEquations3D(BoussinesqEquations):

    def __init__(self, *args, **kwargs):
        """ 
        Initialize class and set up variables that will be used in eqns:
            T1 - Temperature fluctuations from static state
            T1_z - z-derivative of T1
            p    - Pressure, magic
            u    - Horizontal velocity (x)
            u_z   - z-derivative of u
            v    - Horizontal velocity (y)
            v_z  - z-derivative of v
            w    - Vertical velocity
            w_z  - z-derivative of w
        """
        variables=['T1','T1_z','p','u','v', 'w','Ox', 'Oy', 'Oz']
        super(BoussinesqEquations3D, self).__init__(*args, variables=variables, **kwargs)

    def _set_subs(self, kx=0, ky=0):
        """
        Sets up substitutions that are useful for the Boussinesq equations or for outputs
        """
        if self.dimensions == 1:
            self.problem.parameters['j'] = 1j
            self.problem.substitutions['dx(f)'] = "j*kx*(f)"
            self.problem.parameters['kx'] = kx
            self.problem.substitutions['dy(f)'] = "j*ky*(f)"
            self.problem.parameters['ky'] = ky
 
        self.problem.substitutions['UdotGrad(A, A_z)'] = '(u * dx(A) + v * dy(A) + w * A_z)'
        self.problem.substitutions['Lap(A, A_z)'] = '(dx(dx(A)) + dy(dy(A)) + dz(A_z))'

    def set_equations(self, kx = 0, ky = 0):
        """
        Set the Boussinesq, Incompressible equations. The same form is used here as in the 2D equations.
        """
        self._set_subs(kx=kx, ky=ky)

        logger.debug('Adding Eqn: Incompressibility constraint')
        self.problem.add_equation("dx(u) + dy(v) + dz(w) = 0")
        logger.debug('Adding Eqn: Energy')
        self.problem.add_equation("dt(T1) - P*Lap(T1, T1_z) + w*T0_z           = -UdotGrad(T1, T1_z)")
        logger.debug('Adding Eqn: Momentum, x')
        self.problem.add_equation("dt(u)  + R*(dy(Oz) - dz(Oy))  + dx(p)       =  v*Oz - w*Oy ")
        logger.debug('Adding Eqn: Momentum, y')
        self.problem.add_equation("dt(v)  + R*(dz(Ox) - dx(Oz))  + dy(p)       =  w*Ox - u*Oz ")
        logger.debug('Adding Eqn: Momentum, z')
        self.problem.add_equation("dt(w)  + R*(dx(Oy) - dy(Ox))  + dz(p) - T1  =  u*Oy - v*Ox ")
        logger.debug('Adding Eqn: T1_z defn')
        self.problem.add_equation("T1_z - dz(T1) = 0")
        logger.debug('Adding Eqn: X Vorticity defn')
        self.problem.add_equation("Ox - dy(w) + dz(v) = 0")
        logger.debug('Adding Eqn: Y Vorticity defn')
        self.problem.add_equation("Oy - dz(u) + dx(w) = 0")
        logger.debug('Adding Eqn: Z Vorticity defn')
        self.problem.add_equation("Oz - dx(v) + dy(u) = 0")
