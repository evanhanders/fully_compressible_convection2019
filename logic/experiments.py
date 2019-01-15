import numpy as np
import logging
logger = logging.getLogger(__name__)

try:
    from functions import global_noise
except:
    from sys import path
    path.insert(0, './logic')
    from logic.functions import global_noise

class BoussinesqConvection:

    def __init__(self, equations, Rayleigh=1e4, Prandtl=1, IH=True, fixed_t=False, stable=False):
        self.equations = equations
        self._set_parameters(Rayleigh, Prandtl, IH=IH, fixed_t=fixed_t, stable=stable)
        self._set_subs()
        return

    def set_IC(self, solver, de_domain, A0=1e-6, **kwargs):
        """
        Set initial conditions as random noise.  I *think* characteristic
        temperature perturbutations are on the order of P, as in the energy
        equation, so our perturbations should be small, comparably (to start
        at a low Re even at large Ra, this is necessary)
        """
        # initial conditions
        T_IC = solver.state['T1']
        T_z_IC = solver.state['T1_z']
            
        noise = global_noise(self.equations.de_domain, **kwargs)
        noise.set_scales(self.equations.de_domain.dealias, keep_data=True)
        T_IC.set_scales(self.equations.de_domain.dealias, keep_data=True)
        self.T0.set_scales(self.equations.de_domain.dealias, keep_data=True)
        T_IC['g'] = A0*self.P*np.sin(np.pi*self.equations.de_domain.z_de/self.equations.de_domain.Lz)*noise['g']*self.T0['g']
        T_IC.differentiate('z', out=T_z_IC)
        logger.info("Starting with T1 perturbations of amplitude A0 = {:g}".format(A0))


    def _set_parameters(self, Rayleigh, Prandtl, IH=True, fixed_t=False, stable=False):
        """
        Set up important parameters of the problem for boussinesq convection. Assumes domain spans
        z = [0, 1]
        """
        self.T0_z      = self.equations.de_domain.new_ncc()
        self.T0        = self.equations.de_domain.new_ncc()

        if IH:	
            if fixed_t or stable:
                self.T0_z['g'] = 0.5 - self.equations.de_domain.z
                self.T0['g']   = 0.5*(self.equations.de_domain.z - self.equations.de_domain.z**2)
            else:
                self.T0_z['g'] = -self.equations.de_domain.z
                self.T0['g']   = 0.5 - 0.5*self.equations.de_domain.z**2
        else:
            self.T0_z['g'] = -1
            self.T0['g']   = self.equations.de_domain.Lz/2 - self.equations.de_domain.z

        self.equations.problem.parameters['T0'] = self.T0
        self.equations.problem.parameters['T0_z'] = self.T0_z

        # Characteristic scales (things we've non-dimensionalized on)
        self.equations.problem.parameters['t_buoy']   = 1.
        self.equations.problem.parameters['v_ff']     = 1.

        self.equations.problem.parameters['Rayleigh'] = Rayleigh
        self.equations.problem.parameters['Prandtl']  = Prandtl

        self.P = (Rayleigh * Prandtl)**(-1./2)
        self.R = (Rayleigh / Prandtl)**(-1./2)
        self.equations.problem.parameters['P'] = (Rayleigh * Prandtl)**(-1./2)
        self.equations.problem.parameters['R'] = (Rayleigh / Prandtl)**(-1./2)
        self.thermal_time = (Rayleigh / Prandtl)**(1./2)

        self.equations.problem.parameters['Lz'] = self.equations.de_domain.Lz
        if self.equations.de_domain.dimensions >= 2:
            self.equations.problem.parameters['Lx'] = self.equations.de_domain.Lx
        if self.equations.de_domain.dimensions >= 3:
            self.equations.problem.parameters['Ly'] = self.equations.de_domain.Ly


    def _set_subs(self):
        """
        Sets up substitutions that are useful for the Boussinesq equations or for outputs
        """
        if self.equations.de_domain.dimensions == 1:
            self.equations.problem.substitutions['plane_avg(A)'] = 'A'
            self.equations.problem.substitutions['plane_std(A)'] = '0'
            self.equations.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lz'
        elif self.equations.de_domain.dimensions == 2:
            self.equations.problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
            self.equations.problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'
            self.equations.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz'

            self.equations.problem.substitutions['v']         = '0'
            self.equations.problem.substitutions['dy(A)']     = '0'
            self.equations.problem.substitutions['Ox']        = '(dy(w) - dz(v))'
            self.equations.problem.substitutions['Oz']        = '(dx(v) - dy(u))'
        else:
            self.equations.problem.substitutions['plane_avg(A)'] = 'integ(A, "x", "y")/Lx/Ly'
            self.equations.problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'
            self.equations.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Ly/Lz'

            self.equations.problem.substitutions['v_fluc'] = '(v - plane_avg(v))'


        #Diffusivities; diffusive timescale
        self.equations.problem.substitutions['chi']= '(v_ff * Lz * P)'
        self.equations.problem.substitutions['visc_nu'] = '(v_ff * Lz * R)'
        self.equations.problem.substitutions['t_therm'] = '(Lz**2/chi)'
        
        self.equations.problem.substitutions['vel_rms']   = 'sqrt(u**2 + v**2 + w**2)'
        self.equations.problem.substitutions['enstrophy'] = '(Ox**2 + Oy**2 + Oz**2)'

        self.equations.problem.substitutions['u_fluc'] = '(u - plane_avg(u))'
        self.equations.problem.substitutions['w_fluc'] = '(w - plane_avg(w))'
        self.equations.problem.substitutions['KE'] = '(0.5*vel_rms**2)'

        self.equations.problem.substitutions['Re'] = '(vel_rms / visc_nu)'
        self.equations.problem.substitutions['Pe'] = '(vel_rms / chi)'
        
        self.equations.problem.substitutions['enth_flux_z']  = '(w*(T1+T0))'
        self.equations.problem.substitutions['kappa_flux_z'] = '(-P*(T1_z+T0_z))'
        self.equations.problem.substitutions['conv_flux_z']  = '(enth_flux_z + kappa_flux_z)'
        self.equations.problem.substitutions['delta_T']      = 'vol_avg(right(T1 + T0) - left(T1 + T0))' 
        self.equations.problem.substitutions['Nu']           = '(conv_flux_z/vol_avg(kappa_flux_z))'
        #Goluskin's defn
        self.equations.problem.substitutions['Nu_IH_1'] = "(1/(8*interp(plane_avg(T0+T1), z=0.5)))"
