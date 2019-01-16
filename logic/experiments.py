import numpy as np
import logging
logger = logging.getLogger(__name__)

class BoussinesqConvection:

    def __init__(self, de_domain, de_problem, Rayleigh=1e4, Prandtl=1, IH=True, fixed_t=False, stable=False):
        self.de_domain = de_domain
        self.de_problem = de_problem
        self._set_parameters(Rayleigh, Prandtl, IH=IH, fixed_t=fixed_t, stable=stable)
        self._set_subs()
        return

    def _set_parameters(self, Rayleigh, Prandtl, IH=True, fixed_t=False, stable=False):
        """
        Set up important parameters of the problem for boussinesq convection. Assumes domain spans
        z = [0, 1]
        """
        self.T0_z      = self.de_domain.new_ncc()
        self.T0        = self.de_domain.new_ncc()

        if IH:	
            if fixed_t or stable:
                self.T0_z['g'] = 0.5 - self.de_domain.z
                self.T0['g']   = 0.5*(self.de_domain.z - self.de_domain.z**2)
            else:
                self.T0_z['g'] = -self.de_domain.z
                self.T0['g']   = 0.5 - 0.5*self.de_domain.z**2
        else:
            self.T0_z['g'] = -1
            self.T0['g']   = self.de_domain.Lz/2 - self.de_domain.z

        self.de_problem.problem.parameters['T0'] = self.T0
        self.de_problem.problem.parameters['T0_z'] = self.T0_z

        # Characteristic scales (things we've non-dimensionalized on)
        self.de_problem.problem.parameters['t_buoy']   = 1.
        self.de_problem.problem.parameters['v_ff']     = 1.

        self.de_problem.problem.parameters['Rayleigh'] = Rayleigh
        self.de_problem.problem.parameters['Prandtl']  = Prandtl

        self.P = (Rayleigh * Prandtl)**(-1./2)
        self.R = (Rayleigh / Prandtl)**(-1./2)
        self.de_problem.problem.parameters['P'] = (Rayleigh * Prandtl)**(-1./2)
        self.de_problem.problem.parameters['R'] = (Rayleigh / Prandtl)**(-1./2)
        self.thermal_time = (Rayleigh / Prandtl)**(1./2)

        self.de_problem.problem.parameters['Lz'] = self.de_domain.Lz
        if self.de_domain.dimensions >= 2:
            self.de_problem.problem.parameters['Lx'] = self.de_domain.Lx
        if self.de_domain.dimensions >= 3:
            self.de_problem.problem.parameters['Ly'] = self.de_domain.Ly


    def _set_subs(self):
        """
        Sets up substitutions that are useful for the Boussinesq equations or for outputs
        """
        if self.de_domain.dimensions == 1:
            self.de_problem.problem.substitutions['plane_avg(A)'] = 'A'
            self.de_problem.problem.substitutions['plane_std(A)'] = '0'
            self.de_problem.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lz'
        elif self.de_domain.dimensions == 2:
            self.de_problem.problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
            self.de_problem.problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'
            self.de_problem.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz'

            self.de_problem.problem.substitutions['v']         = '0'
            self.de_problem.problem.substitutions['dy(A)']     = '0'
            self.de_problem.problem.substitutions['Ox']        = '(dy(w) - dz(v))'
            self.de_problem.problem.substitutions['Oz']        = '(dx(v) - dy(u))'
        else:
            self.de_problem.problem.substitutions['plane_avg(A)'] = 'integ(A, "x", "y")/Lx/Ly'
            self.de_problem.problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'
            self.de_problem.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Ly/Lz'

            self.de_problem.problem.substitutions['v_fluc'] = '(v - plane_avg(v))'


        #Diffusivities; diffusive timescale
        self.de_problem.problem.substitutions['chi']= '(v_ff * Lz * P)'
        self.de_problem.problem.substitutions['visc_nu'] = '(v_ff * Lz * R)'
        self.de_problem.problem.substitutions['t_therm'] = '(Lz**2/chi)'
        
        self.de_problem.problem.substitutions['vel_rms']   = 'sqrt(u**2 + v**2 + w**2)'
        self.de_problem.problem.substitutions['enstrophy'] = '(Ox**2 + Oy**2 + Oz**2)'

        self.de_problem.problem.substitutions['u_fluc'] = '(u - plane_avg(u))'
        self.de_problem.problem.substitutions['w_fluc'] = '(w - plane_avg(w))'
        self.de_problem.problem.substitutions['KE'] = '(0.5*vel_rms**2)'

        self.de_problem.problem.substitutions['Re'] = '(vel_rms / visc_nu)'
        self.de_problem.problem.substitutions['Pe'] = '(vel_rms / chi)'
        
        self.de_problem.problem.substitutions['enth_flux_z']  = '(w*(T1+T0))'
        self.de_problem.problem.substitutions['kappa_flux_z'] = '(-P*(T1_z+T0_z))'
        self.de_problem.problem.substitutions['conv_flux_z']  = '(enth_flux_z + kappa_flux_z)'
        self.de_problem.problem.substitutions['delta_T']      = 'vol_avg(right(T1 + T0) - left(T1 + T0))' 
        self.de_problem.problem.substitutions['Nu']           = '(conv_flux_z/vol_avg(kappa_flux_z))'
        #Goluskin's defn
        self.de_problem.problem.substitutions['Nu_IH_1'] = "(1/(8*interp(plane_avg(T0+T1), z=0.5)))"
