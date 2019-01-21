from collections import OrderedDict

import numpy as np
import logging
logger = logging.getLogger(__name__)

try:
    from functions import mpi_makedirs
except:
    from sys import path
    path.insert(0, './logic')
    from functions import mpi_makedirs

class IdealGasAtmosphere:
    """
    An abstract class which contains many of the attributes of an
    ideal gas atmosphere, to be extended to more specific cases.

    Attributes:
        params  - An OrderedDict of scalar atmospheric parameters
        gamma   - The adiabatic index of the atmosphere
        Cp, Cv  - Specific heat at constant pressure, volume
        g       - gravity, constant. (default value sets T_ad_z = -1)
        R       - The ideal gas constant
        T_ad_z  - The adiabatic temperature gradient, as in Anders&Brown2017
        domain  - A DedalusDomain object in which the atmosphere will be built
        atmo_fields  - An OrderedDict which contains fields for the variables:
                   T0, T0_z, T0_zz, rho0, rho0_z, ln_rho0, ln_rho0_z, p0, s0, s0_z
    """


    def __init__(self, **kwargs):
        """ Initializes the atmosphere

        Parameters
        ----------
        de_domain       : A DedalusDomain object
            Contains info about the dedalus domain
        de_problem      : A DedalusProblem object
            Contains info about the dedalus problem and solver
        kwargs          : Dictionary
            Additional keyword arguments for the self._set_parameters function
        """
        self.de_domain   = None
        self.de_problem  = None
        self.params      = OrderedDict()
        self.atmo_fields = OrderedDict()
        self._initialize_atmosphere(**kwargs)

    def prepare_atmosphere(self, de_domain, de_problem, *args, **kwargs):
        self.set_domain(de_domain)
        self.set_problem(de_problem)
        self._set_parameters(*args, **kwargs);
        self._set_subs()

    def _initialize_atmosphere(self, gamma=5./3, R=1, g=None):
        self.rho_fit     = None
        self.params['R'] = R
        self.params['gamma'] = gamma
        self.params['Cp'] = gamma*R/(gamma-1.)
        self.params['Cv'] = self.params['Cp'] - R
        if not('g' in self.params.keys()):
            if g is None:
                self.params['g'] = self.params['Cp']
            else:
                self.params['g'] = 1
        self.params['T_ad_z'] = - self.params['g'] / self.params['Cp']

        logger.info('Initialized IdealGasAtmosphere:')
        logger.info('   R = {:.2g}, gamma = {:.2g}, Cp = {:.2g}, Cv = {:.2g}'.format(R, gamma, self.params['Cp'], self.params['Cv']))
        logger.info('   g = {:.2g} and T_ad_z = {:.2g}'.format(self.params['g'], self.params['T_ad_z']))

    def _set_parameters(self):
        fds = ['T0', 'T0_z', 'T0_zz', 'rho0', 'ln_rho0', 'ln_rho0_z']
        for f in fds:
            self.atmo_fields[f] = self.de_domain.new_ncc()

    def _set_subs(self):
        self.de_problem.problem.substitutions['P0']   = '(rho0*T0)'
        self.de_problem.problem.substitutions['s0']   = '((1/Cv)*log(T0) - ln_rho0)'

        self.de_problem.problem.substitutions['ln_rho_full'] = '(ln_rho0 + ln_rho1)'
        self.de_problem.problem.substitutions['rho_full']    = '(rho0*exp(ln_rho1))'
        self.de_problem.problem.substitutions['T_full']      = '(T0 + T1)'
        self.de_problem.problem.substitutions['P_full']      = '(rho_full*T_full)'
        self.de_problem.problem.substitutions['s_full']      = '((1/Cv)*log(T_full) - ln_rho_full)'
       
        self.de_problem.problem.substitutions['rho_fluc']    = '(rho_full - rho0)'
        self.de_problem.problem.substitutions['P_fluc']      = '(P_full - P0)'
        self.de_problem.problem.substitutions['s_fluc']      = '(s_full - s0)'

    def set_domain(self, de_domain):
        self.de_domain = de_domain
    
    def set_problem(self, de_problem):
        self.de_problem = de_problem

    def save_atmo_file(self, out_dir):
        import h5py
        mpi_makedirs(out_dir)
        out_file = out_dir + '/atmosphere.h5'
        with h5py.File('{:s}'.format(out_file), 'w') as f:
            f['z'] = self.de_domain.generate_vertical_profile(de_domain.z)
            for k, p in self.params.items():
                f[k] = p
            for k, fd in self.atmo_fields.items():
                f[k] = self.de_domain.generate_vertical_profile(fd, scales=1)

class Polytrope(IdealGasAtmosphere):
    """
    An extension of an IdealGasAtmosphere for a polytropic
    """
    def __init__(self, *args, **kwargs):
        super(Polytrope, self).__init__(*args, **kwargs)
    
    def _initialize_atmosphere(self, epsilon=0, n_rho=3, aspect_ratio=4, gamma=5./3, R=1):
        self.params['m_ad']     = 1/(gamma - 1)
        self.params['m']        = self.params['m_ad'] - epsilon
        self.params['g']        = R * (self.params['m'] + 1)
        self.params['n_rho']    = n_rho
        self.params['Lz']       = np.exp(self.params['n_rho']/self.params['m']) - 1
        self.params['Lx']       = aspect_ratio*self.params['Lz']
        self.params['Ly']       = aspect_ratio*self.params['Lz']
        self.params['delta_s']  = -epsilon*np.log(self.params['Lz'] + 1)
        super(Polytrope, self)._initialize_atmosphere(gamma=gamma, R=R)
        self.params['t_buoy']   = np.sqrt(np.abs(self.params['Lz']*self.params['Cp'] / self.params['g'] / self.params['delta_s']))
        logger.info('Initialized Polytrope:')
        logger.info('   epsilon = {:.2e}, m = {:.8g}, m_ad = {:.2g}'.format(epsilon, self.params['m'], self.params['m_ad']))
        logger.info('   Lz = {:.2g}, Lx/Ly = {:.2g}'.format(self.params['Lz'], self.params['Lx']))
        logger.info('   t_buoy = {:.2g}'.format(self.params['t_buoy']))

    def _set_atmo_structure(self):
        '''
        Sets up all atmospheric fields (T0, T0_z, T0_zz, rho0, rho0_z, ln_rho0,
        ln_rho0_z, p0, p0_z, s0, s0_z) according to a polytropic stratification
        of the form:

            T0 = (Lz + 1 - z)
            rho0 = T0**m 
        '''
        T0 = (self.params['Lz'] + 1 - self.de_domain.z)
        rho0 = T0**self.params['m']
        ln_rho0 = np.log(rho0)

        self.atmo_fields['T0']['g']      = T0
        self.atmo_fields['T0_z']['g']    = -1 
        self.atmo_fields['T0_zz']['g']   = 0
        self.atmo_fields['rho0']['g']    = rho0
        self.atmo_fields['ln_rho0']['g'] = ln_rho0
        self.atmo_fields['ln_rho0'].differentiate('z', out=self.atmo_fields['ln_rho0_z'])
    
    def _set_parameters(self, Rayleigh, Prandtl):
        super(Polytrope, self)._set_parameters()
        self._set_atmo_structure()

        self.params['mu0']    = nu_top  = np.sqrt(Prandtl*(self.params['g']*self.params['Lz']**3*np.abs(self.params['delta_s']/self.params['Cp']/Rayleigh)))
        self.params['kappa0'] = chi_top = nu_top / Prandtl
        self.params['t_therm_top'] = self.params['Lz']**2/chi_top
        self.params['t_therm_bot'] = self.params['Lz']**2/(chi_top*np.exp(2*self.params['n_rho']))
        logger.info('Diffusivities set by Ra = {:.2g}, Pr = {:.2g}'.format(Rayleigh, Prandtl))
        logger.info('   t_therm_top = {:.2g}, t_therm_bot = {:.2g}'.format(self.params['t_therm_top'], self.params['t_therm_bot']))

        for k, fd in self.atmo_fields.items():
            self.de_problem.problem.parameters[k] = fd
            fd.set_scales(1, keep_data=True)

        for k, p in self.params.items():
            self.de_problem.problem.parameters[k] = p

    def _set_subs(self):
        """
        Sets up substitutions that are useful for the Boussinesq equations or for outputs
        """
        super(Polytrope, self)._set_subs()

        #Diffusivities; diffusive timescale
        self.de_problem.problem.substitutions['chi0'] = '(kappa0/rho0)'
        self.de_problem.problem.substitutions['nu0']  = '(mu0/rho0)'
        self.de_problem.problem.substitutions['phi']  = '(-g*(1 + Lz - z))'


    def set_output_subs(self):
        if self.de_domain.dimensions == 1:
            self.de_problem.problem.substitutions['plane_avg(A)'] = 'A'
            self.de_problem.problem.substitutions['plane_std(A)'] = '0'
            self.de_problem.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lz'
        elif self.de_domain.dimensions == 2:
            self.de_problem.problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
            self.de_problem.problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'
            self.de_problem.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz'
        else:
            self.de_problem.problem.substitutions['plane_avg(A)'] = 'integ(A, "x", "y")/Lx/Ly'
            self.de_problem.problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'
            self.de_problem.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Ly/Lz'

        self.de_problem.problem.substitutions['KE']        = '(0.5*rho_full*vel_rms**2)'
        self.de_problem.problem.substitutions['PE']        = '(rho_full*phi)'
        self.de_problem.problem.substitutions['IE']        = '(rho_full*Cv*T_full)'
        self.de_problem.problem.substitutions['h']         = '(IE + P_full)'
        self.de_problem.problem.substitutions['PE_fluc']   = '(rho_fluc*phi)'
        self.de_problem.problem.substitutions['IE_fluc']   = '(rho_full*Cv*T1 + rho_fluc*Cv*T0)'
        self.de_problem.problem.substitutions['h_fluc']    = '(IE_fluc + P_fluc)'

    
        self.de_problem.problem.substitutions['u_rms']      = 'sqrt(u**2)'
        self.de_problem.problem.substitutions['v_rms']      = 'sqrt(v**2)'
        self.de_problem.problem.substitutions['w_rms']      = 'sqrt(w**2)'
        self.de_problem.problem.substitutions['Re_rms']     = '(vel_rms / nu_full)'
        self.de_problem.problem.substitutions['Pe_rms']     = '(vel_rms / chi_full)'
        self.de_problem.problem.substitutions['Ma_iso_rms'] = '(vel_rms/sqrt(T_full))'
        self.de_problem.problem.substitutions['Ma_ad_rms']  = '(vel_rms/sqrt(gamma*T_full))'

        self.de_problem.problem.substitutions['enth_flux_z']    = '(w*h)'
        self.de_problem.problem.substitutions['KE_flux_z']      = '(w*KE)'
        self.de_problem.problem.substitutions['PE_flux_z']      = '(w*PE)'
        self.de_problem.problem.substitutions['viscous_flux_z'] = '(-rho_full * nu_full * (u*Sig_xz + v*Sig_yz + w*Sig_zz))'
        self.de_problem.problem.substitutions['F_conv_z']       = '(enth_flux_z + KE_flux_z + PE_flux_z + viscous_flux_z)'

        self.de_problem.problem.substitutions['F_cond_z']      = '(-kappa_full*dz(T_full))'
        self.de_problem.problem.substitutions['F_cond_fluc_z'] = '(-kappa_full*T1_z - kappa_fluc*T0_z)'
        self.de_problem.problem.substitutions['F_cond0_z']     = '(-kappa0*T0_z)'
        self.de_problem.problem.substitutions['F_cond_ad_z']   = '(-kappa_full*T_ad_z)'

        self.de_problem.problem.substitutions['Nu'] = '((F_conv_z + F_cond_z - F_cond_ad_z)/vol_avg(F_cond_z - F_cond_ad_z))'



