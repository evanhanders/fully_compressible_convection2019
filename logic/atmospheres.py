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
        kwargs          : Dictionary
            Additional keyword arguments for the self._set_parameters function
        """
        self.atmo_params = OrderedDict()
        self.atmo_params['Lz'] = None
        self.atmo_fields = OrderedDict()
        self._prepare_scalar_info(**kwargs)

    def _prepare_scalar_info(self):
        """ Abstract function. Must set self.atmo_params['Lz'] properly in child classes """
        pass

    def _set_thermodynamics(self, gamma=5./3, R=1):
        self.atmo_params['R'] = R
        self.atmo_params['gamma'] = gamma
        self.atmo_params['Cp'] = gamma*R/(gamma-1.)
        self.atmo_params['Cv'] = self.atmo_params['Cp'] - R
        if not('g' in self.atmo_params.keys()):
            self.atmo_params['g'] = self.atmo_params['Cp']
        self.atmo_params['T_ad_z'] = - self.atmo_params['g'] / self.atmo_params['Cp']

        logger.info('Initialized IdealGasAtmosphere:')
        logger.info('   R = {:.2g}, gamma = {:.2g}, Cp = {:.2g}, Cv = {:.2g}'.format(R, gamma, self.atmo_params['Cp'], self.atmo_params['Cv']))
        logger.info('   g = {:.2g} and T_ad_z = {:.2g}'.format(self.atmo_params['g'], self.atmo_params['T_ad_z']))

    def _make_atmo_fields(self, de_domain, adtl_fds=None):
        fds = ['T0', 'T0_z', 'T0_zz', 'rho0', 'ln_rho0', 'ln_rho0_z', 'phi', 'chi0', 'nu0']
        if type(adtl_fds) is list:
            fds += adtl_fds
        for f in fds:
            self.atmo_fields[f] = de_domain.new_ncc()

    def _set_subs(self, de_problem):
        de_problem.problem.substitutions['P0']   = '(rho0*T0)'
        de_problem.problem.substitutions['s0']   = '(R*((1/(gamma-1))*log(T0) - ln_rho0))'

        de_problem.problem.substitutions['ln_rho_full'] = '(ln_rho0 + ln_rho1)'
        de_problem.problem.substitutions['rho_full']    = '(rho0*exp(ln_rho1))'
        de_problem.problem.substitutions['T_full']      = '(T0 + T1)'
        de_problem.problem.substitutions['P_full']      = '(rho_full*T_full)'
        de_problem.problem.substitutions['s_full']      = '(R*((1/(gamma-1))*log(T_full) - ln_rho_full))'
       
        de_problem.problem.substitutions['rho_fluc']    = '(rho_full - rho0)'
        de_problem.problem.substitutions['P_fluc']      = '(P_full - P0)'
        de_problem.problem.substitutions['s_fluc']      = '(s_full - s0)'

    def save_atmo_file(self, out_dir, de_domain):
        import h5py
        mpi_makedirs(out_dir)
        out_file = out_dir + '/atmosphere.h5'
        with h5py.File('{:s}'.format(out_file), 'w') as f:
            f['z'] = de_domain.generate_vertical_profile(de_domain.z)
            for k, p in self.atmo_params.items():
                f[k] = p
            for k, fd in self.atmo_fields.items():
                f[k] = de_domain.generate_vertical_profile(fd, scales=1)

class Polytrope(IdealGasAtmosphere):
    """
    An extension of an IdealGasAtmosphere for a polytropic stratification,
    nondimensionalized on the atmosphere's temperature gradient and the
    isothermal sound crossing time at the top.
    """
    def __init__(self, *args, **kwargs):
        super(Polytrope, self).__init__(*args, **kwargs)

    def _prepare_scalar_info(self, epsilon=0, n_rho=3, gamma=5./3, R=1):
        self.atmo_params['m_ad']     = 1/(gamma - 1)
        self.atmo_params['m']        = self.atmo_params['m_ad'] - epsilon
        self.atmo_params['g']        = R * (self.atmo_params['m'] + 1)
        self.atmo_params['n_rho']    = n_rho
        self.atmo_params['Lz']       = np.exp(self.atmo_params['n_rho']/self.atmo_params['m']) - 1
        self.atmo_params['delta_s']  = -R * epsilon*np.log(self.atmo_params['Lz'] + 1)
        super(Polytrope, self)._set_thermodynamics(gamma=gamma, R=R)
        self.atmo_params['t_buoy']   = np.sqrt(np.abs(self.atmo_params['Lz']*self.atmo_params['Cp'] / self.atmo_params['g'] / self.atmo_params['delta_s']))
        logger.info('Initialized Polytrope:')
        logger.info('   epsilon = {:.2e}, m = {:.8g}, m_ad = {:.2g}'.format(epsilon, self.atmo_params['m'], self.atmo_params['m_ad']))
        logger.info('   Lz = {:.2g}, t_buoy = {:.2g}'.format(self.atmo_params['Lz'], self.atmo_params['t_buoy']))

    def build_atmosphere(self, de_domain):
        """
        Sets up all atmospheric fields (T0, T0_z, T0_zz, rho0, rho0_z, ln_rho0,
        ln_rho0_z, p0, p0_z, s0, s0_z) according to a polytropic stratification
        of the form:

            T0 = (Lz + 1 - z)
            rho0 = T0**m 

        chi0 and nu0 are set at the experiment level.
        """ 
        self._make_atmo_fields(de_domain)
        T0 = (self.atmo_params['Lz'] + 1 - de_domain.z)
        rho0 = T0**self.atmo_params['m']
        ln_rho0 = np.log(rho0)

        self.atmo_fields['T0']['g']      = T0
        self.atmo_fields['T0_z']['g']    = -1 
        self.atmo_fields['T0_zz']['g']   = 0
        self.atmo_fields['rho0']['g']    = rho0
        self.atmo_fields['ln_rho0']['g'] = ln_rho0
        self.atmo_fields['ln_rho0'].differentiate('z', out=self.atmo_fields['ln_rho0_z'])
        self.atmo_fields['phi']['g'] = -self.atmo_params['g']*(T0)
