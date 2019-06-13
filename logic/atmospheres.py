from collections import OrderedDict

import numpy as np
import logging
logger = logging.getLogger(__name__)

try:
    from functions import mpi_makedirs
except:
    from sys import path
    path.insert(0, './logic')
    from logic.functions import mpi_makedirs

class IdealGasAtmosphere:
    """
    An abstract class which contains many of the attributes of an
    ideal gas atmosphere, to be extended to more specific cases.

    Attributes:
        atmo_params : OrderedDict
            Contains scalar atmospheric parameters (g, gamma, R, Cp, Cv, T_ad_z).
            NOTE: Child classes must specify atmo_params['Lz']
        atmo_fields : OrderedDict
            Contains Dedalus Fields for the variables:
                   T0, T0_z, T0_zz, rho0, ln_rho0, ln_rho0_z, phi, chi0, nu0
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

    def _prepare_scalar_info(self, *args, **kwargs):
        """ Abstract function. Must set self.atmo_params['Lz'] properly in child classes """
        pass
    
    def build_atmosphere(self, *args, **kwargs):
        """ Abstract function. Must call _make_atmo_fields in child classes """
        pass

    def _set_thermodynamics(self, gamma=5./3, R=1):
        """ Specify thermodynamic constants in the atmosphere. Assumes constant gravity.

        Parameters
        ----------
        gamma : float
            Adiabatic index of the gas
        R : float
            Ideal gas constant so that P = R * rho * T
        """
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

    def _make_atmo_fields(self, de_domain, add_fields=None):
        """ Initializes atmospheric fields

        Parameters
        ----------
        de_domain : DedalusDomain
            Contains info on the domain in which the problem is being solved
        add_fields  : list of strings
            Additional fields to add to self.atmo_fields
        """
        fds = ['T0', 'T0_z', 'T0_zz', 'rho0', 'ln_rho0', 'ln_rho0_z', 'phi', 'chi0', 'nu0']
        if type(add_fields) is list:
            fds += add_fields
        for f in fds:
            self.atmo_fields[f] = de_domain.new_ncc()

    def _set_subs(self, de_problem):
        """ Sets atmospheric substitutions in problem.

        Parameters
        ----------
        de_problem : DedalusProblem
            Contains info on the dedalus problem being solved.
        """
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
        """ Saves a file containing all atmospheric fields and parameters.

        Parameters
        ----------
        out_dir : string
            String containing path to output directory
        de_domain : DedalusDomain
            Contains information about the domain on which the atmosphere is built
        """
        import h5py
        mpi_makedirs(out_dir)
        out_file = '{:s}/atmosphere.h5'.format(out_dir)
        with h5py.File('{:s}'.format(out_file), 'w') as f:
            f['z'] = de_domain.generate_vertical_profile(de_domain.z)
            for k, p in self.atmo_params.items():
                f[k] = p
            for k, fd in self.atmo_fields.items():
                f[k] = de_domain.generate_vertical_profile(fd, scales=1)

class Polytrope(IdealGasAtmosphere):
    """
    An extension of an IdealGasAtmosphere for a polytropic stratification,
    nondimensionalized by default on the atmosphere's temperature gradient and the
    isothermal sound crossing time at the top.

    This class adds these additional keys to self.atmo_params:
        m_ad, m, n_rho, Lz, delta_s, t_buoy, t_therm
    """
    def __init__(self, *args, **kwargs):
        super(Polytrope, self).__init__(*args, **kwargs)

    def _prepare_scalar_info(self, epsilon=0, n_rho=3, gamma=5./3, R=1):
        """ Sets up scalar parameters in the atmosphere.

        Parameters
        ----------
            epsilon : float
                Superadiabatic excess of the polytrope (e.g., Anders & Brown 2017)
            n_rho : float
                Number of density scale heights of polytrope
            gamma : float
                Adiabatic index
            R : float
                Ideal gas constant (P = R rho T)
        """
        self.atmo_params['m_ad']     = 1/(gamma - 1)
        self.atmo_params['m']        = self.atmo_params['m_ad'] - epsilon
        self.atmo_params['g']        = R * (self.atmo_params['m'] + 1)
        self.atmo_params['n_rho']    = n_rho
        self.atmo_params['Lz']       = np.exp(self.atmo_params['n_rho']/self.atmo_params['m']) - 1
        self.atmo_params['delta_s']  = -R * epsilon*np.log(self.atmo_params['Lz'] + 1)
        super(Polytrope, self)._set_thermodynamics(gamma=gamma, R=R)
        self.atmo_params['t_buoy']   = np.sqrt(np.abs(self.atmo_params['Lz']*self.atmo_params['Cp'] / self.atmo_params['g'] / self.atmo_params['delta_s']))
        self.atmo_params['t_therm']  = 0 #fill in set_diffusivities
        logger.info('Initialized Polytrope:')
        logger.info('   epsilon = {:.2e}, m = {:.8g}, m_ad = {:.2g}'.format(epsilon, self.atmo_params['m'], self.atmo_params['m_ad']))
        logger.info('   Lz = {:.2g}, t_buoy = {:.2g}'.format(self.atmo_params['Lz'], self.atmo_params['t_buoy']))

    def build_atmosphere(self, de_domain):
        """
        Sets up atmosphere according to a polytropic stratification of the form:
            T0 =  1 + (Lz - z)
            rho0 = T0**m 

        Parameters
        ----------
            de_domain : DedalusDomain
                Contains information about the domain where the problem is specified
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

    def set_diffusivites(self, chi_top, nu_top):
        """
        Specifies diffusivity profiles of initial conditions. Initial diffusivites go like 1/rho,
        such that the dynamic diffusivites are constant in the initial atmosphere.

        Parameters
        ----------
        chi_top : float
            Thermal diffusivity at top of atmosphere (length^2 / time)
        nu_top : float
            Viscous diffusivity at top of atmosphere (length^2 / time)
        """
        self.atmo_fields['chi0']['g'] = chi_top/self.atmo_fields['rho0']['g']
        self.atmo_fields['nu0']['g']  =  nu_top/self.atmo_fields['rho0']['g']
        Lz = self.atmo_params['Lz']
        self.atmo_params['t_therm'] = Lz**2/np.mean(self.atmo_fields['chi0'].interpolate(z=Lz/2)['g'])
        [self.atmo_fields[k].set_scales(1, keep_data=True)  for k in ('chi0', 'nu0', 'rho0')]
        logger.info('Atmosphere set with top of atmosphere chi = {:.2e}, nu = {:.2e}'.format(chi_top, nu_top))
        logger.info('Atmospheric (midplane t_therm)/t_buoy = {:.2e}'.format(self.atmo_params['t_therm']/self.atmo_params['t_buoy']))
