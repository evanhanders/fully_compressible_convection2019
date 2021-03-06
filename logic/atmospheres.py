from collections import OrderedDict

import numpy as np
import logging
logger = logging.getLogger(__name__)

from mpi4py import MPI

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
        fds = ['T0', 'T0_z', 'T0_zz', 'rho0', 'ln_rho0', 'ln_rho0_z', 'phi', 'chi0', 'nu0', 'kappa0', 'mu0']
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
        z = de_domain.generate_vertical_profile(de_domain.z)
        if de_domain.domain.distributor.rank == 0:
            f = h5py.File('{:s}'.format(out_file), 'w')
            f['z'] = z

        for k, p in self.atmo_params.items():
            r = p
            if de_domain.domain.distributor.rank == 0:
                f[k] = r
        for k, fd in self.atmo_fields.items():
            r = de_domain.generate_vertical_profile(fd, scales=1)
            if de_domain.domain.distributor.rank == 0:
                f[k] = r

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
        self.atmo_params['epsilon']  = epsilon
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

        Parameters:
        ----------
        chi_top : float
            Thermal diffusivity at top of atmosphere (length^2 / time)
        nu_top : float
            Viscous diffusivity at top of atmosphere (length^2 / time)
        """
        self.atmo_fields['chi0']['g'] = chi_top/self.atmo_fields['rho0']['g']
        self.atmo_fields['nu0']['g']  =  nu_top/self.atmo_fields['rho0']['g']
        self.atmo_fields['kappa0']['g'] =  chi_top
        self.atmo_fields['mu0']['g']  =  nu_top
        Lz = self.atmo_params['Lz']
        self.atmo_params['t_therm'] = Lz**2/np.mean(self.atmo_fields['chi0'].interpolate(z=Lz)['g'])
        [self.atmo_fields[k].set_scales(1, keep_data=True)  for k in ('chi0', 'nu0', 'rho0', 'mu0', 'kappa0')]
        logger.info('Atmosphere set with top of atmosphere chi = {:.2e}, nu = {:.2e}'.format(chi_top, nu_top))
        logger.info('Atmospheric (midplane t_therm)/t_buoy = {:.2e}'.format(self.atmo_params['t_therm']/self.atmo_params['t_buoy']))

class DepthFinder:

    def __init__(self, nz=256):
        # Bases and domain
        import dedalus.public as de
        z_basis = de.Chebyshev('z', nz, interval=(0, 1), dealias=1)
        domain = de.Domain([z_basis], np.float64, comm=MPI.COMM_SELF)
        self.T0         = domain.new_field()
        self.T0_z       = domain.new_field()
        self.ln_rho0_z  = domain.new_field()
        self.H_rho      = domain.new_field()
        self.ln_rho0    = domain.new_field()
        self.z = domain.grid(0)

    def atmosphere_solve(self, g, Lz):
        """ Given T0, gravity, and the true domain depth, solves hydrostatic equilibrium for ln_rho """
        self.T0.differentiate('z', out=self.T0_z)
        self.T0_z['g'] /= Lz
        self.ln_rho0_z['g'] = -(self.T0_z['g'] + g)/self.T0['g']
        self.ln_rho0_z.antidifferentiate('z', ('right', 0), out=self.ln_rho0)
        self.ln_rho0['g'] *= Lz
        self.H_rho['g'] = -1/self.ln_rho0_z['g']

class TwoLayerIH(IdealGasAtmosphere):
    """
    An extension of an IdealGasAtmosphere for a very simple, 2-layer atmosphere,
    nondimensionalized by default on the atmosphere's adiabatic temperature gradient and the
    isothermal sound crossing time at the top.
    """
    def __init__(self, *args, **kwargs):
        super(TwoLayerIH, self).__init__(*args, **kwargs)

    def _prepare_scalar_info(self, epsilon=1e-4, n_rho_T=3, n_rho_B=1, gamma=5./3, R=1, tolerance=1e-12, stable_top=False):
        """ Sets up scalar parameters in the atmosphere.

        Parameters
        ----------
            epsilon : float
                Magnitude of the IH / flux
            gamma : float
                Adiabatic index
            R : float
                Ideal gas constant (P = R rho T)
        """
        self.atmo_params['n_rho_T']       = n_rho_T
        self.atmo_params['n_rho_B']       = n_rho_B
        self.atmo_params['m_ad'] = self.m_ad = 1/(gamma-1)
        self.atmo_params['g']    = self.g =  gamma*self.m_ad
        if stable_top: epsilon *= -1
        self.atmo_params['epsilon'] = self.epsilon = epsilon
        self.atmo_params['stable_top'] = self.stable_top = stable_top

        Ls = None
        if MPI.COMM_WORLD.rank == 0:
            finder = DepthFinder()

            L_T = np.exp(n_rho_T/(self.m_ad)) - 1
            L_B = np.exp((n_rho_T+n_rho_B)/(self.m_ad)) - 1 - L_T

            max_diff = 1
            while max_diff > tolerance:
                Lz = L_B + L_T
                logger.info('max_diff : {:.2e}'.format(max_diff))
                logger.info('Lz: {:.2g}, LT: {:.2g}, LB: {:.2g}'.format(Lz, L_T, L_B))

                A = -1./2
                B = L_B
                C = -(A*Lz**2 + B*Lz)
                z = Lz*finder.z
                if stable_top: eff_eps = self.epsilon/L_B
                else: eff_eps = self.epsilon/L_T
                finder.T0['g'] = 1 - (z-Lz) + eff_eps*(A*z**2 + B*z + C)
                finder.atmosphere_solve(self.g, Lz)

                #Problem: grad T is approaching the value of g, passing through zero, and so is ln_rho0_z. So H_rho explodes.

                H_rho_bot_T  = np.mean(finder.H_rho.interpolate(z=L_B/Lz)['g'])
                H_rho_bot_B = np.mean(finder.H_rho.interpolate(z=0)['g'])
                print('Hrho', H_rho_bot_T, H_rho_bot_B)

                this_n_rho_T = np.mean(finder.ln_rho0.interpolate(z=(L_B)/Lz)['g'])
                this_n_rho_B = np.mean(finder.ln_rho0.interpolate(z=0)['g']) - this_n_rho_T

                print('nrho', this_n_rho_T, this_n_rho_B)

                delta_nrho_T = n_rho_T - this_n_rho_T
                delta_nrho_B = n_rho_B - this_n_rho_B

                print('dnrho', delta_nrho_T, delta_nrho_B)

                delta_L_B, delta_L_T =  delta_nrho_B*H_rho_bot_B*np.exp(-delta_nrho_B), delta_nrho_T*H_rho_bot_T*np.exp(-delta_nrho_T)

                print('dL', delta_L_B, delta_L_T)

                max_diff = np.abs(np.array((delta_L_B/L_B, delta_L_T/L_T))).max()
                L_B += delta_L_B
                L_T += delta_L_T
            Lz = L_B + L_T
            Ls = (Lz, L_B, L_T)
        Lz, L_B, L_T = MPI.COMM_WORLD.bcast(Ls, root=0)
        self.atmo_params['Lz']   = self.Lz = Lz
        self.atmo_params['L_T'] = self.L_T = L_T
        self.atmo_params['L_B'] = self.L_B = L_B
        logger.info('Solving in TwoLayer atmosphere with length scales:')
        logger.info('   L_T = {:.2g}, L_B = {:.2g}'.format(L_T, L_B))
        super(TwoLayerIH, self)._set_thermodynamics(gamma=gamma, R=R)

    def build_atmosphere(self, de_domain):
        """
        Sets up atmosphere according to a polytropic stratification of the form:
            T0 =  1 + (Lz - z) + quadratic
            rho0 = T0**m 

        Parameters
        ----------
            de_domain : DedalusDomain
                Contains information about the domain where the problem is specified
        """ 
        self._make_atmo_fields(de_domain)
        z = de_domain.z
        epsilon = self.atmo_params['epsilon']
        A = -1./2
        B = self.L_B
        C = -(A*self.Lz**2 + B*self.Lz)

        if self.stable_top: eff_eps = self.epsilon/self.L_B
        else: eff_eps = self.epsilon/self.L_T
        T0 = 1 + self.atmo_params['T_ad_z']*(z - self.Lz) + eff_eps*(A*z**2 + B*z + C) 
        self.atmo_fields['T0']['g']      = T0
        self.atmo_fields['T0'].differentiate('z', out=self.atmo_fields['T0_z'])
        self.atmo_fields['T0_z'].differentiate('z', out=self.atmo_fields['T0_zz'])

        self.atmo_fields['T0'].set_scales(1, keep_data=True)
        self.atmo_fields['T0_z'].set_scales(1, keep_data=True)
        self.atmo_fields['ln_rho0_z']['g'] = - (self.atmo_params['g'] + self.atmo_fields['T0_z']['g'] ) / self.atmo_fields['T0']['g']
        self.atmo_fields['ln_rho0_z'].antidifferentiate('z', ('right', 0), out=self.atmo_fields['ln_rho0'])
        self.atmo_fields['ln_rho0'].set_scales(1, keep_data=True)
        self.atmo_fields['rho0']['g'] = np.exp(self.atmo_fields['ln_rho0']['g'])

        self.atmo_fields['phi']['g'] = -self.atmo_params['g']*(1 + self.atmo_params['T_ad_z']*(z - self.Lz))

        s0 = de_domain.domain.new_field()
        s0['g'] = self.atmo_params['Cp']*(1/self.atmo_params['gamma'])*(np.log(self.atmo_fields['T0']['g']) - (self.atmo_params['gamma']-1)*self.atmo_fields['ln_rho0']['g'])
        if self.stable_top:
            self.atmo_params['delta_s'] = np.abs(np.mean(s0.interpolate(z=self.L_B)['g'])-np.mean(s0.interpolate(z=0)['g']))
            self.atmo_params['t_buoy']   = np.sqrt(np.abs(self.L_B*self.atmo_params['Cp'] / self.atmo_params['g'] / self.atmo_params['delta_s']))
        else:
            self.atmo_params['delta_s'] = np.abs(np.mean(s0.interpolate(z=self.Lz)['g'])-np.mean(s0.interpolate(z=self.L_B)['g']))
            self.atmo_params['t_buoy']   = np.sqrt(np.abs(self.L_T*self.atmo_params['Cp'] / self.atmo_params['g'] / self.atmo_params['delta_s']))
        self.atmo_params['t_therm']  = 0 #fill in set_diffusivities
        logger.info('Initialized TriLayer:')
        logger.info('   epsilon = {:.2e}'.format(epsilon))
        logger.info('   Lz = {:.4g}, t_buoy = {:.4g}'.format(self.atmo_params['Lz'], self.atmo_params['t_buoy']))

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
        self.atmo_fields['chi0']['g'] =  chi_top/self.atmo_fields['rho0']['g']
        self.atmo_fields['nu0']['g']  =  nu_top/self.atmo_fields['rho0']['g']
        self.atmo_fields['kappa0']['g'] =  chi_top
        self.atmo_fields['mu0']['g']  =  nu_top
        Lz = self.atmo_params['Lz']
        self.atmo_params['t_therm'] = Lz**2/np.mean(self.atmo_fields['chi0'].interpolate(z=Lz/2)['g'])
        [self.atmo_fields[k].set_scales(1, keep_data=True)  for k in ('kappa0', 'mu0', 'chi0', 'nu0', 'rho0')]
        logger.info('Atmosphere set with top of atmosphere chi = {:.2e}, nu = {:.2e}'.format(chi_top, nu_top))
        logger.info('Atmospheric (midplane t_therm)/t_buoy = {:.2e}'.format(self.atmo_params['t_therm']/self.atmo_params['t_buoy']))

class TriLayerIH(IdealGasAtmosphere):
    """
    An extension of an IdealGasAtmosphere for a very simple, 3-layer RZ/CZ/RZ stratification,
    nondimensionalized by default on the atmosphere's adiabatic temperature gradient and the
    isothermal sound crossing time at the top.
    """
    def __init__(self, *args, **kwargs):
        super(TriLayerIH, self).__init__(*args, **kwargs)

    def _prepare_scalar_info(self, epsilon=1e-4, n_rho_cz=3, n_rho_rzT=4, n_rho_rzB=0.5, gamma=5./3, R=1, tolerance=1e-12):
        """ Sets up scalar parameters in the atmosphere.

        Parameters
        ----------
            epsilon : float
                Magnitude of the IH / flux
            gamma : float
                Adiabatic index
            R : float
                Ideal gas constant (P = R rho T)
        """
        self.atmo_params['n_rho_rzT']       = n_rho_rzT
        self.atmo_params['n_rho_rzB']       = n_rho_rzB
        self.atmo_params['n_rho_cz']        = n_rho_cz
        self.atmo_params['m_ad'] = self.m_ad = 1/(gamma-1)
        self.atmo_params['g']    = self.g =  gamma*self.m_ad
        self.atmo_params['epsilon'] = self.epsilon = epsilon

        Ls = None
        if MPI.COMM_WORLD.rank == 0:
            finder = DepthFinder()

            L_RT = np.exp(n_rho_rzT/(self.m_ad-self.epsilon)) - 1
            L_C  = np.exp((n_rho_cz+n_rho_rzT)/(self.m_ad-self.epsilon)) - 1 - L_RT
            L_RB = np.exp((n_rho_cz+n_rho_rzT+n_rho_rzB)/(self.m_ad-self.epsilon)) - 1 - L_RT - L_C

            max_diff = 1
            while max_diff > tolerance:
                Lz = L_RB + L_C + L_RT
                logger.info('max_diff : {:.2e}'.format(max_diff))
                logger.info('Lz: {:.2g}, Ls: {:.2g}, {:.2g}, {:.2g}'.format(L_RB+L_C+L_RT, L_RB, L_C, L_RT))

                A = 1./6
                B = -(L_RB + L_C/2)/2
                C = L_RB*(L_RB+L_C)/2
                D = -(A*Lz**3 + B*Lz**2 + C*Lz)
                z = Lz*finder.z
                finder.T0['g'] = 1 - (z-Lz) + self.epsilon*(A*z**3 + B*z**2 + C*z + D)
                finder.atmosphere_solve(self.g, Lz)

                H_rho_bot_rzB = np.mean(finder.H_rho.interpolate(z=0)['g'])
                H_rho_bot_cz  = np.mean(finder.H_rho.interpolate(z=L_RB/Lz)['g'])
                H_rho_bot_rzT = np.mean(finder.H_rho.interpolate(z=(L_RB+L_C)/Lz)['g'])

                this_n_rho_rzT = np.mean(finder.ln_rho0.interpolate(z=(L_RB+L_C)/Lz)['g'])
                this_n_rho_cz  = np.mean(finder.ln_rho0.interpolate(z=(L_RB)/Lz)['g']) - n_rho_rzT
                this_n_rho_rzB = np.mean(finder.ln_rho0.interpolate(z=0)['g']) - n_rho_cz - n_rho_rzT

                delta_nrho_rzT = n_rho_rzT - this_n_rho_rzT
                delta_nrho_cz  = n_rho_cz  - this_n_rho_cz
                delta_nrho_rzB = n_rho_rzB - this_n_rho_rzB
                import matplotlib.pyplot as plt
                plt.plot(finder.ln_rho0['g'])
                plt.show()

                delta_L_RB, delta_L_C, delta_L_RT =  np.array((delta_nrho_rzB*H_rho_bot_rzB, delta_nrho_cz*H_rho_bot_cz, delta_nrho_rzT*H_rho_bot_rzT))
                if np.abs(delta_L_RT/L_RT) > 0.1:
                    delta_L_RT /= np.abs(delta_L_RT) / ( 0.1*np.abs(delta_L_RT/L_RT))
                    delta_L_C = 0
                    delta_L_RB = 0
                elif np.abs(delta_L_C/L_C) > 0.:
                    delta_L_C /= np.abs(delta_L_C) / ( 0.1*np.abs(delta_L_C/L_C))
                    delta_L_RB = 0
                if np.abs(delta_L_RB/L_RB) > 0.1:
                    delta_L_RB /= np.abs(delta_L_RB) /( 0.1*np.abs(delta_L_RB/L_RB))

                max_diff = np.abs(np.array((delta_L_RB/L_RB, delta_L_C/L_C, delta_L_RT/L_RT))).max()
                L_RB += delta_L_RB
                L_C  += delta_L_C
                L_RT += delta_L_RT
            Lz = L_RB + L_C + L_RT
            Ls = (Lz, L_RB, L_C, L_RT)
        Lz, L_RB, L_C, L_RT = MPI.COMM_WORLD.bcast(Ls, root=0)
        self.atmo_params['Lz'] = self.Lz = Lz
        self.atmo_params['L_RT'] = self.L_RT = L_RT
        self.atmo_params['L_C'] = self.L_C = L_C
        self.atmo_params['L_RB'] = self.L_RB = L_RB
        logger.info('Solving in TriLayer atmosphere with length scales:')
        logger.info('   L_RT = {:.2g}, L_C = {:.2g}, L_RB = {:.2g}'.format(L_RT, L_C, L_RB))
        super(TriLayerIH, self)._set_thermodynamics(gamma=gamma, R=R)

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
        z = de_domain.z
        epsilon = self.atmo_params['epsilon']
        A = 1./6
        B = -(1./2)*(self.L_RB + self.L_C/2)
        C = (1./2)*(self.L_RB)*(self.L_RB + self.L_C)
        D = -(A*self.Lz**3 + B*self.Lz**2 + C*self.Lz)
        T0 = 1 + self.atmo_params['T_ad_z']*(z - self.Lz) + epsilon*(A*z**3 + B*z**2 + C*z + D) 
        self.atmo_fields['T0']['g']      = T0
        self.atmo_fields['T0'].differentiate('z', out=self.atmo_fields['T0_z'])
        self.atmo_fields['T0_z'].differentiate('z', out=self.atmo_fields['T0_zz'])

        self.atmo_fields['T0'].set_scales(1, keep_data=True)
        self.atmo_fields['T0_z'].set_scales(1, keep_data=True)
        self.atmo_fields['ln_rho0_z']['g'] = - (self.atmo_params['g'] + self.atmo_fields['T0_z']['g'] ) / self.atmo_fields['T0']['g']
        self.atmo_fields['ln_rho0_z'].antidifferentiate('z', ('right', 0), out=self.atmo_fields['ln_rho0'])
        self.atmo_fields['ln_rho0'].set_scales(1, keep_data=True)
        self.atmo_fields['rho0']['g'] = np.exp(self.atmo_fields['ln_rho0']['g'])

        self.atmo_fields['phi']['g'] = -self.atmo_params['g']*(1 + self.atmo_params['T_ad_z']*(z - self.Lz))

        s0 = de_domain.domain.new_field()
        s0['g'] = self.atmo_params['Cp']*(1/self.atmo_params['gamma'])*(np.log(self.atmo_fields['T0']['g']) - (self.atmo_params['gamma']-1)*self.atmo_fields['ln_rho0']['g'])
        self.atmo_params['delta_s'] = np.abs(np.mean(s0.interpolate(z=self.L_RB+self.L_C)['g'])-np.mean(s0.interpolate(z=self.L_RB)['g']))
        
        self.atmo_params['t_buoy']   = np.sqrt(np.abs(self.atmo_params['L_C']*self.atmo_params['Cp'] / self.atmo_params['g'] / self.atmo_params['delta_s']))
        self.atmo_params['t_therm']  = 0 #fill in set_diffusivities
        logger.info('Initialized TriLayer:')
        logger.info('   epsilon = {:.2e}'.format(epsilon))
        logger.info('   Lz = {:.2g}, t_buoy = {:.2g}'.format(self.atmo_params['Lz'], self.atmo_params['t_buoy']))

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
        self.atmo_fields['chi0']['g'] =  chi_top/self.atmo_fields['rho0']['g']
        self.atmo_fields['nu0']['g']  =  nu_top/self.atmo_fields['rho0']['g']
        self.atmo_fields['kappa0']['g'] =  chi_top
        self.atmo_fields['mu0']['g']  =  nu_top
        Lz = self.atmo_params['Lz']
        self.atmo_params['t_therm'] = Lz**2/np.mean(self.atmo_fields['chi0'].interpolate(z=Lz/2)['g'])
        [self.atmo_fields[k].set_scales(1, keep_data=True)  for k in ('kappa0', 'mu0', 'chi0', 'nu0', 'rho0')]
        logger.info('Atmosphere set with top of atmosphere chi = {:.2e}, nu = {:.2e}'.format(chi_top, nu_top))
        logger.info('Atmospheric (midplane t_therm)/t_buoy = {:.2e}'.format(self.atmo_params['t_therm']/self.atmo_params['t_buoy']))
