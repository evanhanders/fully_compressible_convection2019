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

    def __init__(self, atmosphere, de_domain, de_problem):
        """Initialize the class

        Parameters
        ----------
        atmosphere  : Atmospehre object
            Contains info on the atmosphere in which FC equations will be solved
        de_domain   : DedalusDomain object
            Contains info regarding domain on which equations will be solved
        de_problem  : DedalusProblem object
            Contains info regarding problem on which equations will be solved
        """
        self.atmosphere = atmosphere
        self.de_domain  = de_domain
        self.de_problem = de_problem
        return


class FullyCompressibleEquations(Equations):
    """
    An abstract class containing the fully compressible equations which must
    be extended to specify the type of diffusivities
    """
    def __init__(self, thermal_BC_dict, velocity_BC_dict, *args, ncc_cutoff=1e-10, kx=0, ky=0):
        super(FullyCompressibleEquations, self).__init__(*args)

        variables = ['u','u_z','v', 'v_z', 'w','w_z','T1', 'T1_z', 'ln_rho1']
        if not self.de_domain.dimensions == 3:
            variables.remove('v')
            variables.remove('v_z')
        self.de_problem.set_variables(variables, ncc_cutoff=ncc_cutoff)
        self._set_parameters()
        self._set_basic_subs(kx=kx, ky=ky)
        self._set_diffusion_subs()
        self._set_output_subs()
        self._set_equations()
        self._set_BC(thermal_BC_dict, velocity_BC_dict)

    def _set_equations(self):
        """ 
        Sets the fully compressible equations of in a ln_rho / T formulation,
        as in e.g., Lecoanet et al. 2014 or Anders, Lecoanet, and Brown 2019.
        """ 
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

    def _set_BC(self, thermal_BC_dict, velocity_BC_dict):
        """
        Sets the velocity and thermal boundary conditions at the upper and lower boundaries.  
        Choose one thermal type of BC and one velocity type of BC to set those conditions.  
        See set_thermal_BC() and set_velocity_BC() functions for more info.

        Parameters
        ----------

        """
        self.dirichlet_set = []
        self._set_thermal_BC(thermal_BC_dict)
        self._set_velocity_BC(velocity_BC_dict)
        for key in self.dirichlet_set:
            self.de_problem.problem.meta[key]['z']['dirichlet'] = True
            
    def _set_thermal_BC(self, BC_dict):
        """
        Sets the thermal boundary conditions at the top and bottom of the atmosphere.  

        Parameters
        ----------

        """
        BC_set = False
        for k in BC_dict.keys():
            if BC_dict[k]: BC_set = True
        if not BC_set: BC_dict['flux_temp'] = True

        if BC_dict['flux']:
            logger.info("Thermal BC: fixed flux (full form)")
            self.de_problem.problem.add_bc("fixed_flux_L_LHS = fixed_flux_L_RHS")
            self.de_problem.problem.add_bc("fixed_flux_R_LHS = fixed_flux_R_RHS")
            self.dirichlet_set.append('T1_z')
        elif BC_dict['temp']:
            logger.info("Thermal BC: fixed temperature (T1)")
            self.de_problem.problem.add_bc( "left(T1) = 0")
            self.de_problem.problem.add_bc("right(T1) = 0")
            self.dirichlet_set.append('T1')
        elif BC_dict['flux_temp']:
            logger.info("Thermal BC: fixed flux/fixed temperature")
            self.de_problem.problem.add_bc("fixed_flux_L_LHS = fixed_flux_L_RHS")
            self.de_problem.problem.add_bc("right(T1)  = 0")
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('T1')
        elif BC_dict['temp_flux']:
            logger.info("Thermal BC: fixed temperature/fixed flux")
            self.de_problem.problem.add_bc("left(T1)    = 0")
            self.de_problem.problem.add_bc("fixed_flux_R_LHS = fixed_flux_R_RHS")
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('T1')
        else:
            logger.error("Incorrect thermal boundary conditions specified")
            raise

    def _set_velocity_BC(self, BC_dict):
        """
        Sets the velocity boundary conditions at the top and bottom of the atmosphere.  
        Boundaries are additionally impenetrable (w = 0 at top and bottom)

        Parameters
        ----------

        """
        BC_set = False
        for k in BC_dict.keys():
            if BC_dict[k]: BC_set = True
        if not BC_set: BC_dict['stress_free'] = True
            
        # horizontal velocity boundary conditions
        if BC_dict['stress_free']:
            logger.info("Horizontal velocity BC: stress free")
            self.de_problem.problem.add_bc("left(u_z) = 0")
            self.de_problem.problem.add_bc("right(u_z) = 0")
            self.dirichlet_set.append('u_z')
            if self.de_domain.dimensions == 3:
                self.de_problem.problem.add_bc("left(v_z) = 0")
                self.de_problem.problem.add_bc("right(v_z) = 0")
                self.dirichlet_set.append('v_z')
        elif BC_dict['no_slip']:
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

    def _set_basic_subs(self, kx = 0, ky = 0):
        if self.de_domain.dimensions == 1:
            self.de_problem.problem.parameters['j'] = 1j
            self.de_problem.problem.parameters['kx'] = kx
            self.de_problem.problem.parameters['kx'] = ky
            self.de_problem.problem.substitutions['dx(f)'] = "j*kx*(f)"
            self.de_problem.problem.substitutions['dy(f)'] = "j*ky*(f)"
        if not self.de_domain.dimensions == 3:
            self.de_problem.problem.parameters['v']   = 0
            self.de_problem.problem.parameters['v_z'] = 0 
            self.de_problem.problem.substitutions['dy(A)'] = '0*A'

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

        self.de_problem.problem.substitutions['Cv_inv']   = '(1/Cv)'
        self.de_problem.problem.substitutions['rho_full'] = '(rho0*exp(ln_rho1))'
        self.de_problem.problem.substitutions['rho_fluc'] = '(rho0*(exp(ln_rho1)-1))'
        self.de_problem.problem.substitutions['T_full']   = '(T0 + T1)'
        self.de_problem.problem.substitutions['P']        = '(R*rho_full*T_full)'
        self.de_problem.problem.substitutions['P0']       = '(R*rho0*T0)'
        self.de_problem.problem.substitutions['P1']       = '(P - P0)'
        self.de_problem.problem.substitutions['vel_rms']  = 'sqrt(u**2 + v**2 + w**2)'
        self.de_problem.problem.substitutions['s1']       = '(Cv*log(1+T1/T0) - R*ln_rho1)'
        self.de_problem.problem.substitutions['s_full']   = '(Cv*log(T_full) - R*(ln_rho0 + ln_rho1))'

    def _set_output_subs(self):
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
        self.de_problem.problem.substitutions['TE']        = '(KE + PE + IE)'
        self.de_problem.problem.substitutions['h']         = '(IE + P)'
        self.de_problem.problem.substitutions['PE_fluc']   = '(rho_fluc*phi)'
        self.de_problem.problem.substitutions['IE_fluc']   = '(rho_full*Cv*T1 + rho_fluc*Cv*T0)'
        self.de_problem.problem.substitutions['TE_fluc']   = '(KE + PE_fluc + IE_fluc)'
        self.de_problem.problem.substitutions['h_fluc']    = '(IE_fluc + P1)'

        self.de_problem.problem.substitutions['u_rms']      = 'sqrt(u**2)'
        self.de_problem.problem.substitutions['v_rms']      = 'sqrt(v**2)'
        self.de_problem.problem.substitutions['u_perp_rms'] = 'sqrt(u**2 + v**2)'
        self.de_problem.problem.substitutions['w_rms']      = 'sqrt(w**2)'
        self.de_problem.problem.substitutions['Re_rms']   = '(vel_rms*Lz/nu)'
        self.de_problem.problem.substitutions['Pe_rms']   = '(vel_rms*Lz/chi)'
        self.de_problem.problem.substitutions['Ma_iso_rms'] = '(vel_rms/sqrt(T_full))'
        self.de_problem.problem.substitutions['Ma_ad_rms']  = '(vel_rms/sqrt(gamma*T_full))'

        self.de_problem.problem.substitutions['enth_flux_z']    = '(w*h)'
        self.de_problem.problem.substitutions['KE_flux_z']      = '(w*KE)'
        self.de_problem.problem.substitutions['PE_flux_z']      = '(w*PE)'
        self.de_problem.problem.substitutions['viscous_flux_z'] = '(-rho_full * nu * (u*Sig_xz + v*Sig_yz + w*Sig_zz))'
        self.de_problem.problem.substitutions['F_conv_z']       = '(enth_flux_z + KE_flux_z + PE_flux_z + viscous_flux_z)'

        self.de_problem.problem.substitutions['F_cond_z']      = '(-1)*(kappa_full*dz(T_full))'
        self.de_problem.problem.substitutions['F_cond_fluc_z'] = '(-1)*(kappa_full*T1_z + kappa_fluc*T0_z)'
        self.de_problem.problem.substitutions['F_cond0_z']     = '(-1)*(rho0*chi0*T0_z)'
        self.de_problem.problem.substitutions['F_cond_ad_z']   = '(-1)*(kappa_full*T_ad_z)'

        self.de_problem.problem.substitutions['Nu'] = '((F_conv_z + F_cond_z - F_cond_ad_z)/vol_avg(F_cond_z - F_cond_ad_z))'

    def _set_parameters(self):
        for k, fd in self.atmosphere.atmo_fields.items():
            self.de_problem.problem.parameters[k] = fd
            fd.set_scales(1, keep_data=True)

        for k, p in self.atmosphere.atmo_params.items():
            self.de_problem.problem.parameters[k] = p

        if self.de_domain.dimensions > 1:
            self.de_problem.problem.parameters['Lx'] = self.de_domain.domain.bases[0].interval[1] - self.de_domain.domain.bases[0].interval[0]
        if self.de_domain.dimensions > 2:
            self.de_problem.problem.parameters['Ly'] = self.de_domain.domain.bases[1].interval[1] - self.de_domain.domain.bases[1].interval[0]

    def _set_diffusion_subs(self):
        pass


class KappaMuFCE(FullyCompressibleEquations):
    '''
    An extension of the fully compressible equations where the diffusivities are
    set based on kappa and mu, not chi and nu.
    '''

    def __init__(self, *args, **kwargs):
        super(KappaMuFCE, self).__init__(*args, **kwargs)

    def _set_diffusion_subs(self):
        """ Assumes chi0 = k / rho0 """
        self.de_problem.problem.substitutions['kappa_full']   = '(rho0*chi0)'
        self.de_problem.problem.substitutions['kappa_full_z'] = '(0)'
        self.de_problem.problem.substitutions['kappa_fluc']   = '(0)'
        self.de_problem.problem.substitutions['chi']          = '(chi0*exp(-ln_rho1))'
        self.de_problem.problem.substitutions['mu_full']      = '(rho0*nu0)'
        self.de_problem.problem.substitutions['mu_full_z']    = '(0)'
        self.de_problem.problem.substitutions['mu_fluc']      = '(0)'
        self.de_problem.problem.substitutions['nu']           = '(nu0*exp(-ln_rho1))'

        self.de_problem.problem.substitutions['scale_m_z']  = '(T0)'
        self.de_problem.problem.substitutions['scale_m']    = '(T0)'
        self.de_problem.problem.substitutions['scale_e']    = '(T0)'
        self.de_problem.problem.substitutions['scale_c']    = '(T0)'


        #Viscous subs -- momentum equation     
        self.de_problem.problem.substitutions['visc_u']   = "( (mu_full)*(Lap(u, u_z) + 1/3*Div(dx(u), dx(v), dx(w_z))) + (mu_full_z)*(Sig_xz))"
        self.de_problem.problem.substitutions['visc_v']   = "( (mu_full)*(Lap(v, v_z) + 1/3*Div(dy(u), dy(v), dy(w_z))) + (mu_full_z)*(Sig_yz))"
        self.de_problem.problem.substitutions['visc_w']   = "( (mu_full)*(Lap(w, w_z) + 1/3*Div(  u_z, dz(v), dz(w_z))) + (mu_full_z)*(Sig_zz))"                
#        self.de_problem.problem.substitutions['L_visc_u'] = "(visc_u/rho0)"
#        self.de_problem.problem.substitutions['L_visc_v'] = "(visc_v/rho0)"
#        self.de_problem.problem.substitutions['L_visc_w'] = "(visc_w/rho0)"                
        self.de_problem.problem.substitutions['L_visc_u'] = "(visc_u/T0)"
        self.de_problem.problem.substitutions['L_visc_v'] = "(visc_v/T0)"
        self.de_problem.problem.substitutions['L_visc_w'] = "(visc_w/T0)"                
        self.de_problem.problem.substitutions['R_visc_u'] = "(visc_u/rho_full - (L_visc_u))"
        self.de_problem.problem.substitutions['R_visc_v'] = "(visc_v/rho_full - (L_visc_v))"
        self.de_problem.problem.substitutions['R_visc_w'] = "(visc_w/rho_full - (L_visc_w))"




        self.de_problem.problem.substitutions['thermal'] = ('( ((1/Cv))*(kappa_full*Lap(T1, T1_z) + kappa_full_z*T1_z) )')
#        self.de_problem.problem.substitutions['L_thermal'] = ('thermal/rho0')
        self.de_problem.problem.substitutions['L_thermal'] = ('thermal/T0')
        self.de_problem.problem.substitutions['R_thermal'] = ('( thermal/rho_full - (L_thermal) + ((1/Cv)/(rho_full))*(kappa_full*T0_zz + kappa_full_z*T0_z) )' )
        self.de_problem.problem.substitutions['source_terms'] = '0'
        #Viscous heating
        self.de_problem.problem.substitutions['R_visc_heat']  = " (mu_full/rho_full*(1/Cv))*(dx(u)*Sig_xx + dy(v)*Sig_yy + w_z*Sig_zz + Sig_xy**2 + Sig_xz**2 + Sig_yz**2)"

        #Fixed-flux BC. LHS = RHS at boundary. L = left (lower). R = right (upper)
        self.de_problem.problem.substitutions['fixed_flux_L_LHS'] = "left(T1_z)"
        self.de_problem.problem.substitutions['fixed_flux_L_RHS'] = "(0)"
        self.de_problem.problem.substitutions['fixed_flux_R_LHS'] = "right(T1_z)"
        self.de_problem.problem.substitutions['fixed_flux_R_RHS'] = "(0)"

        super(KappaMuFCE, self)._set_diffusion_subs()


class AEKappaMuFCE(Equations):

    def __init__(self, thermal_BC_dict, avg_field_dict, *args, ncc_cutoff=1e-10, kx=0, ky=0):
        #Needs these parameters:
        super(AEKappaMuFCE, self).__init__(*args)
        variables = ['T1', 'T1_z', 'ln_rho1', 'M1']

        self.de_problem.set_variables(variables, ncc_cutoff=ncc_cutoff)
        self._set_parameters(avg_field_dict)
        self._set_equations()
        self._set_BCs(thermal_BC_dict)
    
    def _set_equations(self):
        """ 
        Sets the horizontally-averaged, stationary FC equations.
        
        Four equations are used in the FC BVP:
        (1) A simple definition of T1_z
        (2) A mass-accounting equation
        (3) Modified hydrostatic equilibrium
        (4) Vertical energy balance

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
        self.de_problem.problem.substitutions['w'] = '(w_prof_IVP)'
        self.de_problem.problem.substitutions['mod_udotgradW'] = '((udotgradW_horiz) + w * dz(w) )'
        self.de_problem.problem.substitutions['mod_viscous']   = ('(mu/AE_rho_full) * ( (4./3.)*dz(dz(w)))')
        self.de_problem.problem.substitutions['F_superad_initial'] = '-kappa*(T0_z - T_ad_z)'

        logger.debug('setting T1_z eqn')
        self.de_problem.problem.add_equation("dz(T1) - T1_z = 0")

        logger.debug('setting mass eqn')
        self.de_problem.problem.add_equation("dz(M1) = AE_rho_fluc")

        logger.debug('Setting energy equation')
        self.de_problem.problem.add_equation(("kappa*dz(T1_z) = dz(Xi * F_conv) - kappa*dz(T0_z - T_ad_z)"))
        
        logger.debug('Setting HS equation')
        self.de_problem.problem.add_equation(("T1_z + T1*dz(ln_rho0) + T0*dz(ln_rho1) ="+\
                              "-T1 * dz(ln_rho1)"))# - mod_udotgradW + mod_viscous "))
        
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
        for key in ['T1', 'T1_z', 'M1']:
            self.de_problem.problem.meta[key]['z']['dirichlet'] = True

    def _set_parameters(self, field_averager):
        for k in ['Xi', 'w_prof_IVP', 'udotgradW_horiz', 'mu', 'F_conv', 'kappa']:
            this_field = self.de_domain.new_ncc()
            this_field['g'] = field_averager[k]
            self.de_problem.problem.parameters[k] = this_field 
        for k, fd in self.atmosphere.atmo_fields.items():
            self.de_problem.problem.parameters[k] = fd
            fd.set_scales(1, keep_data=True)

        for k, p in self.atmosphere.atmo_params.items():
            self.de_problem.problem.parameters[k] = p


