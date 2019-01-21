import numpy as np
import logging
logger = logging.getLogger(__name__)
from collections import OrderedDict

def initialize_output(de_domain, de_problem, data_dir,
                      max_writes=20, max_vol_writes=2, output_dt=0.25, output_vol_dt=1,
                      mode="overwrite", volumes_output=False, coeff_output=False):
    """
    Sets up Dedalus output tasks for a Boussinesq convection run.

    Parameters
    ----------
    de_domain       : DedalusDomain object
        Contains information about the dedalus domain of the simulation
    de_problem      : DedalusProblem object
        Contains information aobut the dedalus problem & solver of the simulation
    data_dir        : string
        path to root data directory
    max_writes      : int, optional
        Maximum number of simulation output writes per file
    max_vol_writes  : int, optional
        Maximum number os imulations output writes per 3D volume file
    output_dt       : float, optional
        Simulation time between output writes
    output_vol_dt   : float, optional
        Simulation time between 3D volume output writes.
    mode            : string, optional
        Write mode for dedalus, "overwrite" or "append"
    volumes_output  : bool, optional
        If True, write 3D volumes
    coeff_output    : bool, optional
        If True, write coefficient data
    """

    analysis_tasks = analysis_tasks = OrderedDict()

    analysis_profile = de_problem.solver.evaluator.add_file_handler(data_dir+"profiles", max_writes=max_writes, parallel=False, sim_dt=output_dt, mode=mode)
    analysis_profile.add_task("plane_avg(T1)", name="T1")
    analysis_profile.add_task("plane_avg(T_full)", name="T_full")
    analysis_profile.add_task("plane_avg(Ma_iso_rms)", name="Ma_iso")
    analysis_profile.add_task("plane_avg(Ma_ad_rms)", name="Ma_ad")
    analysis_profile.add_task("plane_avg(ln_rho1)", name="ln_rho1")
    analysis_profile.add_task("plane_avg(rho_full)", name="rho_full")
    analysis_profile.add_task("plane_avg(KE)", name="KE")
    analysis_profile.add_task("plane_avg(PE)", name="PE")
    analysis_profile.add_task("plane_avg(IE)", name="IE")
    analysis_profile.add_task("plane_avg(PE_fluc)", name="PE_fluc")
    analysis_profile.add_task("plane_avg(IE_fluc)", name="IE_fluc")
    analysis_profile.add_task("plane_avg(KE + PE + IE)", name="TE")
    analysis_profile.add_task("plane_avg(KE + PE_fluc + IE_fluc)", name="TE_fluc")

    analysis_profile.add_task("plane_avg(KE_flux_z)", name="KE_flux_z")
    analysis_profile.add_task("plane_avg(PE_flux_z)", name="PE_flux_z")
    analysis_profile.add_task("plane_avg(w*(IE))", name="IE_flux_z")
    analysis_profile.add_task("plane_avg(enth_flux_z)",  name="enthalpy_flux_z")
    analysis_profile.add_task("plane_avg(viscous_flux_z)",  name="viscous_flux_z")
#    analysis_profile.add_task("plane_avg(kappa_flux_z)", name="kappa_flux_z")
#    analysis_profile.add_task("plane_avg(kappa_flux_fluc)", name="kappa_flux_fluc_z")
#    analysis_profile.add_task("plane_avg(kappa_flux_z - kappa_adiabatic_flux_z_G75)", name="kappa_flux_z_minus_ad_G75")
#    analysis_profile.add_task("plane_avg(kappa_flux_z - kappa_adiabatic_flux_z_AB17)", name="kappa_flux_z_minus_ad_AB17")
#    analysis_profile.add_task("plane_avg(kappa_flux_z-kappa_adiabatic_flux_z_G75)/vol_avg(Nusselt_norm_G75)", name="norm_kappa_flux_z_G75")
#    analysis_profile.add_task("plane_avg(kappa_flux_z-kappa_adiabatic_flux_z_AB17)/vol_avg(Nusselt_norm_AB17)", name="norm_kappa_flux_z_AB17")
#    analysis_profile.add_task("plane_avg(Nusselt_G75)", name="Nusselt_G75")
#    analysis_profile.add_task("plane_avg(Nusselt_AB17)", name="Nusselt_AB17")
#    analysis_profile.add_task("plane_avg(u_rms)", name="u_rms")
#    analysis_profile.add_task("plane_avg(w_rms)", name="w_rms")
#    analysis_profile.add_task("plane_avg(vel_rms)", name="vel_rms")
#    analysis_profile.add_task("plane_avg(Re_rms)", name="Re_rms")
#    analysis_profile.add_task("plane_avg(Pe_rms)", name="Pe_rms")
#    analysis_profile.add_task("plane_avg(enstrophy)", name="enstrophy")
#    analysis_profile.add_task("plane_std(enstrophy)", name="enstrophy_std")        
#    analysis_profile.add_task("plane_avg(Rayleigh_global)", name="Rayleigh_global")
#    analysis_profile.add_task("plane_avg(Rayleigh_local)",  name="Rayleigh_local")
#    analysis_profile.add_task("plane_avg(s_fluc)", name="s_fluc")
#    analysis_profile.add_task("plane_std(s_fluc)", name="s_fluc_std")
#    analysis_profile.add_task("plane_avg(s_mean)", name="s_mean")
#    analysis_profile.add_task("plane_avg(s_fluc + s_mean)", name="s_tot")
#    analysis_profile.add_task("plane_avg(dz(s_fluc))", name="grad_s_fluc")        
#    analysis_profile.add_task("plane_avg(dz(s_mean))", name="grad_s_mean")        
#    analysis_profile.add_task("plane_avg(dz(s_fluc + s_mean))", name="grad_s_tot")
#    analysis_profile.add_task("plane_avg(g*dz(s_fluc)*Cp_inv)", name="brunt_squared_fluc")        
#    analysis_profile.add_task("plane_avg(g*dz(s_mean)*Cp_inv)", name="brunt_squared_mean")        
#    analysis_profile.add_task("plane_avg(g*dz(s_fluc + s_mean)*Cp_inv)", name="brunt_squared_tot")
#    analysis_profile.add_task("interp(sqrt(u**2 + v**2), z={})".format(0.95*self.Lz), name="mag_vh_0.95")
#    analysis_profile.add_task("interp(u, z={})".format(0.95*self.Lz), name="vh_0.95")
#    analysis_profile.add_task("interp(s_fluc, z={})".format(0.95*self.Lz), name="s_0.95")
#    analysis_profile.add_task("interp(u, z={})".format(self.Lz), name="vh_top")
#    analysis_profile.add_task("interp(s_fluc, z={})".format(self.Lz), name="s_top")
#
#    analysis_tasks['profile'] = analysis_profile
#
#    analysis_scalar = solver.evaluator.add_file_handler(data_dir+"scalar", max_writes=max_writes, parallel=False,
#                                                        mode=mode, **kwargs)
#    analysis_scalar.add_task("vol_avg(KE)", name="KE")
#    analysis_scalar.add_task("vol_avg(PE)", name="PE")
#    analysis_scalar.add_task("vol_avg(IE)", name="IE")
#    analysis_scalar.add_task("vol_avg(PE_fluc)", name="PE_fluc")
#    analysis_scalar.add_task("vol_avg(IE_fluc)", name="IE_fluc")
#    analysis_scalar.add_task("vol_avg(KE + PE + IE)", name="TE")
#    analysis_scalar.add_task("vol_avg(KE + PE_fluc + IE_fluc)", name="TE_fluc")
#    analysis_scalar.add_task("vol_avg(u_rms)", name="u_rms")
#    analysis_scalar.add_task("vol_avg(w_rms)", name="w_rms")
#    analysis_scalar.add_task("vol_avg(Re_rms)", name="Re_rms")
#    analysis_scalar.add_task("vol_avg(Pe_rms)", name="Pe_rms")
#    analysis_scalar.add_task("vol_avg(Ma_iso_rms)", name="Ma_iso")
#    analysis_scalar.add_task("vol_avg(Ma_ad_rms)", name="Ma_ad")
#    analysis_scalar.add_task("vol_avg(enstrophy)", name="enstrophy")
#    analysis_scalar.add_task("vol_avg(Nusselt_G75)", name="Nusselt_G75")
#    analysis_scalar.add_task("vol_avg(Nusselt_AB17)", name="Nusselt_AB17")
#    analysis_scalar.add_task("vol_avg(Nusselt_norm_G75)", name="Nusselt_norm_G75")
#    analysis_scalar.add_task("vol_avg(Nusselt_norm_AB17)", name="Nusselt_norm_AB17")
#    analysis_scalar.add_task("log(left(plane_avg(rho_full))/right(plane_avg(rho_full)))", name="n_rho")
#    analysis_scalar.add_task("(plane_avg(right(kappa_flux_z)) - plane_avg(left(kappa_flux_z)))", name="flux_equilibration")
#    analysis_scalar.add_task("(plane_avg(right(kappa_flux_z)) - plane_avg(left(kappa_flux_z)))/plane_avg(left(kappa_flux_z))",name="flux_equilibration_pct")
#        
#    analysis_tasks['scalar'] = analysis_scalar
#
#    if coeff_output:
#        analysis_coeff = solver.evaluator.add_file_handler(data_dir+"coeffs", max_writes=max_writes, parallel=False,
#                                                           mode=mode, **kwargs)
#        analysis_coeff.add_task("s_fluc", name="s", layout='c')
#        analysis_coeff.add_task("s_fluc - plane_avg(s_fluc)", name="s'", layout='c')
#        analysis_coeff.add_task("T1+T0", name="T", layout='c')
#        analysis_coeff.add_task("T1+T0 - plane_avg(T1+T0)", name="T'", layout='c')
#        analysis_coeff.add_task("ln_rho1+ln_rho0", name="ln_rho", layout='c')
#        analysis_coeff.add_task("ln_rho1+ln_rho0 - plane_avg(ln_rho1+ln_rho0)", name="ln_rho'", layout='c')
#        analysis_coeff.add_task("u", name="u", layout='c')
#        analysis_coeff.add_task("w", name="w", layout='c')
#        analysis_coeff.add_task("enstrophy", name="enstrophy", layout='c')
#        analysis_coeff.add_task("(u_z - dx(w))", name="vorticity", layout='c')
#        analysis_tasks['coeff'] = analysis_coeff
#    
#    if de_domain.dimensions == 2:
#        # Analysis
#        slices = de_problem.solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=output_dt, max_writes=max_writes, mode=mode)
#        slices.add_task("s_fluc", name='s1')
#        slices.add_task("(s_fluc - plane_avg(s_fluc))", name='s_fluc')
#        slices.add_task("enstrophy")
#        slices.add_task("vel_rms")
#        slices.add_task("u")
#        slices.add_task("w")
#        analysis_tasks['slices'] = slices
#
##        if coeff_output:
##            coeffs = de_problem.solver.evaluator.add_file_handler(data_dir+'coeffs', sim_dt=output_dt, max_writes=max_writes, mode=mode)
##            coeffs.add_task("T1+T0", name="T", layout='c')
##            coeffs.add_task("T1 - plane_avg(T1)", name="T'", layout='c')
##            coeffs.add_task("w", layout='c')
##            coeffs.add_task("u", layout='c')
##            coeffs.add_task("enstrophy", layout='c')
##            analysis_tasks['coeffs'] = coeffs
##
##    if de_domain.dimensions == 3:
##        slices = de_problem.solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=output_dt, max_writes=max_writes, mode=mode)
##        slices.add_task("interp(T1 + T0,         y={})".format(de_domain.Ly/2), name='T')
##        slices.add_task("interp(T1 + T0,         z={})".format(0.95*de_domain.Lz), name='T near top')
##        slices.add_task("interp(T1 + T0,         z={})".format(de_domain.Lz/2), name='T midplane')
##        slices.add_task("interp(w,         y={})".format(de_domain.Ly/2), name='w')
##        slices.add_task("interp(w,         z={})".format(0.95*de_domain.Lz), name='w near top')
##        slices.add_task("interp(w,         z={})".format(de_domain.Lz/2), name='w midplane')
##        slices.add_task("interp(enstrophy,         y={})".format(de_domain.Ly/2),    name='enstrophy')
##        slices.add_task("interp(enstrophy,         z={})".format(0.95*de_domain.Lz), name='enstrophy near top')
##        slices.add_task("interp(enstrophy,         z={})".format(de_domain.Lz/2),    name='enstrophy midplane')
##        analysis_tasks['slices'] = slices
##
##        analysis_tasks['profiles'].add_task('plane_avg(Oz)', name="z_vorticity")
##        analysis_tasks['scalar'].add_task('vol_avg(Rossby)', name='Ro')
##
##        if volumes_output:
##            analysis_volume = de_problem.solver.evaluator.add_file_handler(data_dir+'volumes', sim_dt=output_vol_dt, max_writes=max_vol_writes, mode=mode)
##            analysis_volume.add_task("T1 + T0", name="T")
##            analysis_volume.add_task("enstrophy", name="enstrophy")
##            analysis_volume.add_task("Oz", name="z_vorticity")
##            analysis_tasks['volumes'] = analysis_volume

    return analysis_tasks

