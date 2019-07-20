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
    analysis_scalar = de_problem.solver.evaluator.add_file_handler(data_dir+"scalar", max_writes=max_writes, parallel=False, sim_dt=output_dt, mode=mode)

    basic_fields = ['u_rms', 'w_rms', 'vel_rms', 'T1', 'T1_z', 'T_full', 'ln_rho1', 'rho_full', 'enstrophy', 's1', 's_full']
    if de_domain.dimensions == 3: basic_fields += ['v_rms', 'u_perp_rms']
    fluid_numbers = ['Re_rms', 'Pe_rms', 'Ma_iso_rms', 'Ma_ad_rms', 'Nu']
    energies = ['KE', 'PE', 'IE', 'TE', 'PE_fluc', 'IE_fluc', 'TE_fluc']
    fluxes = ['KE_flux_z', 'PE_flux_z', 'enth_flux_z', 'viscous_flux_z', 'F_cond_fluc_z', 'F_cond_z']
    out_fields = basic_fields + fluid_numbers + energies + fluxes

    for field in out_fields:
        analysis_profile.add_task("plane_avg({})".format(field), name=field)
        analysis_scalar.add_task("vol_avg({})".format(field), name=field)
   
    analysis_profile.add_task("plane_avg(sqrt(T1**2))", name="T1_rms")
    analysis_scalar.add_task( "vol_avg(sqrt(T1**2))", name="T1_rms")
    analysis_profile.add_task("plane_avg(g*dz(s_full)/Cp)", name="brunt_squared")
    analysis_scalar.add_task( "vol_avg(  g*dz(s_full)/Cp)", name="brunt_squared")
    analysis_scalar.add_task( "integ(  rho_full - rho0)", name="M1")
    analysis_scalar.add_task("(plane_avg(right(F_cond_z)) - plane_avg(left(F_cond_z)))", name="flux_equilibration")
    analysis_scalar.add_task("(plane_avg(right(F_cond_z)) - plane_avg(left(F_cond_z)))/plane_avg(left(F_cond_z))",name="flux_equilibration_pct")

    analysis_tasks['profile'] = analysis_profile
    analysis_tasks['scalar'] = analysis_scalar

    if de_domain.dimensions == 2:
        # Analysis
        slices = de_problem.solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=output_dt, max_writes=max_writes, mode=mode)
        for field in ['s1', 'enstrophy', 'vel_rms', 'u', 'w', 'T1', 'Vort_y']:
            slices.add_task(field)
        slices.add_task("(s1 - plane_avg(s1))", name='s1_plane_fluc')
        analysis_tasks['slices'] = slices

    if de_domain.dimensions == 3:
        Lx, Ly, Lz = de_domain.resolution
        slices = de_problem.solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=output_dt, max_writes=max_writes, mode=mode)
        slices.add_task("interp(s_full,         y={})".format(Ly/2), name='s')
        slices.add_task("interp(s_full,         z={})".format(0.95*Lz), name='s near top')
        slices.add_task("interp(s_full,         z={})".format(Lz/2), name='s midplane')
        slices.add_task("interp(w,         y={})".format(Ly/2), name='w')
        slices.add_task("interp(w,         z={})".format(0.95*Lz), name='w near top')
        slices.add_task("interp(w,         z={})".format(Lz/2), name='w midplane')
        slices.add_task("interp(enstrophy,         y={})".format(Ly/2),    name='enstrophy')
        slices.add_task("interp(enstrophy,         z={})".format(0.95*Lz), name='enstrophy near top')
        slices.add_task("interp(enstrophy,         z={})".format(Lz/2),    name='enstrophy midplane')
        analysis_tasks['slices'] = slices

        if volumes_output:
            analysis_volume = de_problem.solver.evaluator.add_file_handler(data_dir+'volumes', sim_dt=output_vol_dt, max_writes=max_vol_writes, mode=mode)
            analysis_volume.add_task("T1 + T0", name="T")
            analysis_volume.add_task("enstrophy", name="enstrophy")
            analysis_volume.add_task("Oz", name="z_vorticity")
            analysis_tasks['volumes'] = analysis_volume

    return analysis_tasks

