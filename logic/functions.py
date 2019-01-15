import numpy as np
import logging
logger = logging.getLogger(__name__)
from collections import OrderedDict



def global_noise(de_domain, seed=42, **kwargs):
    """
    Create a field fielled with random noise of order 1.  Modify seed to
    get varying noise, keep seed the same to directly compare runs.
    """
    # Random perturbations, initialized globally for same results in parallel
    gshape = de_domain.domain.dist.grid_layout.global_shape(scales=de_domain.dealias)
    slices = de_domain.domain.dist.grid_layout.slices(scales=de_domain.dealias)
    rand = np.random.RandomState(seed=seed)
    noise = rand.standard_normal(gshape)[slices]

    # filter in k-space
    noise_field = de_domain.domain.new_field()
    noise_field.set_scales(de_domain.dealias, keep_data=False)
    noise_field['g'] = noise
    filter_field(noise_field, **kwargs)

    return noise_field

def filter_field(field, frac=0.25):
    """
    Filter a field in coefficient space by cutting off all coefficient above
    a given threshold.  This is accomplished by changing the scale of a field,
    forcing it into coefficient space at that small scale, then coming back to
    the original scale.

    Inputs:
        field   - The dedalus field to filter
        frac    - The fraction of coefficients to KEEP POWER IN.  If frac=0.25,
                    The upper 75% of coefficients are set to 0.
    """
    dom = field.domain
    logger.info("filtering field {} with frac={} using a set-scales approach".format(field.name,frac))
    orig_scale = field.meta[:]['scale']
    field.set_scales(frac, keep_data=True)
    field['c']
    field['g']
    field.set_scales(orig_scale, keep_data=True)

def initialize_output(solver, de_domain, data_dir,
                      max_writes=20, output_dt=0.25, max_vol_writes=2,
                      mode="overwrite", volumes_output=False, coeff_output=False, volumes_dt=1):
    """
    Sets up output from runs.
    """

    # Analysis
    analysis_tasks = OrderedDict()
    profiles = solver.evaluator.add_file_handler(data_dir+'profiles', sim_dt=output_dt, max_writes=max_writes, mode=mode)
    profiles.add_task("plane_avg(T1+T0)", name="T")
    profiles.add_task("plane_avg(dz(T1+T0))", name="Tz")
    profiles.add_task("plane_avg(T1)", name="T1")
    profiles.add_task("plane_avg(u)", name="u")
    profiles.add_task("plane_avg(w)", name="w")
    profiles.add_task("plane_avg(enstrophy)", name="enstrophy")
    profiles.add_task("plane_avg(Nu)", name="Nu")
    profiles.add_task("plane_avg(Re)", name="Re")
    profiles.add_task("plane_avg(Pe)", name="Pe")
    profiles.add_task("plane_avg(enth_flux_z)", name="enth_flux")
    profiles.add_task("plane_avg(kappa_flux_z)", name="kappa_flux")
    profiles.add_task("plane_avg(kappa_flux_z + enth_flux_z)", name="tot_flux")

    analysis_tasks['profiles'] = profiles

    scalar = solver.evaluator.add_file_handler(data_dir+'scalar', sim_dt=output_dt, max_writes=max_writes, mode=mode)
    scalar.add_task("vol_avg(T1)", name="IE")
    scalar.add_task("vol_avg(KE)", name="KE")
    scalar.add_task("vol_avg(T1) + vol_avg(KE)", name="TE")
    scalar.add_task("0.5*vol_avg(u_fluc*u_fluc+w_fluc*w_fluc)", name="KE_fluc")
    scalar.add_task("0.5*vol_avg(u*u)", name="KE_x")
    scalar.add_task("0.5*vol_avg(w*w)", name="KE_z")
    scalar.add_task("0.5*vol_avg(u_fluc*u_fluc)", name="KE_x_fluc")
    scalar.add_task("0.5*vol_avg(w_fluc*w_fluc)", name="KE_z_fluc")
    scalar.add_task("vol_avg(plane_avg(u)**2)", name="u_avg")
    scalar.add_task("vol_avg((u - plane_avg(u))**2)", name="u1")
    scalar.add_task("vol_avg(Nu)", name="Nu")
    scalar.add_task("vol_avg(Re)", name="Re")
    scalar.add_task("vol_avg(Pe)", name="Pe")
    scalar.add_task("delta_T", name="delta_T")
    analysis_tasks['scalar'] = scalar


    if de_domain.dimensions == 2:
        # Analysis
        slices = solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=output_dt, max_writes=max_writes, mode=mode)
        slices.add_task("T1 + T0", name='T')
        slices.add_task("enstrophy")
        slices.add_task("vel_rms")
        slices.add_task("u")
        slices.add_task("w")
        analysis_tasks['slices'] = slices

        if coeff_output:
            coeffs = solver.evaluator.add_file_handler(data_dir+'coeffs', sim_dt=output_dt, max_writes=max_writes, mode=mode)
            coeffs.add_task("T1+T0", name="T", layout='c')
            coeffs.add_task("T1 - plane_avg(T1)", name="T'", layout='c')
            coeffs.add_task("w", layout='c')
            coeffs.add_task("u", layout='c')
            coeffs.add_task("enstrophy", layout='c')
            analysis_tasks['coeffs'] = coeffs

    if de_domain.dimensions == 3:
        slices = solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=output_dt, max_writes=max_writes, mode=mode)
        slices.add_task("interp(T1 + T0,         y={})".format(de_domain.Ly/2), name='T')
        slices.add_task("interp(T1 + T0,         z={})".format(0.95*de_domain.Lz), name='T near top')
        slices.add_task("interp(T1 + T0,         z={})".format(de_domain.Lz/2), name='T midplane')
        slices.add_task("interp(w,         y={})".format(de_domain.Ly/2), name='w')
        slices.add_task("interp(w,         z={})".format(0.95*de_domain.Lz), name='w near top')
        slices.add_task("interp(w,         z={})".format(de_domain.Lz/2), name='w midplane')
        slices.add_task("interp(enstrophy,         y={})".format(de_domain.Ly/2),    name='enstrophy')
        slices.add_task("interp(enstrophy,         z={})".format(0.95*de_domain.Lz), name='enstrophy near top')
        slices.add_task("interp(enstrophy,         z={})".format(de_domain.Lz/2),    name='enstrophy midplane')
        analysis_tasks['slices'] = slices

        analysis_tasks['profiles'].add_task('plane_avg(Oz)', name="z_vorticity")
        analysis_tasks['scalar'].add_task('vol_avg(Rossby)', name='Ro')

        if volumes_output:
            analysis_volume = solver.evaluator.add_file_handler(data_dir+'volumes', sim_dt=volumes_dt, max_writes=max_vol_writes, mode=mode)
            analysis_volume.add_task("T1 + T0", name="T")
            analysis_volume.add_task("enstrophy", name="enstrophy")
            analysis_volume.add_task("Oz", name="z_vorticity")
            analysis_tasks['volumes'] = analysis_volume

    return analysis_tasks

