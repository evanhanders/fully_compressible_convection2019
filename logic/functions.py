import numpy as np
import os
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

def clean_makedir(data_dir):
    import mpi4py.MPI
    if mpi4py.MPI.COMM_WORLD.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.makedirs('{:s}/'.format(data_dir))
        logdir = os.path.join(data_dir,'logs')
        if not os.path.exists(logdir):
            os.mkdir(logdir)

