import numpy as np
import os
import logging
logger = logging.getLogger(__name__)
from collections import OrderedDict



def global_noise(de_domain, seed=42, **kwargs):
    """
    Create a field fielled with random noise of order 1.  Modify seed to
    get varying noise, keep seed the same to directly compare runs.

    Parameters
    ----------
    de_domain   : A DedalusDomain object
        Contains information about the simulation domain
    seed        : int, optional
        The seed used in determining the random noise field
    kwargs      : dict, optional
        Additional keyword arguments for the filter_field() function
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
    a given threshold.  This is accomplished by reducing the scale of the field,
    and then forcing the field into coefficient and then grid space.

    Parameters
    ----------
    field   : a Field object from the Dedalus package
        The field to filter
    frac    : float, optional
        A number between 0 and 1, the fraction of coefficients to keep power in.
    """
    dom = field.domain
    logger.info("filtering field {} with frac={} using a set-scales approach".format(field.name,frac))
    orig_scale = field.meta[:]['scale']
    field.set_scales(frac, keep_data=True)
    field['c']
    field['g']
    field.set_scales(orig_scale, keep_data=True)

def mpi_makedirs(data_dir):
    """Create a directory in an MPI-safe way.

    Parameters
    ----------
    data_dir    : string
        The path to the directory being created (either a local path or global path)
    """
    import mpi4py.MPI
    if mpi4py.MPI.COMM_WORLD.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.makedirs('{:s}/'.format(data_dir))
