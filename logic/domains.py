import logging
logger = logging.getLogger(__name__)

import numpy as np
from mpi4py import MPI
from dedalus import public as de
from dedalus.core.field import Field


class DedalusDomain:
    """
    A simple class which contains a dedalus domain, as well as a bunch of info about
    the domain, for use in broad functions.

    Attributes:
    -----------
        atmosphere       : An atmosphere class
            The atmosphere on which the domain will be built
        resolution       : Tuple of integers
            A 2- or 3- element tuple containing (nz, nx) or (nz, nx, ny)
        aspect           : float
            The aspect ratio of the domain. Lx = Ly = aspect*Lz.
        bases            : List of Dedalus basis objects.
            A list of the basis objects in the domain
        domain           : The dedalus domain object
            The thing that this class is wrapping for easy access / expansion of
        dealias          : Float. 
            multiplicative factor for # of gridpoints on dealiased grid.
        mesh             : List of integers.
            Mesh for dividing processor. The product of list elements must == # of cpus.
        comm             : MPI communicator.
            The MPI communicator object for dedalus (generally COMM_WORLD or COMM_SELF)
        dtype            : A NumPy datatype
            The grid dtype of dedalus.
        x, y, z          : NumPy Arrays
            The gridpoints of the domain for x,y,z (dealias=1)
        x_de, y_de, z_de : NumPy Arrays
            The gridpoints of the domain for x,y,z (dealias=dealias)
    """

    def __init__(self, atmosphere, resolution, aspect, mesh=None, 
                       dealias=3/2, comm=MPI.COMM_WORLD, dtype=np.float64):
        """
        Initializes a 2- or 3D domain. Horizontal directions (x, y) are Fourier decompositions,
        Vertical direction is either a Chebyshev (if nz, Lz are integers) or a compound
        Chebyshev (if nz, Lz are lists) domain.

        Parameters
        ----------
        All parameters match class attributes, as described in the class docstring.

        """
        self.atmosphere = atmosphere
        self.resolution = resolution
        self.aspect     = aspect
        self.dimensions  = len(self.resolution)
        self.mesh        = mesh
        self.dealias     = dealias
        self.dtype       = dtype
        self.comm        = comm

        #setup horizontal directions
        L_horiz = self.aspect*self.atmosphere.atmo_params['Lz']
        self.bases = []
        if self.dimensions >= 2:
            x_basis = de.Fourier('x', self.resolution[1], interval=(-L_horiz/2, L_horiz/2), dealias=dealias)
            self.bases += [x_basis]
        if self.dimensions == 3:
            y_basis = de.Fourier('y', self.resolution[2], interval=(-L_horiz/2, L_horiz/2), dealias=dealias)
            self.bases += [y_basis]

        #setup vertical direction
        if isinstance(self.resolution[0], list):
            Lz_int = 0
            z_basis_list = []
            for Lz_i, nz_i in zip(self.atmosphere.atmo_params['Lz_list'], self.resolution[0]):
                Lz_top = Lz_i + Lz_int
                z_basis = de.Chebyshev('z', nz_i, interval=[Lz_int, Lz_top], dealias=dealias)
                z_basis_list.append(z_basis)
                Lz_int = Lz_top
            self.Lz = Lz_int
            self.nz = np.sum(nz)
            self.Lz_list = Lz
            self.nz_list = nz
            z_basis = de.Compound('z', tuple(z_basis_list), dealias=dealias)
        else:
            z_basis = de.Chebyshev('z', self.resolution[0], interval=(0, self.atmosphere.atmo_params['Lz']), dealias=dealias)
            self.Lz_list = None
            self.nz_list = None

        #create domain
        self.bases += [z_basis]
        self.domain = de.Domain(self.bases, grid_dtype=dtype, mesh=mesh, comm=comm)

        #store grid data
        if self.dimensions >= 2:
            self.x    = self.domain.grid(0)
            self.x_de = self.domain.grid(0, scales=dealias)
        else:
            self.x, self.x_de = None, None
        if self.dimensions == 3:
            self.y    = self.domain.grid(1)
            self.y_de = self.domain.grid(1, scales=dealias)
        else:
            self.y, self.y_de = None, None
        self.z    = self.domain.grid(-1)
        self.z_de = self.domain.grid(-1, scales=dealias)

        self.atmosphere.build_atmosphere(self)

    def new_ncc(self):
        """ Creates a new dedalus field in this domain and sets its Fourier meta as constant """
        field = self.domain.new_field()
        if self.dimensions >= 2:
            field.meta['x']['constant'] = True
        if self.dimensions >= 3:
            field.meta['y']['constant'] = True            
        return field

    def generate_vertical_profile(self, field, scales=1):
        gslices = self.domain.dist.grid_layout.slices(scales=scales)
        global_array = np.zeros(int(scales*self.nz))
        local_array  = np.zeros_like(global_array)

        if type(field) is Field:
            field.set_scales(scales, keep_data=True)
            indices = list(field['g'].shape)
            for i in range(len(indices)-1):
                indices[i] = 0
            indices[-1] = range(indices[-1])
            local_array[gslices[-1]] = field['g'][indices]
        else:
            indices = list(field.shape)
            for i in range(len(indices)-1):
                indices[i] = 0
            indices[-1] = range(indices[-1])
            local_array[gslices[-1]]  = field[indices]

        self.domain.dist.comm_cart.Allreduce(local_array, global_array, op=MPI.SUM)
        if not(self.mesh is None):
            global_array /= mesh[0]
        return global_array
