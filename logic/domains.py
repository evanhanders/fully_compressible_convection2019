import numpy as np

import logging
logger = logging.getLogger(__name__)

from dedalus import public as de

from mpi4py import MPI

class DedalusDomain:
    """
    A simple class which contains a dedalus domain, as well as a bunch of info about
    the domain, for use in broad functions.

    Attributes:
        nx, ny      -- Integers. The coefficient resolution in x, y
        Lx, Ly      -- Integers. The size of the domain in simulation units in x, y
        nz          -- Integer or List. Coefficient resolution (z).
        Lz          -- Integer or List. Size of domain in simulation units (z)
        dimension   -- An integer. Number of problem dimensions (1-, 2-, 3D), (z, (x,z), (x,y,z)) 
        bases       -- A list of the basis objects in the domain
        domain      -- The dedalus domain object
        dealias     -- A float. multiplicative factor for # of gridpoints on dealiased grid.
        mesh        -- A list. Mesh for dividing processor. np.prod(mesh) must == # of cpus.
        comm        -- MPI communicator for dedalus.
        dtype       -- A numpy datatype, the grid dtype of dedalus.
        x, y, z     -- Arrays containing the gridpoints of the domain for x,y,z (dealias=1)
        x_de, y_de, z_de -- same as x, y, z, but with dealias.
    """

    def __init__(self, nx, ny, nz, Lx, Ly, Lz, 
                 dimensions=2, mesh=None, dealias=3/2, comm=MPI.COMM_WORLD, dtype=np.float64):
        """
        Initializes a 2- or 3D domain. Horizontal directions (x, y) are Fourier decompositions,
        Vertical direction is either a Chebyshev (if nz, Lz are integers) or a compound
        Chebyshev (if nz, Lz are lists) domain.

        Function inputs which line up with class attributes are described in class docstring.
        """
        self.nx, self.ny, self.nz = nx, ny, nz
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.dimensions  = dimensions 
        self.mesh        = mesh
        self.dealias     = dealias
        self.dtype       = dtype
        self.comm        = comm

        #setup horizontal directions
        self.bases = []
        if dimensions >= 2:
            x_basis = de.Fourier('x', nx, interval=(-Lx/2, Lx/2), dealias=dealias)
            self.bases += [x_basis]
        if dimensions == 3:
            y_basis = de.Fourier('y', ny, interval=(-Ly/2, Ly/2), dealias=dealias)
            self.bases += [y_basis]

        #setup vertical direction
        if isinstance(nz, list) and isinstance(Lz, list):
            Lz_int = 0
            z_basis_list = []
            for Lz_i, nz_i in zip(Lz, nz):
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
            z_basis = de.Chebyshev('z', nz, interval=(0, Lz), dealias=dealias)
            self.Lz_list = None
            self.nz_list = None

        #create domain
        self.bases += [z_basis]
        self.domain = de.Domain(self.bases, grid_dtype=dtype, mesh=mesh, comm=comm)

        #store grid data
        if dimensions >= 2:
            self.x    = self.domain.grid(0)
            self.x_de = self.domain.grid(0, scales=dealias)
        else:
            self.x, self.x_de = None, None
        if dimensions == 3:
            self.y    = self.domain.grid(1)
            self.y_de = self.domain.grid(1, scales=dealias)
        else:
            self.y, self.y_de = None, None
        self.z    = self.domain.grid(-1)
        self.z_de = self.domain.grid(-1, scales=dealias)
        print(self.z_de, self.z_de.shape)

    def new_ncc(self):
        ''' Creates a new dedalus field and sets its Fourier meta as constant '''
        field = self.domain.new_field()
        if self.dimensions >= 2:
            field.meta['x']['constant'] = True
        if self.dimensions >= 3:
            field.meta['y']['constant'] = True            
        return field

