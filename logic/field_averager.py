from collections import OrderedDict
from dedalus.extras.flow_tools import GlobalFlowProperty

import h5py
from mpi4py import MPI

import numpy as np
import logging
logger = logging.getLogger(__name__)

try:
    from functions import mpi_makedirs
except:
    from sys import path
    path.insert(0, './logic')
    from logic.functions import mpi_makedirs

class FieldAverager:

    FIELDS = None
    OUT_DIR = 'averager'

    def __init__(self, solver_IVP, de_domain_IVP, root_dir, file_dir=None):
        
        self.solver_IVP    = solver_IVP
        self.de_domain_IVP = de_domain_IVP

        self.rank = de_domain_IVP.domain.dist.comm_cart.rank
        if len(de_domain_IVP.domain.dist.mesh) == 0:
            self.nz_per_proc = int(de_domain_IVP.resolution[0]/de_domain_IVP.domain.dist.comm_cart.size)
        else:
            self.nz_per_proc = int(de_domain_IVP.resolution[0]/de_domain_IVP.domain.dist.mesh[-1])

        self.flow = GlobalFlowProperty(solver_IVP, cadence=1)

        self.measured_profiles, self.avg_profiles, self.local_l2 = OrderedDict(), OrderedDict(), OrderedDict()

        for k, fd in self.FIELDS.items():
            self.flow.add_property('plane_avg({})'.format(fd), name='{}'.format(k))
            self.measured_profiles[k] = np.zeros((2, self.nz_per_proc))
            self.avg_profiles[k]     = np.zeros( self.nz_per_proc )
            self.local_l2[k]     = np.zeros( self.nz_per_proc )

        self.avg_times        = np.zeros(2)
        self.elapsed_avg_time = 0

        self.n_files_saved    = 0
        if file_dir is None:
            file_dir = self.OUT_DIR
        self.file_dir = '{:s}/{:s}/'.format(root_dir, file_dir)
        mpi_makedirs(self.file_dir)


    def get_local_profile(self, prof_name):
        this_field = self.flow.properties['{}'.format(prof_name)]['g']
        if self.de_domain_IVP.dimensions == 3:
            profile = this_field[0,0,:]
        else:
            profile = this_field[0,:]
        return profile

    def local_to_global_average(self, profile):
        loc, glob = [np.zeros(self.de_domain_IVP.resolution[0]) for i in range(2)]
        if len(self.de_domain_IVP.domain.dist.mesh) == 0:
            loc[self.nz_per_proc*self.rank:self.nz_per_proc*(self.rank+1)] = profile 
        elif self.rank < self.de_domain_IVP.domain.dist.mesh[-1]:
            loc[self.nz_per_proc*self.rank:self.nz_per_proc*(self.rank+1)] = profile
        self.de_domain_IVP.domain.dist.comm_cart.Allreduce(loc, glob, op=MPI.SUM)
        return glob

    def find_global_max(self, profile):
        loc, glob = [np.zeros(1) for i in range(2)]
        if len(self.de_domain_IVP.domain.dist.mesh) == 0:
            loc[0] = np.max(profile)
        elif self.rank < self.de_domain_IVP.domain.dist.mesh[-1]:
            loc[0] = np.max(profile)
        self.de_domain_IVP.domain.dist.comm_cart.Allreduce(loc, glob, op=MPI.MAX)
        return glob[0]
            
    def update_avgs(self):
        first = False

        #Times
        self.avg_times[0] = self.solver_IVP.sim_time
        this_dt = self.avg_times[0] - self.avg_times[1]
        if self.elapsed_avg_time == 0:
            first = True
        self.elapsed_avg_time += this_dt
        self.avg_times[1] = self.avg_times[0]
    
        #Profiles
        for k in self.FIELDS.keys():
            self.measured_profiles[k][0,:] = self.get_local_profile(k)
            if first:
                self.avg_profiles[k] *= 0
                self.local_l2[k] *= 0
            else:
                old_avg = self.avg_profiles[k]/(self.elapsed_avg_time - this_dt)
                self.avg_profiles[k] += (this_dt/2)*np.sum(self.measured_profiles[k], axis=0)
                new_avg = self.avg_profiles[k]/self.elapsed_avg_time
                self.local_l2[k] = np.abs((new_avg - old_avg)/new_avg)

            self.measured_profiles[k][1,:] = self.measured_profiles[k][0,:]
        
    def save_file(self):
        """  Saves profiles dict to file """
        z_profile = self.local_to_global_average(self.de_domain_IVP.z.flatten())

        if self.rank == 0:
            file_name = self.file_dir + "profile_dict_file_{:04d}.h5".format(self.n_files_saved+1)
            with h5py.File(file_name, 'w') as f:
                for k, item in self.avg_profiles.items():
                    f[k] = self.local_to_global_average(item/self.elapsed_avg_time)
                f['z'] = z_profile 
            self.n_files_saved += 1

    def reset_fields(self):
        """ Reset all local fields after doing a BVP """
        for fd, info in self.FIELDS.items():
            self.avg_profiles[fd]  *= 0
            self.measured_profiles[fd]  *= 0
            self.local_l2[fd]  *= 0
            self.avg_times *= 0
        self.avg_times[1] = self.solver_IVP.sim_time
        self.elapsed_avg_time = 0

class AveragerFCAE(FieldAverager):

    FIELDS = OrderedDict([
                        ('F_conv',          'F_conv_z'),
                        ('F_tot_superad',   '(F_conv_z + F_cond_z - F_cond_ad_z)'),
                        ('udotgradW_horiz', '(u*dx(w) + v*dy(w))'),
                        ('w_prof_IVP',      'w'),
                        ('T1',              'T1'),
                        ('ln_rho1',         'ln_rho1'),
                        ('mu',              'mu_full'),
                        ('kappa',           'kappa_full'),
                            ])
    OUT_DIR = 'averager_FCAE'

