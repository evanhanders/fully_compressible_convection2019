import os
import logging
from collections import OrderedDict

import h5py
import numpy as np
from mpi4py import MPI

logger = logging.getLogger(__name__.split('.')[-1])

class FileReader:
    """ A general class for reading and interacting with Dedalus output data.

    Attributes:
    -----------

    """

    def __init__(self, run_dir, sub_dirs=['slices',], num_files=[None,], start_file=1, comm=MPI.COMM_WORLD, **kwargs):
        """
        Initializes the file reader.

        Arguments:
        ----------
        root_dir : string
            Path to root dedalus directory.
        sub_dirs : list, optional
            Subdirectories to read files in
        num_files : list, optional
            Number of files to read for each subdirectory. If None, read them all.
        start_file : integer, optional 
            File number to start reading
        comm : mpi4py Communicator, optional
            communicator over which to break upfiles evenly for even file tasks
        """
        self.run_dir = os.path.expanduser(run_dir)
        self.sub_dirs = sub_dirs
        self.start_file = start_file
        self.file_lists = OrderedDict()
        self.comm = comm

        for d, n in zip(sub_dirs, num_files):
            files = []
            for f in os.listdir('{:s}/{:s}/'.format(self.run_dir, d)):
                if f.endswith('.h5'):
                    file_num = int(f.split('.h5')[0].split('_s')[-1])
                    if file_num < self.start_file: continue
                    if n is not None and file_num > n: continue
                    files.append(['{:s}/{:s}/{:s}'.format(self.run_dir, d, f), file_num])
            self.file_lists[d], nums = zip(*sorted(files, key=lambda x: x[1]))
        self._distribute_files(**kwargs)

    def _distribute_files(self, distribution='one', writes=20):
        """
        Distribute files across MPI processes according to a given type of file distribution.
        Currently, these types of file distributions are implemented:

        'even'   : evenly distribute over all mpi processes
        'writes' : Split up files based on the number of writes in each file
        'single' : First process takes all file tasks

        Arguments:
        ----------
        distribution : string, optional
            Type of distribution
        writes  : num, optional
            If distribution type is "writes," number of writes to clump together
        """
        self.local_file_lists = OrderedDict()
        self.distribution_comms = OrderedDict()
        for k, files in self.file_lists.items():
            if distribution.lower() == 'single':
                self.distribution_comms[k] = None
                if self.comm.rank >= 1:
                    self.local_file_lists[k] = None
                else:
                    self.local_file_lists[k] = files
            elif distribution.lower() == 'even':
                if len(files) <= self.comm.size:
                    if self.comm.rank >= len(files):
                        self.local_file_lists[k] = None
                        self.distribution_comms[k] = None
                    else:
                        self.local_file_lists[k] = [files[self.comm.rank],]
                        self.distribution_comms[k] = self.comm.Create(self.comm.Get_group().Incl(np.arange(len(files))))
                else:
                    files_per = len(files) / self.comm.size
                    excess_files = len(files) % self.comm.size
                    if self.comm.rank >= excess_files:
                        print(self.comm.rank*files_per+excess_files, (self.comm.rank+1)*files_per+excess_files)
                        self.local_file_lists[k] = list(files[int(self.comm.rank*files_per+excess_files):int((self.comm.rank+1)*files_per+excess_files)])
                    else:
                        self.local_file_lists[k] = list(files[int(self.comm.rank*(files_per+1)):int((self.comm.rank+1)*(files_per+1))])
                    self.distribution_comms[k] = self.comm
            elif distribution.lower() == 'writes':
                logger.error('NOT YET IMPLEMENTED')
                import sys
                sys.exit()
                file_writes, local_write = np.zeros(1), np.zeros(1)
                if self.comm.rank == 0:
                    with h5py.File(files[0], 'r') as f:
                        local_write[0] = len(f['scales']['write_number'].value)
                self.comm.Allreduce(local_write, file_writes, op=MPI.SUM)

                assignments = (np.arange(len(files)*file_writes[0])+1)/writes
                if np.ceil(assignments[-1]) < self.comm.size: #too many processors
                    start_file = int(np.floor(self.comm.rank * writes / file_writes))
                    end_file   = int(np.ceil((self.comm.rank) * writes / file_writes))
                    if end_file > len(files):
                        self.local_file_lists[k] = None
                        self.distribution_comms[k] = None
                    else:
                        self.local_file_lists[k] = files[start_file:end_file]
                        self.distribution_comms[k] = comm.Create(comm.Get_group().Incl(np.arange(np.ceil(assignments[-1]))))
#                else: #more than one group per process
            
    def read_file(self, filename, bases=[], tasks=[]):
        """ reads dedalus tasks """
        out_bases = OrderedDict()
        out_tasks = OrderedDict()
        with h5py.File(filename, 'r') as f:
            for b in bases:
                out_bases[b] = f['scales'][b]['1.0'].value
            out_write_num = f['scales']['write_number'].value
            out_sim_time = f['scales']['sim_time'].value
            for t in tasks:
                out_tasks[t] = f['tasks'][t].value
        return out_bases, out_tasks, out_write_num, out_sim_time

if __name__ == '__main__':
    FileReader('FC_poly_Ra1e3_Pr1_n3_eps1e-4_a4_2D_Tflux_temp_Vstress_free_64x128_AE')
            

