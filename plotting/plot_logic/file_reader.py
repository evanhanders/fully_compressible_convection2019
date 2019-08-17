import os
import logging
from collections import OrderedDict

import h5py
import numpy as np
from mpi4py import MPI

logger = logging.getLogger(__name__.split('.')[-1])


class FileReader:
    """ 
    A general class for reading and interacting with Dedalus output data.

    Attributes:
    -----------
    comm : mpi4py Comm
        The MPI communicator to use for parallel post-processing distribution
    distribution_comms : OrderedDict
        subdivided communicators based on desired processor distribution, split by sub_dirs
    file_lists : OrderedDict
        Contains lists of string paths to all files being processed for each dir in sub_dirs
    idle : list
        A list of bools. True if the local processor is not responsible for any files
    local_file_lists : OrderedDict
        lists of string paths to output files that this processor is responsible for reading, split by sub_dirs
    run_dir : str
        Path to root dedalus directory
    sub_dirs : list
        List of strings of subdirectories in run_dir to consider files in
    """

    def __init__(self, run_dir, sub_dirs=['slices',], num_files=[None,], start_file=1, comm=MPI.COMM_WORLD, **kwargs):
        """
        Initializes the file reader.

        Arguments:
        ----------
        run_dir : string
            As defined in class-level docstring
        sub_dirs : list, optional
            As defined in class-level docstring
        num_files : list, optional
            Number of files to read in each subdirectory. If None, read them all.
        start_file : integer, optional 
            File number to start reading from (1 by default)
        comm : mpi4py Comm, optional
            As defined in class-level docstring
        **kwargs : Additional keyword arguments for the self._distribute_files() function.
        """
        self.run_dir    = os.path.expanduser(run_dir)
        self.sub_dirs   = sub_dirs
        self.file_lists = OrderedDict()
        self.comm       = comm

        for d, n in zip(sub_dirs, num_files):
            files = []
            for f in os.listdir('{:s}/{:s}/'.format(self.run_dir, d)):
                if f.endswith('.h5'):
                    file_num = int(f.split('.h5')[0].split('_s')[-1])
                    if file_num < start_file: continue
                    if n is not None and file_num > n: continue
                    files.append(['{:s}/{:s}/{:s}'.format(self.run_dir, d, f), file_num])
            self.file_lists[d], nums = zip(*sorted(files, key=lambda x: x[1]))

        self.local_file_lists = OrderedDict()
        self.distribution_comms = OrderedDict()
        self.idle = []
        self._distribute_files(**kwargs)


    def _distribute_files(self, distribution='one'):
        """
        Distribute files across MPI processes according to a given type of file distribution.
        Currently, these types of file distributions are implemented:

        'even'   : evenly distribute over all mpi processes
        'single' : First process takes all file tasks

        Arguments:
        ----------
        distribution : string, optional
            Type of MPI file distribution
        """
        for k, files in self.file_lists.items():
            self.idle[k] = False
            if distribution.lower() == 'single':
                self.distribution_comms[k] = None
                if self.comm.rank >= 1:
                    self.local_file_lists[k] = None
                    self.idle[k] = True
                else:
                    self.local_file_lists[k] = files
            elif distribution.lower() == 'even':
                if len(files) <= self.comm.size:
                    if self.comm.rank >= len(files):
                        self.local_file_lists[k] = None
                        self.distribution_comms[k] = None
                        self.idle[k] = True
                    else:
                        self.local_file_lists[k] = [files[self.comm.rank],]
                        self.distribution_comms[k] = self.comm.Create(self.comm.Get_group().Incl(np.arange(len(files))))
                else:
                    files_per = int(np.floor(len(files) / self.comm.size))
                    excess_files = int(len(files) % self.comm.size)
                    if self.comm.rank >= excess_files:
                        self.local_file_lists[k] = list(files[int(self.comm.rank*files_per+excess_files):int((self.comm.rank+1)*files_per+excess_files)])
                    else:
                        self.local_file_lists[k] = list(files[int(self.comm.rank*(files_per+1)):int((self.comm.rank+1)*(files_per+1))])
                    self.distribution_comms[k] = self.comm
          

    def read_file(self, file_name, bases=[], tasks=[]):
        """ 
        Opens a dedalus file and reads out the specific bases and tasks.
        Additionally reads the simulation time and write number of each write.

        Arguments:
        ----------
        file_name : str
            string path to the file being opened
        bases : list, optional
            The names of the bases to pull from the file, e.g., 'x', 'z'
        tasks : list, optional
            The output tasks to pull from the file, e.g., 'vorticity', 'entropy'

        Outputs:
        --------
        out_bases : OrderedDict
            The desired bases. Dictionary keys are the elements of input list 'bases'
        out_tasks : OrderedDict
            The desired tasks. Dictionary keys are the elements of input list 'tasks'
        out_write_num : NumPy array
            The write number of each write in the file
        out_sim_time : NumPy array
            The simulation time of each write in the file
        """
        out_bases = OrderedDict()
        out_tasks = OrderedDict()
        with h5py.File(file_name, 'r') as f:
            for b in bases:
                out_bases[b] = f['scales/{:s}/1.0'.format(b)][()]
            out_write_num = f['scales']['write_number'][()]
            out_sim_time = f['scales']['sim_time'][()]
            for t in tasks:
                out_tasks[t] = f['tasks'][t][()]
        return out_bases, out_tasks, out_write_num, out_sim_time
