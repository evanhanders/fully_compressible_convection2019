#http://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
# has great notes on how to manipulate matplotlib legends.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from scipy import stats
import dedalus.public as de

import h5py
import os

from mpi4py import MPI
MPI_COMM_WORLD = MPI.COMM_WORLD

import logging
logger = logging.getLogger(__name__.split('.')[-1])

COLORS=('darkslategray', 'fuchsia', 'orangered', \
        'firebrick', 'indigo', 'lightgoldenrodyellow', 'darkmagenta',\
        'turquoise', 'mediumvioletred', 'mediumorchid', 'saddlebrown',\
        'slateblue', 'darkcyan')
STYLES=('-', '--', '-.', ':')
GREY=(0.8,0.8,0.8)

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def get_l2_ends(field_l2_values):
    field_l2_ends = np.zeros(field_l2_values.shape[0])
    for i in range(field_l2_values.shape[0]):
        if len(np.where(field_l2_values[i,:] != 0)[0]) == 0:
            field_l2_ends[i] = 1
            continue
        field_l2_ends[i] = \
            field_l2_values[i, np.where(field_l2_values[i,:] != 0)[0][-1]]
    return field_l2_ends


class PlotBuddy:
    """
    A general class for interacting with dedalus data from a convection-type
    simulation.  Has functionality for locating .h5 files in appropriate subdirectories,
    and also for grabbing a specified subset of those data files.

    Attributes:
    -----------
        root_dir        - A base path to the simulation's output directory, which contains
                          the various different file handler directories
        start_file      - An integer, which corresponds to the number of the first data file
                          from each file handler that will be read in.
        file_dirs       - A list of the names of file handler directories inside of root_dir to
                          get the files from
        comm            - An MPI communication group over which file handling/plotting duties are split
        cw_size         - The number of processors in comm
        cw_rank         - This processor's rank in comm.
        idle            - If True, too many processes were given for the task at hand, and
                          the local processor will just sit there, rather than mess things up.
        n_files_global   - The total number of files being processed by the comm world
        files_below     - The number of files "below" this processor, e.g. the number of files
                          being looked at by processes of lower rank than this one.
        n_files_local         - The number of files the current process is responsible for
        file_number_start - The starting file number for the local process
        file_number_end   - The ending file number for the local process
        local_files     - A dictionary with len(file_dirs) entries, with the path to each file
                          the local process is responsible for.
        global_files    - Same as local_files, but for ALL files in the comm world.
        atmosphere      - A dictionary containing information about the atmosphere,
                          read from the file 'root_dir/atmosphere/atmosphere.h5'
        x               - If simulation is 2- or 3-D, this is a 1D array of x values
        y               - If simulation is 3-D, this is a 1D array of y values
        z               - A 1-D array of z values
        xy_xs, xy_ys    - A mesh of x and y values for plotting horizontal cuts
        xz_xs, xz_zs    - A mesh of x and z values for plotting vertical cuts
        yz_ys, yz_zs    - A mesh of y and z values for plotting vertical cuts
        global_good_indices - A list of lists, each of which contains the "good" indices of
                              the corresponding file in the comm world
        local_good_indices  - Like global_good_indice, but only for the local files.
        global_times        - A list of all of the simulation times.
        sim_start_time      - The first element of global_times.
        local_times         - Like global_times, but only for the local files.
        max_writes_per_file - The maximum number of writes in a file (in the time direction)
        local_total_writes  - The total number of good writes, locally.
        local_writes_below  - The number of total good writes on processes whose rank is lower
                              than the local one.
    """ 

    def __init__(self, root_dir, n_files=1e6, start_file=1,\
                    file_dirs=['slices', 'scalar', 'profiles'], comm=MPI_COMM_WORLD):
        """
        Initializes the PlotBuddy.  Finds all of the important files in a directory,
        then splits them up between processors and figures out which ones the local process
        is responsible for.

        Inputs:
        -------
            root_dir    - The base path to the simulation data
            start_file  - The number of the first file to grab
            n_files   - The maximum number of simulation files to access
            file_dirs   - The file handler sub-directories to keep track of
            comm        - The comm group to split files over
        """

        # Store some basics.
        if root_dir[-1] != '/':
            root_dir += '/'
        self.root_dir       = root_dir
        self.start_file     = start_file
        self.file_dirs      = file_dirs

        # Read in atmosphere information and get files
        self._read_atmosphere()
        self._get_all_files()

        self.cw_size        = comm.Get_size()
        self.cw_rank        = comm.Get_rank()
        self.n_files_global  = len(self.global_files[file_dirs[0]])

        # If more files exist than are asked for, only do as many as are requested
        if self.n_files_global > n_files:
            self.n_files_global = n_files
            for d in self.file_dirs:
                self.global_files[d] = self.global_files[d][:n_files]

        logger.info('Getting {} files from {:s}'.\
                        format(self.n_files_global,self.root_dir))

        # Get all of the files associated with this process and figure out how 
        # many files are on processes below this one.
        self.n_files_local = int(np.floor(self.n_files_global/self.cw_size))
        if self.cw_rank < (self.n_files_global % self.cw_size) \
                and self.cw_size > 1:
            self.n_files_local += 1
            files_below = self.cw_rank * self.n_files_local
        elif self.cw_size == 1:
            files_below = 0
        else:
            files_below = (self.n_files_local+1)*(self.n_files_global % self.cw_size)\
                          + (self.cw_rank - (self.n_files_global % self.cw_size))\
                          * self.n_files_local

        # If the user has given too many CPUs (more than there are files), then
        # make a new comm group with only as many as are needed.
        if self.n_files_local == 0:
            self.idle, self.comm = True, None
            self.cw_rank, self.cw_size = -1, -1
            return
        else:
            self.idle = False
            n_group = self.n_files_global
            if n_group > self.cw_size:
                n_group = self.cw_size
            self.comm = comm.Create(comm.Get_group().Incl(np.arange(n_group)))
            self.cw_rank = self.comm.rank
            self.cw_size = self.comm.size

        # Store information about the local files
        self.files_below = int(files_below)
        self.file_number_start = self.files_below + 1
        self.file_number_end = self.files_below +  self.n_files_local
        
        upper = self.n_files_local + self.files_below
        lower = self.files_below
        self.local_files = dict()
        for d in self.file_dirs:
            self.local_files[d] = self.global_files[d][lower:upper]

        #Figure out temporal and spatial information of these simulations.
        self._get_domain_info()
        self._get_time_info()

    def _get_all_files(self):
        '''
            Looks in all subdirectories of root_dir specified in self.file_dirs,
            and pulls out the .h5 files located in them.  Stores them in the
            global_files class attribute.
        '''
        self.global_files = dict()
        for d in self.file_dirs:
            file_list = []
            for f in os.listdir(self.root_dir+d):
                if f.endswith('.h5'):
                    if int(f.split('.')[0].split('_')[-1][1:]) < self.start_file:
                        continue
                    file_list.append([self.root_dir+d+'/'+f, \
                         int(f.split('.')[0].split('_')[-1][1:])])
          
            self.global_files[d] = sorted(file_list, key=lambda x: x[1])

    def _get_domain_info(self):
        '''
            Grabs out the profiles of x, y, and z for the local processor.
            Makes a mesh of x/z, x/y, and y/z. Only grabs these from the
            1st type of file in file_dirs.
        '''
        if self.idle: return
        base =  {'x' : None, 'y' : None, 'z' : None}
        self.x, self.y, self.z = [None]*3

        f = h5py.File("{:s}".format(self.local_files[self.file_dirs[0]][0][0]))
        for k in f['scales'].keys():
            if type(f['scales'][k]) == h5py._hl.group.Group:
               base[k] =  f['scales'][k]['1.0'][:]
        f.close()
        self.x = base['x']
        self.y = base['y']
        self.z = base['z']
        logger.info('bases found:')
        logger.info('          x = {}'.format(self.x))
        logger.info('          y = {}'.format(self.y))
        logger.info('          z = {}'.format(self.z))

    def _get_time_info(self):
        '''
            Look at each file, and gather info about the time in each of the files.
            If checkpoints happened, it's possible that there are some repeats of
            timesteps, so check to see if there are, and never look at the repeated
            points.  Store time and repeat information about local files, and about
            the global file group.
        '''
        if self.idle: return

        #Get all of the times
        times = []
        loc_times,  glob_times = np.zeros(self.comm.size, dtype=np.int64), np.zeros(self.comm.size, dtype=np.int64)
        loc_writes, glob_writes = np.zeros(self.n_files_global, dtype=np.int64), np.zeros(self.n_files_global, dtype=np.int64)
        for i, item in enumerate(self.local_files[self.file_dirs[-1]]):
            with h5py.File("{:s}".format(item[0]), 'r') as f:
                t = np.array(f['scales']['sim_time'][:], dtype=np.float32)
            times.append(t)
            loc_times[self.comm.rank] += t.shape[0]
            loc_writes[self.files_below+i] = t.shape[0]

        #Communicate max writes/file, and how many time points there are.
        self.comm.Allreduce(loc_times, glob_times, op=MPI.SUM)
        self.comm.Allreduce(loc_writes, glob_writes, op=MPI.SUM)

        #Store all time points in a big array, communicate to get full time array.
        local_sim_times = np.zeros(int(np.sum(glob_times)))
        global_sim_times = np.zeros_like(local_sim_times)
        start_indx = int(round(int(np.sum(glob_times[:self.comm.rank]))))
        for t in times:
            local_sim_times[start_indx:start_indx+len(t)] = t
            start_indx += len(t)
        self.comm.Allreduce(local_sim_times, global_sim_times, op=MPI.SUM)

        #In case there are weird data points, get rid of them. Don't keep repeat times.
        unique_times, unique_indices = np.unique(global_sim_times, return_index=True)
        good_glob_indices, good_loc_indices = [], []
        for i in range(self.n_files_global):
            good_glob_indices.append(list())
            good_loc_indices.append(list())
            glob_indices = np.arange(glob_writes[i]) + np.sum(glob_writes[:i])
            loc_indices = np.arange(glob_writes[i])
            for j, indx in enumerate(glob_indices):
                if indx in unique_indices:
                    good_glob_indices[i].append(indx)
                    good_loc_indices[i].append(loc_indices[j])
        
        #Store info about non-repeated times.
        self.all_good_local_indices     = good_loc_indices
        self.all_good_global_indices    = good_glob_indices
        self.local_good_global_indices  = good_glob_indices[self.file_number_start-1:self.file_number_end]
        self.local_good_local_indices   = good_loc_indices[self.file_number_start-1:self.file_number_end]
        self.global_writes              = glob_writes
        self.global_times               = unique_times
        self.sim_start_time             = self.global_times[0]
        self.global_writes_per_file     = [len(l) for l in self.all_good_global_indices]
        self.local_writes_per_file      = [len(l) for l in self.local_good_global_indices]

#        # File indices should start from 0
#        for i in range(len(self.local_good_indices)):
#            self.local_good_indices[i] = np.array(self.local_good_indices[i], dtype=np.int64)
#            self.local_good_indices[i] -= int(np.round(np.sum(tot_f_writes[:self.file_number_start+i-1])))
            
        # Get out the "good" indices from our local files 
        local_times_full = []
        for i,indices in enumerate(self.local_good_local_indices):
            for j in indices:
                local_times_full.append(times[i][j])

        self.local_times                = np.array(local_times_full)
        self.max_writes_per_file        = np.max(self.global_writes)
        self.local_writes               = int(self.local_times.shape[0])
        if self.local_writes != 0:    
            self.local_writes_below      = int(self.global_times[self.global_times < self.local_times[0]].shape[0])
        else:
            self.local_writes_below      = int(np.sum(np.array([len(l) for l in good_indices[:self.file_number_start]])))

        # Get file start / end times
        self.local_start_times, self.global_start_times = np.zeros(self.global_writes.shape), np.zeros(self.global_writes.shape)
        self.local_end_times, self.global_end_times = np.zeros(self.global_writes.shape), np.zeros(self.global_writes.shape)
        for i, indices in enumerate(self.local_good_local_indices):
            self.local_start_times[self.files_below+i] = np.min(times[i][indices])
            self.local_end_times[self.files_below+i] = np.max(times[i][indices])
        
        self.comm.Allreduce(self.local_start_times, self.global_start_times, op=MPI.SUM)
        self.comm.Allreduce(self.local_end_times, self.global_end_times, op=MPI.SUM)
        self.local_start_times = self.local_start_times[self.local_start_times != 0]
        self.local_end_times = self.local_end_times[self.local_end_times != 0]

    def _read_atmosphere(self):
        '''
        Reads atmospheric parameters from the file root_dir/atmosphere/atmosphere.h5
        '''
        file_name = self.root_dir + '/atmosphere/atmosphere.h5'
        self.atmosphere = dict()
        try:
            f = h5py.File('{:s}'.format(file_name))
            for key in f.keys():
                self.atmosphere[key] = f[key].value
        except:
               logger.info("atmosphere file seems to be missing, or somethign else went wrong.") 



    def grab_full_task(self, file_list, good_indices, profile_name=['s']):
        """
            Grabs the save field(s) specified in profile_name.  Only selects 
            info from the specified files in the specified
            indices of each file that the user selects.

            Inputs:
            -------
            file_list - A list of lists, each of which has a file name as its first element,
                        in the manner of self.local_files or self.global_files
            good_indices - A list of lists, which contain the "good" indices of each of
                           the corresponding files in file_list
            profile_name - A list of profiles to grab from the tasks of the file.

            Returns:
            --------
            A dictionary, which has as many keys as the length of profile_name. The
            keys are the elements of profile_name, and stored in each key is a Numpy
            array of Time x simulation data from the files in file_list.
        """
        if self.idle: return
        bigger_fields = dict()
        
        if len(file_list) > len(good_indices):
            file_list = file_list[:len(good_indices)]
        count = 0
        for i, f_list in enumerate(file_list):
            if len(good_indices[i]) == 0:
                print('File {}/{} on process {} is bad, skipping'.format(i+1, len(file_list), self.cw_rank))
                continue
            logger.info('opening file {}'.format(f_list[0]))
            with h5py.File("{:s}".format(f_list[0]), 'r') as f:
                for j in range(len(profile_name)):
                    try:
                        field = np.array(f['tasks'][str(profile_name[j])], dtype=np.float32)
                    except:
                        field = np.array(np.abs(f['tasks'][str(profile_name[j])]), dtype=np.float32)
                    if profile_name[j] not in bigger_fields.keys():
                        #Create space in memory for fields from all files
                        shape = list(field.shape)
                        shape[0] = np.sum(np.array([len(l) for l in good_indices]))
                        bigger_fields[profile_name[j]] = np.zeros(shape, dtype=np.float32)
                    bigger_fields[profile_name[j]][count:count+len(good_indices[i])] =  field[good_indices[i]]
            count += len(good_indices[i])
        return bigger_fields

