import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 9})

from collections import OrderedDict
import os
from sys import stdout
from sys import path
path.insert(0, './plot_logic')
from plot_logic.file_reader import FileReader
from plot_logic.plot_grid import ColorbarPlotGrid

import numpy as np
from mpi4py import MPI

import logging
logger = logging.getLogger(__name__.split('.')[-1])


class ProfileColormesh:
    def __init__(self, field, basis='z', cmap='RdBu_r', pos_def=False):
        self.field = field
        self.basis = basis
        self.cmap = cmap
        self.pos_def = pos_def
        self.xx, self.yy = None, None

class ProfilePlotter():

    def __init__(self, root_dir, file_dir='profiles', fig_name='structures', start_file=1, n_files=None, **kwargs):
        self.reader = FileReader(root_dir, sub_dirs=[file_dir,], num_files=[n_files,], start_file=1, distribution='even', **kwargs)
        self.fig_name = fig_name
        self.out_dir  = '{:s}/{:s}/'.format(root_dir, fig_name)
        if self.reader.comm.rank == 0 and not os.path.exists('{:s}'.format(self.out_dir)):
            os.mkdir('{:s}'.format(self.out_dir))
        self.colormeshes = []

    def add_colormesh(self, *args, **kwargs):
        self.colormeshes.append(ProfileColormesh(*args, **kwargs))

    def get_profiles(self, tasks, bases):
        my_tsks = []
        my_times = []
        my_writes = []
        my_num_writes = 0
        min_writenum = None

        comm = self.reader.distribution_comms[self.reader.sub_dirs[0]]
        files = self.reader.local_file_lists[self.reader.sub_dirs[0]]
        if len(files) == 0:
            return [None]*3
        for i, f in enumerate(files):
            if self.reader.comm.rank == 0:
                print('on file {}/{}...'.format(i+1, len(self.reader.local_file_lists[self.reader.sub_dirs[0]])))
                stdout.flush()
            bs, tsk, writenum, times = self.reader.read_file(f, bases=bases, tasks=tasks)
            my_tsks.append(tsk)
            my_times.append(times)
            my_writes.append(writenum)
            my_num_writes += len(times)
            if i == 0:
                min_writenum = np.min(writenum)

        glob_writes = np.zeros(1, dtype=np.int32)
        glob_min_writenum = np.zeros(1, dtype=np.int32)
        my_num_writes = np.array(my_num_writes, dtype=np.int32)
        min_writenum = np.array(min_writenum, dtype=np.int32)
        comm.Allreduce(my_num_writes, glob_writes, op=MPI.SUM)
        comm.Allreduce(min_writenum, glob_min_writenum, op=MPI.MIN)

        profiles = OrderedDict()
        times = np.zeros(glob_writes[0])
        times_buff = np.zeros(glob_writes[0])
        for i, t in enumerate(tasks):
            for j in range(len(my_tsks)):         
                field = my_tsks[j][t].squeeze()
                n_prof = field.shape[-1]
                if j == 0:
                    buff = np.zeros((glob_writes[0],n_prof))
                    profiles[t] = np.zeros((glob_writes[0], n_prof))
                t_indices = my_writes[j]-glob_min_writenum[0]
                profiles[t][t_indices,:] = field
                if i == 0:
                    times[t_indices] = my_times[j]
            comm.Allreduce(profiles[t], buff, op=MPI.SUM)
            comm.Allreduce(times, times_buff, op=MPI.SUM)
            profiles[t][:,:] = buff[:,:]
            times[:] = times_buff
        return profiles, bs, times
                
    def plot_colormeshes(self, dpi=200, **kwargs):
        grid = ColorbarPlotGrid(1,1, **kwargs)
        ax = grid.axes['ax_0-0']
        cax = grid.cbar_axes['ax_0-0']
        tasks = []
        bases = []
        for cm in self.colormeshes:
            if cm.field not in tasks:
                tasks.append(cm.field)
            if cm.basis not in bases:
                bases.append(cm.basis)

        if self.reader.local_file_lists[self.reader.sub_dirs[0]] is None:
            return
        profiles, bs, times = self.get_profiles(tasks, bases)

        for cm in self.colormeshes:
            basis = bs[cm.basis]
            yy, xx = np.meshgrid(basis, times)
            k = cm.field
            data = profiles[k]

            if self.reader.comm.rank == 0:
                print('writing plot {}'.format(k))
                stdout.flush()


            vals = np.sort(data.flatten())
            if cm.pos_def:
                vmin, vmax = 0, vals[int(0.99*len(vals))]
            else:
                vals = np.sort(np.abs(vals))
                vmax = vals[int(0.99*len(vals))]
                vmin = -vmax
            plot = ax.pcolormesh(xx, yy, data, cmap=cm.cmap, vmin=vmin, vmax=vmax, rasterized=True)
            cb = plt.colorbar(plot, cax=cax, orientation='horizontal')
            cb.solids.set_rasterized(True)
            cb.set_ticks((vmin, vmax))
            cb.set_ticklabels(('{:.2e}'.format(vmin), '{:.2e}'.format(vmax)))
            cax.xaxis.set_ticks_position('bottom')
            cax.text(0.5, 0.25, '{:s}'.format(k))

            grid.fig.savefig('{:s}/{:s}_{:s}.png'.format(self.out_dir, self.fig_name, k), dpi=dpi, bbox_inches='tight')
            ax.clear()
            cax.clear()
