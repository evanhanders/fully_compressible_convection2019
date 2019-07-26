import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 9})

import os
from sys import stdout
from sys import path
path.insert(0, './plot_logic')
from plot_logic.file_reader import FileReader
from plot_logic.plot_grid import ColorbarPlotGrid

import numpy as np

import logging
logger = logging.getLogger(__name__.split('.')[-1])


class Colormesh:
    def __init__(self, field, x_basis='x', y_basis='z', remove_mean=False, remove_x_mean=False, remove_y_mean=False, cmap='RdBu_r', pos_def=False):
        self.field = field
        self.x_basis = x_basis
        self.y_basis = y_basis
        self.remove_mean = remove_mean
        self.remove_x_mean = remove_x_mean
        self.remove_y_mean = remove_y_mean
        self.cmap = cmap
        self.pos_def = pos_def
        self.xx, self.yy = None, None

class SlicePlotter():

    def __init__(self, root_dir, file_dir='slices', fig_name='snapshots', start_file=1, n_files=None, **kwargs):
        self.reader = FileReader(root_dir, sub_dirs=[file_dir,], num_files=[n_files,], start_file=1, distribution='even', **kwargs)
        self.fig_name = fig_name
        self.out_dir  = '{:s}/{:s}/'.format(root_dir, fig_name)
        if self.reader.comm.rank == 0 and not os.path.exists('{:s}'.format(self.out_dir)):
            os.mkdir('{:s}'.format(self.out_dir))
        self.colormeshes = []

    def setup_grid(self, *args, **kwargs):
        self.grid = ColorbarPlotGrid(*args, **kwargs)

    def add_colormesh(self, *args, **kwargs):
        self.colormeshes.append(Colormesh(*args, **kwargs))


    def _groom_grid(self):
        axs, caxs = [], []
        for i in range(self.grid.ncols):
            for j in range(self.grid.nrows):
                k = 'ax_{}-{}'.format(i,j)
                if k in self.grid.axes.keys():
                    axs.append(self.grid.axes[k])
                    caxs.append(self.grid.cbar_axes[k])
        return axs, caxs

    def plot_colormeshes(self, start_fig=1, dpi=200):
        axs, caxs = self._groom_grid()
        tasks = []
        bases = []
        for cm in self.colormeshes:
            if cm.field not in tasks:
                tasks.append(cm.field)
            if cm.x_basis not in bases:
                bases.append(cm.x_basis)
            if cm.y_basis not in bases:
                bases.append(cm.y_basis)

        if self.reader.local_file_lists[self.reader.sub_dirs[0]] is None:
            return

        for i, f in enumerate(self.reader.local_file_lists[self.reader.sub_dirs[0]]):
            if self.reader.comm.rank == 0:
                print('on file {}/{}...'.format(i+1, len(self.reader.local_file_lists)))
                stdout.flush()
            bs, tsk, writenum, times = self.reader.read_file(f, bases=bases, tasks=tasks)

            if i == 0:
                for cm in self.colormeshes:
                    x = bs[cm.x_basis]
                    y = bs[cm.y_basis]
                    cm.yy, cm.xx = np.meshgrid(y, x)

            for j, n in enumerate(writenum):
                if self.reader.comm.rank == 0:
                    print('writing plot {}/{} on process 0'.format(j+1, len(writenum)))
                    stdout.flush()
                for k in range(len(tasks)):
                    field = np.squeeze(tsk[tasks[k]][j,:])
                    xx, yy = self.colormeshes[k].xx, self.colormeshes[k].yy
                    if self.colormeshes[k].remove_mean:
                        field -= np.mean(field)
                    elif self.colormeshes[k].remove_x_mean:
                        field -= np.mean(field, axis=0)
                    elif self.colormeshes[k].remove_y_mean:
                        field -= np.mean(field, axis=1)

                    vals = np.sort(field.flatten())
                    if self.colormeshes[k].pos_def:
                        vmin, vmax = 0, vals[int(0.99*len(vals))]
                    else:
                        vals = np.sort(np.abs(vals))
                        vmax = vals[int(0.99*len(vals))]
                        vmin = -vmax
                    plot = axs[k].pcolormesh(xx, yy, field, cmap=self.colormeshes[k].cmap, vmin=vmin, vmax=vmax, rasterized=True)
                    cb = plt.colorbar(plot, cax=caxs[k], orientation='horizontal')
                    cb.solids.set_rasterized(True)
                    cb.set_ticks((vmin, vmax))
                    cb.set_ticklabels(('{:.2e}'.format(vmin), '{:.2e}'.format(vmax)))
                    caxs[k].xaxis.set_ticks_position('bottom')
                    caxs[k].text(0.5, 0.25, '{:s}'.format(tasks[k]))

                plt.suptitle('t = {:.4e}'.format(times[j]))
                self.grid.fig.savefig('{:s}/{:s}_{:06d}.png'.format(self.out_dir, self.fig_name, n+start_fig), dpi=dpi, bbox_inches='tight')
                for ax in axs: ax.clear()
                for cax in caxs: cax.clear()
