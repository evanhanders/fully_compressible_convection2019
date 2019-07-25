import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import OrderedDict

import os
from sys import stdout
from sys import path
path.insert(0, './plot_logic')
from plot_logic.file_reader import FileReader
from plot_logic.plot_grid import PlotGrid

import numpy as np

import logging
logger = logging.getLogger(__name__.split('.')[-1])


class ScalarFigure(PlotGrid):
    def __init__(self, *args, fig_name=None, **kwargs):
        super(ScalarFigure, self).__init__(*args, **kwargs)
        self.panels = []
        self.panel_fields = []
        self.fig_name = fig_name
        for i in range(self.ncols):
            for j in range(self.nrows):
                self.panels.append('ax_{}-{}'.format(i,j))
                self.panel_fields.append([])

    def add_field(self, panel, field):
        self.panel_fields[panel].append(field)

class ScalarPlotter():

    def __init__(self, root_dir, file_dir='scalar', fig_name='traces', start_file=1, n_files=None, **kwargs):
        self.reader = FileReader(root_dir, sub_dirs=[file_dir,], num_files=[n_files,], start_file=1, distribution='single', **kwargs)
        self.fig_name = fig_name
        self.out_dir  = '{:s}/{:s}/'.format(root_dir, fig_name)
        if self.reader.comm.rank == 0 and not os.path.exists('{:s}'.format(self.out_dir)):
            os.mkdir('{:s}'.format(self.out_dir))
        self.fields = []

    def load_figures(self, fig_list):
        self.figures = fig_list
        for fig in self.figures:
            for field_list in fig.panel_fields:
                for fd in field_list:
                    if fd not in self.fields:
                        self.fields.append(fd)

    def _read_fields(self):
        if self.reader.local_file_lists[self.reader.sub_dirs[0]] is None:
            return
        self.trace_data = OrderedDict()
        for f in self.fields: self.trace_data[f] = []
        self.trace_data['sim_time'] = []
        for i, f in enumerate(self.reader.local_file_lists[self.reader.sub_dirs[0]]):
            bs, tsk, writenum, times = self.reader.read_file(f, bases=[], tasks=self.fields)
            for f in self.fields: self.trace_data[f].append(tsk[f].flatten())
            self.trace_data['sim_time'].append(times)

        for f in self.fields: self.trace_data[f] = np.concatenate(tuple(self.trace_data[f]))
        self.trace_data['sim_time'] = np.concatenate(tuple(self.trace_data['sim_time']))

    def _save_traces(self):
        if self.reader.local_file_lists[self.reader.sub_dirs[0]] is None:
            return
        import h5py
        with h5py.File('{:s}/full_traces.h5'.format(self.out_dir), 'w') as f:
            for k, fd in self.trace_data.items():
                f[k] = fd

    def plot_figures(self, dpi=200):
        self._read_fields()

        for j, fig in enumerate(self.figures):
            for i, k in enumerate(fig.panels):
                ax = fig.axes[k]
                for fd in fig.panel_fields[i]:
                    print(self.trace_data['sim_time'], self.trace_data[fd])
                    ax.plot(self.trace_data['sim_time'], self.trace_data[fd], label=fd)
                ax.legend(fontsize=8, loc='best')
            ax.set_xlabel('sim_time')
            if fig.fig_name is None:
                fig_name = self.fig_name + '_{}'.format(j)
            else:
                fig_name = fig.fig_name


            fig.fig.savefig('{:s}/{:s}.png'.format(self.out_dir, fig_name), dpi=dpi, bbox_inches='tight')
        self._save_traces()
