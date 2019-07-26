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
from plot_logic.plot_grid import PlotGrid
from scipy.interpolate import RegularGridInterpolator

from mpi4py import MPI

import numpy as np
import h5py

import logging
logger = logging.getLogger(__name__.split('.')[-1])


class PdfPlotter():

    def __init__(self, root_dir, file_dir='slices', fig_name='pdfs', start_file=1, n_files=None, **kwargs):
        self.reader = FileReader(root_dir, sub_dirs=[file_dir,], num_files=[n_files,], start_file=1, distribution='even', **kwargs)
        self.fig_name = fig_name
        self.out_dir  = '{:s}/{:s}/'.format(root_dir, fig_name)
        if self.reader.comm.rank == 0 and not os.path.exists('{:s}'.format(self.out_dir)):
            os.mkdir('{:s}'.format(self.out_dir))
        self.pdfs = OrderedDict()
        self.pdf_stats = OrderedDict()
    
    def _get_interpolated_slices(self, pdf_list, bases=['x', 'z'], cheby_basis='z'):
        #Read data
        tasks = []
        for i, f in enumerate(self.reader.local_file_lists[self.reader.sub_dirs[0]]):
            if self.reader.comm.rank == 0:
                print('reading file {}/{}...'.format(i+1, len(self.reader.local_file_lists[self.reader.sub_dirs[0]])))
                stdout.flush()
            bs, tsk, writenum, times = self.reader.read_file(f, bases=bases, tasks=pdf_list)
            tasks.append(tsk)
            if i == 0:
                total_shape = list(tsk[pdf_list[0]].shape)
            else:
                total_shape[0] += tsk[pdf_list[0]].shape[0]

        # Put data on an even grid
        x, y = bs[bases[0]], bs[bases[1]]
        yy, xx = np.meshgrid(y, x)
        if bases[0] == cheby_basis:
            even_x = np.linspace(x.min(), x.max(), len(x))
            even_y = y
        elif bases[1] == cheby_basis:
            even_x = x
            even_y = np.linspace(y.min(), y.max(), len(y))
        eyy, exx = np.meshgrid(even_y, even_x)

        full_data = OrderedDict()
        for k in pdf_list: full_data[k] = np.zeros(total_shape)
        count = 0
        for i in range(len(tasks)):
            for j in range(tasks[i][pdf_list[0]].shape[0]):
                for k in pdf_list:
                    interp = RegularGridInterpolator((x.flatten(), y.flatten()), tasks[i][k][j,:], method='linear')
                    full_data[k][count,:] = interp((exx, eyy))
                count += 1
        return full_data

    def _calculate_pdf_statistics(self):
        for k, data in self.pdfs.items():
            pdf, x_vals, dx = data

            mean = np.sum(x_vals*pdf*dx)
            stdev = np.sqrt(np.sum((x_vals-mean)**2*pdf*dx))
            skew = np.sum((x_vals-mean)**3*pdf*dx)/stdev**3
            kurt = np.sum((x_vals-mean)**4*pdf*dx)/stdev**4
            self.pdf_stats[k] = (mean, stdev, skew, kurt)



    def calculate_pdfs(self, pdf_list, bins=100, **kwargs):
        this_comm = self.reader.distribution_comms[self.reader.sub_dirs[0]]
        if this_comm is None:
            return

        full_data = self._get_interpolated_slices(pdf_list, **kwargs)

        # Create histograms of data
        bounds = OrderedDict()
        minv, maxv = np.zeros(1), np.zeros(1)
        buffmin, buffmax = np.zeros(1), np.zeros(1)
        for k in pdf_list:
            minv[0] = np.min(full_data[k])
            maxv[0] = np.max(full_data[k])
            this_comm.Allreduce(minv, buffmin, op=MPI.MIN)
            this_comm.Allreduce(maxv, buffmax, op=MPI.MAX)
            bounds[k] = (np.copy(buffmin[0]), np.copy(buffmax[0]))
            buffmin *= 0
            buffmax *= 0

            loc_hist, bin_edges = np.histogram(full_data[k], bins=bins, range=bounds[k])
            loc_hist = np.array(loc_hist, dtype=np.float64)
            global_hist = np.zeros_like(loc_hist, dtype=np.float64)
            this_comm.Allreduce(loc_hist, global_hist, op=MPI.SUM)
            local_counts, global_counts = np.zeros(1), np.zeros(1)
            local_counts[0] = np.prod(full_data[k].shape)
            this_comm.Allreduce(local_counts, global_counts, op=MPI.SUM)

            dx = bin_edges[1]-bin_edges[0]
            x_vals = bin_edges[:-1] + dx/2
            pdf = global_hist/global_counts/dx
            self.pdfs[k] = (pdf, x_vals, dx)
        self._calculate_pdf_statistics()

        

    def plot_pdfs(self, dpi=150):

        grid = PlotGrid(1,1, row_in=5, col_in=8.5)
        ax = grid.axes['ax_0-0']
        
        for k, data in self.pdfs.items():
            pdf, xs, dx = data
            mean, stdev, skew, kurt = self.pdf_stats[k]
            title = r'$\mu$ = {:.2g}, $\sigma$ = {:.2g}, skew = {:.2g}, kurt = {:.2g}'.format(mean, stdev, skew, kurt)
            ax.set_title(title)
            ax.axvline(mean, c='orange')

            ax.plot(xs, pdf, lw=2, c='k')
            ax.fill_between((mean-stdev, mean+stdev), pdf.min(), pdf.max(), color='orange', alpha=0.5)
            ax.fill_between(xs, 1e-16, pdf, color='k', alpha=0.5)
            ax.set_xlim(xs.min(), xs.max())
            ax.set_ylim(pdf.min(), pdf.max())
            ax.set_yscale('log')
            ax.set_xlabel(k)
            ax.set_ylabel('P({:s})'.format(k))

            grid.fig.savefig('{:s}/{:s}_pdf.png'.format(self.out_dir, k), dpi=dpi, bbox_inches='tight')
            ax.clear()

        self._save_pdfs()

    def _save_pdfs(self):
        with h5py.File('{:s}/pdf_data.h5'.format(self.out_dir), 'w') as f:
            for k, data in self.pdfs.items():
                pdf, xs, dx = data
                this_group = f.create_group(k)
                for d, n in ((pdf, 'pdf'), (xs, 'xs')):
                    dset = this_group.create_dataset(name=n, shape=d.shape, dtype=np.float64)
                    f['{:s}/{:s}'.format(k, n)][:] = d
                dset = this_group.create_dataset(name='dx', shape=(1,), dtype=np.float64)
                f['{:s}/dx'.format(k)][0] = dx

