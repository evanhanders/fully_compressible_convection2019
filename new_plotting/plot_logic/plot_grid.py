import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from collections import OrderedDict

def fl_int(num):
    return int(np.floor(num))


class PlotGrid:
    """
    Sets up an even plot grid with a given number of rows and columns.
    Axes objects are stored in self.axes, with keys like 'ax_0-1', where
    the numbers refer to the column, then row of the plot (so they go
    left to right, then top to bottom)
    """

    def __init__(self, nrows, ncols, padding=50, col_in=3, row_in=3):
        self.nrows     = nrows
        self.ncols     = ncols
        self.width     = int(np.round(ncols*col_in))
        self.height    = int(np.round(nrows*row_in))
        self.padding   = padding
        self.fig       = plt.figure(figsize=(self.width, self.height))
        self.gs        = gridspec.GridSpec(1000,1000) #height units, then width units
        self.col_size       = fl_int((1000 - padding*(self.ncols-1))/self.ncols) 
        self.row_size       = fl_int((1000 - padding*(self.nrows-1))/self.nrows) 
        self.axes      = OrderedDict()
        self._make_subplots()

    def _make_subplots(self):
        for i in range(self.ncols):
            for j in range(self.nrows):
                self.axes['ax_{}-{}'.format(i,j)] = plt.subplot(self.gs.new_subplotspec(
                                                     (j*(self.row_size+self.padding), i*(self.col_size+self.padding)),
                                                     self.row_size, self.col_size))

    def full_row_ax(self, row_num):
        """ row number indexing starts at 0 """
        for i in range(self.ncols):
            del self.axes['ax_{}-{}'.format(i, row_num)]
        self.axes['ax_0-{}'.format(row_num)] = plt.subplot(self.gs.new_subplotspec(
                                                    (row_num*(self.row_size+self.padding), 0),
                                                    self.row_size, 1000))

    def full_col_ax(self, col_num):
        """ col number indexing starts at 0 """
        for i in range(self.nrows):
            del self.axes['ax_{}-{}'.format(col_num, i)]
        self.axes['ax_{}-0'.format(col_num)] = plt.subplot(self.gs.new_subplotspec(
                                                    (0, col_num*(self.col_size+self.padding)),
                                                    1000, self.col_size))

class ColorbarPlotGrid(PlotGrid):
    
    def __init__(self, *args, **kwargs):
        self.cbar_axes = OrderedDict()
        super(ColorbarPlotGrid, self).__init__(*args, **kwargs)

    def _make_subplots(self):
        for i in range(self.ncols):
            for j in range(self.nrows):
                self.axes['ax_{}-{}'.format(i,j)] = plt.subplot(self.gs.new_subplotspec(
                                                     (fl_int(j*(self.row_size+self.padding) + 0.2*self.row_size), fl_int(i*(self.col_size+self.padding))),
                                                     fl_int(self.row_size*0.8), fl_int(self.col_size)))
                self.cbar_axes['ax_{}-{}'.format(i,j)] = plt.subplot(self.gs.new_subplotspec(
                                                     (fl_int(j*(self.row_size+self.padding)), fl_int(i*(self.col_size+self.padding))),
                                                     fl_int(self.row_size*0.1), fl_int(self.col_size)))
    def full_row_ax(self, row_num):
        """ row number indexing starts at 0 """
        for i in range(self.ncols):
            del self.axes['ax_{}-{}'.format(i, row_num)]
            self.axes['ax_0-{}'.format(row_num)] = plt.subplot(self.gs.new_subplotspec(
                                                (fl_int(row_num*(self.row_size+self.padding) + 0.2*self.row_size), 0),
                                                fl_int(self.row_size*0.8), 1000))
            self.cbar_axes['ax_0-{}'.format(row_num)] = plt.subplot(self.gs.new_subplotspec(
                                                     (fl_int(row_num*(self.row_size+self.padding)), 0),
                                                     fl_int(self.row_size*0.1), 1000))

    def full_col_ax(self, col_num):
        """ col number indexing starts at 0 """
        for i in range(self.nrows):
            del self.axes['ax_{}-{}'.format(col_num, i)]
        self.axes['ax_{}-0'.format(col_num)] = plt.subplot(self.gs.new_subplotspec(
                                            (0, fl_int(col_num*(self.col_size+self.padding))),
                                            1000, fl_int(self.col_size)))
        self.cbar_axes['ax_{}-0'.format(col_num)] = plt.subplot(self.gs.new_subplotspec(
                                                     (0, fl_int(col_num*(self.col_size+self.padding))),
                                                     1000, fl_int(self.col_size)))

