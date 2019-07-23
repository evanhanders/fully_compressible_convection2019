import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from collections import OrderedDict


class PlotGrid:
    """
    Sets up an even plot grid with a given number of rows and columns.
    Axes objects are stored in self.axes, with keys like 'ax_0-1', where
    the numbers refer to the column, then row of the plot (so they go
    left to right, then top to bottom)
    """

    def __init__(self, nrows, ncols, padding=25, col_in=3, row_in=3):
        self.nrows     = nrows
        self.ncols     = ncols
        self.width     = ncols*col_in
        self.height    = nrows*row_in
        self.padding   = padding
        self.fig       = plt.figure(figsize=(self.width, self.height))
        self.gs        = gridspec.GridSpec(1000,1000) #height units, then width units
        col_size       = int((1000 - padding*(self.ncols-1))/self.ncols) 
        row_size       = int((1000 - padding*(self.nrows-1))/self.nrows) 
        self.col_size, self.row_size = col_size, row_size
        self.axes      = OrderedDict()
        for i in range(self.ncols):
            for j in range(self.nrows):
                self.axes['ax_{}-{}'.format(i,j)] = plt.subplot(self.gs.new_subplotspec(
                                                     (j*(row_size+padding), i*(col_size+padding)),
                                                     row_size, col_size))

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
        self.axes['ax_0-{}'.format(row_num)] = plt.subplot(self.gs.new_subplotspec(
                                                    (0, col_num*(self.col_size+self.padding)),
                                                    1000, self.col_size))
