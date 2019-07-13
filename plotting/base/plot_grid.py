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
        print(self.width, self.height)
        self.fig       = plt.figure(figsize=(self.width, self.height))
        self.gs        = gridspec.GridSpec(1000,1000) #height units, then width units
        col_size       = int((1000 - padding*(self.ncols-1))/self.ncols) 
        row_size       = int((1000 - padding*(self.nrows-1))/self.nrows) 
        self.axes      = OrderedDict()
        for i in range(self.ncols):
            for j in range(self.nrows):
                self.axes['ax_{}-{}'.format(i,j)] = plt.subplot(self.gs.new_subplotspec(
                                                     (j*(row_size+padding), i*(col_size+padding)),
                                                     row_size, col_size))
