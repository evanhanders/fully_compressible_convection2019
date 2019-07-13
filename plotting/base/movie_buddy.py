from plot_buddy_base import *
import matplotlib
matplotlib.rcParams.update({'font.size': 11})
from scipy.stats.mstats import mode
from scipy import integrate
import matplotlib.colorbar as colorbar
import matplotlib.gridspec as gridspec

import logging
logger = logging.getLogger(__name__)

class MovieBuddy(PlotBuddy):
    """
        An extension of the PlotBuddy class which creates 2D colormap
        images of snapshots from dedalus runs.  This class' functionality
        includes making a figure with a large number of subplots of different sizes.

        Attributes:
            plots          - A python list, which will be filled with dictionaries containing
                            important information regarding each axis subplot.
            data          - A dictionary of NumPy arrays, containing simulation data which will be plotted
            figure        - The matplotlib figure which will be plotted on.
            gs            - A matplotlib grid spec which maps out where subplots can be on figure.
            fig_dims      - A tuple containing the dimensions, in inches, of the figure.
            gridspec_dims - A tuple containing the gridspec dimensions of the figure.
            dpi           - The dpi at which to save the output figure.
    """
     
    def __init__(self, directory, dpi=200, fig_dims=(8,5.5), gridspec_dims=(1,1), profile_post_file=None, **kwargs):
        """
        Calls the parent PlotBuddy __init__ function, then initializes the MovieBuddy's figure.
        """
        super(MovieBuddy, self).__init__(directory, **kwargs)
        if self.idle: return

        self.dpi            = dpi
        self.fig_dims       = fig_dims
        self.gridspec_dims  = gridspec_dims

        self.figure = plt.figure(figsize=fig_dims)
        self.gs     = gridspec.GridSpec(*gridspec_dims)
        self.plots  = list()
        self.data   = dict()

    def add_subplot(self, field, xvals='x', yvals='z', cmap='RdYlBu_r',
                        sub_t_x_avg=False, sub_t_avg=False, bare=False,
                    symm_cbar=True):
        """
            Adds a subplot to the list of subplots that the plotter will track.  Also stores
            important information about that subplot

            field       -- the field being plottedin this subplot
            xvals       -- the physical dimension of the problem on the horizontal axis ('x', 'y')
            yvals       -- the physical dimension of the problem on the vertical axis   ('y', 'z')
            cmap        -- The colormap of the plot
            sub_t_avg   -- If True, subtract the time averaged mean value from each movie frame.
            sub_t_x_avg -- If True, subtract the time- and horizontally-averaged profile from each movie frame.
            bare        -- If bare, all labels are removed.
        """
        if self.idle: return
        plot_info = dict()
        plot_info['field']       = field
        plot_info['xvals']       = xvals
        plot_info['yvals']       = yvals
        plot_info['cmap']        = cmap
        plot_info['sub_t_x_avg'] = sub_t_x_avg
        plot_info['sub_t_avg']   = sub_t_avg
        plot_info['bare']        = bare
        plot_info['symm_cbar']   = symm_cbar
        plot_info['t_avg']       = 0
        plot_info['t_x_avg']     = 0
        plot_info['max_val']     = (0, 0, 0) #max abs(), true max, true min
        plot_info['max_val_t_x'] = (0, 0, 0)
        plot_info['max_val_t']   = (0, 0, 0)
        self.plots.append(plot_info)

    def prepare_subplots(self, cutoff=0.25, filetype='slices'):
        """
        Read data from disk, and calculate important properties for each data profile
        (the horizontal average, total average, and a rough estimate of the maximum value
        over time for each profile.

        ***Must be run before plotting. 

        Inputs:
        -------
            cutoff - A percentage.  The percentage of total points to ignore when
                           finding the maximum value for colorbar purposes.  The purpose
                           of this is to cut off outlier points such that they don't kill
                           the color scale.
        """
        if self.idle: return

        # Go through all the subplots to see which fields we need.
        field_names = []
        for i, ax in enumerate(self.plots):
            if ax['field'] not in field_names:
                field_names.append(ax['field'])

        # Pull those fields out of the files
        self.data   = self.grab_full_task(self.local_files[filetype],
                                              self.local_good_local_indices,
                                              profile_name=field_names)

        # Calculate important global properties of each field
        t_avgs   = dict()
        t_x_avgs = dict()
        maxs     = dict()
        maxs_t_x    = dict()
        maxs_t    = dict()
        for k in field_names:
            if len(self.data[k].shape) == 4:
                if self.data[k].shape[1] == 1:
                    self.data[k] = self.data[k][:,0,:,:]
                elif self.data[k].shape[2] == 1:
                    self.data[k] = self.data[k][:,:,0,:]
                elif self.data[k].shape[3] == 1:
                    self.data[k] = self.data[k][:,:,:,0]
            t_avgs[k]   = self.get_time_avg(self.data[k])
            t_x_avgs[k] = self.get_time_horiz_avg(self.data[k])
            # We cut off the highest "cutoff%" data values in case there are explosions, etc.
            maxs[k]     = self.get_max_value(self.data[k], cutoff=cutoff)
            maxs_t_x[k] = self.get_max_value(self.data[k] - t_x_avgs[k], cutoff=cutoff)
            maxs_t[k]   = self.get_max_value(self.data[k] - t_avgs[k], cutoff=cutoff)

        # Store info back in plots
        for i, ax in enumerate(self.plots):
            self.plots[i]['t_avg']     = t_avgs[ax['field']]
            self.plots[i]['t_x_avg']   = t_x_avgs[ax['field']]
            self.plots[i]['max_val']   = maxs[ax['field']]
            self.plots[i]['max_val_t_x']   = maxs_t_x[ax['field']]
            self.plots[i]['max_val_t']   = maxs_t[ax['field']]

    def get_max_value(self, field, cutoff=0.25):
        """
        Given a multi-dimensional NumPy array, where the data are arranged in terms of
        (Time axis) x (Horizontal axes) x (Vertical axis), this function calculates the
        maximum value of the field across all processors.  The top few percentage of points
        are cut off in order to avoid ringing messing everything up (cutoff is how many
        % of points are removed)

        Inputs:
        -------
            field -- A NumPy array containing the values you want the maximum of.

        Outputs:
        --------
            The maximum global value in the allowed range
        """
        if self.idle: return

        flattened_abs = np.sort(np.abs(field.flatten()))
        flattened     = np.sort(field.flatten())
        local_max_abs = flattened_abs[int((100-cutoff)/100*len(flattened))-1]
        local_max     = flattened[int((100-cutoff)/100*len(flattened))-1]
        local_min     = flattened[int(cutoff/100*len(flattened))]

        returns = []
        for v, op in zip((local_max_abs, local_max, local_min), (MPI.MAX, MPI.MAX, MPI.MIN)):
            local_value    = np.zeros(1, np.float64)
            global_value   = np.zeros(1, np.float64)
            local_value[0] = v
            self.comm.Allreduce(local_value, global_value, op=op)
            returns.append(global_value[0])
        return returns


    def get_time_horiz_avg(self, field):
        """
        Given a multi-dimensional NumPy array, where the data are arranged in terms of
        (Time axis) x (Horizontal axes) x (Vertical axis), this function calculates the
        horizontal average at each time step, then finds what the global average is over
        all times.

        Inputs:
        -------
            field -- A NumPy array of 3 or 4 dimensions, (time, x, z) or (time, x, y, z)
                     containing values that you want the horizontal average of.
        Outputs:
        --------
            the mean horizontal profile, a NumPy array whose size is equal to the last
            axis of the input array.
        """
        if self.idle: return

        # Figure out how many total writes there are, for taking the average
        local_counts  = np.zeros(1, np.uint32)
        global_counts = np.zeros(1, np.uint32)
        local_counts[0] = field.shape[0]
        self.comm.Allreduce(local_counts, global_counts, op=MPI.SUM)

        while len(field.shape) > 2: #works for 2D or 3D, evenly spaced horizontal points.
            field = np.mean(field, axis=1)

        # Calculate the mean profile across all times, from all processes.
        collapsed_profile   = np.sum(field, axis=0)
        local_sum           = np.zeros(collapsed_profile.shape, dtype=np.float64)
        local_sum[:]        = collapsed_profile[:]
        global_sum          = np.zeros(local_sum.shape, dtype=np.float64)
        self.comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
        return global_sum/global_counts[0]

    def get_time_avg(self, field):
        """
        Given a multi-dimensional NumPy array, where the data are arranged in terms of
        (Time axis) x (Horizontal axes) x (Vertical axis), this function calculates the
        average value at each time step, then finds what the global average is over
        all times.

        Inputs:
        -------
            field -- A NumPy array of 3 or 4 dimensions, (time, x, z) or (time, x, y, z)
                     containing values that you want the average of.
        Outputs:
        --------
            the mean value of the profile, a float
        """
        if self.idle: return

        horiz_avg = self.get_time_horiz_avg(field)

        # Some outputs will be saved with Z already out of the picture.  If not, we need to
        # account for its potential unevenness.
        if len(horiz_avg) > 1:
            integral = integrate.cumtrapz(horiz_avg, self.z, initial=0)
            return integral[-1]/np.max(self.z) # Not perfect integration, but good enough for movie plots.
        else:
            return horiz_avg


    def plot_colormesh(self, ax, field, time_index, global_cmap=True):
        """
            Plots a 2D colormap of a specified set of data on a specified matplotlib axis.

            Inputs:
            -------
                ax          -- The matplotlib axis on which to plot
                field       -- a string containing the field name to plot
                time_index  -- time index of this plot on local processor
        """
        if self.idle: return
        this_p = None
        for p in self.plots:
            if p['field'] == field:
                this_p = p
                break
        if this_p is None:
            logger.info("Plot {} not properly prepared".format(field))
            raise
       
        plot_data = self.data[field][time_index,:,:]
        if global_cmap:
            if this_p['sub_t_x_avg']:
                if this_p['symm_cbar']:
                    c_max, c_min = this_p['max_val_t_x'][0], -1*this_p['max_val_t_x'][0]
                else:
                    c_max, c_min = this_p['max_val_t_x'][1:]
            elif this_p['sub_t_avg']:
                if this_p['symm_cbar']:
                    c_max, c_min = this_p['max_val_t'][0], -1*this_p['max_val_t'][0]
                else:
                    c_max, c_min = this_p['max_val_t'][1:]
            else:
                if this_p['symm_cbar']:
                    c_max, c_min = this_p['max_val'][0], -1*this_p['max_val'][0]
                else:
                    c_max, c_min = this_p['max_val'][1:]
        else:
            if this_p['symm_cbar']:
                c_max, c_min = np.abs(plot_data).max(), -np.abs(plot_data).max()
            else:
                c_max, c_min = plot_data.max(), plot_data.min()

        if this_p['xvals'] == 'x' and this_p['yvals'] == 'z':
            ys, xs = np.meshgrid(self.z, self.x)
        elif this_p['xvals'] == 'y' and this_p['yvals'] == 'z':
            ys, xs = np.meshgrid(self.z, self.y)
        elif this_p['xvals'] == 'x' and this_p['yvals'] == 'y':
            ys, xs = np.meshgrid(self.y, self.x)
        else:
            logger.info("xvals, yvals choices not known, exiting")
            raise


        # Plot things
        plot = ax.pcolormesh(xs, ys, plot_data, cmap=this_p['cmap'], vmin=c_min, vmax=c_max)

        # Add color bar
        divider = make_axes_locatable(ax)
        cax, kw = colorbar.make_axes(ax, fraction=0.07, pad=0.03, aspect=2, anchor=(0,0), location='top')
        cbar = colorbar.colorbar_factory(cax, plot, **kw)
        trans = cax.get_yaxis_transform()
        cax.set_xticklabels([])
        cax.set_xticks([])
        cax.tick_params(axis=u'both', which=u'both',length=0)

        # Go back to focusing on the main axis.
        plt.axes(ax)

        # Add labels and stuff, if appropriate to do so.
        if not this_p['bare']:
            xticks = np.array([np.min(xs), np.max(xs)])
            yticks = np.array([np.min(ys), np.max(ys)])
            plt.xticks(xticks, [r'${:1.2f}$'.format(tick) for tick in xticks], fontsize=11)
            plt.yticks(yticks, [r'${:1.2f}$'.format(tick) for tick in yticks], fontsize=11)
            fancy_str = []
            for val in (c_min, c_max):
                c_str = '${:1.2e}'.format(val)
                if 'e+0' in c_str:
                    new_str = c_str.replace('e+0', '\\times 10^{')
                elif 'e-0' in c_str:
                    new_str = c_str.replace('e-0', '\\times 10^{-')
                else:
                    new_str = c_str.replace('e', '\\times 10^{')
                new_str += '}$'
                fancy_str.append(new_str)

            if this_p['symm_cbar']:
                cax.annotate(r'$\pm${:s}'.format(fancy_str[1]), (1.02, 0.01), size=11, color='black', xycoords=trans, clip_on=True)
            else:
                cax.annotate(r'{:s}, {:s}'.format(fancy_str[0], fancy_str[1]), (1.02,0.01), size=11, color='black', xycoords=trans, clip_on=True)
#                cax.annotate(r'{:s}'.format(fancy_str[1]), (1.02,0.01), size=11, color='black', xycoords=trans)
            cax.annotate(r'{:s}'.format(field), (0,1.25), size=11, color='black', xycoords=trans, clip_on=True)
        else:
            plt.xticks([], [])
            plt.yticks([], [])


