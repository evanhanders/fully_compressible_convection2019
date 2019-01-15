from plot_buddy_base import *
plt.style.use('ggplot')
from collections import OrderedDict

class ScalarBuddy(PlotBuddy):
    """
    A Class which inherits all of the functionality of PlotBuddy, and then extends it
    such that it is useful in tracking scalars.  

    New Attributes (See PlotBuddy for inherited attributes):
    -----------
    self.tracked_scalars
    self.pulled_scalars
    self.plots


    """

    def __init__(self, *args, **kwargs):
        super(ScalarBuddy, self).__init__(*args, **kwargs)
        self.reset()
        if self.cw_rank != 0:
            self.idle = True
        else:
            self.file_number_start = 1
            self.all_good_local_indices = [np.array(a) - int(np.round(np.sum(self.global_writes_per_file[:self.file_number_start+i-1]))) for i,a in enumerate(self.all_good_global_indices)]
            self.local_writes_per_file = self.global_writes_per_file
            self.local_times = self.global_times.flatten()
            self.local_total_writes = int(self.local_times.shape[0])
            self.local_files = self.global_files

    def track_scalar(self, scalar):
        if self.idle: return
        self.tracked_scalars.append(scalar)

    def pull_tracked_scalars(self):
        if self.idle: return
        self.pulled_scalars = self.grab_full_task(  self.local_files['scalar'],\
                                                        self.all_good_local_indices,\
                                                        profile_name=self.tracked_scalars)
        for k in self.tracked_scalars:
            self.pulled_scalars[k] = self.pulled_scalars[k].flatten()

    def add_plot(self, plot_name, y_fields, x_field = 'sim_time', start_x_avg = None, log_x=False, log_y=False):
        if self.idle: return
        
        self.plots.append(OrderedDict())

        if isinstance(start_x_avg, type(None)):
            start_x_avg = self.sim_start_time

        self.plots[-1]['y_fields'] = y_fields
        self.plots[-1]['x_field'] = x_field
        self.plots[-1]['log_x']    = log_x
        self.plots[-1]['log_y']    = log_y
        self.plots[-1]['start_x_avg'] = start_x_avg

        if '.png' not in plot_name:
            plot_name += '.png'
        self.plots[-1]['plot_name'] = plot_name

    def reset(self):
        self.tracked_scalars = list()
        self.plots = []
        self.pulled_scalars = None

    def make_plots(self, figsize=None, outdir='scalar_plots', dpi=300, t_buoy=True, do_avg=True):
        '''
            Create all of the plots!
        '''
        if self.idle: return
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        output_directory = self.root_dir + '/' + outdir + '/'
        if self.cw_rank == 0 and not os.path.exists('{:s}'.format(output_directory)):
            os.mkdir('{:s}'.format(output_directory))
        logger.info('saving figures to {}'.format(output_directory))

        for i, plot in enumerate(self.plots):
            fig = plt.figure(figsize=figsize)
            logger.info('Plotting {}:  {} v {}'.format(plot['plot_name'], plot['y_fields'], plot['x_field']))

            x_div = 1
            if plot['x_field'] == 'sim_time':
                x_f = self.local_times
                if t_buoy:
                    try:
                        x_div = self.atmosphere['t_buoy']
                    except:
                        logger.info('no t_buoy found')
            else:
                x_f = self.pulled_scalars[plot['x_field']]
            if not isinstance(plot['y_fields'], str):
                for j, y_f in enumerate(plot['y_fields']):
                    mean_val = np.mean(self.pulled_scalars[y_f][x_f >= plot['start_x_avg']])
                    label='{:s}; mean={:.4g}'.format(y_f, mean_val)
                    if not do_avg:
                        label = label.split(';')[0]
                    plt.plot(x_f/x_div, self.pulled_scalars[y_f], color=colors[j], label=label)
                    if do_avg:
                        xs, ys = [plot['start_x_avg'], np.max(x_f)], [mean_val]*2
                        plt.plot(np.array(xs)/x_div, ys, color=colors[j])
            else:
                mean_val = np.mean(self.pulled_scalars[plot['y_fields']][x_f >= plot['start_x_avg']])
                label='{:s}; mean={:.4g}'.format(plot['y_fields'], mean_val)
                if not do_avg:
                    label = label.split(';')[0]
                plt.plot(x_f/x_div, self.pulled_scalars[plot['y_fields']], color=colors[0], label=label)
                print(self.pulled_scalars[plot['y_fields']])
                xs, ys = [plot['start_x_avg'], np.max(x_f)], [mean_val]*2
                if do_avg:
                    plt.plot(np.array(xs)/x_div, ys, color=colors[0])
                    plt.ylabel(plot['y_fields'])
            plt.legend(loc='best')
            if t_buoy and plot['x_field'] == 'sim_time':
                try:
                    plt.xlabel(r'Time (buoyancy units, $t_b = {:.2g}$)'.format(self.atmosphere['t_buoy']))
                except:
                    plt.xlabel('Time')
            else:
                plt.xlabel(plot['x_field'])
            if plot['log_x']:
                plt.xscale('log')
            if plot['log_y']:
                plt.yscale('log')

            plt.savefig(output_directory+plot['plot_name'], dpi=dpi, bbox_inches='tight', figsize=figsize)
            plt.close()

    def save_scalars(self, outdir='scalar_plots'):
        if self.idle: return
        if self.cw_rank != 0: return

        output_directory = self.root_dir + '/' + outdir + '/'
        if self.cw_rank == 0 and not os.path.exists('{:s}'.format(output_directory)):
            os.mkdir('{:s}'.format(output_directory))
        logger.info('saving file to {}'.format(output_directory))

        with h5py.File('{}/scalar_values.h5'.format(output_directory), 'w') as f:
            for k, data in self.pulled_scalars.items():
                f[k] = data
            f['sim_time'] = self.local_times
            try:
                f['t_buoy'] = self.atmosphere['t_buoy']
                f['t_therm'] = self.atmosphere['t_therm']
            except:
                logger.info('cant find thermal time and buoyancy time in file')



        
