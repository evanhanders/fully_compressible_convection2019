from plot_buddy_base import *
import matplotlib.ticker as ticker
from scipy.stats import mode
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from collections import OrderedDict


def find_root(x_vals, y_vals): #For now, just find one root
    """
    This class finds the root of a curve, given the x and y values
    """
    sign_bottom = y_vals[0]
    sign_flip = y_vals/sign_bottom
    guess = x_vals[np.where(sign_flip < 0)[0][0]]
#    guess = x_vals[np.argmin(np.abs(y_vals))]
    lower = (guess + x_vals[0])/2.0
    upper = (x_vals[-1] + guess)/2.0
    function = interp1d(x_vals, y_vals)

    root = brentq(function, lower, upper)
    return root


class ProfileBuddy(PlotBuddy):
    """
    An extension of the PlotBuddy class for examining profiles which have been averaged over time

    Attributes:
    -----------
        avg_time_unit
        avg_time
        out_dir
        output_dir

    """
    
    def __init__(self, root_dir, avg_writes=100, **kwargs):
        """
        Initializes the profile buddy by figuring out how long to average over, and also
        setting up variables for the output directories.
        """
        super(ProfileBuddy, self).__init__(root_dir, **kwargs)
        self.reset()
        if self.idle: return
        self.avg_writes = avg_writes
        self.local_profiles = 1
        self.create_comm_world()

    def create_comm_world(self):
        '''
        Creates a smaller comm group that is only the size of the number of l2_avgs that
        are going to be done in this run
        '''
        if self.idle: return
        self.average_groups = int(np.ceil(len(self.global_times) / self.avg_writes))
        
        # Reorganize comm world
        if self.average_groups < self.cw_size:
            if self.cw_rank < self.average_groups:
                self.comm = self.comm.Create(self.comm.Get_group().Incl(np.arange(self.average_groups)))
                self.cw_rank = self.comm.rank
                self.cw_size = self.comm.size
            else:
                self.idle = True
                return
        
        self.local_profiles = int(np.floor(self.average_groups/self.cw_size))
        if self.cw_rank <  int(self.average_groups % self.cw_size):
            self.local_profiles += 1
            self.averages_below = self.local_profiles*self.cw_rank
        else:
            self.averages_below = (self.local_profiles + 1)*(self.average_groups % self.cw_size)
            self.averages_below += (self.local_profiles)*(self.cw_rank - self.average_groups % self.cw_size)

        tot_write_num = np.cumsum(self.global_writes_per_file)
        first = tot_write_num > self.averages_below
        last  = tot_write_num <= self.averages_below + self.local_profiles*self.avg_writes
        if self.averages_below + self.local_profiles*self.avg_writes > tot_write_num[last][-1] and\
            self.averages_below + self.local_profiles*self.avg_writes < tot_write_num[-1]:
            last[np.where(last == True)[0][-1] + 1] = True
        indices = list(np.arange(len(tot_write_num), dtype=int)[first*last])

        for d in self.file_dirs:
            self.local_files[d]  = []
            for i in indices:
                self.local_files[d] += [self.global_files[d][i]] #workaround b/c list, not numpy array


        self.start_avg_index = tot_write_num[first*last][0] - self.averages_below - tot_write_num[0]
        self.end_avg_index = tot_write_num[first*last][-1] - (self.averages_below + self.local_profiles*self.avg_writes)
        self.tot_write_num = tot_write_num

        logger.info('breaking up profiles across {} processes'.format(self.cw_size))

    def add_plot(self, plot_name, y_fields, x_field = 'z', log_y=False, sum_fields = False, coeffs=False):
        if self.idle: return
        self.plots.append(dict())
        self.plots[-1]['y_fields'] = y_fields
        self.plots[-1]['x_field']  = x_field
        self.plots[-1]['log_y']    = log_y
        if '.png' not in plot_name:
            plot_name += '.png'
        self.plots[-1]['plot_name'] = plot_name
        self.plots[-1]['sum_fields'] = sum_fields
        self.plots[-1]['coeffs']     = coeffs

    def reset(self):
        if self.idle: return
        self.tracked_profiles = list()
        self.tracked_powers = list()
        self.plots = []
        self.pulled_profiles = None
        self.averaged_profiles = None
        self.start_times = None
        self.x_waves = None

    def track_profile(self, profile, powers=False):
        if self.idle: return
        if powers:
            self.tracked_powers.append(profile)
        else:
            self.tracked_profiles.append(profile)

    def pull_tracked_profiles(self):
        if self.idle: return
        if self.pulled_profiles is None:
            self.pulled_profiles = self.grab_full_task( self.local_files['profiles'],\
                                                        self.all_good_local_indices,\
                                                        profile_name = self.tracked_profiles)

        else:
            local_profiles = self.grab_full_task( self.local_files['profiles'],\
                                                        self.all_good_local_indices,\
                                                        profile_name = self.tracked_profiles)
            for k in local_profiles.keys:
                self.pulled_profiles[k] = local_profiles[k]
        try:
            power_profiles = self.grab_full_task( self.local_files['powers'],\
                                                      self.all_good_local_indices,\
                                                      profile_name = self.tracked_powers)
            for k, item in power_profiles.items():
                self.pulled_profiles[k] = np.squeeze(item)
        except:
            logger.info('no powers found')

        for k in self.tracked_profiles:
            self.pulled_profiles[k] = np.squeeze(self.pulled_profiles[k])

    def get_power_spectrum(self, field='u', depth=0.05):
        if self.idle: return
        slices = self.grab_full_task(self.local_files['slices'], self.all_good_local_indices, profile_name=[field])

        right_depth = self.z - depth*self.atmosphere['Lz']
        depth_ind   = np.argmin(np.abs(right_depth))

        right_shape = list(slices[field].shape)
        right_shape.pop(-1)
        power = np.zeros(right_shape)
        waves = 2*np.pi*np.fft.fftfreq(right_shape[-1], self.x[1]-self.x[0]) #assumes even x spacing
        for i in range(right_shape[0]):
            spectrum = np.fft.fft(slices[field][i,:,depth_ind]) #only works for 2D
            power[i,:] = (spectrum * np.conj(spectrum)).real
        if isinstance(self.pulled_profiles, type(None)):
            self.pulled_profiles = OrderedDict()
        self.pulled_profiles[field+'_power'] = power
        self.tracked_profiles.append(field + '_power')
        self.x_waves = waves
        

    def take_avg(self, do_l2=False):
        if self.idle: return
        if do_l2:
            print('ERROR, NOT IMPLEMENTED')
            return

        self.averaged_profiles = OrderedDict()
        self.start_times = []
        base_indx = None
        for j, k in enumerate((self.tracked_powers + self.tracked_profiles)):
            self.averaged_profiles[k] = np.zeros((self.local_profiles,self.pulled_profiles[k].shape[-1]))
            for i in range(self.local_profiles):
                last_ind = self.start_avg_index+self.avg_writes*(i+1)
                if self.tot_write_num[-1]-1 < last_ind:
                    last_ind = self.tot_write_num[-1]-1
                self.averaged_profiles[k][i,:] += np.sum(self.pulled_profiles[k][int(self.start_avg_index+self.avg_writes*i):int(last_ind),:], axis=0)/self.avg_writes
                if j == 0:
                    self.start_times += [self.global_times[int(self.start_avg_index+self.avg_writes*i)]]

    def save_profiles(self, out_dir):
        """
        Saves the averaged profiles on the first process.
        """
        if self.idle: return
        output_directory = self.root_dir + '/' + out_dir + '/'
        self.comm.Barrier()
        f_name = output_directory + '/profile_info.h5'
        if self.cw_rank == 0 and not os.path.exists('{:s}'.format(output_directory)):
            os.mkdir('{:s}'.format(output_directory))

        local_holder = np.zeros((self.average_groups, self.averaged_profiles[self.tracked_profiles[0]].shape[-1]))
        global_holder = np.zeros((self.average_groups, self.averaged_profiles[self.tracked_profiles[0]].shape[-1]))
        global_profiles = OrderedDict()

        do_powers = False
        for k in self.averaged_profiles.keys():
            if '_power' in k:
                do_powers = True
                continue
            local_holder[self.averages_below:self.averages_below+self.local_profiles] = self.averaged_profiles[k]
            self.comm.Allreduce(local_holder, global_holder, op=MPI.SUM)
            global_profiles[k] = 1*global_holder
            global_holder *= 0
            local_holder *= 0

        if do_powers:
            local_holder = np.zeros((self.average_groups, len(self.x_waves)))
            global_holder = np.zeros((self.average_groups,len(self.x_waves)))
            for k in self.averaged_profiles.keys():
                if '_power' not in k:
                    continue
                local_holder[self.averages_below:self.averages_below+self.local_profiles] = self.averaged_profiles[k]
                self.comm.Allreduce(local_holder, global_holder, op=MPI.SUM)
                global_profiles[k] = 1*global_holder
                global_holder *= 0
                local_holder *= 0
            

        if self.cw_rank == 0:
            with h5py.File(f_name, 'w') as f:
                for k, data in global_profiles.items():
                    f[k] = data
                f['z'] = self.z
                try:
                    f['x_waves'] = self.x_waves
                except:
                    logger.info('failed to add x_waves to file')

    def make_plots(self, figsize=(10,6), out_dir='profile_plots', dpi=300, plot_root=False):
        if self.idle: return
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#        colors = plt.rcParams['axes.color_cycle']
        output_directory = self.root_dir + '/' + out_dir + '/'
        self.comm.Barrier()
        if self.cw_rank == 0 and not os.path.exists('{:s}'.format(output_directory)):
            os.mkdir('{:s}'.format(output_directory))
        logger.info('saving figures to {}'.format(output_directory))

        for i, plot in enumerate(self.plots):
            if len(self.tracked_profiles) > 0:
                key = self.tracked_profiles[0]
            else:
                key = self.tracked_powers[0]
            for j in range(self.averaged_profiles[key].shape[0]):
                fig = plt.figure(figsize=figsize)
                logger.info('Plotting {}: {} v {} # {}'.format(plot['plot_name'], plot['y_fields'], plot['x_field'], j+1))

                if not plot['coeffs']:
                    if plot['x_field'] == 'z':
                        x_f = self.z
                    else:
                        logger.error('Unrecognized x-field chosen')
                        return
                else:
                    x_f = np.arange(self.averaged_profiles[plot['y_fields'][0]][j,:].shape[-1])
                if not isinstance(plot['y_fields'], str):
                    for k, y_f in enumerate(plot['y_fields']):
                        y = self.averaged_profiles[y_f][j,:]
                        root_str = ''
                        if plot['log_y']:
                            if plot_root:
                                root = find_root(x_f, self.averaged_profiles[y_f][j,:])
                                plt.axvline(root, color=colors[k])
                                root_str += '; root={:.5g}'.format(root)
                            plt.plot(x_f[y > 0], y[y > 0], color=colors[k], label='{:s}'.format(y_f) + root_str)
                            plt.plot(x_f[y < 0], -y[y < 0], color=colors[k], ls='--')
                        else:
                            if plot_root:
                                root = find_root(x_f, self.averaged_profiles[y_f][j,:])
                                plt.axvline(root, color=colors[k])
                                root_str += '; root={:.5g}'.format(root)
                            print(self.averaged_profiles[y_f].shape, y_f)
                            plt.plot(x_f, self.averaged_profiles[y_f][j,:], color=colors[k % len(colors)], label='{:s}'.format(y_f) + root_str)
                    if plot['sum_fields']:
                        sum_f = np.zeros_like(self.averaged_profiles[y_f][j,:])
                        for k, y_f in enumerate(plot['y_fields']):
                            sum_f += self.averaged_profiles[y_f][j,:]
                        root_str = ''
                        if plot['log_y']:
                            if plot_root:
                                root = find_root(x_f, sum_f)
                                plt.axvline(root, color=colors[len(plot['y_fields'])])
                                root_str += '; root={:.5g}'.format(root)
                            plt.plot(x_f[sum_f > 0], sum_f[sum_f > 0], color=colors[len(plot['y_fields'])], label='sum' + root_str)
                            plt.plot(x_f[sum_f < 0], -sum_f[sum_f < 0], color=colors[len(plot['y_fields'])], ls = '--')
                        else:
                            if plot_root:
                                root = find_root(x_f, sum_f)
                                plt.axvline(root, color=colors[len(plot['y_fields'])])
                                root_str += '; root={:.5g}'.format(root)
                            plt.plot(x_f, sum_f, color=colors[len(plot['y_fields'])], label='sum' + root_str)
                else:
                    plt.plot(x_f, self.averaged_profiles[plot['y_fields']][j,:], color=colors[0], label='{:s}'.format(plot['y_fields']))
                plt.legend(loc='best')
                plt.xlabel(plot['x_field'])
                plt.xlim(np.min(x_f), np.max(x_f))
                if plot['log_y'] or plot['coeffs']:
                    plt.yscale('log')
                name = plot['plot_name'].split('.png')
                plt.title('Start avg time: {}'.format(self.start_times[j]))
                name = name[0] + '_{:04d}.png'.format(j + self.averages_below)
                plt.savefig(output_directory+name, dpi=dpi, bbox_inches='tight', figsize=figsize)
                plt.close()

        
        

