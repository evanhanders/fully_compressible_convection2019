"""
Script for plotting a movie of the evolution of a 2D convection run.  
This script plots time evolution of the fields specified in 'fig_type'

Usage:
    plot_slices.py [options]

Options:
    --root_dir=<root_dir>               Directory pointing to 'slices', etc
    --fig_name=<fig_name>               Base name of each saved figure [default: snapshots]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Dedalus output file to start at [default: 1]
    --n_files=<num_files>               Number of files to plot [default: 100000]
    --dpi=<dpi>                         Image pixel density [default: 100]
    --fig_type=<fig_type>               Type of figure to plot
                                            1 - T_fluc, w, enstrophy
                                        [default: 1]
    --static_cbar                       If flagged, don't evolve the cbar with time
"""
from docopt import docopt
args = docopt(__doc__)
from base.plot_buddy import *
import logging
logger = logging.getLogger(__name__)


write_num_1 = int(args['--start_fig'])
n_files     = int(args['--n_files'])
start_file  = int(args['--start_file'])

root_dir    = args['--root_dir']
if root_dir is None:
    logger.info('No dedalus output dir specified, exiting')
    import sys
    sys.exit()
fig_name   = args['--fig_name']
out_dir     = '{:s}/{:s}/'.format(root_dir, fig_name)

if int(args['--fig_type']) == 1:
    fnames = [('s1', {}), ('Vort_y', {})]#, ('enstrophy', {'symm_cbar': False, 'cmap': 'Purples'})]
buddy1 =  MovieBuddy(root_dir, file_dirs=['slices'], start_file=start_file, n_files=n_files)

if buddy1.cw_rank == 0 and not os.path.exists('{:s}'.format(out_dir)):
    os.mkdir('{:s}'.format(out_dir))

for f in fnames:
    buddy1.add_subplot(f[0], **f[1])
buddy1.prepare_subplots(filetype='slices')

for i in range(buddy1.local_writes):
    logger.info('Plotting {}/{}'.format(i+1,buddy1.local_writes))
    grid   = PlotGrid(len(fnames), 1, col_in=10, row_in=2.5)
    for j, f in enumerate(fnames):
        if type(f) is tuple:
            f = f[0]
        buddy1.plot_colormesh(grid.axes['ax_0-{}'.format(j)], f, i, global_cmap=args['--static_cbar'])
    plt.suptitle('t = {:.4g}'.format(buddy1.local_times[i]))
    grid.fig.savefig('{:s}/{:s}_{:06d}.png'.format(out_dir, fig_name, buddy1.local_writes_below+write_num_1+i), dpi=200, bbox_inches='tight')
    plt.close(grid.fig)
