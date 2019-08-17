"""
Script for plotting time-averaged profiles

Usage:
    plot_profiles.py <root_dir> [options]

Options:
    --fig_dir=<fig_dir>                 Output directory for figures [default: profile_plots]
    --fig_width=<fig_width>             Figure width in inches [default: 12]
    --fig_height=<fig_height>           Figure height in inches [default: 10]
    --start_file=<start_file>           Dedalus output file to start at [default: 1]
    --n_files=<num_files>               Number of files to plot [default: 100000]
    --dpi=<dpi>                         Image pixel density [default: 150]
    --avg_writes=<time>               Average over this many profiles [default: 20]
"""
import logging
logger = logging.getLogger(__name__)
from docopt import docopt
args = docopt(__doc__)
from base.plot_buddy import ProfileBuddy
from mpi4py import MPI

root_dir = args['<root_dir>']
fig_dir  = args['--fig_dir']
start_file = int(args['--start_file'])
n_files = int(args['--n_files'])
dpi=int(args['--dpi'])

fig_dims = (float(args['--fig_width']), float(args['--fig_height']))

if isinstance(root_dir, type(None)):
    logger.info('No root directory specified; exiting')
    import sys
    sys.exit()

plotter = ProfileBuddy(root_dir, avg_writes=float(args['--avg_writes']), n_files=n_files, start_file=start_file)

fluxes = ['F_cond_fluc_z', 'enth_flux_z', 'PE_flux_z', 'KE_flux_z', 'viscous_flux_z']
[plotter.track_profile(s) for s in fluxes]
fields = ['T1', 'enstrophy', 'Re_rms']
for f in fields:
    plotter.track_profile(f)
plotter.pull_tracked_profiles()
plotter.take_avg()
plotter.add_plot('fluxes_v_time.png', fluxes, sum_fields=True)
for f in fields:
    plotter.add_plot('profile_{}.png'.format(f), f)
plotter.save_profiles(fig_dir)
plotter.make_plots(out_dir=fig_dir)


