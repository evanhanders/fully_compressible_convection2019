"""
Script for plotting files which follow entropy perturbations, velocity fields, 
horizontal power spectra, and entropy gradient

Usage:
    plot_scalars.py [options]

Options:
    --root_dir=<root_dir>               Directory pointing to 'slices', 'scalars', 
                                         'profiles' folders for relevant run
    --fig_dir=<fig_dir>                 Output directory for figures [scalar_plots]
    --fig_width=<fig_width>             Figure width in inches [default: 12]
    --fig_height=<fig_height>           Figure height in inches [default: 10]
    --start_file=<start_file>           Dedalus output file to start at [default: 1]
    --n_files=<num_files>               Number of files to plot [default: 100000]
    --dpi=<dpi>                         Image pixel density [default: 150]
    --avg_start_time=<time>             Average simulation start time
    --no_avg                            If true, don't plot mean
"""
import logging
logger = logging.getLogger(__name__)
from docopt import docopt
args = docopt(__doc__)
from base.plot_buddy import ScalarBuddy
from mpi4py import MPI

root_dir = args['--root_dir']
fig_dir  = args['--fig_dir']
avg_start_time = args['--avg_start_time']
if not isinstance(avg_start_time, type(None)):
    avg_start_time = float(avg_start_time)
start_file = int(args['--start_file'])
n_files = int(args['--n_files'])
dpi=int(args['--dpi'])
no_avg=True#args['--no_avg']

fig_dims = (float(args['--fig_width']), float(args['--fig_height']))

if isinstance(root_dir, type(None)):
    logger.info('No root directory specified; exiting')
    import sys
    sys.exit()

plotter = ScalarBuddy(root_dir, n_files=n_files, start_file=start_file)
try:
    plotter.track_scalar('Nu')
    plotter.track_scalar('Re_rms')
    energies = ['KE', 'IE_fluc', 'PE_fluc', 'TE_fluc']
    [plotter.track_scalar(s) for s in energies]
    plotter.pull_tracked_scalars()

    plotter.add_plot('Energy_v_time.png', energies, start_x_avg=avg_start_time)
    plotter.add_plot('nu_v_time.png', 'Nu', start_x_avg=avg_start_time)
    plotter.add_plot('re_v_time.png', 'Re_rms', start_x_avg=avg_start_time)
    plotter.make_plots(do_avg=not(no_avg))
    plotter.save_scalars()
except:
    raise
    logger.info("Not a rotating run, no Rossby plot")

plotter.reset()

