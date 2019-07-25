"""
Script for plotting traces of evaluated scalar quantities vs. time.

Usage:
    plot_scalars.py [options]

Options:
    --root_dir=<root_dir>               Directory pointing to 'slices', 'scalars', 
                                         'profiles' folders for relevant run
    --fig_name=<fig_name>               Output directory for figures [default: traces]
    --start_file=<start_file>           Dedalus output file to start at [default: 1]
    --n_files=<num_files>               Number of files to plot [default: 100000]
    --dpi=<dpi>                         Image pixel density [default: 150]
"""
import logging
logger = logging.getLogger(__name__)
from docopt import docopt
args = docopt(__doc__)
from plot_logic.scalars import ScalarFigure, ScalarPlotter

root_dir = args['--root_dir']
fig_name  = args['--fig_name']
start_file = int(args['--start_file'])
n_files     = args['--n_files']
if n_files is not None: n_files = int(n_files)


# Nu vs time
fig1 = ScalarFigure(1, 1, col_in=6, fig_name='nu_trace')
fig1.add_field(0, 'Nu')

# Re vs. time
fig2 = ScalarFigure(1, 1, col_in=6, fig_name='re_trace')
fig2.add_field(0, 'Re_rms')

# Flux equilibration vs. time
fig3 = ScalarFigure(1, 1, col_in=6, fig_name='flux_equil')
fig3.add_field(0, 'flux_equilibration')

# Energies
fig4 = ScalarFigure(5, 1, col_in=4, row_in=1.5, fig_name='energies')
fig4.add_field(0, 'KE')
fig4.add_field(1, 'KE')
fig4.add_field(0, 'IE_fluc')
fig4.add_field(2, 'IE_fluc')
fig4.add_field(0, 'PE_fluc')
fig4.add_field(3, 'PE_fluc')
fig4.add_field(0, 'TE_fluc')
fig4.add_field(4, 'TE_fluc')

# Load in figures and make plots
plotter = ScalarPlotter(root_dir, file_dir='scalar', fig_name=fig_name, start_file=start_file, n_files=n_files)
plotter.load_figures([fig1, fig2, fig3, fig4])
plotter.plot_figures(dpi=int(args['--dpi']))
