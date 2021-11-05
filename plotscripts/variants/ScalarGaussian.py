"""
ScalarGaussian.py SGO_sim plot_dir

Makes the following plots for fig:ScalarGaussian:

SGO_b.pdf  Simulated time series of states
SGO_c.pdf  Simulated time series of observations
SGO_d.pdf  Decoded time series of states

"""
DEBUG = False
import sys
import os.path

import numpy

import plotscripts.utilities


def main(argv=None):
    """Make plots SGO_%.pdf for % in b,c,d

    """

    import matplotlib  # pylint: disable=import-outside-toplevel

    global DEBUG
    if DEBUG:
        matplotlib.rcParams['text.usetex'] = False
    else:
        matplotlib.use('PDF')
    import matplotlib.pyplot  # pylint: disable=import-outside-toplevel

    if argv is None:  # Usual case
        argv = sys.argv[1:]
    sim_file, fig_dir = argv

    params = {
        'axes.labelsize': 18,  # Plotting parameters for latex
        #'text.fontsize': 15,
        'legend.fontsize': 15,
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'Computer Modern Roman',
        'xtick.labelsize': 15,
        'ytick.labelsize': 15
    }
    matplotlib.rcParams.update(params)

    data = plotscripts.utilities.read_data(sim_file)
    X = plotscripts.utilities.axis(data=data[0],
                                   magnitude=False,
                                   label=r'$t$',
                                   ticks=numpy.arange(0, 100.1, 25))

    def _plot(Y):
        fig = matplotlib.pyplot.figure(figsize=(3.5, 2.5))
        ax = plotscripts.utilities.SubPlot(fig, (1, 1, 1), X, Y, color='b')
        ax.set_ylim(-0.02, 1.02)
        fig.subplots_adjust(bottom=0.15)  # Make more space for label
        fig.subplots_adjust(left=.15, bottom=.18)
        return (ax, fig)

    ax, fig_b = _plot(
        plotscripts.utilities.axis(data=data[1],
                                   magnitude=False,
                                   label=r'$S(t)$',
                                   ticks=numpy.arange(0, 1.1, 1)))
    ax, fig_d = _plot(
        plotscripts.utilities.axis(data=data[3],
                                   magnitude=False,
                                   label=r'$S(t)$',
                                   ticks=numpy.arange(0, 1.1, 1)))

    ax, fig_c = _plot(
        plotscripts.utilities.axis(data=data[2],
                                   magnitude=False,
                                   label=r'$y(t)$',
                                   ticks=numpy.arange(-4, 4.1, 4)))
    ax.set_ylim(-5, 5)
    fig_c.subplots_adjust(left=.2)
    if DEBUG:
        matplotlib.pyplot.show()
    else:
        fig_b.savefig(os.path.join(fig_dir, 'SGO_b.pdf'))
        fig_c.savefig(os.path.join(fig_dir, 'SGO_c.pdf'))
        fig_d.savefig(os.path.join(fig_dir, 'SGO_d.pdf'))
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:
