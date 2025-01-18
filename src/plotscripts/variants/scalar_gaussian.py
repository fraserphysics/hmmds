"""
ScalarGaussian.py SGO_sim plot_dir

Makes the following plots for fig:ScalarGaussian:

SGO_b.pdf  Simulated time series of states
SGO_c.pdf  Simulated time series of observations
SGO_d.pdf  Decoded time series of states

"""
import sys
import os.path
import argparse

import numpy

import plotscripts.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(
        description='Make 3 plots for the ScalarGaussian figure')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('data_path', type=str, help="path to data")
    parser.add_argument('fig_path', type=str, help="path to figure")
    return parser.parse_args(argv)


def main(argv=None):
    """Make plots SGO_%.pdf for % in b,c,d
    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    data = plotscripts.utilities.read_data(args.data_path)
    x = plotscripts.utilities.Axis(data=data[0],
                                   magnitude=False,
                                   label=r'$t$',
                                   ticks=numpy.arange(0, 100.1, 25))

    def _plot(y):
        fig = pyplot.figure(figsize=(2.5, 2.0))
        axis = plotscripts.utilities.sub_plot(fig, (1, 1, 1), x, y, color='b')
        axis.set_ylim(-0.02, 1.02)
        fig.subplots_adjust(bottom=0.15)  # Make more space for label
        fig.subplots_adjust(left=.15, bottom=.18)
        fig.tight_layout()
        return (axis, fig)

    _, fig_b = _plot(
        plotscripts.utilities.Axis(data=data[1],
                                   magnitude=False,
                                   label=r'$S(t)$',
                                   ticks=numpy.arange(0, 1.1, 1)))
    _, fig_d = _plot(
        plotscripts.utilities.Axis(data=data[3],
                                   magnitude=False,
                                   label=r'$S(t)$',
                                   ticks=numpy.arange(0, 1.1, 1)))

    axis_c, fig_c = _plot(
        plotscripts.utilities.Axis(data=data[2],
                                   magnitude=False,
                                   label=r'$y(t)$',
                                   ticks=numpy.arange(-4, 4.1, 4)))
    axis_c.set_ylim(-5, 5)
    fig_c.subplots_adjust(left=.2)
    if args.show:
        pyplot.show()
    else:
        fig_b.savefig(os.path.join(args.fig_path, 'SGO_b.pdf'))
        fig_c.savefig(os.path.join(args.fig_path, 'SGO_c.pdf'))
        fig_d.savefig(os.path.join(args.fig_path, 'SGO_d.pdf'))
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:
