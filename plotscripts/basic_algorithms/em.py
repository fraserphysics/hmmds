""" EM.py: Creates Fig. 2.8 of the book

python EM.py outfile.pdf
"""
import sys
import argparse

import numpy
#from mpl_toolkits.mplot3d import Axes3D  # for  "projection='3d'".

import plotscripts.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Make GaussMix.pdf')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('fig_path', type=str, help="path to figure")
    return parser.parse_args(argv)


def main(argv=None):
    """Make plots of auxiliary function Q and consequent 1-d map for EM
    algorithm.

    """

    args, matplotlib, pyplot = plotscripts.utilities.import_and_parse(
        parse_args, argv)

    fig = pyplot.figure(figsize=(9, 5))
    axis_0 = fig.add_subplot(1, 2, 1, projection='3d', azim=-109, elev=30)
    axis_0.set_xlabel(r'$\theta$')
    axis_0.set_ylabel(r"$\theta'$")
    axis_0.set_zlabel(r"$Q(\theta',\theta)$")
    x_s = numpy.arange(0.1, 0.9, 0.05)
    y_s = numpy.arange(0.2, 0.8, 0.05)
    n_x = len(x_s)
    n_y = len(y_s)
    z_s = numpy.empty((n_x, n_y)).T
    for i in range(n_x):
        x = x_s[i]
        for j in range(n_y):
            y = y_s[j]
            z_s[j,
                i] = (1 + 2 * x) * numpy.log(y) + (1 + 2 *
                                                   (1 - x)) * numpy.log(1 - y)
    axis_0.set_xticks(numpy.arange(0.2, 0.8, .2))
    axis_0.set_yticks(numpy.arange(0.3, 0.8, .2))
    x_grid, y_grid = numpy.meshgrid(x_s, y_s)
    axis_0.plot_surface(
        x_grid,
        y_grid,
        z_s,
        rstride=1,
        cstride=1,
        cmap=matplotlib.cm.hsv,  # pylint: disable=no-member
        linewidth=1)
    axis_1 = fig.add_subplot(1, 2, 2)
    x = numpy.arange(0, 1.1, 1)
    y = 0.25 + x / 2.0
    axis_1.plot(x, x, label='slope 1 referece')
    axis_1.plot(x, y, label=r'$\cal{T}(\theta)$')
    axis_1.set_xlabel(r'$\theta$')
    axis_1.set_ylabel(r"$\cal{T}(\theta)$")
    axis_1.legend(loc='lower right')
    ticks = numpy.arange(0, 1.1, 0.25)
    axis_1.set_xticks(ticks)
    axis_1.set_yticks(ticks)
    if args.show:
        pyplot.show()
    else:
        fig.savefig(args.fig_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
# Local Variables:
# mode: python
# End:
