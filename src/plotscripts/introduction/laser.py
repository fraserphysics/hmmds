"""laser.py plot the laser data

ToDo: Delete and use code in laser directories

"""
import sys
import argparse

import numpy

import plotscripts.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Make GaussMix.pdf')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('--data',
                        type=str,
                        help="path to LP5.DAT",
                        default='../../raw_data/LP5.DAT')
    parser.add_argument('--fig_path',
                        type=str,
                        help="path to figure",
                        default='../../build/figs/introduction/lp5.pdf')
    return parser.parse_args(argv)


def main(argv=None):
    """Initially just look at the data.

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    data = read_data(args.data)
    fig = LaserLP5(pyplot, data)

    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


def read_data(data_file):
    """Read in "data_file" as an array.  ToDo: replace with
    hmmds.applications.laser.utilities.read_tang
    """
    with open(data_file, 'r') as file:
        lines = file.readlines(
        )  # There are only 28278 lines and memory is big and cheap

    assert lines[0].split()[0] == 'BEGIN'
    assert lines[-1].split()[-1] == 'END'
    return numpy.array([[float(x) for x in line.split()] for line in lines[1:-1]
                       ]).T


def LaserLP5(pyplot, data):
    fig = pyplot.figure(figsize=(7, 5))
    x = plotscripts.utilities.Axis(data=data[0], magnitude=False, label=r'$t$')
    y = plotscripts.utilities.Axis(data=data[1], magnitude=False, label=r'$y$')
    plotscripts.utilities.sub_plot(fig, (1, 1, 1), x, y)
    return fig


if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:
