"""
TrainChar.py ../../derived_data/synthetic/TrainChar ../../figs/TrainChar.pdf

"""
import sys
import argparse

import numpy

import plotscripts.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    if not argv:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Make plot of many training characteristics')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('data_path', type=str, help="path to data")
    parser.add_argument('fig_path', type=str, help="path to figure")
    return parser.parse_args(argv)


def main(argv=None):
    """Call with arguments: data_file, fig_file

    """

    args = parse_args(argv)
    matplotlib, pyplot = plotscripts.utilities.import_matplotlib_pyplot(args)
    plotscripts.utilities.update_matplotlib_params(matplotlib)

    fig, axis = pyplot.subplots(1, 1, figsize=(6, 3))
    axis.set_xlabel(r'$n$')
    axis.set_ylabel(r'$\frac{\log(P(y_1^T|\theta(n))}{T}$')
    with open(args.data_path, 'r') as data_file:
        data = numpy.array([[float(part)
                             for part in line.split()]
                            for line in data_file.readlines()])
    _, n_seeds = data.shape
    for i in range(1, n_seeds):
        axis.semilogx(data[:, 0] + 1, data[:, i])
        #axis.plot(data[:,0], data[:,i])
    fig.subplots_adjust(bottom=0.15)  # Make more space for label

    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)  #Make sure to save it as a .pdf
    return 0


if __name__ == "__main__":
    sys.exit(main())
# Local Variables:
# mode: python
# End:
