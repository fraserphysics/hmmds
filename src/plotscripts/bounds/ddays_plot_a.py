"""ddays_plot_a.py Makes figure for dynamics days 2025

python ddays_plot_a.py input_path output_path

"""

import sys
import argparse
import pickle

import numpy
import numpy.linalg
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as pyplot


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Debugging plot')
    parser.add_argument('--start',
                        type=int,
                        default=10133,
                        help='Plot particles at 2 times starting here')
    parser.add_argument('input', type=str, help='Path to data')
    parser.add_argument('fig_path', type=str, help='Path to figure file')
    args = parser.parse_args(argv)
    return args


def plot_selected(axes, x_all, bins, indices: set, shift, t_x):
    """Plot points selected by indices shifted by steps

    Args:
        axes: Plot on this
        x_all: Long vector time series
        bins: For vertical lines
        indices: Unshifted times
        shift: Plots points at times = indices + shift
        t_x: Mark point t_x + shift
    """
    index_array = numpy.asarray([index for index in indices])
    shifted_indices = (index_array + shift,)
    axes.plot(
        x_all[shifted_indices, 0],
        x_all[shifted_indices, 2],
        markeredgecolor='none',
        color='#1f77b4',
        marker='.',
        markersize=2.5,
        linestyle='None',
    )
    axes.plot(x_all[t_x + shift, 0],
              x_all[t_x + shift, 2],
              marker='x',
              markersize=10,
              color='red')
    for boundary in bins:
        axes.plot((boundary,) * 2, (0, 50), color='black', linewidth=.5)
    axes.set_xlim(-22, 22)
    axes.set_ylim(0, 50)


def main(argv=None):
    """Plot some stuff
    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    with open(args.input, 'rb') as file_:
        dict_in = pickle.load(file_)
    bins = dict_in['bins']
    x_all = dict_in['x_all']
    y_q = numpy.digitize(x_all[:, 0], bins)
    n_all = len(y_q)

    n_times = 4
    figure, axeses = pyplot.subplots(nrows=2,
                                     ncols=n_times,
                                     figsize=(6, 3),
                                     sharex=True,
                                     sharey=True)
    indices = set(numpy.arange(len(x_all) - n_times))
    shift = 0
    t_x = args.start

    for shift in range(n_times):
        plot_selected(axeses[0, shift], x_all, bins, indices, shift, t_x)
        indices = indices & set(
            numpy.nonzero(y_q[shift:] == y_q[t_x + shift])[0])
        plot_selected(axeses[1, shift], x_all, bins, indices, shift, t_x)
    axeses[0, 0].set_ylabel('Forecast')
    axeses[1, 0].set_ylabel('Update')
    axeses[1, 0].set_yticks([])

    #pyplot.show()
    figure.tight_layout()
    figure.savefig(args.fig_path)


if __name__ == "__main__":
    sys.exit(main())
