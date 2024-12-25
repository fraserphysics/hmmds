"""ddays_plot_b.py Makes figure for dynamics days 2025

python ddays_plot_b.py input_path output_path

"""

import sys
import argparse
import pickle

import numpy
import numpy.linalg

import plotscripts.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Debugging plot')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('--start',
                        type=int,
                        default=72,
                        help='Plot particles at 2 times starting here')
    parser.add_argument('input', type=str, help='Path to data')
    parser.add_argument('fig_path', type=str, help='Path to figure file')
    args = parser.parse_args(argv)
    return args


def plot_point(axes, x, color, label=None):
    axes.plot(x[0],
              x[2],
              markeredgecolor='none',
              marker='.',
              markersize=5,
              linestyle='None',
              color=color,
              label=label)


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
    clouds = dict_in['clouds']

    n_times = 2
    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)
    figure, axeses = pyplot.subplots(nrows=2,
                                     ncols=n_times,
                                     figsize=(6, 6),
                                     sharex=True,
                                     sharey=True)

    for i in range(args.start, args.start + 2):
        forecast = clouds[(i, 'forecast')]
        update = dict_in['clouds'][(i, 'update')]

        # Plot points of forecast and update
        for j, cloud in enumerate((forecast, update)):
            axes = axeses[j, (i - args.start) % 2]
            for particle in cloud:
                plot_point(axes, particle.x, '#1f77b4')
            for boundary in bins:
                axes.plot((boundary,) * 2, (0, 50), color='black', linewidth=.5)
            axes.set_xlim(-22, 22)
            axes.set_ylim(0, 50)
        axes = axeses[0, (i - args.start) % 2]
        plot_point(axes, forecast[0].x, '#1f77b4', f'{i=}')
        axes.legend()
    axeses[0, 0].set_ylabel(r'$\rm{Forecast}$')
    axeses[1, 0].set_ylabel(r'$\rm{Update}$')
    axeses[1, 0].set_yticks([])

    #pyplot.show()
    figure.tight_layout()
    figure.savefig(args.fig_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
