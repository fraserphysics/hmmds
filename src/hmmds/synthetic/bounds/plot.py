"""plot.py: Plots for debugging.

python plot.py input_path

"""

import sys
import argparse
import pickle

import numpy
import matplotlib
from matplotlib.colors import TABLEAU_COLORS as color_dict

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as pyplot


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Debuggin plot')
    parser.add_argument('--start',
                        type=int,
                        default=0,
                        help='Plot particles at 5 times starting here')
    parser.add_argument('input', type=str, help='Path to data')
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

    figure, axeses = pyplot.subplots(nrows=5,
                                     ncols=2,
                                     sharex=True,
                                     sharey=True,
                                     figsize=(5, 9))
    y_q = dict_in['y_q']
    colors = list(color_dict.values())
    for i in range(args.start, args.start + 5):
        if (i, 'forecast') not in dict_in['clouds']:
            continue
        forecast = dict_in['clouds'][(i, 'forecast')]
        update = dict_in['clouds'][(i, 'update')]
        for j, cloud in enumerate((forecast, update)):
            if len(cloud) == 0:
                continue
            axes = axeses[i % 5, j]
            for particle in cloud:
                color = colors[particle.parent % len(colors)]
                plot_point(axes, particle.x, color)
            if j == 0:
                plot_point(axes,
                           cloud[0].x,
                           color,
                           label=f'n={len(cloud)} y[{i}]={y_q[i]}')
                axes.legend(loc='upper right')
            axes.set_xlim(-22, 22)
            axes.set_ylim(0, 50)
    pyplot.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
