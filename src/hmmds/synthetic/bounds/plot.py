"""plot.py: Plots for debugging.

python plot.py --start 50 input_path

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

    parser = argparse.ArgumentParser(description='Debugging plot')
    parser.add_argument('--printR',
                        action='store_true',
                        help="Print R matrices")
    parser.add_argument('--start',
                        type=int,
                        default=0,
                        help='Plot particles at 5 times starting here')
    parser.add_argument('--parent',
                        type=int,
                        default=-1,
                        help='Track descendants of parent')
    parser.add_argument('input', type=str, help='Path to data')
    args = parser.parse_args(argv)
    return args


def plot_box(axes, particle):
    colors = 'red green blue'.split()
    x = particle.x
    neighbor = x + particle.neighbor

    def plot_line(i):
        end = x + numpy.dot(particle.Q, particle.R[:, i])
        axes.plot((x[0], end[0]), (x[2], end[2]), color=colors[i])

    for i in range(3):
        plot_line(i)

    axes.plot((x[0], neighbor[0]), (x[2], neighbor[2]), color='black', linestyle='dotted')
    axes.plot(x[0],
              x[2],
              markeredgecolor='none',
              marker='.',
              markersize=8,
              linestyle='None')


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

    figure, axeses = pyplot.subplots(nrows=5, ncols=3, figsize=(7, 9))
    y_q = dict_in['y_q']
    colors = list(color_dict.values())
    for i in range(args.start, args.start + 5):
        if (i, 'forecast') not in dict_in['clouds']:
            continue
        forecast = dict_in['clouds'][(i, 'forecast')]
        update = dict_in['clouds'][(i, 'update')]
        axes = axeses[i % 5, 2]
        for particle in update:
            if particle.parent == args.parent or particle.parent == -1:
                plot_box(axes, particle)
        axes.set_xticks([])
        axes.set_yticks([])
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
    # Print QRs for updates in last cloud
    for particle in update:
        if args.printR:
            qr = numpy.matmul(particle.Q, particle.R)
            print(f'QR=\n{qr}')
            print(f'neighbor={particle.neighbor.T}')
    pyplot.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
