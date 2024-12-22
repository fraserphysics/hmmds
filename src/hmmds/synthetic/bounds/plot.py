"""plot.py: Plots for debugging.

python plot.py --start 50 input_path

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
    parser.add_argument('--print_box', action='store_true', help="Print boxes")
    parser.add_argument('--start',
                        type=int,
                        default=0,
                        help='Plot particles at 5 times starting here')
    parser.add_argument('input', type=str, help='Path to data')
    args = parser.parse_args(argv)
    return args


def plot_box(axes, particle):
    colors = 'red green blue'.split()
    x = particle.x

    def plot_line(i):
        end = x + particle.box[i]
        axes.plot((x[0], end[0]), (x[2], end[2]), color=colors[i])

    for i in range(3):
        plot_line(i)

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


class Close:

    def __init__(self, true_x, particles):
        self.true_x = true_x
        self.particles = particles
        self.distances = numpy.asarray(
            [self.distance(particle) for particle in particles])
        self.indices = numpy.argsort(self.distances)

    def distance(self, particle):
        """Max norm
        """
        delta = particle.x - self.true_x
        return numpy.abs(delta).max()

    def nth(self, n):
        """Return the distance and particle that are nth closest

        """
        index = self.indices[n]
        return self.distances[index], self.particles[index]

    def delta_svd(self, n):
        """Calculate (true_x - nth closest) in svd coordinates
        """
        particle = self.particles[self.indices[n]]
        U, S, VT = numpy.linalg.svd(particle.box)
        delta = self.true_x - particle.x
        return numpy.dot((U / S).T, delta)


def main(argv=None):
    """Plot some stuff
    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    with open(args.input, 'rb') as file_:
        dict_in = pickle.load(file_)
    y_q = dict_in['y_q']
    x_all = dict_in['x_all']
    clouds = dict_in['clouds']

    figure, axeses = pyplot.subplots(nrows=5, ncols=3, figsize=(8, 9))

    for i in range(args.start, args.start + 5):
        if (i, 'forecast') not in clouds:
            continue
        forecast = clouds[(i, 'forecast')]
        if (i, 'update') in dict_in['clouds']:
            update = dict_in['clouds'][(i, 'update')]
        else:
            update = []

        close = Close(x_all[i], forecast)
        distance, closest = close.nth(0)
        axes = axeses[i % 5, 2]
        plot_box(axes, closest)

        # Plot points of forecast and update
        for j, cloud in enumerate((forecast, update)):
            if len(cloud) == 0:
                continue
            axes = axeses[i % 5, j]
            # Mark the "true" state with an x
            axes.plot(x_all[i, 0], x_all[i, 2], marker='x')
            for particle in cloud:
                plot_point(axes, particle.x, 'blue')

                # Plot boxes
                if len(cloud) > 1:
                    _, next_closest = close.nth(1)
                    plot_box(axes, next_closest)
                distance, closest = close.nth(0)
                plot_box(axes, closest)

            axes.set_xlim(-22, 22)
            axes.set_ylim(0, 50)
        # Print information in legend of leftmost plot
        axes = axeses[i % 5, 0]
        plot_point(axes,
                   closest.x,
                   'red',
                   label=f'y[{i}]={y_q[i]} {distance=:.3f} n={len(forecast)} ')
        axes.legend(loc='upper right')

        numpy.set_printoptions(precision=2)
        print(f'n[{i}]={len(forecast)} {distance=:.2f} true x = {x_all[i]}')
        numpy.set_printoptions(precision=8)

    # Print box for closest
    if args.print_box:
        print(f'box=\n{closest.box}')
    pyplot.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
