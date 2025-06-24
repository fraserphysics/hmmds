"""plot_clouds.py: Plots for debugging.

python plot_clouds.py --start 50 r_threshold/0.003

"""

import sys
import argparse
import pickle
import os

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
    parser.add_argument('--plot_lims',
                        action='store_true',
                        help="Plot clouds in standard frames")
    parser.add_argument('--start',
                        type=int,
                        default=0,
                        help='Plot particles at 5 times starting here')
    parser.add_argument('--save',
                        type=str,
                        help='path to result.  Show if not set')
    parser.add_argument('input', type=str, help='Path to data directory')
    args = parser.parse_args(argv)
    return args


def plot_box(axes, x_box):
    """Plot the box for a particle
    """
    colors = 'red green blue'.split()
    x = x_box[:3]
    box = x_box[3:].reshape((3, 3))

    def plot_line(i):
        end = x + box[i]
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

    def __init__(self, true_x, states_boxes):
        assert true_x.shape == (3,)
        assert states_boxes.shape[1] == 12
        n_particles = states_boxes.shape[0]

        self.true_x = true_x
        self.states_boxes = states_boxes
        self.distances = numpy.asarray([
            lengths.max() for lengths in numpy.abs(states_boxes[:, :3] - true_x)
        ])
        self.indices = numpy.argsort(self.distances)

        assert self.distances.shape == (n_particles,)
        assert self.indices.shape == (n_particles,)

    def nth(self, n):
        """Return the distance and particle that are nth closest

        """
        index = self.indices[n]
        return self.distances[index], self.states_boxes[index]

    def delta_svd(self, n):
        """Calculate (true_x - nth closest) in svd coordinates
        """
        x_box = self.states_boxes[self.indices[n]]
        U, S, VT = numpy.linalg.svd(x_box[3:].reshape((3, 3)))
        delta = self.true_x - x_box[:3]
        return numpy.dot((U / S).T, delta)


def read_particles(path, n_start, length):
    """Read states_boxes arrays for forcast and update clouds of particles
    """
    result = {}

    with open(path, 'rb') as file_:
        for n in range(n_start * 2):
            numpy.load(file_)
        for n in range(n_start, n_start + length):
            result[(n, 'forecast')] = numpy.load(file_)
            try:
                result[(n, 'update')] = numpy.load(file_)
            except EOFError:
                result[(n, 'update')] = []
                break
    return result


def main(argv=None):
    """Plot some stuff
    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    npy_path = os.path.join(args.input, 'states_boxes.npy')
    dict_path = os.path.join(args.input, 'dict.pkl')
    with open(dict_path, 'rb') as file_:
        dict_in = pickle.load(file_)
    y_q = dict_in['y_q']
    x_all = dict_in['x_all']
    gamma = dict_in['gamma']
    offset = 50  # First resample from 200,000 to 20,000 with some
    # nice parameters
    log_gamma = numpy.log(gamma)[offset:]
    cum_sum = numpy.cumsum(log_gamma)
    h_hat = -cum_sum / numpy.arange(1, len(cum_sum) + 1) / 0.15
    print(f'{h_hat[-10:]=}\n args:')
    for key, value in dict_in['args'].__dict__.items():
        print(f'    {key}: {value}')

    figure, axeses = pyplot.subplots(nrows=5, ncols=3, figsize=(8, 9))

    clouds = read_particles(npy_path, args.start, 5)
    for i in range(args.start, args.start + 5):
        if (i, 'forecast') not in clouds:
            break
        forecast = clouds[(i, 'forecast')]
        update = clouds[(i, 'update')]

        close = Close(x_all[i], forecast)
        distance, closest = close.nth(0)
        axes = axeses[i % 5, 2]  # Plot the box for the particle
        plot_box(axes, closest)  # closest to the true trajectory

        # Plot points of forecast and update
        for j, states_boxes in enumerate((forecast, update)):
            # j = 0 -> forecast, j = 1 -> update
            if len(states_boxes) == 0:
                continue
            axes = axeses[i % 5, j]
            # Mark the "true" state with an x
            axes.plot(x_all[i, 0], x_all[i, 2], marker='x')
            x_0s = states_boxes[:, 0]
            x_2s = states_boxes[:, 2]
            axes.plot(x_0s,
                      x_2s,
                      markeredgecolor='none',
                      marker='.',
                      markersize=5,
                      linestyle='None',
                      color='blue')
            # Plot boxes
            if len(states_boxes) > 1:
                _, next_closest = close.nth(1)
                plot_box(axes, next_closest)
            distance, closest = close.nth(0)
            plot_box(axes, closest)

            if args.plot_lims:
                axes.set_xlim(-22, 22)
                axes.set_ylim(0, 50)
        # Print information in legend of leftmost plot
        axes = axeses[i % 5, 0]
        plot_point(axes,
                   closest,
                   'red',
                   label=f'y[{i}]={y_q[i]} {distance=:.3f} n={len(forecast)} ')
        axes.legend(loc='upper right')  # Faster than loc="best"
        # Print p(y[t]|y[0:t]) in legend of center plot
        axes = axeses[i % 5, 1]
        plot_point(axes,
                   closest,
                   'red',
                   label=f'p={len(update)/len(forecast):.3f} ')
        axes.legend(loc='upper right')  # Faster than loc="best"

        numpy.set_printoptions(precision=2)
        print(f'n[{i}]={len(forecast)} {distance=:.2f} true x = {x_all[i]}')
        numpy.set_printoptions(precision=8)

    # Print box for closest
    if args.print_box:
        print(f'box=\n{closest.box}')
    if args.save:
        figure.savefig(args.save)
    else:
        pyplot.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
