"""plot.py: Plots for debugging.

python plot.py input_path

"""

import sys
import argparse
import pickle

import numpy
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as pyplot


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Debuggin plot')
    parser.add_argument('input', type=str, help='Path to data')
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    """Plot some stuff
    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    with open(args.input, 'rb') as file_:
        dict_in = pickle.load(file_)
    for key, value in dict_in.items():
        if isinstance(value, numpy.ndarray):
            print(f'{key} {value.shape}')
        else:
            print(f'{key} {type(value)} {len(value)=}')
    figure, axeses = pyplot.subplots(nrows=6, ncols=2)
    x = dict_in['initial_positions']
    x_all = dict_in['x_all']
    y_q = dict_in['y_q']
    gamma = dict_in['gamma']
    axeses[0, 0].plot(gamma)
    axeses[0, 1].plot(y_q)
    for i in range(20, 25):
        x = dict_in['clouds'][2 * i]
        if len(x.shape) != 2:
            continue
        axes = axeses[i % 5 + 1, 0]
        axes.plot(x[:, 0],
                  x[:, 2],
                  markeredgecolor='none',
                  marker='.',
                  markersize=5,
                  linestyle='None')
        axes.set_xlim(-20, 20)
        axes.set_ylim(0, 50)

        x = dict_in['clouds'][2 * i + 1]
        if len(x.shape) != 2:
            continue
        axes = axeses[i % 5 + 1, 1]
        axes.plot(x[:, 0],
                  x[:, 2],
                  markeredgecolor='none',
                  marker='.',
                  markersize=5,
                  linestyle='None',
                  label=f'y[{i}]={y_q[i]}')
        axes.set_xlim(-20, 20)
        axes.set_ylim(0, 50)
        axes.legend()
    pyplot.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
