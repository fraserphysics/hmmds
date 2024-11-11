"""laser_figures.py: Make pdfs from data.  EG,

python laser_figures.py --LaserHist LaserHist LaserHist.pdf
                          FigType   DataPath  TargetPath
"""

import sys
import argparse
import pickle
import numpy

import plotscripts.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Make laser figures for book')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('--LaserLP5',
                        action='store_true',
                        help='250 simulated laser data points')
    parser.add_argument('--LaserLogLike',
                        action='store_true',
                        help='Dependence of log likelihood on s and b')
    parser.add_argument('--LaserStates',
                        action='store_true',
                        help='Phase portrait from particle EKF')
    parser.add_argument('--LaserForecast',
                        action='store_true',
                        help='400 simulated laser data points')
    parser.add_argument('--LaserHist',
                        action='store_true',
                        help='Histogram of first 600 data points')
    parser.add_argument('data', type=str, help='path to data')
    parser.add_argument('result', type=str, help='path to result')
    return parser.parse_args(argv)


Tasks = {}  # Keys are function names.  Values are functions.


def register(func):
    """Decorator that puts function in Tasks dictionary
    """
    Tasks[func.__name__] = func
    return func


@register
def LaserLP5(data, matplotlib, pyplot):
    """Plot real and simulated laser data over fit range.
    """
    fig = pyplot.figure(figsize=(7, 4))
    ax = fig.add_subplot()
    ax.plot(data['training_simulated_observations'], label='Simulation')
    ax.plot(data['training_laser_data'], label='Laser Data')
    ax.legend()
    return fig


@register
def LaserLogLike(data, matplotlib, pyplot):
    """s b log_likelihood"""
    s = data['s']
    b = data['b']
    log_like = data['log_likelihood']
    fig = pyplot.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1, projection='3d', azim=-13, elev=20)
    ax.set_xlabel('$b$')
    ax.set_ylabel('$s$')
    ax.set_zlabel(r'$\log(P(y_1^{250}|\theta))$')
    X, Y = numpy.meshgrid(b, s)
    ax.contour(X, Y, log_like, zdir='z', offset=log_like.min())
    surf = ax.plot_surface(X,
                           Y,
                           log_like,
                           rstride=1,
                           cstride=1,
                           linewidth=1,
                           edgecolors='k',
                           cmap=matplotlib.cm.winter)
    return fig


@register
def LaserStates(data, matplotlib, pyplot):
    """forward_means"""
    fig = pyplot.figure()
    ax = fig.add_subplot()
    x_t = data['forward_means']
    ax.plot(x_t[:, 0], x_t[:, 2])
    ax.plot(x_t[:, 0], x_t[:, 2], linestyle='', marker='.', color='black')
    return fig


@register
def LaserForecast(data, matplotlib, pyplot):
    """simulated_observations over testing range"""
    fig = pyplot.figure(figsize=(7, 4))
    ax = fig.add_subplot()
    t_start = data['t_start']
    t_stop = data['t_stop']
    times = numpy.arange(t_start, t_stop)
    ax.plot(times, data['forecast_observations'], label='Forecast')
    ax.plot(times, data['next_data'], label='Laser Data')
    ax.set_xlabel(r'$t$')
    ax.legend()
    return fig


@register
def LaserHist(data, matplotlib, pyplot):
    """count"""
    count = data['count']
    fig = pyplot.figure()
    ax = fig.add_subplot()
    x = numpy.arange(0, len(count))
    ax.bar(x[:100], count[:100])
    ax.set_yticks(numpy.arange(0, 25, 10))
    ax.set_ylabel('Counts')
    ax.set_xlabel('$x$')
    ax.set_xticks([0, 5, 50, 93, 100])
    fig.subplots_adjust(bottom=0.12)  # Make more space for label
    return fig


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    args, matplotlib, pyplot = plotscripts.utilities.import_and_parse(
        parse_args, argv)

    with open(args.data, 'rb') as file_:
        data = pickle.load(file_)

    for name, function in Tasks.items():
        if getattr(args, name):
            figure = function(data, matplotlib, pyplot)
            if args.show:
                pyplot.show()
            figure.savefig(args.result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
