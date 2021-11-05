""" TSintro.py <fine> <coarse> <quantized> <plot_file>
"""
DEBUG = False
import sys

import numpy

import plotscripts.utilities


def main(argv=None):
    """Make time series picture with fine, coarse, and quantized Lorenz
    data.

    """
    global DEBUG

    import matplotlib  # pylint: disable=import-outside-toplevel

    if DEBUG:
        matplotlib.rcParams['text.usetex'] = False
    else:
        matplotlib.use('PDF')
    import matplotlib.pyplot  # pylint: disable=import-outside-toplevel
    if argv is None:  # Usual case
        argv = sys.argv[1:]
    name_fine, name_coarse, name_quantized, plot_file = argv

    def read_data(name):
        with open(name, 'r') as file:
            return numpy.array([
                [float(x) for x in line.split()] for line in file.readlines()
            ]).T

    fine = read_data(name_fine)
    coarse = read_data(name_coarse)
    quantized = read_data(name_quantized)

    params = {
        'axes.labelsize': 12,
        #'text.fontsize': 10,
        'legend.fontsize': 10,
        'text.usetex': True,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11
    }
    matplotlib.rcParams.update(params)
    fig = matplotlib.pyplot.figure(figsize=(6, 4))
    X = plotscripts.utilities.axis(data=fine[0],
                                   magnitude=False,
                                   label=r'$\tau$')
    Y = plotscripts.utilities.axis(data=fine[1],
                                   magnitude=False,
                                   ticks=numpy.arange(-10, 10.1, 10),
                                   label=r'$x_1(\tau)$')
    ax = plotscripts.utilities.SubPlot(fig, (2, 1, 1), X, Y, color='b')
    ax.plot(coarse[0], coarse[1], 'ro')
    ax.set_ylim(-17, 17)
    ax.set_xlim(0, 6)

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(quantized[0], quantized[1], 'kd')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$y(t)$')
    ax.set_ylim(0.5, 4.5)
    ax.set_yticks(numpy.arange(1, 4.1, 1))
    ax.set_xticks(numpy.arange(0, 40.1, 10))
    fig.subplots_adjust(hspace=0.3)

    if DEBUG:
        matplotlib.pyplot.show()
    else:
        fig.savefig(plot_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
