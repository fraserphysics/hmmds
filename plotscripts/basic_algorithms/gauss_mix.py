"""gauss_mix.py: Makes GaussMix.pdf

"""

import sys
import pickle
import argparse

import numpy

import plotscripts.utilities


def parse_args(argv=None):
    """ Convert command line arguments into a namespace
    """

    if not argv:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Make GaussMix.pdf')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('dict_file', type=str, help="path to data")
    parser.add_argument('fig_path', type=str, help="path to figure")
    return parser.parse_args(argv)


def main(argv=None):
    """Make Fig. 2.7.  Illustration of EM estimation for Gaussian Mixture
    model.

    """
    args = parse_args(argv)
    matplotlib, pyplot = plotscripts.utilities.import_matplotlib_pyplot(args)
    plotscripts.utilities.update_matplotlib_params(matplotlib)

    _dict = pickle.load(open(args.dict_file, 'rb'))

    def subplot(axis, i_label):
        x = numpy.arange(-6, 6, 0.05)

        def gauss(mean, var):
            difference = x - mean
            return (1 / (numpy.sqrt(2 * numpy.pi * var))) * numpy.exp(
                -difference * difference / (2 * var))

        def mix(alpha, means):
            return alpha * gauss(means[0], 1) + (1 - alpha) * gauss(means[1], 1)

        for i, label in i_label:
            y = mix(_dict['alpha'][i], _dict['mu_i'][i])
            axis.plot(x, y, label=label)
        axis.legend()

    fig, axes = pyplot.subplots(2, 1, figsize=(6, 5))
    subplot(axes[0], ((0, r'$\theta(1)$'), (-1, r'$\theta$')))
    subplot(axes[1], ((1, r'$\theta(2)$'),))
    x = _dict['Y']
    axes[1].plot(x, numpy.ones(len(x)) * 0.01, 'rd')

    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)  #Make sure to save it as a .pdf
    return 0


if __name__ == "__main__":
    sys.exit(main())
# Local Variables:
# mode: python
# End:
