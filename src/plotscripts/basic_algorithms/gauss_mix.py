"""gauss_mix.py: Makes GaussMix.pdf

"""

import sys
import pickle
import argparse

import numpy

import plotscripts.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Make GaussMix.pdf')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('dict_file', type=str, help="path to data")
    parser.add_argument('fig_path', type=str, help="path to figure")
    return parser.parse_args(argv)


def main(argv=None):
    """Make fig:GaussMix  Illustration of EM estimation for Gaussian Mixture
    model.

    """

    args, matplotlib, pyplot = plotscripts.utilities.import_and_parse(
        parse_args, argv)

    _dict = pickle.load(open(args.dict_file, 'rb'))
    assert set(_dict.keys()) == set(('Y', 'alpha', 'mu_i'))

    # Y is a set of 10 observations.  alpha and mu_i are sequences of
    # model parameters.

    def subplot(axis, i_labels):
        """Plot the distributions for means[i], alpha[i].
        
        Args:
            axis: An axis from pyplot.subplots
            i_labels: A tuple of pairs (i, label)

        """
        x = numpy.arange(-6, 6, 0.05)

        def gauss(mean, var):
            """Return y values for Normal(mean, var) over range of x."""
            difference = x - mean
            return (1 / (numpy.sqrt(2 * numpy.pi * var))) * numpy.exp(
                -difference * difference / (2 * var))

        def mix(alpha, means):
            _var = 1.0
            return alpha * gauss(means[0], _var) + (1 - alpha) * gauss(
                means[1], _var)

        for i, label in i_labels:
            y = mix(_dict['alpha'][i], _dict['mu_i'][i])
            axis.plot(x, y, label=label)
        axis.legend()

    fig, axes = pyplot.subplots(2, 1, figsize=(6, 5))

    # Plot the distributions for the initial and true parameters
    subplot(axes[0], (
        (0, r'$\theta(1)$'),  # Initial parameters
        (-1, r'$\theta$')  # True parameters
    ))

    # Plot the distribution for parameters after one and two EM
    # iterations
    subplot(axes[1], (
        (1, r'$\theta(2)$'),  # Parameters after one iteration
        (2, r'$\theta(3)$'),  # Parameters after two iterations
    ))

    # Plot the observations
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
