"""interpolate.py Plot quadratic fit to 3 points

interpolate.py --abcd a b c d

"""
import sys
import argparse

import numpy

import plotscripts.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(
        description='Plot quadratic fit to 3 points')
    parser.add_argument('--show',
                        action='store_false',
                        help="display figure using Qt5")
    parser.add_argument('--abcd',
                        nargs=4,
                        type=float,
                        default=(1.0, 3.0, 2.0, 0.5),
                        help="variables")
    return parser.parse_args(argv)


def alpha_beta_gamma(a, b, c, d):
    return (a + c - 2 * b) / (2 * d * d), (c - a) / (2 * d), b


def y(a, b, c, d, x):
    alpha, beta, gamma = alpha_beta_gamma(a, b, c, d)
    return x * x * alpha + x * beta + gamma


def dydx(a, b, c, d, x):
    alpha, beta, gamma = alpha_beta_gamma(a, b, c, d)
    return 2 * x * alpha + beta


def peak(a, b, c, d):
    x = d * (a - c) / (2 * (a + c - 2 * b))
    y_x = y(a, b, c, d, x)
    return x, y_x


def main(argv=None):
    """Display the data points, a quadratic interpolation, its
    derivative, and the peak.

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    a, b, c, d = args.abcd
    print(f"{a=}, {b=}, {c=}, {d=}")
    x = numpy.linspace(-1.1, 1.1, 50) * d
    y_values = y(a, b, c, d, x)
    dy_values = dydx(a, b, c, d, x)
    peak_x, peak_y = peak(a, b, c, d)

    fig, (ax_y, ax_d) = pyplot.subplots(nrows=2, figsize=(6, 8))
    ax_y.sharex(ax_d)
    ax_y.plot(x, y_values)
    ax_y.plot([-d, 0, d], [a, b, c], marker="x", linestyle="")
    ax_y.plot(peak_x, peak_y, marker="*")
    ax_d.plot(x, dy_values)
    pyplot.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
