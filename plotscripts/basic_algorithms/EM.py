""" EM.py: Creates Fig. 2.8 of the book

python EM.py outfile.pdf
"""
Debug = False
import sys
import numpy
import matplotlib

matplotlib.use('PDF')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for  "projection='3d'".


def main(argv=None):
    """Make plots of auxiliary function Q and consequent 1-d map for EM
    algorithm.

    """
    global Debug
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    fig_name = argv[0]
    if fig_name == 'debug':
        Debug = True

    params = {
        'axes.labelsize': 12,
        #'text.fontsize': 10,
        'legend.fontsize': 10,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'text.usetex': True,
    }
    if Debug:
        params['text.usetex'] = False
    matplotlib.rcParams.update(params)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 2, 1, projection='3d', azim=-109, elev=30)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r"$\theta'$")
    ax.set_zlabel(r"$Q(\theta',\theta)$")
    xs = numpy.arange(0.1, 0.9, 0.05)
    ys = numpy.arange(0.2, 0.8, 0.05)
    n_x = len(xs)
    n_y = len(ys)
    zs = numpy.empty((n_x, n_y)).T
    for i in range(n_x):
        x = xs[i]
        for j in range(n_y):
            y = ys[j]
            zs[j, i] = (1 + 2 * x) * numpy.log(y) + (1 + 2 *
                                                     (1 - x)) * numpy.log(1 - y)
    ax.set_xticks(numpy.arange(0.2, 0.8, .2))
    ax.set_yticks(numpy.arange(0.3, 0.8, .2))
    X, Y = numpy.meshgrid(xs, ys)
    ax.plot_surface(X,
                    Y,
                    zs,
                    rstride=1,
                    cstride=1,
                    cmap=matplotlib.cm.hsv,  # pylint: disable=no-member
                    linewidth=1)
    ax = fig.add_subplot(1, 2, 2)
    x = numpy.arange(0, 1.1, 1)
    y = 0.25 + x / 2.0
    ax.plot(x, x, label='slope 1 referece')
    ax.plot(x, y, label=r'$\cal{T}(\theta)$')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r"$\cal{T}(\theta)$")
    ax.legend(loc='lower right')
    ticks = numpy.arange(0, 1.1, 0.25)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if Debug:
        plt.show()
    else:
        fig.savefig(fig_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
# Local Variables:
# mode: python
# End:
