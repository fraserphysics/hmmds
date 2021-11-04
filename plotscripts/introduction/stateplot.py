"""The script that makes the cover figure.

Call with optional arguments: --data_dir, --base_name, --fig_path

    data_dir is the directory that has the state files

    base_name When it is "state" the data files are "state0", "state1"
    ,..., "state11".

    fig_path, eg, figs/Statesintro.pdf.  Where the figure gets written

"""

import sys
import argparse


def parse_args(argv=None):
    """ Convert command line arguments into a namespace
    """

    if not argv:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Make plot for cover of book')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('--data_dir',
                        type=str,
                        default="derived_data/synthetic")
    parser.add_argument('--base_name', type=str, default="state")
    parser.add_argument('--fig_path', type=str, default="figs/Statesintro.pdf")
    return parser.parse_args(argv)


def main():
    '''
    '''

    import matplotlib

    args = parse_args()

    if args.show:
        matplotlib.use('Qt5Agg')
    else:
        matplotlib.use('PDF')  # Permits absence of enviroment variable DISPLAY
    import matplotlib.pyplot  # Must be after matplotlib.use

    # Colors for the states
    plotcolor = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [0, 1, 1],  # Cyan
        [1, 0, 1],  # Magenta
        [.95, .95, 0],  # Yellow
        [0, 0, 0]  # Black
    ]

    fig = matplotlib.pyplot.figure(figsize=(15, 15))
    n_subplot = 0
    skiplist = [1, 2, 5, 6]  # Positions for the combined plot

    # The first loop is to graph each individual set of points, the
    # second is to get all of them at once.
    def subplot(ax, n_state, markersize):
        name = '{0}/{1}{2}'.format(args.data_dir, args.base_name, n_state)
        xlist = []
        ylist = []
        zlist = []
        for line in open(name, 'r').readlines():  #Read the data file
            x, y, z = [float(w) for w in line.split()]
            xlist.append(x)
            ylist.append(y)
            zlist.append(z)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(xlist,
                zlist,
                color=plotcolor[n_state % 7],
                marker=',',
                markersize=markersize,
                linestyle='None')
        ax.set_xlim(-20, 20)
        ax.set_ylim(0, 50)

    for n_state in range(0, 12):  # The last file is state11.
        n_subplot += 1
        while n_subplot in skiplist:  #This is to make space for putting in the
            n_subplot += 1  #figure with all the assembled pieces.

        ax = fig.add_subplot(4, 4, n_subplot)
        # There are 2 kinds of calls to fig.add_subplot; one here
        # that's 4x4 and one before the next loop that's 2x2
        subplot(ax, n_state, 1)

    ax = fig.add_subplot(2, 2, 1)
    for n_state in range(0, 12):
        subplot(ax, n_state, 2)
    if args.show:
        matplotlib.pyplot.show()
    fig.savefig(args.fig_path)  #Make sure to save it as a .pdf
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:
