"""The script that makes the cover figure.

Call with optional arguments: --data_dir, --base_name, --fig_path

    data_dir is the directory that has the state files

    base_name When it is "state" the data files are "state0", "state1"
    ,..., "state11".

    fig_path, eg, figs/Statesintro.pdf.  Where the figure gets written

"""

import sys
import argparse

import plotscripts.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

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


def main(argv=None):
    """Make the cover figure.
    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

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

    fig = pyplot.figure(figsize=(15, 15))
    n_subplot = 0
    skiplist = [1, 2, 5, 6]  # Positions for the combined plot

    def subplot(axis, state, markersize):
        """Plot points decoded as state.
        """
        name = f'{args.data_dir}/{args.base_name}{state}'
        x_list = []
        z_list = []
        with open(name, 'r', encoding='utf-8') as data_file:
            for line in data_file.readlines():
                x, _, z = [float(w) for w in line.split()]
                x_list.append(x)
                z_list.append(z)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.plot(x_list,
                  z_list,
                  color=plotcolor[state % 7],
                  markeredgecolor='none',
                  marker='.',
                  markersize=markersize,
                  linestyle='None')
        axis.set_xlim(-20, 20)
        axis.set_ylim(0, 50)

    markersize = 3
    # Make separate plots for each decoded state.
    for state in range(0, 12):  # The last file is state11.
        n_subplot += 1
        while n_subplot in skiplist:  #This is to make space for putting in the
            n_subplot += 1  #figure with all the assembled pieces.
        # There are 2 kinds of calls to fig.add_subplot; one here
        # that's 4x4 and one before the next loop that's 2x2
        subplot(fig.add_subplot(4, 4, n_subplot), state, markersize)

    # Make a single plot in which colors identify the different states.
    axis_all = fig.add_subplot(2, 2, 1)
    for state in range(0, 12):
        subplot(axis_all, state, markersize)

    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:
