"""stateplots.py Makes separate figures of point clouds for back cover

python stateplots.py  --data_dir ../../../build/derived_data/synthetic --base_name state --fig_path ./

Call with optional arguments: --data_dir, --base_name, --fig_path

    data_dir is the directory that has the state files

    base_name When it is "state" the data files are "state0", "state1"
    ,..., "state11".

    fig_path, eg, .  Where the figures get written

"""

import sys
import argparse
import os

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
        axis.set_axis_off()

    markersize = 3
    # Make separate figures for each decoded state.
    for state in range(0, 12):  # The last file is state11.
        fig, axes = pyplot.subplots(nrows=1, figsize=(4, 4))
        subplot(axes, state, markersize)
        fig_path = os.path.join(args.fig_path, f'state{state}.png')
        fig.savefig(fig_path)

    if args.show:
        pyplot.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:
