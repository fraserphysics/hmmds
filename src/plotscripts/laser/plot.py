"""plot.py: Illustrate performance of filter.
"""

import sys
import argparse
import pickle

import plotscripts.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Explore 1-d map')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('data', type=str, help='path to data file')
    parser.add_argument('fig_path', type=str, help='path to figure')
    return parser.parse_args(argv)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    data = pickle.load(open(args.data, 'rb'))

    fig, ((filtered, filtered_0, observations),
          (filtered_short, filtered_0_short,
           observations_short)) = pyplot.subplots(nrows=2,
                                                  ncols=3,
                                                  figsize=(10, 10))
    fig.text(0.05, 0.02, f'cross entropy: {data["cross_entropy"]}')

    filtered.plot(data['forward_means'][:, 0], data['forward_means'][:, 2])
    filtered.set_title(r"Filtered $x_2$ vs $x_0$")
    filtered_0.plot(data['forward_means'][:, 0])
    filtered_0.set_title(r"Filtered $x_0$ vs $n$")
    observations.plot(data['observations'])
    observations.set_title(r"Laser Data vs $n$")

    # Set start and stop to illustrate a complete orbit
    start = 252
    stop = 427
    x = list(range(start, stop))

    filtered_short.plot(data['forward_means'][start:stop, 0],
                        data['forward_means'][start:stop, 2])
    filtered_short.set_title(r"Subset Filtered $x_2$ vs $x_0$")
    filtered_0_short.plot(x, data['forward_means'][start:stop, 0])
    filtered_0_short.set_title(r"Subset Filtered $x_0$ vs $n$")
    observations_short.plot(x, data['observations'][start:stop])
    observations_short.set_title(r"Subset Laser Data vs $n$")
    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
