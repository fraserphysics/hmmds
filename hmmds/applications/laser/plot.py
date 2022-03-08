"""plot.py A temporary script for debugging
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
                        action='store_false',
                        help="display figure using Qt5")
    parser.add_argument('--data',
                        type=str,
                        default='test_ekf',
                        help='path to data file')
    parser.add_argument('--fig_path',
                        type=str,
                        default='fig_ekf.pdf',
                        help='path to figure')
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
                                                  figsize=(6, 10))

    filtered.plot(data['forward_means'][:, 0], data['forward_means'][:, 2])
    filtered_0.plot(data['forward_means'][:, 0])
    observations.plot(data['observations'])

    # Set start and stop to illustrate a complete orbit
    start = 78
    stop = 253

    filtered_short.plot(data['forward_means'][start:stop, 0],
                        data['forward_means'][start:stop, 2])
    filtered_0_short.plot(data['forward_means'][start:stop, 0])
    observations_short.plot(data['observations'][start:stop])
    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
