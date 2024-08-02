"""model_viz.py Read a model file and write a graphviz representation

Method HMM.viz does the work
"""

import argparse
import pickle
import sys

import pygraphviz


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Create and write a vizualization")
    parser.add_argument('model_path', type=str)
    parser.add_argument('write_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    """Create an hmm and write it as a pickle.
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    with open(args.model_path, 'rb') as _file:
        model = pickle.load(_file)

    if args.write_path[-4:] == '.pdf':
        model.viz(pdf_path=args.write_path)

    if args.write_path[-4:] == '.dot':
        model.viz(dot_path=args.write_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
