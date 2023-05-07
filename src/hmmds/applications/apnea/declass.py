"""declass.py: Read hmm with output model with classes and write hmm
with only the underlying output model

"""
import sys
import argparse
import pickle


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Remove classes from output model")
    parser.add_argument('input', type=str, help='path to initial model')
    parser.add_argument('output', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    with open(args.input, 'rb') as _file:
        old_args, old_hmm = pickle.load(_file)

    del old_hmm.y_mod['class']

    with open(args.output, 'wb') as _file:
        pickle.dump((old_args, old_hmm), _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
