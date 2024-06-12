"""apnea_train.py reads an initial model trains it and writes the result

"""
import sys
import glob
import os.path
import pickle
import argparse

import numpy.random

import hmmds.applications.apnea.utilities
import hmmds.applications.apnea.model_init
import hmm.base


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Read initial model, train, write result")
    hmmds.applications.apnea.utilities.common_arguments(parser)
    parser.add_argument('--type',
                        type=str,
                        default="masked",
                        help='Type of data, eg, "masked" or "unmasked"')
    parser.add_argument('initial_path', type=str, help="path to initial model")
    parser.add_argument('write_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    with open(args.initial_path, 'rb') as _file:
        model = pickle.load(_file)

    if 'class' in model.y_mod:
        reader = model.read_y_with_class
    else:
        reader = model.read_y_no_class
    y_data = list(
        # model.read_y_with_class calls
        # self.args.read_y_class(self.args, record_name)
        hmm.base.JointSegment(reader(record)) for record in args.records)

    numpy.seterr(divide='raise', invalid='raise')
    model.multi_train(y_data, args.iterations)

    model.strip()
    with open(args.write_path, 'wb') as _file:
        pickle.dump(model, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
