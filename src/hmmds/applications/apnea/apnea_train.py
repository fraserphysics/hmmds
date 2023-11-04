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
from hmmds.applications.apnea.utilities import State


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

    y_data = list(
        hmm.base.JointSegment(model.read_y_with_class(record))
        for record in args.records)

    model.multi_train(y_data, args.iterations)
    #hmmds.applications.apnea.utilities.print_chain_model(
    #    model.y_mod, model.alpha.sum(axis=0), model.args.state_key2state_index)

    model.strip()
    with open(args.write_path, 'wb') as _file:
        pickle.dump(model, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
