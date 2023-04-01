"""apnea_train.py reads an initial model trains it and writes the result

From the Makefile:


${MODELS}/model_%:
        python apnea_train.py --root ${ROOT} $* ${MODELS}/initial_$* $@

$* is the pattern %
$@ is the target

Training data for A2 is a01 -- a20, for Low, Medium, High determined by the pass 1 report

"""
import sys
import glob
import os.path
import pickle
import argparse

import numpy.random

import hmmds.applications.apnea.utilities


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Read initial model, train, write result")
    hmmds.applications.apnea.utilities.common_arguments(parser)
    parser.add_argument('model_name',
                        type=str,
                        help="eg, A2 or Low.  Determines training data")
    parser.add_argument('initial_path', type=str, help="path to initial model")
    parser.add_argument('write_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


def make_data_level(common, level):
    """ Make data for training with expert information

    Args:
        common: Information for apnea work
        level: One of High, Medium, Low

    Returns:
        list of hmm.base.JointSegment instances

    """

    with open(common.pass1 + '.pickle', 'rb') as _file:
        pass1 = pickle.load(_file)

    return_list = []
    for record in pass1:
        if not record.level == level:
            continue
        name = record.name
        if name[0] == 'x':
            continue

        # Was heart_rate_respiration_bundle_data
        # return_list.append(
        #     hmmds.applications.apnea.utilities.fixme
        #    fixme(name, common))
    return return_list


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)
    rng = numpy.random.default_rng()

    if args.model_name in 'A2 A3 A4'.split():
        y_data = hmmds.applications.apnea.utilities.list_heart_rate_respiration_data(
            args.a_names, args)

    elif args.model_name == 'outlier':
        y_data = hmmds.applications.apnea.utilities.list_heart_rate_respiration_data(
            args.all_names, args)

    elif args.model_name in 'C2 C3 C4'.split():
        y_data = hmmds.applications.apnea.utilities.list_heart_rate_respiration_data(
            args.c_names, args)

    elif args.model_name in 'Low Medium High'.split():
        y_data = make_data_level(args, args.model_name)

    else:
        raise RuntimeError('Unknown model_name: {0}'.format(args.model_name))

    with open(args.initial_path, 'rb') as _file:
        _, model = pickle.load(_file)

    model.multi_train(y_data, args.iterations)

    with open(args.write_path, 'wb') as _file:
        pickle.dump(model, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
