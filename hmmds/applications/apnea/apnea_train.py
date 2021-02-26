"""apnea_train.py reads an initial model trains it and writes the result

From the Makefile:


${MODELS}/model_%:
        python apnea_train.py ${COMMON_ARGS} $* ${MODELS}/initial_$* $@

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

import utilities


def make_data_a2(args):
    """Prepare training data for A2 from all a records

    Args:
        args:

    Returns:
        A list of ducts

    use glob.glob to get list of record names and
    utilities.heart_rate_respiration_data to construct the list

    """

    return_list = []
    for hr_path in glob.glob(args.heart_rate + '/a*'):
        r_path = '{0}/{1}'.format(args.respiration, os.path.basename(hr_path))
        return_list.append(
            utilities.heart_rate_respiration_data(hr_path, r_path))
    return return_list


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser("Read initial model, train, write result")
    utilities.common_args(parser)
    parser.add_argument('model_name',
                        type=str,
                        help="eg, A2 or Low.  Determines training data")
    parser.add_argument('initial_path', type=str, help="path to initial model")
    parser.add_argument('write_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    args.rng = numpy.random.default_rng()

    if args.model_name == 'A2':
        y_data = utilities.pattern_heart_rate_respiration_data(args, ['a'])
    else:
        raise RuntimeError('Unknown model_name: {0}'.format(args.model_name))

    with open(args.initial_path, 'rb') as _file:
        model = pickle.load(_file)

    model.multi_train(y_data, args.iterations)

    with open(args.write_path, 'wb') as _file:
        pickle.dump(model, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
