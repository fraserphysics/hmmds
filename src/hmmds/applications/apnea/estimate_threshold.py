"""estimate_threshold.py Use purpose built hmm to estimate best
thresholds for records.

python estimate_threshold.py ../../../../build/derived_data/apnea/models/threshold --records a01 a02

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

    parser = argparse.ArgumentParser("Estimate best thresholds")
    hmmds.applications.apnea.utilities.common_arguments(parser)
    parser.add_argument("threshold_model",
                        type=str,
                        help="Path to model for calculating thresholds")
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    with open(args.threshold_model, 'rb') as _file:
        model = pickle.load(_file)
    print(f'{model.__str__()}')
    args.AR_order = model.y_mod['hr_respiration'].ar_order
    reader = hmmds.applications.apnea.model_init.read_lphr_respiration

    for record in args.records:
        y_data = [hmm.base.JointSegment(reader(args, record))]
        estimate = model.estimate_missing(y_data, 'threshold')
        print(
            f'{record} {estimate.mean():5.2f} {estimate.min()=:5.2f} {estimate.max()=:5.2f}'
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
