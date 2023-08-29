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
    parser.add_argument(
        '--AR_order',
        type=int,
        default=-1,
        help='Change AR order after estimating state probabilities')
    parser.add_argument('initial_path', type=str, help="path to initial model")
    parser.add_argument('write_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


TYPES = {}  # Is populated by @register decorated functions.  The keys
# are function names, and the values are functions for reading data.


def register(func):
    """Decorator that puts function in TYPES dictionary"""
    #See https://realpython.com/primer-on-python-decorators/
    TYPES[func.__name__] = func
    return func


@register
def masked(args):
    return [
        hmm.base.JointSegment(
            hmmds.applications.apnea.utilities.read_slow_class(args, record))
        for record in args.records
    ]


@register
def unmasked(args):
    return [
        hmm.base.JointSegment(
            hmmds.applications.apnea.utilities.read_slow(args, record))
        for record in args.records
    ]


@register
def class_peak(args):

    with open(args.boundaries, 'rb') as _file:
        boundaries = pickle.load(_file)

    return [
        hmm.base.JointSegment(
            hmmds.applications.apnea.utilities.read_slow_class_peak(
                args, boundaries, record)) for record in args.records
    ]


@register
def class_peak_interval(args):

    with open(args.boundaries, 'rb') as _file:
        boundaries = pickle.load(_file)

    return [
        hmm.base.JointSegment(
            hmmds.applications.apnea.utilities.read_slow_class_peak_interval(
                args, boundaries, record)) for record in args.records
    ]


def new_ar_order(model, ar_order, y_data):
    """Set parameters for new AR order using weights from orignal model.

    Args:
        model: Original HMM
        ar_order: For new observation model
        y_data: Calculate weights for this data

    Return: y_mod

    """
    old_slow = model.y_mod['slow']
    n_states, old_ar_order_plus_1 = old_slow.coefficients.shape
    if ar_order + 1 > old_ar_order_plus_1:
        model.y_mod.truncate = ar_order

    # Call multi_train to get weights for fitting observation model
    # parameters
    model.multi_train(y_data, 1)
    weights = model.alpha

    model.y_mod['slow'] = hmm.C.AutoRegressive(
        numpy.empty((n_states, ar_order)),
        numpy.empty(n_states),  # offset
        numpy.ones(n_states),  # variance
        old_slow._rng,
        alpha=old_slow.alpha,
        beta=old_slow.beta)
    model.y_mod.observe(y_data)

    # Result will be nonsense if fit is rank deficient
    model.y_mod.reestimate(weights)
    return model.y_mod


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    with open(args.initial_path, 'rb') as _file:
        old_args, model = pickle.load(_file)

    y_data = TYPES[args.type](args)

    if args.AR_order >= 0:
        y_mod = new_ar_order(model, args.AR_order, y_data)
        with open(args.initial_path, 'rb') as _file:
            old_args, model = pickle.load(_file)
        model.y_mod = y_mod
    model.multi_train(y_data, args.iterations)
    hmmds.applications.apnea.utilities.print_chain_model(
        model.y_mod, model.alpha.sum(axis=0), old_args.state_key2state_index)

    model.strip()
    with open(args.write_path, 'wb') as _file:
        pickle.dump((old_args, model), _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
