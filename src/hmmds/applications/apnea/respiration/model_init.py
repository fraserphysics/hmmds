"""model_init.py Create initial HMM models with apnea observations

"""
import sys
import os.path
import pickle
import argparse

import numpy

import hmm.base
import hmm.simple

import hmm.C
import hmmds.applications.apnea.develop
import hmmds.applications.apnea.respiration.utilities as utilities
from utilities import State


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Create and write/pickle an initial model")
    utilities.common_arguments(parser)
    parser.add_argument("--alpha_beta",
                        type=float,
                        nargs=2,
                        default=(5.0e1, 1.0e1),
                        help="Paramters of inverse gamma prior for variance")
    parser.add_argument(
        'key',
        type=str,
        help='One of the functions registered in the source, eg, apnea_dict')
    parser.add_argument('write_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


MODELS = {}  # Is populated by @register decorated functions.  The keys
# are function names, and the values are functions


def register(func):
    """Decorator that puts function in MODELS dictionary"""
    #See https://realpython.com/primer-on-python-decorators/
    MODELS[func.__name__] = func
    return func


@register  # Model for "c" records
def c_model(args, rng):
    """Return an hmm that finds all minutes normal

    """

    n_states = 2
    p_state_initial = numpy.array([1.0, 0])
    p_state_time_average = numpy.array([1.0, 0])
    p_state2state = hmm.simple.Prob(numpy.ones((n_states, n_states))) / 2

    class_index2state_indices = {0: [0], 1: [1]}

    y_model = hmm.base.JointObservation({
        'slow':
            hmm.observe_float.Gauss(numpy.array([50, -1e6]),
                                    numpy.ones(2) * 1.0e4, rng),
        'class':
            hmm.base.ClassObservation(class_index2state_indices),
    })

    # Create and return the hmm
    hmm_ = hmmds.applications.apnea.develop.HMM(p_state_initial,
                                                p_state_time_average,
                                                p_state2state, y_model, args,
                                                rng)
    args.read_y_class = utilities.read_slow_class
    args.read_raw_y = utilities.read_slow

    # Next two lines are for debugging more complicated models
    state_key2state_index = {0: 0, 1: 1}
    state_dict = {0: State([0], [1.0], y_model), 1: State([0], [1.0], y_model)}

    return hmm_, state_dict, state_key2state_index


@register  # Model that uses respiration signal
def respiration2(args, rng):
    """Return an hmm with two states that model the respiration signal

    """

    n_states = 2
    p_state_initial = numpy.ones(n_states) / n_states
    p_state_time_average = p_state_initial.copy()
    p_state2state = hmm.simple.Prob(numpy.ones((n_states, n_states))) / 2

    class_index2state_indices = {0: [0], 1: [1]}

    y_model = hmm.base.JointObservation({
        'respiration':
            hmm.observe_float.Gauss(numpy.array([1.0, 3.0]),
                                    numpy.ones(2) * 2.0, rng),
        'class':
            hmm.base.ClassObservation(class_index2state_indices),
    })

    # Create and return the hmm
    hmm_ = hmmds.applications.apnea.develop.HMM(p_state_initial,
                                                p_state_time_average,
                                                p_state2state, y_model, args,
                                                rng)
    args.read_y_class = utilities.read_respiration_class
    args.read_raw_y = utilities.read_respiration

    # Next two lines are for debugging more complicated models
    state_key2state_index = {0: 0, 1: 1}
    state_dict = {0: State([0], [1.0], y_model), 1: State([0], [1.0], y_model)}

    return hmm_, state_dict, state_key2state_index


def main(argv=None):
    """Create an hmm and write it as a pickle.
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)
    rng = numpy.random.default_rng(3)

    # Run the function specified by args.key
    model, state_dict, state_key2state_index = MODELS[args.key](args, rng)
    assert model.p_state_initial.min() >= 0

    model.strip()
    args.state_dict = state_dict
    args.state_key2state_index = state_key2state_index
    with open(args.write_path, 'wb') as _file:
        pickle.dump(model, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
