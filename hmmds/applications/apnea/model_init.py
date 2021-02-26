"""model_init.py Create initial HMM models with apnea observations

From Makefile:

python model_init.py  ${COMMON_ARGS} $* $@

$* is one of: A2 C1 High Medium Low
"""
import sys
import os.path
import pickle
import argparse

import numpy

import hmm.base

import utilities
import observation


def random_1d_prob(rng: numpy.random.Generator, length: int) -> numpy.ndarray:
    """Draw a probability vector from a uniform distribution
    Args:
       rng: Random number generator
       length: Dimension of returned vector

    Returns:
        A normalized probability mass function
    """
    vector = rng.random(length)
    return vector / vector.sum()


def random_conditional_prob(rng: numpy.random.Generator,
                            shape: tuple) -> hmm.simple.Prob:
    """Draw a random conditional distribution P_{a|b}
    Args:
       rng: Random number generator
       shape: (|A|,|B|) where |A| is the number of possible values of a

    Returns:
        A normalized conditional distribution
    """
    return hmm.simple.Prob(rng.random(shape)).normalize()


# State transition probabilities for the Low and Medium models
P_SS_LowMedium = hmm.simple.Prob(
    numpy.array(
        [
            #1  2  3  4  5  6  7  8  9  10
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #1
            [1, 1, 1, 1, 1, 0, 1, 0, 0, 0],  #2
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  #3
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],  #4
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],  #5
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  #6
            [0, 1, 0, 0, 0, 0, 1, 1, 1, 0],  #7
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  #8
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],  #9
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  #10
        ],
        dtype=numpy.float64))

# State transition probabilities for the High model
P_SS_High = hmm.simple.Prob(
    numpy.array(
        [
            #1  2  3  4  5  6  7  8  9  10 11 12 13 14
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #1
            [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],  #2
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #3
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #4
            [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  #5
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #6
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  #7
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  #8
            [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],  #9
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  #10
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],  #11
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],  #12
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  #13
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]  #14
        ],
        dtype=numpy.float64),)


def filtered_heart_rate_model(n_states, rng, ar_order=4):
    """Make an initial observation model
    """
    offset = numpy.zeros(n_states)
    variance = numpy.ones(n_states)
    ar_coefficients = numpy.ones((n_states, ar_order)) / ar_order
    return observation.FilteredHeartRate(ar_coefficients, offset, variance, rng)


def respiration_model(n_states, rng, dimension=3):
    """Make an initial observation model
    """
    mu = numpy.zeros((n_states, dimension))
    sigma = numpy.zeros((n_states, dimension, dimension))
    for state in range(n_states):
        sigma[state, :, :] = numpy.eye(dimension)
    return observation.Respiration(mu, sigma, rng)


def filtered_heart_rate_respiration_bundle_model(n_states,
                                                 bundle2state,
                                                 rng,
                                                 ar_order=4,
                                                 dimension=3):
    """Make an initial observation model
    """
    hr_mod = filtered_heart_rate_model(n_states, rng, ar_order=ar_order)
    res_mod = respiration_model(n_states, rng, dimension=dimension)

    underlying_model = observation.FilteredHeartRate_Respiration(
        hr_mod, res_mod, rng)

    return hmm.base.Observation_with_bundles(underlying_model, bundle2state,
                                             rng)


MODELS = {}  # Is populated by @register decorated functions.  The keys
# are function names, and the values are functions


def register(func):
    """Decorator that puts function in MODELS dictionary"""
    #See https://realpython.com/primer-on-python-decorators/
    MODELS[func.__name__] = func
    return func


@register
def A2(args: argparse.Namespace) -> hmm.base.HMM:
    """Two states, no bundles, AR-4 for heart rate, single Gaussian for
    respiration

    Args:
        args: Collection of information for the apnea project.
            args.rng is a numpy random number generator instance

    Use data from a01 to initialize parameters of the observation model
    """
    n_states = 2

    y_model = observation.FilteredHeartRate_Respiration(
        filtered_heart_rate_model(n_states, args.rng),
        respiration_model(n_states, args.rng), args.rng)

    y_data = utilities.pattern_heart_rate_respiration_data(args, ['a'])
    # a list with a dict for each a-file

    model = hmm.base.HMM(
        random_1d_prob(args.rng, 2),  # p_state_initial
        random_1d_prob(args.rng, 2),  # p_state_time_average
        random_conditional_prob(args.rng, (2, 2)),  # p_state2state
        y_model,
        args.rng)

    model.initialize_y_model(y_data)
    return model


@register
def C1(args):
    """One state, no bundles, AR-4 for heart rate, single Gaussian for
    respiration

    """
    n_states = 1

    y_model = observation.FilteredHeartRate_Respiration(
        filtered_heart_rate_model(n_states, args.rng),
        respiration_model(n_states, args.rng), args.rng)

    name = 'c01'
    y_data = [
        utilities.heart_rate_respiration_data(
            os.path.join(args.heart_rate, name),
            os.path.join(args.respiration, name))
    ]

    model = hmm.base.HMM(
        numpy.ones((1,), numpy.float64),  # p_state_initial
        numpy.ones((1,), numpy.float64),  # p_state_time_average
        numpy.ones((1, 1), numpy.float64),  # p_state2state
        y_model,
        args.rng)

    model.initialize_y_model(y_data)
    return model


@register
def Low(args):
    n_states = 10

    bundle2state = {
        0: numpy.arange(6, dtype=numpy.int32),
        1: numpy.arange(6, 10, dtype=numpy.int32)
    }

    y_model = filtered_heart_rate_respiration_bundle_model(
        n_states, bundle2state, args.rng)

    name = 'c01'
    y_data = [
        utilities.heart_rate_respiration_bundle_data(
            os.path.join(args.heart_rate, name),
            os.path.join(args.respiration, name),
            args.expert,
            name,
        )
    ]

    model = hmm.base.HMM(random_1d_prob(args.rng, n_states),
                         random_1d_prob(args.rng, n_states),
                         hmm.simple.Prob(P_SS_LowMedium).normalize(), y_model,
                         args.rng)
    model.initialize_y_model(y_data)
    return model


@register
def Medium(args):
    return Low(args)


@register
def High(args):
    n_states = 14

    bundle2state = {
        0: numpy.arange(7, dtype=numpy.int32),
        1: numpy.arange(7, 14, dtype=numpy.int32)
    }

    y_model = filtered_heart_rate_respiration_bundle_model(
        n_states, bundle2state, args.rng)

    name = 'a01'
    y_data = [
        utilities.heart_rate_respiration_bundle_data(
            os.path.join(args.heart_rate, name),
            os.path.join(args.respiration, name),
            args.expert,
            name,
        )
    ]

    model = hmm.base.HMM(random_1d_prob(args.rng, n_states),
                         random_1d_prob(args.rng, n_states),
                         hmm.simple.Prob(P_SS_High).normalize(), y_model,
                         args.rng)
    model.initialize_y_model(y_data)
    return model


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser("Create and write/pickle an initial model")
    utilities.common_args(parser)
    parser.add_argument('key', type=str, help='One of A2 C1 High Medium Low')
    parser.add_argument('write_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    args.rng = numpy.random.default_rng()

    # Run the function specified by args.key and apply deallocate to
    # the result to delete alpha, beta, and gamma
    model = MODELS[args.key](args).deallocate()

    with open(args.write_path, 'wb') as _file:
        pickle.dump(model, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
