"""model_init.py Create initial HMM models with apnea observations

A rule modified from Rules.mk:

${MODELS}/initial_%: model_init.py utilities.py observation.py
	python model_init.py --root ${ROOT} $* $@

The pattern % or $* selects one of the registered functions in this
module.

"""
import sys
import os.path
import pickle
import argparse

import numpy

import hmm.base
import hmm.simple

import hmm.C
import hmmds.applications.apnea.utilities
import hmmds.applications.apnea.observation
import develop
import utilities


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Create and write/pickle an initial model")
    hmmds.applications.apnea.utilities.common_arguments(parser)
    # args.records is None if --records is not on command line
    parser.add_argument('--records',
                        type=str,
                        nargs='+',
                        help='--records a01 x02 -- ')
    parser.add_argument(
        'key',
        type=str,
        help='One of the functions registered in the source, eg, A4')
    parser.add_argument('write_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


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


def _make_hmm(y_model,
              p_state_initial,
              p_state_time_average,
              p_state2state,
              names,
              args,
              rng,
              Class=develop.hmm):
    """Create a hmm using parameters defined in the caller

    Args:
        y_model: P(y[t]|s[t]) and functions to support reestimation
        p_state_initial:
        p_state_time_average:
        p_state2state: p[t=1] = numpy.dot(p[t=0], p_state2state)
        names: List of record names, eg, 'c01 a05 x35'.split()
        rng: Random number generator

    Return: hmm initialized with data specified by names

    Unsupervised, ie, no bundles. AR-4 for heart rate. 3-d
    multivariate Gaussian for respiration.

    """

    _hmm = Class.HMM(p_state_initial, p_state_time_average, p_state2state,
                     y_model, rng)

    y_data = hmmds.applications.apnea.utilities.list_heart_rate_respiration_data(
        names, args)
    _hmm.initialize_y_model(y_data)
    return _hmm


def _make_hr_resp_model(nu: numpy.ndarray, variances: numpy.ndarray,
                        alpha: numpy.ndarray, beta: numpy.ndarray, rng):
    """Create an observation model

    Args:
        nu: Denominator for respiration variance prior.
        variances: Numerator for respiration variance prior.
        alpha: Denominator for heart rate variance prior.
        beta: Numerator for heart rate variance.
        rng: Random number generator

    Return: observation model

    """
    n_states = len(nu)

    # Prior for respiration variance
    psi = numpy.empty((n_states, 3, 3))
    for state in range(n_states):
        psi[state, :, :] = numpy.eye(3) * nu[state] * variances[state]

    return hmmds.applications.apnea.observation.FilteredHeartRate_Respiration(
        filtered_heart_rate_model(n_states, rng, alpha=alpha, beta=beta),
        respiration_model(n_states, rng, Psi=psi, nu=nu), rng)


def filtered_heart_rate_model(n_states, rng, ar_order=4, **kwargs):
    """Make an initial observation model
    """
    offset = numpy.zeros(n_states)
    variance = numpy.ones(n_states)
    ar_coefficients = numpy.ones((n_states, ar_order)) / ar_order
    return hmmds.applications.apnea.observation.FilteredHeartRate(
        ar_coefficients, offset, variance, rng, **kwargs)


def respiration_model(n_states, rng, dimension=3, **kwargs):
    """Make an initial observation model
    """
    mu = numpy.zeros((n_states, dimension))
    sigma = numpy.zeros((n_states, dimension, dimension))
    for state in range(n_states):
        sigma[state, :, :] = numpy.eye(dimension)
    return hmmds.applications.apnea.observation.Respiration(
        mu, sigma, rng, **kwargs)


def filtered_heart_rate_respiration_bundle_model(n_states,
                                                 bundle2state,
                                                 rng,
                                                 ar_order=4,
                                                 dimension=3):
    """Make an initial observation model
    """
    hr_mod = filtered_heart_rate_model(n_states, rng, ar_order=ar_order)
    res_mod = respiration_model(n_states, rng, dimension=dimension)

    underlying_model = hmmds.applications.apnea.observation.FilteredHeartRate_Respiration(
        hr_mod, res_mod, rng)

    return hmm.base.Observation_with_bundles(underlying_model, bundle2state,
                                             rng)


class State:
    """For defining HMM graph

    Args:
        successors: List of names (dict keys) of successor states
        probabilities: List of float probabilities for successors
        times: List of inetger times
        bundle: Integer class
    """

    def __init__(self, successors, probabilities, bundle):
        self.successors = successors
        self.probabilities = probabilities
        self.bundle = bundle


def dict2hmm(state_dict, underlying_y_model, y_data, rng):
    """Translate definition based on dict to HMM

    Args:
        state_dict: state_dict[state_name] is a State instance
        y_model:
        y_data: A single hmm.base.BundleSegment
        rng: A random number generator
    """

    n_states = len(state_dict)
    n_times = len(y_data)
    bundle2state = {}
    p_state_initial = numpy.ones(n_states) / n_states
    p_state_time_average = numpy.ones(n_states) / n_states
    p_state2state = hmm.simple.Prob(numpy.zeros((n_states, n_states)))

    name2index = {}
    index2name = []
    for index, name in enumerate(state_dict.keys()):
        name2index[name] = index
        index2name.append(name)

    for name, state in state_dict.items():
        index = name2index[name]
        for successor, probability in zip(state.successors,
                                          state.probabilities):
            p_state2state[index, name2index[successor]] = probability
        if state.bundle in bundle2state:
            bundle2state[state.bundle].append(index)
        else:
            bundle2state[state.bundle] = [index]

    p_state2state.normalize()

    y_model = hmm.base.ObservationWithBundles(underlying_y_model, bundle2state,
                                              rng)

    y_model.observe(y_data)
    weights = hmm.simple.Prob(y_model.calculate()).normalize()
    y_model.reestimate(weights)

    # Create and return the hmm
    return hmm.C.HMM(p_state_initial, p_state_time_average, p_state2state,
                     y_model, rng)


MODELS = {}  # Is populated by @register decorated functions.  The keys
# are function names, and the values are functions


def register(func):
    """Decorator that puts function in MODELS dictionary"""
    #See https://realpython.com/primer-on-python-decorators/
    MODELS[func.__name__] = func
    return func


@register
def test(args, rng):
    for key, value in args.__dict__.items():
        print(f'{key}: {value}')


@register
def masked_dict(args, rng):

    n_before = 18  # Fast states before peak
    n_after = 30  # Fast states after peak
    n_slow = 3  # Slow states with transitions to themselves

    state_dict = {}
    # Define slow states with transitions to themselves
    for i in range(n_slow - 1):
        state_dict[f'slow_{i}'] = State([f'slow_{i}', f'slow_{i+1}'], [.5, .5],
                                        0)
    state_dict[f'slow_{n_slow-1}'] = State([-n_before], [1.0], 0)

    # Define fast states that only have forward transitions
    for t in range(-n_before, n_after):
        state_dict[t] = State([t + 1], [1.0], t + n_before + 1)
    # Close loop
    state_dict[n_after] = State(['slow_0'], [1.0], n_after + n_before + 1)

    n_states = len(state_dict)
    ar_order = 3
    ar_coefficients = numpy.ones((n_states, ar_order))
    offset = numpy.ones(n_states)
    variances = numpy.ones(n_states)
    # I think right variance is between .05 and .001, and there are
    # about 50,000 samples per state in a01
    alpha = numpy.ones(n_states) * 1.0e3
    beta = numpy.ones(n_states) * 1.0e2

    underlying = hmm.C.AutoRegressive(ar_coefficients,
                                      offset,
                                      variances,
                                      rng,
                                      alpha=alpha,
                                      beta=beta)

    return dict2hmm(state_dict, underlying,
                    [utilities.read_tagged_ecg("a01", args, n_before, n_after)],
                    rng)

    keys = list(state_dict.keys())
    for key in keys:
        print(
            f"state_dict[{key}].successors={state_dict[key].successors} bundle={state_dict[key].bundle}"
        )
    sys.exit(0)


@register
def masked3_300(args, rng) -> develop.HMM:
    """ Like AR3_300 but no bad state and bundle observations

    """

    assert isinstance(args.records, list)

    # Define observation model and the data
    n_forward = 10
    p_self = 0.05
    n_states = 300
    ar_order = 3
    ar_coefficients = numpy.ones((n_states, ar_order))
    offset = numpy.ones(n_states)
    variances = numpy.ones(n_states)
    alpha = numpy.ones(n_states) * 1.0e2
    beta = numpy.ones(n_states) * 1.0e2

    underlying = hmm.C.AutoRegressive(ar_coefficients,
                                      offset,
                                      variances,
                                      rng,
                                      alpha=alpha,
                                      beta=beta)
    # Allowing state[0:n_forward] only when peak is detected should
    # force one cycle through states for each detected peak.
    y_model = hmm.base.ObservationWithBundles(underlying, {
        0: numpy.arange(n_forward),
        1: numpy.arange(n_forward, n_states)
    }, rng)

    a01_masked_data = utilities.read_masked_ecg("a01", args)
    n_minute = 60 * 100
    # 25*60*100 = 1.5e5
    y_data = [a01_masked_data[75 * n_minute:100 * n_minute]]

    return circulant(n_states,
                     y_model,
                     y_data,
                     rng,
                     n_forward=n_forward,
                     p_self=p_self)


@register
def C1(args, rng):
    """One state, no bundles, AR-4 for heart rate, single Gaussian for
    respiration

    """
    n_states = 1

    y_model = hmmds.applications.apnea.observation.FilteredHeartRate_Respiration(
        filtered_heart_rate_model(n_states, rng),
        respiration_model(n_states, rng), rng)

    name = 'c01'
    y_data = [
        hmmds.applications.apnea.utilities.heart_rate_respiration_data(
            name, args)
    ]

    model = develop.HMM(
        numpy.ones((1,), numpy.float64),  # p_state_initial
        numpy.ones((1,), numpy.float64),  # p_state_time_average
        numpy.ones((1, 1), numpy.float64),  # p_state2state
        y_model,
        rng)

    model.initialize_y_model(y_data)
    return model


@register
def outlier(args, rng) -> develop.HMM:
    """Single state.  For finding model that makes all outliers plausible.

    Args:
        rng: A numpy random number generator instance
        args: A Collection of information for the apnea project.

    """

    # Along with a sum of weights, nu and alpha are in the denominator
    # of the MAP reestimate for the Multivariate respiration and
    # Autoregressive heart rate models respectively.  Since in all of
    # the data combined there are 4,073,904 samples, setting nu and
    # alpha to 1e7 ensures that the prior will dominate the data.
    # Some data is not plausible if the prior for the variance of the
    # AR heart rate model is less than 1e6, indicating that magnitude
    # of outliers is 1e3.  I am surprised that a prior variance of
    # 1e-7 works for the respiration model.

    # Here are results of experiments and my choices
    #                  Fails       OK          Choice
    # (nu, variances)  (1e7, 1e-1) (1e7, 1e0)  (1e7, 1e9)
    # (alpha, beta)    (1e7, 1e13) (1e7, 1e14) (1e7, 1e15)

    nu = numpy.array([1.0e7])
    variances = numpy.array([1.0e9])
    alpha = numpy.array([1.0e7])
    beta = numpy.array([1.0e15])

    p_state_initial = numpy.array([1.0])
    p_state_time_average = numpy.array([1.0])
    p_state2state = numpy.array([[1.0]])

    return _make_hmm(_make_hr_resp_model(nu, variances, alpha, beta, rng),
                     p_state_initial, p_state_time_average, p_state2state,
                     args.all_names, args, rng)


def _two(names, args, rng):
    """Model with two states.

    Args:
        names: Record names for initialization data
        rng: A numpy random number generator instance
        args: A Collection of information for the apnea project.

    State 0 is for outliers.  The other states will get fit to the
    rest of the data.

    """
    nu = numpy.array([1.0e7, 10.0])
    variances = numpy.array([1.0e9, 1.0])
    alpha = numpy.array([1.0e7, 10.0])
    beta = numpy.array([1.0e15, 1.0])

    p_state_initial = numpy.array([.01, .99])
    p_state_time_average = numpy.array([.01, .99])
    p_state2state = numpy.array([  #
        [.99, .01],  #
        [.01, .99],
    ])

    return _make_hmm(_make_hr_resp_model(nu, variances, alpha, beta,
                                         rng), p_state_initial,
                     p_state_time_average, p_state2state, names, args, rng)


def _three(names, args, rng) -> develop.HMM:
    """Model with three states.
    """

    nu = numpy.array([1.0e7, 10.0, 10.0])
    variances = numpy.array([1.0e9, 1.0, 1.0])
    alpha = numpy.array([1.0e7, 10.0, 10.0])
    beta = numpy.array([1.0e15, 1.0, 1.0])

    p_state_initial = numpy.array([.01, .01, .98])
    p_state_time_average = numpy.array([.01, .495, .495])
    p_state2state = numpy.array([  #
        [.99, .001, .001],  #
        [.001, .01, .989],
        [.001, .989, .01]
    ])

    return _make_hmm(_make_hr_resp_model(nu, variances, alpha, beta,
                                         rng), p_state_initial,
                     p_state_time_average, p_state2state, names, args, rng)


def _four(names, args, rng) -> develop.HMM:
    """Model with four states.
    """

    nu = numpy.array([1.0e7, 10.0, 10.0, 10])
    variances = numpy.array([1.0e9, 1.0, 1.0, 1.0])
    alpha = numpy.array([1.0e7, 10.0, 10.0, 10.0])
    beta = numpy.array([1.0e15, 1.0, 1.0, 1.0])

    p_state_initial = numpy.array([.01, .01, .01, .97])
    p_state_time_average = numpy.array([.01, .33, .33, .33])
    p_state2state = numpy.array([  #
        [.99, .001, .001, .008],  #
        [.001, .01, .01, .979],
        [.001, .979, .01, .01],
        [.001, .01, .979, .01]
    ])

    return _make_hmm(_make_hr_resp_model(nu, variances, alpha, beta,
                                         rng), p_state_initial,
                     p_state_time_average, p_state2state, names, args, rng)


@register
def A2(args, rng) -> develop.HMM:
    """Two states initialized with apnea records
    """
    return _two(args.a_names, args, rng)


@register
def C2(args, rng) -> develop.HMM:
    """Two states initialized with normal records
    """
    return _two(args.c_names, args, rng)


@register
def A3(args, rng) -> develop.HMM:
    """Three states initialized with apnea records
    """
    return _three(args.a_names, args, rng)


@register
def C3(args, rng) -> develop.HMM:
    """Three states initialized with normal records
    """
    return _three(args.c_names, args, rng)


@register
def A4(args, rng) -> develop.HMM:
    """Four states initialized with apnea records
    """
    return _four(args.a_names, args, rng)


@register
def C4(args, rng) -> develop.HMM:
    """Four states initialized with normal records
    """
    return _four(args.c_names, args, rng)


@register
def Low(args, rng):
    n_states = 10

    bundle2state = {
        0: numpy.arange(6, dtype=numpy.int32),
        1: numpy.arange(6, 10, dtype=numpy.int32)
    }

    y_model = filtered_heart_rate_respiration_bundle_model(
        n_states, bundle2state, rng)

    # Without several c-names for initialization, training fails and
    # reports that the data is not plausible
    y_data = [
        hmmds.applications.apnea.utilities.heart_rate_respiration_bundle_data(
            name, args) for name in 'c01 c02 c03 c04 c05 c06'.split()
    ]

    model = develop.HMM(random_1d_prob(rng, n_states),
                        random_1d_prob(rng, n_states),
                        hmm.simple.Prob(P_SS_LowMedium).normalize(), y_model,
                        rng)
    assert model.p_state_initial.min() > 0
    model.initialize_y_model(y_data)
    assert model.p_state_initial.min() > 0
    return model


@register
def Medium(args, rng):
    return Low(args, rng)


@register
def High(args, rng):
    n_states = 14

    bundle2state = {
        0: numpy.arange(7, dtype=numpy.int32),
        1: numpy.arange(7, 14, dtype=numpy.int32)
    }

    y_model = filtered_heart_rate_respiration_bundle_model(
        n_states, bundle2state, rng)

    # Without several a-names for initialization, training fails and
    # reports that the data is not plausible
    y_data = [
        hmmds.applications.apnea.utilities.heart_rate_respiration_bundle_data(
            name, args) for name in 'a01 a02 a03 a04 a05 a06'.split()
    ]

    model = develop.HMM(random_1d_prob(rng, n_states),
                        random_1d_prob(rng, n_states),
                        hmm.simple.Prob(P_SS_High).normalize(), y_model, rng)
    model.initialize_y_model(y_data)
    return model


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)
    rng = numpy.random.default_rng(3)

    # Run the function specified by args.key
    model = MODELS[args.key](args, rng)
    assert model.p_state_initial.min() > 0

    model.strip()
    with open(args.write_path, 'wb') as _file:
        pickle.dump(model, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
