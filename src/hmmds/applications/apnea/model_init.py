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


def _make_hmm(y_model, p_state_initial, p_state_time_average, p_state2state,
              names, args, rng, Class=develop.hmm):
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
def AR1k20(args, rng) -> develop.HMM:
    """Normally states progress monotonically around a loop of 20
    normal states.  Transitions go from a state to itself or to the
    next state in the loop.  An extra state, called bad, exists to
    cover anomalous observations.  A pair of low probability
    transisitons connects each of the normal states to the bad state,
    and the bad state transitions to itself with high probability.

    """

    assert isinstance(args.records, list)
    n_states = 21

    # Define state probability parameters
    p_state_initial = numpy.ones(n_states) * 1e-3
    p_state_initial[0] = 1.0 - p_state_initial.sum()  # Break symmetry
    p_state_time_average = numpy.ones(n_states) / n_states
    p_state2state = numpy.zeros((n_states, n_states))
    normal = p_state2state[:-1, :-1]

    small = 1e-3
    bad = n_states - 1
    for state in range(n_states - 1):
        normal[state, state] = .5 - small
        normal[state - 1, state] = .5
        p_state2state[state, bad] = small
        p_state2state[bad, state] = small
    p_state2state[bad, bad] = 1.0 - small * (n_states - 1)

    # Define observation model and the data
    n_context = 4
    coefficients = numpy.ones((n_states, n_context))
    variances = numpy.ones(n_states)
    n_history = 1000
    # A weak prior is sufficient for normal states because the network
    # structure ensures that they will all have about the same weight.
    # Lead noise in the data goes to +/- 10 mV, and there are about
    # 1e7 total observations in the training data.  So the prior for
    # the bad state ensures that it will have a variance of about 100
    # that will lead noise plausible.
    alpha = numpy.ones(n_states) * 1.0e2
    beta = numpy.ones(n_states) * 1.0e2
    alpha[bad] = 1.0e8
    beta[bad] = 1.0e20
    # 1e10 Lets the bad state model the R peak
    # 1e16 with 20 iterstions lets R state model noise 2 cycles per beat
    # 1e13 with 10 iterations lets R state model noise 2 cycles per beat
    # 1e20 Train on a01 only.  Nice result
    # 1e20 Train on a01 x02 b01 c05.  Result not nice.

    # 1e20 Train starting with model trained on a01 only then train on
    # a01 x02 b01 c05.  Performs well on all training data
    y_model = hmmds.applications.apnea.observation.ECG(coefficients,
                                                       variances,
                                                       rng,
                                                       alpha=alpha,
                                                       beta=beta,
                                                       n_history=n_history)
    paths = [
        os.path.join(args.root, 'raw_data/Rtimes', f'{name}.ecg')
        for name in args.records
    ]
    y_data = [
        hmmds.applications.apnea.observation.read_ecg(path) for path in paths
    ]

    # Create and initialize the hmm
    model = hmm.C.HMM(p_state_initial, p_state_time_average, p_state2state,
                      y_model, rng)
    model.initialize_y_model(y_data)

    return model


@register
def ECG300(args, rng) -> develop.HMM:
    r"""300 regular states in a loop and one _bad_ state for trouble
    with leads.  The bad state has transitions to itself and all of
    the other states.  The regular states don't have transitions to
    themselves.  They can step forward around the loop by 1 to 10
    steps which makes it possible to model pulse rates from 20 to 200
    beats per minute.  Since the data is sampled at 100Hz, I get:

    Pulse (bpm)  Period (sec)  Samples/beat
    20           3.0           300
    200          0.3            30
    60           1.0           100

    Make P(i+j|i) \propto 1/j for j in [1,...,10] and get expected
    value of j = 3.414 or 68.28 cylces per minute.  P(i|t=0) = [.99,
    .01/300, ...].  P(i|bad) = P(bad|i) = (1.0e-3)/300, P(bad|bad) =
    (1.0-1.0e-3)

    """

    assert isinstance(args.records, list)
    n_states = 301
    bad = n_states - 1
    epsilon = 1.0e-5
    n_forward = 10

    # Define state probability parameters
    p_state_initial = numpy.ones(n_states) * 0.01 /(n_states-1)
    p_state_initial[0] = .99
    p_state_initial /= p_state_initial.sum()

    p_state_time_average = numpy.ones(n_states)/n_states

    p_state2state = hmm.simple.Prob(numpy.zeros((n_states, n_states)))
    row = numpy.zeros(n_states-1)
    row[1:n_forward+1] = 1/numpy.arange(1,n_forward+1)
    for i in range(1,n_forward+1):
        row[i] = 1/i
    row /= row.sum()
    for i in range(n_states-1):
        p_state2state[i,:-1] = numpy.roll(row, i)
    p_state2state[bad,:] = epsilon
    p_state2state[:,bad] = epsilon
    p_state2state[bad,bad] = 1.0 - (n_states-1)*epsilon
    p_state2state.normalize()

    # Define observation model and the data
    n_context = 4
    coefficients = numpy.ones((n_states, n_context))
    variances = numpy.ones(n_states)
    n_history = 1000
    alpha = numpy.ones(n_states) * 1.0e2
    beta = numpy.ones(n_states) * 1.0e2
    alpha[bad] = 1.0e8
    beta[bad] = 1.0e20
    y_model = hmmds.applications.apnea.observation.ECG(coefficients,
                                                       variances,
                                                       rng,
                                                       alpha=alpha,
                                                       beta=beta,
                                                       n_history=n_history)
    paths = [
        os.path.join(args.root, 'raw_data/Rtimes', f'{name}.ecg')
        for name in args.records
    ]
    y_data = [
        hmmds.applications.apnea.observation.read_ecg(path) for path in paths
    ]

    # Create and initialize the hmm
    model = hmm.C.HMM(p_state_initial, p_state_time_average, p_state2state,
                      y_model, rng)
    n_y = len(y_data[0])
    state_sequence = numpy.zeros((n_y,), dtype=int)
    for i in range(n_y):
        state_sequence[i] = i%n_states
    model.initialize_y_model(y_data, state_sequence)

    return model

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

    with open(args.write_path, 'wb') as _file:
        pickle.dump(model, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
