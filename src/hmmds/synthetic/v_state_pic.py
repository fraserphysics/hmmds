"""VStatePic.py data_dir y_name state_file_preface

Creates varg_stateN (N in 0:11) in the directory named by data_dir

"""

import sys
from os.path import join
import argparse

import numpy
import numpy.random

import hmm.base
import hmm.observe_float
import hmm.simple


def parse_args(argv):
    """Parse the command line.
    """

    parser = argparse.ArgumentParser(
        description=
        "Make data for figure of states from vector autoregressive model")
    parser.add_argument('--n_states', type=int, default=12)
    parser.add_argument('data_dir', type=str)
    parser.add_argument('data_in', type=str)
    parser.add_argument('out_preface', type=str)
    return parser.parse_args(argv)


def main(argv=None):
    """Call with arguments: data_dir vector_file

    Writes files named ['state%d'%n for n in range(nstates)] to the
    data_dir.  Each file consists of points in vector_file that are
    decoded to the the state number specified in the name.  The states
    are assigned by using the model in model_file to Viterbi decode
    the data in data_file.

    """

    # This is repetitive boilerplate, pylint: disable = duplicate-code
    if argv is None:  # Usual case
        argv = sys.argv[1:]
    args = parse_args(argv)

    # Read in time series of vectors
    with open(join(args.data_dir, args.data_in), encoding='utf-8',
              mode='r') as file_:
        vectors = numpy.array(
            [list(map(float, line.split())) for line in file_.readlines()])
    y = (vectors,)

    n_t, vector_dim = vectors.shape
    n_t -= 1
    context_dim = vector_dim + 1
    assert context_dim == 4

    # Make initial model
    model = make_varg_hmm(args.n_states, vector_dim, context_dim, vectors, 1.0e6, 4.0e6)

    # Train while loosening prior.  Recall update formula: Cov[s] =
    # (Psi + rrsum)/(wsum[s]+nu+dimension+1).  FixMe: Don't modify
    # from outside.  Call some kind of method of model.y_mod instead.
    for scale in ( 1, 1e-5, .5, .5):
        model.y_mod.nu *= scale
        model.y_mod.Psi *= scale
        model.train(y, 10)

    # Do Viterbi decoding
    states = model.decode(y)

    # Write the vectors that were decoded for each state.
    # pylint: disable = consider-using-f-string, consider-using-with
    state_files = list(
        open(join(args.data_dir, 'varg_state' + str(state)),
             encoding='utf-8',
             mode='w') for state in range(args.n_states))
    for t in range(n_t):
        print('%7.4f %7.4f %7.4f' % tuple(vectors[t]),
              file=state_files[states[t]])
    return 0


def make_varg_hmm(n_states, out_dimension, context_dimension, vectors, nu, psi):
    """Returns a normalized random initial model.

    Args:
        n_states:
        out_dimension:
        context_dimension:
        vectors: A sequence of observations

    Return:
        A model that is consistent with the data "vectors".

    """

    # Make VARG observation model
    rng = numpy.random.default_rng(0)

    # Generate a random time for each state
    t_state = rng.integers(1, high=len(vectors), size=n_states)

    # Setup forecast and to work perfectly for state s at time t_state[state]
    a_forecast = numpy.zeros((n_states, out_dimension, context_dimension))
    for state in range(n_states):
        # Assign one column at a time
        a_forecast[state, :, 0] = [1, 0, 0]
        a_forecast[state, :, 1] = [0, 1, 0]
        a_forecast[state, :, 2] = [0, 0, 1]
        a_forecast[state, :,
                   3] = vectors[t_state[state]] - vectors[t_state[state] - 1]

    # Setup covariance of each state to be the covariance of all of the data
    covariance = numpy.cov(vectors.T)
    assert covariance.shape == (3, 3)
    covariance_state = numpy.empty((n_states, 3, 3))
    for state in range(n_states):
        covariance_state[state] = covariance

    # Make other parameters for HMM
    p_s0 = hmm.simple.Prob(rng.random((1, n_states))).normalize()[0]
    p_s0_ergodic = hmm.simple.Prob(rng.random((1, n_states))).normalize()[0]
    p_s_to_s = hmm.simple.Prob(rng.random((n_states, n_states))).normalize()

    # Make the model
    model = hmm.base.HMM(
        p_s0, p_s0_ergodic, p_s_to_s,
        hmm.observe_float.VARG(a_forecast, covariance_state, rng, nu=nu, Psi=psi))
    assert model.p_state_initial.shape == (12,)
    return model


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
