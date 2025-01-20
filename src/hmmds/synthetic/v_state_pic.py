"""VStatePic.py data_dir y_name state_file_preface

Creates varg_stateN (N in 0:11) in the directory named by data_dir

"""

import sys
from os.path import join
import argparse

import numpy
import numpy.random

import utilities
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
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--nu', type=float, default=1.0)
    parser.add_argument('--Psi', type=float, default=4.0)
    parser.add_argument('--t_sample', type=float, default=0.15)
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
    assert vector_dim == 3

    # Initial 5 state model has plausible approximate symmetry
    model_5 = make_varg_hmm(args, vector_dim, context_dim, vectors, n_states=5)
    # Overwrite states with fits to fixed points
    for state, sign in enumerate((-1, 0, 1)):
        fixed_point = utilities.FixedPoint(sign=sign)
        model_5.y_mod.a_mean[state, 0:3,
                             0:3] = fixed_point.dPhi_dx(args.t_sample)
        model_5.y_mod.a_mean[state, :, 3] = fixed_point.fixed_point
    iterations = 70
    model_5.train(y, iterations)

    # I don't understand why the trained 12 state model has so many
    # states along the x_2 axis
    model_12 = make_varg_hmm(args,
                             vector_dim,
                             context_dim,
                             vectors,
                             n_states=12)
    model_12.y_mod.a_mean[:5, :, :] = model_5.y_mod.a_mean
    model_12.y_mod.sigma[:5, :, :] = model_5.y_mod.sigma
    model_12.train(y, iterations)

    # Do Viterbi decoding
    states = model_12.decode(y)

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


def make_varg_hmm(args,
                  out_dimension,
                  context_dimension,
                  vectors,
                  n_states=None):
    """Returns a normalized random initial model.

    Args:
        args: From command line need: n_states, random_seed, nu, Psi, t_sample
        out_dimension: 3-d Lorenz data
        context_dimension: 4-d Lorenz + offset
        vectors: A sequence of observations

    Return:
        A model that is consistent with the data "vectors".

    """

    assert out_dimension == 3
    assert context_dimension == 4

    if n_states is None:
        n_states = args.n_states

    # Make VARG observation model
    rng = numpy.random.default_rng(args.random_seed)
    a_forecast = numpy.zeros((n_states, out_dimension, context_dimension))
    for state in range(n_states):
        # Setup forecast to work perfectly for state at time t_state
        a_forecast[state, 0:3, 0:3] = numpy.eye(out_dimension)
        t_state = rng.integers(1, high=len(vectors))
        a_forecast[state, :, 3] = vectors[t_state] - vectors[t_state - 1]

    # Setup covariance of each state to be the covariance of all of the data
    covariance = numpy.cov(vectors.T)
    assert covariance.shape == (3, 3)
    covariance_state = numpy.empty((n_states, 3, 3))
    for state in range(n_states):
        covariance_state[state] = covariance

    # Make other parameters for HMM.  Rely on differing a_forecast to
    # break symmetry
    p_s0 = hmm.simple.Prob(numpy.ones((1, n_states))).normalize()[0]
    p_s0_ergodic = hmm.simple.Prob(numpy.ones((1, n_states))).normalize()[0]
    p_s_to_s = hmm.simple.Prob(numpy.ones((n_states, n_states))).normalize()

    # Make the model
    model = hmm.base.HMM(
        p_s0, p_s0_ergodic, p_s_to_s,
        hmm.observe_float.VARG(a_forecast,
                               covariance_state,
                               rng,
                               nu=args.nu,
                               Psi=args.Psi))
    assert model.p_state_initial.shape == (n_states,)
    return model


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
