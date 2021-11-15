"""VStatePic.py data_dir y_name state_file_preface

Creates varg_stateN (N in 0..11) in the directory named by data

"""

import sys
from os.path import join
import pickle
import argparse

import numpy
import numpy.random

import hmm.base
import hmm.observe_float
import hmm.simple

import hmmds.synthetic.MakeModel


def main(argv=None):
    """Call with arguments: data_dir vector_file

    Writes files named ['state%d'%n for n in range(nstates)] to the
    data_dir.  Each file consists of points in vector_file that are
    decoded to the the state number specified in the name.  The states
    are assigned by using the model in model_file to Viterbi decode
    the data in data_file.

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description=
        "Make data for figure of states from vector autoregressive model")
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--data_in', type=str)
    parser.add_argument('--out_preface', type=str)

    args = parser.parse_args(argv)

    # Read in time series of vectors
    vectors = numpy.array([
        list(map(float, line.split()))
        for line in hmmds.synthetic.MakeModel.skip_header(
            open(join(args.data_dir, args.data_in), 'r'))
    ])
    n_y, Odim = vectors.shape
    n_y -= 1
    Cdim = Odim + 1
    assert Cdim == 4
    n_states = 12

    y = (vectors,)
    model = MakeVARG_HMM(n_states, Odim, Cdim, vectors)  # Make initial model
    # Recall update formula:   Cov[s] = (Psi + rrsum)/(wsum[s]+nu+dimension+1)
    for nu, psi in ((1e6, 4e6), (4.0, 1.0), (1.0, 0.25), (0.0, 0.0)):
        model.y_mod.nu = nu
        model.y_mod.Psi = numpy.eye(3) * psi
        model.train(y, 10)  # ToDo: Printed LLps is not monotonic
    states = model.decode(y)  # Do Viterbi decoding

    f = list(
        open(join(args.data_dir, 'varg_state' + str(s)), 'w')
        for s in range(n_states))
    for t in range(n_y):
        print('%7.4f %7.4f %7.4f' % tuple(vectors[t]), file=f[states[t]])
    return 0


def MakeVARG_HMM(n_states, out_dimension, context_dimension, vectors):
    """Returns a normalized random initial model
    """

    # Make VARG observation model
    rng = numpy.random.default_rng(0)
    ts = rng.integers(1, len(vectors), n_states)
    a_forecast = numpy.zeros((n_states, out_dimension, context_dimension))
    # Set up forecast to work perfectly for state s at time ts[s]
    for s in range(n_states):
        # Assign one column at a time
        a_forecast[s, :, 0] = [1, 0, 0]
        a_forecast[s, :, 1] = [0, 1, 0]
        a_forecast[s, :, 2] = [0, 0, 1]
        a_forecast[s, :, 3] = vectors[ts[s]] - vectors[ts[s] - 1]
    mean = numpy.mean(vectors, axis=0)
    assert mean.shape == (3,)
    covariance = numpy.cov(vectors.T)
    assert covariance.shape == (3, 3)
    covariance_state = numpy.empty((n_states, 3, 3))
    for s in range(n_states):
        covariance_state[s] = covariance
    y_model = hmm.observe_float.VARG(a_forecast, covariance_state, rng)

    # Make other parameters for HMM
    P_S0 = hmm.simple.Prob(rng.random((1, n_states))).normalize()[0]
    P_S0_ergodic = hmm.simple.Prob(rng.random((1, n_states))).normalize()[0]
    P_ScS = hmm.simple.Prob(rng.random((n_states, n_states))).normalize()

    model = hmm.base.HMM(P_S0, P_S0_ergodic, P_ScS, y_model)
    assert model.p_state_initial.shape == (12,)
    return model


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
