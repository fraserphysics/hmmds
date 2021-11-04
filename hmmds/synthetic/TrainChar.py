""" TrainChar.py y_data out_data

Read an integer time series from the file "y_data" and write 5
training characteristics to a file named "out_data".
"""
import sys

import numpy
import numpy.random

import hmm.simple
from hmm.simple import HMM  # change to c version if/when written


def random_hmm(Card_Y, N_states, seed):
    '''
    '''
    rng = numpy.random.default_rng(seed)

    def random_prob(shape):
        return hmm.simple.Prob(rng.random(shape)).normalize()

    P_S0 = random_prob((1, N_states))[0]
    P_S0_ergodic = random_prob((1, N_states))[0]
    P_ScS = random_prob((N_states, N_states))
    P_YcS = random_prob((N_states, Card_Y))
    observation_model = hmm.simple.Observation(P_YcS, rng)
    return HMM(P_S0, P_S0_ergodic, P_ScS, observation_model, rng)


def main(argv=None):
    if argv is None:  # Usual case
        argv = sys.argv[1:]
    assert len(argv) == 2
    y_file, out_name = argv

    niterations = 500
    N_states = 12
    Y = numpy.array([int(line) for line in open(y_file, 'r')], numpy.int32)
    Card_Y = Y.max()
    Y -= 1
    n_seeds = 5
    LL = numpy.empty((niterations, n_seeds))
    for seed in range(n_seeds):
        model = random_hmm(Card_Y, N_states, seed)
        LL[:, seed] = model.train(Y, niterations, display=False)
    f = open(out_name, 'w')
    for i in range(niterations):
        print('%3d' % i, (n_seeds * ' %7.4f') % tuple(LL[i]), file=f)
    f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
