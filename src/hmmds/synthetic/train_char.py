""" TrainChar.py y_data out_data

Read an integer time series from the file "y_data" and write 5
training characteristics to a file named "out_data".
"""
import sys
import argparse

import numpy
import numpy.random

import hmm.simple
# Instead of hmm.simple.HMM, use c version if/when written


def parse_args(argv):
    """Parse the command line.
    """
    parser = argparse.ArgumentParser(
        description='Train starting from several different intial models')
    parser.add_argument('--n_iterations', type=int, default=5)
    parser.add_argument('--n_states', type=int, default=12)
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('in_path', type=str)
    parser.add_argument('out_path', type=str)
    return parser.parse_args(argv)


def random_hmm(n_y, n_states, seed):
    """Create and return a hmm.simple.HMM
    """
    rng = numpy.random.default_rng(seed)

    def random_prob(shape):
        return hmm.simple.Prob(rng.random(shape)).normalize()

    p_s0 = random_prob((1, n_states))[0]
    p_s0_ergodic = random_prob((1, n_states))[0]
    p_s_to_s = random_prob((n_states, n_states))
    p_s_to_y = random_prob((n_states, n_y))
    observation_model = hmm.simple.Observation(p_s_to_y, rng)
    return hmm.simple.HMM(p_s0, p_s0_ergodic, p_s_to_s, observation_model, rng)


def main(argv=None):
    """Make data for "Training characteristics" figure.

    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    y = numpy.array([int(line) for line in open(args.in_path, 'r')],
                    numpy.int32)
    # EG y.shape = (20000,) and values are from set {1,2,3,4}
    n_y = y.max()
    y -= 1
    log_likelihood = numpy.empty((args.n_iterations, args.n_seeds))
    for seed in range(args.n_seeds):
        model = random_hmm(n_y, args.n_states, seed)
        log_likelihood[:, seed] = model.train(y,
                                              args.n_iterations,
                                              display=False)
    with open(args.out_path, 'w') as output:
        for i in range(args.n_iterations):
            print('%3d' % i,
                  (args.n_seeds * ' %7.4f') % tuple(log_likelihood[i]),
                  file=output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
