''' MakeModel.py <H_dir> <data_dir> <data_file> <model_file>
 EG. python MakeModel.py data lorenz.4 m12s.4y
'''

import sys
import os.path
import pickle
import itertools
import argparse

import numpy
import numpy.random

import hmm.base
import hmm.simple


def skip_header(_file):
    """Skip lines that start with #.
    Args:
        _file: An open file object

    """
    return itertools.dropwhile(lambda line: line.startswith("#"), _file)


def read_data(data_dir, data_file):
    '''Read quantized data and return as numpy array.  Shift values by -1
    so that minimum for plots can be 1 while still using [0,n) for hmm
    code.

    '''
    with open(os.path.join(data_dir, data_file), 'r') as file_:
        y = numpy.array([int(line) - 1 for line in skip_header(file_)],
                        numpy.int32)
    return y, y.max() + 1


def main(argv=None):
    '''Call with arguments: n, data_dir, data_file, model_file

    n = number of iterations

    data_dir = directory for data and resulting model

    data_file = name of data file

    model_file = name of the file into which resulting model to be written
    '''

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Make data for figure on cover")
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('n_iterations', type=int)
    parser.add_argument('data_dir', type=str)
    parser.add_argument('data_file', type=str)
    parser.add_argument('model_file', type=str)
    args = parser.parse_args(argv)

    nstates = 12

    y_data, cardy = read_data(args.data_dir, args.data_file)

    # Set random values of initial model parameters
    rng = numpy.random.default_rng(args.random_seed)
    p_state_initial = hmm.simple.Prob(rng.random((1, nstates))).normalize()[0]
    p_state_time_average = hmm.simple.Prob(rng.random(
        (1, nstates))).normalize()[0]
    p_state2state = hmm.simple.Prob(rng.random((nstates, nstates))).normalize()
    p_state2y = hmm.simple.Prob(rng.random((nstates, cardy))).normalize()

    # Train the model
    y_mod = hmm.simple.Observation(p_state2y, rng)
    mod = hmm.base.HMM(p_state_initial, p_state_time_average, p_state2state,
                       y_mod, rng)
    mod.train(y_data, args.n_iterations)

    # Strip and then save model in <model_file>
    mod.deallocate()
    with open(os.path.join(args.data_dir, args.model_file), 'wb') as _file:
        pickle.dump(mod, _file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
