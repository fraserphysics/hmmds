"""like_lor.py Make data for plot of cross entropy vs number of states

"""

import sys
import argparse
import pickle

import numpy
import scipy.sparse

import hmmds.synthetic.bounds.lorenz


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(
        description='Illustrate cross entropy vs number of states')
    parser.add_argument('--log_resolution',
                        type=float,
                        nargs=3,
                        default=[4.0, -0.1, -0.5],
                        help='Range of log of quantization resolution base 2')
    parser.add_argument('--n_relax',
                        type=int,
                        default=500,
                        help='Number of steps to relax to attractor')
    parser.add_argument('--n_train',
                        type=int,
                        default=100000,
                        help='Training sample size')
    parser.add_argument('--n_test',
                        type=int,
                        default=1000,
                        help='Testing sample size')
    parser.add_argument('--n_quantized',
                        type=int,
                        default=4,
                        help='Number of different values of measurements')
    parser.add_argument('--x_initial',
                        type=float,
                        nargs=3,
                        default=numpy.ones(3),
                        help='Initial conditions')
    parser.add_argument('--t_sample',
                        type=float,
                        default=0.15,
                        help='Sample interval')
    parser.add_argument('--min_prob',
                        type=float,
                        default=1e-4,
                        help='Minimum conditional observation probability')
    parser.add_argument('result', type=str, help='Where to write result')
    return parser.parse_args(argv)


def make_data(args: argparse.Namespace):
    """
    Args:
        args: Command line arguments

    Return: (xyz_train, q_train, q_test)

    """
    assert args.n_quantized % 2 == 0  # Must be even for quantization
    # scheme that has boundary at 0
    n_total = args.n_relax + args.n_train + args.n_test
    initial = numpy.array(args.x_initial)
    assert initial.shape == (3,)
    x_all = hmmds.synthetic.bounds.lorenz.n_steps(initial, n_total,
                                                  args.t_sample)
    assert x_all.shape == (n_total, 3)
    xyz_train = x_all[args.n_relax:args.n_relax + args.n_train]
    x_train = xyz_train[:, 0]
    x_min = x_train.min()
    x_max = x_train.max()
    assert x_min < 0 < x_max
    size = max(x_max, -x_min) / (args.n_quantized / 2)
    bins = numpy.linspace(-size, size, args.n_quantized + 1)[1:-1]
    q_train = numpy.digitize(x_train, bins)
    x_test = x_all[args.n_relax + args.n_train:, 0]
    q_test = numpy.digitize(x_test, bins)

    assert xyz_train.shape == (args.n_train, 3)
    assert q_train.min() == 0
    assert q_train.max() == args.n_quantized - 1
    assert q_train.shape == (args.n_train,)
    assert q_test.shape == (args.n_test,)

    return xyz_train, q_train, q_test


class Model:
    """Contains an HMM based on a true state sequence

    Args:
        quantized: A time series of discrete scalar measurements
        true_states: The true sequencs of 3d states
        args: The command line arguments

    """

    # pylint: disable = too-few-public-methods
    def __init__(self, quantized, true_states, resolution, n_quantized):

        n_t = len(quantized)
        assert len(true_states) == n_t

        # Make a map from quantized true_states to integer indices
        index_true = {}
        for float_state in true_states:
            key = tuple((float_state / resolution).astype(numpy.int32))
            if key not in index_true:
                index_true[key] = len(index_true)

        n_states = len(index_true)

        dok_state_state = scipy.sparse.dok_array((n_states, n_states))
        dok_measurement_state = scipy.sparse.dok_array((n_quantized, n_states))

        # Count transitions state <- state and measurement <- state
        old_index = index_true[tuple(
            (true_states[0] / resolution).astype(numpy.int32))]
        dok_measurement_state[quantized[0], old_index] += 1
        for t in range(1, len(true_states)):
            float_state = true_states[t]
            new_index = index_true[tuple(
                (float_state / resolution).astype(numpy.int32))]
            dok_state_state[new_index, old_index] += 1
            dok_measurement_state[quantized[t], new_index] += 1
            old_index = new_index

        # Normalize to get probabilities from counts and translate to csr
        # total = numpy.empty(n_states) out=total in sum does not work
        # Sum m[i,j] over i, ie, new_index, to get n[1,j]
        total = dok_state_state.sum(axis=0)
        assert total.shape == (n_states,)
        self.p_state_state = dok_state_state.multiply(1 / total).tocsr()
        assert self.p_state_state.shape == (n_states, n_states)

        self.p_state_0 = total / total.sum()
        assert self.p_state_0.shape == (n_states,)

        total = dok_measurement_state.sum(axis=0)
        self.p_measurement_state = dok_measurement_state.multiply(
            1 / total).tocsr()
        assert self.p_measurement_state.shape == (n_quantized, n_states)

    def log_likelihood(self, quantized):
        """Calculate the log likelihood of the model self

        Args:
            quantized: A sequence of measurements

        Return: log(P(quantized|self))

        This is the forward algorithm using sparse arrays and not
        saving the alpha array.

        """
        _, n_states = self.p_measurement_state.shape
        # Because csr operations yield csr arrays, I simply start with
        # temp as a csr array
        temp = scipy.sparse.csr_array(self.p_state_0.reshape((1, n_states)))
        result = 0.0
        for y_t in quantized:
            # Now temp is the forecast or prior, ie, temp[0,i] = P(s[t]=i|y[:t])
            temp = self.p_measurement_state[[y_t]].multiply(temp)
            # getrow(y) is the likelihood, ie, getrow(y)[i] =
            # P(y[t]|s[t]=i)
            assert temp.shape == (1, n_states)
            # Now temp[0,i] = P(y[t], s[t]=i|y[:t])
            p_y = temp.sum()
            # p_y is P(y[t]|y[:t])
            result += numpy.log(p_y)
            temp = self.p_state_state.dot(temp.T / p_y).T
            assert temp.shape == (1, n_states)
            # Now temp is the new forecast, temp[i] =
            # P(s[t+1]=i|y[:t+1])
        return result


def main(argv=None):
    """Study cross entropy vs number of states

    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    result = {'args': args}
    xyz_train, q_train, q_test = make_data(args)
    for log_resolution in numpy.arange(*args.log_resolution):
        resolution = 2.0**log_resolution
        hmm = Model(q_train, xyz_train, resolution, args.n_quantized)
        log_likelihood = hmm.log_likelihood(q_test)
        n_states = len(hmm.p_state_0)
        print(
            f'log_resolution={log_resolution:4.1f}, resolution={resolution:5.1f} '
            + f'n_states={n_states:3d} log_likelihood={log_likelihood:6.1f}')
        result[log_resolution] = {
            'log_likelihood': log_likelihood,
            'n_states': n_states
        }

    with open(args.result, 'wb') as _file:
        pickle.dump(result, _file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
