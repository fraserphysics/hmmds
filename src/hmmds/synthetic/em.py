r"""em.py makes data for Fig. 2.7, fig:GaussMix, ,"Two iterations of
the EM Algorithm."

It fits a Gaussian mixture model with two means and fixed variance to
simulated data.

In the book the mixture parameter is $\lambda$.  Because lambda is a
reserved word in python, I use alpha for the mixture parameter here.

"""
import sys
import argparse
import pickle

import numpy
import numpy.random


def parse_args(argv):  # pylint: disable=missing-function-docstring

    parser = argparse.ArgumentParser(
        description="Make data for illustrating EM algorithm")
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--n_iterations', type=int, default=2)
    parser.add_argument('out_path', type=str)
    parser.add_argument('--print',
                        action='store_true',
                        help='Print intermediate results for degugging')
    return parser.parse_args(argv)


def main(argv=None):
    """Call: python em.py em.pickle
    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    rng = numpy.random.default_rng(args.random_seed)

    def _print(*dummy, **kwargs):
        """Print to stdout if "--print" was on command line.
        """
        if args.print:
            print(*dummy, **kwargs)

    mu = [-2.0, 2.0]  # Means of the two components
    y_sequence = []  # To hold the sequence of observations
    n_y = 10  # Number of observations
    em_iterations = args.n_iterations

    # Create the data
    _print('# Here are the observations')
    for t in range(n_y):
        state = rng.choice([0, 1])
        y_sequence.append(rng.normal(mu[state],
                                     1.0))  # Draw from \Normal(mu[s], 1)

    _print((n_y * '%5.2f ') % tuple(y_sequence))

    # Initial model parameters
    alpha = [0.5]
    mu_i = [[-1.0, 1.0]]

    for i in range(em_iterations):
        _print(
            f'i={i} alpha={alpha[i]:6.3f} mu0={mu_i[i][0]:6.3f} mu1={mu_i[i][1]:6.3f}\n'
        )
        state_probs = []  # State probabilities
        sums = 0.0  # Sum_t prob(class_0|Y[t])
        # Estimate state probabilities for each observation
        for t in range(n_y):
            p0t = (alpha[i] / numpy.sqrt(2 * numpy.pi)) * numpy.exp(
                (-(y_sequence[t] - mu_i[i][0])**2) / 2)
            p1t = ((1 - alpha[i]) / numpy.sqrt(2 * numpy.pi)) * numpy.exp(
                (-(y_sequence[t] - mu_i[i][1])**2) / 2)
            state_probs.append(p0t / (p0t + p1t))
            sums += state_probs[t]
        _print('w(t)        ',
               (len(state_probs) * '%5.2f ') % tuple(state_probs))
        _print('w(t)y(t)    ', (len(state_probs) * '%5.2f ') %
               tuple(y_sequence[t] * state_probs[t] for t in range(n_y)))
        _print('(1-w(t))y(t)', (len(state_probs) * '%5.2f ') %
               tuple(y_sequence[t] * (1 - state_probs[t]) for t in range(n_y)))
        # Reestimate model parameters mu_i and alpha
        sum0 = 0.0
        sum1 = 0.0
        for t in range(n_y):
            sum0 += y_sequence[t] * state_probs[t]
            sum1 += y_sequence[t] * (1.0 - state_probs[t])
        alpha.append(sums / n_y)
        mu_i.append([sum0 / sums, sum1 / (n_y - sums)])
    _print(
        f'i={i} alpha={alpha[-1]:6.3f} mu0={mu_i[-1][0]:6.3f} mu1={mu_i[-1][1]:6.3f}\n'
    )

    mu_i.append(mu)  # Record initial model
    alpha.append(0.5)  # Record initial model
    with open(args.out_path, 'wb') as _file:
        pickle.dump({
            'Y': y_sequence,
            'alpha': alpha,
            'mu_i': mu_i
        },
                    _file,
                    protocol=2)


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
