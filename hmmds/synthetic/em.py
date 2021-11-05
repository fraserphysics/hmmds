''' em.py makes data for Fig. 2.7, fig:GaussMix, of the book
'''
import sys
import argparse
import pickle

import numpy
import numpy.random


def main(argv=None):
    """Call: python em.py em.pickle
    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Make data for illustrating EM algorithm")
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--n_iterations', type=int, default=2)
    parser.add_argument('out_path', type=str)
    parser.add_argument('--print', action='store_true')
    args = parser.parse_args(argv)
    rng = numpy.random.default_rng(args.random_seed)

    def _print(*dummy, **kwargs):
        if args.print:
            print(*dummy, **kwargs)

    mu = [-2.0, 2.0]  # Means of the two components
    Y = []  # To hold the sequence of observations
    T = 10  # Number of observations
    em_iterations = args.n_iterations

    # Create the data
    _print('# Here are the observations')
    for t in range(T):
        s = rng.choice([0, 1])
        y = rng.normal(mu[s], 1.0)  # Draw from \Normal(mu[s], 1)
        Y.append(y)
    _print((T * '%5.2f ') % tuple(Y))

    # Initial model parameters
    alpha = [0.5]
    mu_i = [[-1.0, 1.0]]

    for i in range(em_iterations):
        _print('i=%d alpha=%6.3f mu0=%6.3f mu1=%6.3f\n' %
               (i, alpha[i], mu_i[i][0], mu_i[i][1]))
        ps = []  # State probabilities
        sums = 0.0  # Sum_t prob(class_0|Y[t])
        # Estimate state probabilities for each observation
        for t in range(T):
            p0t = (alpha[i] / numpy.sqrt(2 * numpy.pi)) * numpy.exp(
                (-(Y[t] - mu_i[i][0])**2) / 2)
            p1t = ((1 - alpha[i]) / numpy.sqrt(2 * numpy.pi)) * numpy.exp(
                (-(Y[t] - mu_i[i][1])**2) / 2)
            ps.append(p0t / (p0t + p1t))
            sums += ps[t]
        _print('w(t)        ', (len(ps) * '%5.2f ') % tuple(ps))
        _print('w(t)y(t)    ',
               (len(ps) * '%5.2f ') % tuple(Y[t] * ps[t] for t in range(T)))
        _print('(1-w(t))y(t)', (len(ps) * '%5.2f ') %
               tuple(Y[t] * (1 - ps[t]) for t in range(T)))
        # Reestimate model parameters mu_i and alpha
        sum0 = 0.0
        sum1 = 0.0
        for t in range(T):
            sum0 += Y[t] * ps[t]
            sum1 += Y[t] * (1.0 - ps[t])
        alpha.append(sums / T)
        mu_i.append([sum0 / sums, sum1 / (T - sums)])
    _print('i=%d alpha=%6.3f mu0=%6.3f mu1=%6.3f\n' %
           (em_iterations, alpha[-1], mu_i[-1][0], mu_i[-1][1]))

    mu_i.append(mu)  # Record initial model
    alpha.append(0.5)  # Record initial model
    pickle.dump({
        'Y': Y,
        'alpha': alpha,
        'mu_i': mu_i
    },
                open(args.out_path, 'wb'),
                protocol=2)


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
