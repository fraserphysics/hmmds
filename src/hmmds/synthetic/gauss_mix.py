r"""gauss_mix.py makes data for fig:GaussMix.

python gauss_mix.py out_dir

It fits a Gaussian mixture model with two means and fixed variance to
simulated data.  Write the following files:

out_dir/gauss_mix.pkl          Values of alpha and the vector mu for each
                                    EM iteration

out_dir/gauss_mix_theta.tex    A LaTeX table of parameter values for each
                                 EM iteration

out_dir/gauss_mix_weights.tex  A LaTeX table of weighted y values for each
                                 EM iteration

In the book the mixture parameter is $\lambda$.  Because lambda is a
reserved word in python, I use alpha for the mixture parameter here.

"""
import sys
import argparse
import pickle
import os

import numpy
import numpy.random


def parse_args(argv):  # pylint: disable=missing-function-docstring

    parser = argparse.ArgumentParser(
        description="Make data for illustrating EM algorithm")
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--em_iterations', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('out_dir', type=str)
    return parser.parse_args(argv)


def simulate(rng, alpha, mu, n_samples):
    """ Draw samples from the specified mixture model
    """
    y_sequence = []
    for t in range(n_samples):
        state = rng.choice([0, 1], p=[alpha, 1 - alpha])
        y_sequence.append(rng.normal(mu[state],
                                     1.0))  # Draw from \Normal(mu[s], 1)
    return numpy.array(y_sequence)


def em_step(y_sequence, alpha, mu, weights):
    """ One step of EM iteration
    """
    sums = 0.0  # Sum_t prob(class_0|Y[t])
    # Estimate state probabilities for each observation
    for i, y in enumerate(y_sequence):
        p0t = (alpha / numpy.sqrt(2 * numpy.pi)) * numpy.exp(
            (-(y - mu[0])**2) / 2)
        p1t = ((1 - alpha) / numpy.sqrt(2 * numpy.pi)) * numpy.exp(
            (-(y - mu[1])**2) / 2)
        weights[i] = p0t / (p0t + p1t)
        sums += weights[i]
    # Reestimate model parameters mu and alpha
    sum0 = 0.0
    sum1 = 0.0
    for prob, y in zip(weights, y_sequence):
        sum0 += y * prob
        sum1 += y * (1.0 - prob)
    new_alpha = sums / len(y_sequence)
    new_mu = [sum0 / sums, sum1 / (len(y_sequence) - sums)]
    return new_alpha, new_mu


def write_theta_tex(alpha, mu, _file):
    """ Write in latex tabular format
    """
    _file.write('''\\begin{tabular}{|l|d{4.2}*{2}{d{5.2}}|}
 \ch{} & \multicolumn{1}{r}{$\lambda$}
    & \multicolumn{1}{r}{$\mu_1$}
    & \multicolumn{1}{r}{$\mu_2$} \\\\ \hline
''')
    for i, key in (
        (-1, 'theta_{\\text{true}}'),
        (0, 'theta(1)'),
        (1, 'theta(2)'),
        (2, 'theta(3)'),
    ):
        _file.write(
            f'$\\{key}$ & {alpha[i]:5.2f} & {mu[i][0]:5.2f} & {mu[i][1]:5.2f} \\\\\n'
        )
    _file.write('\hline \end{tabular}\n')


def write_weights_tex(weights, y_sequence, _file):
    """ Write in latex tabular format
    """
    _file.write('''
    \def\cs{\hspace{0.10em}}%
    \def\st{\\rule{0pt}{2.25ex}}%
    \\begin{tabular}{r@{}|c@{\cs}|*{10}{@{\cs}.@{\cs}|}}
    \ch{} & \ch{$t$} & \ch{1} & \ch{2} & \ch{3} & \ch{4} & \ch{5} & \ch{6} & \ch{7} & \ch{8} & \ch{9} & \ch{10} \\\\ \cline{2-12}
''')

    def write_row(key, values, extra=None):
        if extra:
            _file.write(f'{extra}& {key}')
        else:
            _file.write(f'& {key}')
        for value in values:
            _file.write(f'& {value:4.2f}')
        _file.write('\\\\\n')

    write_row('$\\ti{y}{t}$', y_sequence)
    _file.write('\cline{2-12} \n')
    for i in range(2):
        write_row('$\\ti{w}{t}$', weights[i])
        write_row('$\\ti{w}{t}\\ti{y}{t}$',
                  weights[i] * y_sequence,
                  extra='$\\ti{\\theta}' + f'{i+1}$')
        write_row('$(1-\\ti{w}{t})\\ti{y}{t}$', (1 - weights[i]) * y_sequence)
        _file.write('\cline{2-12} \n')
    _file.write('\end{tabular}\n')


def main(argv=None):
    """Call: python gauss_mix.py gauss_mix.pickle
    """

    # This is repetitive boilerplate, pylint: disable = duplicate-code
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    rng = numpy.random.default_rng(args.random_seed)

    true_mu = [-2.0, 2.0]  # Means of the two components
    true_alpha = .5  # Probability of first mixture component

    weights = numpy.empty((args.em_iterations, args.n_samples))
    y_sequence = simulate(rng, true_alpha, true_mu, args.n_samples)

    # Initial model parameters
    alpha = [0.5]
    mu = [[-1.0, 1.0]]

    for iteration in range(args.em_iterations):
        new_alpha, new_mu = em_step(y_sequence, alpha[-1], mu[-1],
                                    weights[iteration, :])
        alpha.append(new_alpha)
        mu.append(new_mu)
    mu.append(true_mu)  # Record initial model
    alpha.append(true_alpha)  # Record initial model

    with open(os.path.join(args.out_dir, 'gauss_mix_theta.tex'),
              encoding='utf-8',
              mode='w') as _file:
        write_theta_tex(alpha, mu, _file)

    with open(os.path.join(args.out_dir, 'gauss_mix_weights.tex'),
              encoding='utf-8',
              mode='w') as _file:
        write_weights_tex(weights, y_sequence, _file)

    with open(os.path.join(args.out_dir, 'gauss_mix.pkl'), 'wb') as _file:
        pickle.dump({'Y': y_sequence, 'alpha': alpha, 'mu': mu}, _file)


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
