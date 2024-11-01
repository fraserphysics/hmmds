""" scalar_gaussian.py: Make and use HMMs with scalar gaussian observations

Use: python scalar_gaussian.py tex_path data_dir

In data_dir writes the following files:

data_dir/SGO: Empty flag indicating success
data_dir/SGO_sim: Text with columns: time simulated_state simulated_observation decoded_state
data_dir/SGO_models: Text descriptions of models
tex_path: LaTeX \defs

"""

import sys
import os.path
import argparse

import numpy
import numpy.random

import hmm.base
import hmm.observe_float


def parse_args(argv):  # pylint: disable=missing-function-docstring

    parser = argparse.ArgumentParser(
        description="Make data for illustrating scalar gaussian observations")
    parser.add_argument('--time_steps',
                        type=int,
                        default=100,
                        help='length of time series')
    parser.add_argument('--iterations_good',
                        type=int,
                        default=50,
                        help='training iterations for good model')
    parser.add_argument('--iterations_fail',
                        type=int,
                        default=10,
                        help='training iterations for bad start')
    parser.add_argument('--random_seed',
                        type=int,
                        default=3,
                        help='Seed for random number generator')
    parser.add_argument('tex_path',
                        type=str,
                        help='For strings to include in LaTeX')
    parser.add_argument('out_dir', type=str, help='Data for plotting')
    return parser.parse_args(argv)


def make_2a(rng):
    """Make HMM for plot a in fig:ScalarGaussian
    """
    p_ss = numpy.array([[0.93, 0.07], [0.13, 0.87]])
    p_s0 = numpy.array([13. / 20, 7. / 20.])
    mu = numpy.array([-1.0, 1.0])
    var = numpy.ones(2)
    return hmm.base.HMM(p_s0, p_s0, p_ss, hmm.observe_float.Gauss(mu, var, rng),
                        rng)


def make_2e(rng):
    """Make HMM for plot e in fig:ScalarGaussian
    """
    p_ss = numpy.array([[0.5, 0.5], [0.5, 0.5]])
    p_s0 = numpy.array([0.5, 0.5])
    mu = numpy.array([-2.0, 2.0])
    var = numpy.ones(2) * 2
    return hmm.base.HMM(p_s0, p_s0, p_ss, hmm.observe_float.Gauss(mu, var, rng),
                        rng)


def make_3a(rng):
    """Make HMM for plot a in fig:MLEfail
    """
    p_ss = numpy.array([[0.5, 0.5], [0.5, 0.5]])
    p_s0 = numpy.array([0.5, 0.5])
    mu = numpy.array([0.0, 3.6])
    var = numpy.array([4.0**2, 0.5**2])
    return hmm.base.HMM(p_s0, p_s0, p_ss, hmm.observe_float.Gauss(mu, var, rng),
                        rng)


def write_latex(hmm, prefix, file_):
    """Write \defs for inclusion in LaTeX

    Args:
        hmm: An HMM with Gauss observation models
        prefix: Eg, MLEfaila
        file_: A file object open for writing

    """
    for key, value in (
        ('Paa', hmm.p_state2state[0, 0]),
        ('Pba', hmm.p_state2state[1, 0]),
        ('Pab', hmm.p_state2state[0, 1]),
        ('Pbb', hmm.p_state2state[1, 1]),
        ('mua', hmm.y_mod.mu[0]),
        ('mub', hmm.y_mod.mu[1]),
        ('vara', hmm.y_mod.variance[0]),
        ('varb', hmm.y_mod.variance[1]),
    ):
        if abs(value) > 0.01:
            file_.write(f'\def\{prefix}{key}{{{value:4.2f}}}\n')
        else:
            file_.write(f'\def\{prefix}{key}{{{value:.1e}}}\n')


def main(argv=None):
    """Data for Fig. 3.2: "An HMM with scalar Gaussian observations".

    """

    # This is repetitive boilerplate, pylint: disable = duplicate-code
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    rng = numpy.random.default_rng(args.random_seed)

    model_2a = make_2a(rng)
    model_2e = make_2e(rng)
    model_3a = make_3a(rng)

    with open(args.tex_path, 'w', encoding='utf-8') as file_:
        write_latex(model_2a, 'ScalarGaussiana', file_)
        write_latex(model_2e, 'ScalarGaussiane', file_)
        write_latex(model_3a, 'MLEfaila', file_)

    with open(os.path.join(args.out_dir, 'SGO_train'), 'w',
              encoding='utf-8') as sgo_train:
        sgo_train.write(f'initial model_2a=\n{model_2a}\n')
        sgo_train.write(f'initial model_2e=\n{model_2e}\n')
        sgo_train.write(f'initial model_3a=\n{model_3a}\n')
    state_sequence, temp = model_2a.simulate(args.time_steps)
    y_sequence = (numpy.array(temp, numpy.float64),)
    estimated_states = model_2a.decode(y_sequence)

    model_2e.train(y_sequence, n_iterations=args.iterations_good, display=False)
    model_3a.train(y_sequence, n_iterations=args.iterations_fail, display=True)
    differences = numpy.abs(y_sequence[0] - model_3a.y_mod.mu[1])
    compare = (differences < 1e-3).nonzero()[0]
    assert compare.shape == (1,)

    with open(os.path.join(args.out_dir, 'SGO_sim'), encoding='utf-8',
              mode='w') as sgo_sim:
        for t in range(args.time_steps):
            sgo_sim.write(
                f'{t:2d} {state_sequence[t]:1d} {y_sequence[0][t]:7.3f} {estimated_states[t]:1d}\n'
            )

    with open(os.path.join(args.out_dir, 'SGO_train'), 'a',
              encoding='utf-8') as sgo_train:
        sgo_train.write(
            f'model_2e after {args.iterations_good} training iterations=\n{model_2e}\n'
        )
        sgo_train.write(
            f'\nmodel_3a after {args.iterations_fail} training iterations=\n{model_3a}'
        )

    with open(args.tex_path, 'a', encoding='utf-8') as file_:
        file_.write(f'\def\MLEfailIterations{{{args.iterations_fail}}}\n')
        file_.write(f'\def\MLEfailt{{{compare[0]}}}\n')
        write_latex(model_2e, 'ScalarGaussianf', file_)
        write_latex(model_3a, 'MLEfailb', file_)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:
