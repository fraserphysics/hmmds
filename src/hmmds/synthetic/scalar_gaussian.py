""" ScalarGaussian.py data_dir

Writes file SGO_sim with the following columns:

    time simulated_state simulated_observation decoded_state

Writes file SGO_train containing:

    Two trained models

"""

import sys
import os.path

import numpy
import numpy.random

import hmm.base
import hmm.observe_float

rng = numpy.random.default_rng(4)

p_ss = numpy.array([[0.93, 0.07], [0.13, 0.87]])
p_s0 = numpy.array([13. / 20, 7. / 20.])
mu = numpy.array([-1.0, 1.0])
var = numpy.ones(2)
model_2a = hmm.base.HMM(p_s0, p_s0, p_ss, hmm.observe_float.Gauss(mu, var, rng),
                        rng)

p_ss = numpy.array([[0.5, 0.5], [0.5, 0.5]])
p_s0 = numpy.array([0.5, 0.5])
mu = numpy.array([-2.0, 2.0])
var = numpy.ones(2) * 2
model_2e = hmm.base.HMM(p_s0, p_s0, p_ss, hmm.observe_float.Gauss(mu, var, rng),
                        rng)

p_ss = numpy.array([[0.5, 0.5], [0.5, 0.5]])
p_s0 = numpy.array([0.5, 0.5])
mu = numpy.array([0.0, 3.6])
var = numpy.array([4.0**2, 0.5**2])
model_3a = hmm.base.HMM(p_s0, p_s0, p_ss, hmm.observe_float.Gauss(mu, var, rng),
                        rng)


def main(argv=None):
    """Data for Fig. 3.2: "An HMM with scalar Gaussian observations".

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]
    data_dir = argv[0]

    n_t = 100

    state_sequence, temp = model_2a.simulate(n_t)
    y_sequence = (numpy.array(temp, numpy.float64),)

    estimated_states = model_2a.decode(y_sequence)
    model_2e.train(y_sequence, n_iterations=50, display=False)
    model_3a.train(y_sequence, n_iterations=5, display=False)

    with open(os.path.join(data_dir, 'SGO_sim'), encoding='utf-8',
              mode='w') as sgo_sim:
        for t in range(n_t):
            sgo_sim.write(
                f'{t:2d} {state_sequence[t]:1d} {y_sequence[0][t]:7.3f} {estimated_states[t]:1d}\n'
            )

    with open(os.path.join(data_dir, 'SGO_train'), 'w',
              encoding='utf-8') as sgo_train:
        sgo_train.write('model_2e after 50 training iterations=\n{model_2e}\n')
        sgo_train.write(
            f'\nmodel_3a after 2 training iterations=\n{model_3a}\n')

    return 0


if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:
