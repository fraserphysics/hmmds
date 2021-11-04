''' ScalarGaussian.py data_dir

Writes file SGO_sim with the following columns:

    time simulated_state simulated_observation decoded_state

Writes file SGO_train containing:

    Two trained models

'''

import sys
import os.path

import numpy
import numpy.random

import hmm.base
import hmm.observe_float

rng = numpy.random.default_rng(4)

P_SS = [[0.93, 0.07], [0.13, 0.87]]
P_S0 = [13. / 20, 7. / 20.]
mu = [-1.0, 1.0]
var = numpy.ones(2)
model_2a = hmm.base.HMM(P_S0, P_S0, P_SS, hmm.observe_float.Gauss(mu, var, rng),
                        rng)

P_SS = [[0.5, 0.5], [0.5, 0.5]]
P_S0 = [0.5, 0.5]
mu = [-2.0, 2.0]
var = numpy.ones(2) * 2
model_2e = hmm.base.HMM(P_S0, P_S0, P_SS, hmm.observe_float.Gauss(mu, var, rng),
                        rng)

P_SS = [[0.5, 0.5], [0.5, 0.5]]
P_S0 = [0.5, 0.5]
mu = [0.0, 3.6]
var = numpy.array([4.0**2, 0.126**2])
model_3a = hmm.base.HMM(P_S0, P_S0, P_SS, hmm.observe_float.Gauss(mu, var, rng),
                        rng)


def main(argv=None):

    if argv is None:  # Usual case
        argv = sys.argv[1:]
    data_dir = argv[0]
    T = 100
    s, y = model_2a.simulate(T)
    y = (numpy.array(y, numpy.float64),)
    s_hat = model_2a.decode(y)

    f = open(os.path.join(data_dir, 'SGO_sim'), 'w')
    for t in range(T):
        print('%2d %1d %7.3f %1d' % (t, s[t], y[0][t], s_hat[t]), file=f)
    model_2e.train(y, n_iterations=50, display=False)
    f = open(os.path.join(data_dir, 'SGO_train'), 'w')
    print('model_2e after 50 training iterations=\n%s' % model_2e, file=f)
    model_3a.train(y, n_iterations=6, display=False)
    print('\nmodel_3a after 6 training iterations=\n%s' % model_3a, file=f)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:
