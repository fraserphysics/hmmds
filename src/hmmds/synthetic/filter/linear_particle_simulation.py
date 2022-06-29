"""linear_particle_simulation.py: Demonstrate particle filter on linear Gaussian system

Imitate linear_map_simulation.py, but only do forward filter.
"""
from __future__ import annotations  # Enables, eg, (self: System

import pickle
import sys

import numpy
import numpy.random

import hmm.state_space
import hmm.particle
from hmmds.synthetic.filter import linear_map_simulation


# Not worth reducing locals from 23 to 20 pylint: disable = too-many-locals
def main(argv=None,
         make_system=linear_map_simulation.make_linear_stationary,
         additional_args=(linear_map_simulation.system_args,)):
    """
    x_{t+1} = A x_t + B V_n

    y_t = C x_t + D W_n
    """

    # For making system with x_dim = 2 and y_dim = 1
    #

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = linear_map_simulation.parse_args(argv, additional_args)

    rng = numpy.random.default_rng(args.random_seed)

    dt_fine = 2 * numpy.pi / (args.omega * args.sample_rate)
    dt_coarse = dt_fine * args.sample_ratio

    system_fine, initial_fine = make_system(args, dt_fine, rng)
    system_coarse, initial_coarse = make_system(args, dt_coarse, rng)

    x_fine, y_fine = system_fine.simulate_n_steps(initial_fine, args.n_fine)
    x_coarse, y_coarse = system_coarse.simulate_n_steps(initial_coarse,
                                                        args.n_coarse)

    system = hmm.particle.LinearSystem(
        system_coarse.state_map, system_coarse.state_noise_covariance,
        system_coarse.observation_map,
        system_coarse.observation_noise_covariance, initial_coarse.mean,
        initial_coarse.covariance, rng)
    n_times = len(y_coarse)
    n_particles = numpy.ones(n_times, dtype=int) * 100
    n_particles[0:3] *= 10

    # Like lorenz_particle_simulation, pylint: disable = duplicate-code
    _, forward_means, forward_covariances, log_likelihood = system.forward_filter(
        y_coarse, n_particles, threshold=0.5)
    print(f"log_likelihood: {log_likelihood}")

    with open(args.data, 'wb') as _file:
        pickle.dump(
            {
                'dt_fine': dt_fine,
                'dt_coarse': dt_coarse,
                'x_fine': x_fine,
                'y_fine': y_fine,
                'x_coarse': x_coarse,
                'y_coarse': y_coarse,
                'forward_means': forward_means,
                'forward_covariances': forward_covariances,
            }, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# mode:python
# End:
