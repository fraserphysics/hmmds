"""mimic.py Imitate linear_simulation.py but use EKF

"""
import sys

import hmm.examples.ekf
import hmm.state_space

import linear_simulation


def make_system(args, d_t, rng):
    """Make a system instance
    
    Args:
        args: Command line arguments
        d_t: Sample interval
        rng:

    Returns:
        (A system instance, an initial state, an inital distribution)

    """

    under, stationary_distribution = linear_simulation.make_system(
        args, d_t, rng)
    system = hmm.examples.ekf.Linear(under, d_t)
    result = hmm.state_space.EKF(system, d_t, rng)
    assert isinstance(result.system.under, hmm.state_space.LinearGaussian)
    assert isinstance(result.system, hmm.examples.ekf.Linear)
    assert isinstance(result, hmm.state_space.EKF)
    return result, stationary_distribution


def main():
    """
    """
    args = linear_simulation.parse_args(sys.argv[1:])
    return linear_simulation.main(sys.argv[1:], make_system=make_system)


if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# mode:python
# End:
