"""mimic.py Make linear_simulation.py exercise all methods of EKF.

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

    Wrap under, a LinearGaussian instance, in system, an Linear
    instance.  Then wrap system in result, an EKF instance.  The goal
    is to get linear_simulation.main to exercise all of the EKF
    methods.

    """

    under, stationary_distribution = linear_simulation.make_linear_gaussian(
        args, d_t, rng)
    system = hmm.examples.ekf.Linear(under, d_t)
    result = hmm.state_space.EKF(system, d_t, None)  # rng in EKF not used
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
