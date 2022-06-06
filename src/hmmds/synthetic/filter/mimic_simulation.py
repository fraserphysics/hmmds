"""mimic_simulation.py Make linear_simulation.py exercise all methods
of NonStationary.

"""
import sys

import hmm.examples.ekf
import hmm.state_space

import linear_map_simulation


def make_system(args, dt, rng):
    """Make a system instance
    
    Args:
        args: Command line arguments
        dt: Sample interval
        rng:

    Returns:
        (A system instance, an initial state, an inital distribution)

    Wrap under, a LinearGaussian instance, in system, an Linear
    instance.  Then wrap system in result, an EKF instance.  The goal
    is to get linear_map_simulation.main to exercise all of the
    NonStationary methods.

    """

    under, stationary_distribution = linear_map_simulation.make_linear_stationary(
        args, dt, rng)
    system = hmm.examples.ekf.Linear(under, dt)
    result = hmm.state_space.NonStationary(
        system, dt, None)  # rng in Nonstationary not used
    assert isinstance(result.system.under, hmm.state_space.LinearStationary)
    assert isinstance(result.system, hmm.examples.ekf.Linear)
    assert isinstance(result, hmm.state_space.NonStationary)
    return result, stationary_distribution


def main():
    """
    """
    return linear_map_simulation.main(sys.argv[1:], make_system=make_system)


if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# mode:python
# End:
