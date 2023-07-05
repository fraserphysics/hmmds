"""develop.py Use this file for developing code.  Once code is stable
move it to other files.  That protocol permits directing gnu-make to
re-make only what's necessary.  """

from __future__ import annotations  # Enables, eg, (self: HMM,

# Some of this is dead code.

import sys
import os.path
import pickle
import argparse

import numpy
import sortedcontainers
import anytree

import hmm.base
import hmm.C

import hmmds.applications.apnea.utilities


# FixMe: Not using hmm.C.HMM_SPARSE because: 1. It's not much faster
# in training for ~600 state apnea models 2. It's method forward
# crashes.  3. It doesn't have a decode method
class HMM(hmm.C.HMM):  #HMM_SPARSE or HMM
    """Holds state transition probabilities constant

    """

    def __init__(self: HMM,
                 *args,
                 untrainable_indices=None,
                 untrainable_values=None):
        """Option of holding some elements of p_state2state constant
        in reestimation.

        """
        hmm.C.HMM.__init__(self, *args)
        self.untrainable_indices = untrainable_indices
        self.untrainable_values = untrainable_values

    def reestimate(self: HMM):
        """Variant can hold some self.p_state2state values constant.

        Reestimates observation model parameters.

        """

        hmm.C.HMM.reestimate(self)
        if self.untrainable_indices is None or len(
                self.untrainable_indices) == 0:
            return
        self.p_state2state[self.untrainable_indices] = self.untrainable_values
        return

    def likelihood(self: HMM, y) -> numpy.ndarray:
        """Calculate p(y[t]|y[:t]) for t < len(y)

        Args:
            y: A single segment appropriate for self.y_mod.observe([y])

        """
        self.y_mod.observe([y])
        state_likelihood = self.y_mod.calculate()
        length = len(state_likelihood)  # Less than len(y) if y_mod is
        # autoregressive
        result = numpy.empty(length)
        last = numpy.copy(self.p_state_initial)
        for t in range(length):
            last *= state_likelihood[t]
            last_sum = last.sum()  # Probability of y[t]|y[:t]
            result[t] = last_sum
            if last_sum > 0.0:
                last /= last_sum
            else:
                print(f'Zero likelihood at {t=}.  Reset.')
                last = numpy.copy(self.p_state_initial)
            self.p_state2state.step_forward(last)
        return result

    # Also want relative weighting of slow and fast data
    def class_estimate(self: HMM, y: list, more_specific: float = 1) -> list:
        """ Estimate a sequence of classes
        Args:
            y: List with single element that is time series of measurements
            more_specific: >1 increase the number of normal minutes

        Returns:
            Time series of class identifiers
        """
        class_model = self.y_mod['class']
        del self.y_mod['class']
        t_seg = self.y_mod.observe(y)
        self.n_times = self.y_mod.n_times
        assert len(t_seg) == 2
        assert t_seg[-1] == self.n_times

        self.alpha = numpy.empty((self.n_times, self.n_states))
        self.beta = numpy.empty((self.n_times, self.n_states))
        self.gamma_inv = numpy.empty((self.n_times,))
        self.state_likelihood = self.y_mod.calculate()
        assert self.state_likelihood[0, :].sum() > 0.0
        log_likelihood = self.forward()
        self.backward()
        weights = self.alpha * self.beta

        def weights_per_minute(state_list, samples_per_minute=40):
            minutes = self.n_times // samples_per_minute
            remainder = self.n_times % samples_per_minute
            if remainder == 0:
                result = weights[:, state_list].sum(axis=1).reshape(
                    -1, samples_per_minute).sum(axis=1)
            else:
                result = weights[:-remainder, state_list].sum(axis=1).reshape(
                    -1, samples_per_minute).sum(axis=1)
            assert result.shape == (minutes,)
            return result

        weights_normal = weights_per_minute(class_model.class2state[0])
        weights_apnea = weights_per_minute(class_model.class2state[1])
        result = weights_apnea > weights_normal * more_specific
        self.y_mod['class'] = class_model  # Restore for future use
        return result


class ItemScoreT:
    """For list that is sorted by a function of score and t

    Args:
        number: Value for sorting, ie, score + more_specific*t
        node: Pointer to associated FixMe (was BundleNode)
        reference: Key for list sorted by score
    """

    def __init__(self, number: float, node, reference: float):
        assert isinstance(reference, float)
        self.number = number
        self.node = node
        self.reference = reference
        self.key = (number, node)

    def __eq__(self, other):
        return self.key == other.key

    def __lt__(self, other):
        return self.key[0] < other.key[0]


def main(argv=None):
    """ Put small tests here
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    nodes_by_priority = sortedcontainers.SortedList()
    return 0


if __name__ == "__main__":
    sys.exit(main())
