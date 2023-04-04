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
import observation


class HMM(hmm.C.HMM):
    """Holds state transition probabilities constant

    """

    def reestimate(self: HMM):
        """Variant that holds self.p_state2state constant.

        Reestimates observation model parameters.

        """

        self.alpha *= self.beta  # Saves allocating a new array for
        alpha_beta = self.alpha  # the result

        self.p_state_time_average = alpha_beta.sum(axis=0)  # type: ignore
        self.p_state_initial = numpy.copy(alpha_beta[0])
        for x in (self.p_state_time_average, self.p_state_initial):
            x /= x.sum()
        self.y_mod.reestimate(alpha_beta)

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

    # Dead code
    def _bundle_decode(self: HMM,
                       y: list,
                       fudge: float = 0.03,
                       power: float = 0.5,
                       max_leaves: int = int(1e4)) -> list:
        """Search for a high likelihood bundle sequence

        Args:
            y: List with single element that is time series of measurements
            fudge: Preference given to longer sequences for extension
            power: Preference given to longer sequences for extension
            max_leaves: Termination criterion

        Returns:
            Time series of bundle identifiers

        Need large value of fudge to get code to finish with
        reasonable value of max_leaves.  With a large value of
        max_leaves, the complexity of this code is exponential in the
        length of the data.

        This is sort of Viterbi decoding for bundles.  It's better
        than bundle_weight if it's important to get sequence correct,
        but if correct classification at each time is more important
        use bundle_weight.  In Viterbi, I could track relative
        probabilities because all trajectories were compared with the
        same length.  To compare trajectories of different lengths and
        prevent underflow, I use log probability for score.

        """

        # bundle2state_dict[bundle] = list of constituent states
        bundle2state_dict = self.y_mod.bundle2state
        n_bundles = self.y_mod.n_bundle
        # bundle_and_state[bundle, state] = True iff state \in bundle
        bundle_and_state = self.y_mod.bundle_and_state

        # Take the data and calculate likelihood of states
        assert len(y) == 1
        t_seg = self.y_mod.underlying_model.observe(y)
        n_times = t_seg[-1]
        likelihood = self.y_mod.underlying_model.calculate()

        # Leaves sorted by score + fudge * t ** power
        leaves_by_priority = sortedcontainers.SortedList()
        # Sorted scores of leaves
        leaves_by_score = sortedcontainers.SortedList()

        def remove_from_lists(node: BundleNode):
            """Remove entries in the lists leaves_by_priority and leaves_by_score

            Args:
                node: Former leaf node

            This is called when children are attached to node.

            """
            priority_item = node.priority_item
            score_item = priority_item.reference
            leaves_by_priority.remove(priority_item)
            leaves_by_score.remove(score_item)

        def propagate(node):
            t = node.depth  # Length of bundle sequences for children
            children = []
            # P(s(t+1), y(t+1)|c_0^t, y_0^t)
            p_state_and_y = numpy.dot(self.p_state2state,
                                      node.p_state) * likelihood[t]
            # P(s(t+1)|c_0^t, y_0^{t+1})
            p_state = p_state_and_y / p_state_and_y.sum()

            for b_child in range(n_bundles):
                # For sequence b_0^t that ends in b_child, calculate
                # conditional distribution of states, p_state, and
                # score = log(prob(b_0^t|y_0^t)

                # P(c(t+1), s(t+1) |c_0^t, y_0^{t+1})
                p_state_and_child = p_state * bundle_and_state[b_child, :]

                # P(c(t+1)|c_0^t, y_0^{t+1})
                p_child = p_state_and_child.sum()
                if p_child == 0:  # Child is not possible
                    continue

                # P(s(t+1)|c_0^{t+1}, y_0^{t+1})
                p_state_given_child = p_state_and_child / p_child
                score = float(node.score + numpy.log(p_child))

                # Assign results to data structures
                child = BundleNode(b_child, score, p_state_given_child, node,
                                   None)
                priority_item = ItemScoreT(score + fudge * t**power, child,
                                           score)
                child.priority_item = priority_item  # Enable removal
                leaves_by_priority.add(priority_item)
                leaves_by_score.add(score)
                children.append(child)

            node.children = children
            return t, children

        # Set up for iterating
        score = 0.0
        bundle_id = -1
        parent = None
        children = None
        best_end = None  # Node at the end of a complete path
        root = BundleNode(bundle_id, score, self.p_state_time_average, parent,
                          children)
        # Put root in lists of leaves
        leaves_by_priority.add(ItemScoreT(score, root, score))
        leaves_by_score.add(score)

        while best_end is None or leaves_by_score[0] < best_end.score:
            n_leaves = len(leaves_by_priority)
            if n_leaves > max_leaves and best_end is not None:
                break
            if n_leaves > 10 * max_leaves:
                return None  # Indicate failure to caller
            leaf_score = leaves_by_priority.pop()
            leaves_by_score.remove(
                leaf_score.reference)  # Remove the corresponding item
            node = leaf_score.node

            if n_leaves % 1_000 == 0:
                print(
                    'n_leaves= {0} depth= {1} n_times={5} priority= {2:7.4f} range(priority) = {3:7.4f} {4:7.4f}'
                    .format(n_leaves, node.depth, leaf_score.number,
                            leaves_by_priority[0].number,
                            leaves_by_priority[-1].number, n_times))

            assert isinstance(node, BundleNode)
            t, children = propagate(node)
            if t + 1 < n_times:
                continue  # Sequence is incomplete
            # Each child represents a complete decoded bundle sequence
            for child in children:
                remove_from_lists(child)
                if best_end is None or child.score > best_end.score:
                    print(
                        """new best_end: n_leaves={0} score={1} worst={2} best={3}"""
                        .format(len(leaves_by_priority), child.score,
                                leaves_by_score[0], leaves_by_score[-1]))
                    best_end = child
        # Backtrack to find sequence of bundles
        b_sequence = []
        node = best_end
        while node.bundle_id >= 0:
            b_sequence.append(node.bundle_id)
            node = node.parent
        b_sequence.reverse()
        return b_sequence

    def old_bundle_decode(self: HMM, y):
        """

        Returns:
            numpy.ndarray A 1-d sequence of integers representing the time series of bundles
        """
        bundle_sequence = super().broken_decode(y)
        print('Returned from broken_decode')
        return bundle_sequence


class ItemScoreT:
    """For list that is sorted by a function of score and t

    Args:
        number: Value for sorting, ie, score + fudge*t
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
