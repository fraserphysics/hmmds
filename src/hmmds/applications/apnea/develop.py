"""develop.py Use this file for developing code.  Once code is stable
move it to other files.  That protocol permits directing gnu-make to
re-make only what's necessary.  """

from __future__ import annotations  # Enables, eg, (self: HMM,

import sys
import os.path
import pickle
import argparse

import numpy
import sortedcontainers
import anytree

import hmm.base

import hmmds.applications.apnea.utilities
import observation


class ItemScoreT:
    """For list that is sorted by a function of score and t

    Args:
        number: Value for sorting, ie, score + fudge*t
        node: Pointer to associated BundleNode
        reference: Key for list sorted by score
    """

    def __init__(self, number: float, node: BundleNode, reference: float):
        assert isinstance(reference, float)
        self.number = number
        self.node = node
        self.reference = reference
        self.key = (number, node)

    def __eq__(self, other):
        return self.key == other.key

    def __lt__(self, other):
        return self.key[0] < other.key[0]


class BundleNode(anytree.NodeMixin):
    """Defines b_0^t, a partial sequence of bundles, and related probabilities

    Args:

        bundle_id: Integer label of bundle
        score: Log probability data y_0^t given b_0^t
        p_state: The probability of states at t given y_0^t and b_0^t
        parent: The parent node
        children: A list of child nodes

    """

    def __init__(
        self: BundleNode,
        bundle_id,
        score,
        p_state,
        parent,
        children,
    ):
        self.bundle_id = bundle_id
        self.score = score
        self.p_state = p_state
        self.parent = parent
        if children:
            self.children = children
        # Calling code should assign priority_item
        self.priority_item = None


class HMM(hmm.base.HMM):
    """This subclass provides methods to estimate the sequence of bundles
    based on measured data.

    """

    def log_probs(  # pylint: disable = arguments-differ
            self: HMM,
            t_start: int = 0,
            t_stop: int = 0,
            t_skip: int = 0,
            last_0=None) -> float:
        """Recursively calculate state probabilities, P(s[t]|y[0:t])

        Args:
            t_start: Use self.state_likelihood[t_start] first
            t_stop: Use self.state_likelihood[t_stop-1] last
            t_skip: Number of time steps from when "last" is valid till t_start
            last_0: Optional initial distribution of states

        Returns:
            Log (base e) likelihood of HMM given entire observation sequence

        The same as forward but this returns the sequence of
        log conditional likelihoods.

        """
        if t_stop == 0:
            # Reduces to ignoring t_start and t_stop and operating on
            # a single segment
            assert t_start == 0
            t_stop = len(self.state_likelihood)

        if last_0 is None:
            last_0 = self.p_state_initial

        last = numpy.copy(last_0).reshape(-1)
        for t in range(t_skip):
            # The following replaces last in place with last *
            # p_state2state ** t_skip
            self.p_state2state.step_forward(last)

        for t in range(t_start, t_stop):
            last *= self.state_likelihood[t]  # Element-wise multiply
            assert last.sum() > 0
            self.gamma_inv[t] = 1 / last.sum()
            last *= self.gamma_inv[t]
            self.alpha[t, :] = last
            self.p_state2state.step_forward(last)
        return -numpy.log(self.gamma_inv[t_start:t_stop])

    def bundle_weight(self: HMM, y: list, fudge=[1, 1]) -> numpy.ndarray:
        """Return the sequence of most likely bundles (not the most likely
        sequence of bundles)

        Args:
            y: List with single element that is time series of measurements
            fudge: At each time multiply the calculated bundle probabilities by
                fudge

        The complexity of this code is linear in the length of the
        data.  The method calls observe, calculate, forward and
        backward to calculate a weight array of state probabilities
        for each time given all of the data.  Then at each time, it
        pools the state weights to get the bundle weights it reports.

        """

        n_bundles = self.y_mod.n_bundle
        # bundle_and_state[bundle, state] = True iff state \in bundle
        bundle_and_state = self.y_mod.bundle_and_state

        fudge = numpy.array(fudge)

        # Take the data and calculate likelihood of states
        assert len(y) == 1
        t_seg = self.y_mod.underlying_model.observe(y)
        assert len(t_seg) == 2
        self.n_times = t_seg[-1]
        self.state_likelihood = self.y_mod.underlying_model.calculate()
        # The records are between 6 and 10 hours
        assert 3600 < self.n_times < 6000, 'self.n_times={0}'.format(
            self.n_times)

        self.alpha = numpy.empty((self.n_times, self.n_states))
        self.beta = numpy.empty((self.n_times, self.n_states))
        self.gamma_inv = numpy.empty((self.n_times,))

        # Forward and backward use all the data with arguments 0,0
        log_likelihood = self.forward(0, 0)
        self.backward(0, 0)
        state_weight = self.alpha * self.beta
        assert state_weight.shape == (self.n_times, self.n_states)
        bundle_weight = numpy.dot(state_weight, bundle_and_state.T) * fudge
        bundle_sequence = bundle_weight.argmax(axis=1)
        assert len(bundle_sequence) == self.n_times
        return bundle_sequence

    def bundle_decode(self: HMM,
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


def main(argv=None):
    """ Put small tests here
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    nodes_by_priority = sortedcontainers.SortedList()
    nodes_by_priority.add(ItemScoreT(.1, 1))
    nodes_by_priority.add(ItemScoreT(.1, 2))
    nodes_by_priority.add(ItemScoreT(.2, 1))
    nodes_by_priority.add(ItemScoreT(0, 1))
    print('nodes_by_score range from {0} to {1}'.format(
        nodes_by_priority[0].number, nodes_by_priority[-1].number))
    return 0


if __name__ == "__main__":
    sys.exit(main())
