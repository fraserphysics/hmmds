"""prune.py Deletes little used fast chains from a trained heart rate model

Because this code assumes that the keys to the dictionary "state_dict"
are the same integers as the indices to probabilities in the hmm
"model", it can't be applied to a result of itself.

"""
import sys
import pickle
import argparse
import copy

import numpy.random

import hmmds.applications.apnea.utilities
import hmmds.applications.apnea.model_init
import hmm.base
from hmmds.applications.apnea.utilities import State


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Delete little used fast chains")
    hmmds.applications.apnea.utilities.common_arguments(parser)
    parser.add_argument('--exercise', action='store_true', help='debug')
    parser.add_argument('initial_path', type=str, help="path to initial model")
    parser.add_argument('write_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


class Chain:
    """Collect data about a chain

    Args:
        model: A trained hmm
        state_dict: Definition of initial hmm
        switch_key: Points to state that has link to start of chain
        chain_key: Points to first state in chain
        switch_keys: list of pointers to all switch states

    """

    def __init__(self, model, state_dict, switch_key, chain_key, switch_keys):
        # _key is an index of a state, eg model.p_state_initial[_key]
        # _index is an index for state arrays successor or probabilities
        self.switch_key = switch_key
        self.start_key = chain_key
        p_time_average = model.p_state_time_average[switch_key]
        self.probability = p_time_average * model.p_state2state[switch_key,
                                                                chain_key]

        # Identify all states in the chain
        self.chain_keys = []
        successor_key = chain_key
        old_p = model.p_state_time_average[chain_key]

        while not successor_key in switch_keys:
            self.chain_keys.append(successor_key)
            state = state_dict[successor_key]
            assert len(state.successors) == 1
            state_p = model.p_state_time_average[successor_key]
            r_diff = abs(old_p - state_p) / state_p
            assert r_diff < 1e-2
            # These probabilities depend on state likelihoods which
            # are similar bunt not exactly uniform along chains.
            old_p = state_p
            successor_key = state.successors[0]

    def __call__(self):
        return self.probability


def sort_chains(model, state_dict, switch_keys):
    """
    Args:
        model:
        state_dict:
        switch_keys:
    """
    chains = []
    for switch_key in switch_keys:
        for start_index in state_dict[switch_key].successors:
            if start_index in switch_keys:
                continue
            chains.append(
                Chain(model, state_dict, switch_key, start_index, switch_keys))
    chains.sort(key=lambda x: x())
    return chains


def view(model, state_dict, state_key):
    """Print successors of a state and the transition probabilities

    Args:
        model: A trained hmm
        state_dict: From model_init.py
        state_key: Index of state in model and key of state in state_dict
    """
    p_time_average = model.p_state_time_average[state_key]
    print(f'{p_time_average=}')
    state = state_dict[state_key]
    for successor in state.successors:
        print(
            f'{successor:3d} {p_time_average*model.p_state2state[state_key, successor]}'
        )


def prune_chain(chain, old_model, old_state_dict):
    """Delete a chain from a model and state_dict

    Args:
        chain: Chain instance
        old_model: HMM instance
        old_state_dict: Dictionary that defines an hmm

    Return: (new_model, new_state_dict)
    """
    model = copy.deepcopy(old_model)
    state_dict = old_state_dict.copy()

    # Delete chain from new_state_dict
    for state_key in chain.chain_keys:
        del state_dict[state_key]

    switch_state = state_dict[chain.switch_key]
    start_index = switch_state.successors.index(chain.start_key)
    switch_state.successors.pop(start_index)
    switch_state.probabilities = numpy.delete(switch_state.probabilities,
                                              [start_index])

    # Delete chain from model
    model.p_state_initial[chain.start_key] = 0.0
    norm = model.p_state_initial.sum()
    model.p_state_initial /= norm

    model.p_state_time_average[chain.start_key] = 0.0
    norm = model.p_state_time_average.sum()
    model.p_state_time_average /= norm

    model.p_state2state[chain.switch_key, chain.start_key] = 0.0
    model.p_state2state.normalize()

    return model, state_dict


def exercise(model, state_dict, switch_keys):
    """Print results to support debugging.

    """

    chains = sort_chains(model, state_dict, switch_keys)
    iterations = len(chains)
    for _ in range(iterations):
        print(f"{'switch':6s} {'start':5s} {'length':6s} {'probability':11s}")
        for chain in chains:
            print(
                f'{chain.switch_key:6d} {chain.start_key:5d} {len(chain.chain_keys):6d} {chain.probability:11.8f}'
            )

        model, state_dict = prune_chain(chains[0], model, state_dict)
        a = chains[0].switch_key
        b = chains[0].start_key
        print(
            f'model.p_state2state[{a},{b}]={model.p_state2state[a,b]} {model.p_state2state[a,:].sum()=}'
        )
        chains = sort_chains(model, state_dict, switch_keys)


def main(argv=None):
    """ Analyze chains.
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    with open(args.initial_path, 'rb') as _file:
        old_args, model = pickle.load(_file)
    state_dict = old_args.state_dict
    normal_switch = 290
    apnea_switch = 291
    switch_keys = [normal_switch, apnea_switch]

    if args.exercise:  # For testing
        exercise(model, state_dict, switch_keys)
        return 0

    chains = sort_chains(model, state_dict, switch_keys)
    new_model, new_state_dict = prune_chain(chains[0], model, state_dict)
    old_args.state_dict = new_state_dict

    with open(args.write_path, 'wb') as _file:
        pickle.dump((old_args, new_model), _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
