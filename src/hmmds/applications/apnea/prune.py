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
from hmmds.applications.apnea.model_init import dict2hmm


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Delete little used fast chains")
    hmmds.applications.apnea.utilities.common_arguments(parser)
    parser.add_argument('paths',
                        type=str,
                        nargs='+',
                        help="path to initial model and result")
    parser.add_argument('--debug',
                        action='store_true',
                        help='Run debugging code')
    parser.add_argument('--print',
                        action='store_true',
                        help='print summary of input model')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


class Chain:
    """Collect data about a chain

    Args:
        model: A trained hmm
        state_dict: Definition of initial hmm
        switch_key: Points to state that has link to start of chain
        start_key: Points to first state in chain
        switch_keys: list of pointers to all switch states
        key2index:

    """

    def __init__(self, model, state_dict, switch_key, start_key, switch_keys,
                 key2index: dict):
        self.switch_key = switch_key
        self.start_key = start_key
        switch_index = key2index[switch_key]
        start_index = key2index[start_key]
        p_time_average = model.p_state_time_average[switch_index]
        self.probability = p_time_average * model.p_state2state[switch_index,
                                                                start_index]

        # Identify all states in the chain
        self.chain_keys = []
        successor_key = start_key
        old_p = model.p_state_time_average[start_index]

        while not successor_key in switch_keys:
            self.chain_keys.append(successor_key)
            state = state_dict[successor_key]
            assert len(state.successors
                      ) == 1, f'{start_key=} {successor_key=} {str(state)}'
            state_p = model.p_state_time_average[key2index[successor_key]]
            r_diff = abs(old_p - state_p) / state_p
            assert r_diff < 1.5e-2, f'{r_diff=}'
            # These probabilities depend on state likelihoods which
            # are similar but not exactly uniform along chains.
            old_p = state_p
            successor_key = state.successors[0]

    def __call__(self):
        return self.probability


def sort_chains(model, state_dict, switch_keys, key2index):
    """
    Args:
        model:
        state_dict:
        switch_keys:
        key2index:
    """
    chains = []
    for switch_key in switch_keys:
        for start_key in state_dict[switch_key].successors:
            if start_key in switch_keys or start_key.find('noise') > 0:
                continue
            chains.append(
                Chain(model, state_dict, switch_key, start_key, switch_keys,
                      key2index))
    chains.sort(key=lambda x: x())
    return chains


def prune_observation(model, chain_indices):
    """Create a new observation model from model.y_mod with states
    specified by chain_indices deleted

    """

    slow = model.y_mod['slow']
    coefficients = numpy.delete(slow.coefficients, chain_indices, axis=0)
    variances = numpy.delete(slow.variance, chain_indices, axis=0)
    alpha = numpy.delete(slow.alpha, chain_indices, axis=0)
    beta = numpy.delete(slow.beta, chain_indices, axis=0)

    slow_model = hmm.C.AutoRegressive(coefficients[:, :-1], coefficients[:, -1],
                                      variances, model.rng, alpha, beta)
    return {'slow': slow_model}


def prune_chain(chain: Chain, model, state_dict: dict, key2index: dict, args):
    """Delete a chain from a model and state_dict

    Args:
        chain: Function removes this chain
        model: Old HMM instance
        state_dict: Definition of old hmm
        key2index: Map for old hmm

    Return: (new_model, new_state_dict, new_key2index)
    """
    # Strategy: Build new_state_dict with values from old_model.  Then
    # call model_init.dict2hmm to make a new model.

    new_state_dict = {}
    for state_key, state in state_dict.items():
        if state_key in chain.chain_keys:
            continue
        # Create new state for new model]
        state_index = key2index[state_key]
        successors = []
        probabilities = []
        trainable = []
        for successor_key, trainable_ in zip(state.successors, state.trainable):
            if successor_key in chain.chain_keys:
                continue
            successors.append(successor_key)
            trainable.append(trainable_)
            successor_index = key2index[successor_key]
            probability = model.p_state2state[state_index, successor_index]
            probabilities.append(probability)
        new_state_dict[state_key] = State(successors,
                                          probabilities,
                                          state.class_index,
                                          trainable,
                                          prior=state.prior)

    chain_indices = [key2index[key] for key in chain.chain_keys]
    new_model, new_key2index = dict2hmm(new_state_dict,
                                        prune_observation(model, chain_indices),
                                        model.rng,
                                        truncate=args.AR_order)
    return new_model, new_state_dict, new_key2index


def print_summary(state_dict):
    for key, value in state_dict.items():
        chain_position = key[key.rfind('_') + 1:]
        if chain_position.isdigit() and int(chain_position) > 0:
            continue
        print(f'{key}:  {value}')


def debug(old_args, model, state_dict, key2index):
    """Exercise sort_chains and prune_chain and print result
    """
    while len(state_dict) > 100:

        switch_keys = set(())
        for key in state_dict.keys():
            if key.find('switch') >= 0:
                switch_keys.add(key)
        assert switch_keys == set('normal_switch apnea_switch'.split())

        chains = sort_chains(model, state_dict, switch_keys, key2index)
        model, state_dict, key2index = prune_chain(chains[0], model, state_dict,
                                                   key2index, old_args)
    print_summary(state_dict)


def main(argv=None):
    """Analyze chains in a model and write the model after removing
    one chain.

    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    with open(args.paths[0], 'rb') as _file:
        old_args, model = pickle.load(_file)
    state_dict = old_args.state_dict
    key2index = old_args.state_key2state_index

    switch_keys = set(())
    for key in state_dict.keys():
        if key.find('switch') >= 0:
            switch_keys.add(key)
    assert switch_keys == set('normal_switch apnea_switch'.split())

    if args.debug:
        debug(old_args, model, state_dict, key2index)
        return 0

    if args.print:
        print_summary(state_dict)
        return 0

    chains = sort_chains(model, state_dict, switch_keys, key2index)
    new_model, old_args.state_dict, old_args.state_key2state_index = prune_chain(
        chains[0], model, state_dict, key2index, old_args)

    with open(args.paths[1], 'wb') as _file:
        pickle.dump((old_args, new_model), _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
