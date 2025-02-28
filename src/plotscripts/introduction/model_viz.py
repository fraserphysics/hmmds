"""model_viz.py Read a model file and write a graphviz representation

"""

import argparse
import pickle
import sys
import os

import numpy

# https://graphviz.readthedocs.io/en/stable/manual.html
import graphviz


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Create and write a vizualization")
    parser.add_argument('--base_name', type=str, default="state")
    parser.add_argument('--data_dir',
                        type=str,
                        default="derived_data/synthetic")
    parser.add_argument('--threshold', type=float, default=5e-3)
    parser.add_argument('--image_path', type=str, default='./')
    parser.add_argument(
        '--layout',
        type=str,
        default='sfdp',
        help='one of: circo dot fdp neato osage patchwork sfdp twopi')
    parser.add_argument('model_path', type=str)
    parser.add_argument('write_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    return args


def viz(hmm, args, graph):
    """Put nodes and edges of hmm in graph

    """

    n_states = len(hmm.p_state_initial)

    class State:

        def __init__(state, index, p_successor, threshold):
            state.index = index
            state.successors = []
            state.image = os.path.join(args.image_path, f'state{index}.png')
            for successor, probability in enumerate(p_successor):
                if probability > threshold:
                    state.successors.append(successor)
            # Calculate average position of state
            name = f'{args.data_dir}/{args.base_name}{index}'
            xz_list = []
            with open(name, 'r', encoding='utf-8') as data_file:
                for line in data_file.readlines():
                    x, _, z = [float(w) for w in line.split()]
                    xz_list.append((x, z))
            state.xz = numpy.array(xz_list).mean(axis=0) * 1.19
            assert state.xz.shape == (2,)

    state_dict = {}
    for index in range(n_states):
        state_dict[index] = state = State(index, hmm.p_state2state[index],
                                          args.threshold)
        graph.node(f'{index}',
                   label='',
                   shape='rectangle',
                   image=state.image,
                   pos=f'{state.xz[0]},{state.xz[1]}!')
    for state in state_dict.values():
        for state_f in state.successors:
            graph.edge(f'{state.index}',
                       f'{state_f}',
                       penwidth='8',
                       color='blue')


def main(argv=None):
    """ Read hmm and write graphviz representation
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    with open(args.model_path, 'rb') as _file:
        model = pickle.load(_file)

    graph = graphviz.Digraph(format='pdf', strict=True, engine=args.layout)
    graph.graph_attr['nodesep'] = '1.0'
    viz(model, args, graph)
    graph.render(args.write_path, view=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
