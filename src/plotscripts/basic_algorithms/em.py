"""em.py Illustrate EM algorithm convergence and difference between Log likelihood and Q

"""
import sys
import pickle
import argparse

import numpy
import numpy.linalg

import plotscripts.utilities


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Illustrate convergence of EM")
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('data_path', type=str, help="path to calculated data")
    parser.add_argument('fig_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    return args


def plot_trajectory(axeses, data_dict):
    level_axes, eig_axes = axeses
    trajectory = data_dict['trajectory']
    q_loops = data_dict['q_loops']
    l_loops = data_dict['l_loops']
    max_uv = data_dict['max_uv']
    d_phi = data_dict['d_phi']

    level_axes.plot(trajectory[:, 0],
                    trajectory[:, 1],
                    color='blue',
                    label='EM iterates')
    level_axes.plot(trajectory[:, 0],
                    trajectory[:, 1],
                    marker='.',
                    color='black',
                    linestyle='',
                    markersize=6)

    for iteration, uv in enumerate(trajectory):
        if iteration % 3 != 1:
            continue
        q_loop = q_loops[iteration]
        l_loop = l_loops[iteration]
        if iteration == 1:
            q_label = r'$Q$ Level Set'
            l_label = r'Likelihood Level Set'
        else:
            q_label = ''
            l_label = ''
        level_axes.plot(q_loop[:, 0],
                        q_loop[:, 1],
                        color='green',
                        label=q_label)
        level_axes.plot(l_loop[:, 0], l_loop[:, 1], color='red', label=l_label)
    level_axes.set_ylabel('$v$')
    level_axes.legend()

    eig_axes.plot(trajectory[:, 0],
                  trajectory[:, 1],
                  color='blue',
                  label='EM iterates')
    eig_axes.plot(trajectory[:, 0],
                  trajectory[:, 1],
                  marker='.',
                  color='black',
                  linestyle='',
                  markersize=6)

    values, vectors = numpy.linalg.eig(d_phi)
    for (value, vector, color) in zip(values, vectors.T, ('red', 'green')):
        assert value > 0, f'{value=} {vector=}'
        delta = vector * numpy.log(value) / 50
        for start, label in ((max_uv + delta,
                              rf'eigenvector $\lambda \approx${value:.2f}'),
                             (max_uv - delta, '')):
            eig_axes.plot([start[0], max_uv[0]], [start[1], max_uv[1]],
                          label=label,
                          color=color,
                          alpha=1.0,
                          linewidth=2)

    eig_axes.set_xlabel('$u$')
    eig_axes.set_ylabel('$v$')
    eig_axes.legend()

    return


def main(argv=None):
    """
    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    with open(args.data_path, 'rb') as _file:
        data_dict = pickle.load(_file)

    fig, axeses = pyplot.subplots(2,
                                  1,
                                  figsize=(4, 8),
                                  sharex=True,
                                  sharey=True)
    plot_trajectory(axeses, data_dict)

    fig.tight_layout()
    fig.savefig(args.fig_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
