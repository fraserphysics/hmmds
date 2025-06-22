"""filter_to_tex.py  Make a latex table from a directory of filter results

python filter_to_tex.py s_augment/0.01 s_augment/0.003 s_augment.tex

"""

import sys
import os
import argparse
import pickle

import numpy


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='''Create a latex table, eg:
python filter_to_tex.py s_augment/0.01 s_augment/0.003 s_augment.tex''')
    parser.add_argument('dirs', type=str, nargs='+', help='paths')
    args = parser.parse_args(argv)
    return args


def entropy(dir_):
    """Estimate entropy and find minimum number of particles

    Args:
        dir_
    """

    with open(os.path.join(dir_, 'dict.pkl'), 'rb') as file_:
        dict_in = pickle.load(file_)
    gamma = dict_in['gamma']
    n_update = numpy.zeros(len(gamma), dtype=int)
    with open(os.path.join(dir_, 'states_boxes.npy'), 'rb') as file_:
        for n in range(len(gamma)):
            try:
                numpy.load(file_)
                n_update[n] = len(numpy.load(file_))
            except EOFError:
                break
    argmin = n_update[100:].argmin() + 100
    n_min = n_update[argmin]

    offset = 50
    if n_min == 0:
        log_gamma = numpy.log(gamma[offset:argmin - 5]).sum()
        n_gamma = argmin - 5 - offset
    else:
        log_gamma = numpy.log(gamma[offset:]).sum()
        n_gamma = len(gamma) - offset
    h_hat = -log_gamma / (n_gamma * 0.15)
    return h_hat, argmin, n_min


def time(dir_):
    """extract execution time from log file
    """
    with open(os.path.join(dir_, 'log.txt'), 'r', encoding='utf-8') as file_:
        for line in file_.readlines():
            n_elapsed = line.find('elapsed')
            if n_elapsed <= 0:
                continue
            n_start = line[:n_elapsed].rfind(' ')
            return line[n_start:n_elapsed]


def main(argv=None):
    """Plot data from particle filter simulations for entropy estimation
    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    dirs = args.dirs[:-1]
    tex_file = args.dirs[-1]

    table = [
        r"""\begin{tabular*}{1.0\linewidth}{rrrrr} \\
%s &$t_{\text{min}}$&$n_{\text{min}}$&$\hat h$ & time \\ \hline
""" % os.path.basename(tex_file).replace('.tex', '').replace('_', '-')
    ]

    for dir_ in dirs:
        h_hat, argmin, n_min = entropy(dir_)
        key = os.path.basename(dir_)
        table.append(
            f'''{key} & {argmin} & {n_min} & {h_hat:.3f} & {time(dir_)} \\\\
''')
    last = table[-1][:-3]
    table[-1] = last

    table.append(r"""
\end{tabular*}
    """)
    with open(tex_file, encoding='utf-8', mode='w') as _file:
        _file.write(''.join(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
