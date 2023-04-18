"""table.py Make a table of scores for each of the CINC2000 records

Example:
python table.py model_dir output.tex

"""
import sys
import os.path
import pickle
import argparse
import glob
import os

import numpy

import hmmds.applications.apnea.ECG.utilities


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Make a table of scores")
    hmmds.applications.apnea.ECG.utilities.common_arguments(parser)
    parser.add_argument('--states',
                        type=str,
                        default='states',
                        help='relative states directory')
    parser.add_argument('--likelihood',
                        type=str,
                        default='likelihood',
                        help='relative likelihood directory')
    parser.add_argument('model_dir', type=str, help='Path to data')
    parser.add_argument('output', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.ECG.utilities.join_common(args)
    return args


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    states_dir = os.path.join(args.model_dir, args.states)
    likelihood_dir = os.path.join(args.model_dir, args.likelihood)
    states_set = set(
        os.path.basename(path) for path in glob.glob(f'{states_dir}/*'))
    likelihood_set = set(
        os.path.basename(path) for path in glob.glob(f'{likelihood_dir}/*'))
    joint_list = list(states_set & likelihood_set)
    dict_names = {}
    for record_name in joint_list:
        with open(os.path.join(states_dir, record_name), 'rb') as _file:
            states = pickle.load(_file)
        with open(os.path.join(likelihood_dir, record_name), 'rb') as _file:
            likelihoods = pickle.load(_file)
        assert len(states) == len(likelihoods)
        indices = numpy.nonzero(states != 0)[0]
        n_ok = len(indices)
        n_all = len(states)
        log_likelihood = numpy.log(likelihoods[indices]).sum()
        dict_names[record_name] = {
            'ok_frac': n_ok / n_all,
            'cross_entropy': log_likelihood / n_ok
        }
    joint_list.sort(key=lambda name: dict_names[name]['cross_entropy'],
                    reverse=True)
    message = [
        r"""\begin{tabular}{|lrl|lrl|lrl|}
\hline
name & $-h(X|\theta)$ & plausible &name & $-h(X|\theta)$ & plausible &name & $-h(X|\theta)$ & plausible \\ \hline
"""
    ]
    for i, name in enumerate(joint_list):
        value = dict_names[name]
        message.append(
            f" {name}  &   {value['cross_entropy']:7.4f}     &     {value['ok_frac']:7.5f} "
        )
        if i % 3 == 2:
            message.append("""\\\\
""")
        else:
            message.append(" & ")
    message.append(r"""
\hline
\end{tabular}
    """)
    with open(args.output, encoding='utf-8', mode='w') as _file:
        _file.write(''.join(message))
    return 0


if __name__ == "__main__":
    sys.exit(main())
