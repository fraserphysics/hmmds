"""self_table.py Make a table of scores for each of the CINC2000 records

Example:
python self_table.py model_dir output.tex

This version uses the models that are trained on only one record.  The
modeles are in files with names like a01_self_AR3/unmasked_trained.

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
    parser.add_argument('--template',
                        type=str,
                        default='%s_self_AR3',
                        help='For directory names')
    parser.add_argument('self_models', type=str, help='Path to data')
    parser.add_argument('output', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.ECG.utilities.join_common(args)
    return args


def get_records(args):
    records = {}
    string = args.self_models + f'/*{args.template[4:]}'
    paths = glob.glob(string)
    names = (os.path.basename(path)[:3] for path in paths)
    assert len(paths) > 5

    for name, path in zip(names, paths):
        with open(os.path.join(path, 'states'), 'rb') as _file:
            states = pickle.load(_file)
        with open(os.path.join(path, 'likelihood'), 'rb') as _file:
            likelihoods = pickle.load(_file)
        records[name] = {'states': states, 'likelihoods': likelihoods}

    for record in records.values():
        states = record['states']
        likelihoods = record['likelihoods']
        indices = numpy.nonzero(states != 0)[0]
        n_ok = len(indices)
        n_all = len(states)
        log_likelihood = numpy.log(likelihoods[indices]).sum()
        record['ok_frac'] = n_ok / n_all
        record['cross_entropy'] = log_likelihood / n_ok

    return records


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    records = get_records(args)
    sorted_names = list(records.keys())
    sorted_names.sort(key=lambda name: records[name]['cross_entropy'],
                      reverse=True)

    message = [
        r"""\begin{tabular}{|lrl|lrl|lrl|}
\hline
name & $-h(X|\theta)$ & plausible &name & $-h(X|\theta)$ & plausible &name & $-h(X|\theta)$ & plausible \\ \hline
"""
    ]
    for i, name in enumerate(sorted_names):
        value = records[name]
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
