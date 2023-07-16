"""state_probabilities.py Make a latex table of the probability of
state over training data for a selection of records.

Example:
python state_probabilities.py --records  a01 b02 c01 --trim_start 25 model_path output.tex

"""
import sys
import copy
import pickle
import argparse

import hmmds.applications.apnea.utilities
import hmm.base


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser(
        "Calculate probability of states given data")
    hmmds.applications.apnea.utilities.common_arguments(parser)

    parser.add_argument('model', type=str, help='Path to model')
    parser.add_argument('output', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


def latex_table(y_mod_slow, weights_class, classless, key2index):
    result = [r'''\begin{tabular}{llrrrrr}
\hline \\
''']
    nl = '\\\\\n'
    result.append(
        f'{"index ":6s}&{"name":14s}&{"w class":9s}&{"w/o class":9s}&{"variance":9s}&{"a/b":6s}&{"alpha":9s}{nl}'
    )
    for key_, index in key2index.items():
        key = key_.replace('_', '\_')
        result.append(
            f'{index:6d}&{key:14s}&{weights_class[index]:9.3g}&{classless[index]:9.3g}&{y_mod_slow.variance[index]:9.3g}&{y_mod_slow.beta[index]/y_mod_slow.alpha[index]:6.1f}&{y_mod_slow.alpha[index]:9.2e}{nl}'
        )
    result.append(r'''
\end{tabular}
''')
    return result


def main(argv=None):
    """ Make a latex table for debugging class estimation
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    y_data = [
        hmm.base.JointSegment(
            hmmds.applications.apnea.utilities.read_slow_class(args, record))
        for record in args.records
    ]

    with open(args.model, 'rb') as _file:
        old_args, model = pickle.load(_file)
    original_slow = copy.deepcopy(model.y_mod['slow'])

    weights = model.weights(y_data).sum(axis=0)

    # Calculate weights without class
    y_data = [
        hmm.base.JointSegment(
            hmmds.applications.apnea.utilities.read_slow(args, record))
        for record in args.records
    ]

    with open(args.model, 'rb') as _file:
        old_args, model = pickle.load(_file)
    del model.y_mod['class']

    classless = model.weights(y_data).sum(axis=0)

    result = latex_table(original_slow, weights, classless,
                         old_args.state_key2state_index)

    with open(args.output, encoding='utf-8', mode='w') as _file:
        _file.write(''.join(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())
