""" like_table.py Make a table of likelihoods for models of heart rate time series

Example:
python like_table.py model_dir output.tex

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

import hmmds.applications.apnea.utilities
import hmm.base


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser(
        "Calculate likelihood of heart rate models")
    hmmds.applications.apnea.utilities.common_arguments(parser)

    parser.add_argument('--model_template',
                        type=str,
                        default='%s/%s_unmasked',
                        help='For paths to models')
    parser.add_argument('--model_dir',
                        type=str,
                        default='../../../../build/derived_data/apnea/models',
                        help='Path to trained models')
    parser.add_argument('output', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


def get_records(args, names):
    """

    Return: Dict with values (model, y_data)

    """
    records = {}
    for name in names:
        model_path = args.model_template % (args.model_dir, name)

        with open(model_path, 'rb') as _file:
            old_args, model = pickle.load(_file)

        y_data = hmm.base.JointSegment(
            hmmds.applications.apnea.utilities.read_slow_respiration(
                args, name))

        records[name] = (model, y_data)
    return records


def calculate_log_likelihoods(record_dict, names):
    result = {}
    for data_name in names:
        data = record_dict[data_name][1]
        for model_name in names:
            model = record_dict[model_name][0]
            likelihood = model.likelihood(data)
            if likelihood.min() <= 0:
                result[(data_name, model_name)] = -numpy.inf
            else:
                result[(data_name,
                        model_name)] = numpy.log(likelihood).sum() / len(data)
    return result


def calculate_distances(log_likelihoods, names):
    n = len(names)
    ac_totals = {}
    distance = numpy.empty((n, n))
    for i, data_name in enumerate(names):
        ac_totals[data_name] = {'a': 0, 'c': 0}
        for j, model_name in enumerate(names):
            distance[i, j] = -log_likelihoods[(data_name, model_name)]
            if j != i:
                ac_totals[data_name][model_name[0]] += log_likelihoods[(
                    data_name, model_name)]

    return distance, ac_totals


def classify(args, names: list, reference_models: dict) -> dict:
    """Classify each record as normal or apnea

    Args:
        names: strings that identify records, eg, "a01"
        reference_models: keys are names of records, values are hmms

    Return dict of sums of log_likelihoods of classes
    """
    result = {}
    for name in names:
        sums = {'a': 0.0, 'c': 0.0}
        data = hmm.base.JointSegment(
            hmmds.applications.apnea.utilities.read_slow_respiration(
                args, name))
        for model_name, model in reference_models.items():
            if model_name == name:
                continue
            likelihood = model.likelihood(data)
            assert likelihood.min() > 0
            sums[model_name[0]] += numpy.log(likelihood).sum() / len(data)
        result[name] = sums['a'] - sums['c']
    return result


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)
    names = args.a_names + args.c_names

    names_5 = 'a06 a09 a10 a11 c07 c08 c09 c10'.split()
    names2index = {}
    for i, name in enumerate(names_5):
        names2index[name] = i
    records = get_records(args, names_5)
    log_likelihood = calculate_log_likelihoods(records, names_5)
    distance, ac_totals = calculate_distances(log_likelihood, names_5)
    for i, data_name in enumerate(names_5):
        sorted_names = sorted(names_5,
                              key=lambda name: distance[i, names2index[name]])
        print(f'{data_name} {sorted_names}')
        print('    ', end='')
        for name in sorted_names:
            print(f'{distance[i, names2index[name]]:6.3f} ', end='')
        print('\n')

    for name, value in ac_totals.items():
        difference = value["a"] - value["c"]
        print(f'{name} {difference}')

    print('')

    classifications = classify(
        args, names, dict((name, value[0]) for name, value in records.items()))
    for name in names:
        print(f'{name} {classifications[name]}')

    message = [r"\begin{tabular}{l|" + "c" * len(names_5) + "}"]
    for i, data_name in enumerate(names_5):
        message.append(f'{data_name}')
        for model_name in sorted(
                names_5, key=lambda name: distance[i, names2index[name]]):
            message.append(f' & {model_name}')
        if i < len(names_5) - 1:
            message.append('\\\\\n')
    message.append(r"""
\end{tabular}
    """)
    with open(args.output, encoding='utf-8', mode='w') as _file:
        _file.write(''.join(message))
    return 0


if __name__ == "__main__":
    sys.exit(main())
