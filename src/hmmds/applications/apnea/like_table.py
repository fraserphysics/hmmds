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
    # EG, build/derived_data/apnea/models/a01_declass

    #parser.add_argument('output', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


def get_records(args, names):
    records = {}
    for name in names:
        model_path = args.model_template % (args.model_dir, name)

        with open(model_path, 'rb') as _file:
            old_args, model = pickle.load(_file)

        y_data = hmm.base.JointSegment(
            hmmds.applications.apnea.utilities.read_slow_respiration(
                args, name))

        records[name] = (model, y_data[25:])
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
                result[(data_name, model_name)] = numpy.log(
                    model.likelihood(data)).sum() / len(data)
    return result


def calculate_distances(log_likelihoods, names):
    n = len(names)
    distance = numpy.empty((n, n))
    for i, data_name in enumerate(names):
        for j, model_name in enumerate(names):
            distance[i, j] = -log_likelihoods[(data_name, model_name)]
    return distance


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)
    names = args.a_names + args.c_names

    names_5 = names[-15:-5]
    names2index = {}
    for i, name in enumerate(names_5):
        names2index[name] = i
    records = get_records(args, names_5)
    log_likelihood = calculate_log_likelihoods(records, names_5)
    distance = calculate_distances(log_likelihood, names_5)
    for i, data_name in enumerate(names_5):
        sorted_names = sorted(names_5,
                              key=lambda name: distance[i, names2index[name]])
        print(f'{data_name} {sorted_names}')
    return 0
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
