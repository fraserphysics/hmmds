"""cross_score.py Evaluate apnea classification performance of models
trained on single a records.

python cross_score.py > report.txt

Arguments: Path to models, path to data, and path to result.

Data structure of result:

result['a03']['b02'] is a dict for the results of applying the model
trained on a03 to the b02 record.  Its keys are:

    'expert': Time series of minute by minute classifications by expert
    'hmm': Time series of minute by minute classifications by hmm
    'a->n': Number of minutes hmm said were normal that expert said were apnea
    'n->a': Number of minutes hmm said were apnea that expert said were normal

"""
import sys
import os
import argparse
import glob
import pickle

import numpy

import hmmds.applications.apnea.utilities
import hmm.base


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(
        "Compare classification of minutes by HMM and expert")
    hmmds.applications.apnea.utilities.common_arguments(parser)
    parser.add_argument('--model_template',
                        type=str,
                        default='%s/%s_masked',
                        help='For paths to models')
    parser.add_argument('--model_dir',
                        type=str,
                        default='../../../../build/derived_data/apnea/models',
                        help='Path to trained models')
    parser.add_argument(
        '--data_names',
        type=str,
        nargs='+',
        default=[f'a{x:02d}' for x in range(1, 21)] +
        [f'b{x:02d}' for x in range(1, 5)] +
        [f'c{x:02d}' for x in range(1, 11)],
    )
    parser.add_argument(
        '--model_names',
        type=str,
        nargs='+',
        default=[f'a{x:02d}' for x in range(1, 21)],
    )
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


def read_models(names, args) -> dict:
    """Read the hmm trained on each a record.

    """
    result = {}
    for name in names:
        model_path = args.model_template % (args.model_dir, name)
        with open(model_path, 'rb') as _file:
            old_args, result[name] = pickle.load(_file)
            y_mod = result[name].y_mod
    return result


def read_records(names, args) -> dict:
    """Read the estimated heart rate time series and expert
    classification for each record to be analyzed.

    """
    result = {}
    for name in names:
        y_data = [
            hmm.base.JointSegment(
                hmmds.applications.apnea.utilities.read_slow(args, name))
        ]
        path = os.path.join(args.root, 'raw_data/apnea/summary_of_training')
        expert = hmmds.applications.apnea.utilities.read_expert(path, name)
        result[name] = {'y_data': y_data, 'expert': expert}
    return result


def analyze(models: dict, records: dict, fudge: float) -> dict:
    """Compare expert and hmm classifications

    Args:
        models: key, eg, 'a01', value hmm
        records: key, eg, 'a01', value dict with keys: 'y_data' and 'expert'

    Return: dict, eg result['a01']['c01'] with keys: 'expert hmm a->n n->a'.split()
    """
    result = {}
    for model_name, model in models.items():
        result[model_name] = {}
        for data_name, value in records.items():
            expert = numpy.array(value['expert'], dtype='bool')
            y_data = value['y_data']
            class_model = model.y_mod['class']
            try:  # Exception if model finds data impossible
                hmm_class = model.class_estimate(y_data, fudge)
                length = min(len(hmm_class), len(expert))
                assert 1000 > length > 200, f'{length=}.  Expected about 480 for 8 hours'

                def and_not(a, b):
                    return (a[:length] & numpy.logical_not(b[:length])).sum()

                result[model_name][data_name] = {
                    'expert': expert,
                    'hmm': hmm_class,
                    'length': length,
                    'a->n': and_not(hmm_class, expert),
                    'n->a': and_not(expert, hmm_class)
                }
            except RuntimeError:
                model.y_mod['class'] = class_model
                result[model_name][data_name] = None

    return result


def print_result(models, records, analysis):
    false_alarm = 0
    missed_detection = 0
    print('     ', end='')
    for data_name in records.keys():
        print(f'{data_name:4s}', end='')
    print()
    for model_name in models.keys():
        print(f"{model_name:3s}:", end='')
        for data_name in records.keys():
            result = analysis[model_name][data_name]
            if result is None:
                print('  --', end='')
            else:
                fa = result['a->n']
                md = result['n->a']
                false_alarm += fa
                missed_detection += md
                fraction = (fa + md) / result['length']
                percent = int(100 * (1 - fraction))
                print(f"{percent:4d}", end='')
        print()
    print(f'{false_alarm=} {missed_detection=}')
    return 0


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    models = read_models(args.model_names, args)
    records = read_records(args.data_names, args)
    specific = 0.05
    analysis = analyze(models, records, specific)
    print_result(models, records, analysis)
    return 0


if __name__ == "__main__":
    sys.exit(main())
