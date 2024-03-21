"""bare_likelihoods.py Pickle the likelihoods of models in
$(MODELS)/bare for each record in args.all_names

python bare_likelihoods.py $(MODELS)/bare result.pkl

result['a01']['a02'] = log(prob(data[a02]|bare[a01]))
"""
from __future__ import annotations

import sys
import argparse
import os
import typing
import pickle

import numpy

import utilities


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser(
        "Fit function for threshold of each record")
    utilities.common_arguments(parser)
    parser.add_argument('bare', type=str, help="path to directory of models")
    parser.add_argument('result_path', type=str, help="path to result pickle")
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def main(argv=None):
    """Calculate various statistics and parameters for f(record) ->
    threshold, and write to a pickle file

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    model_names = os.listdir(args.bare)
    assert len(model_names) > 1
    record_names = args.all_names

    result = dict((model_name, {}) for model_name in model_names)
    for model_name in model_names:
        for record_name in record_names:
            model_path = os.path.join(args.bare, model_name)
            model_record = utilities.ModelRecord(model_path, record_name)
            p_yt_steps = model_record.model.likelihood(
                model_record.y_raw_data[0])
            log_likelihood = numpy.log(p_yt_steps).sum()
            result[model_name][record_name] = log_likelihood
    with open(args.result_path, 'wb') as _file:
        pickle.dump(result, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
