"""test_alignment.py Compare lengths of time series associated with records.

python test_alignment.py > report.txt

"""
import sys
import os
import argparse
import pickle
import datetime

import numpy
import pint

import hmmds.applications.apnea.utilities
import hmm.base

PINT = pint.UnitRegistry()


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser("Compare lengths of time series")
    hmmds.applications.apnea.utilities.common_arguments(parser)
    parser.add_argument(
        '--model',
        nargs='?',
        type=argparse.FileType('rb'),
        default='../../../../build/derived_data/apnea/models/two_ar6_masked6',
        help='Path to model')
    parser.add_argument(
        '--record_names',
        type=str,
        nargs='+',
        default='a05 a06 a07'.split(),
    )
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


class Record:

    def __init__(self, name, model, args):
        """Read and calculate time series for name

        """
        self.name = name
        self.fields = 'ecg hr_respiration estimated_class expert'.split()
        with open(os.path.join(args.ecg_dir, f'{name}.ecg')) as _file:
            ecg_dict = pickle.load(_file)  # keys 'ecg' 'times'.  Times are in
            # seconds and sample frequency is
            # centiseconds
        self.ecg = (ecg_dict['ecg'], 100 * PINT('Hz'))

        # Pad prepended to hr_respiration by read
        y_data = model.read_y_no_class(name)
        length = len(y_data['hr_respiration']) - model.args.AR_order
        for key, value in y_data.items():
            if key == 'hr_respiration':
                continue
            assert len(value) == length, f'{key} {len(value)} != {length}'
        f_sample = model.args.model_sample_frequency
        self.hr_respiration = (y_data['hr_respiration'], f_sample)

        joint_data = [hmm.base.JointSegment(y_data)]
        samples_per_minute = int(f_sample.to('1/minute').magnitude)
        # class_estimate calls weights, calls y_mod.observe, calls
        # _concatenate which uses first ar_order samples for context.
        self.estimated_class = (model.class_estimate(joint_data,
                                                     samples_per_minute,
                                                     1.0), 1 / PINT('minute'))

        path = os.path.join(args.root, 'raw_data/apnea/summary_of_training')
        self.expert = (hmmds.applications.apnea.utilities.read_expert(
            path, name), 1 / PINT('minute'))
        for attribute in self.fields:
            data, frequency = getattr(self, attribute)
            length = len(data)
            if attribute == 'hr_respiration':
                length -= model.args.AR_order
            last_time = (length / frequency).to('seconds').magnitude
            setattr(self, attribute, (data, frequency, last_time))
        self.ecg_expert = self.ecg[-1] - self.expert[-1]

    def head(self, extra=None):
        """Print head of each column
        """
        print(f'{"":3s} ', end='')
        for attribute in self.fields:
            print(f'{attribute:9.9s} ', end='')
        print(f'{"ecg-exp":9.9s} ', end='')
        if extra:
            print(extra, end='')
        print('')

    def print(self, extra=None):
        """Print row
        """
        print(f'{self.name:3s} ', end='')
        for attribute in self.fields:
            formatted_time = str(
                datetime.timedelta(seconds=getattr(self, attribute)[-1]))
            print(f'{formatted_time:9.9s} ', end='')
        d_t = self.ecg_expert
        if d_t > 0:
            formatted_time = ' ' + str(datetime.timedelta(seconds=d_t))
        else:
            formatted_time = '-' + str(datetime.timedelta(seconds=-d_t))
        print(f'{formatted_time:9.9s} ', end='')
        if extra is not None:
            print(f'  {extra}', end='')
        print('')


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)
    model = pickle.load(args.model)

    records = {}
    record_names = args.a_names + args.b_names + args.c_names
    #record_names = args.reocrd_names
    for name in record_names:
        records[name] = Record(name, model, args)
    records[list(records.keys())[0]].head('est-exp')
    for name in record_names:
        record = records[name]
        len_estimate = len(record.estimated_class[0])
        len_expert = len(record.expert[0])
        delta = len_estimate - len_expert
        if True:  #len_estimate != len_expert:
            records[name].print(delta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
