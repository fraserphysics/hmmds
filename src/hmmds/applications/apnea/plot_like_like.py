"""plot_like_like.py.  Plot log-likelihood of heart rate model against
log-likelihood of ecg model.

"""

import sys
import os.path
import argparse
import pickle
import typing

import pint
import numpy
import numpy.linalg
import scipy.signal

import utilities
import plotscripts.utilities
import hmm.base

PINT = pint.UnitRegistry()


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(
        description='Plot of likelihoods of heart rate model and ecg model')
    parser.add_argument('--name',
                        type=str,
                        default='a06',
                        help='Records to analyze')
    
    # Likelihood is in eg, build/derived_data/ECG/a06_self_AR3/likelihood
    parser.add_argument('--ECG_dir',
                        type=str,
                        default='../../../../build/derived_data/ECG/',
                        help='Path to likelihood data for reading')
    parser.add_argument('--likelihood_sample_frequency',
                        type=int,
                        default=100,
                        help='Samples per second for the likelihoods')

    # Model is in eg, build/derived_data/apnea/models/a06_unmasked
    parser.add_argument('--model_template',
                        type=str,
                        default='%s/%s_unmasked',
                        help='For paths to models')
    parser.add_argument('--model_dir',
                        type=str,
                        default='../../../../build/derived_data/apnea/models',
                        help='Path to trained models of the joint filtered heart rate data')

    # Joint filtered heart rate data will be read by
    # utilities.read_slow_respiration(args, record) which needs
    # args.derived_apnea_data, args.heart_rate_sample_frequency, and
    # args.trim_start.  All of those args are provided by utilities.py

    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    utilities.common_arguments(parser)
    args = parser.parse_args(argv)
    utilities.join_common(args)
    args.likelihood_sample_frequency *= PINT('Hz')
    return args


def read_joint_model(args):
    model_path = args.model_template % (args.model_dir, args.name)
    with open(model_path, 'rb') as _file:
            old_args, result = pickle.load(_file)
    return result

def read_ecg_likelihood(args):
    path = os.path.join(args.ECG_dir, f'{args.name}_self_AR3/likelihood')
    with open(path, 'rb') as _file:
        likelihood = pickle.load(_file)
    final_time = len(likelihood)/args.likelihood_sample_frequency
    return likelihood

def read_data(args):
    """Read data joint slow-respiration data
    """
    data = hmm.base.JointSegment(
        utilities.read_slow_respiration(
            args, args.name))
    final_time = len(data)/args.heart_rate_sample_frequency
    return data
        
def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]
    #args = parse_args(argv)
    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    model = read_joint_model(args)
    ecg_log_likelihood = numpy.log(read_ecg_likelihood(args))
    joint_heart_rate_data = read_data(args)
    joint_log_likelihood = numpy.log(model.likelihood(joint_heart_rate_data))
    # Hack to circumvent "Cannot operate with Quantity and Quantity of
    # different registries."
    f_like = int(args.likelihood_sample_frequency.to('1/minutes').magnitude)
    f_joint = int(args.heart_rate_sample_frequency.to('1/minutes').magnitude)
    f_ratio = f_like // f_joint
    n_joint = min(len(joint_heart_rate_data), len(ecg_log_likelihood)//f_ratio)
    assert len(ecg_log_likelihood)/f_ratio >= n_joint, f'{len(ecg_log_likelihood)/f_ratio=} {f_ratio=} {n_joint=}'
    x = numpy.empty(n_joint)
    for t in range(n_joint):
        x[t] = ecg_log_likelihood[t*f_ratio:(t+1)*f_ratio].min()
    fig, (ecg_axes, joint_axes, both_axes) = pyplot.subplots(nrows=3,figsize=(6,8))
    # Make t axis in minutes and share
    ecg_axes.sharex(joint_axes)
    ecg_minutes = numpy.arange(len(ecg_log_likelihood))/f_like
    joint_minutes = numpy.arange(len(joint_log_likelihood))/f_joint
    ecg_axes.plot(ecg_minutes, ecg_log_likelihood)
    ecg_axes.set_ylabel('LL_ecg')
    joint_axes.plot(joint_minutes, joint_log_likelihood)
    joint_axes.set_ylabel('joint_ecg')
    both_axes.plot(x,joint_log_likelihood, linestyle='', marker='.')
    both_axes.set_xlabel('ecg')
    both_axes.set_ylabel('joint')

    if args.show:
        pyplot.show()
#    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
