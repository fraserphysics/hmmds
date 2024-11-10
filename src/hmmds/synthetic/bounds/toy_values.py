"""toy_values.py Read computation write in form for LaTeX

"""
from __future__ import annotations  # Enables, eg, (self: LocalNonStationary

import sys
import argparse
import pickle
import os

import numpy
import scipy.special


def parse_args(argv, arg_list):
    """Parse a command line.
    Args:
        argv: The command line
        arg_list: List pickle file names, eg, like_lor

    
    """
    parser = argparse.ArgumentParser(
        description='Translate picked values to LaTex')
    for name in arg_list:
        parser.add_argument(name, type=str, help=f'Path to pickle file')
    parser.add_argument('result', type=str, help='path for result')
    return parser.parse_args(argv)


FunctionDict = {}  # Keys are function names.  Values are functions.


def register(func):
    """Decorator that puts function in FunctionDict
    """
    FunctionDict[func.__name__] = func
    return func


@register
def data_h_view(dict_in):
    """return expressions for y_step unit_state_noise observation_noise_multiplier dt

    Args:
        dict_in: Result of running h_cli.py

    """
    t_start = dict_in['t_start']
    t_stop = dict_in['t_stop']
    dict_in['T_Max_Var'] = numpy.argmax(dict_in['y_variances'][t_start:t_stop])
    dict_in['T_Max_Var_Plus_One'] = dict_in['T_Max_Var'] + 1
    dict_in['StateNoise'] = dict_in['unit_state_noise'][0, 0]
    dict_in['ObservationNoise'] = dict_in['observation_noise_multiplier'][0, 0]

    result = {}
    for key in 'y_step atol dt T_Max_Var T_Max_Var_Plus_One StateNoise ObservationNoise'.split(
    ):
        result[f"toyHview{key.replace('_', '')}"] = dict_in[key]
    return result


@register
def toy_h(dict_in):
    """return expressions for several args

    toyhSigmaEta toyhSigmaEpsilon toyhDelta

    Args:
        dict_in: Result of running toy_h.py

    """
    result = {}

    args = dict_in['args']
    for key in 'n_t dev_measurement dev_state_generate dev_state_filter y_step'.split(
    ):
        result[f"toyToyh{key.replace('_', '')}"] = getattr(args, key)
    return result


@register
def benettin(dict_in):
    """return LaTeX expressions for values in dict_in and its args value

    Args:
        dict_in: Result of running benettin.py

    """
    args = dict_in['args']
    for key in 'n_runs dev_state grid_size t_run'.split():
        dict_in[key] = getattr(args, key)

    dict_in['LambdaOne'] = dict_in["spectrum"][0]
    dict_in['N_times'] = int(args.t_run / args.time_step)
    dict_in['ratio'] = args.dev_state / args.grid_size

    for key in 'sde_mean sde_std augmented_mean augmented_std'.split():
        dict_in[key] = dict_in[key][0]
    dict_in['DeltaLambda'] = dict_in['augmented_mean'] - dict_in['sde_mean']

    result = {}
    for key in 'n_runs dev_state grid_size N_times LambdaOne t_run ratio sde_mean sde_std augmented_mean augmented_std DeltaLambda'.split(
    ):
        result[f"toyBenettin{key.replace('_', '')}"] = dict_in[key]
    return result


@register
def like_lor(dict_in):
    """return values from running like_lor.py

    Args:
        dict_in: Result of running like_lor.py

    I'm not sure that the results are used in the LaTeX.
    """
    args = dict_in['args']
    result = {}

    # Get some values from args
    for key in 'n_train n_test n_quantized t_sample min_prob'.split():
        result[f"toyLikeLor{key.replace('_', '')}"] = getattr(args, key)

    # Find the key for the largest number of states (right end of plot)
    min_resolution = 100.0
    for key in dict_in.keys():
        if not isinstance(key, float):
            continue
        if key < min_resolution:
            min_resolution = key

    # Get some values from the right end of the plot
    log_likelihood = dict_in[min_resolution]['log_likelihood']
    dict_in['n_states'] = dict_in[min_resolution]['n_states']
    dict_in[
        'cross_entropy'] = -log_likelihood / args.n_test  # nats per time step
    dict_in['cross_entropy_bits'] = dict_in['cross_entropy'] / numpy.log(2)

    for key in 'n_states cross_entropy cross_entropy_bits'.split():
        result[f"toyLikeLor{key.replace('_', '')}"] = dict_in[key]

    return result


def main(argv=None):
    """Study dependence of relative entropy of Kalman filters on
    actual sample time and assumed observation noise.

    """
    if argv is None:
        argv = sys.argv[1:]
    # List of targets in Rules.mk
    arg_list = 'data_h_view toy_h benettin like_lor'.split()
    args = parse_args(argv, arg_list)

    result = {}

    for name in arg_list:
        path = getattr(args, name)
        with open(path, 'rb') as file_:
            dict_in = pickle.load(file_)
        result.update(FunctionDict[name](dict_in))

    # Calculations of entropies per sample time
    t_sample = result['toyLikeLortsample']
    entropy = result['toyBenettinLambdaOne'] * t_sample
    entropy_bits = entropy / numpy.log(2)
    cross_entropy = result['toyLikeLorcrossentropy']
    cross_entropy_bits = cross_entropy / numpy.log(2)

    result['toyLikeLorGapPercent'] = 100 * (cross_entropy - entropy) / entropy
    result['toyLikeLorEntropyBits'] = entropy_bits
    result['toyLikeLorEntropy'] = entropy

    replacements = {
        1e-4: '10^{-4}',
        1e-5: '10^{-5}',
        1e-6: '10^{-6}',
        1e-7: '10^{-7}',
        1e-8: '10^{-8}',
        1e-9: '10^{-9}',
        1e-10: '10^{-10}'
    }

    for key, value in result.items():
        if value in replacements:
            result[key] = replacements[value]
        if key == 'toyLikeLorGapPercent':
            result[key] = f'{int(value)}'
        elif isinstance(value, int) and value >= 1000:
            result[key] = f'{value:,}'
        elif isinstance(value, float) and (0.01 < abs(value) < 10):
            result[key] = f'{value:.3f}'
        elif isinstance(value, float) and (0.001 < abs(value) < 10):
            result[key] = f'{value:.4f}'

    # Calculate values for the caption of fig:toyH
    erf_inv_sqrt_8 = scipy.special.erf(1 / numpy.sqrt(8))
    result['toyErfSqrtEight'] = f'{erf_inv_sqrt_8:.4}'
    result['toyLogErfSqrtEight'] = f'{numpy.log(erf_inv_sqrt_8):.4}'
    with open(args.result, 'w', encoding='utf-8') as file_:
        for key, value in result.items():
            print(f'\def\{key}{{{value}}}', file=file_)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
