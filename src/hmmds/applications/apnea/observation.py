"""
"""
from __future__ import annotations  # Enables, eg, (self: Respiration,

import typing

import numpy
import numpy.random
import numpy.linalg

import hmm
import hmm.observe_float

#ToDo: This small value of Small is required to run pass1 and perhaps to train
#model_High Why?
Small = 1.0e-20


class Respiration(hmm.observe_float.MultivariateGaussian):
    """Observation model for respiration signal.

    Args:
        mu[n_states, 3]: Mean of distribution for each state
        sigma[n_states, 3, 3]: Covariance matrix for each state
        rng: Random number generator with state

    """
    _parameter_keys = "mu sigma".split()

    def __init__(self, *args, **kwargs):
        kwargs['small'] = Small
        super().__init__(*args, **kwargs)

    def random_out(self: Respiration, s: int) -> numpy.ndarray:
        raise RuntimeError('random_out not implemented for Respiration')


class ECG(hmm.observe_float.LinearContext):
    r"""Nonlinear auto-regressive observation model for raw ecg measurements.

    Args:
        coefficients[n_states, 4]: Parameters of mu
        variance[n_states]: Residual variance for each state
        rng: Random number generator with state
        alpha: Denominator part of prior for variance
        beta: Numerator part of prior for variance
        small: Throw error if likelihood at any time is less than small
        n_history: mu is a functions of y[t-n_history:t]

    The models is likelihood[t,i] = Normal(mu[i](y[:t]), var[i]) at
    y[t].  Note that mu[i](y[:t]) is not linear or affine.  The
    function for state i is:

    mu[i](y[:t]) = a[i][0]*rms(y[t-1000:t]) +
                   a[i][1]*mean(y[t-1000:t]) + a[i][2] * y[t-1] + a[i][3]

    I hope that the model for each state fits a particular phase of a heart beat well.

    """
    _parameter_keys = "ar_coefficients offset variance".split()

    def __init__(self: ECG, *args, **kwargs):
        if 'n_history' in kwargs:
            n_history = kwargs['n_history']
            del (kwargs['n_history'])
        else:
            n_history = 1000
        super().__init__(*args, **kwargs)
        self.n_history = n_history

    def _concatenate(self: ECG, segments) -> tuple:
        """Calculate and assign self.context

        Args:
            y_segs: Independent measurement sequences.  Each sequence
            is a 1-d numpy array.

        Returns:
            (all_data, segment boundaries)

        """
        length = 0
        t_seg = [0]
        for seg in segments:
            length += len(seg)
            t_seg.append(length)
        all_data = numpy.empty(length)
        self.context = numpy.ones((length, self.context_dimension)) * -9
        for i in range(len(t_seg) - 1):
            all_data[t_seg[i]:t_seg[i + 1]] = segments[i]
        squared = all_data * all_data

        def assign_context(t, sign=-1):
            # Default context is from history
            if sign == -1:
                start = t - self.n_history
                stop = t
                self.context[t, 2] = all_data[t - 1]
            # There is no history.  Context is from future
            elif sign == 0:
                start = t + 1
                stop = t + self.n_history
                self.context[t, 2] = all_data[t]
            # There is not enough history.  Context is from future
            elif sign == 1:
                start = t + 1
                stop = t + self.n_history
                self.context[t, 2] = all_data[t - 1]
            else:
                raise ValueError(f'{sign=} not -1 or 1')
            self.context[t, 0] = numpy.sqrt(squared[start:stop].mean())
            self.context[t, 1] = all_data[start:stop].mean()
            self.context[t, 3] = 1.0

        for t in range(self.n_history, len(all_data)):
            assign_context(t)
        # For times without relevant history, use future values
        # instead.  It's not right, but it's not too bad, and it's
        # only for the first 10 seconds of an eight hour record.
        for t_start in t_seg[:-1]:
            assign_context(t_start, sign=0)
            for t in range(t_start + 1, t_start + self.n_history):
                assign_context(t, sign=1)
        return all_data, t_seg


class FilteredHeartRate(hmm.observe_float.AutoRegressive):
    r"""Observation model for filtered heart rate measurements.

    Args:
        ar_coefficients[n_states, ar_order]: Auto-regressive coefficients
        variance[n_states]: Residual variance for each state
        rng: Random number generator with state

    Model: likelihood[t,i] = Normal(mu_{t,i}, var[i]) at _y[t]
           where mu_{t,i} = ar_coefficients[i] \cdot _y[t-n_ar:t] + offset[i]
           for t too close to segment boundaries, fill with self._y[boundary]

    Data: Simply the scalar filtered heart rate

    ToDo: Treatment of boundaries between segments is to pad the data
    with copies of the first measurement to provide AR context.

    """
    _parameter_keys = "ar_coefficients offset variance".split()

    # Use a state with wide variance to catch bad data points.  To do
    # that, attach prior parameters to each state separately.  Also
    # calculate AP not likelihood.
    def __init__(self, *args, **kwargs):
        kwargs['small'] = Small
        super().__init__(*args, **kwargs)

    def random_out(self: FilteredHeartRate, s: int) -> numpy.ndarray:
        raise RuntimeError('random_out not implemented for FilteredHeartRate')

    def _concatenate(self: FilteredHeartRate, y_segs_in) -> tuple:
        """Pad the segments and call super

        Args:
            y_segs: Independent measurement sequences.  Each sequence
            is a 1-d numpy array.

        Returns:
            (all_data, segment boundaries)

        Padding is for context that will be assigned in super() and
        used in calculate() and reestimate().  While this treatment is
        not exactly correct, I think it's OK because a fairly constant
        heart rate is plausible.

        """
        modified_y_segs = []
        for seg in y_segs_in:
            new_seg = numpy.empty(self.ar_order + len(seg))
            new_seg[:self.ar_order] = seg[0]
            new_seg[self.ar_order:] = seg
            modified_y_segs.append(new_seg)
            # super will drop ar_order from each new_seg
        return super()._concatenate(modified_y_segs)


class FilteredHeartRate_Respiration(hmm.base.BaseObservation):
    """For measurements of filtered heart rate and respiration combined.

    Args:
        filtered_heart_rate: A model instance
        respiration: A model instance
        rng: Random number generator with state

    Model: likelihood[t,i] = product of component model likelihoods
    (Assumes observation components are conditionally independent
    given state)

    Data: (filtered_heart_rate_data, respiration_data)

    """
    _parameter_keys = "filtered_heart_rate_model respiration_model".split()

    def __init__(  # pylint: disable = super-init-not-called
        self: FilteredHeartRate_Respiration,
        filtered_heart_rate_model: FilteredHeartRate,
        respiration_model: Respiration,
        rng: numpy.random.Generator,
    ):
        assert isinstance(filtered_heart_rate_model, FilteredHeartRate)
        assert isinstance(respiration_model, Respiration)
        assert isinstance(rng, numpy.random.Generator)

        self.filtered_heart_rate_model = filtered_heart_rate_model
        self.respiration_model = respiration_model
        self._rng = rng

    def random_out(self: FilteredHeartRate_Respiration,
                   s: int) -> numpy.ndarray:
        raise RuntimeError(f'random_out not implemented for {self.__class__}')

    def __str__(self: FilteredHeartRate_Respiration) -> str:
        rv = 'Model %s instance\n\n' % self.__class__
        rv += 'FilteredHeartRate component:\n{0}\n'.format(
            self.filtered_heart_rate_model.__str__())
        rv += 'Respiration component:\n{0}'.format(
            self.respiration_model.__str__())
        return rv

    def observe(self: FilteredHeartRate_Respiration,
                y_list: list) -> numpy.ndarray:
        """Attach observations to self.filtered_heart_rate_model and
        self.respiration_model

        Args:
            y_list: Each element is a dict with keys 'filtered_heart_rate_data'
                and 'respiration_data'

        Returns:
            (numpy.ndarray): Segment boundaries

        """
        # Don't use self._concatenate because sub-models need their
        # own t_segs to calculate likelihoods at borders
        self.t_seg = self.filtered_heart_rate_model.observe(
            list((y_dict['filtered_heart_rate_data'] for y_dict in y_list)))
        t_seg_r = self.respiration_model.observe(
            list((y_dict['respiration_data'] for y_dict in y_list)))

        # Ensure that segment boundaries match
        diff = t_seg_r - self.t_seg
        assert numpy.abs(diff).max() == 0.0

        self.n_times = self.t_seg[-1]
        return self.t_seg

    def _concatenate(self: FilteredHeartRate_Respiration,
                     y_list: list) -> tuple:
        """
        Concatenate list of observation sequences.
        
        Args:
           y_list Elements are dicts with keys
               'filtered_heart_rate_data' and 'respiration_data'

        Returns:
            (dict, t_seg) with keys of dict being 'filtered_heart_rate_data' and 'respiration_data'

        """

        filtered_heart_rate_data, t_seg_f = self.filtered_heart_rate_model._concatenate(
            list((y_dict['filtered_heart_rate_data'] for y_dict in y_list)))

        respiration_data, t_seg_r = self.respiration_model._concatenate(
            list((y_dict['respiration_data'] for y_dict in y_list)))

        assert tuple(t_seg_f) == tuple(t_seg_r)
        return {
            'filtered_heart_rate_data': filtered_heart_rate_data,
            'respiration_data': respiration_data
        }, t_seg_f

    def calculate(self: FilteredHeartRate_Respiration) -> numpy.ndarray:
        """
        Calculate and return likelihoods.

        Returns:
            self.p_y[t,i] = P(y(t)|s(t)=i)

        Assume conditional independence of observation components given state
        """
        resp_like = self.respiration_model.calculate()
        hr_like = self.filtered_heart_rate_model.calculate()
        self._likelihood = hr_like * resp_like
        return self._likelihood

    def reestimate(
        self: FilteredHeartRate_Respiration,
        w: numpy.ndarray,
    ):
        """
        Estimate new model parameters.  self._y already assigned

        Args:
            w: Weights. w[t,s] = Prob(state[t]=s) given data and
                old model

        """
        self.filtered_heart_rate_model.reestimate(w)
        self.respiration_model.reestimate(w)


################Code below is for testing##################################
import sys
import argparse
import os
import pickle


def read_ecg(path):
    with open(path, 'rb') as _file:
        return pickle.load(_file)['raw']


def parse_args(argv):
    """ Example for reference and testing
    """

    parser = argparse.ArgumentParser(description='Do not use this code')
    parser.add_argument('--root',
                        type=str,
                        default='../../../../',
                        help='root of hmmds project')
    parser.add_argument('--rtimes',
                        type=str,
                        default='raw_data/Rtimes',
                        help='Relative path to directory of ecg files')
    parser.add_argument('--n_history', type=int, default=10)
    parser.add_argument('--record_names',
                        type=str,
                        nargs='+',
                        default='a01 x02 b01 c05'.split(),
                        help='Records to process')
    return parser.parse_args(argv)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    records = dict([(name, {}) for name in args.record_names])
    for name, value in records.items():
        value['data'] = read_ecg(
            os.path.join(args.root, args.rtimes, name + '.ecg'))

    coefficients = numpy.ones((2, 4))
    variance = numpy.ones(2)
    rng = numpy.random.default_rng()

    ecg = ECG(coefficients, variance, rng, n_history=5)

    segments = [records[name]['data'][:10] for name in args.record_names]
    ecg.observe(segments)
    print(f'{ecg._y.shape=} {ecg.context.shape=}')
    for i, (y, context) in enumerate(zip(ecg._y, ecg.context)):
        if i % 10 == 0:
            print('')
        print(f'{i:2d} {y:6.3f} {context[2]:6.3f}')

    return 0


if __name__ == "__main__":
    sys.exit(main())
