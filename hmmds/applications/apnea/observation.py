"""
"""
from __future__ import annotations  # Enables, eg, (self: Respiration,
import numpy
import numpy.random
import numpy.linalg

import hmm

#ToDo: This small value of Small is required to run pass1 and perhaps to train
#model_High Why?
Small = 1.0e-50


class Respiration(hmm.base.Observation_0):
    """Observation model for respiration signal.

    Args:
        mu[n_states, 3]: Mean of distribution for each state
        sigma[n_states, 3, 3]: Covariance matrix for each state
        rng: Random number generator with state

    ToDo: Since this is a simple multivariate Gaussian model, it be in
    the hmm package.

    """
    _parameter_keys = "mu sigma".split()

    def __init__(  # pylint: disable = super-init-not-called
        self: Respiration,
        mu: numpy.ndarray,
        sigma: numpy.ndarray,
        rng: numpy.random.Generator,
        inverse_wishart_a=4,
        inverse_wishart_b=0.1,
    ):
        # Check arguments
        self.n_states, dimension = mu.shape
        assert dimension == 3
        assert sigma.shape == (self.n_states, dimension, dimension)
        assert isinstance(rng, numpy.random.Generator)

        # Assign arguments to self
        self.mu = mu
        self.sigma = sigma
        self.inverse_sigma = numpy.empty((self.n_states, dimension, dimension))
        self.norm = numpy.empty(self.n_states)
        for i in range(self.n_states):
            self.inverse_sigma[i, :, :] = numpy.linalg.inv(self.sigma[i, :, :])
            determinant = numpy.linalg.det(sigma[i, :, :])
            self.norm[i] = 1 / numpy.sqrt(
                (2 * numpy.pi)**dimension * determinant)
        self._rng = rng
        self.inverse_wishart_a = inverse_wishart_a
        self.inverse_wishart_b = inverse_wishart_b

    def random_out(self: Respiration, s: int) -> numpy.ndarray:
        raise RuntimeError('random_out not implemented for Respiration')

    def __str__(self: Respiration) -> str:
        save = numpy.get_printoptions()['precision']
        numpy.set_printoptions(precision=3)
        rv = 'Model %s instance\n' % self.__class__
        for i in range(self.n_states):
            rv += 'For state %d:\n' % i
            rv += ' inverse_sigma = \n%s\n' % self.inverse_sigma[i]
            rv += ' mu = %s' % self.mu[i]
            rv += ' norm = %f\n' % self.norm[i]
        numpy.set_printoptions(precision=save)
        return rv

    def calculate(self: Respiration) -> numpy.ndarray:
        """
        Calculate and return likelihoods.

        Returns:
            self.p_y[t,i] = P(y(t)|s(t)=i)

        """
        assert self._y.shape == (self.n_times, 3)
        for t in range(self.n_times):
            for i in range(self.n_states):
                d = (self._y[t] - self.mu[i])
                dQd = numpy.dot(d, numpy.dot(self.inverse_sigma[i], d))
                if dQd > 300:  # Underflow
                    self._likelihood[t, i] = 0
                else:
                    self._likelihood[t, i] = self.norm[i] * numpy.exp(-dQd / 2)
            if self._likelihood[t, :].sum() < Small:
                raise ValueError(
                    'Observation is not possible from any state.  self.likelihood[{0},:]={1}'
                    .format(t, self._likelihood[t, :]))
        return self._likelihood

    def reestimate(
        self: Respiration,
        w: numpy.ndarray,
    ):
        """
        Estimate new model parameters.  self._y already assigned

        Args:
            w: Weights; Prob(state[t]=s) given data and
                old model

        """
        y = self._y
        dim = 3
        wsum = w.sum(axis=0)
        self.mu = (numpy.inner(y.T, w.T) / wsum).T
        # Inverse Wishart prior parameters.  Without data sigma_sq = b/a
        for i in range(self.n_states):
            rrsum = numpy.zeros((dim, dim))
            for t in range(self.n_times):
                r = y[t] - self.mu[i]
                rrsum += w[t, i] * numpy.outer(r, r)
            self.sigma = (self.inverse_wishart_b * numpy.eye(dim) +
                          rrsum) / (self.inverse_wishart_a + wsum[i])
            det = numpy.linalg.det(self.sigma)
            assert (det > 0.0)
            self.inverse_sigma[i, :, :] = numpy.linalg.inv(self.sigma)
            self.norm[i] = 1.0 / (numpy.sqrt((2 * numpy.pi)**dim * det))


class FilteredHeartRate(hmm.base.Observation_0):
    r"""Observation model for filtered heart rate measurements.

    Args:
        ar_coefficients[n_states, ar_order]: Auto-regressive coefficients
        variance[n_states]: Residual variance for each state
        rng: Random number generator with state

    Model: likelihood[t,i] = Normal(mu_{t,i}, var[i]) at _y[t]
           where mu_{t,i} = ar_coefficients[i] \cdot _y[t-n_ar:t] + offset[i]
           for t too close to segment boundaries, fill with self._y[boundary]

    Data: Simply the scalar filtered heart rate

    ToDo: Treatment of boundaries between segments is not quite right.
    2021-2-24 I think it's best to shorten each segment by the length
    of ar_coefficients.  Since this is simply an auto-regressive
    model, it could be in the hmm package.

    """
    _parameter_keys = "ar_coefficients offset variance".split()

    def __init__(  # pylint: disable = super-init-not-called
        self: FilteredHeartRate,
        ar_coefficients: numpy.ndarray,
        offset: numpy.ndarray,
        variance: numpy.ndarray,
        rng: numpy.random.Generator,
    ):
        assert len(variance.shape) == 1
        assert len(offset.shape) == 1
        assert len(ar_coefficients.shape) == 2

        self.n_states, self.ar_order = ar_coefficients.shape

        assert offset.shape[0] == self.n_states
        assert variance.shape[0] == self.n_states
        assert isinstance(rng, numpy.random.Generator)

        # Store offset in self.ar_coefficients_offset for convenience in
        # both calculating likelihoods and in re-estimation
        self.ar_coefficients_offset = numpy.empty(
            (self.n_states, self.ar_order + 1))
        self.ar_coefficients_offset[:, :self.ar_order] = ar_coefficients
        self.ar_coefficients_offset[:, self.ar_order] = offset

        self.variance = variance
        self.norm = numpy.empty(self.n_states)
        for i in range(self.n_states):
            self.norm[i] = 1 / numpy.sqrt(2 * numpy.pi * self.variance[i])
        self._rng = rng

    def random_out(self: FilteredHeartRate, s: int) -> numpy.ndarray:
        raise RuntimeError('random_out not implemented for FilteredHeartRate')

    def __str__(self: FilteredHeartRate) -> str:
        save = numpy.get_printoptions()['precision']
        numpy.set_printoptions(precision=3)
        rv = 'Model %s instance\n' % self.__class__
        for i in range(self.n_states):
            rv += 'For state %d:\n' % i
            rv += ' variance = \n%s\n' % self.variance[i]
            rv += ' ar_coefficients = %s' % self.ar_coefficients_offset[i, :-1]
            rv += ' offset = %s' % self.ar_coefficients_offset[i, -1]
            rv += ' norm = %f\n' % self.norm[i]
        numpy.set_printoptions(precision=save)
        return rv

    def observe(self: FilteredHeartRate, y_segs) -> numpy.ndarray:
        """Attach observations to self as self._y and attach context to self

        Args:
            y_segs: Independent measurement sequences.  Each sequence
            is a 1-d numpy array.

        Returns:
            (numpy.ndarray): Segment boundaries

        context will be used in calculate and reestimate.
        After values get assigned, context[t, :-1] = previous
        observations, and context[t, -1] = 1.0

        ToDo: Truncate at segment boundaries for context in base
        version in hmm.  Pad with copies in subclass for heart rate.

        """
        self.t_seg = super().observe(y_segs)  # assigns self._y
        self.context = numpy.ones((self.n_times, self.ar_order + 1))

        for t in range(self.ar_order, self.n_times):
            self.context[t, :-1] = self._y[t - self.ar_order:t]
        # Near each segment boundary, t_0, assign context[t,i] = y[t_0] if t-i<t_0
        for t_0 in self.t_seg[:-1]:
            for t in range(t_0, t_0 + self.ar_order):
                self.context[t, 0:self.ar_order - t + t_0] = self._y[t_0]
        return self.t_seg

    def calculate(self: FilteredHeartRate) -> numpy.ndarray:
        """
        Calculate and return likelihoods.

        Returns:
            self.p_y[t,i] = P(y(t)|s(t)=i)

        """
        assert self._y.shape == (self.n_times,)

        for t in range(self.n_times):
            delta = self._y[t] - numpy.dot(self.ar_coefficients_offset,
                                           self.context[t])
            self._likelihood[t, :] = self.norm * numpy.exp(-delta * delta /
                                                           (2 * self.variance))

            if self._likelihood[t, :].sum() < Small:
                raise ValueError(
                    'Observation is not possible from any state.  self.likelihood[{0},:]={1}'
                    .format(t, self._likelihood[t, :]))
        return self._likelihood

    def reestimate(
        self: FilteredHeartRate,
        w: numpy.ndarray,
    ):
        """
        Estimate new model parameters.  self._y already assigned

        Args:
            w: Weights. w[t,s] = Prob(state[t]=s) given data and
                old model

        """
        y = self._y
        mask = w >= Small  # Small weights confuse the residual
        # calculation in least_squares()
        w2 = mask * w
        wsum = w2.sum(axis=0)
        w1 = numpy.sqrt(w2)  # n_times x n_states array of weights

        # Inverse Wishart prior parameters.  Without data, variance = b/a
        a = 4
        b = 16
        for i in range(self.n_states):
            w_y = w1[:, i] * self._y
            w_context = (w1[:, i] * self.context.T).T
            fit, residuals, rank, singular_values = numpy.linalg.lstsq(
                w_context, w_y, rcond=None)
            assert rank == self.ar_order + 1
            self.ar_coefficients_offset[i, :] = fit
            delta = w_y - numpy.inner(w_context, fit)
            sum_squared_error = numpy.inner(delta, delta)
            self.variance[i] = (b + sum_squared_error) / (a + wsum[i])
            self.norm[i] = 1 / numpy.sqrt(2 * numpy.pi * self.variance[i])


class FilteredHeartRate_Respiration(hmm.base.Observation_0):
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
        raise RuntimeError(
            'random_out not implemented for FilteredHeartRate_Respiration')

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
        Args:
           y_list Elements are dicts with keys
               'filtered_heart_rate_data' and 'respiration_data'

        Returns:
            (dict, t_seg) with keys of dict being 'filtered_heart_rate_data' and 'respiration_data'

        """

        filtered_heart_rate_data, t_seg_f = self.filtered_heart_rate_model._concatenate(
            list((y_dict['filtered_heart_rate_data'] for y_dict in y_list)))
        respiration_data, t_seg_r = self.filtered_heart_rate_model._concatenate(
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
        self._likelihood = self.filtered_heart_rate_model.calculate(
        ) * self.respiration_model.calculate()
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
