"""
"""
from __future__ import annotations  # Enables, eg, (self: Respiration,
import numpy
import numpy.random
import numpy.linalg

import hmm

Small = 1.0e-25

class Respiration(hmm.base.Observation_0):
    """ Observation model for respiration signal.

    Args:
        mu[n_states, 3]: Mean of distribution for each state
        sigma[n_states, 3, 3]: Covariance matrix for each state
        rng: Random number generator with state

    """
    _parameter_keys = "mu sigma".split()
    def __init__(  # pylint: disable = super-init-not-called
        self:Respiration,
        mu: numpy.ndarray,
        sigma: numpy.ndarray,
        rng: numpy.random.Generator,
    ):
        self.n_states, three = mu.shape
        assert three == 3
        assert sigma.shape == (self.n_states, 3, 3)
        assert isinstance(rng, numpy.random.Generator)

        self.mu = mu
        self.sigma = sigma
        self.inverse_sigma = numpy.empty((self.n_states,3,3))
        self.norm = numpy.empty(self.n_states)
        for i in range(self.n_states):
            self.inverse_sigma[i,:,:] = numpy.linalg.inv(self.sigma[i,:,:])
            determinant =  numpy.linalg.det(sigma[i,:,:])
            self.norm[i] = 1/numpy.sqrt((2*numpy.pi)**3 * determinant)

    def random_out(self: Respiration, s: int)->numpy.ndarray:
        raise RuntimeError('random_out not implemented for Respiration')

    def __str__(self: Respiration
                )->str:
        save = numpy.get_printoptions()['precision']
        numpy.set_printoptions(precision=3)
        rv = 'Model %s instance\n'%self.__class__
        for i in range(self.n_states):
            rv += 'For state %d:\n'%i
            rv += ' inverse_sigma = \n%s\n'%self.inverse_sigma[i]
            rv += ' mu = %s'%self.mu[i]
            rv += ' norm = %f\n'%self.norm[i]
        numpy.set_printoptions(precision=save)
        return rv

    def calculate(self:Respiration)->numpy.ndarray:
        """
        Calculate and return likelihoods.

        Returns:
            self.p_y[t,i] = P(y(t)|s(t)=i)

        """
        assert self._y.shape == (self.n_times, 3)
        for t in range(self.n_times):
            for i in range(self.n_states):
                d = (self._y[t]-self.mu[i])
                dQd = numpy.dot(d, numpy.dot(self.inverse_sigma[i], d))
                if dQd > 300: # Underflow
                    self._likelihood[t,i] = 0
                else:
                    self._likelihood[t,i] = self.norm[i]*numpy.exp(-dQd/2)
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
        self.mu = (numpy.inner(y.T, w.T)/wsum).T
        # Inverse Wishart prior parameters.  Without data sigma_sq = b/a
        a = 4
        b = 0.1
        for i in range(self.n_states):
            rrsum = numpy.zeros((dim,dim))
            for t in range(self.n_times):
                r = y[t]-self.mu[i]
                rrsum += w[t,i]*numpy.outer(r, r)
            self.sigma = (b*numpy.eye(dim) + rrsum)/(a + wsum[i])
            det = numpy.linalg.det(self.sigma)
            assert (det > 0.0)
            self.inverse_sigma[i,:,:] = numpy.linalg.inv(self.sigma)
            self.norm[i] = 1.0/(numpy.sqrt((2*numpy.pi)**dim*det))

class FilteredHeartRate(hmm.base.Observation_0):
    """ Observation model for filtered heart rate measurements.

    Args:
        ar_coefficients[n_states, ar_order]: Auto-regressive coefficients
        variance[n_states]: Residual variance for each state
        rng: Random number generator with state

    Model: likelihood[t,i] = Normal(mu_{t,i}, var[i]) at _y[t]
           where mu_{t,i} = ar_coefficients[i] \cdot _y[t-n_ar:t] + offset[i]
           for t too close to segment boundaries, fill with self._y[boundary]

    Data: Simply the scalar filtered heart rate
    """
    _parameter_keys = "ar_coefficients offset variance".split()
    def __init__(  # pylint: disable = super-init-not-called
        self:FilteredHeartRate,
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
        self.ar_coefficients_offset = numpy.empty((self.n_states, self.ar_order+1))
        self.ar_coefficients_offset[:, :self.ar_order] = ar_coefficients
        self.ar_coefficients_offset[:, self.ar_order] = offset

        self.variance = variance
        self.norm = numpy.empty(self.n_states)
        for i in range(self.n_states):
            self.norm[i] = 1/numpy.sqrt(2*numpy.pi * self.variance[i])

    def random_out(self: FilteredHeartRate, s: int)->numpy.ndarray:
        raise RuntimeError('random_out not implemented for FilteredHeartRate')

    def __str__(self: FilteredHeartRate
                )->str:
        save = numpy.get_printoptions()['precision']
        numpy.set_printoptions(precision=3)
        rv = 'Model %s instance\n'%self.__class__
        for i in range(self.n_states):
            rv += 'For state %d:\n'%i
            rv += ' variance = \n%s\n'%self.variance[i]
            rv += ' ar_coefficients = %s'%self.ar_coefficients_offset[i, :-1]
            rv += ' offset = %s'%self.ar_coefficients_offset[i, -1]
            rv += ' norm = %f\n'%self.norm[i]
        numpy.set_printoptions(precision=save)
        return rv

    def observe(self: FilteredHeartRate, y_segs) -> numpy.ndarray:
        """Attach observations to self as self._y and attach context to self

        Args:
            y_segs: Independent measurement sequences.  Structure
                specified by implementation of self._concatenate() by
                subclasses.

        Returns:
            (numpy.ndarray): Segment boundaries

        context will be used in calculate and reestimate.
        After values get assigned, context[t, :-1] = previous
        observations, and context[t, -1] = 1.0
        """
        self.t_seg = super().observe(y_segs)  # assigns self._y
        self.context = numpy.ones((self.n_times, self.ar_order + 1))

        for t in range(self.ar_order,self.n_times):
            self.context[t, :-1] = self._y[t-self.ar_order:t]
        # Near each segment boundary, t_0, assign context[t,i] = y[t_0] if t-i<t_0
        for t_0 in self.t_seg[:-1]:
            for t in range(t_0, t_0 + self.ar_order):
                self.context[t, 0:self.ar_order-t+t_0] = self._y[t_0]
        return self.t_seg

    def calculate(self:FilteredHeartRate)->numpy.ndarray:
        """
        Calculate and return likelihoods.

        Returns:
            self.p_y[t,i] = P(y(t)|s(t)=i)

        """
        assert self._y.shape == (self.n_times,)

        for t in range(self.n_times):
            delta = self._y[t] - numpy.dot(self.ar_coefficients_offset,
                                           self.context[t])
            self._likelihood[t,:] = self.norm*numpy.exp(
                -delta*delta/(2*self.variance))
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
        mask = w >= Small    # Small weights confuse the residual
                             # calculation in least_squares()
        w2 = mask*w
        wsum = w2.sum(axis=0)
        w1 = numpy.sqrt(w2)      # n_times x n_states array of weights

        # Inverse Wishart prior parameters.  Without data, variance = b/a
        a = 4
        b = 16
        for i in range(self.n_states):
            w_y = w1[:,i] * self._y
            w_context = (w1[:,i] * self.context.T).T
            fit, residuals, rank, singular_values = numpy.linalg.lstsq(w_context, w_y)
            assert rank == self.ar_order + 1
            self.ar_coefficients_offset[i, :] = fit
            delta = w_y - numpy.inner(w_context, fit)
            sum_squared_error = numpy.inner(delta,delta)
            self.variance[i] = (b+sum_squared_error)/(a+wsum[i])
            self.norm[i] = 1/numpy.sqrt(2*numpy.pi*self.variance[i])
