"""
"""
from __future__ import annotations  # Enables, eg, (self: Respiration,
import numpy
import numpy.random
import numpy.linalg

import hmm

# ToDo: Make super Observation_0.  Requires putting _concatenate() in
# Observation_0
class Respiration(hmm.base.IntegerObservation):
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
        save = numpy.get_printoptions
        numpy.set_printoptions(precision=3)
        rv = 'Model %s instance\n'%self.__class__
        for i in range(self.n_states):
            rv += 'For state %d:\n'%i
            rv += ' inverse_sigma = \n%s\n'%self.inverse_sigma[i]
            rv += ' mu = %s'%self.mu[i]
            rv += ' norm = %f\n'%self.norm[i]
        numpy.set_printoptions(save)
        return rv

    def calculate(self:Respiration)->numpy.ndarray:
        """
        Calculate and return likelihoods.

        Returns:
            self.p_y[t,i] = P(y(t)|s(t)=i)

        """
        assert self._y.shape == (self.n_times, 3)
        self._likelihood = numpy.empty((self.n_times, self.n_states))
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
