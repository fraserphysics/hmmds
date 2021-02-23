"""test_apnea.py: T run "$ python -m pytest test_apnea.py" or
"$ python -m pytest path"

"""
# Copyright (c) 2021 Andrew M. Fraser
import os
import unittest

import numpy
import numpy.testing
import scipy.linalg

import hmm.base

import observation
import utilities

class BaseClass(unittest.TestCase):
    """Holds common values and methods used by other test classes.

    Because this class has no method names that start with "test",
    unittest will not discover and run any of the methods.  However
    the methods of this class can be assigned to names that start with
    "test" in subclasses where they will be discovered and run.

    """
    # Values here can be accessed by instances of subclasses as, eg,
    # self.n_states.  # This imitates proj_hmm/hmm/tests/test_simple.py

    n_states = 12
    respiration_path = '../../../derived_data/apnea/respiration/'
    heartrate_path = '../../../derived_data/apnea/low_pass_heart_rate'
    data_names = os.listdir(respiration_path)
    n_names = len(data_names)
    a_names = list(filter(lambda name: name[0]=='a', data_names))
    b_names = list(filter(lambda name: name[0]=='b', data_names))
    c_names = list(filter(lambda name: name[0]=='c', data_names))
    x_names = list(filter(lambda name: name[0]=='x', data_names))

    respiration = {}
    heart_rate = {}
    for name in data_names:
        respiration[name] = utilities.read_respiration(
            os.path.join(respiration_path, name))[:,1:]  #drop times
        heart_rate[name] = utilities.read_low_pass_heart_rate(
            os.path.join(heartrate_path, name))[:,-1]  # filtered only

    def random_name(self):
        """ Return the name of a data file drawn at random"""
        index = self.rng.integers(self.n_names)
        return self.data_names[index]

    def random_index(self, name):
        """Return a random index for a named data file"""
        length = len(self.respiration[name])
        return self.rng.integers(length)

class TestRespiration(BaseClass):
    """ Test hmmds.applications.apnea.observation.Respiration
    """

    def setUp(self):
        self.n_files = 2
        self.dimension = 3
        self.rng = numpy.random.default_rng(0)
        mu = numpy.zeros((self.n_states, self.dimension))
        sigma = numpy.zeros((self.n_states, self.dimension, self.dimension))
        for state in range(self.n_states):
            data = list((self.respiration[self.random_name()] for n in range(self.n_files)))
            joined = numpy.concatenate(data)
            n_joined = joined.shape[0]
            mu[state,:] = joined.sum()/n_joined
            delta = joined - mu[state,:]
            sigma[state,:,:] = (numpy.dot(delta.T, delta))/n_joined
        self.model = observation.Respiration(mu, sigma, self.rng)
        self.test_data = list((
            self.respiration[self.random_name()] for n in range(self.n_files)))

    def test_random_out(self):
        with self.assertRaises(RuntimeError):
            self.model.random_out(0)

    def test_str(self):
        string = self.model.__str__()
        n_instance = string.find('For')
        part = string[n_instance:60]
        self.assertTrue(part == 'For state 0')

    def test_observe(self):
        t_seg = self.model.observe(self.test_data)
        self.assertTrue(len(t_seg) == self.n_files+1)
        self.assertTrue(self.model._y.shape == (self.model.n_times,self.dimension))
        self.assertTrue(t_seg[-1] == self.model._y.shape[0])
        self.assertTrue(t_seg[-1] > 1000)

    def test_calculate(self):
        self.model.observe(self.test_data)
        likelihood = self.model.calculate()
        self.assertTrue(likelihood.shape == (self.model.n_times,self.n_states))
        self.assertTrue(likelihood.min() >= 0)
        self.assertTrue(likelihood.max() < 10)

    def test_reestimate(self):
        self.model.observe(self.test_data)
        n_times = self.model.t_seg[-1]
        # Create a weight array
        w = numpy.zeros((n_times, self.n_states))
        for i in range(n_times):
            w[i, i%self.n_states] = 1
        self.model.reestimate(w)
        for norm in self.model.norm:
            self.assertTrue( abs(norm - 1.33) < .2)

class TestFilteredHeartRate(BaseClass):
    """ Test hmmds.applications.apnea.observation.FilteredHeartRate
    """

    def setUp(self):
        self.n_files = 2
        self.ar_order = 4
        self.rng = numpy.random.default_rng(0)
        ar_coefficients = numpy.ones((self.n_states, self.ar_order))/self.ar_order
        offset = numpy.zeros(self.n_states)
        variance = numpy.ones(self.n_states)

        for state in range(self.n_states):
            data = list((self.heart_rate[self.random_name()] for n in range(self.n_files)))
            joined = numpy.concatenate(data)
            n_joined = joined.shape[0]
            average = joined.sum()/n_joined
            delta = joined - average
            variance[state] = (delta*delta).sum()/n_joined
            offset[state] = average

        self.model = observation.FilteredHeartRate(
            ar_coefficients, offset, variance, self.rng)

        self.test_data = list((
            self.heart_rate[self.random_name()] for n in range(self.n_files)))

    def test_random_out(self):
        with self.assertRaises(RuntimeError):
            self.model.random_out(0)

    def test_str(self):
        string = self.model.__str__()
        n_instance = string.find('For')
        part = string[n_instance:n_instance+12]
        self.assertTrue(part == 'For state 0:')

    def test_observe(self):
        t_seg = self.model.observe(self.test_data)
        self.assertTrue(len(t_seg) == self.n_files+1)
        self.assertTrue(self.model._y.shape == (self.model.n_times,))
        self.assertTrue(t_seg[-1] == self.model._y.shape[0])
        self.assertTrue(t_seg[-1] > 1000)

    def test_calculate(self):
        self.model.observe(self.test_data)
        likelihood = self.model.calculate()
        self.assertTrue(likelihood.shape == (self.model.n_times,self.n_states))
        self.assertTrue(likelihood.min() >= 0)
        self.assertTrue(likelihood.max() < 10)

    def test_reestimate(self):
        self.model.observe(self.test_data)
        n_times = self.model.t_seg[-1]
        # Create a weight array
        w = numpy.zeros((n_times, self.n_states))
        for i in range(n_times):
            w[i, i%self.n_states] = 1
        self.model.reestimate(w)
        for norm in self.model.norm:
            self.assertTrue( abs(norm - 0.09) < .005)
