"""test_apnea.py: To run "$ py.test test.py" or
"$ python -i -m unittest test.TestRespiration"

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
    unittest will not discover and run any of the methods.

    """
    # Values here can be accessed by instances of subclasses as, eg,
    # self.n_states.  # This imitates proj_hmm/hmm/tests/test_simple.py

    n_states = 12
    common = utilities.Common('../../../')
    train_names = common.a_names + common.b_names + common.c_names

    respiration = {}
    heart_rate = {}
    expert = {}
    for name in common.all_names:
        # Build both so that the lengths of each are the same
        both = utilities.heart_rate_respiration_data(name, common)
        respiration[name] = both['respiration_data']
        heart_rate[name] = both['filtered_heart_rate_data']
        if name[0] == 'x':
            continue

        # Read expert annotations
        samples_per_minute = 10
        expert[name] = utilities.read_expert(common.expert,
                                             name).repeat(samples_per_minute)

    def random_name(self, names):
        """ Return the name of a data file drawn at random"""
        index = self.rng.integers(len(names))
        return names[index]

    def random_index(self, name):
        """Return a random index for a named data file"""
        length = len(self.respiration[name])
        return self.rng.integers(length)

    def setUp(self):
        self.n_files = 2

        self.rng = numpy.random.default_rng(0)

        # Use same records for both data sets
        data_names = list((self.random_name(self.common.all_names)
                           for n in range(self.n_files)))

        # Make respiration model
        self.dimension = 3
        mu = numpy.zeros((self.n_states, self.dimension))
        sigma = numpy.zeros((self.n_states, self.dimension, self.dimension))
        for state in range(self.n_states):
            data = list((self.respiration[name] for name in data_names))
            joined = numpy.concatenate(data)
            n_joined = joined.shape[0]
            mu[state, :] = joined.sum() / n_joined
            delta = joined - mu[state, :]
            sigma[state, :, :] = (numpy.dot(delta.T, delta)) / n_joined
        self.respiration_model = observation.Respiration(mu, sigma, self.rng)

        # Make filtered heart rate model
        self.ar_order = 4
        ar_coefficients = numpy.ones(
            (self.n_states, self.ar_order)) / self.ar_order
        offset = numpy.zeros(self.n_states)
        variance = numpy.ones(self.n_states)

        for state in range(self.n_states):
            data = list((self.heart_rate[name] for name in data_names))
            joined = numpy.concatenate(data)
            n_joined = joined.shape[0]
            average = joined.sum() / n_joined
            delta = joined - average
            variance[state] = (delta * delta).sum() / n_joined
            offset[state] = average

        self.filtered_heart_rate_model = observation.FilteredHeartRate(
            ar_coefficients, offset, variance, self.rng)

    def random_out(self):
        """ A test for subclasses
        """
        with self.assertRaises(RuntimeError):
            self.model.random_out(0)

    def calculate(self):
        """A test for subclasses

        """
        self.model.observe(self.test_data)
        likelihood = self.model.calculate()
        self.assertTrue(likelihood.shape == (self.model.n_times, self.n_states))
        self.assertTrue(likelihood.min() >= 0)
        self.assertTrue(likelihood.max() < 10)

    def string(self):
        """A test for subclasses

        """
        string = self.model.__str__()
        n_instance = string.find('For')
        part = string[n_instance:n_instance + 12]
        self.assertTrue(part == 'For state 0:')

    def reestimate(self):
        """Prepare for test_reestimate in subclasses
        """
        self.model.observe(self.test_data)
        n_times = self.model.t_seg[-1]
        # Create a weight array
        w = numpy.zeros((n_times, self.n_states))
        for i in range(n_times):
            w[i, i % self.n_states] = 1
        self.model.reestimate(w)


class TestRespiration(BaseClass):
    """ Test hmmds.applications.apnea.observation.Respiration
    """

    def setUp(self):
        super().setUp()
        self.model = self.respiration_model
        self.test_data = list(
            (self.respiration[self.random_name(self.common.all_names)]
             for n in range(self.n_files)))

    test_calculate = BaseClass.calculate
    test_str = BaseClass.string
    test_random_out = BaseClass.random_out

    def test_observe(self):
        t_seg = self.model.observe(self.test_data)
        self.assertTrue(len(t_seg) == self.n_files + 1)
        self.assertTrue(self.model._y.shape == (self.model.n_times,
                                                self.dimension))
        self.assertTrue(t_seg[-1] == self.model._y.shape[0])
        self.assertTrue(t_seg[-1] > 1000)

    def test_reestimate(self):
        self.reestimate()
        for norm in self.model.norm:
            self.assertTrue(abs(norm - 1.33) < .2)


class TestFilteredHeartRate(BaseClass):
    """ Test hmmds.applications.apnea.observation.FilteredHeartRate
    """

    def setUp(self):
        super().setUp()
        self.model = self.filtered_heart_rate_model

        self.test_data = list(
            (self.heart_rate[self.random_name(self.common.all_names)]
             for n in range(self.n_files)))

    test_calculate = BaseClass.calculate
    test_str = BaseClass.string
    test_random_out = BaseClass.random_out

    def test_observe(self):
        t_seg = self.model.observe(self.test_data)
        self.assertTrue(len(t_seg) == self.n_files + 1)
        self.assertTrue(self.model._y.shape == (self.model.n_times,))
        self.assertTrue(t_seg[-1] == self.model._y.shape[0])
        self.assertTrue(t_seg[-1] > 1000)

    def test_reestimate(self):
        self.reestimate()
        for norm in self.model.norm:
            self.assertTrue(abs(norm - 0.099) < .01)


class TestFilteredHeartRate_Respiration(BaseClass):
    """ Test hmmds.applications.apnea.observation.FilteredHeartRate_Respiration
    """

    def setUp(self):
        super().setUp()
        self.n_files = 5
        self.model = observation.FilteredHeartRate_Respiration(
            self.filtered_heart_rate_model, self.respiration_model, self.rng)

        # Use same records for both test data sets
        self.data_names = list(
            (self.random_name(self.train_names) for n in range(self.n_files)))
        self.test_data = list(({
            'respiration_data': self.respiration[name],
            'filtered_heart_rate_data': self.heart_rate[name]
        } for name in self.data_names))

    test_calculate = BaseClass.calculate
    test_str = BaseClass.string
    test_random_out = BaseClass.random_out

    def test_observe(self):
        t_seg = self.model.observe(self.test_data)
        self.assertTrue(len(t_seg) == self.n_files + 1)
        self.assertTrue(t_seg[-1] > 1000)

    def test_reestimate(self):
        self.reestimate()
        for norm in self.model.filtered_heart_rate_model.norm:
            self.assertTrue(abs(norm - 0.13) < .01)
        for norm in self.model.respiration_model.norm:
            self.assertTrue(abs(norm - 1.37) < .02)


class TestBundle(TestFilteredHeartRate_Respiration):
    """Creates an instance of observation.FilteredHeartRate_Respiration
    and uses it as the underlying model in an instance of
    hmm.base.Observation_with_bundles.

    That is the kind of model that will use the expert annotations to
    train.

    """

    def setUp(self):
        super().setUp()

        # Assign states to bundles
        self.bundle2state = {
            0: numpy.arange(6, dtype=numpy.int32),
            1: numpy.arange(6, self.n_states, dtype=numpy.int32)
        }

        # Get the FilteredHeartRate_Respiration instance from super
        self.model = hmm.base.Observation_with_bundles(self.model,
                                                       self.bundle2state,
                                                       self.rng)

        # The test data is a list with elements of type
        # Bundle_segment.  For each Bundle_segment bundles is a time
        # series of classifications and y is a dict with items for
        # filtered heart rate data and respiration data.
        self.test_data = []
        for name in self.data_names:
            self.test_data.append(
                utilities.heart_rate_respiration_bundle_data(name, self.common))

    def test_reestimate(self):
        """ Difference from super is "underlying"
        """
        self.reestimate()
        for norm in self.model.underlying_model.filtered_heart_rate_model.norm:
            self.assertTrue(abs(norm - 0.13) < .01)
        for norm in self.model.underlying_model.respiration_model.norm:
            self.assertTrue(abs(norm - 1.37) < .02)
