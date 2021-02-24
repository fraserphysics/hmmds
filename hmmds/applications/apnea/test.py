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
    expert_path = '../../../raw_data/apnea/summary_of_training'
    data_names = os.listdir(respiration_path)
    a_names = list(filter(lambda name: name[0] == 'a', data_names))
    b_names = list(filter(lambda name: name[0] == 'b', data_names))
    c_names = list(filter(lambda name: name[0] == 'c', data_names))
    x_names = list(filter(lambda name: name[0] == 'x', data_names))
    train_names = a_names + b_names + c_names

    respiration = {}
    heart_rate = {}
    expert = {}
    for name in data_names:
        # Read all fields of files
        raw_r = utilities.read_respiration(os.path.join(respiration_path, name))
        raw_h = utilities.read_low_pass_heart_rate(
            os.path.join(heartrate_path, name))

        # Ensure that measurement times are the same.  ToDo: Why are
        # there more heart_rate data points?
        n_r = len(raw_r)
        assert len(raw_h) > n_r
        time_difference = raw_r[:, 0] - raw_h[:n_r, 0]
        assert numpy.abs(time_difference).max() == 0.0

        respiration[name] = raw_r[:, 1:]  # Don't store time data
        heart_rate[name] = raw_h[:n_r, -1]  # Store only filtered heart rate
        assert len(respiration[name]) == len(heart_rate[name])
        if name[0] == 'x':
            continue

        # Read expert annotations
        samples_per_minute = 10
        expert[name] = utilities.read_expert(expert_path,
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
        data_names = list(
            (self.random_name(self.data_names) for n in range(self.n_files)))

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


class TestRespiration(BaseClass):
    """ Test hmmds.applications.apnea.observation.Respiration
    """

    def setUp(self):
        super().setUp()
        self.model = self.respiration_model
        self.test_data = list(
            (self.respiration[self.random_name(self.data_names)]
             for n in range(self.n_files)))

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
        self.assertTrue(len(t_seg) == self.n_files + 1)
        self.assertTrue(self.model._y.shape == (self.model.n_times,
                                                self.dimension))
        self.assertTrue(t_seg[-1] == self.model._y.shape[0])
        self.assertTrue(t_seg[-1] > 1000)

    def test_calculate(self):
        self.model.observe(self.test_data)
        likelihood = self.model.calculate()
        self.assertTrue(likelihood.shape == (self.model.n_times, self.n_states))
        self.assertTrue(likelihood.min() >= 0)
        self.assertTrue(likelihood.max() < 10)

    def test_reestimate(self):
        self.model.observe(self.test_data)
        n_times = self.model.t_seg[-1]
        # Create a weight array
        w = numpy.zeros((n_times, self.n_states))
        for i in range(n_times):
            w[i, i % self.n_states] = 1
        self.model.reestimate(w)
        for norm in self.model.norm:
            self.assertTrue(abs(norm - 1.33) < .2)


class TestFilteredHeartRate(BaseClass):
    """ Test hmmds.applications.apnea.observation.FilteredHeartRate
    """

    def setUp(self):
        super().setUp()
        self.model = self.filtered_heart_rate_model

        self.test_data = list(
            (self.heart_rate[self.random_name(self.data_names)]
             for n in range(self.n_files)))

    def test_random_out(self):
        with self.assertRaises(RuntimeError):
            self.model.random_out(0)

    def test_str(self):
        string = self.model.__str__()
        n_instance = string.find('For')
        part = string[n_instance:n_instance + 12]
        self.assertTrue(part == 'For state 0:')

    def test_observe(self):
        t_seg = self.model.observe(self.test_data)
        self.assertTrue(len(t_seg) == self.n_files + 1)
        self.assertTrue(self.model._y.shape == (self.model.n_times,))
        self.assertTrue(t_seg[-1] == self.model._y.shape[0])
        self.assertTrue(t_seg[-1] > 1000)

    def test_calculate(self):  # ToDo move to super?
        self.model.observe(self.test_data)
        likelihood = self.model.calculate()
        self.assertTrue(likelihood.shape == (self.model.n_times, self.n_states))
        self.assertTrue(likelihood.min() >= 0)
        self.assertTrue(likelihood.max() < 10)

    def test_reestimate(self):  # ToDo move to super?
        self.model.observe(self.test_data)
        n_times = self.model.t_seg[-1]
        # Create a weight array
        w = numpy.zeros((n_times, self.n_states))
        for i in range(n_times):
            w[i, i % self.n_states] = 1
        self.model.reestimate(w)
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
        data_names = list(
            (self.random_name(self.train_names) for n in range(self.n_files)))
        self.test_data = list(({
            'respiration_data': self.respiration[name],
            'filtered_heart_rate_data': self.heart_rate[name]
        } for name in data_names))

    def test_random_out(self):
        with self.assertRaises(RuntimeError):
            self.model.random_out(0)

    def test_str(self):
        string = self.model.__str__()
        n_instance = string.find('For')
        part = string[n_instance:n_instance + 12]
        self.assertTrue(part == 'For state 0:')

    def test_observe(self):
        t_seg = self.model.observe(self.test_data)
        self.assertTrue(len(t_seg) == self.n_files + 1)
        self.assertTrue(t_seg[-1] > 1000)

    def test_calculate(self):  # ToDo move to super?
        self.model.observe(self.test_data)
        likelihood = self.model.calculate()
        self.assertTrue(likelihood.shape == (self.model.n_times, self.n_states))
        self.assertTrue(likelihood.min() >= 0)
        self.assertTrue(likelihood.max() < 10)

    def test_reestimate(self):  # ToDo move to super?
        self.model.observe(self.test_data)
        n_times = self.model.t_seg[-1]
        # Create a weight array
        w = numpy.zeros((n_times, self.n_states))
        for i in range(n_times):
            w[i, i % self.n_states] = 1
        self.model.reestimate(w)
        for norm in self.model.filtered_heart_rate_model.norm:
            self.assertTrue(abs(norm - 0.13) < .01)
        for norm in self.model.respiration_model.norm:
            self.assertTrue(abs(norm - 1.37) < .02)


# class TestBundle(TestFilteredHeartRate_Respiration):
#     def setUp(self):
#         super().setUp()
#         self.bundle2state = {
#             0:np.arange(7, dtype=np.int32),
#             1:np.arange(7, 14, dtype=np.int32)
#             }
#         self.model = hmm.base.Observation_with_bundles(self.model, self.bundle2state, self.rng)
