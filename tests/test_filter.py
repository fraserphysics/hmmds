"""test_filter.py Tests modules in src/hmmds/synthetic/filter.

$ python -m pytest tests/test_filter.py or $ py.test --pdb tests/test_filter.py

"""

import unittest
import pickle

import numpy
import numpy.testing
import numpy.random

import hmmds.synthetic.filter.distribution
import hmmds.synthetic.filter.linear_map_simulation
import hmmds.synthetic.filter.linear_particle_simulation
import hmmds.synthetic.filter.linear_sde_simulation
import hmmds.synthetic.filter.log_likelihood
import hmmds.synthetic.filter.lorenz_particle_simulation
import hmmds.synthetic.filter.lorenz_simulation
import hmmds.synthetic.filter.mimic_simulation


def test_distribution(tmp_path):
    """Simply run the code.  Don't check results"""
    dir_path = tmp_path / "distribution_test"
    dir_path.mkdir()
    file_path = dir_path / "pickled_result"
    args = f"--n_samples 100 --a 0.05 --b 0.2 {file_path}".split()
    hmmds.synthetic.filter.distribution.main(argv=args)


def test_linear_map_simulation(tmp_path):
    """Exercise the code.  No checks of results."""
    dir_path = tmp_path / "linear_map_simulation"
    dir_path.mkdir()
    file_path = dir_path / "linear_map_simulation_result"
    args = f"--n_fine 100 --n_coarse 100 {file_path}".split()
    hmmds.synthetic.filter.linear_map_simulation.main(argv=args)


def test_linear_particle_simulation(tmp_path):
    """Exercise the code.  No checks of results."""
    dir_path = tmp_path / "linear_particle_simulation"
    dir_path.mkdir()
    file_path = dir_path / "linear_particle_simulation_result"
    args = f"--n_fine 5 --n_coarse 5 {file_path}".split()
    hmmds.synthetic.filter.linear_particle_simulation.main(argv=args)


def test_linear_sde_simulation(tmp_path):
    """Exercise the code.  No checks of results."""
    dir_path = tmp_path / "linear_sde_simulation"
    dir_path.mkdir()
    file_path = dir_path / "linear_sde_simulation_result"
    args = f"--n_fine 5 --n_coarse 5 {file_path}".split()
    hmmds.synthetic.filter.linear_sde_simulation.main(argv=args)


def test_log_likelihood(tmp_path):
    """Exercise the code.  No checks of results."""
    dir_path = tmp_path / "log_likelihood"
    dir_path.mkdir()
    file_path = dir_path / "log_likelihood_result"
    args = f"--n_samples 100 --n_b 5 --b_range .8 1.2 {file_path}".split()
    hmmds.synthetic.filter.log_likelihood.main(argv=args)


def test_lorenz_particle_simulation(tmp_path):
    """Exercise the code.  No checks of results."""
    dir_path = tmp_path / "lorenz_particle_simulation"
    dir_path.mkdir()
    file_path = dir_path / "lorenz_particle_simulation_result"
    args = f"--n_fine 5 --n_coarse 5 {file_path}".split()
    hmmds.synthetic.filter.lorenz_particle_simulation.main(argv=args)


def test_lorenz_simulation(tmp_path):
    """Exercise the code.  No checks of results."""
    dir_path = tmp_path / "lorenz_simulation"
    dir_path.mkdir()
    file_path = dir_path / "lorenz_simulation_result"
    args = f"--n_fine 5 --n_coarse 5 {file_path}".split()
    hmmds.synthetic.filter.lorenz_simulation.main(argv=args)


def test_mimic_simulation(tmp_path):
    """Exercise the code.  No checks of results."""
    dir_path = tmp_path / "mimic_simulation"
    dir_path.mkdir()
    file_path = dir_path / "mimic_simulation_result"
    args = f"--n_fine 5 --n_coarse 5 {file_path}".split()
    hmmds.synthetic.filter.mimic_simulation.main(argv=args)
