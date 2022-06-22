"""test_laser.py Tests modules in src/hmmds/applications/laser.

$ python -m pytest tests/test_laser.py or $ py.test --pdb tests/test_laser.py

"""

import unittest
import pickle
import os

import numpy
import numpy.testing
import numpy.random

import hmmds.applications.laser.optimize_ekf
import hmmds.applications.laser.optimize_particle
import hmmds.applications.laser.particle


def source_paths():
    """Get paths to laser data and parameters from GUI
    """
    source_dir = os.path.dirname(
        hmmds.applications.laser.optimize_particle.__file__)
    explore_path = os.path.join(source_dir, 'explore.txt')
    lp5_path = os.path.join(source_dir, 'LP5.DAT')
    return lp5_path, explore_path


def optimize_ekf(output_path):
    """Run optimization and write result to output_path
    """
    lp5_path, explore_path = source_paths()
    args = f"""--parameter_type GUI_out --laser_data {lp5_path} --length 10
    --method Powell {explore_path}  {output_path}""".split()
    hmmds.applications.laser.optimize_ekf.main(argv=args)


def test_optimize_ekf(tmp_path):
    """This takes 16 seconds"""
    dir_path = tmp_path / "foo"
    dir_path.mkdir()
    parameter_path = dir_path / "parameters"
    optimize_ekf(parameter_path)

    with open(parameter_path, 'rb') as _file:
        assert _file.read().find(str.encode('success')) > 0


def test_particle(tmp_path):
    """This test fails if it is run after test_optimize_particle."""
    dir_path = tmp_path / "foo"
    dir_path.mkdir()
    parameter_path = dir_path / "parameters"
    optimize_ekf(parameter_path)

    result_path = dir_path / "result"
    lp5_path, _ = source_paths()
    args = f"""--laser_data {lp5_path} --n_times 50 {parameter_path}
    {result_path}""".split()
    hmmds.applications.laser.particle.main(argv=args)


def test_optimize_particle(tmp_path):
    """this takes 20 seconds"""
    dir_path = tmp_path / "optimize_particle"
    dir_path.mkdir()
    file_path = dir_path / "result"
    lp5_path, explore_path = source_paths()
    args = f"""--parameter_type GUI_out --laser_data {lp5_path} --length 10
    --method Powell --n_particles 40 {explore_path}  {file_path}""".split()
    hmmds.applications.laser.optimize_particle.main(argv=args)
    with open(file_path, 'rb') as _file:
        assert _file.read().find(str.encode('success')) > 0
