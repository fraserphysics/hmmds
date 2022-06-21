"""test_synthetic.py Tests modules in src/hmmds/synthetic.

Other files have tests for modules in subdirectories, eg,
test_filter.py will test src/hmmds/synthetic/filter.

$ python -m pytest tests/test_synthetic.py or $ py.test --pdb tests/test_synthetic.py

"""

import unittest
import pickle

import numpy
import numpy.testing
import numpy.random

import hmmds.synthetic.em
import hmmds.synthetic.lorenz
import hmmds.synthetic.make_model
import hmmds.synthetic.scalar_gaussian
import hmmds.synthetic.state_pic
import hmmds.synthetic.train_char
import hmmds.synthetic.v_state_pic


def test_em(tmp_path, capsys):
    """Check the length of what em.py prints and that it writes a
    readable pickle file."""
    dir_path = tmp_path / "em_test"
    dir_path.mkdir()
    file_path = dir_path / "pickle_dict"
    args = f"--print {file_path}".split()
    hmmds.synthetic.em.main(argv=args)
    with open(file_path, 'rb') as _file:
        _dict = pickle.load(_file)
    assert set(_dict.keys()) == set('Y alpha mu_i'.split())
    captured = capsys.readouterr()
    parts = captured.out.split()
    assert len(captured.out) == 653
    assert parts[0] == "#"
    assert parts[-1] == "1.850"


def test_lorenz(tmp_path):
    """Exercise the code.  No checks of results."""
    dir_path = tmp_path / "lorenz_test"
    dir_path.mkdir()
    quant_path = dir_path / "quant"
    xyz_path = dir_path / "xyz"
    n_samples = 100  # Makes 2 samples in coarse
    args = f"""--quantfile {quant_path} --xyzfile {xyz_path} --TSintro {dir_path}
    --n_samples {n_samples}""".split()
    hmmds.synthetic.lorenz.main(argv=args)


def test_make_model(tmp_path):
    """Exercise the code.  No checks of results."""
    dir_path = tmp_path / "make_model"
    dir_path.mkdir()

    # Make training data
    quant_path = dir_path / "quant"
    xyz_path = dir_path / "xyz"
    n_samples = 100
    args = f"""--quantfile {quant_path} --xyzfile {xyz_path}
    --n_samples {n_samples}""".split()
    hmmds.synthetic.lorenz.main(argv=args)

    n_train = 5
    args = f"""{n_train} {dir_path} quant m12s.4y""".split()
    hmmds.synthetic.make_model.main(argv=args)


def test_scalar_gaussian(tmp_path):
    """Exercise the code.  No checks of results."""
    dir_path = tmp_path / "scalar_gaussian"
    dir_path.mkdir()

    args = f"""{dir_path}""".split()
    hmmds.synthetic.scalar_gaussian.main(argv=args)


def test_state_pic(tmp_path):
    """Exercise the code.  No checks of results."""
    dir_path = tmp_path / "state_pic"
    dir_path.mkdir()

    # Make training data
    quant_path = dir_path / "quant"
    xyz_path = dir_path / "xyz"
    n_samples = 100
    args = f"""--quantfile {quant_path} --xyzfile {xyz_path}
    --n_samples {n_samples}""".split()
    hmmds.synthetic.lorenz.main(argv=args)

    # Train model
    n_train = 5
    args = f"""{n_train} {dir_path} quant m12s.4y""".split()
    hmmds.synthetic.make_model.main(argv=args)

    args = f"""{dir_path} quant xyz m12s.4y""".split()
    hmmds.synthetic.state_pic.main(argv=args)


def test_train_char(tmp_path):
    """Exercise the code.  No checks of results."""
    dir_path = tmp_path / "train_char"
    dir_path.mkdir()

    # Make training data
    quant_path = dir_path / "quant"
    xyz_path = dir_path / "xyz"
    n_samples = 100
    args = f"""--quantfile {quant_path} --xyzfile {xyz_path}
    --n_samples {n_samples}""".split()
    hmmds.synthetic.lorenz.main(argv=args)

    out_path = dir_path / "TrainChar"
    n_iterations = 5
    n_seeds = 2  # Number of initial models to train
    args = f"""--n_iterations {n_iterations} --n_seeds {n_seeds} {quant_path}
    {out_path}""".split()
    hmmds.synthetic.train_char.main(argv=args)


def test_v_state_pic(tmp_path):
    """Exercise the code.  No checks of results."""
    dir_path = tmp_path / "v_state_pic"
    dir_path.mkdir()

    # Make training data
    quant_path = dir_path / "quant"
    xyz_path = dir_path / "xyz"
    n_samples = 100
    args = f"""--quantfile {quant_path} --xyzfile {xyz_path}
    --n_samples {n_samples}""".split()
    hmmds.synthetic.lorenz.main(argv=args)

    args = f"""{dir_path} xyz varg_state""".split()
    hmmds.synthetic.v_state_pic.main(argv=args)
