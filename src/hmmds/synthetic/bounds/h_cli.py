"""h_cli.py A command line interface script that mimics h_view.py

Use: python h_cli.py output.pkl

This script writes data for building figures without using the GUI.

"""

import sys
import typing
import pickle
import argparse

import PyQt5.QtWidgets

import hmmds.synthetic.bounds.h_view
import hmm.state_space


def parse_args(argv):
    """Parse a command line.
    """
    parser = argparse.ArgumentParser(
        description='Write data file like h_view.py')
    parser.add_argument('result', type=str, help='write result to this path')
    return parser.parse_args(argv)


class MainWindow(hmmds.synthetic.bounds.h_view.MainWindow):
    """Wrap h_view.MainWindow for calling without GUI.

    """

    def __init__(self):
        super().__init__()  # Runs self.update_filter

    def make_dict(self):
        """Return variable values and results of call to
        self.system.forward_filter in dict.

        """
        result = dict(((name, getattr(self, name)) for name in '''
        forecast_means forecast_covariances update_means
        update_covariances y y_means  y_variances log_probabilities'''.split()))

        for name, variable in self.variable.items():
            assert name not in result
            result[name] = variable()
        return result


def main(argv=None):
    """ Write pickle like h_view.py

    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    app = PyQt5.QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()  # This initialization runs simulation
    result = main_window.make_dict()

    non_stationary = main_window.system
    assert isinstance(non_stationary,
                      hmmds.synthetic.bounds.lorenz.LocalNonStationary)
    sde = non_stationary.system
    assert isinstance(sde, hmm.state_space.SDE)

    # For LaTeX book, get attributes of the non_stationary filter and
    # the stochastic differential equation (SDE)
    result['y_step'] = getattr(non_stationary, 'y_step')
    for name in '''unit_state_noise observation_noise_multiplier dt ivp_args
    atol'''.split():
        result[name] = getattr(sde, name)

    # Hack to get plot range into LaTeX
    result['t_start'] = 0
    result['t_stop'] = 120

    with open(args.result, 'wb') as file_:
        pickle.dump(result, file_)
    return 0


if __name__ == '__main__':
    sys.exit(main())
