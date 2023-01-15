"""h_cli.py A command line interface script that mimics h_view.py

I wrote this to write data for building figures without launching the GUI.

"""

import sys
import typing
import pickle
import argparse

import PyQt5.QtWidgets

import hmmds.synthetic.bounds.h_view


def parse_args(argv):
    """Parse a command line.
    """
    parser = argparse.ArgumentParser(
        description='Write data file like h_view.py')
    parser.add_argument('result', type=str, help='write result to this path')
    return parser.parse_args(argv)


class Wall(hmmds.synthetic.bounds.h_view.MainWindow):
    """
    """

    def __init__(self):
        super().__init__()

    def save(self, out_path):
        dump_dict = {}
        for name in 'forecast_means forecast_covariances update_means update_covariances y y_means  y_variances log_probabilities'.split(
        ):
            dump_dict[name] = getattr(self, name)
        for name, variable in self.variable.items():
            dump_dict[name] = variable()
        with open(out_path, 'wb') as _file:
            pickle.dump(dump_dict, _file)


def main(argv=None):
    """ Write pickle like h_view.py

    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    app = PyQt5.QtWidgets.QApplication(sys.argv)
    wall = Wall()
    wall.save(args.result)
    return 0


if __name__ == '__main__':
    sys.exit(main())
