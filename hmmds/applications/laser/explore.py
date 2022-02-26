"""explore.py For exploring parameters of Lorenz system.

Derived from
https://www.pythonguis.com/tutorials/creating-your-first-pyqt-window/

"""

import sys  # We need sys so that we can pass argv to QApplication
import os
import argparse
import pdb

import PyQt5.QtWidgets  # QApplication, QMainWindow, QPushButton
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpacerItem, \
    QVBoxLayout, QWidget, QPushButton

import pyqtgraph

import numpy
import numpy.linalg
import scipy.optimize

import plotscripts.utilities

from hmmds.synthetic.filter.lorenz_sde import lorenz_integrate


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Explore 1-d map')
    parser.add_argument('--show',
                        action='store_false',
                        help="display figure using Qt5")
    return parser.parse_args(argv)


def plot_for_r(r):
    argv = sys.argv[1:]
    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)
    figure, (time, x_map) = pyplot.subplots(nrows=2, ncols=1, sharex=True)

    n_x = 100
    x_array = numpy.linspace(1 / n_x, 6, n_x)
    x_initial = numpy.empty((n_x, 3))
    x_final = numpy.empty((n_x, 3))
    t_array = numpy.empty(n_x)

    fixed_point = FixedPoint(r)

    for i, delta_x in enumerate(x_array):
        x_initial[i] = fixed_point.initial_state(delta_x)
        t_array[i], x_final[i] = fixed_point.map_time(x_initial[i])
    x_values = x_initial[:, 0]
    time.plot(x_values, t_array, label='t')
    x_map.plot(x_values, x_final[:, 0], label='final x')
    x_map.plot(x_values, x_values, label='initial x')
    for axis in time, x_map:
        axis.legend()

    pyplot.show()
    return 0


class FixedPoint:
    """Characterizes the focus of the Lorenz system at x_i > 0
    """

    def __init__(self, r):
        s = 10.0
        b = 8.0 / 3
        self.r = r
        root = numpy.sqrt(b * (r - 1))
        self.fixed_point = numpy.array([root, root, r - 1])
        df_dx = numpy.array([  # derivative of x_dot wrt x
            [-s, s, 0], [1, -1, -root], [root, root, -b]
        ])
        values, right_vectors = numpy.linalg.eig(df_dx)
        left_vectors = numpy.linalg.inv(right_vectors)
        for i in range(3):
            assert numpy.allclose(numpy.dot(left_vectors[i], df_dx),
                                  values[i] * left_vectors[i])
            assert numpy.allclose(numpy.dot(df_dx, right_vectors[:, i]),
                                  values[i] * right_vectors[:, i])
        assert values[
            0].imag == 0.0, f"First eigenvalue is not real: values={values}"
        self.projection = numpy.dot(right_vectors[:, 1:],
                                    left_vectors[1:, :]).real
        # projection onto subspace of complex eigenvectors
        self.image_2d = numpy.dot(numpy.array([[1, 0, 0], [0, 0, 1]]),
                                  self.projection)
        # Components 0 and 2 of projection
        assert numpy.allclose(numpy.dot(self.projection, right_vectors[:, -1]),
                              right_vectors[:, -1])
        self.omega = numpy.abs(values[-1].imag)
        self.period = 2 * numpy.pi / self.omega
        self.relax = values[-1].real

    def initial_state(self, delta_x):
        """Find initial state that is distance delta_x from fixed point
        """
        coefficients = numpy.linalg.lstsq(self.image_2d,
                                          numpy.array([delta_x, 0]),
                                          rcond=None)[0]
        return numpy.dot(self.projection, coefficients) + self.fixed_point

    def map_time(self, x_initial):
        """Find time and position that x_initial maps to x[2] = r-1
        """
        h_max = 0.0025
        tenths = numpy.empty((20, 3))
        t_step = self.period / 10
        # Integrate at least once because x_initial is on boundary
        x_last = lorenz_integrate(x_initial, 0, t_step, h_max=h_max, r=self.r)
        for i in range(20):
            x_next = lorenz_integrate(x_last, 0, t_step, h_max=h_max, r=self.r)
            if x_next[2] > self.r - 1 > x_last[2]:
                break
            x_last = x_next
        else:
            raise RuntimeError("Failed to find bracket")

        def func(time):
            x_time = lorenz_integrate(x_last, 0, time, h_max=h_max, r=self.r)
            result = x_time[2] - (self.r - 1)
            return result

        delta_t = scipy.optimize.brentq(func, 0, t_step)
        t_final = (i + 1) * t_step + delta_t
        x_final = lorenz_integrate(x_last, 0, delta_t, h_max=h_max, r=self.r)
        return t_final, x_final


def make_data(r):
    big_t = 10000
    t_sample = 0.03
    data = numpy.empty((big_t, 3))
    relaxed = lorenz_integrate(numpy.ones(3), 0.0, 500.0, r=r)
    data[0] = relaxed
    for t in range(1, big_t):
        data[t] = lorenz_integrate(data[t - 1], 0, t_sample, r=r)
    return data


class MainWindow(PyQt5.QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Exploring Lorenz Parameters")

        # Configure the main window
        layout0 = PyQt5.QtWidgets.QHBoxLayout()
        control_layout = PyQt5.QtWidgets.QVBoxLayout()
        plot_layout = PyQt5.QtWidgets.QVBoxLayout()
        layout0.addLayout(control_layout)
        layout0.addLayout(plot_layout)

        # Define widgets of control section
        quit_button = QPushButton('Quit', self)
        quit_button.clicked.connect(self.close)
        slider_layout = PyQt5.QtWidgets.QHBoxLayout()
        self.slider = {}  # A dict so that I can print all values someday
        for name, minimum, maximum in (
            ('Coarse', 20.0, 30.0),
            ('Medium', 0, .5),
            ('Fine', 0, .005),
        ):
            self.slider[name] = Variable(name, minimum, maximum, self)
            slider_layout.addWidget(self.slider[name])
        # Enable access like self.r.  Perhaps better to access self.slider['r']
        self.__dict__.update(self.slider)

        # Layout control section
        control_layout.addWidget(quit_button)
        control_layout.addLayout(slider_layout)

        # Define widgets for plot section
        phase_portrait = pyqtgraph.GraphicsLayoutWidget(title="Phase Portrait")
        pp_plot = phase_portrait.addPlot()
        self.pp_curve = pp_plot.plot(pen='r')

        time_series = pyqtgraph.GraphicsLayoutWidget(title="Time Series")
        ts_plot = time_series.addPlot()
        self.ts_curve = ts_plot.plot(pen='g')

        # Layout plot section
        plot_layout.addWidget(phase_portrait)
        plot_layout.addWidget(time_series)

        self.update_plot()  # Plot data for initial settings

        # Make self the central widget
        widget = PyQt5.QtWidgets.QWidget()
        widget.setLayout(layout0)
        self.setCentralWidget(widget)

    def update_plot(self):
        r = self.Coarse.x + self.Medium.x + self.Fine.x
        print(f'r={r}')
        data = make_data(r)
        self.pp_curve.setData(data[:, 0], data[:, 2])
        self.ts_curve.setData(data[:200, 0]**2)


class Variable(QWidget):
    """Provide sliders and spin boxes to manipulate variable.

    Args:
        label:
        minimum:
        maximum:
        main_window: For access to method update_plot
        parent:  I don't understand this
    """

    def __init__(self,
                 label: str,
                 minimum: float,
                 maximum: float,
                 main_window,
                 parent=None):
        super(Variable, self).__init__(parent=parent)

        # Instantiate widgets
        self.label = QLabel(self)
        self.slider = PyQt5.QtWidgets.QSlider(self)
        self.spin = PyQt5.QtWidgets.QDoubleSpinBox(self)

        # Attach arguments to self and widgets
        self.label.setText(label)
        self.spin.setMinimum(minimum)
        self.spin.setMaximum(maximum)
        self.minimum = minimum
        self.maximum = maximum
        self.main_window = main_window

        # Modify widget properties
        self.spin.setDecimals(3)
        # self.slider.minimum() = 0, self.slider.maximum() = 99
        self.dx_dslide = (maximum - minimum) / (self.slider.maximum() -
                                                self.slider.minimum())
        self.slider.setValue(
            int((self.slider.maximum() + self.slider.minimum()) / 2))
        self.x = (minimum + maximum) / 2
        self.spin.setValue(self.x)

        # Connect signals to slots after setting value so that the slots won't be called
        self.slider.valueChanged.connect(self.slider_changed)
        self.spin.valueChanged.connect(self.spin_changed)

        # Define layout
        self.setFixedWidth(120)
        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.addWidget(self.label)
        self.verticalLayout.addWidget(self.slider)
        self.verticalLayout.addWidget(self.spin)
        self.resize(self.sizeHint())

    def spin_changed(self, value):
        self.x = value
        self.slider.disconnect()  # Avoid loop with setValue
        self.slider.setValue(self.slider.minimum() +
                             int((value - self.minimum) / self.dx_dslide))
        self.slider.valueChanged.connect(self.slider_changed)
        self.main_window.update_plot()

    def slider_changed(self, value):
        self.x = self.minimum + float(value) * self.dx_dslide
        self.spin.disconnect()  # Avoid loop with setValue
        self.spin.setValue(self.x)
        self.spin.valueChanged.connect(self.spin_changed)
        self.main_window.update_plot()


# see ~/projects/not_active/metfie/gui_eos.py
if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
