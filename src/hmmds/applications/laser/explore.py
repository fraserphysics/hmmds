"""explore.py For exploring parameters of Lorenz system.

Derived from
https://www.pythonguis.com/tutorials/creating-your-first-pyqt-window/

"""

# PyQt5 is hopeless: pylint: skip-file
import sys  # We need sys so that we can pass argv to QApplication
import os

import PyQt5.QtWidgets
import pyqtgraph

import numpy
import numpy.linalg
import scipy.optimize

import plotscripts.utilities
import hmmds.applications.laser.utilities

from hmmds.synthetic.filter.lorenz_sde import lorenz_integrate


def make_data(
        values,  # MainWindow,
        laser_t: float,
        over_sample: int) -> numpy.ndarray:
    """Integrate the Lorenz system

    Args:
        values  Instance that provides access to Variables
        laser_t Sampling period of laser data
        over_sample Number of synthetic data points per laser data point
    """
    s = values.s()
    r = values.r()
    b = values.b()
    fixed_point = hmmds.applications.laser.utilities.FixedPoint(r, s, b)
    initial_state = lorenz_integrate(  #
        fixed_point.initial_state(values.delta_x()), 0.0, values.delta_t(), s,
        r, b)
    t_sample = laser_t * values.t_ratio() / over_sample
    n_samples = int(values.T_total() / t_sample)

    data = numpy.empty((n_samples, 3))
    data[0] = initial_state
    for i in range(1, n_samples):
        data[i] = lorenz_integrate(data[i - 1], 0, t_sample, s, r, b)
    return data


class MainWindow(PyQt5.QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Exploring Lorenz Parameters")

        # Configure the main window
        layout0 = PyQt5.QtWidgets.QHBoxLayout()
        control_layout = PyQt5.QtWidgets.QVBoxLayout()
        buttons_layout = PyQt5.QtWidgets.QHBoxLayout()
        sliders1_layout = PyQt5.QtWidgets.QHBoxLayout()
        control2_layout = PyQt5.QtWidgets.QHBoxLayout()
        control_layout.addLayout(buttons_layout)
        control_layout.addLayout(sliders1_layout)
        control_layout.addLayout(control2_layout)
        plot_layout = PyQt5.QtWidgets.QVBoxLayout()
        layout0.addLayout(control_layout)
        layout0.addLayout(plot_layout)

        # Define widgets of button section
        quit_button = PyQt5.QtWidgets.QPushButton('Quit', self)
        quit_button.clicked.connect(self.close)
        write_button = PyQt5.QtWidgets.QPushButton('Write values to file', self)
        write_button.clicked.connect(self.write_values)
        read_button = PyQt5.QtWidgets.QPushButton('Read values from file', self)
        read_button.clicked.connect(self.read_values)
        likelihood_button = PyQt5.QtWidgets.QPushButton('Calculate Likelihood',
                                                        self)
        likelihood_button.clicked.connect(self.calculate_likelihood)

        # Layout button section
        buttons_layout.addWidget(quit_button)
        buttons_layout.addWidget(write_button)
        buttons_layout.addWidget(read_button)
        buttons_layout.addWidget(likelihood_button)

        self.variable = {}  # A dict so that I can print all values someday

        # Layout first row of sliders
        for name, minimum, maximum in (
            ('s', 1.0, 3.0),
            ('r', 10.0, 25.0),
            ('b', 0.2, 3.0),
            ('delta_x', 0, 6),
            ('delta_t', 0, 5.0),
        ):
            self.variable[name] = Variable(name, minimum, maximum, self)
            sliders1_layout.addWidget(self.variable[name])

        # Layout second row of sliders
        for name, minimum, maximum in (
            ('t_ratio', 1.0, 20.0),
            ('x_ratio', 5.0, 20.0),
            ('offset', 10, 20),
            ('T_total', 10, 50),
        ):
            self.variable[name] = Variable(name, minimum, maximum, self)
            control2_layout.addWidget(self.variable[name])

        # Enable access like self.r.  Perhaps better to access self.variable['r']
        self.__dict__.update(self.variable)

        # Define widgets for plot section
        phase_portrait = pyqtgraph.GraphicsLayoutWidget(title="Phase Portrait")
        pp_plot = phase_portrait.addPlot()
        self.pp_curve = pp_plot.plot(pen='g')

        time_series = pyqtgraph.GraphicsLayoutWidget(title="Time Series")
        ts_plot = time_series.addPlot()
        ts_plot.addLegend()
        self.ts_curve = ts_plot.plot(pen='g', name='simulation')
        self.laser_plot = ts_plot.plot(pen='r', name='LP5DAT')

        likelihood = pyqtgraph.GraphicsLayoutWidget(
            title="Log Likelihood per Step")
        ll_plot = likelihood.addPlot()
        ll_plot.addLegend()
        self.like_curve = ll_plot.plot(pen='g', name='log likelihood')

        # Layout plot section
        plot_layout.addWidget(phase_portrait)
        plot_layout.addWidget(time_series)
        plot_layout.addWidget(likelihood)

        # Make self the central widget
        widget = PyQt5.QtWidgets.QWidget()
        widget.setLayout(layout0)
        self.setCentralWidget(widget)

        # Read the laser data
        self.laser_data = hmmds.applications.laser.utilities.read_tang(
            '../../../../raw_data/LP5.DAT')
        assert self.laser_data.shape == (2, 2876)

        self.update_plot()  # Plot data for initial settings

    def get_parameters(self):
        """Return a Parameters instance for make_non_stationary with
        values from self

        """
        s = self.variable['s']()
        r = self.variable['r']()
        b = self.variable['b']()
        delta_t = self.variable['delta_t']()
        delta_x = self.variable['delta_x']()
        t_ratio = self.variable['t_ratio']()
        x_ratio = self.variable['x_ratio']()
        offset = self.variable['offset']()
        fixed_point = hmmds.applications.laser.utilities.FixedPoint(r, s, b)
        initial_state = lorenz_integrate(  #
            fixed_point.initial_state(delta_x), 0.0, delta_t, s, r, b)
        values = {
            's': s,
            'r': r,
            'b': b,
            'initial_state': initial_state,
        }
        return hmmds.applications.laser.utilities.Parameters(
            s,
            r,
            b,
            *initial_state,
            t_ratio,
            x_ratio,
            offset,
        )

    def calculate_likelihood(self):
        n_data = 100
        non_stationary, initial_distribution, _ = hmmds.applications.laser.utilities.make_non_stationary(
            self.get_parameters(), None)
        like_per_step = numpy.empty(n_data)
        laser_data = self.laser_data[1:, :n_data].T
        result = non_stationary.log_likelihood(initial_distribution, laser_data,
                                               like_per_step)
        print(f"""objective_function = {result}""")
        self.like_curve.setData(like_per_step)

    def write_values(self):
        with open('explore.txt', 'w') as file_:
            for name, variable in self.variable.items():
                file_.write(f'{name} {variable()}\n')

    def read_values(self):
        with open('explore.txt', 'r') as file_:
            for line in file_.readlines():
                name, value_str = line.split()
                value = float(value_str)
                self.variable[name].setValue(value)

    def update_plot(self):
        over_sample = 5
        laser_t = 0.04
        data = make_data(self, laser_t, over_sample)
        self.pp_curve.setData(data[:, 0], data[:, 2])
        simulated_data = self.x_ratio() * data[:, 0]**2 + self.offset()
        n_simulated = len(simulated_data)
        n_data = int(n_simulated / over_sample)
        times = numpy.linspace(0,
                               n_simulated * laser_t / over_sample,
                               n_simulated,
                               endpoint=False)
        self.ts_curve.setData(times, simulated_data)
        self.laser_plot.setData(self.laser_data[:, :n_data].T)

        non_stationary, initial_distribution, _ = hmmds.applications.laser.utilities.make_non_stationary(
            self.get_parameters(), None)
        like_per_step = numpy.empty(n_data)
        laser_data = self.laser_data[1:, :n_data].T
        result = non_stationary.log_likelihood(initial_distribution, laser_data,
                                               like_per_step)
        self.like_curve.setData(like_per_step)


class Variable(PyQt5.QtWidgets.QWidget):
    """Provide sliders and spin boxes to manipulate variable.

    Args:
        label:
        minimum:
        maximum:
        main_window: For access to method update_plot
        parent:  I don't understand this
    """

    def __init__(
            self,  # Variable
            label: str,
            minimum: float,
            maximum: float,
            main_window,
            parent=None):
        super(Variable, self).__init__(parent=parent)

        # Instantiate widgets
        self.label = PyQt5.QtWidgets.QLabel(self)
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
        self.spin.setSingleStep(0.001)
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
        self.verticalLayout = PyQt5.QtWidgets.QVBoxLayout(self)
        self.verticalLayout.addWidget(self.label)
        self.verticalLayout.addWidget(self.slider)
        self.verticalLayout.addWidget(self.spin)
        self.resize(self.sizeHint())

    def __call__(self):
        return self.x

    def setValue(self, value):
        self.spin.setValue(value)

    def spin_changed(self, value):
        self.x = value
        self.slider.disconnect()  # Avoid loop with setValue
        self.slider.setValue(self.slider.minimum() +
                             int((value - self.minimum) / self.dx_dslide))
        self.slider.valueChanged.connect(self.slider_changed)
        self.main_window.update_plot()

    def slider_changed(
            self,  # Variable
            value):
        self.x = self.minimum + float(value) * self.dx_dslide
        self.spin.disconnect()  # Avoid loop with setValue
        self.spin.setValue(self.x)
        self.spin.valueChanged.connect(self.spin_changed)
        self.main_window.update_plot()


if __name__ == '__main__':
    app = PyQt5.QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
