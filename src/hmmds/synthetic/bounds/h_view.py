"""h_view.py For exploring cross entropy of extended Kalman filter

ToDo: Discrete y to get probability masses instead of densities
ToDo: Calculate and plot log p(y[t]|y[:t])

"""
from __future__ import annotations  # Enables, eg, (self: LocalNonStationary

# PyQt5 is hopeless: pylint: skip-file
import sys  # We need sys so that we can pass argv to QApplication
import os
import typing
import pickle

import PyQt5.QtWidgets
import pyqtgraph

import numpy
import numpy.linalg
import scipy.linalg

import hmm.state_space
import hmmds.synthetic.filter.lorenz_sde
import hmmds.synthetic.bounds.lorenz


def ellipse(mean, covariance, i_a=1, i_b=2):
    """ Calculate points on x^T \Sigma^{-1} x = 1

    Args:
        mean: 3-vector
        covariance: 3x3 array
        i_a: index of first component
        i_b: index of second component
    """
    mean_2 = numpy.array([mean[i_a], mean[i_b]])
    covariance_2 = numpy.array([[covariance[i_a, i_a], covariance[i_a, i_b]],
                                [covariance[i_b, i_a], covariance[i_b, i_b]]])
    sqrt_cov_2 = scipy.linalg.sqrtm(covariance_2)
    n_points = 100
    theta = numpy.linspace(0, 2 * numpy.pi, n_points, endpoint=True)
    z = numpy.array([numpy.sin(theta), numpy.cos(theta)]).T
    result = numpy.dot(z, sqrt_cov_2) + mean_2

    vals3, vecs3 = numpy.linalg.eigh(covariance)
    vals2, vecs2 = numpy.linalg.eigh(covariance_2)
    vals_sqrt, vecs_sqrt = numpy.linalg.eigh(sqrt_cov_2)
    return result


class MainWindow(PyQt5.QtWidgets.QMainWindow):
    """

-----------------------------------------------------------------
 Buttons              |                      |
                      |                      |
                      |                      |
                      |                      |
 Sliders              |  x_0[t] vs t         |  x_0[t] vs x_2[t]
                      |                      |
                      |                      |
                      |                      |
                      |                      |
-----------------------------------------------------------------
                      |                      |
                      |                      |
                      | Sqrt Sigma[0,0] vs t |
 Sliders              | Error y vs t         |  Ellipses for t
                      |                      |
                      |                      |
                      |                      |
                      |                      |
-----------------------------------------------------------------
                      |                      |
                      |                      |
                      |                      |
 Sliders              | log p(y[t]|y[:t])    |  Ellipses for t+1
                      |                      |
                      |                      |
                      |                      |
                      |                      |
-----------------------------------------------------------------

    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Exploring Lorenz Parameters")

        # Layout the main window skeleton
        layout0 = PyQt5.QtWidgets.QHBoxLayout()
        control_layout = PyQt5.QtWidgets.QVBoxLayout()
        plot_layout = PyQt5.QtWidgets.QHBoxLayout()
        layout0.addLayout(control_layout)
        layout0.addLayout(plot_layout)

        # Layout the control section skeleton
        buttons_layout = PyQt5.QtWidgets.QHBoxLayout()
        sliders1_layout = PyQt5.QtWidgets.QHBoxLayout()
        sliders2_layout = PyQt5.QtWidgets.QHBoxLayout()
        sliders3_layout = PyQt5.QtWidgets.QHBoxLayout()
        control_layout.addLayout(buttons_layout)
        control_layout.addLayout(sliders1_layout)
        control_layout.addLayout(sliders2_layout)
        control_layout.addLayout(sliders3_layout)

        # Layout the plot section skeleton
        plot_right = PyQt5.QtWidgets.QVBoxLayout()
        plot_left = PyQt5.QtWidgets.QVBoxLayout()
        plot_layout.addLayout(plot_left)
        plot_layout.addLayout(plot_right)
        time_series = pyqtgraph.GraphicsLayoutWidget(title="Time Series")
        error = pyqtgraph.GraphicsLayoutWidget(title="Error")
        probability = pyqtgraph.GraphicsLayoutWidget(title="Probability")
        phase_portrait = pyqtgraph.GraphicsLayoutWidget(title="Phase Portrait")
        ekf_t = pyqtgraph.GraphicsLayoutWidget(title="ekf t")
        ekf_t_plus = pyqtgraph.GraphicsLayoutWidget(title="ekf t+1")
        plot_left.addWidget(time_series)
        plot_left.addWidget(error)
        plot_left.addWidget(probability)
        plot_right.addWidget(phase_portrait)
        plot_right.addWidget(ekf_t)
        plot_right.addWidget(ekf_t_plus)

        # Define and layout widgets of button section
        quit_button = PyQt5.QtWidgets.QPushButton('Quit', self)
        quit_button.clicked.connect(self.close)
        buttons_layout.addWidget(quit_button)

        debug_button = PyQt5.QtWidgets.QPushButton('Debug', self)
        debug_button.clicked.connect(self.debug)
        buttons_layout.addWidget(debug_button)

        self.hold_button = PyQt5.QtWidgets.QPushButton('SciPy', self)
        self.hold_button.clicked.connect(self.hold)
        buttons_layout.addWidget(self.hold_button)

        save_button = PyQt5.QtWidgets.QPushButton('Save', self)
        save_button.clicked.connect(self.save)
        buttons_layout.addWidget(save_button)

        self.variable = {}  # A dict for the variables

        # Layout first row of sliders.  "updates" is a list of methods
        # to call when a varible changes
        for name, title, minimum, maximum, initial, updates in (
            ('n_times', 'Nt', 5, 500, 500,
             'update_data update_filter update_plot'),
            ('n_view', 'N_view', 1, 100, 30, 'update_plot'),
            ('t_view', 't_view', 0, 500, 15, 'update_plot'),
        ):
            self.variable[name] = IntVariable(title, minimum, maximum, initial,
                                              self, updates)
            sliders1_layout.addWidget(self.variable[name])

        # Layout second row of sliders
        for name, title, minimum, maximum, initial, updates in (
            ('time_step', 'ts', 0.05, 0.5, 0.25,
             'update_system update_data update_filter update_plot'),
            ('y_step', 'Dy*100', 1.0e-4, 2.0e-2, 0.01,
             'update_filter update_plot'),
            ('dev_observation', '\u03c3y*1000', 0.001, 0.1, 0.01,
             'update_system update_data update_filter update_plot'
            ),  # sigma epsilon observation noise
        ):
            self.variable[name] = FloatVariable(title, minimum, maximum,
                                                initial, self, updates)
            sliders2_layout.addWidget(self.variable[name])
        self.variable['y_step'].modify_scale(1.0/100.0)
        self.variable['dev_observation'].modify_scale(1.0e-3)

        # Layout third row of sliders
        for name, title, minimum, maximum, initial, updates in (
            ('view_theta', '\u03b8', -3.0, 3.0, 0.0,
             'update_plot'),  # View theta
            ('view_phi', '\u03c6', -3.0, 3.0, 0.0, 'update_plot'),  # View phi
            ('dev_state', '\u03C3x*1e4', 1.0e-3, 2.0e-2, 0.01,
             'update_system update_data update_filter update_plot'
            ),  # sigma eta state noise
        ):
            self.variable[name] = FloatVariable(title, minimum, maximum,
                                                initial, self, updates)
            sliders3_layout.addWidget(self.variable[name])
        self.variable['dev_state'].modify_scale(1.0e-4)

        # Enable access like self.n_times() instead of self.variable['n_times']()
        self.__dict__.update(self.variable)

        # Define widgets for plot section

        # Upper left. Time series x[0]
        temp = time_series.addPlot()
        temp.addLegend()
        self.ts_curve = temp.plot(pen='g', name="y")
        self.ts_point = temp.plot(symbolPen='w', symbol='+', symbolSize=30)
        self.y_forecast = temp.plot(pen='r', name="forecast")

        # Center left.  y[t] - mean of p(y[t]|y[:t])
        temp = error.addPlot()
        temp.addLegend()
        self.error_curve = temp.plot(pen='g', name='error')
        self.error_point = temp.plot(symbolPen='w', symbol='+', symbolSize=30)
        self.sigma_curve = temp.plot(pen='r', name='sigma')

        # Lower left. log p(y[t]|y[:t])
        temp = probability.addPlot()
        temp.addLegend()
        self.probability_curve = temp.plot(pen='g', name='log p(y[t]|y[:t])')
        self.probability_point = temp.plot(symbolPen='w',
                                           symbol='+',
                                           symbolSize=30)

        # Upper right. x[0] vs x[2]
        temp = phase_portrait.addPlot()
        temp.addLegend()
        self.pp_curve = temp.plot(pen='g', name='x')
        self.pp_forecast = temp.plot(pen='r', name='forecast')
        self.pp_update = temp.plot(pen='y', name='update')
        self.pp_point = temp.plot(symbolPen='w', symbol='+', symbolSize=30)

        # Center right.  ekf at t
        temp = ekf_t.addPlot()
        temp.addLegend()
        self.ekf_forecast_t_curve = temp.plot(pen='r', name='forecast')
        self.ekf_update_t_curve = temp.plot(pen='g', name='update')

        # Lower right.  ekf at t+1
        temp = ekf_t_plus.addPlot()
        temp.addLegend()
        self.ekf_forecast_t_plus_curve = temp.plot(pen='r', name='forecast')
        self.ekf_update_t_plus_curve = temp.plot(pen='g', name='update')

        # Make self the central widget
        widget = PyQt5.QtWidgets.QWidget()
        widget.setLayout(layout0)
        self.setCentralWidget(widget)

        # Do all calculations to initialize
        self.update_system()
        self.update_data()
        self.update_filter()
        self.update_plot()

    def update_system(self):
        """Make a new self.system
        """
        s = 10.0
        r = 28.0
        b = 8.0 / 3

        rng = numpy.random.default_rng(3)
        d_t = self.time_step()
        # In state_space.SDE.forecast, the covariance ends up being dt
        # * state_noise_scale**2.  So dividing by sqrt(dt) here makes
        # self.dev_state the actual noise scale.
        state_noise_scale = self.dev_state() / numpy.sqrt(d_t)
        self.__dict__.update(hmmds.synthetic.bounds.lorenz.make_system(
            s, r, b, state_noise_scale, self.dev_observation(), d_t, rng))
        self.system = self.SciPy
        self.initial_distribution = hmm.state_space.MultivariateNormal(
            self.initial_state,
            numpy.eye(3) * state_noise_scale**2,
            self.stationary_distribution.rng)

    def update_data(self):
        """Make new self.x and self.y
        """
        # Reinitialize rng for reproducibility
        self.system.rng = numpy.random.default_rng(3)
        self.x, self.y = self.system.simulate_n_steps(self.initial_distribution,
                                                      self.n_times(),
                                                      self.y_step())

    def update_filter(self):
        """Run extended Kalman filter to get forecast and updated distributions

        ToDo: Fix me

        hmm.state_space.LinearStationary.forward_filter Does not even see forecast
        hmm.state_space.NonStationary.forward_step Does not save forecast
        hmm.lorenz_sde.SDE.forecast
        hmm.state_space.SDE.update
        """
        self.forecast_means, self.forecast_covariances, self.update_means, self.update_covariances, self.y_means, self.y_variances, temp_prob = self.system.forward_filter(
            self.initial_distribution, self.y,
            self.y_step())
        self.log_probabilities = numpy.log(numpy.maximum(temp_prob, 1.0e-20))

    def update_plot(self):
        # Calculate range of samples to display
        n_max = max(1,
                    min(self.n_times(), int(self.t_view() + self.n_view() / 2)))
        assert 1 <= n_max <= self.n_times()

        n_min = min(self.n_times() - 1,
                    max(0, int(self.t_view() - self.n_view() / 2)))
        assert 0 <= n_min < self.n_times()

        assert n_min < n_max
        times = numpy.arange(n_min, n_max)
        t_now = self.t_view()

        # Plot time series
        self.ts_curve.setData(times, self.y[n_min:n_max, 0])
        self.y_forecast.setData(times, self.y_means[n_min:n_max])
        self.ts_point.setData([
            t_now,
        ], [self.y[t_now, 0]])

        # Plot phase portrait
        self.pp_curve.setData(self.x[n_min:n_max, 0], self.x[n_min:n_max, 2])
        self.pp_point.setData([self.x[t_now, 0]], [self.x[t_now, 2]])
        self.pp_update.setData(self.update_means[n_min:n_max, 0],
                               self.update_means[n_min:n_max, 2])
        self.pp_forecast.setData(self.forecast_means[n_min:n_max, 0],
                                 self.forecast_means[n_min:n_max, 2])

        # Plot errors
        error = self.y[:, 0] - self.y_means

        self.sigma_curve.setData(times,
                                 numpy.sqrt(self.y_variances[n_min:n_max]))
        self.error_curve.setData(times, error[n_min:n_max])
        self.error_point.setData([
            t_now,
        ], [
            error[t_now],
        ])

        # Plot ellipses for t
        temp = ellipse(self.forecast_means[t_now],
                       self.forecast_covariances[t_now])
        self.ekf_forecast_t_curve.setData(temp[:, 0], temp[:, 1])
        temp = ellipse(self.update_means[t_now], self.update_covariances[t_now])
        self.ekf_update_t_curve.setData(temp[:, 0], temp[:, 1])

        # Plot probability vs t
        self.probability_curve.setData(times,
                                       self.log_probabilities[n_min:n_max])
        self.probability_point.setData([t_now], [self.log_probabilities[t_now]])

        # Plot ellipses for t+1
        t_next = self.t_view() + 1
        temp = ellipse(self.forecast_means[t_next],
                       self.forecast_covariances[t_next])
        self.ekf_forecast_t_plus_curve.setData(temp[:, 0], temp[:, 1])
        temp = ellipse(self.update_means[t_next],
                       self.update_covariances[t_next])
        self.ekf_update_t_plus_curve.setData(temp[:, 0], temp[:, 1])

    def debug(self):
        """Print debugging information
        """

        def analyze_print(name, distribution=None, mean=None, covariance=None):
            if distribution:
                assert mean is None
                assert covariance is None
                mean = distribution.mean
                covariance = distribution.covariance
            vals, vecs = numpy.linalg.eigh(covariance)
            # Check that the column vecs[:,0] is a left eigenvector
            assert numpy.allclose(numpy.dot(covariance, vecs[:, 0]),
                                  vecs[:, 0] * vals[0])
            print(f"""   {name}
mean = {mean}
covariance =
{covariance}
eigenvectors are columns of:
{vecs}
Eigenvalues={vals}
""")
            return

        now = self.t_view()

        print(f"""------------------------------------------------------------
   y[{now}]={self.y[now]}""")
        analyze_print(f'forecast[{now}]',
                      mean=self.forecast_means[now],
                      covariance=self.forecast_covariances[now])
        analyze_print(f'update[{now}]',
                      mean=self.update_means[now],
                      covariance=self.update_covariances[now])

    def hold(self):
        """Choose Cython or SciPy
        """
        old_text = self.hold_button.text()
        if old_text == 'Cython':
            self.hold_button.setText('SciPy')
        elif old_text == 'SciPy':
            self.hold_button.setText('Cython')
        else:
            raise RuntimeError
        self.system = getattr(self, self.hold_button.text())
        self.update_data()
        self.update_filter()
        self.update_plot()

    def save(self):
        """Save slider settings and plotted data

        ToDo: Actually save the stuff using pickle.  Maybe pop up a
        chooser for path to file """
        dump_dict = {}
        for name in 'forecast_means forecast_covariances update_means update_covariances y_means log_probabilities'.split():
            dump_dict[name] = getattr(self, name)
        for name, variable in self.variable.items():
            dump_dict[name] = variable()
        with open('data_h_view', 'wb') as _file:
            pickle.dump(dump_dict,_file)

    def read_values(self):
        """Maybe write this to read slider values
        """
        pass


class IntVariable(PyQt5.QtWidgets.QWidget):
    """Provide sliders and spin boxes for an integer variable.

    Args:
        title: For display
        minimum:
        maximum:
        main_window: For access to method update_plot
        parent:  I don't understand this

    """

    def modify_properties(self: IntVariable, maximum: int, minimum: int,
                          initial: int):
        """Isolate differences between float and int Varibles
        """
        self.minimum = minimum
        self.maximum = maximum
        self.spin = PyQt5.QtWidgets.QSpinBox(self)
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.dx_dslide = 1
        self.x = initial
        self.slider.setValue(self.x)

    def modify_scale(self, scale):
        """Set scale for __call__
        """
        self.scale = scale

    def __init__(self: IntVariable,
                 title: str,
                 minimum: int,
                 maximum: int,
                 initial: int,
                 main_window,
                 update_names,
                 scale=1,
                 parent=None):
        super(IntVariable, self).__init__(parent=parent)

        self.update_names = update_names.split()
        # Instantiate widgets
        self.label = PyQt5.QtWidgets.QLabel(self)
        self.slider = PyQt5.QtWidgets.QSlider(self)
        self.modify_properties(maximum, minimum, initial)

        # Attach arguments to self and widgets
        self.scale=scale
        self.label.setText(title)
        self.spin.setRange(minimum, maximum)
        self.spin.setValue(self.x)
        self.main_window = main_window

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

    def __call__(self: IntVariable):
        return self.x * self.scale

    def setValue(self: IntVariable, value):
        self.spin.setValue(value)

    def new_spin_value(self: IntVariable, value):
        """Seperate method to isolate difference between int and float classes
        """
        return value

    def new_slider_value(self: IntVariable, value):
        """Seperate method to isolate difference between int and float classes
        """
        return value

    def update(self: IntVariable):
        """Call each of the update methods for this variable.
        """
        for name in self.update_names:
            getattr(self.main_window, name)()

    def spin_changed(self: IntVariable, value):
        self.x = value
        self.slider.disconnect()  # Avoid loop with setValue
        self.slider.setValue(self.new_spin_value(value))
        self.slider.valueChanged.connect(self.slider_changed)
        self.update()

    def slider_changed(self: IntVariable, value):
        self.x = self.new_slider_value(value)
        self.spin.disconnect()  # Avoid loop with setValue
        self.spin.setValue(self.x)
        self.spin.valueChanged.connect(self.spin_changed)
        self.update()


class FloatVariable(IntVariable):
    """Provide sliders and spin boxes to manipulate float variable.

    Args:
        title: For display
        minimum:
        maximum:
        main_window: For access to method update_plot
        parent:  I don't understand this
    """

    def modify_properties(self: FloatVariable, maximum: float, minimum: float,
                          initial: float):
        """Isolate differences between float and int Varibles

        Args:
            maximum: Actual float value
            minimum: Actual float value
            initial: Actual float value
        """

        self.spin = PyQt5.QtWidgets.QDoubleSpinBox(self)

        self.maximum = maximum
        self.minimum = minimum
        if maximum > 0.01:
            self.spin.setDecimals(3)
            self.spin.setSingleStep(0.001)
        # self.slider.minimum() = 0, self.slider.maximum() = 99
        self.dx_dslide = (maximum - minimum) / (self.slider.maximum() -
                                                self.slider.minimum())
        self.slider.setValue(self.new_spin_value(initial))
        self.x = initial

    def new_spin_value(self: FloatVariable, value: float):
        """Seperate method to isolate difference between int and float classes

        Args:
            value: Float from spin

        Return: Integer for slider
        
        """
        return self.slider.minimum() + int(
            (value - self.minimum) / self.dx_dslide)

    def new_slider_value(self: FloatVariable, value: int):
        """Seperate method to isolate difference between int and float classes
        """
        return self.minimum + float(value) * self.dx_dslide


if __name__ == '__main__':
    app = PyQt5.QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
