"""h_view.py For exploring cross entropy of extended Kalman filter

ToDo: Discrete y to get probability masses instead of densities
ToDo: Calculate and plot log p(y[t]|y[:t])
ToDo: Calculate update_means and update_covariances

"""

# PyQt5 is hopeless: pylint: skip-file
import sys  # We need sys so that we can pass argv to QApplication
import os
import typing

import PyQt5.QtWidgets
import pyqtgraph

import numpy
import numpy.linalg
import scipy.linalg

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
    print(f'''eigenvalues3={vals3} {vals3.max()/vals3.min():.3g}
eigenvalues2={vals2}  {vals2.max()/vals2.min():.3g}
vals_sqrt=   {vals_sqrt}  {vals_sqrt.max()/vals_sqrt.min():.3g}
''')
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

        run_button = PyQt5.QtWidgets.QPushButton('Run', self)
        run_button.clicked.connect(self.run)
        buttons_layout.addWidget(run_button)

        filter_button = PyQt5.QtWidgets.QPushButton('Filter', self)
        filter_button.clicked.connect(self.filter)
        buttons_layout.addWidget(filter_button)

        save_button = PyQt5.QtWidgets.QPushButton('Save', self)
        save_button.clicked.connect(self.save)
        buttons_layout.addWidget(save_button)

        self.variable = {}  # A dict so that I can print all values

        # Layout first row of sliders.  "updates" is a list of methods
        # to call when a varible changes
        for name, title, minimum, maximum, updates in (
            ('n_times', 'Nt', 5, 500, 'update_data update_filter update_plot'),
            ('n_view', 'N_view', 1, 100, 'update_plot'),
            ('t_view', 't_view', 0, 500, 'update_plot'),
        ):
            self.variable[name] = IntVariable(title, minimum, maximum, self,
                                              updates)
            sliders1_layout.addWidget(self.variable[name])

        # Layout second row of sliders
        for name, title, minimum, maximum, updates in (
            ('time_step', 'ts', 0.05, 0.5,
             'update_system update_data update_filter update_plot'),
            ('y_step', 'y_step*100', 1.0e-3, 2.0e-2,
             'update_filter update_plot'),
            ('dev_observation', 'Dev_y', 0.001, 0.1,
             'update_system update_data update_filter update_plot'
            ),  # sigma epsilon observation noise
        ):
            self.variable[name] = FloatVariable(title, minimum, maximum, self,
                                                updates)
            sliders2_layout.addWidget(self.variable[name])

        # Layout third row of sliders
        for name, title, minimum, maximum, updates in (
            ('view_theta', '\u03b8', -3.0, 3.0, 'update_plot'),  # View theta
            ('view_phi', '\u03c6', -3.0, 3.0, 'update_plot'),  # View phi
                # Times 1e4, fixme scale by root dt
            ('dev_state', 'Dev_x*1e4', 1.0e-3, 1.0e-1,
             'update_system update_data update_filter update_plot'
            ),  # sigma eta state noise
        ):
            self.variable[name] = FloatVariable(title, minimum, maximum, self,
                                                updates)
            sliders3_layout.addWidget(self.variable[name])

        # Enable access like self.n_times() instead of self.variable['n_times']()
        self.__dict__.update(self.variable)

        # Define widgets for plot section

        # Time series x[0]
        temp = time_series.addPlot()
        self.ts_curve = temp.plot(pen='g')
        self.ts_point = temp.plot(symbolPen='w', symbol='+', symbolSize=30)

        # y[t] - mean of p(y[t]|y[:t])
        temp = error.addPlot()
        self.error_curve = temp.plot(pen='g')
        self.error_point = temp.plot(symbolPen='w', symbol='+', symbolSize=30)
        self.sigma_curve = temp.plot(pen='r')

        # log p(y[t]|y[:t])
        temp = probability.addPlot()
        self.probability_curve = temp.plot(pen='g')
        self.probability_point = temp.plot(symbolPen='w',
                                           symbol='+',
                                           symbolSize=30)

        # x[0] vs x[2]
        temp = phase_portrait.addPlot()
        self.pp_curve = temp.plot(pen='g')

        # ekf at t
        temp = ekf_t.addPlot()
        self.ekf_forecast_t_curve = temp.plot(pen='r')
        self.ekf_update_t_curve = temp.plot(pen='g')

        # ekf at t+1
        temp = ekf_t_plus.addPlot()
        self.ekf_forecast_t_plus_curve = temp.plot(pen='r')
        self.ekf_update_t_plus_curve = temp.plot(pen='g')

        # Make self the central widget
        widget = PyQt5.QtWidgets.QWidget()
        widget.setLayout(layout0)
        self.setCentralWidget(widget)

        # Do all calculations to initialize
        self.update_system()
        self.n_times.spin.setValue(500)

    def update_system(self):
        """Make a new self.system
        """
        s = 10.0
        r = 28.0
        b = 8.0 / 3

        rng = numpy.random.default_rng(3)
        d_t = self.time_step()
        state_noise_scale = self.dev_state() * numpy.sqrt(d_t) / 1.0e4
        self.system, self.stationary_distribution, self.initial_state = hmmds.synthetic.bounds.lorenz.make_system(
            s, r, b, state_noise_scale, self.dev_observation(), d_t, rng)

    def update_data(self):
        """Make new self.x and self.y
        """
        # Reinitialize rng for reproducibility
        self.system.rng = numpy.random.default_rng(3)
        self.x, self.y = self.system.simulate_n_steps(
            self.stationary_distribution, self.n_times(),
            self.y_step() / 100)

    def update_filter(self):
        """Run extended Kalman filter to get forecast and updated distributions

        ToDo: Fix me

        hmm.state_space.LinearStationary.forward_filter Does not even see forecast
        hmm.state_space.NonStationary.forward_step Does not save forecast
        hmm.lorenz_sde.SDE.forecast
        hmm.state_space.SDE.update
        """
        self.forecast_means, self.forecast_covariances, self.update_means, self.update_covariances, self.y_means, self.y_variances, self.y_probabilities = self.system.forward_filter(
            self.stationary_distribution, self.y, self.y_step())

    def update_plot(self):
        # Calculate range of samples to display
        n_max = max(1,
                    min(self.n_times(), int(self.t_view() + self.n_view() / 2)))
        assert 1 <= n_max <= self.n_times()

        n_min = min(self.n_times() - 1,
                    max(0, int(self.t_view() - self.n_view() / 2)))
        assert 1 <= n_min <= self.n_times() - 1

        assert n_min < n_max
        times = numpy.arange(n_min, n_max)

        t_now = self.t_view()
        # Plot time series
        self.ts_curve.setData(times, self.y[n_min:n_max, 0])
        self.ts_point.setData([
            t_now,
        ], [self.y[t_now, 0]])

        # Plot phase portrait
        self.pp_curve.setData(self.x[n_min:n_max, 0], self.x[n_min:n_max, 2])

        # Plot errors
        error = self.y[:, 0] - self.y_means

        # FixMe: 100
        self.sigma_curve.setData(times,
                                 100 * numpy.sqrt(self.y_variances[n_min:n_max]))
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
        self.probability_curve.setData(times, self.y_probabilities[n_min:n_max])
        self.probability_point.setData([
            t_now,
        ], [
            self.y_probabilities[t_now],
        ])

        # Plot ellipses for t+1
        t_next = self.t_view() + 1
        temp = ellipse(self.forecast_means[t_next],
                       self.forecast_covariances[t_next])
        self.ekf_forecast_t_plus_curve.setData(temp[:, 0], temp[:, 1])
        temp = ellipse(self.update_means[t_next],
                       self.update_covariances[t_next])
        self.ekf_update_t_plus_curve.setData(temp[:, 0], temp[:, 1])

    def run(self):
        pass

    def filter(self):
        pass

    def save(self):
        """Save slider settings and plotted data

        ToDo: Actually save the stuff using pickle.  Maybe pop up a
        chooser for path to file """
        with open('foo.txt', 'w') as file_:
            for name, variable in self.variable.items():
                file_.write(f'{name} {variable()}\n')

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

    def modify_properties(self, maximum, minimum):
        """Isolate differences between float and int Varibles
        """

        self.spin = PyQt5.QtWidgets.QSpinBox(self)
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.dx_dslide = 1
        self.x = int((minimum + maximum) / 2)
        self.slider.setValue(self.x)

    def __init__(
            self,  # IntVariable
            title: str,
            minimum: int,
            maximum: int,
            main_window,
            update_names,
            parent=None):
        super(IntVariable, self).__init__(parent=parent)

        self.update_names = update_names.split()
        # Instantiate widgets
        self.label = PyQt5.QtWidgets.QLabel(self)
        self.slider = PyQt5.QtWidgets.QSlider(self)
        self.modify_properties(maximum, minimum)

        # Attach arguments to self and widgets
        self.label.setText(title)
        self.spin.setRange(minimum, maximum)
        self.spin.setValue(self.x)
        self.minimum = minimum
        self.maximum = maximum
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

    def __call__(self):
        return self.x

    def setValue(self, value):
        self.spin.setValue(value)

    def new_spin_value(self, value):
        """Seperate method to isolate difference between int and float classes
        """
        return value

    def new_slider_value(
            self,  # IntVariable
            value):
        """Seperate method to isolate difference between int and float classes
        """
        return value

    def update(self):
        """Call each of the update methods for this variable.
        """
        for name in self.update_names:
            getattr(self.main_window, name)()

    def spin_changed(self, value):
        self.x = value
        self.slider.disconnect()  # Avoid loop with setValue
        self.slider.setValue(self.new_spin_value(value))
        self.slider.valueChanged.connect(self.slider_changed)
        self.update()

    def slider_changed(self, value):
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

    def modify_properties(
            self,  # FloatVariable
            maximum,
            minimum):
        """Isolate differences between float and int Varibles
        """

        self.spin = PyQt5.QtWidgets.QDoubleSpinBox(self)

        if maximum > 0.01:
            self.spin.setDecimals(3)
            self.spin.setSingleStep(0.001)
        # self.slider.minimum() = 0, self.slider.maximum() = 99
        self.dx_dslide = (maximum - minimum) / (self.slider.maximum() -
                                                self.slider.minimum())
        self.slider.setValue(
            int((self.slider.maximum() + self.slider.minimum()) / 2))
        self.x = (minimum + maximum) / 2

    def new_spin_value(self, value):
        """Seperate method to isolate difference between int and float classes
        """
        return self.slider.minimum() + int(
            (value - self.minimum) / self.dx_dslide)

    def new_slider_value(
            self,  # FloatVariable
            value):
        """Seperate method to isolate difference between int and float classes
        """
        return self.minimum + float(value) * self.dx_dslide


if __name__ == '__main__':
    app = PyQt5.QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
