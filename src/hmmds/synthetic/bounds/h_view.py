"""h_view.py For exploring cross entropy of extended Kalman filter

"""

# PyQt5 is hopeless: pylint: skip-file
import sys  # We need sys so that we can pass argv to QApplication
import os
import typing

import PyQt5.QtWidgets
import pyqtgraph

import numpy
import numpy.linalg

from hmmds.synthetic.filter.lorenz_sde import lorenz_integrate


# values: n_times, n_view, t_view, time_step
def make_data(values) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
#def make_data(values: MainWindow) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
    """Integrate the Lorenz system

    Args:
        values  Instance that provides access to Variables
        laser_t Sampling period of laser data
        over_sample Number of synthetic data points per laser data point
    """
    s = 10.0
    r = 28.0
    b = 8.0 / 3
    initial_state = lorenz_integrate(numpy.ones(3), 0.0, 100.0, s, r, b)
    t_sample = values.time_step()
    n_samples = values.n_times()

    def observe(x):
        return x[0:1]
    x = numpy.empty((n_samples, 3))
    y = numpy.empty((n_samples, 1))
    x[0] = initial_state
    y[0] = observe(x[0])
    for i in range(1, n_samples):
        x[i] = lorenz_integrate(x[i - 1], 0, t_sample, s, r, b)
        y[i] = observe(x[i])
    return x, y


class MainWindow(PyQt5.QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Exploring Lorenz Parameters")

        # Layout the main window skeleton
        layout0 = PyQt5.QtWidgets.QHBoxLayout()
        control_layout = PyQt5.QtWidgets.QVBoxLayout()
        plot_layout = PyQt5.QtWidgets.QVBoxLayout()
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
        time_series = pyqtgraph.GraphicsLayoutWidget(title="Time Series")
        error = pyqtgraph.GraphicsLayoutWidget(title="Error")
        probability = pyqtgraph.GraphicsLayoutWidget(title="Probability")
        phase_portrait = pyqtgraph.GraphicsLayoutWidget(title="Phase Portrait")
        plot_layout.addWidget(time_series)
        plot_layout.addWidget(error)
        plot_layout.addWidget(probability)
        plot_layout.addWidget(phase_portrait)
        
        # Define and layout widgets of button section
        run_button = PyQt5.QtWidgets.QPushButton('Run', self)
        run_button.clicked.connect(self.run)
        buttons_layout.addWidget(run_button)
        
        filter_button = PyQt5.QtWidgets.QPushButton('Filter', self)
        filter_button.clicked.connect(self.filter)
        buttons_layout.addWidget(filter_button)
        
        save_button = PyQt5.QtWidgets.QPushButton('Save', self)
        save_button.clicked.connect(self.save)
        buttons_layout.addWidget(save_button)
        
        quit_button = PyQt5.QtWidgets.QPushButton('Quit', self)
        quit_button.clicked.connect(self.close)
        buttons_layout.addWidget(quit_button)
        
        self.variable = {}  # A dict so that I can print all values

        # Layout first row of sliders
        for name, title, minimum, maximum in (
            ('n_times', 'Nt', 5, 500),
            ('n_view', 'N_view', 1, 100),
            ('t_view', 't_view', 0, 500),
        ):
            self.variable[name] = IntVariable(title, minimum, maximum, self)
            sliders1_layout.addWidget(self.variable[name])

        # Layout second row of sliders
        for name, title, minimum, maximum in (
            ('time_step', 'ts', 0.05, 0.5),
            ('y_step', 'y_step', 0.05, 0.5),
            ('dev_observation', '\u03c3\u03b5', 0.001, 0.1),  # sigma epsilon observation noise
        ):
            self.variable[name] = FloatVariable(title, minimum, maximum, self)
            sliders2_layout.addWidget(self.variable[name])

        # Layout third row of sliders
        for name, title, minimum, maximum in (
            ('view_theta', '\u03b8', -3.0, 3.0),  # View theta
            ('view_phi', '\u03c6', -3.0, 3.0),  # View phi
            ('dev_state', '\u03c3\u03b7', 1.0e-7, 1.0e-5),  # sigma eta state noise
        ):
            self.variable[name] = FloatVariable(title, minimum, maximum, self)
            sliders3_layout.addWidget(self.variable[name])

        # Enable access like self.n_times.  Perhaps better to access self.variable['n_times']
        self.__dict__.update(self.variable)

        # Define widgets for plot section

        ts_plot = time_series.addPlot()
        self.ts_curve = ts_plot.plot(pen='g')

        error_plot = error.addPlot()
        self.error_curve = error_plot.plot(pen='g')

        probability_plot = probability.addPlot()
        self.probability_curve = probability_plot.plot(pen='g')
        
        pp_plot = phase_portrait.addPlot()
        self.pp_curve = pp_plot.plot(pen='g')

        # Make self the central widget
        widget = PyQt5.QtWidgets.QWidget()
        widget.setLayout(layout0)
        self.setCentralWidget(widget)

        self.update_plot()  # Plot data for initial settings

    def run(self):
        pass
    def filter(self):
        pass
    def save(self):
        with open('foo.txt', 'w') as file_:
            for name, variable in self.variable.items():
                file_.write(f'{name} {variable()}\n')

    def read_values(self):
        with open('explore.txt', 'r') as file_:
            for line in file_.readlines():
                name, value_str = line.split()
                value = float(value_str)
                self.variable[name].setValue(value)

    def update_plot(self):
        x, y = make_data(self)
        assert len(x) == len(y) == self.n_times()

        # Calculate range of samples to display
        n_max = min(self.n_times(), int(self.t_view() + self.n_view()/2))
        n_min = max(1, int(self.t_view() - self.n_view()/2))
        assert n_min < self.n_times() - 1
        assert n_max > 2
        assert n_min < n_max
        
        self.pp_curve.setData(x[n_min:n_max, 0], x[n_min:n_max, 2])
        
        times = numpy.arange(n_min, n_max)* self.time_step()
        self.ts_curve.setData(times, y[n_min:n_max,0])
        self.error_curve.setData(times, y[n_min:n_max,0])
        self.probability_curve.setData(times, y[n_min:n_max,0])


class IntVariable(PyQt5.QtWidgets.QWidget):
    """Provide sliders and spin boxes to manipulate integer variable.

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
            self,  # Variable
            title: str,
            minimum: int,
            maximum: int,
            main_window,
            parent=None):
        super(IntVariable, self).__init__(parent=parent)

        # Instantiate widgets
        self.label = PyQt5.QtWidgets.QLabel(self)
        self.slider = PyQt5.QtWidgets.QSlider(self)
        self.modify_properties(maximum, minimum)
        self.spin.setValue(self.x)

        # Attach arguments to self and widgets
        self.label.setText(title)
        self.spin.setMinimum(minimum)
        self.spin.setMaximum(maximum)
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
    def spin_changed(self, value):
        self.x = value
        self.slider.disconnect()  # Avoid loop with setValue
        self.slider.setValue(self.new_spin_value)
        self.slider.valueChanged.connect(self.slider_changed)
        self.main_window.update_plot()


    def new_slider_value(self,   # IntVariable
                         value):
        """Seperate method to isolate difference between int and float classes
        """
        return value
    
    def slider_changed(
            self,
            value):
        self.x = self.new_slider_value(value)
        self.spin.disconnect()  # Avoid loop with setValue
        self.spin.setValue(self.x)
        self.spin.valueChanged.connect(self.spin_changed)
        self.main_window.update_plot()

class FloatVariable(IntVariable):
    """Provide sliders and spin boxes to manipulate float variable.

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
        
        self.spin = PyQt5.QtWidgets.QDoubleSpinBox(self)

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
        return self.slider.minimum() + int((value - self.minimum) / self.dx_dslide)
    
    def new_slider_value(self, # FloatVariable
                         value):
        """Seperate method to isolate difference between int and float classes
        """
        return self.minimum + float(value) * self.dx_dslide
    
if __name__ == '__main__':
    app = PyQt5.QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
