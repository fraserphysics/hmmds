"""explore.py For exploring ECG and derived time series

Derived from hmmds/applications/laser/explore.py

"""

# PyQt5 is hopeless: pylint: skip-file
import sys  # We need sys so that we can pass argv to QApplication
import os

import PyQt5.QtWidgets
import pyqtgraph

import numpy
import numpy.linalg
import scipy.optimize


class MainWindow(PyQt5.QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Exploring Lorenz Parameters")
        # Configure the main window
        layout0 = PyQt5.QtWidgets.QHBoxLayout()
        controls_layout = PyQt5.QtWidgets.QVBoxLayout()

        # Make all buttons fit in a fixed size
        container = PyQt5.QtWidgets.QWidget(self)
        container.setFixedSize(300, 500)
        controls_layout.addWidget(container)
        buttons_layout = PyQt5.QtWidgets.QVBoxLayout(container)
        
        sliders_layout = PyQt5.QtWidgets.QHBoxLayout()
        plot_layout = PyQt5.QtWidgets.QVBoxLayout()
        layout0.addLayout(controls_layout)
        controls_layout.addLayout(sliders_layout)
        layout0.addLayout(plot_layout)

        # Define widgets of button section
        quit_button = PyQt5.QtWidgets.QPushButton('Quit', self)
        quit_button.clicked.connect(self.close)

        record_box = PyQt5.QtWidgets.QLineEdit(self)
        record_ok = PyQt5.QtWidgets.QPushButton('Record', self)
        record_ok.clicked.connect(self.new_record)

        model_box = PyQt5.QtWidgets.QLineEdit(self)
        model_ok = PyQt5.QtWidgets.QPushButton('Model', self)
        model_ok.clicked.connect(self.new_model)

        ecg_button = PyQt5.QtWidgets.QPushButton('Plot ECG', self)
        ecg_button.clicked.connect(self.plot_ecg)

        hr_button = PyQt5.QtWidgets.QPushButton('Plot Heart Rate', self)
        ecg_button.clicked.connect(self.plot_hr)

        resp_button = PyQt5.QtWidgets.QPushButton('Plot Resp', self)
        resp_button.clicked.connect(self.plot_resp)

        like_button = PyQt5.QtWidgets.QPushButton('Log Likelihood', self)
        like_button.clicked.connect(self.plot_like)

        # Layout button section
        buttons_layout.addWidget(quit_button)
        
        record_layout = PyQt5.QtWidgets.QHBoxLayout()
        record_layout.addWidget(record_ok)
        record_layout.addWidget(record_box)
        buttons_layout.addLayout(record_layout)

        model_layout = PyQt5.QtWidgets.QHBoxLayout()
        model_layout.addWidget(model_ok)
        model_layout.addWidget(model_box)
        buttons_layout.addLayout(model_layout)

        buttons_layout.addWidget(ecg_button)
        buttons_layout.addWidget(hr_button)
        buttons_layout.addWidget(resp_button)
        buttons_layout.addWidget(like_button)

        
        self.variable = {}  # A dict to hold variables
        # Layout row of sliders
        for name, minimum, maximum in (
            ('T', 0, 540),
            ('Delta T', 5, 540),
        ):
            self.variable[name] = Variable(name, minimum, maximum, self)
            sliders_layout.addWidget(self.variable[name])

        # Define widgets for plot section

        ecg = pyqtgraph.GraphicsLayoutWidget(title="ECG")
        ecg_plot = ecg.addPlot()
        self.ecg_curve = ecg_plot.plot(pen='g')
        self.qrs_times = ecg_plot.plot(pen=None, symbol='+', symbolSize=15, symbolBrush=('b'))

        hr = pyqtgraph.GraphicsLayoutWidget(title="Heart Rate")
        hr_plot = hr.addPlot()
        self.hr_curve = hr_plot.plot(pen='g')

        resp = pyqtgraph.GraphicsLayoutWidget(title="Resp Components")
        resp_plot = resp.addPlot()
        self.resp_curve = resp_plot.plot(pen='g')

        like = pyqtgraph.GraphicsLayoutWidget(title="Log Likelihood")
        like_plot = like.addPlot()
        self.like_curve = like_plot.plot(pen='g')

        # Layout plot section
        for widget in (ecg, hr, resp, like):
            plot_layout.addWidget(widget)

        # Make self the central widget
        widget = PyQt5.QtWidgets.QWidget()
        widget.setLayout(layout0)
        self.setCentralWidget(widget)

    def update_plot(self):
        pass

    def new_record(self):
        pass

    def new_model(self):
        pass

    def plot_window(self, plot, times, values):
        """ Display T to T + Delta T in a plot

        Args:
            plot: a PyQt plot object
            times: 1-d array of x axis values
            values: 1-d array of y axis values
        """
        window = [self.variable['T'](), self.variable['T']() + self.variable['Delta T']()]
        start, stop = numpy.searchsorted(times, window)
        plot.setData(times[start:stop], values[start:stop])

    def plot_ecg(self):
        fine_times = numpy.arange(0, 540, .1)
        ecg = numpy.sin(fine_times)
        self.plot_window(self.ecg_curve, fine_times, ecg)
        coarse_times = fine_times[::10]
        qrs = numpy.sin(coarse_times)
        self.plot_window(self.qrs_times, coarse_times, qrs)

    def plot_hr(self):
        pass

    def plot_resp(self):
        pass

    def plot_like(self):
        pass


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
        self.spin.setDecimals(1)
        self.spin.setSingleStep(0.1)
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
