"""explore.py For exploring ECG and derived time series

Derived from hmmds/applications/laser/explore.py

"""

# PyQt5 is hopeless: pylint: skip-file
import sys  # Need sys to pass argv to QApplication
import os
import pickle

import PyQt5.QtWidgets
import pyqtgraph

import numpy
import numpy.linalg
import scipy.optimize
import pint

import rtimes2hr # For read_rtimes FixMe: Move to utilities

PINT = pint.UnitRegistry()

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

        self.root_box = PyQt5.QtWidgets.QLineEdit(self)
        self.root_box.setText('../../../..')
        root_ok = PyQt5.QtWidgets.QPushButton('Root', self)
        root_ok.clicked.connect(self.new_root)

        self.record_box = PyQt5.QtWidgets.QLineEdit(self)
        self.record_box.setText('a01')
        record_ok = PyQt5.QtWidgets.QPushButton('Record', self)
        record_ok.clicked.connect(self.new_record)

        self.model_box = PyQt5.QtWidgets.QLineEdit(self)
        self.model_box.setText('p1model_A4')
        model_ok = PyQt5.QtWidgets.QPushButton('Model', self)
        model_ok.clicked.connect(self.new_model)

        ecg_button = PyQt5.QtWidgets.QPushButton('Plot ECG', self)
        ecg_button.clicked.connect(self.read_ecg)

        hr_button = PyQt5.QtWidgets.QPushButton('Plot Heart Rate', self)
        hr_button.clicked.connect(self.read_hr)

        resp_button = PyQt5.QtWidgets.QPushButton('Plot Resp', self)
        resp_button.clicked.connect(self.read_resp)

        like_button = PyQt5.QtWidgets.QPushButton('Log Likelihood', self)
        like_button.clicked.connect(self.calculate_like)

        # Layout button section
        buttons_layout.addWidget(quit_button)

        root_layout = PyQt5.QtWidgets.QHBoxLayout()
        root_layout.addWidget(root_ok)
        root_layout.addWidget(self.root_box)
        buttons_layout.addLayout(root_layout)

        record_layout = PyQt5.QtWidgets.QHBoxLayout()
        record_layout.addWidget(record_ok)
        record_layout.addWidget(self.record_box)
        buttons_layout.addLayout(record_layout)

        model_layout = PyQt5.QtWidgets.QHBoxLayout()
        model_layout.addWidget(model_ok)
        model_layout.addWidget(self.model_box)
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
        self.qrs_times = ecg_plot.plot(pen=None,
                                       symbol='+',
                                       symbolSize=15,
                                       symbolBrush=('b'))

        hr = pyqtgraph.GraphicsLayoutWidget(title="Heart Rate")
        hr_plot = hr.addPlot()
        self.hr_curve = hr_plot.plot(pen='g')

        resp = pyqtgraph.GraphicsLayoutWidget(title="Resp Components")
        resp_plot = resp.addPlot()
        self.resp_curve0 = resp_plot.plot(pen='w')
        self.resp_curve1 = resp_plot.plot(pen='r')

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

    def update_plots(self):
        self.plot_ecg()
        self.plot_hr()
        self.plot_resp()

    def new_root(self):
        text = self.root_box.text()
        print(f'{text=}')

    def new_record(self):
        text = self.record_box.text()
        print(f'{text=}')

    def new_model(self):
        text = self.model_box.text()
        print(f'{text=}')

    def plot_window(self, plot, times, values):
        """ Display T to T + Delta T in a plot

        Args:
            plot: a PyQt plot object
            times: 1-d array of x axis values
            values: 1-d array of y axis values
        """
        window = [
            self.variable['T'](),
            self.variable['T']() + self.variable['Delta T']()
        ]
        start, stop = numpy.searchsorted(times, window)
        plot.setData(times[start:stop], values[start:stop])

    def plot_ecg(self):
        self.plot_window(self.ecg_curve, self.ecg_dict['times'], self.ecg_dict['signal'])
        #self.plot_window(self.qrs_times, coarse_times, qrs)

    def plot_hr(self):
        self.plot_window(self.hr_curve, self.hr_dict['times'], self.hr_dict['signal'])

    def plot_resp(self):
        for curve, i in zip((self.resp_curve0, self.resp_curve1),(0,1)):
            self.plot_window(curve, self.resp_dict['times'].magnitude, self.resp_dict['signal'][i])

    def read_ecg(self):
        prefix = os.path.join(
            self.root_box.text(),
            'raw_data/Rtimes',f'{self.record_box.text()}')
        with open(prefix+'.ecg', 'rb') as _file:
            _dict = pickle.load(_file)
            ecg = _dict['raw']
            times = _dict['times']
        rtimes, n_ecg = rtimes2hr.read_rtimes(prefix+'.rtimes')
        self.ecg_dict = {'times':times, 'signal':ecg}
        self.plot_ecg()

    def read_hr(self):
        path = os.path.join(
            self.root_box.text(),
            'build/derived_data/apnea/Lphr',
            f'{self.record_box.text()}.lphr')
        with open(path, 'rb') as _file:
            hr_dict = pickle.load(_file)
        signal = hr_dict['hr_low_pass']
        self.hr_dict = {'times':numpy.arange(len(signal)), 'signal':signal}
        self.plot_hr()

    def read_resp(self):
        path = os.path.join(
            self.root_box.text(),
            'build/derived_data/apnea/Respire',
            f'{self.record_box.text()}.resp')
        with open(path, 'rb') as _file:
            _dict = pickle.load(_file)
            times = _dict['times']
            components = _dict['components']  # c_0, c_1, norm of psd
        self.resp_dict = {'times':times, 'signal':components.T}
        self.plot_resp()

    def calculate_like(self):
        pass


class Variable(PyQt5.QtWidgets.QWidget):
    """Provide sliders and spin boxes to manipulate variable.

    Args:
        label:
        minimum:
        maximum:
        main_window: For access to method update_plots
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
        self.main_window.update_plots()

    def slider_changed(
            self,  # Variable
            value):
        self.x = self.minimum + float(value) * self.dx_dslide
        self.spin.disconnect()  # Avoid loop with setValue
        self.spin.setValue(self.x)
        self.spin.valueChanged.connect(self.spin_changed)
        self.main_window.update_plots()


if __name__ == '__main__':
    app = PyQt5.QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
