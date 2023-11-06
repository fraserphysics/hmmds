"""explore.py For exploring apnea from heart rate time series

"""

# PyQt5 is hopeless: pylint: skip-file
import sys  # Need sys to pass argv to QApplication
import os
import pickle

import PyQt5.QtWidgets
import PyQt5.QtCore
import pyqtgraph

import numpy
import numpy.linalg
import scipy.optimize
import scipy.signal
import pint

import utilities

PINT = pint.UnitRegistry()


def parse_args(argv=None):
    """ Access default args in utilities
    """
    if argv is None:
        argv = sys.argv[1:]
    import argparse
    parser = argparse.ArgumentParser("Access default args in utilities")
    utilities.common_arguments(parser)
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


class MainWindow(PyQt5.QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.args = parse_args()
        self.setWindowTitle("Examine Apnea Model Performance")
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
        self.root_box.setText('../../../../..')
        root_ok = PyQt5.QtWidgets.QPushButton('Root', self)
        root_ok.clicked.connect(self.new_root)

        self.record_box = PyQt5.QtWidgets.QLineEdit(self)
        self.record_box.setText('a03')
        record_ok = PyQt5.QtWidgets.QPushButton('Record', self)
        record_ok.clicked.connect(self.new_record)

        read_button = PyQt5.QtWidgets.QPushButton(r'Read and Calculate', self)
        read_button.clicked.connect(self.read_calculate_plot)

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

        buttons_layout.addWidget(read_button)

        self.variable = {}  # A dict to hold variables
        # Layout row of sliders
        for name, minimum, maximum, initial in (
            ('T', 0, 540, 0),
            ('fine T', -50, 50, 0),
                # From .1 second to 9 hours in minutes
            ('Delta T', numpy.log(.1 / 60), numpy.log(540), numpy.log(540)),
        ):
            self.variable[name] = Variable(name, minimum, maximum, initial,
                                           self)
            sliders_layout.addWidget(self.variable[name])

        # Define widgets for plot section

        ecg = pyqtgraph.GraphicsLayoutWidget(title="ECG")
        ecg_plot = ecg.addPlot()
        ecg_plot.addLegend()
        self.ecg_dict = {'curves': [ecg_plot.plot(pen='g', name='ecg')]}

        hr = pyqtgraph.GraphicsLayoutWidget(title="Heart Rate")
        hr_plot = hr.addPlot()
        hr_plot.addLegend()
        self.hr_dict = {
            'curves': [
                hr_plot.plot(pen='g', name='hr'),
                hr_plot.plot(pen='r', name='filtered'),
            ]
        }

        filter = pyqtgraph.GraphicsLayoutWidget(title="Filtered HR")
        filter_plot = filter.addPlot()
        filter_plot.addLegend()
        self.filter_dict = {
            'curves': [
                filter_plot.plot(pen='r', name='fast'),
                filter_plot.plot(pen='y', name='envelope')
            ]
        }
        resp = pyqtgraph.GraphicsLayoutWidget(title="Resp")
        resp_plot = resp.addPlot()
        resp_plot.addLegend()
        self.resp_dict = {
            'curves': [
                resp_plot.plot(pen='r', name='resp_a'),
                resp_plot.plot(pen='y', name='resp_c'),
                resp_plot.plot(pen='g', name='resp_n')
            ]
        }

        SpO2_ = pyqtgraph.GraphicsLayoutWidget(title="Classification")
        SpO2_plot = SpO2_.addPlot()
        SpO2_plot.addLegend()
        self.SpO2_dict = {'curves': [SpO2_plot.plot(pen='r', name='SpO2')]}

        class_ = pyqtgraph.GraphicsLayoutWidget(title="Classification")
        class_plot = class_.addPlot()
        class_plot.addLegend()
        self.class_dict = {'curves': [class_plot.plot(pen='r', name='expert')]}

        # Layout plot section
        for widget in (ecg, hr, filter, resp, SpO2_, class_):
            plot_layout.addWidget(widget)

        # Make self the central widget
        widget = PyQt5.QtWidgets.QWidget()
        widget.setLayout(layout0)
        self.setCentralWidget(widget)

    def update_plots(self  # MainWindow
                    ):
        for _dict in (self.ecg_dict, self.hr_dict, self.filter_dict,
                      self.resp_dict, self.SpO2_dict, self.class_dict):
            self.plot_window(**_dict)

    def open_file_dialog(self, directory=None):
        if directory is None:
            directory = self.root_box.text()
        dialog = PyQt5.QtWidgets.QFileDialog(self)
        dialog.setDirectory(directory)
        dialog.setFileMode(PyQt5.QtWidgets.QFileDialog.FileMode.ExistingFiles)
        dialog.setViewMode(PyQt5.QtWidgets.QFileDialog.ViewMode.List)
        filename, ok = dialog.getOpenFileName()
        return filename

    def new_root(self):
        pass  # No action

    def new_record(self):
        pass  # No action

    def plot_window(self, curves=None, signals=None):
        """ Display T to T + Delta T in a plot

        Keyword args:
            curves: A list of PyQt plot objects
            signals: A list of pairs (times, values).  Times with pint
        """
        if signals is None:
            print(f'No signals to plot.')
            return
        start = self.variable['T']() + self.variable['fine T']() / 200
        stop = start + numpy.exp(self.variable['Delta T']())
        window = [start, stop]
        for curve, signal in zip(curves, signals):
            minutes = signal[0].to('minutes').magnitude
            start, _stop = numpy.searchsorted(minutes, window)
            stop = min(_stop, len(minutes), len(signal[1]))
            curve.setData(minutes[start:stop], signal[1][start:stop])

    def read_calculate_plot(self):
        """ Read data, caclulate, and plot
        """
        self.read_calculate()
        self.plot_ecg()
        self.plot_hr()
        self.plot_filter()
        self.plot_resp()
        self.plot_SpO2()
        self.plot_classification()

    def signal_path(self, signal):
        """ Return path to signal

        Args:
            signal: ecg, state, likelihood or heart_rate
        """
        return f'{self.root_box.text()}/build/derived_data/ECG/{self.record_box.text()}_self_AR3/{signal}'

    def read_calculate(self):
        """Do all this in one method because calculations depend on
        each other

        """
        record_name = f'{self.record_box.text()}'
        # Read ECG
        prefix = os.path.join(self.root_box.text(), 'raw_data/Rtimes',
                              record_name)
        with open(prefix + '.ecg', 'rb') as _file:
            _dict = pickle.load(_file)
        self.ecg = _dict['raw']
        self.ecg_times = _dict['times'] * PINT('seconds')

        top_freq = 2 * PINT('1/minute')  # Used in both hr_notch and
        # respiration derived from
        # heart rate

        # Read heart rate
        with open(self.signal_path('heart_rate'), 'rb') as _file:
            pickle_dict = pickle.load(_file)
        sample_frequency = pickle_dict['sample_frequency']
        self.hr_signal = pickle_dict['hr'].to('1/minute').magnitude
        notch = (12 * PINT('1/minute'), 18 * PINT('1/minute'))
        self.hr_notch = utilities.notch_hr(self.hr_signal, 1 / sample_frequency,
                                           notch, top_freq)
        self.hr_times = numpy.arange(len(self.hr_signal)) / sample_frequency

        # Calculate spectral filters using fft method in utilities
        # with normalization for amplitude of heart rate signal.
        divisor = utilities.Pass1(record_name,
                                  self.args).statistic_2() / self.args.norm_avg
        self.filters = utilities.hr_2_respiration(
            self.hr_signal / divisor,
            1 / sample_frequency,
            sample_period_out=PINT('minute') / 120,
            bandpass_center=17 / PINT('minute'),
            bandpass_width=3 / PINT('minute'),
            smooth_width=top_freq,
        )

    def plot_ecg(self):
        """"""

        self.ecg_dict['signals'] = [(self.ecg_times, self.ecg)]

        self.plot_window(**self.ecg_dict)

    def plot_hr(self):
        self.hr_dict['signals'] = [(self.hr_times, self.hr_signal),
                                   (self.hr_times, self.hr_notch)]
        self.plot_window(**self.hr_dict)

    def plot_filter(self):
        """plot spectral filters

        """
        self.filter_dict['signals'] = [
            (self.filters['times'], self.filters['fast']),
            (self.filters['times'], self.filters['respiration'])
        ]
        self.plot_window(**self.filter_dict)

    def plot_resp(self):
        """plot wfdb respiration signals

        """
        record_name = self.record_box.text()
        if record_name not in 'a01 a02 a03 a04 b01 c01 c02 c03'.split():
            signals = (numpy.ones(480) * .5,) * 3
            times = numpy.arange(480) * PINT('minutes')
        else:
            with open(f'../../../../../raw_data/apnea/ers/{record_name}er.pkl',
                      'rb') as _file:
                record_dict = pickle.load(_file)
            freq, unit = record_dict['sample_frequency']
            sample_frequency = freq * PINT(unit)
            signals = tuple(
                record_dict[x] for x in ('Resp A', 'Resp C', 'Resp N'))
            times = numpy.arange(len(signals[0])) / sample_frequency

        self.resp_dict['signals'] = [(times, signal) for signal in signals]
        self.plot_window(**self.resp_dict)

    def plot_SpO2(self  # MainWindow
                 ):
        record_name = self.record_box.text()
        if record_name not in 'a01 a02 a03 a04 b01 c01 c02 c03'.split():
            signal = numpy.ones(480) * .5
            times = numpy.arange(480) * PINT('minutes')
        else:
            with open(f'../../../../../raw_data/apnea/ers/{record_name}er.pkl',
                      'rb') as _file:
                record_dict = pickle.load(_file)
            freq, unit = record_dict['sample_frequency']
            sample_frequency = freq * PINT(unit)
            signal = record_dict['SpO2']
            times = numpy.arange(len(signal)) / sample_frequency
        self.SpO2_dict['signals'] = [(times, signal)]
        self.plot_window(**self.SpO2_dict)

    def plot_classification(self  # MainWindow
                           ):
        record_name = self.record_box.text()
        if record_name[0] == 'x':
            expert_class = numpy.ones(480) * .5
        else:
            expert_class = utilities.read_expert(self.args.expert, record_name)
        expert_times = numpy.arange(len(expert_class)) * PINT('minutes')
        self.class_dict['signals'] = [(expert_times, expert_class)]
        self.plot_window(**self.class_dict)


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
            initial: float,
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
        initial_int = self.slider.minimum() + int(
            (initial - self.minimum) / self.dx_dslide)
        self.slider.setValue(initial_int)
        self.x = initial
        self.spin.setValue(initial)

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
        if self.label.text() == 'Specific':
            self.main_window.new_classification()
        self.main_window.update_plots()


if __name__ == '__main__':
    app = PyQt5.QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
