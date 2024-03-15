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

import hmm.C
import utilities
import develop

PINT = pint.UnitRegistry()


def parse_args(argv=None):
    """ This is for debugging
    """
    if argv is None:
        argv = sys.argv[1:]
    import argparse
    parser = argparse.ArgumentParser("Only for debugging")
    utilities.common_arguments(parser)
    args = parser.parse_args(argv)
    args.trim_start = 25
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
        self.root_box.setText('../../../..')
        root_ok = PyQt5.QtWidgets.QPushButton('Root', self)
        root_ok.clicked.connect(self.new_root)

        self.record_box = PyQt5.QtWidgets.QLineEdit(self)
        self.record_box.setText('a03')
        record_ok = PyQt5.QtWidgets.QPushButton('Record', self)
        record_ok.clicked.connect(self.new_record)

        self.model_box = PyQt5.QtWidgets.QLineEdit(self)
        self.model_box.setText(
            f'{self.root_box.text()}/build/derived_data/apnea/models/default')
        model_ok = PyQt5.QtWidgets.QPushButton('Model', self)
        model_ok.clicked.connect(self.new_model)

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

        model_layout = PyQt5.QtWidgets.QHBoxLayout()
        model_layout.addWidget(model_ok)
        model_layout.addWidget(self.model_box)
        buttons_layout.addLayout(model_layout)

        buttons_layout.addWidget(read_button)

        self.variable = {}  # A dict to hold variables
        # Layout row of sliders
        for name, minimum, maximum, initial in (
            ('T', 0, 540, 0),
            ('fine T', -50, 50, 0),
                # From .1 second to 9 hours in minutes
            ('Delta T', numpy.log(.1 / 60), numpy.log(540), numpy.log(540)),
            ('Specific', numpy.log(1.0e-20), numpy.log(1.0e20), numpy.log(1)),
        ):
            self.variable[name] = Variable(name, minimum, maximum, initial,
                                           self)
            sliders_layout.addWidget(self.variable[name])

        # Define widgets for plot section

        ecg = pyqtgraph.GraphicsLayoutWidget(title="ECG")
        ecg_plot = ecg.addPlot()
        ecg_plot.addLegend()
        self.ecg_dict = {
            'curves': [
                ecg_plot.plot(pen='g', name='ecg'),
                ecg_plot.plot(
                    pen=None,
                    symbol='+',
                    symbolSize=15,
                    symbolBrush=('b'),
                    name='state 32',
                )
            ]
        }

        hr = pyqtgraph.GraphicsLayoutWidget(title="Heart Rate")
        hr_plot = hr.addPlot()
        hr_plot.addLegend()
        self.hr_dict = {
            'curves': [
                hr_plot.plot(pen=pyqtgraph.mkPen(color=(128, 255, 128),
                                                 width=2),
                             name='hr'),
                hr_plot.plot(pen=pyqtgraph.mkPen(color=(255, 128, 128),
                                                 width=2),
                             name='slow'),
            ]
        }

        filter = pyqtgraph.GraphicsLayoutWidget(title="Filtered HR")
        filter_plot = filter.addPlot()
        filter_plot.addLegend()
        self.filter_dict = {
            'curves': [
                filter_plot.plot(pen=pyqtgraph.mkPen(color=(255, 128, 128),
                                                     width=2),
                                 name='respiration'),
                filter_plot.plot(pen='y', name='envelope'),
                filter_plot.plot(pen='g', name='resp_pass'),
            ]
        }

        like = pyqtgraph.GraphicsLayoutWidget(title="Likelihood")
        like_plot = like.addPlot()
        like_plot.addLegend()
        self.like_dict = {
            'curves': [like_plot.plot(pen='g', name='likelihood')]
        }

        viterbi = pyqtgraph.GraphicsLayoutWidget(title="Decoded States")
        viterbi_plot = viterbi.addPlot()
        viterbi_plot.addLegend()
        self.viterbi_dict = {
            'curves': [viterbi_plot.plot(pen='g', name='state')]
        }

        class_ = pyqtgraph.GraphicsLayoutWidget(title="Classification")
        class_plot = class_.addPlot()
        class_plot.addLegend()
        self.class_dict = {
            'curves': [
                class_plot.plot(pen=pyqtgraph.mkPen(color=(255, 64, 0),
                                                    width=3),
                                name='expert'),
                class_plot.plot(
                    pen=pyqtgraph.mkPen(color=(0, 200, 255),
                                        width=2,
                                        style=PyQt5.QtCore.Qt.DotLine),
                    name='hmm')
            ]
        }

        # Layout plot section
        for widget in (ecg, hr, filter, like, viterbi, class_):
            plot_layout.addWidget(widget)

        # Make self the central widget
        widget = PyQt5.QtWidgets.QWidget()
        widget.setLayout(layout0)
        self.setCentralWidget(widget)

    def update_plots(self  # MainWindow
                    ):
        for _dict in (self.ecg_dict, self.hr_dict, self.filter_dict,
                      self.like_dict, self.viterbi_dict, self.class_dict):
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

    def read_model(self):
        file_name = self.model_box.text()
        with open(file_name, 'rb') as _file:
            self.model = pickle.load(_file)

    def new_model(self):
        path_list = [self.root_box.text()
                    ] + 'build derived_data apnea models'.split()
        path = os.path.join(*path_list)
        file_name = self.open_file_dialog(path)
        self.model_box.setText(file_name)
        self.read_model()

    def plot_window(self, curves=None, signals=None):
        """ Display T to T + Delta T in a plot

        Keyword args:
            curves: A list of PyQt plot objects
            signals: A list of pairs (times, values).  Times with pint
        """
        if signals is None:
            print(f'No signals to plot.')
            return
        start = self.variable['T']() + self.variable['fine T']() / 50
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
        self.plot_like()
        self.plot_states()
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

        # Read apnea model.
        self.read_model()

        # Read ECG
        prefix = os.path.join(self.root_box.text(), 'raw_data/Rtimes',
                              f'{self.record_box.text()}')
        with open(prefix + '.ecg', 'rb') as _file:
            _dict = pickle.load(_file)
        self.ecg = _dict['raw']
        self.ecg_times = _dict['times'] * PINT('seconds')

        # Read decoded ecg states and find places where state is 32
        with open(self.signal_path('states'), 'rb') as _file:
            states = pickle.load(_file)
        self.r_times = numpy.nonzero(states == 32)[0]

        # Read heart rate
        if 'config' in self.model.args:
            config = self.model.args.config
        else:
            config = None
        self.hr_instance = utilities.HeartRate(
            self.args,
            self.record_box.text(),
            config,
            False  # normalize
        )  # Sets: hr_sample_frequency, raw_hr
        if self.record_box.text()[0] != 'x':
            # Set expert, (normal->0, apnea->1)
            self.hr_instance.read_expert()
        # Set slow, notch, respiration, envelope
        self.hr_instance.filter_hr()
        self.hr_signal = self.hr_instance.raw_hr
        self.slow = self.hr_instance.slow
        self.hr_times = numpy.arange(len(
            self.hr_signal)) / self.hr_instance.hr_sample_frequency

        # Calculate spectral filters using fft method in utilities
        self.filters = {
            'times': self.hr_times,
            'resp_pass': self.hr_instance.resp_pass,
            'envelope': self.hr_instance.envelope,
            'respiration': self.hr_instance.respiration,
        }

        # Get observations for apnea model
        read_y = self.model.read_y_no_class(self.record_box.text())
        self.y_data = [hmm.base.JointSegment(read_y)]

        # Calculate hmm classification
        self.new_classification()

        # Viterbi decode states and calculate likelihood
        class_model = self.model.y_mod['class']
        del self.model.y_mod['class']
        self.states = self.model.decode(self.y_data)
        self.time_states = numpy.arange(len(
            self.states)) / self.model.args.model_sample_frequency

        self.weight = self.model.weights(self.y_data).sum(axis=0)
        self.like = -numpy.log(self.model.gamma_inv)
        self.time_like = numpy.arange(len(
            self.like)) / self.model.args.model_sample_frequency

        self.model.y_mod['class'] = class_model

    def plot_ecg(self):
        """"""

        self.ecg_dict['signals'] = [(self.ecg_times, self.ecg),
                                    (self.ecg_times[self.r_times],
                                     self.ecg[self.r_times])]

        self.plot_window(**self.ecg_dict)

    def plot_hr(self):
        self.hr_dict['signals'] = [
            (self.hr_times, self.hr_signal),
            (self.filters['times'], self.slow),
        ]
        self.plot_window(**self.hr_dict)

    def plot_like(self):
        if not hasattr(self, 'time_like'):
            return
        self.like_dict['signals'] = [(self.time_like, self.like)]
        self.plot_window(**self.like_dict)

    def plot_states(self):
        if not hasattr(self, 'time_states'):
            return
        self.viterbi_dict['signals'] = [(self.time_states, self.states)]
        self.plot_window(**self.viterbi_dict)

    def plot_classification(self  # MainWindow
                           ):
        expert_class = self.model_record.class_from_expert
        hmm_class = self.model_record.class_from_model
        expert_times = numpy.arange(len(expert_class)) * PINT('minutes')
        self.class_dict['signals'] = [(expert_times, expert_class),
                                      (expert_times, hmm_class)]
        self.plot_window(**self.class_dict)

    def new_classification(
            self,  # MainWindow
    ):
        self.model_record = utilities.ModelRecord(self.model_box.text(),
                                                  self.record_box.text())
        self.model_record.classify(numpy.exp(self.variable['Specific']()))
        self.model_record.score()
        self.plot_classification()

    def plot_filter(self):
        """plot spectral filters

        """
        self.filter_dict['signals'] = [
            (self.filters['times'], self.filters['respiration']),
            (self.filters['times'], self.filters['envelope']),
            (self.filters['times'], self.filters['resp_pass']),
        ]
        self.plot_window(**self.filter_dict)


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
