"""explore.py For exploring ECG and derived time series


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
import scipy.signal
import pint

import hmm.C
import utilities

PINT = pint.UnitRegistry()


class MainWindow(PyQt5.QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Examine ECG Model Performance")
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
        self.record_box.setText('a01')
        record_ok = PyQt5.QtWidgets.QPushButton('Record', self)
        record_ok.clicked.connect(self.new_record)

        self.data_dir_box = PyQt5.QtWidgets.QLineEdit(self)
        self.data_dir_box.setText(
            f'{self.root_box.text()}/build/derived_data/ECG/')
        data_dir_ok = PyQt5.QtWidgets.QPushButton('Data Dir', self)
        data_dir_ok.clicked.connect(self.new_data_dir)

        read_button = PyQt5.QtWidgets.QPushButton('Read Data', self)
        read_button.clicked.connect(self.read_data)

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

        data_dir_layout = PyQt5.QtWidgets.QHBoxLayout()
        data_dir_layout.addWidget(data_dir_ok)
        data_dir_layout.addWidget(self.data_dir_box)
        buttons_layout.addLayout(data_dir_layout)

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

        viterbi = pyqtgraph.GraphicsLayoutWidget(title="Decoded States")
        viterbi_plot = viterbi.addPlot()
        viterbi_plot.addLegend()
        self.viterbi_dict = {
            'curves': [
                viterbi_plot.plot(pen='g', name='state'),
                viterbi_plot.plot(
                    pen=None,
                    symbol='+',
                    symbolSize=15,
                    symbolBrush=('r'),
                    name='noise',
                )
            ]
        }

        like = pyqtgraph.GraphicsLayoutWidget(title="Likelihood")
        like_plot = like.addPlot()
        like_plot.addLegend()
        self.like_dict = {
            'curves': [like_plot.plot(pen='g', name='likelihood')]
        }

        hr = pyqtgraph.GraphicsLayoutWidget(title="Heart Rate")
        hr_plot = hr.addPlot()
        hr_plot.addLegend()
        self.hr_dict = {'curves': [hr_plot.plot(pen='r', name='hr'),]}

        # Layout plot section
        for widget in (ecg, viterbi, like, hr):
            plot_layout.addWidget(widget)

        # Make self the central widget
        widget = PyQt5.QtWidgets.QWidget()
        widget.setLayout(layout0)
        self.setCentralWidget(widget)

    def update_plots(self):
        for _dict in (self.ecg_dict, self.viterbi_dict, self.hr_dict,
                      self.like_dict):
            self.plot_window(**_dict)

    def open_file_dialog(self, directory=None):
        if directory is None:
            directory = self.root_box.text()
        filename = str(
            PyQt5.QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select Directory", directory))
        return filename

    def new_root(self):
        pass  # No action

    def new_data_dir(self):
        path_list = [self.root_box.text()] + 'build derived_data ECG'.split()
        path = os.path.join(*path_list)
        file_name = self.open_file_dialog(path)
        self.data_dir_box.setText(file_name)

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

    def read_data(self):
        self.read_ecg()
        self.read_viterbi()
        self.read_like()
        self.read_hr()

    def read_ecg(self):
        prefix = os.path.join(self.root_box.text(), 'raw_data/Rtimes',
                              f'{self.record_box.text()}')
        with open(prefix + '.ecg', 'rb') as _file:
            _dict = pickle.load(_file)
            ecg = _dict['raw']
            ecg_times = _dict['times'] * PINT('seconds')
        self.ecg_dict['signals'] = [
            (ecg_times, ecg),
        ]
        self.plot_window(**self.ecg_dict)

    def signal_path(self, signal):
        """ Return path to signal

        Args:
            signal: ecg, state, likehood or heart_rate
        """
        data_dir = self.data_dir_box.text()
        if data_dir.find('trained') == -1:
            return f'{self.data_dir_box.text()}/{self.record_box.text()}_self_AR3/{signal}'
        else:  # For eg, a01_trained_AR3/states/x01
            return f'{self.data_dir_box.text()}/{signal}/{self.record_box.text()}'

    def read_viterbi(self):
        """Read the state data.
        """
        with open(self.signal_path('states'), 'rb') as _file:
            states = pickle.load(_file)
        times = numpy.arange(0, len(states)) / (100 * 60) * PINT('minutes')
        indices = numpy.nonzero((states[1:-1] == 0) &
                                ((states[:-2] != states[1:-1])  # Leading edge
                                 |
                                 (states[2:] != states[1:-1]))  # Trailing edge
                               )[0] + 1
        self.viterbi_dict['signals'] = [(times, states),
                                        (times[indices], states[indices])]
        self.plot_window(**self.viterbi_dict)

        # Find places where state is 32 and put + in ecg plot there.
        indices = numpy.nonzero(states == 32)[0]
        ecg_times, ecg = self.ecg_dict['signals'][0]
        self.ecg_dict['signals'] = [(ecg_times, ecg),
                                    (ecg_times[indices], ecg[indices])]
        self.plot_window(**self.ecg_dict)

    def read_like(self):
        with open(self.signal_path('likelihood'), 'rb') as _file:
            likelihood = pickle.load(_file)
        times = numpy.arange(0, len(likelihood)) / (100 * 60) * PINT('minutes')
        self.like_dict['signals'] = [(times, numpy.log(likelihood))]
        self.plot_window(**self.like_dict)

    def read_hr(self):
        with open(self.signal_path('heart_rate'), 'rb') as _file:
            pickle_dict = pickle.load(_file)
        hr_states = pickle_dict['hr'].to('1/minute').magnitude
        times = numpy.arange(len(hr_states)) / pickle_dict['sample_frequency']

        self.hr_dict['signals'] = [
            (times, hr_states),
        ]
        self.plot_window(**self.hr_dict)


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
        self.main_window.update_plots()


if __name__ == '__main__':
    app = PyQt5.QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
