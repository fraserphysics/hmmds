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

import rtimes2hr  # For read_rtimes FixMe: Move to utilities
import utilities

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

        self.ecg_hmm_box = PyQt5.QtWidgets.QLineEdit(self)
        self.ecg_hmm_box.setText('trained_20_ECG')
        ecg_hmm_ok = PyQt5.QtWidgets.QPushButton('ECG HMM', self)
        ecg_hmm_ok.clicked.connect(self.new_ecg_hmm)

        self.model0_box = PyQt5.QtWidgets.QLineEdit(self)
        self.model0_box.setText('p1model_A4')
        model0_ok = PyQt5.QtWidgets.QPushButton('Model0', self)
        model0_ok.clicked.connect(self.new_model0)

        self.model1_box = PyQt5.QtWidgets.QLineEdit(self)
        self.model1_box.setText('p1model_C2')
        model1_ok = PyQt5.QtWidgets.QPushButton('Model1', self)
        model1_ok.clicked.connect(self.new_model1)

        ecg_button = PyQt5.QtWidgets.QPushButton('Read ECG', self)
        ecg_button.clicked.connect(self.read_ecg)

        viterbi_button = PyQt5.QtWidgets.QPushButton('Viterbi', self)
        viterbi_button.clicked.connect(self.viterbi)

        hr_button = PyQt5.QtWidgets.QPushButton('Read Heart Rate', self)
        hr_button.clicked.connect(self.read_hr)

        resp_button = PyQt5.QtWidgets.QPushButton('Read Resp', self)
        resp_button.clicked.connect(self.read_resp)

        like_button = PyQt5.QtWidgets.QPushButton('Calculate Likelihood', self)
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

        ecg_hmm_layout = PyQt5.QtWidgets.QHBoxLayout()
        ecg_hmm_layout.addWidget(ecg_hmm_ok)
        ecg_hmm_layout.addWidget(self.ecg_hmm_box)
        buttons_layout.addLayout(ecg_hmm_layout)

        model0_layout = PyQt5.QtWidgets.QHBoxLayout()
        model0_layout.addWidget(model0_ok)
        model0_layout.addWidget(self.model0_box)
        buttons_layout.addLayout(model0_layout)

        model1_layout = PyQt5.QtWidgets.QHBoxLayout()
        model1_layout.addWidget(model1_ok)
        model1_layout.addWidget(self.model1_box)
        buttons_layout.addLayout(model1_layout)

        buttons_layout.addWidget(ecg_button)
        buttons_layout.addWidget(viterbi_button)
        buttons_layout.addWidget(hr_button)
        buttons_layout.addWidget(resp_button)
        buttons_layout.addWidget(like_button)

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
                    name='qrs',
                )
            ]
        }

        viterbi = pyqtgraph.GraphicsLayoutWidget(title="Decoded States")
        viterbi_plot = viterbi.addPlot()
        viterbi_plot.addLegend()
        self.viterbi_dict = {
            'curves': [viterbi_plot.plot(pen='g', name='state')]
        }

        hr = pyqtgraph.GraphicsLayoutWidget(title="Heart Rate")
        hr_plot = hr.addPlot()
        hr_plot.addLegend()
        self.hr_dict = {'curves': [hr_plot.plot(pen='g', name='hr')]}

        resp = pyqtgraph.GraphicsLayoutWidget(title="Resp Components")
        resp_plot = resp.addPlot()
        resp_plot.addLegend()
        self.resp_dict = {
            'curves': [
                resp_plot.plot(pen='w', name='resp0'),
                resp_plot.plot(pen='r', name='resp1')
            ]
        }

        like = pyqtgraph.GraphicsLayoutWidget(title="Log Likelihood")
        like_plot = like.addPlot()
        like_plot.addLegend()
        self.like_dict = {
            'curves': [
                like_plot.plot(pen='g', name='model0'),
                like_plot.plot(pen='w', name='model1'),
            ]
        }

        # Layout plot section
        for widget in (ecg, viterbi, hr, resp, like):
            plot_layout.addWidget(widget)

        # Make self the central widget
        widget = PyQt5.QtWidgets.QWidget()
        widget.setLayout(layout0)
        self.setCentralWidget(widget)

    def update_plots(self):
        for _dict in (self.ecg_dict, self.viterbi_dict, self.hr_dict,
                      self.resp_dict, self.like_dict):
            self.plot_window(**_dict)

    def new_root(self):
        pass  # No action

    def new_ecg_hmm(self):
        pass  # No action

    def new_record(self):
        pass  # No action

    def new_model0(self):
        pass  # No action

    def new_model1(self):
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
            start, stop = numpy.searchsorted(minutes, window)
            curve.setData(minutes[start:stop], signal[1][start:stop])

    def read_ecg(self):
        prefix = os.path.join(self.root_box.text(), 'raw_data/Rtimes',
                              f'{self.record_box.text()}')
        with open(prefix + '.ecg', 'rb') as _file:
            _dict = pickle.load(_file)
            ecg = _dict['raw']
            ecg_times = _dict['times'] * PINT('seconds')
        rtimes, n_ecg = rtimes2hr.read_rtimes(prefix + '.rtimes')
        indices = numpy.searchsorted(
            ecg_times.to('seconds').magnitude,
            rtimes.to('seconds').magnitude)
        self.ecg_dict['signals'] = [(ecg_times, ecg), (rtimes, ecg[indices])]
        self.plot_window(**self.ecg_dict)

    def viterbi(self):
        """Read the model and the data.  Then run model.decode.
        """
        path = os.path.join(self.root_box.text(),
                            'build/derived_data/apnea/models',
                            f'{self.ecg_hmm_box.text()}')
        with open(path, 'rb') as _file:
            hmm = pickle.load(_file)
        prefix = os.path.join(self.root_box.text(), 'raw_data/Rtimes',
                              f'{self.record_box.text()}')
        with open(prefix + '.ecg', 'rb') as _file:
            _dict = pickle.load(_file)
            ecg = _dict['raw']
            ecg_times = _dict['times'] * PINT('seconds')

        print('start decode')
        states = hmm.decode([ecg])
        print('finished decode')
        self.viterbi_dict['signals'] = [(ecg_times, states)]
        self.plot_window(**self.viterbi_dict)

    def read_hr(self):
        path = os.path.join(self.root_box.text(),
                            'build/derived_data/apnea/Lphr',
                            f'{self.record_box.text()}.lphr')
        with open(path, 'rb') as _file:
            pickle_dict = pickle.load(_file)
        hr = pickle_dict['hr'].to('1/minute').magnitude
        times = numpy.arange(len(hr)) / pickle_dict['sample_frequency']
        self.hr_dict['signals'] = [(times, hr)]
        self.plot_window(**self.hr_dict)

    def read_resp(self):
        path = os.path.join(self.root_box.text(),
                            'build/derived_data/apnea/Respire',
                            f'{self.record_box.text()}.resp')
        with open(path, 'rb') as _file:
            _dict = pickle.load(_file)
            times = _dict['times']
            components = _dict['components'].T
        # Last component is norm of psd
        self.resp_dict['signals'] = [(times, components[0]),
                                     (times, components[1])]
        self.plot_window(**self.resp_dict)

    def calculate_like(self):
        """Calculate log(prob(y[t]|y[:t],model)) for all t and two models

        """
        derived_apnea_data_dir = os.path.join(self.root_box.text(),
                                              'build/derived_data/apnea')
        model_dir = os.path.join(derived_apnea_data_dir, 'models')
        with open(os.path.join(model_dir, self.model0_box.text()),
                  'rb') as _file:
            self.model0 = pickle.load(_file)
        with open(os.path.join(model_dir, self.model1_box.text()),
                  'rb') as _file:
            self.model1 = pickle.load(_file)

        class Args:

            def __init__(self):
                """Namespace to pass to utility function
                """
                self.heart_rate_dir = os.path.join(derived_apnea_data_dir,
                                                   'Lphr')
                self.respiration_dir = os.path.join(derived_apnea_data_dir,
                                                    'Respire')

        data = [
            utilities.heart_rate_respiration_data(self.record_box.text(),
                                                  Args())
        ]
        times = data[0]['times']
        probabilities = []
        for model in (self.model0, self.model1):
            model.y_mod.observe(data)
            probabilities.append((times, model.log_probs()))
        self.like_dict['signals'] = probabilities
        self.plot_window(**self.like_dict)


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
