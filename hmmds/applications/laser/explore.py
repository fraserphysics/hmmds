""" explore.py For exploring parameters of Lorenz system.

Derived from https://www.pythonguis.com/tutorials/plotting-pyqtgraph/

"""

import sys  # We need sys so that we can pass argv to QApplication
import os
import argparse
import pdb

from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpacerItem, \
    QVBoxLayout, QWidget, QPushButton
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg

import numpy
import numpy.linalg
import scipy.optimize

import plotscripts.utilities

from hmmds.synthetic.filter.lorenz_sde import lorenz_integrate

def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Explore 1-d map')
    parser.add_argument('--show',
                        action='store_false',
                        help="display figure using Qt5")
    return parser.parse_args(argv)
def plot_for_r(r):
    argv = sys.argv[1:]
    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)
    figure, (time, x_map) = pyplot.subplots(nrows=2, ncols=1, sharex=True)

    n_x = 100
    x_array = numpy.linspace(1/n_x,6,n_x)
    x_initial = numpy.empty((n_x,3))
    x_final =  numpy.empty((n_x,3))
    t_array = numpy.empty(n_x)
    
    fixed_point = FixedPoint(r)

    for i, delta_x in enumerate(x_array):
        x_initial[i] = fixed_point.initial_state(delta_x)
        t_array[i], x_final[i] = fixed_point.map_time(x_initial[i])
    x_values = x_initial[:,0]
    time.plot(x_values, t_array/fixed_point.period, label='t')
    x_map.plot(x_values, x_final[:,0], label='final x')
    x_map.plot(x_values, x_values, label='initial x')
    for axis in time, x_map:
        axis.legend()
    
    pyplot.show()
    return 0

class FixedPoint:
    """Characterizes the focus of the Lorenz system at x_i > 0
    """
    def __init__(self, r):
        s = 10.0
        b = 8.0/3
        self.r = r
        root = numpy.sqrt(b*(r-1))
        self.fixed_point = numpy.array([root, root, r-1])
        df_dx = numpy.array([ # derivative of x_dot wrt x
            [-s,  s,   0],
            [1,   -1, -root],
            [root, root, -b]])
        values, right_vectors = numpy.linalg.eig(df_dx)
        left_vectors = numpy.linalg.inv(right_vectors)
        for i in range(3):
            assert numpy.allclose(numpy.dot(left_vectors[i], df_dx), values[i]*left_vectors[i])
            assert numpy.allclose(numpy.dot(df_dx, right_vectors[:,i]), values[i]*right_vectors[:,i])
        assert values[0].imag == 0.0,f"First eigenvalue is not real: values={values}"
        self.projection = numpy.dot(right_vectors[:,1:], left_vectors[1:,:]).real
        # projection onto subspace of complex eigenvectors
        self.image_2d = numpy.dot(numpy.array([[1,0,0],[0,0,1]]), self.projection)
        # Components 0 and 2 of projection
        assert numpy.allclose(numpy.dot(self.projection, right_vectors[:,-1]), right_vectors[:,-1])
        self.omega = numpy.abs(values[-1].imag)
        self.period = 2*numpy.pi/self.omega
        self.relax = values[-1].real
    def initial_state(self, delta_x):
        """Find initial state that is distance delta_x from fixed point
        """
        coefficients = numpy.linalg.lstsq(self.image_2d, numpy.array([delta_x,0]),rcond=None)[0]
        return numpy.dot(self.projection, coefficients) + self.fixed_point
    def map_time(self, x_initial):
        """Find time and position that x_initial maps to x[2] = r-1
        """
        h_max = 0.0025
        tenths = numpy.empty((20,3))
        t_step = self.period/10
        # Integrate at least once because x_initial is on boundary
        x_last = lorenz_integrate(x_initial, 0, t_step, h_max=h_max, r=self.r)
        for i in range(20):
            x_next = lorenz_integrate(x_last, 0, t_step, h_max=h_max, r=self.r)
            if x_next[2] > self.r-1 > x_last[2]:
                break
            x_last = x_next
        else:
            raise RuntimeError("Failed to find bracket")
        def func(time):
            x_time = lorenz_integrate(x_last, 0, time, h_max=h_max, r=self.r)
            result = x_time[2] - (self.r-1)
            return result
        delta_t = scipy.optimize.brentq(func, 0, t_step)
        t_final = (i+1)*t_step + delta_t
        x_final = lorenz_integrate(x_last, 0 , delta_t, h_max=h_max, r=self.r)
        return t_final, x_final
def make_data(r):
    big_t = 10000
    t_sample = 0.03
    data = numpy.empty((big_t,3))
    relaxed = lorenz_integrate(numpy.ones(3), 0.0, 500.0, r=r)
    data[0] = relaxed
    for t in range(1,big_t):
        data[t] = lorenz_integrate(data[t-1], 0, t_sample, r=r)
    return data

class Slider(QWidget):
    def __init__(self, minimum, maximum, parent=None):
        super(Slider, self).__init__(parent=parent)
        self.verticalLayout = QVBoxLayout(self)
        self.label = QLabel(self)
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout = QHBoxLayout()
        spacerItem = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Vertical)
        self.horizontalLayout.addWidget(self.slider)
        spacerItem1 = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.resize(self.sizeHint())

        self.minimum = minimum
        self.maximum = maximum
        self.slider.valueChanged.connect(self.setLabelValue)
        self.x = None
        self.setLabelValue(self.slider.value())

    def setLabelValue(self, value):
        self.x = self.minimum + (float(value) / (self.slider.maximum() - self.slider.minimum())) * (
        self.maximum - self.minimum)
        self.label.setText("{0:.6g}".format(self.x))


class Widget(QWidget):
    def __init__(self, parent=None):
        super(Widget, self).__init__(parent=parent)
        quit_button = QPushButton('Quit',self)
        quit_button.clicked.connect(self.close)
        
        self.horizontalLayout = QHBoxLayout(self)
        
        self.w1 = Slider(20.0, 30.0)
        self.horizontalLayout.addWidget(self.w1)
        
        self.w2 = Slider(0, .5)
        self.horizontalLayout.addWidget(self.w2)
        
        self.w3 = Slider(0, .005)
        self.horizontalLayout.addWidget(self.w3)

        self.win = pg.GraphicsWindow(title="Basic plotting examples")
        self.horizontalLayout.addWidget(self.win)
        self.p6 = self.win.addPlot(title="My Plot")
        self.curve = self.p6.plot(pen='r')
        
        self.win2 = pg.GraphicsWindow(title="Basic plotting examples")
        self.horizontalLayout.addWidget(self.win2)
        self.p5 = self.win2.addPlot(title="My Plot")
        self.curve2 = self.p5.plot(pen='g')
        self.update_plot()

        self.w1.slider.valueChanged.connect(self.update_plot)
        self.w2.slider.valueChanged.connect(self.update_plot)
        self.w3.slider.valueChanged.connect(self.update_plot)
    
    def update_plot(self):
        r = self.w1.x + self.w2.x + self.w3.x
        print(f'r={r}')
        data = make_data(r)
        self.curve.setData(data[:,0],data[:,2])
        self.curve2.setData(data[:200,0]**2)

        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
