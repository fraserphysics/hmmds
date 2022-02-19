""" explore.py For exploring parameters of Lorenz system.

Derived from https://www.pythonguis.com/tutorials/plotting-pyqtgraph/

"""

from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpacerItem, \
    QVBoxLayout, QWidget, QPushButton
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os

import numpy
from hmmds.synthetic.filter.lorenz_sde import lorenz_integrate

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
