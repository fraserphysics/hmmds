""" ecg_grid.py Utilities to plot ECGs for physicians

"""
import math

import numpy
import pint

import plotscripts.utilities

PINT = pint.UnitRegistry()


def get_samples(a_in, space):
    '''Sample the interval spanned by a_in at integer spacings of
    space

    Args:
        a_in: Sorted sequence of numbers
        space: Distance between samples

    '''
    bottom = space * math.ceil(a_in[0] / space)
    top = space * round(a_in[-1] / space)
    result = numpy.linspace(bottom, top, round((top - bottom) / space) + 1)
    return result


def y_ticks_labels(y_min, y_max):
    '''Return ticks and labels for the voltage axis of ECG

    Args:
       y_min, y_max: Electric potential in mV

    '''
    y_minor_ticks = numpy.arange(y_min, y_max + 0.05, 0.1)
    y_major_ticks = get_samples(y_minor_ticks, 0.5)
    y_label_values = get_samples(y_minor_ticks, 2.0)
    y_label_indices = numpy.array(
        numpy.searchsorted(y_major_ticks, y_label_values))
    y_labels_text = [f'{y:.0f}' for y in y_label_values]
    y_labels = [''] * len(y_major_ticks)
    for label, index in zip(y_labels_text, y_label_indices):
        y_labels[index] = label
    return y_minor_ticks, y_major_ticks, y_labels


def x_ticks_labels(s_start, s_stop):
    '''Return ticks and labels for the time axis of ECG

    Args:
       s_start, s_stop: Times in seconds

    '''
    x_minor_ticks = numpy.arange(s_start, s_stop + .01, 0.04)
    x_major_ticks = get_samples(x_minor_ticks, .2)
    x_labels = [''] * len(x_major_ticks)

    x_label_values = get_samples(x_major_ticks, 1.0)
    x_label_indices = numpy.array(
        numpy.searchsorted(x_major_ticks, x_label_values))
    formatted = [
        plotscripts.utilities.format_time(t * PINT('seconds'))
        for t in x_label_values
    ]
    for label, index in zip(formatted, x_label_indices):
        if index < len(x_labels):
            x_labels[index] = label
    return x_minor_ticks, x_major_ticks, x_labels


def decorate(axes, s_start, s_stop, y_min, y_max):
    '''Put grid on plot and label axes

    Args:
       y_min, y_max: Electric potential in mV
       s_start, s_stop: Times in seconds

    '''
    y_minor_ticks, y_major_ticks, y_labels = y_ticks_labels(y_min, y_max)
    x_minor_ticks, x_major_ticks, x_labels = x_ticks_labels(s_start, s_stop)
    axes.set_ylim(y_min, y_max)
    axes.set_xlim(s_start, s_stop)
    # From Google AI "matplotlib EKG grid"
    # Customize the grid
    axes.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    axes.grid(which='minor',
              linestyle='-',
              linewidth='0.5',
              alpha=.1,
              color='red')

    axes.set_xticks(x_major_ticks, x_labels)
    axes.set_xticks(x_minor_ticks, minor=True)
    axes.set_yticks(y_major_ticks, y_labels)
    axes.set_yticks(y_minor_ticks, minor=True)
