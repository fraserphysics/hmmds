"""utilities.py: Support for making figures with matplotlib
"""

# Try something like:

# fig, (ax_ecg, ax_onr, ax_O2) = matplotlib.pyplot.subplots(nrows=3, ncols=1, sharex=True)

from __future__ import annotations  # Enables, eg, (self: HMM,

import sys

import numpy as np
# Utilities for axis labels in LaTeX with units in \rm font
label_magnitude_unit = lambda lab, mag, unit: (r'$%s/(10^{%d}\ {\rm{%s}})$' %
                                               (lab, mag, unit))
label_unit = lambda lab, unit: r'$%s/{\rm{%s}}$' % (lab, unit)
label_magnitude = lambda lab, mag: r'$%s/10^{%d}$' % (lab, mag)
magnitude = lambda A: int(np.log10(np.abs(A).max()))

# The use of dicts in __init__ defeats pylint.  Hence the following line.


# pylint: disable=no-member
class Axis:
    """ Class for managing scaling and labeling 1-d axes and data.
    """

    def __init__(self: Axis, **kwargs):  # Any keyword argument is legal
        """ Hides some logic that figures out how to format axis
        labels, ticks and tick labels.
        """
        defaults = dict(
            data=None,  # np array
            magnitude='auto',  # Power of 10.  False to suppress
            ticks='auto',  # np array.  False to suppress ticks
            label=False,  # string, eg 'force'
            units=None,  # eg, 'dyn'
            tick_label_flag=True)
        self.__dict__.update(defaults)
        self.__dict__.update(kwargs)
        # pylint: disable=access-member-before-definition
        if self.magnitude == 'auto' and isinstance(self.data, np.ndarray):
            self.magnitude = magnitude(self.data)
        if self.label is False:
            return
        # Calculate label string
        has_magnitude = not (self.magnitude == 0 or self.magnitude is False)
        units = isinstance(self.units, str)
        self.label = {
            (True, True):
                label_magnitude_unit(self.label, self.magnitude, self.units),
            (True, False):
                label_unit(self.label, self.units),
            (False, True):
                label_magnitude(self.label, self.magnitude),
            (False, False):
                r'$%s$' % (self.label,)
        }[(units, has_magnitude)]

    def get_data(self: Axis):
        """Return self.data possibly scaled. """
        if isinstance(self.magnitude, int):
            return self.data / 10**self.magnitude
        return self.data

    def set_label(self: Axis, func):
        """Apply func, eg, mpl.axis.set_xlabel(), to self._label
        """
        if self.label is not False:
            func(self.label)

    def set_ticks(self: Axis, tick_func, label_func):
        """Apply functions, eg, mpl.axis.set_xticks() and
        mpl.axis,set_xticklabels() to self.ticks
        """
        if self.tick_label_flag is False:
            label_func([])
        if isinstance(self.ticks, str) and self.ticks == 'auto':
            return

        tick_func(self.ticks, minor=False)
        if self.tick_label_flag:
            if np.abs(self.ticks - self.ticks.astype(int)).sum() == 0:
                label_func([r'$%d$' % int(f) for f in self.ticks])
            else:
                label_func([r'$%1.1f$' % f for f in self.ticks])
        return


def sub_plot(fig, position, x, y, plot_flag=True, label=None, color='b'):
    """ Make a subplot for fig using data and format in axis objects x and y
    """
    axis = fig.add_subplot(*position)
    if plot_flag:
        if x.data is None:
            axis.plot(y.get_data(), color=color, label=label)
        else:
            axis.plot(x.get_data(), y.get_data(), color=color, label=label)
    y.set_label(axis.set_ylabel)
    y.set_ticks(axis.set_yticks, axis.set_yticklabels)
    x.set_label(axis.set_xlabel)
    x.set_ticks(axis.set_xticks, axis.set_xticklabels)
    return axis


def read_data(data_file):
    """Read in "data_file" as an array"""
    with open(data_file, 'r') as file_:
        data = [[float(x) for x in line.split()] for line in file_.readlines()]
    return np.array(data).T


def import_and_parse(parse_args, argv):
    """ Boilerplate for plotscripts.

    Args:
        parse_args: Function that parses argv
        args: The command line

    Return: args, matplotlib, pyplot

    """
    import matplotlib  # pylint: disable=import-outside-toplevel

    if not argv:
        argv = sys.argv[1:]
    args = parse_args(argv)

    if args.show:
        matplotlib.use('Qt5Agg')
    else:
        matplotlib.use('PDF')  # Permits absence of enviroment variable DISPLAY
    import matplotlib.pyplot  # pylint: disable=import-outside-toplevel

    params = {
        'axes.labelsize': 12,
        #'text.fontsize': 10,
        'legend.fontsize': 10,
        'text.usetex': True,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11
    }
    matplotlib.rcParams.update(params)

    return args, matplotlib, matplotlib.pyplot


def format_time(time: pint.Quantity) -> str:
    """Return a LaTeX representation of a pint time

    """
    float_seconds = time.to('seconds').magnitude
    assert float_seconds >= 0.0
    int_seconds = int(float_seconds)
    fraction = float_seconds - int_seconds
    int_minutes = int_seconds // 60
    seconds = int_seconds - 60 * int_minutes
    hours = int_minutes // 60
    minutes = int_minutes - 60 * hours
    if fraction < 0.01:
        return f'{hours:2d}:{minutes:02d}:{seconds:02d}'
    return f'${hours:2d}:{minutes:02d}:{seconds+fraction:05.2f}$'


def read_particles(path, n_start, length):
    """Return forcast and update clouds from particle filter
    """
    result = {}

    with open(path, 'rb') as file_:
        for n in range(n_start * 2):
            np.load(file_)
        for n in range(n_start, n_start + length):
            result[(n, 'forecast')] = np.load(file_)
            try:
                result[(n, 'update')] = np.load(file_)
            except EOFError:
                result[(n, 'update')] = []
                break
    return result


#---------------
# Local Variables:
# eval: (python-mode)
# End:
