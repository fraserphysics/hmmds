"""utilities.py:

"""
from __future__ import annotations

import numpy


def read_tang(data_file):
    """Read one of Tang's laser data files as an array.
    """
    with open(data_file, 'r') as file:
        lines = file.readlines(
        )  # There are only 28278 lines and memory is big and cheap

    assert lines[0].split()[0] == 'BEGIN'
    assert lines[-1].split()[-1] == 'END'
    return numpy.array([[float(x) for x in line.split()] for line in lines[1:-1]
                       ]).T


class Parameters:
    """Parameters for laser data.

    A tuple of parameters is passed to objective_funciton which uses
    this class to associate a name with each value and then invokes
    make_non_stationary with a Parameters instance as an argument.

    Subclasses could let you optimize over smaller sets of values.
    """

    # The order of names in variables must match the order in __init__
    variables = """
s r b
x_initial_0 x_initial_1 x_initial_2
t_ratio x_ratio offset
state_noise observation_noise""".split()

    def __init__(
        self,
        s,
        r,
        b,
        x_initial_0,
        x_initial_1,
        x_initial_2,
        t_ratio,
        x_ratio,
        offset,
        state_noise=0.7,
        observation_noise=0.5,
        # The following are not subject to optimization
        fudge=1.0,
        laser_dt=0.04,
    ):
        var_dict = vars()
        for name in self.variables:
            setattr(self, name, var_dict[name])
        self.fudge = fudge  # Roll into state_noise
        self.laser_dt = laser_dt

    def values(self: Parameters):
        return tuple(getattr(self, key) for key in self.variables)

    def __str__(self: Parameters):
        result = ''
        for key in self.variables:
            result += f'{key} {getattr(self,key)}\n'
        return result

    def write(self: Parameters, path):
        with open(path, 'w') as file_:
            file_.write(self.__str__())


def read_parameters(path):
    """Remove the copy in optimize_ekf.py
    """
    in_dict = {}
    with open(path, 'r') as file_:
        for line in file_.readlines():
            parts = line.split()
            if parts[0] in Parameters.variables:  # Skip result strings
                in_dict[parts[0]] = float(parts[1])
    value_list = [in_dict[name] for name in Parameters.variables]
    return Parameters(*value_list)
