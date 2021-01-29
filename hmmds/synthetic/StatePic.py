""" StatePic.py <data_dir> <data_file> <vector_file> <model_file>
EG. ${PYTHON} ${P}/StatePic.py ${D} lorenz.4 m12s.4y lorenz.xyz m12s.4y

1. Create the 12 files data_dir/state0 ... data_dir/state11 each of
   which contain lists of 3-vectors that fall in that state

2. data_dir/states that has a single decoded state trajectory
"""
Copyright = '''
Copyright 2021 Andrew M. Fraser

This file is part of hmmds.

Hmmds is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

Hmmds is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

See the file gpl.txt in the root directory of the hmmds distribution
or see <http://www.gnu.org/licenses/>.
'''
import sys
from os.path import join
import pickle
import argparse

import numpy

import hmm.base

import MakeModel


def main(argv=None):
    '''Call with arguments: data_dir, data_file, vector_file, model_file

    Writes files named ['state%d'%n for n in range(nstates)] to the
    data_dir.  Each file consists of points in vector_file that are
    decoded to the the state number specified in the name.  The states
    are assigned by using the model in model_file to Viterbi decode
    the data in data_file.

    '''

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Make data for figure on cover")
    parser.add_argument('data_dir', type=str)
    parser.add_argument('data_file', type=str)
    parser.add_argument('vector_file', type=str)
    parser.add_argument('model_file', type=str)

    args = parser.parse_args(argv)

    # Read in the output sequence
    y_data, cardy = MakeModel.read_data(args.data_dir, args.data_file)

    # Read in model
    mod = pickle.load(open(join(args.data_dir, args.model_file), 'rb'))
    nstates = mod.p_state_initial.shape[-1]

    # Viterbi decode to get time series of states, ie, state sequence
    ss = mod.decode(y_data)

    # Write state sequence to file named "states"
    f = open(join(args.data_dir, 'states'), 'w')
    for s in ss:
        print(s, file=f)
    f.close()

    # Read in time series of vectors
    vectors = [
        list(map(float, line.split())) for line in MakeModel.skip_header(
            open(join(args.data_dir, args.vector_file), 'r'))
    ]

    # Write vectors to each state file, ie, state0, state1, .. state11
    ssf = open(join(args.data_dir, 'state_sequence'), 'w')
    f = list(
        open(join(args.data_dir, 'state' + str(s)), 'w')
        for s in range(nstates))
    for t in range(len(ss)):
        print('{0:5d} {1:d}'.format(t, ss[t]), file=ssf)
        print('{0:7.4f} {1:7.4f} {2:7.4f}'.format(*vectors[t]), file=f[ss[t]])
    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
