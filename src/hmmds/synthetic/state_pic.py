""" StatePic.py <data_dir> <data_file> <vector_file> <model_file>
EG. ${PYTHON} ${P}/StatePic.py ${D} lorenz.4 m12s.4y lorenz.xyz m12s.4y

1. Create the 12 files data_dir/state0 ... data_dir/state11 each of
   which contain lists of 3-vectors that fall in that state

2. data_dir/states that has a single decoded state trajectory
"""

import sys
import os.path
import pickle
import argparse

import hmmds.synthetic.make_model


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

    # Read in model
    with open(os.path.join(args.data_dir, args.model_file), mode='rb') as _file:
        mod = pickle.load(_file)
    n_states = mod.p_state_initial.shape[-1]

    # Read in the sequence of observations, ie, a time series, and
    # Viterbi decode it to get a sequence of states.
    state_sequence = mod.decode(
        hmmds.synthetic.make_model.read_data(args.data_dir, args.data_file)[0])

    # Write the state sequence to file named "states".
    with open(os.path.join(args.data_dir, 'states'), encoding='utf-8',
              mode='w') as states_file:
        for state in state_sequence:
            print(state, file=states_file)

    # Read in the sequence of original vectors.
    with open(os.path.join(args.data_dir, args.vector_file),
              encoding='utf-8',
              mode='r') as vector_file:
        vector_sequence = [
            map(float, line.split())
            for line in hmmds.synthetic.make_model.skip_header(vector_file)
        ]
    assert len(state_sequence) == len(vector_sequence)

    with open(os.path.join(args.data_dir, 'state_sequence'),
              encoding='utf-8',
              mode='w') as ss_file:

        # Create a list of open files, one for each state
        # pylint: disable = consider-using-with
        state_files = list(
            open(os.path.join(args.data_dir, f'state{state}'),
                 encoding='utf-8',
                 mode='w') for state in range(n_states))

        # Write vectors to each state file, ie, state0, state1, .. state11
        for t, state_t in enumerate(state_sequence):
            ss_file.write(f'{t:5d} {state_t:d}\n')
            # pylint: disable = consider-using-f-string
            state_files[state_t].write(
                '{0:7.4f} {1:7.4f} {2:7.4f}\n'.format(*vector_sequence[t]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
