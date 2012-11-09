""" StatePic.py <data_dir> <data_file> <vector_file> <model_file> 
EG. ${PYTHON} ${P}/StatePic.py ${D} lorenz.4 m12s.4y lorenz.xyz m12s.4y

1. Create the 12 files data_dir/state0 ... data_dir/state11 each of
   which contain lists of 3-vectors that fall in that state

2. data_dir/states that has a single decoded state trajectory
"""

import sys, os.path, pickle, scipy
data_dir, data_file, vector_file, model_file = sys.argv[1:5]

import Scalar
from itertools import dropwhile

def skip_header(lines):
    isheader = lambda line: line.startswith("#")
    return dropwhile(isheader, lines)
    
def read_data(data_dir, data_file):
    # Read in <data_file>
    f = file(os.path.join(data_dir, data_file), 'r')
    lines = skip_header(f.xreadlines())
    y = [int(line)-1 for line in lines]
    f.close()
    return y, max(y)+1

# Read in the output sequence
Y, cardy = read_data(data_dir, data_file)
Y = scipy.array(Y)

# Read in model
mod = pickle.load(file(os.path.join(data_dir, model_file),'r'))
nstates = mod.P_S0.shape[-1] # P_S_0 is a matrix with shape (1,nstates)

# Viterbi decoding
ss = mod.decode(Y)
f = file(os.path.join(data_dir, 'states'),'w')
for s in ss:
    print >>f, s
f.close()
# print "State trajectory=\n",trajectory

# Read in vectors
#vectors = []
f = file(os.path.join(data_dir, vector_file),'r')
lines = skip_header(f.xreadlines())
vectors = [map(float,line.split()) for line in lines]
f.close()

# vs[s] will be a list of vectors that belong in state s
vs = [[] for state in xrange(nstates)]

for t in range(0,len(ss)):
    vs[ss[t]].append(vectors[t])

for s in xrange(nstates):
    f = file(os.path.join(data_dir, "state%d"%s), 'w')
    for v in vs[s]:
        print >>f, v[0], v[1], v[2]
    f.close()

#Local Variables:
#mode:python
#End:
