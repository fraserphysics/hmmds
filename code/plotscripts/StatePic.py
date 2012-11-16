""" StatePic.py <data_dir> <data_file> <vector_file> <model_file> 
EG. ${PYTHON} ${P}/StatePic.py ${D} lorenz.4 m12s.4y lorenz.xyz m12s.4y

1. Create the 12 files data_dir/state0 ... data_dir/state11 each of
   which contain lists of 3-vectors that fall in that state

2. data_dir/states that has a single decoded state trajectory
"""

import sys, os.path, pickle, numpy
data_dir, data_file, vector_file, model_file = sys.argv[1:5]

from itertools import dropwhile

def skip_header(lines):
    isheader = lambda line: line.startswith("#")
    return dropwhile(isheader, lines)
    
def read_data(data_dir, data_file):
    # Read in <data_file>
    f = open(os.path.join(data_dir, data_file), 'r')
    lines = skip_header(f)
    y = [int(line)-1 for line in lines]
    f.close()
    return y, max(y)+1

# Read in the output sequence
Y, cardy = read_data(data_dir, data_file)
Y = numpy.array(Y,numpy.int32)

# Read in model
mod = pickle.load(open(os.path.join(data_dir, model_file),'rb'))
nstates = mod.P_S0.shape[-1] # P_S_0 is a matrix with shape (1,nstates)

# Viterbi decoding
ss = mod.decode(Y)
f = open(os.path.join(data_dir, 'states'),'w')
for s in ss:
    print(s, file=f)
f.close()
# print "State trajectory=\n",trajectory

# Read in vectors
#vectors = []
f = open(os.path.join(data_dir, vector_file),'r')
lines = skip_header(f)
vectors = [list(map(float,line.split())) for line in lines]
f.close()

# vs[s] will be a list of vectors that belong in state s
vs = [[] for state in range(nstates)]

for t in range(0,len(ss)):
    vs[ss[t]].append(vectors[t])

for s in range(nstates):
    f = open(os.path.join(data_dir, "state%d"%s), 'w')
    for v in vs[s]:
        print(v[0], v[1], v[2], file=f)
    f.close()

#Local Variables:
#mode:python
#End:
