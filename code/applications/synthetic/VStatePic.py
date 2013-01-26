"""
VStatePic.py data_dir y_name

Creates varg_stateN (N in 0..11) in the directory named by data

y[0][t] is a numpy.array containing the observation
y[1][t] is a numpy.array containing the context
context = (y[0][t-1],y[0][t-2],...,y[0][t-taumax],1.0)

Copyright (c) 2005, 2007, 2013 Andrew Fraser
This file is part of HMM_DS_Code.

"""

import sys
import os.path
import pickle
import numpy
from hmm import C
from MakeModel import read_data, skip_header
from hmm.VARG import VARG

data_dir,y_name = sys.argv[1:3]

def MakeVARG_HMM(Nstates,Odim,Cdim):
    '''Returns a normalized random initial model
    '''
    from Scalar import make_random as random_p
    from numpy import random
    random.seed(6)
    P_S0 = random_p((1,Nstates))[0]
    P_S0_ergodic = random_p((1,Nstates))[0]
    P_ScS = random_p((Nstates,Nstates))

    Icovs = np.empty((Nstates, Odim, Odim))
    As = np.empty((Nstates, Odim, Cdim))
    ICovs = Nstates*[None]
    for i in range(Nstates):
        ICovs[i,:,:] = np.eye(Odim)/100
        As[i,:,:] = random_p((Odim,Cdim))

    params = (Icovs,
              As,
              30.0,  # a
              0.1    # b
    )
    # Recall:   Cov = (b * scipy.eye(dim_Y) + ZZT)/(a + sum_w[i])
    return HMM(P_S0,P_S0_ergodic,params,P_ScS, VARG)

# Read data and construct Y.
Odim,Cdim,N_states = (3,4,12)
Y = []
f = file(os.path.join(data_dir, y_name), 'r')
tail_one = scipy.ones(1,scipy.float32) # Because Odim = 3 and Cdim = 4
data = scipy.zeros(Odim,scipy.float32)
for line in f:
    context = scipy.concatenate((data,tail_one))
    data = scipy.array(list(map(float,line.split())),scipy.float32)
    Y.append([data,context])
f.close()
Y.pop(0) # first y has no context, delete it

# Now for each t, y[t] = [observation,context] where observation is
# the 3-d state at time t and context is 4-d; the 3-d state at time
# t-1 with 1.0 appended.  1.0 lets the state use a column of A to
# provide a fixed offset
random.seed(6) # 96 is interesting too
model = MakeVARG_HMM(N_states,Odim,Cdim) # Make a random initial model
model.train(Y, 25)
states = model.decode(Y)                 # Do Viterbi decoding
vs = N_states*[None]
for s in range(N_states):
    vs[s] = []  # vs[s] is a list of vectors that belong in state s.
for t in range(0,len(states)):
    vs[states[t]].append(Y[t][0])
for s in range(N_states):
    f = file(os.path.join(data_dir, 'varg_state'+str(s)), 'w')
    for v in vs[s]:
        print(v[0], v[1], v[2], file=f)
                                 
#Local Variables:
#mode:python
#End:
