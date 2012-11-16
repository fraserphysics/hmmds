''' MakeModel.py <H_dir> <data_dir> <data_file> <model_file>
 EG. python MakeModel.py data lorenz.4 m12s.4y
'''

import sys, os.path, pickle, random, numpy

from itertools import dropwhile

def skip_header(lines):
    isheader = lambda line: line.startswith("#")
    return dropwhile(isheader, lines)
    
def read_data(data_dir, data_file):
    '''Read data and return as numpy array
    '''
    lines = skip_header(open(os.path.join(data_dir, data_file), 'r'))
    y = numpy.array([int(line)-1 for line in lines],numpy.int32)
    return y, y.max()+1

def randomP(A):
    """ Fill allocated array A with random normalized probability
    """
    sum = 0
    for i in range(len(A)):
        x = random.random()
        sum += x
        A[i] = x
    A /= sum
    return A

def main(argv=None):

    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    nstates = 12
    n, data_dir, data_file, model_file = argv
    niterations = int(n) # maximum number of iterations
    #from Scalar import HMM
    from C import HMM_SPARSE as HMM
    #from C import HMM

    Y, cardy = read_data(data_dir, data_file)

    random.seed(7)
    P_S0 = randomP(numpy.zeros(nstates))
    P_S0_ergodic = randomP(numpy.zeros(nstates))
    P_ScS = numpy.zeros((nstates,nstates))
    P_YcS = numpy.zeros((nstates,cardy))
    for AA in (P_ScS,P_YcS):
        for A in AA:
            randomP(A)

    # Train the model
    mod = HMM(P_S0,P_S0_ergodic,P_ScS,P_YcS)
    mod.train(Y,N_iter=niterations)

    # Strip alpha, beta, and Py, then save model in <model_file>
    mod.alpha = None
    mod.beta = None
    mod.Py = None
    f = open(os.path.join(data_dir, model_file), 'wb')
    pickle.dump(mod, f)
    f.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
