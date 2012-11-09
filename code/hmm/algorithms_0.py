# algorithms_0.py
#
# Copyright (c) 2004, 2008 Ralf Juengling, Andrew Fraser
# See __init__.py for license information

import numpy

# Changes:

# 2008-8-31 I (Andy Fraser) took Ralf's 2004 code as a starting point
# for hmmpy.  Changed to use precomputed output probabilities and to
# support multiple sequences.  My goals for this version is that the
# code be easy to read.

# Array/Matrix layout (N=number of states, T=number of time steps):

def forward(P_S0,    # Probability vector (1xN matrix) for initial state
            P_ScS,   # NxN matrix of state transition probabilities
            Py,      # List (or iterator) of vectors of
                     # observation probabilities.  Py[t][s] is the
                     # probability that state s would produce y[t]
            gamma,   # Length T array 
            alpha    # TxN array
            ):
    # The following ugly temporary storage cuts forward's runtime by
    # more than a factor of 2
    last = numpy.matrix(P_S0*1.0)   # Allocate and initialize temporary storage
    lastA = last.A    # View as array to make * element-wise multiplicaiton
    for t in xrange(len(Py)):
        lastA *= Py[t]          # Element-wise multiply
        gamma[t] = lastA.sum()
        lastA /= gamma[t]
        alpha[t,:] = lastA[0,:]
        last[:,:] = last*P_ScS  # Vector matrix product
    return # End of forward()

def backward(P_ScS,
            Py,      # Py[t][i] = P(y(t)|S(t)=i)
            gamma,   # Values precomputed by forward()
            beta     # zeros(T, N)
            ):
    
    # initialize
    lastbeta = numpy.mat(numpy.ones(beta.shape[1]))
    lastbetaA = lastbeta.A[0,:]   # Array view so * is element-wise multiply
    pscst = P_ScS.T               # Transpose view
    # iterate
    for t in xrange(len(Py)-1,-1,-1):
        beta[t,:] = lastbetaA
        lastbeta[:,:] = (lastbetaA*Py[t,:]/gamma[t])*pscst
    return # End of backward()

def viterbi(T,     # Number of time steps
            N,     # Number of states
            L_S0,  # Vector of logs of the initial state probabilities
            L_ScS, # Array of logs of state transition probabilities
            L_Py   # List of vectors of logs of observation
                   # probabilities.
            ):

    pred = numpy.zeros((T, N), numpy.int32) # Array of best predecessors
    ss = numpy.zeros((T, 1), numpy.int32)   # State sequence
    nu = L_Py[0] + L_S0
    for t in xrange(1,T):
        omega = L_ScS.T + nu
        pred[t,:] = omega.T.argmax(axis=0)   # Best predecessor
        nu = pred[t,:].choose(omega.T) + L_Py[t]
    lasts = numpy.argmax(nu)
    for t in xrange(T-1,-1,-1):
        ss[t] = lasts
        lasts = pred[t,lasts]
    return ss.flat # End of viterbi

def reestimate_a(Py,    # Observation probabilities.  Py[t,s] is the
                        # probability that state s would produce y[t]
                 gamma, # numpy array shape (T,)
                 alpha, # Array TxN
                 beta,  # Array TxN
                 u_sum, # Numpy array for new P_ScS
                 P_S0,  # Numpy array for new initial state probabilities
                 P_S0_ergodic
                 ):
    """ Accumulation portion of algorithm for reestimating model
        parameters of initial state probabilities and state transition
        probabilities.  Add to passed arrays u_sum, P_S0, and
        P_S0_ergodic without normalizing so that the function can be
        used for multiple segments.
    """
    T,N = alpha.shape
    for t in xrange(T-1):
        u_sum += numpy.outer(alpha[t]/gamma[t+1], Py[t+1]*beta[t+1,:])
    alpha *= beta
    P_S0_ergodic += alpha.sum(axis=0)
    P_S0 += alpha[0]
    return alpha # End of reestimate_a()

def reestimate_s(ScS, u_sum, new_P_S0, new_P_S0_ergodic):
    """ Calculate new model parameters from accumulated sums.  All
    that is required is reformating and normalization.
    """
    assert u_sum.shape == ScS.shape
    ScST = ScS.A.T # To get element wise multiplication and correct /
    ScST *= u_sum.T
    ScST /= ScST.sum(axis=0)
    # Now normalize and return
    return (numpy.mat(new_P_S0/new_P_S0.sum()),
    numpy.mat(new_P_S0_ergodic/new_P_S0_ergodic.sum()),
    numpy.mat(ScS))

#------------------------------
# Local Variables:
# mode: python
# End:
