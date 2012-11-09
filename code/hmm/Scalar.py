# Scalar.py  Copyright (c) 2003, 2007, 2008 Andrew Fraser
""" Implements HMMs with discrete observations.
"""
import random, numpy
import algorithms_0 as ALG
#import algorithms_1 as ALG

def print_V(V):
    for x in V:
        print '%-6.3f'%x,
    print ''
def print_Name_V(name,V):
    print name+' =',
    print_V(V)
def print_Name_VV(name,VV):
    print '  '+name+' ='
    for V in VV:
        print '   ',
        print_V(V)

## ----------------------------------------------------------------------
class HMM:
    """A Hidden Markov Model implementation with the following
    groups of methods:

    Tools for applications: forward(), backward(), train(), decode(),
    reestimate() and simulate()
    """
    def __init__(self, P_S0,P_S0_ergodic,P_ScS,P_YcS):
        """Builds a new Hidden Markov Model"""
        self.N =len(P_S0)
        self.P_S0 = numpy.matrix(P_S0)
        self.P_S0_ergodic = numpy.matrix(P_S0_ergodic)
        self.P_ScS = numpy.matrix(P_ScS)
        self.P_YcS = numpy.array(P_YcS)
        return # End of __init__()
    def Py_calc(self,y):
        """
        Allocate self.Py and assign values self.Py[t,i] = P(y(t)|s(t)=i)
       
        On entry:
        self    is an HMM
        y       is a sequence of observations
        """

        # Check size and initialize self.Py
        self.T = len(y)
        try:
            assert(self.Py.shape is (self.T,self.N))
        except:
            self.Py = numpy.zeros((self.T,self.N),numpy.float64)
        for t in xrange(self.T):
            self.Py[t,:] = self.P_YcS[:,y[t]]
        return self.Py # End of Py_calc()
    def forward(self):
       """
       On entry:
       self       is an HMM
       self.Py    has been calculated
       self.T     is length of Y
       self.N     is number of states
       On return:
       self.gamma[t] = Pr{y(t)=y(t)|y_0^{t-1}}
       self.alpha[t,i] = Pr{s(t)=i|y_0^t}
       return value is log likelihood of all data
       """

       # Check size and initialize alpha and gamma
       try:
           assert(self.alpha.shape==(self.T,self.N) and len(self.gamma)==self.T)
       except:
           self.alpha = numpy.zeros((self.T,self.N),numpy.float64)
           self.gamma = numpy.zeros((self.T,),numpy.float64)
       ALG.forward(self.P_S0, self.P_ScS, self.Py, self.gamma, self.alpha)
       return (numpy.log(self.gamma)).sum() # End of forward()
    def backward(self):
        """
        On entry:
        self    is an HMM
        y       is a sequence of observations
        exp(PyGhist[t]) = Pr{y(t)=y(t)|y_0^{t-1}}
        On return:
        for each state i, beta[t,i] = Pr{y_{t+1}^T|s(t)=i}/Pr{y_{t+1}^T}
        """
       # Check size and initialize beta
        try:
            assert (self.beta.shape == (self.T,self.N))
        except:
           self.beta = numpy.zeros((self.T,self.N),numpy.float64)
        ALG.backward(self.P_ScS, self.Py, self.gamma, self.beta)
        return # End of backward()
    def train(self, y, N_iter=1, display=True):
        # Do (N_iter) BaumWelch iterations
        LLL = []
        for it in xrange(N_iter):
            self.Py_calc(y)
            LLps = self.forward()/len(y)
            if display:
                print "it= %d LLps= %7.3f"%(it,LLps)
            LLL.append(LLps)
            self.backward()
            self.reestimate(y)
        return LLL # End of train()
    def reestimate_s(self):
        """ Reestimate state transition probabilities and initial
        state probabilities.  Given the observation probabilities, ie,
        self.state[s].Py[t], given alpha, beta, gamma, and Py, these
        calcuations are independent of the observation model
        calculations."""
        u_sum = numpy.zeros((self.N,self.N),numpy.float64)
        new_P_S0 = numpy.zeros((self.N,))
        new_P_S0_ergodic = numpy.zeros((self.N,))
        w  = ALG.reestimate_a(self.Py, self.gamma, self.alpha, self.beta, u_sum,
                         new_P_S0, new_P_S0_ergodic)
        # w is simply self.alpha which has been multiplied by beta element-wise
        self.P_S0, self.P_S0_ergodic, self.P_ScS = ALG.reestimate_s(
            self.P_ScS, u_sum, new_P_S0, new_P_S0_ergodic)
        return (w,new_P_S0_ergodic) # End of reestimate_s()
    def reestimate(self,y):
        """ Reestimate all paramters.  In particular, reestimate observation
        model.
        """
        w,sum_w = self.reestimate_s()
        if not type(y) == numpy.ndarray:
            y = numpy.array(y,numpy.int32)
        assert(y.dtype == numpy.int32 and y.shape == (self.T,))
        for yi in xrange(self.P_YcS.shape[1]):
            self.P_YcS[:,yi] = w.take(numpy.where(y==yi)[0],axis=0
                                      ).sum(axis=0)/sum_w
        return # End of reestimate()
    def decode(self,y):
        """Use the Viterbi algorithm to find the most likely state
           sequence for a given observation sequence y"""
        self.Py_calc(y)
        return ALG.viterbi(self.T,self.N,
                         numpy.log(numpy.maximum(self.P_S0,1e-30)),
                         numpy.log(numpy.maximum(self.P_ScS,1e-30)),
                         numpy.log(numpy.maximum(self.Py,1e-30)))
    # End of decode()
    def simulate(self,length,seed=3):
        """generates a random sequence of observations of given length"""
        random.seed(seed)
        # Set up cumulative distributions
        cum_init = numpy.cumsum(self.P_S0_ergodic.A[0])
        cum_tran = numpy.cumsum(self.P_ScS.A,axis=1)
        cum_y = numpy.cumsum(self.P_YcS,axis=1)
        # Initialize lists
        outs = []
        states = []
        def cum_rand(cum): # A little service function
            return numpy.searchsorted(cum,random.random())
        # Select initial state
        i = cum_rand(cum_init)
        # Select subsequent states and call model to generate observations
        for t in xrange(length):
            states.append(i)
            outs.append(cum_rand(cum_y[i]))
            i = cum_rand(cum_tran[i])
        return (states,outs) # End of simulate()
    def link(self, From, To, P):
        """ Create (or remove) a link between state "From" and state "To".
        The strength of the link is a function of both the argument
        "P" and the existing P_ScS array.  Set P_ScS itself if you
        need to set exact values.  Use this method to modify topology
        before training.
        """
        self.P_ScS[From,To] = P
        self.P_ScS[From,:] /= self.P_ScS[From,:].sum()
    def dump_base(self):
        print "dumping a %s with %d states"%(self.__class__,self.N)
        print_Name_V(' P_S0=        ',self.P_S0.A[0])
        print_Name_V(' P_S0_ergodic=',self.P_S0_ergodic.A[0])
        print_Name_VV('P_ScS', self.P_ScS.A)
        return #end of dump_base()
    def dump(self):
        self.dump_base()
        print_Name_VV('P_YcS=', self.P_YcS)
        return #end of dump()
if __name__ == '__main__':  # Test code
    """ Test the code in this file and in algorithms_1.py by manipulating
    the HMM in Figure 1.6 (fig:dhmm) in the book.
    """
    P_S0 = numpy.array([1./3., 1./3., 1./3.])
    P_S0_ergodic = numpy.array([1./7., 4./7., 2./7.])
    P_ScS = numpy.array([
            [0,   1,   0],
            [0,  .5,  .5],
            [.5, .5,   0]
            ],numpy.float64)
    P_YcS = numpy.array([
            [1, 0,     0],
            [0, 1./3., 2./3.],
            [0, 2./3., 1./3.]
            ])
    mod = HMM(P_S0,P_S0_ergodic,P_ScS,P_YcS)
    S,Y = mod.simulate(500)
    Y = numpy.array(Y,numpy.int32)
    E = mod.decode(Y)
    print '%3s, %3s, %3s'%('y','S','E_s')
    for triple in zip(Y,S,E[:10]):
        print '%3d, %3d, %3d'%triple
    L = mod.train(Y,N_iter=10)
    mod.dump()

#--------------------------------
# Local Variables:
# mode: python
# End:
