""" Scalar.py: Implements HMMs with discrete observations.

"""
# Copyright (c) 2003, 2007, 2008, 2012 Andrew M. Fraser
import random, numpy as np

def initialize(x,shape,dtype=np.float64):
    if x == None or x.shape != shape:
        return np.zeros(shape,dtype)
    return x*0
## ----------------------------------------------------------------------
class PROB(np.ndarray):
    '''Subclass of ndarray for probability matrices.  P[a,b] is the
    probability of b given a.  The class has additional methods and is
    designed to enable further subclasses with speed improvements
    implemented by uglier code.

    '''
    # See http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    def assign_col(self,i,col):
        '''
        Replace column of self with data specified by the parameters

        Parameters
        ----------
        i : int
            Column index
        col : array_like
            Column data

        Returns
        -------
        None
        '''
        self[:,i] = col
    def likelihoods(self,v):
        '''likelihoods for vector of data

        Parameters
        ----------
        v : array_like
            Column data

        Returns
        -------
        L : array_like

            Given T = len(v) and self.shape = (M,N), L.shape = (T,M)
            with the interpretation L[t,a] = Prob(v[t]|a)

        '''
        return self[:,v].T
    def inplace_elementwise_multiply(self,A):
        '''
        Multiply self by argument

        Parameters
        ----------
        A : array_like

        Returns
        -------
        None
        '''
        self *= A
    def normalize(self):
        '''
        Make each row a proability that sums to one

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        S = self.sum(axis=1)
        for i in range(self.shape[0]):
            self[i,:] /= S[i]
    def step_forward(self,A):
        '''
        Replace values of argument A with matrix product A*self

        Parameters
        ----------
        A : array_like

        Returns
        -------
        None
        '''
        A[:] = np.dot(A,self)
    def step_back(self,A):
        '''
        Replace values of argument A with matrix product self*A

        Parameters
        ----------
        A : array_like

        Returns
        -------
        None
        '''
        A[:] = np.dot(self,A)
    def values(self):
        '''
        Produce values of self


        This is a hack to free subclasses from the requirement of self
        being an nd_array

        Parameters
        ----------
        None

        Returns
        -------
        v : array_like
        '''
        return self
def make_prob(x):
    x = np.array(x)
    return PROB(x.shape,buffer=x.data)

class HMM:
    """A Hidden Markov Model implementation.

    Parameters
    ----------
    P_S0 : array_like
        Initial distribution of states
    P_S0_ergodic : array_like
        Stationary distribution of states
    P_ScS : array_like
        P_ScS[a,b] = Prob(s(1)=b|s(0)=a)
    P_YcS : array_like
        P_YcS[a,b] = Prob(y(0)=b|s(0)=a)
    prob=make_prob : function, optional
        Function to make conditional probability matrix

    Returns
    -------
    None

    Examples
    --------
    Illustrate/Test some methods by manipulating the HMM in Figure 1.6
    of the book.

    >>> P_S0 = np.array([1./3., 1./3., 1./3.])
    >>> P_S0_ergodic = np.array([1./7., 4./7., 2./7.])
    >>> P_ScS = np.array([
    ...         [0,   1,   0],
    ...         [0,  .5,  .5],
    ...         [.5, .5,   0]
    ...         ],np.float64)
    >>> P_YcS = np.array([
    ...         [1, 0,     0],
    ...         [0, 1./3., 2./3.],
    ...         [0, 2./3., 1./3.]
    ...         ])
    >>> mod = HMM(P_S0,P_S0_ergodic,P_ScS,P_YcS)
    >>> S,Y = mod.simulate(500)
    >>> Y = np.array(Y,np.int32)
    >>> E = mod.decode(Y)
    >>> table = ['%3s, %3s, %3s'%('y','S','Decoded')]
    >>> table += ['%3d, %3d, %3d'%triple for triple in zip(Y,S,E[:10])]
    >>> for triple in table:
    ...     print(triple)
      y,   S, Decoded
      2,   1,   1
      2,   1,   1
      1,   2,   2
      0,   0,   0
      1,   1,   1
      1,   2,   2
      2,   1,   1
      1,   2,   2
      2,   1,   1
      2,   2,   1
    >>> L = mod.train(Y,N_iter=4)
    it= 0 LLps=  -0.920
    it= 1 LLps=  -0.918
    it= 2 LLps=  -0.918
    it= 3 LLps=  -0.917
    >>> print(mod)
    A <class '__main__.HMM'> with 3 states
     P_S0         =0.000 0.963 0.037 
     P_S0_ergodic =0.142 0.580 0.278 
      P_ScS =
       0.000 1.000 0.000 
       0.000 0.519 0.481 
       0.512 0.488 0.000 
      P_YcS =
       1.000 0.000 0.000 
       0.000 0.335 0.665 
       0.000 0.726 0.274 

    """
    def __init__(
        self,         # HMM instance
        P_S0,         # Initial distribution of states
        P_S0_ergodic, # Stationary distribution of states
        P_ScS,        # P_ScS[a,b] = Prob(s(1)=b|s(0)=a)
        P_YcS,        # P_YcS[a,b] = Prob(y(0)=b|s(0)=a)
        prob=make_prob# Function to make conditional probability matrix
        ):
        """Builds a new Hidden Markov Model
        """
        self.N =len(P_S0)
        self.P_S0 = np.array(P_S0)
        self.P_S0_ergodic = np.array(P_S0_ergodic)
        self.P_ScS = prob(P_ScS)
        self.P_YcS = prob(P_YcS)
        self.Py = None
        self.alpha = None
        self.gamma = None
        self.beta = None
        return # End of __init__()
    def Py_calc(
        self,    # HMM
        y        # A sequence of integer observations
        ):
        """
        Allocate self.Py and assign values self.Py[t,i] = P(y(t)|s(t)=i)

        Parameters
        ----------
        y : array_like
            A sequence of integer observations

        Returns
        -------
        None
        """
        # Check size and initialize self.Py
        self.T = len(y)
        self.Py = initialize(self.Py,(self.T,self.N))
        self.Py[:,:] = self.P_YcS.likelihoods(y)
        return # End of Py_calc()
    def forward(self):
        """
        Implements recursive calculation of state probabilities given
        observation probabilities.

        Parameters
        ----------
        None

        Returns
        -------
        L : float
            Average log (base e) likelihood per point of entire observation
            sequence 

        Bullet points
        -------------

        On entry:

        * self       is an HMM

        * self.Py    has been calculated

        * self.T     is length of Y

        * self.N     is number of states

        Bullet points
        -------------

        On return:

        * self.gamma[t] = Pr{y(t)=y(t)|y_0^{t-1}}
        * self.alpha[t,i] = Pr{s(t)=i|y_0^t}
        * return value is log likelihood of all data

        """

        # Ensure allocation and size of alpha and gamma
        self.alpha = initialize(self.alpha,(self.T,self.N))
        self.gamma = initialize(self.gamma,(self.T,))
        last = np.copy(self.P_S0.reshape(-1)) # Copy
        for t in range(self.T):
            last *= self.Py[t]              # Element-wise multiply
            self.gamma[t] = last.sum()
            last /= self.gamma[t]
            self.alpha[t,:] = last
            self.P_ScS.step_forward(last)
        return (np.log(self.gamma)).sum() # End of forward()
    def backward(self):
        """
        Implements the Baum_Welch backwards pass through state conditional
        likelihoods of the obserations.

        Parameters
        ----------
        None

        Returns
        -------
        None
        
        Bullet points
        -------------

        On entry:

        * self    is an HMM

        * self.Py    has been calculated

        Bullet points
        -------------

        On return:

        * for each state i, beta[t,i] = Pr{y_{t+1}^T|s(t)=i}/Pr{y_{t+1}^T}

        """
        # Ensure allocation and size of beta
        self.beta = initialize(self.beta,(self.T,self.N))
        last = np.ones(self.N)
        # iterate
        for t in range(self.T-1,-1,-1):
            self.beta[t,:] = last
            last *= self.Py[t]
            last /= self.gamma[t]
            self.P_ScS.step_back(last)
        return # End of backward()
    def train(self, y, N_iter=1, display=True):
        '''Based on observations y, do N_iter iterations of model reestimation

        Use Baum-Welch algorithm to search for maximum likelihood
        model parameters.

        Parameters
        ----------
        y : array_like
            Sequence of integer observations
        N_iter : int, optional
            Number of iterations
        display : bool, optional
            If True, print the log likelihood per observation for each
            iteration

        Returns
        -------
        LLL : list
            List of log likelihood per observation for each iteration

        '''
        # Do (N_iter) BaumWelch iterations
        LLL = []
        for it in range(N_iter):
            self.Py_calc(y)
            LLps = self.forward()/len(y)
            if display:
                print("it= %d LLps= %7.3f"%(it,LLps))
            LLL.append(LLps)
            self.backward()
            self.reestimate(y)
        return LLL # End of train()
    def reestimate_s(self):
        """Reestimate state transition probabilities and initial
        state probabilities.

        Given the observation probabilities, ie, self.state[s].Py[t],
        given alpha, beta, gamma, and Py, these calcuations are
        independent of the observation model calculations.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        u_sum = np.zeros((self.N,self.N),np.float64)
        for t in range(self.T-1):
            u_sum += np.outer(self.alpha[t]/self.gamma[t+1],
                                 self.Py[t+1]*self.beta[t+1,:])
        self.alpha *= self.beta
        wsum = self.alpha.sum(axis=0)
        self.P_S0_ergodic = np.copy(wsum)
        self.P_S0 = np.copy(self.alpha[0])
        for x in (self.P_S0_ergodic, self.P_S0):
            x /= x.sum()
        assert u_sum.shape == self.P_ScS.shape
        self.P_ScS.inplace_elementwise_multiply(u_sum)
        self.P_ScS.normalize()
        return self.alpha # End of reestimate_s()
    def reestimate(self,y):
        """Reestimate all model paramters. In particular, reestimate
        observation model.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        w = self.reestimate_s()
        if not type(y) == np.ndarray:
            y = np.array(y,np.int32)
        assert(y.dtype == np.int32 and y.shape == (self.T,))
        for yi in range(self.P_YcS.shape[1]):
            self.P_YcS.assign_col(
                yi, w.take(np.where(y==yi)[0],axis=0).sum(axis=0))
        self.P_YcS.normalize()
        return # End of reestimate()
    def decode(self,y):
        """Use the Viterbi algorithm to find the most likely state
           sequence for a given observation sequence y

        Parameters
        ----------
        y : array_like
            Sequence of observations

        Returns
        -------
        ss : array_like
            Maximum likelihood state sequence
        """
        self.Py_calc(y)
        pred = np.zeros((self.T, self.N), np.int32) # Best predecessors
        ss = np.zeros((self.T, 1), np.int32)        # State sequence
        L_S0, L_ScS, L_Py = (np.log(np.maximum(x,1e-30)) for x in
                             (self.P_S0,self.P_ScS.values(),self.Py))
        nu = L_Py[0] + L_S0
        for t in range(1,self.T):
            omega = L_ScS.T + nu
            pred[t,:] = omega.T.argmax(axis=0)   # Best predecessor
            nu = pred[t,:].choose(omega.T) + L_Py[t]
        last_s = np.argmax(nu)
        for t in range(self.T-1,-1,-1):
            ss[t] = last_s
            last_s = pred[t,last_s]
        return ss.flat # End of viterbi
    # End of decode()
    def simulate(self,length,seed=3):
        """generates a random sequence of observations of given length

        Parameters
        ----------

        length : int
            Number of time steps to simulate
        seed : int, optional
            Seed for random number generator

        Returns
        -------

        states : array_like
            Sequence of states
        outs : array_like
            Sequence of observations
        """
        random.seed(seed)
        # Set up cumulative distributions
        cum_init = np.cumsum(self.P_S0_ergodic[0])
        cum_tran = np.cumsum(self.P_ScS.values(),axis=1)
        cum_y = np.cumsum(self.P_YcS.values(),axis=1)
        # Initialize lists
        outs = []
        states = []
        def cum_rand(cum): # A little service function
            return np.searchsorted(cum,random.random())
        # Select initial state
        i = cum_rand(cum_init)
        # Select subsequent states and call model to generate observations
        for t in range(length):
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

        Parameters
        ----------
        From : int
        To : int
        P : float

        Returns
        -------
        None

        FixMe: No test coverage.  Broken for cython code
        """
        self.P_ScS[From,To] = P
        self.P_ScS[From,:] /= self.P_ScS[From,:].sum()
    def __str__(self):
        def print_V(V):
            rv = ''
            for x in V:
                rv += '%-6.3f'%x
            return rv + '\n'
        def print_Name_V(name,V):
            return name+' =' + print_V(V)
        def print_Name_VV(name,VV):
            rv = '  '+name+' =\n'
            for V in VV:
                rv += '   ' + print_V(V)
            return rv

        rv = "A %s with %d states\n"%(self.__class__,self.N)
        rv += print_Name_V(' P_S0        ',self.P_S0)
        rv += print_Name_V(' P_S0_ergodic',self.P_S0_ergodic)
        rv += print_Name_VV('P_ScS', self.P_ScS.values())
        rv += print_Name_VV('P_YcS', self.P_YcS.values())
        return rv[:-1]

def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()

#--------------------------------
# Local Variables:
# mode: python
# End:
