""" Scalar.py: Implements basic HMM algorithms.  Default observation
models are defined in the "Scalar" module.

"""
# Copyright (c) 2003, 2007, 2008, 2012 Andrew M. Fraser
import numpy as np
from Scalar import initialize, Prob, Discrete_Observations, Class_y, make_prob

class HMM:
    """A Hidden Markov Model implementation.

    Parameters
    ----------
    P_S0 : array_like
        Initial distribution of states
    P_S0_ergodic : array_like
        Stationary distribution of states
    P_SS : array_like
        P_SS[a,b] = Prob(s(1)=b|s(0)=a)
    P_YS : array_like
        P_YS[a,b] = Prob(y(0)=b|s(0)=a)
    prob=make_prob : function, optional
        Function to make conditional probability matrix

    Examples
    --------
    Illustrate/Test some methods by manipulating the HMM in Figure 1.6
    of the book.

    >>> P_S0 = np.array([1./3., 1./3., 1./3.])
    >>> P_S0_ergodic = np.array([1./7., 4./7., 2./7.])
    >>> P_SS = np.array([
    ...         [0,   1,   0],
    ...         [0,  .5,  .5],
    ...         [.5, .5,   0]
    ...         ],np.float64)
    >>> P_YS = np.array([
    ...         [1, 0,     0],
    ...         [0, 1./3., 2./3.],
    ...         [0, 2./3., 1./3.]
    ...         ])
    >>> mod = HMM(P_S0,P_S0_ergodic,P_YS,P_SS)
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
    >>> L = mod.train(Y,n_iter=4)
    it= 0 LLps=  -0.920
    it= 1 LLps=  -0.918
    it= 2 LLps=  -0.918
    it= 3 LLps=  -0.917
    >>> print(mod)
    <class '__main__.HMM'> with 3 states
    P_S0         = [ 0.     0.963  0.037]
    P_S0_ergodic = [ 0.142  0.58   0.278]
    P_SS =
    [[ 0.     1.     0.   ]
     [ 0.     0.519  0.481]
     [ 0.512  0.488  0.   ]]
    P_YS =
    [[ 1.     0.     0.   ]
     [ 0.     0.335  0.665]
     [ 0.     0.726  0.274]
    
    """
    def __init__(
        self,         # HMM instance
        P_S0,         # Initial distribution of states
        P_S0_ergodic, # Stationary distribution of states
        P_YS,         # P_YS[a,b] = Prob(y(0)=b|s(0)=a)
        P_SS,         # P_SS[a,b] = Prob(s(1)=b|s(0)=a)
        y_mod=Discrete_Observations,
        prob=make_prob# Function to make conditional probability matrix
        ):
        """Builds a new Hidden Markov Model
        """
        self.n_states = len(P_S0)
        self.P_S0 = np.array(P_S0)
        self.P_S0_ergodic = np.array(P_S0_ergodic)
        self.P_SS = prob(P_SS)
        self.y_mod = y_mod(P_YS)
        self.alpha = None
        self.gamma = None
        self.beta = None
        self.n_y = None
        return # End of __init__()
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

        * self          is an HMM

        * self.P_Y      has been calculated

        * self.n_y      is length of Y

        * self.n_states is number of states

        Bullet points
        -------------

        On return:

        * self.gamma[t] = Pr{y(t)=y(t)|y_0^{t-1}}
        * self.alpha[t,i] = Pr{s(t)=i|y_0^t}
        * return value is log likelihood of all data

        """

        # Ensure allocation and size of alpha and gamma
        self.alpha = initialize(self.alpha, (self.n_y, self.n_states))
        self.gamma = initialize(self.gamma, (self.n_y,))
        last = np.copy(self.P_S0.reshape(-1)) # Copy
        for t in range(self.n_y):
            last *= self.P_Y[t]              # Element-wise multiply
            self.gamma[t] = last.sum()
            last /= self.gamma[t]
            self.alpha[t, :] = last
            self.P_SS.step_forward(last)
        return (np.log(self.gamma)).sum() # End of forward()
    def backward(self):
        """
        Implements the Baum Welch backwards pass through state conditional
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

        * self     is an HMM

        * self.P_Y has been calculated

        Bullet points
        -------------

        On return:

        * for each state i, beta[t,i] = Pr{y_{t+1}^T|s(t)=i}/Pr{y_{t+1}^T}

        """
        # Ensure allocation and size of beta
        self.beta = initialize(self.beta, (self.n_y, self.n_states))
        last = np.ones(self.n_states)
        # iterate
        for t in range(self.n_y-1, -1, -1):
            self.beta[t, :] = last
            last *= self.P_Y[t]
            last /= self.gamma[t]
            self.P_SS.step_back(last)
        return # End of backward()
    def train(self, y, n_iter=1, display=True):
        '''Based on observations y, do n_iter iterations of model reestimation

        Use Baum-Welch algorithm to search for maximum likelihood
        model parameters.

        Parameters
        ----------
        y : array_like
            Sequence of integer observations
        n_iter : int, optional
            Number of iterations
        display : bool, optional
            If True, print the log likelihood per observation for each
            iteration

        Returns
        -------
        LLL : list
            List of log likelihood per observation for each iteration

        '''
        # Do (n_iter) BaumWelch iterations
        LLL = []
        for it in range(n_iter):
            self.P_Y = self.y_mod.calc(y)
            self.n_y = len(self.P_Y)
            LLps = self.forward()/len(y) # log likelihood per step
            if display:
                print("it= %d LLps= %7.3f"%(it, LLps))
            LLL.append(LLps)
            self.backward()
            self.reestimate(y)
        return LLL # End of train()
    def reestimate(self,y):
        """Reestimate state transition probabilities and initial state
        probabilities, then call y_mod.reestimate method to update
        observation model parameters.

        Parameters
        ----------
        y : array
            Sequence of observations

        Returns
        -------
        None

        """
        u_sum = np.zeros((self.n_states, self.n_states), np.float64)
        for t in np.where(self.gamma[1:]>0)[0]: # Skip segment boundaries
            u_sum += np.outer(self.alpha[t]/self.gamma[t+1],
                              self.P_Y[t+1]*self.beta[t+1, :])
        self.alpha *= self.beta
        wsum = self.alpha.sum(axis=0)
        self.P_S0_ergodic = np.copy(wsum)
        self.P_S0 = np.copy(self.alpha[0])
        for x in (self.P_S0_ergodic, self.P_S0):
            x /= x.sum()
        assert u_sum.shape == self.P_SS.shape
        self.P_SS.inplace_elementwise_multiply(u_sum)
        self.P_SS.normalize()
        self.y_mod.reestimate(self.alpha,y)
        return # End of reestimate()
    def decode(self, y):
        """Use the Viterbi algorithm to find the most likely state
           sequence for a given observation sequence y

        Parameters
        ----------
        y : array_like
            Sequence of observations

        Returns
        -------
        ss : array
            Maximum likelihood state sequence

        """
        P_Y = self.y_mod.calc(y)
        self.n_y = len(P_Y)
        pred = np.empty((self.n_y, self.n_states), np.int32) # Best predecessors
        ss = np.ones((self.n_y, 1), np.int32)       # State sequence
        nu = P_Y[0] * self.P_S0
        for t in range(1, self.n_y):
            cost = (self.P_SS.T*nu).T*P_Y[t] # outer(nu,P_Y[t])*P_SS
            pred[t] = cost.argmax(axis=0)    # Best predecessor
            nu = np.choose(pred[t],cost)     # Cost of best paths to each state
            nu /= nu.max()                   # Prevent underflow
        last_s = np.argmax(nu)
        for t in range(self.n_y-1, -1, -1):
            ss[t] = last_s
            last_s = pred[t,last_s]
        return ss.flat # End of viterbi
    # End of decode()
    def class_decode(
        self,  # HMM instance
        y      # Observations
        ):
        '''
        Calculate maximum a posterori probability (MAP) classification
        sequence.

        Parameters
        ----------
        y : array_like
            Sequence of observations

        Returns
        -------
        cs : array
            Maximum likelihood classification sequence

        >>> from scipy.linalg import circulant
        >>> np.set_printoptions(precision=3, suppress=True)
        >>> c2s = {
        ...     0:[0,1],
        ...     1:[2,3],
        ...     2:[4,5],
        ...     }
        >>> P_S0 = np.ones(6)/6.0
        >>> P_SS = circulant([0,  0, 0, 0, .5, .5])
        >>> P_YS = circulant([.4, 0, 0, 0, .3, .3])
        >>> pars = (Discrete_Observations, P_YS, c2s)
        >>> mod = HMM(P_S0, P_S0, pars, P_SS, Class_y, make_prob)
        >>> S,YC = mod.simulate(1000)
        >>> YC = np.array(YC, np.int32)
        >>> p_s = 0.7*P_SS + 0.3/6
        >>> p_y = 0.7*P_YS + 0.3/6
        >>> pars = (Discrete_Observations, p_y, c2s)
        >>> mod = HMM(P_S0, P_S0, pars, p_s, Class_y, make_prob)
        >>> L = mod.train(YC, n_iter=20, display=False)

        Maximum likelihood estimation (training) yeilds a model that
        is similar to the model used to make the data.

        >>> print(mod)
        <class '__main__.HMM'> with 6 states
        P_S0         = [ 0.  1.  0.  0.  0.  0.]
        P_S0_ergodic = [ 0.184  0.161  0.151  0.177  0.174  0.153]
        P_SS =
        [[ 0.     0.485  0.515  0.     0.     0.   ]
         [ 0.     0.     0.351  0.649  0.     0.   ]
         [ 0.     0.     0.     0.484  0.516  0.   ]
         [ 0.     0.     0.     0.     0.543  0.457]
         [ 0.586  0.     0.     0.     0.     0.414]
         [ 0.539  0.461  0.     0.     0.     0.   ]]
        <class 'Scalar.Class_y'> with c2s =
        [[1 1 0 0 0 0]
         [0 0 1 1 0 0]
         [0 0 0 0 1 1]]
        P_YS =
        [[ 0.439  0.281  0.28   0.     0.     0.   ]
         [ 0.     0.362  0.314  0.324  0.     0.   ]
         [ 0.     0.     0.418  0.305  0.277  0.   ]
         [ 0.     0.     0.     0.384  0.283  0.333]
         [ 0.282  0.     0.     0.     0.391  0.328]
         [ 0.301  0.366  0.     0.     0.     0.333]
        
        Here are some of the log liklihoods per observation from the
        sequence of training iterations.  Note that they increase
        monotonically and that at the end of the sequence the change
        per iteration is less that a part in a thousand.

        >>> for i in [0, 1, 2, len(L)-2, len(L)-1]:
        ...     print('%2d: %6.3f'%(i, L[i]))
         0: -1.972
         1: -1.682
         2: -1.657
        18: -1.641
        19: -1.641

        Next, we drop the simulated class data from YC and demonstrate
        Viterbi decoding of the class sequence.  We designed the model
        so that decoding would be good rather than perfect, but there
        are no errors in this short sequence.  In the sequence below
        we've printed the observation y[i], the simulated class c[i]
        and the decoded class d[i], ie,

        y[i]  c[i]  d[i]

        >>> D = mod.class_decode(YC[:5,0])
        >>> for (yc, d) in zip(YC, D):
        ...     print('%3d, %3d, %3d'%(yc[0], yc[1], d))
          2,   0,   0
          3,   1,   1
          0,   2,   2
          2,   0,   0
          1,   0,   0

        '''
        c2s = self.y_mod.c2s
        n_c = len(c2s)
        n_t = len(y)
        s1 = np.arange(self.n_states, dtype=np.int32) #Index for cs_cost -> phi
        c1 = self.y_mod.s2c[s1] # Index for cs_cost -> phi
        P_Y = self.y_mod.y_mod.calc(y) # P_Y[t,s] = prob(Y=y[t]|state=s)

        # Do partial first iteration before loop
        pred = np.empty((n_t,n_c),np.int32)
        phi = self.P_S0_ergodic * P_Y[0]
        nu = np.dot(c2s,phi)
        phi /= np.maximum(1.0e-300, np.dot(np.dot(c2s, phi), c2s))

        for t in range(1,n_t): # Main loop
            # cost for state-state pairs = outer((nu*c2s.*phi),P_Y[t]).*P_SS
            ss_cost = (self.P_SS.T*np.dot(nu,c2s)*phi).T*P_Y[t]
            cs_cost = np.dot(c2s, ss_cost)    # Cost for class-state pairs
            cc_cost = np.dot(cs_cost, c2s.T)  # Cost for class-class pairs
            pred[t] = cc_cost.argmax(axis=0)  # Best predecessor for each class
            nu = np.choose(pred[t], cc_cost)  # Class cost given best history
            nu /= nu.max()
            phi = cs_cost[pred[t,c1],s1] # phi[s] is prob(state=s|best
                                         # class sequence ending in c[s])
            phi /= np.maximum(1.0e-300, np.dot(np.dot(c2s, phi), c2s))
        # Backtrack
        seq = np.empty((n_t,),np.int32)
        last_c = np.argmax(nu)
        for t in range(n_t-1,-1,-1):
            seq[t] = last_c
            last_c = pred[t, last_c]
        return seq

    def simulate(self, length, seed=3):
        """generates a random sequence of observations of given length

        Parameters
        ----------

        length : int
            Number of time steps to simulate
        seed : int, optional
            Seed for random number generator

        Returns
        -------

        states : list
            Sequence of states
        outs : list
            Sequence of observations
        """
        import random
        random.seed(seed)
        # Initialize lists
        outs = []
        states = []
        # Set up cumulative distributions
        cum_init = np.cumsum(self.P_S0_ergodic[0])
        cum_tran = np.cumsum(self.P_SS.values(), axis=1)
        # cum_rand generates random integers from a cumulative distribution
        cum_rand = lambda cum: np.searchsorted(cum, random.random())
        # Select initial state
        i = cum_rand(cum_init)
        # Select subsequent states and call model to generate observations
        for t in range(length):
            states.append(i)
            outs.append(self.y_mod.random_out(i))
            i = cum_rand(cum_tran[i])
        return (states, outs) # End of simulate()
    def link(self, from_, to_, p):
        """ Create (or remove) a link between state "from_" and state "to_".

        The strength of the link is a function of both the argument
        "p" and the existing P_SS array.  Set P_SS itself if you
        need to set exact values.  Use this method to modify topology
        before training.

        Parameters
        ----------
        from_ : int
        to_ : int
        p : float

        Returns
        -------
        None

        FixMe: No test coverage.  Broken for cython code
        """
        self.P_SS[from_,to_] = p
        self.P_SS[from_, :] /= self.P_SS[from_, :].sum()
    def __str__(self):
        save = np.get_printoptions
        np.set_printoptions(precision=3)
        rv = '''
%s with %d states
P_S0         = %s
P_S0_ergodic = %s
P_SS =
%s
%s''' % (self.__class__, self.n_states, self.P_S0, self.P_S0_ergodic,
         self.P_SS.values(), self.y_mod
         )
        np.set_printoptions(save)
        return rv[1:-1]
    def join_ys(
            self,    # HMM instance
            ys       # List of observation sequences
        ):
        """Concatenate and return multiple y sequences.  Also return
        information on sequence boundaries within concatenated list.

        Parameters
        ----------
        ys : list
            A list of observation sequences.  Default int, but must match
            method self.P_Y() if subclassed

        Returns
        -------
        n_seg : int
            Number of component segments
        t_seg : list
            List of ints specifying endpoints of segments within y_all
        y_all : list
            Concatenated list of observations

        """
        t_seg = [0] # List of segment boundaries in concatenated ys
        y_all = []
        for seg in ys:
            y_all += list(seg)
            t_seg.append(len(y_all))
        self.n_y = t_seg[-1]
        return len(t_seg)-1, t_seg, y_all
    def multi_train(
            self,         # HMM instance
            ys,           # List of observation sequences
            n_iter=1,
            boost_w=None, # Optional weight of each observation for reestimation
            display=True
        ):
        '''Train on multiple sequences of observations

        Parameters
        ----------
        ys : list
            list of sequences of integer observations
        n_iter : int, optional
            Number of iterations
        boost_w : array, optional
            Weight of each observation for reestimation
        display : bool, optional
            If True, print the log likelihood per observation for each
            segment and each iteration

        Returns
        -------
        avgs : list
            List of log likelihood per observation for each iteration

        Examples
        --------
        Same model as used to demonstrate train().  Here simulated
        data is broken into three independent segments for training.
        For each iteration, "L[i]" gives the log likelihood per data
        point of segment "i".  Note that L[2] does not improve
        monotonically but that the average over segments does.

        >>> P_S0 = np.array([1./3., 1./3., 1./3.])
        >>> P_S0_ergodic = np.array([1./7., 4./7., 2./7.])
        >>> P_SS = np.array([
        ...         [0,   1,   0],
        ...         [0,  .5,  .5],
        ...         [.5, .5,   0]
        ...         ],np.float64)
        >>> P_YS = np.array([
        ...         [1, 0,     0],
        ...         [0, 1./3., 2./3.],
        ...         [0, 2./3., 1./3.]
        ...         ])
        >>> mod = HMM(P_S0,P_S0_ergodic,P_YS,P_SS)
        >>> S,Y = mod.simulate(600)
        >>> ys = []
        >>> for i in [1,2,0]:
        ...     ys.append(list(Y[200*i:200*(i+1)]))
        >>> A = mod.multi_train(ys,3)
        i=0: L[0]=-0.9162 L[1]=-0.9142 L[2]=-0.9275 avg=-0.9193117
        i=1: L[0]=-0.9156 L[1]=-0.9137 L[2]=-0.9278 avg=-0.9190510
        i=2: L[0]=-0.9153 L[1]=-0.9135 L[2]=-0.9280 avg=-0.9189398

        '''
        n_seg, t_seg, y_all = self.join_ys(ys)
        avgs = n_iter*[None] # Average log likelihood per step
        t_total = t_seg[-1]
        alpha_all = initialize(self.alpha, (t_total, self.n_states))
        beta_all = initialize(self.beta, (t_total, self.n_states))
        gamma_all = initialize(self.gamma, (t_total,))
        P_S0_all = np.empty((n_seg, self.n_states))
        #P_S0_all are state probabilities at the beginning of each segment
        for seg in range(n_seg):
            P_S0_all[seg, :] = self.P_S0.copy()
        for i in range(n_iter):
            if display:
                print('i=%d: '%i, end='')
            tot = 0.0
            # Both forward() and backward() should operate on each
            # training segment and put the results in the
            # corresponding segement of the the alpha, beta and gamma
            # arrays.
            P_Y_all = self.y_mod.calc(y_all)
            for seg in range(n_seg):
                self.n_y = t_seg[seg+1] - t_seg[seg]
                self.alpha = alpha_all[t_seg[seg]:t_seg[seg+1], :]
                self.beta = beta_all[t_seg[seg]:t_seg[seg+1], :]
                self.P_Y = P_Y_all[t_seg[seg]:t_seg[seg+1]]
                self.gamma = gamma_all[t_seg[seg]:t_seg[seg+1]]
                self.P_S0 = P_S0_all[seg, :]
                LL = self.forward() #Log Likelihood
                if display:
                    print('L[%d]=%7.4f '%(seg,LL/self.n_y), end='')
                tot += LL
                self.backward()
                self.P_S0 = self.alpha[0] * self.beta[0]
                self.gamma[0] = -1 # Don't fit transitions between segments
            avgs[i] = tot/t_total
            if i>0 and avgs[i-1] >= avgs[i]:
                print('''
WARNING training is not monotonic: avg[%d]=%f and avg[%d]=%f
'''%(i-1,avgs[i-1],i,avgs[i]))
            if display:
                print('avg=%10.7f'% avgs[i])
            # Associate all of the alpha and beta segments with the
            # states and reestimate()
            self.alpha = alpha_all
            self.beta = beta_all
            self.gamma = gamma_all
            self.P_Y = P_Y_all
            if boost_w != None:
                self.alpha *= BoostW
            self.n_y = len(y_all)
            self.reestimate(y_all)
        self.P_S0[:] = P_S0_all.sum(axis=0)
        self.P_S0 /= self.P_S0.sum()
        return avgs

def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
#--------------------------------
# Local Variables:
# mode: python
# End:
