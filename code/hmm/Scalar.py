""" Scalar.py: Implements scalar observation models.

"""
# Copyright (c) 2003, 2007, 2008, 2012 Andrew M. Fraser
import random
import numpy as np

# cum_rand generates random integers from a cumulative distribution
cum_rand = lambda cum: np.searchsorted(cum, random.random())

def initialize(x, shape, dtype=np.float64):
    if x == None or x.shape != shape:
        return np.empty(shape, dtype)
    return x
## ----------------------------------------------------------------------
class Prob(np.ndarray):
    '''Subclass of ndarray for probability matrices.  P[a,b] is the
    probability of b given a.  The class has additional methods and is
    designed to enable alternative implementations with speed
    improvements implemented by uglier code.

    '''
    # See http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    def assign_col(self, i, col):
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
        self[:, i] = col
    def likelihoods(self, v):
        '''likelihoods for vector of data

        Parameters
        ----------
        v : array_like
            Time series of observations

        Returns
        -------
        L : array

            Given T = len(v) and self.shape = (M,N), L.shape = (T,M)
            with the interpretation L[t,a] = Prob(v[t]|a)

        '''
        return self[:, v].T
    def inplace_elementwise_multiply(self, a):
        '''
        Multiply self by argument

        Parameters
        ----------
        a : array

        Returns
        -------
        None
        '''
        self *= a
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
        s = self.sum(axis=1)
        for i in range(self.shape[0]):
            self[i, :] /= s[i]
    def step_forward(self, a):
        '''
        Replace values of argument a with matrix product a*self

        Parameters
        ----------
        a : array

        Returns
        -------
        None
        '''
        a[:] = np.dot(a, self)
    def step_back(self, a):
        '''
        Replace values of argument a with matrix product self*a

        Parameters
        ----------
        a : array

        Returns
        -------
        None
        '''
        a[:] = np.dot(self, a)
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
        v : array
        '''
        return self
def make_prob(x):
    x = np.array(x)
    return Prob(x.shape, buffer=x.data)

class Discrete_Observations:
    '''The simplest observation model: A finite set of integers.

    More complex observation models should provide the methods:
    *calc*, *random_out*, *reestimate* necessary for the application.

    Parameters
    ----------
    P_YS : array_like
        Conditional probabilites P_YS[s,y]

    '''
    def __init__(self,P_YS):
        self.P_YS = make_prob(P_YS)
        self.cum_y = np.cumsum(self.P_YS, axis=1)
        self.P_Y = None
        return
    def __str__(self):
        return 'P_YS =\n%s'%(self.P_YS,)
    def random_out(self,s):
        ''' For simulation, draw a random observation given state s

        Parameters
        ----------
        s : int
            Index of state

        Returns
        -------
        y : int
            Random observation drawn from distribution conditioned on state s
        '''
        return cum_rand(self.cum_y[s])
    def calc(self,y):
        """
        Calculate and return likelihoods: self.P_Y[t,i] = P(y(t)|s(t)=i)

        Parameters
        ----------
        y : array
            A sequence of integer observations

        Returns
        -------
        P_Y : array, floats
        """
        n_y = len(y)
        n_states = len(self.P_YS)
        self.P_Y = initialize(self.P_Y, (n_y, n_states))
        self.P_Y[:, :] = self.P_YS.likelihoods(y)
        return self.P_Y
    def reestimate(self, # Discrete_Observations instance
                   w, y):
        """
        Estimate new model parameters

        Parameters
        ----------
        w : array
            w[t,s] = Prob(state[t]=s) given data and old model
        y : array
            A sequence of integer observations

        Returns
        -------
        None
        """
        n_y = len(y)
        if not type(y) == np.ndarray:
            y = np.array(y, np.int32)
        assert(y.dtype == np.int32 and y.shape == (n_y,))
        for yi in range(self.P_YS.shape[1]):
            self.P_YS.assign_col(
                yi, w.take(np.where(y==yi)[0], axis=0).sum(axis=0))
        self.P_YS.normalize()
        self.cum_y = np.cumsum(self.P_YS, axis=1)
        return

class Class_y(Discrete_Observations):
    '''Observation model with classification
    
    Parameters
    ----------
    pars : (P_YS,c2s)
    P_YS : array
        P_YS[s,y] is the probability of state s producing output y
    c2s : dict
        classification labels are keys and each value is a list states
        contained in the classification.

    '''
    def __init__(self, # Class_y instance
                 pars):
        y_class, y_pars, c2s = pars
        self.y_mod = y_class(y_pars)
        self.P_Y = None
        self.g = None
        states = {}  # Count states
        for c in c2s:
            for s in c2s[c]:
                states[s] = True
        n_states = len(states)
        n_class = len(c2s)
        self.c2s = np.zeros((n_class, n_states), np.bool)
        self.s2c = np.empty(n_states, dtype=np.int32)
        for c in c2s:
            for s in c2s[c]:
                self.s2c[s] = c
                self.c2s[c,s] = True
        return
    def __str__(self):
        return('%s with c2s =\n%s\n%s'%(
                self.__class__, self.c2s.astype(np.int8), self.y_mod))
    def random_out(self, # Class_y instance
                   s):
        return self.y_mod.random_out(s), self.s2c[s]
    def calc(self, # Class_y instance
             yc):
        """
        Calculate and return likelihoods: self.P_Y[t,i] =
        P(y(t)|s(t)=i)*g(s,c[t])

        Parameters
        ----------
        yc : array_like
            A sequence of y,c pairs.  y[t] = yc[t][0] and c[t] = yc[t][1]

        Returns
        -------
        P_Y : array, floats
        """
        y = yc[:,0]
        c = yc[:,1]
        n_y = len(y)
        n_class, n_states = self.c2s.shape
        self.g = initialize(self.g,(n_y, n_states),np.bool)
        self.g[:,:] = self.c2s[c,:]
        self.P_Y = self.y_mod.calc(y) * self.g
        return self.P_Y
    def reestimate(self,  # Class_y instance
                   w,yc):
        """
        Estimate new model parameters

        Parameters
        ----------
        w : array
            w[t,s] = Prob(state[t]=s) given data and old model
        yc : array
            A sequence of y,c pairs

        Returns
        -------
        None
        """
        self.y_mod.reestimate(w, yc[:,0])
        return

#--------------------------------
# Local Variables:
# mode: python
# End:
