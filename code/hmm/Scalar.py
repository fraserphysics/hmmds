""" Scalar.py: Implements scalar observation models.

"""
# Copyright (c) 2003, 2007, 2008, 2012 Andrew M. Fraser
import numpy as np

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
        '''Likelihoods for vector of data

        Given T = len(v) and self.shape = (M,N), return L with L.shape
        = (T,M) and L[t,a] = Prob(v[t]|a)

        Parameters
        ----------
        v : array_like
            Time series of observations

        Returns
        -------
        L : array

        '''
        return self[:, v].T
    def cost(self, nu, py):
        ''' Efficient calculation of np.outer(nu, py)*self (where * is
        element-wise)
        '''
        return (self.T*nu).T*py
    def inplace_elementwise_multiply(self, a):
        '''
        Replace self with product of self and argument

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
    '''Make a Prob instance.

    Used as an argument of HMM.__init__ so that one may use other
    functions that create instances of other classes instead of Prob,
    eg, one that uses sparse matrices.

    Parameters
    ----------
    x : array_like
        Conditional probabilites x[i,j] the probability of j given i

    Returns
    -------
    p : Prob instance
    '''
    x = np.array(x)
    return Prob(x.shape, buffer=x.data)

class Discrete_Observations:
    '''The simplest observation model: A finite set of integers.

    Parameters
    ----------
    P_YS : array_like
        Conditional probabilites P_YS[s,y]

    '''
    def __init__(self, P_YS):
        self.P_YS = make_prob(P_YS)
        self.cum_y = np.cumsum(self.P_YS, axis=1)
        self.P_Y = None
        self.dtype = [np.int32]
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
        import random
        return  (np.searchsorted(self.cum_y[s],random.random()),)
    def calc(self, y_):
        """
        Calculate and return likelihoods: self.P_Y[t,i] = P(y(t)|s(t)=i)

        Parameters
        ----------
        y_ : list
            Has one element which is a sequence of integer observations

        Returns
        -------
        P_Y : array, floats

        """
        y = y_[0]
        n_y = len(y)
        n_states = len(self.P_YS)
        self.P_Y = initialize(self.P_Y, (n_y, n_states))
        self.P_Y[:, :] = self.P_YS.likelihoods(y)
        return self.P_Y
    def join(self, # Discrete_Observations instance
             ys):
        """Concatenate and return multiple y sequences.

        Also return information on sequence boundaries within
        concatenated list.

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
        n_components = len(ys[0])
        for i in range(n_components):
            y_all.append([])
        for seg in ys:
            for i in range(n_components):
                y_all[i] += list(seg[i])
            t_seg.append(len(y_all[0]))
        for i in range(n_components):
            y_all[i] = np.array(y_all[i], self.dtype[i])
        return len(t_seg)-1, t_seg, y_all
    def reestimate(self,      # Discrete_Observations instance
                   w,         # Weights
                   y_,        # Observations
                   warn=True
                   ):
        """
        Estimate new model parameters

        Parameters
        ----------
        w : array
            w[t,s] = Prob(state[t]=s) given data and old model
        y : list
            y[0] is a sequence of integer observations
        warn : bool
            If True and y[0].dtype != np.int32, print warning

        Returns
        -------
        None
        """
        y = y_[0]
        n_y = len(y)
        if not (type(y) == np.ndarray and y.dtype == np.int32):
            y = np.array(y, np.int32)
            if warn:
                print('Warning: reformatted y in reestimate')
        assert(y.dtype == np.int32 and y.shape == (n_y,)),'''
                y.dtype=%s, y.shape=%s'''%(y.dtype, y.shape)
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
    pars : (y_class, theta, c2s)
    y_class : class
        y_class(theta) should yield a model instance for observations without
        class
    theta : object
        A python object that contains parameter[s] for the observation model
    c2s : dict
        classification labels are keys and each value is a list of states
        contained in the classification.

    '''
    def __init__(self, # Class_y instance
                 pars):
        y_class, y_pars, c2s = pars
        self.y_mod = y_class(y_pars)
        self.dtype = [np.int32, np.int32]
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
        '''Simulate.  Draw a random observation given state s.

        Parameters
        ----------
        s : int
            Index of state

        Returns
        -------
        y, c : int, int
            A tuple consisting of an observation and the class of the state
        '''
        return self.y_mod.random_out(s)[0], self.s2c[s]
    def calc(self, # Class_y instance
             yc):
        """
        Calculate and return likelihoods: P_Y[t,i] = P(y(t)|s(t)=i)*g(s,c[t])

        g(s,c) is a gate function that is one if state s is in class c
        and is zero otherwise.

        Parameters
        ----------
        yc : array_like
            A sequence of y,c pairs.  y[t] = yc[t][0] and c[t] = yc[t][1]

        Returns
        -------
        P_Y : array, floats

        """
        y, c = yc
        n_y = len(y)
        n_class, n_states = self.c2s.shape
        self.g = initialize(self.g,(n_y, n_states),np.bool)
        self.g[:,:] = self.c2s[c,:]
        self.P_Y = self.y_mod.calc((y,)) * self.g
        return self.P_Y
    def reestimate(self,  # Class_y instance
                   w, yc):
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
        self.y_mod.reestimate(w, (yc[0],))
        return

#--------------------------------
# Local Variables:
# mode: python
# End:
