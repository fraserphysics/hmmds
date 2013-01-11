'''C.pyx: cython code for speed up.  The HMM class is 17.9 times
faster than the pure python in base.py and Scalar.py, and the
HMM_SPARSE class is 16.4 times faster when the matrices have no zeros.

'''
#To build: python3 setup.py build_ext --inplace
from hmm import Scalar
from hmm import base
import numpy as np
import scipy.sparse as SS
#warnings.simplefilter('ignore',SS.SparseEfficiencyWarning)
# Imitate http://docs.cython.org/src/tutorial/numpy.html
# http://docs.cython.org/src/userguide/memoryviews.html
cimport cython, numpy as np
DTYPE = np.float64
ITYPE = np.int32
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t ITYPE_t
class HMM(base.HMM):
    '''A Cython subclass of HMM that implments methods forward, backward
    and reestimate-s for speed'''

    @cython.boundscheck(False)
    def forward(self):
        # Ensure allocation and size of alpha and gamma
        self.alpha = Scalar.initialize(self.alpha,(self.n_y,self.n_states))
        self.gamma = Scalar.initialize(self.gamma,(self.n_y,))

        # Make views of numpy arrays
        cdef DTYPE_t [:] gamma = self.gamma
        cdef DTYPE_t [:,:] alpha = self.alpha
        cdef DTYPE_t [:,:] P_SS = self.P_SS
        cdef DTYPE_t [:,:] P_Y = self.P_Y

        # Make double buffer for calculations
        cdef double *_next, *_last
        scratch = np.empty((2,self.n_states))
        scratch[0,:] = self.P_S0
        cdef DTYPE_t [:, :] next_last = scratch

        cdef int t, i, j
        cdef int N = self.n_states
        cdef int T = self.n_y

        # iterate
        for t in range(T):
            _last = &next_last[t%2,0]
            _next = &next_last[(t+1)%2,0]
            for i in range(N):
                _last[i] = _last[i]*P_Y[t,i]
            gamma[t] = 0
            for i in range(N):
                gamma[t] += _last[i]
            for i in range(N):
                _last[i] /= gamma[t]
                alpha[t,i] = _last[i]
            for i in range(N):
                _next[i] = 0
                for j in range(N):
                    _next[i] += _last[j] * P_SS[j,i]
        return (np.log(self.gamma)).sum() # End of forward()
    @cython.boundscheck(False)
    def backward(self):
        # Ensure allocation and size of beta
        self.beta = Scalar.initialize(self.beta,(self.n_y,self.n_states))

        # Make views of numpy arrays
        cdef DTYPE_t [:] gamma = self.gamma
        cdef DTYPE_t [:,:] beta = self.beta
        cdef DTYPE_t [:,:] P_SS = self.P_SS
        cdef DTYPE_t [:,:] P_Y = self.P_Y

        # Make double buffer for calculations
        cdef double *_next, *_last
        scratch = np.ones((2,self.n_states))
        scratch[0,:] = self.P_S0
        cdef DTYPE_t [:, :] next_last = scratch

        cdef int t,i,j
        cdef int N = self.n_states
        cdef int T = self.n_y

        # iterate
        for t in range(T-1,-1,-1):
            _last = &next_last[t%2,0]
            _next = &next_last[(t+1)%2,0]
            for i in range(N):
                beta[t,i] = _last[i]
                _last[i] *= P_Y[t,i]/gamma[t]
            for i in range(N):
                _next[i] = 0
                for j in range(N):
                    _next[i] += P_SS[i,j] * _last[j]
        return # End of backward()
    @cython.boundscheck(False)
    def reestimate(self, y):
        """Reestimate state transition probabilities and initial
        state probabilities.

        Given the observation probabilities, ie, self.state[s].P_Y[t],
        given alpha, beta, gamma, and Py, these calcuations are
        independent of the observation model calculations.

        Parameters
        ----------
        y : sequence

        Returns
        -------
        alpha*beta : array
            State probabilities given all observations

        """

        cdef np.ndarray[DTYPE_t, ndim=1] wsum = np.zeros(
            self.n_states, np.float64)
        cdef np.ndarray[DTYPE_t, ndim=2] usum = np.zeros(
            (self.n_states,self.n_states),np.float64)

        # Make views of numpy arrays
        cdef DTYPE_t [:] gamma = self.gamma
        cdef DTYPE_t [:,:] alpha = self.alpha
        cdef DTYPE_t [:,:] beta = self.beta
        cdef DTYPE_t [:,:] P_Y = self.P_Y
        cdef DTYPE_t [:] _wsum = wsum
        cdef DTYPE_t [:,:] _usum = usum

        cdef int t,i,j
        cdef int N = self.n_states
        cdef int T = self.n_y

        for t in range(T-1):
            if gamma[t] == 0:
                continue       # Skip over segment boundaries
            for i in range(N):
                for j in range(N):
                    _usum[i,j] += alpha[t,i]*beta[t+1,j]*P_Y[t+1,j]/gamma[t+1]
                alpha[t,i] *= beta[t,i]
                _wsum[i] += alpha[t,i]
        #Alpha[T-1,:] *= Beta[T-1,:] but Beta[T-1,:] = 1
        wsum += self.alpha[T-1]
        self.P_S0_ergodic = np.copy(wsum)
        self.P_S0 = np.copy(self.alpha[0])
        for x in (self.P_S0_ergodic, self.P_S0):
            x /= x.sum()
        self.P_SS.inplace_elementwise_multiply(usum)
        self.P_SS.normalize()
        self.y_mod.reestimate(self.alpha, y)
        return # End of reestimate()

class Prob:
    '''Replacement for Scalar.Prob that stores data in sparse matrix
    format.  P[a,b] is the probability of b given a.
    
    For pruning.  Drop x[i,j] if x[i,j] < threshold*max(A[i,:]) and
    x[i,j] < threshold*max(A[:,j]) I don't understand sensitivity of
    training to pruning threshold.

    '''
    def __init__(self, x, threshold=-1):
        self.threshold = threshold
        if SS.isspmatrix_csc(x):
            self.csc = x
        else:
            self.csc = SS.csc_matrix(x)
        self.data = self.csc.data
        self.indptr = self.csc.indptr
        self.indices= self.csc.indices
        self.data = self.csc.data
        self.shape = x.shape
        N,M = x.shape
        self.tcol = np.empty(N)  # Scratch space step_back
        self.trow = np.empty(M)  # Scratch space step_forward
        self.normalize()
    def values(self):
        ''' Return dense version of matrix
        '''
        return self.csc.todense()
    def assign_col(self,i,col):
        '''Implements self[:,i]=col.  Very slow because finding each csc[j,i]
        is slow.  However, this is not used in a loop over T.
        '''
        N,M = self.shape
        for j in range(N):
            self.csc[j,i] = col[j]
    def likelihoods(self,v):
        '''Returns L with L[t,j]=self[j,v[t]], ie, the state likelihoods for
        observations v.
        '''
        N,M = self.shape
        T = len(v)
        L = np.zeros((T,N))
        for t in range(T):
            i = v[t]
            for j in range(self.indptr[i],self.indptr[i+1]):
                J = self.indices[j]
                L[t,J] = self.data[j]
        return L
    def inplace_elementwise_multiply(self,A):
        N,M = self.shape
        for i in range(M):
            for j in range(self.indptr[i],self.indptr[i+1]):
                J = self.indices[j]
                self.data[j] *= A[J,i]
    def cost(self, nu, py):
        ''' Efficient calculation of np.outer(nu, py)*self (* is
        element-wise).  Used in Viterbi decoding.
        '''
        n, m = self.shape
        r = np.zeros(self.shape)
        for i in range(m):
            for j in range(self.indptr[i],self.indptr[i+1]):
                J = self.indices[j]
                r[J,i] = self.data[j] * nu[J] * py[i]
        return r
    def normalize(self):
        '''Divide each row, self[j,:], by its sum.  Then prune based on
        threshold.

        '''
        N,M = self.shape
        max_row = np.zeros(M) # Row of maxima in each column
        max_col = np.zeros(N)
        sum_col = np.zeros(N) # Column of row sums
        for i in range(M):    # Add up the rows
            for j in range(self.indptr[i],self.indptr[i+1]):
                J = self.indices[j]
                sum_col[J] += self.data[j]
        for i in range(M):    # Normalize the rows
            for j in range(self.indptr[i],self.indptr[i+1]):
                J = self.indices[j]
                self.data[j] /= sum_col[J]
        if self.threshold < 0:
            return
        for i in range(M):    # Find max of the rows and columns
            for j in range(self.indptr[i],self.indptr[i+1]):
                J = self.indices[j]
                x = self.data[j]
                if x > max_row[i]:
                    max_row[i] = x
                if x > max_col[J]:
                    max_col[J] = x
        max_row *= self.threshold
        max_col *= self.threshold
        k = self.indptr[0]
        L = self.indptr[0]
        for i in range(M):
            for j in range(L,self.indptr[i+1]):
                J = self.indices[j]
                x = self.data[j]
                if (x > max_row[i] or x > max_col[J]):
                    self.indices[k] = J
                    self.data[k] = x
                    k += 1
                else:
                    print('Prune')
            L = self.indptr[i+1]
            self.indptr[i+1] = k
    def step_back(self,A):
        ''' Implements A[:] = self*A
        '''
        cdef np.ndarray[DTYPE_t, ndim=1] _A = A
        cdef double *a = <double *>_A.data
        cdef np.ndarray[DTYPE_t, ndim=1] data = self.data
        cdef double *_data = <double *>data.data
        cdef np.ndarray[ITYPE_t, ndim=1] indices = self.indices
        cdef int *_indices = <int *>indices.data
        cdef np.ndarray[ITYPE_t, ndim=1] indptr = self.indptr
        cdef int *_indptr = <int *>indptr.data
        cdef np.ndarray[DTYPE_t, ndim=1] tdata = self.tcol
        cdef double *t = <double *>tdata.data
        cdef int N = self.shape[0]
        cdef int M = self.shape[1]
        cdef int i,j,J
        for j in range(M):
            t[j] = 0
        for i in range(N):
            for j in range(_indptr[i],_indptr[i+1]):
                J = _indices[j]
                t[J] += _data[j]*a[i]
        tdata.data = <char *>a
        _A.data = <char *>t
    def step_forward(self,A):
        ''' Implements A[:] = A*self
        '''
        cdef np.ndarray[DTYPE_t, ndim=1] _A = A
        cdef double *a = <double *>_A.data
        cdef np.ndarray[DTYPE_t, ndim=1] data = self.data
        cdef double *_data = <double *>data.data
        cdef np.ndarray[ITYPE_t, ndim=1] indices = self.indices
        cdef int *_indices = <int *>indices.data
        cdef np.ndarray[ITYPE_t, ndim=1] indptr = self.indptr
        cdef int *_indptr = <int *>indptr.data
        cdef np.ndarray[DTYPE_t, ndim=1] tdata = self.trow
        cdef double *t = <double *>tdata.data
        cdef int N = self.shape[0]
        cdef int M = self.shape[1]
        cdef int i,j,J
        for i in range(N):
            t[i] = 0
            for j in range(_indptr[i],_indptr[i+1]):
                J = _indices[j]
                t[i] += _data[j]*a[J]
        tdata.data = <char *>a
        _A.data = <char *>t
        
def make_prob(x):
    return Prob(x)
class Discrete_Observations(Scalar.Discrete_Observations):
    '''The simplest observation model: A finite set of integers.
    Implemented with scipy sparse matrices.

    Parameters
    ----------
    P_YS : array_like
        Conditional probabilites P_YS[s,y]

    '''
    def __init__(self,P_YS):
        self.P_YS = make_prob(P_YS)
        self.P_Y = None
        return
    def __str__(self):
        return 'P_YS =\n%s'%(self.P_YS.todense(),)
    def random_out(self,s):
        ''' For simulation, draw a random observation given state s
        '''
        raise RuntimeError('Simulation not implemented for %s'%self.__class__)
    @cython.boundscheck(False)
    def calc(
        self,    # Discrete_Observations instance
        y_       # A list with a sequence of integer observations
        ):
        """
        Allocate self.P_Y and assign values self.P_Y[t,i] = P(y(t)|s(t)=i)

        Parameters
        ----------
        y : array
            A sequence of integer observations

        Returns
        -------
        P_Y : array
            Array of likelihoods of states.  P_Y.shape = (n_y, n_states)
        """
        # Check size and initialize self.P_Y
        y = y_[0]
        n_y = len(y)
        n_states = self.P_YS.shape[0]
        self.P_Y = Scalar.initialize(self.P_Y, (n_y, n_states))
        YcS = self.P_YS

        cdef np.ndarray[DTYPE_t, ndim=2] Py = self.P_Y
        cdef int pystride = Py.strides[0]
        cdef char *_py = Py.data
        cdef double *py

        cdef np.ndarray[ITYPE_t, ndim=1] Y = y
        cdef int *_y = <int *>Y.data

        cdef np.ndarray[DTYPE_t, ndim=1] data = YcS.data
        cdef double *_data = <double *>data.data
        cdef np.ndarray[ITYPE_t, ndim=1] indices = YcS.indices
        cdef int *_indices = <int *>indices.data
        cdef np.ndarray[ITYPE_t, ndim=1] indptr = YcS.indptr
        cdef int *_indptr = <int *>indptr.data

        cdef int T = n_y
        cdef int t,i,j,J
        for t in range(T):
            py = <double *>(_py+t*pystride)
            i = _y[t]
            for j in range(_indptr[i],_indptr[i+1]):
                J = _indices[j]
                py[J] = _data[j]
        return self.P_Y # End of p_y_calc()
    def reestimate(self,w,y_):
        """
        Estimate new model parameters.  Differs from version in Scalar
        by not updating self.cum_y which simulation requires.

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
        y = y_[0]
        n_y = len(y)
        if not type(y) == np.ndarray:
            y = np.array(y, np.int32)
        assert(y.dtype == np.int32 and y.shape == (n_y,))
        for yi in range(self.P_YS.shape[1]):
            self.P_YS.assign_col(
                yi, w.take(np.where(y==yi)[0], axis=0).sum(axis=0))
        self.P_YS.normalize()
        return
class HMM_SPARSE(HMM):
    '''HMM code that uses sparse matrices for state to state and state to
    observation probabilities.  API matches base.HMM

    '''
    def __init__(self, P_S0, P_S0_ergodic, P_YS, P_SS,
                 y_mod=Discrete_Observations, prob=make_prob):
        base.HMM.__init__(self, P_S0, P_S0_ergodic, P_YS, P_SS, y_mod, prob)

    @cython.boundscheck(False)
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

        * self.P_Y    has been calculated

        * self.n_y     is length of Y

        * self.n_states     is number of states

        Bullet points
        -------------

        On return:

        * self.gamma[t] = Pr{y(t)=y(t)|y_0^{t-1}}
        * self.alpha[t,i] = Pr{s(t)=i|y_0^t}
        * return value is log likelihood of all data

        """
        # Ensure allocation and size of alpha and gamma
        self.alpha = Scalar.initialize(self.alpha,(self.n_y,self.n_states))
        self.gamma = Scalar.initialize(self.gamma,(self.n_y,))

        # Setup direct access to numpy arrays
        cdef np.ndarray[DTYPE_t, ndim=1] gamma = self.gamma
        cdef double *_gamma = <double *>gamma.data

        cdef np.ndarray[DTYPE_t, ndim=1] last = np.copy(self.P_S0.reshape(-1))
        cdef double *_last = <double *>last.data

        cdef np.ndarray[DTYPE_t, ndim=2] Alpha = self.alpha
        cdef char *_alpha = Alpha.data
        cdef int astride = Alpha.strides[0]
        cdef double *a

        cdef np.ndarray[DTYPE_t, ndim=2] Py = self.P_Y
        cdef int pystride = Py.strides[0]
        cdef char *_py = Py.data
        cdef double *py

        PSCS = self.P_SS
        cdef np.ndarray[DTYPE_t, ndim=1] data = PSCS.data
        cdef double *_data = <double *>data.data
        cdef np.ndarray[ITYPE_t, ndim=1] indices = PSCS.indices
        cdef int *_indices = <int *>indices.data
        cdef np.ndarray[ITYPE_t, ndim=1] indptr = PSCS.indptr
        cdef int *_indptr = <int *>indptr.data
        cdef np.ndarray[DTYPE_t, ndim=1] tdata = PSCS.trow
        cdef double *_next = <double *>tdata.data

        cdef int t,i,j
        cdef int N = self.n_states
        cdef int T = self.n_y
        cdef double *_tmp
        cdef double total
        # iterate
        for t in range(T):
            py = <double *>(_py+t*pystride)
            for i in range(N):
                _last[i] = _last[i]*py[i]
            total = 0
            for i in range(N):
                total += _last[i]
            _gamma[t] = total
            a = <double *>(_alpha+t*astride)
            for i in range(N):
                _last[i] /= total
                a[i] = _last[i]

            for i in range(N):  # Block for next = last*self
                _next[i] = 0
                for j in range(_indptr[i],_indptr[i+1]):
                    J = _indices[j]
                    _next[i] += _data[j]*_last[J]

            _tmp = _last
            _last = _next
            _next = _tmp
        return (np.log(self.gamma)).sum() # End of forward()

    @cython.boundscheck(False)
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

        * self.P_Y    has been calculated

        Bullet points
        -------------

        On return:

        * for each state i, beta[t,i] = Pr{y_{t+1}^T|s(t)=i}/Pr{y_{t+1}^T}

        """
        # Ensure allocation and size of beta
        self.beta = Scalar.initialize(self.beta,(self.n_y,self.n_states))

        # Setup direct access to numpy arrays
        cdef np.ndarray[DTYPE_t, ndim=1] gamma = self.gamma
        cdef double *_gamma = <double *>gamma.data

        cdef np.ndarray[DTYPE_t, ndim=1] last = np.ones(self.n_states)
        cdef double *_last = <double *>last.data

        cdef np.ndarray[DTYPE_t, ndim=2] Beta = self.beta
        cdef char *_beta = Beta.data
        cdef int bstride = Beta.strides[0]
        cdef double *b

        cdef np.ndarray[DTYPE_t, ndim=2] Py = self.P_Y
        cdef int pystride = Py.strides[0]
        cdef char *_py = Py.data
        cdef double *py

        PSCS = self.P_SS
        cdef np.ndarray[DTYPE_t, ndim=1] data = PSCS.data
        cdef double *_data = <double *>data.data
        cdef np.ndarray[ITYPE_t, ndim=1] indices = PSCS.indices
        cdef int *_indices = <int *>indices.data
        cdef np.ndarray[ITYPE_t, ndim=1] indptr = PSCS.indptr
        cdef int *_indptr = <int *>indptr.data
        cdef np.ndarray[DTYPE_t, ndim=1] tdata = PSCS.tcol
        cdef double *_next = <double *>tdata.data

        # iterate
        cdef int t,i,j,J
        cdef int N = self.n_states
        cdef int T = self.n_y
        cdef double *_tmp
        cdef double total
        for t in range(T-1,-1,-1):
            py = <double *>(_py+t*pystride)
            b = <double *>(_beta+t*bstride)
            for i in range(N):
                b[i] = _last[i]
                _last[i] *= py[i]/_gamma[t]

            for j in range(N):
                _next[j] = 0
            for i in range(N):
                for j in range(_indptr[i],_indptr[i+1]):
                    J = _indices[j]
                    _next[J] += _data[j]*_last[i]

            _tmp = _last
            _last = _next
            _next = _tmp
        return # End of backward()

#--------------------------------
# Local Variables:
# mode: python
# End:
