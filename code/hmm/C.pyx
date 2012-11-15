'''C.pyx: cython code for speed up.  Only significant improvment comes
from the loop over time in reestimate_s()
'''
import Scalar, numpy as np
# Imitate http://docs.cython.org/src/tutorial/numpy.html
cimport cython, numpy as np
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
class PROB(np.ndarray):
    ''' Subclass of ndarray for probability matrices.  P[a,b] is the
    probability of b given a.  The class has additional methods and is
    designed to enable further subclasses with ugly speed
    optimization.  See
    http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    '''
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
          strides=None, order=None):
        obj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset, strides,
                         order)
        obj.out = np.zeros(shape[1])
        obj.outT = np.dot(obj,obj.out)
        return obj
    def assign_col(self,i,col):
        self[:,i] = col
    def likelihoods(self,v):
        '''likelihoods for vector of observations v
        '''
        return self[:,v].T
    def inplace_elementwise_multiply(self,A):
        self *= A
    def normalize(self):
        S = self.sum(axis=1)
        for i in range(self.shape[0]):
            self[i,:] /= S[i]
    def step_back(self,A):
        ''' need last = np.dot(self.P_ScS,np.ones(self.N))
        '''
        np.dot(self,A,self.outT)
        t = self.outT
        self.outT = A
        return t
    def step_forward(self,A):
        np.dot(A,self,self.out)
        t = self.out
        self.out = A
        return t
def make_prob(x):
    x = np.array(x)
    return PROB(x.shape,buffer=x.data)

class HMM(Scalar.HMM):
    @cython.boundscheck(False)
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

        # Ensure allocation and size of alpha and gamma
        self.alpha = Scalar.initialize(self.alpha,(self.T,self.N))
        self.gamma = Scalar.initialize(self.gamma,(self.T,))
        # Setup direct access to arrays
        cdef np.ndarray[DTYPE_t, ndim=1] gamma = self.gamma
        cdef np.ndarray[DTYPE_t, ndim=2] Py = self.Py
        cdef np.ndarray[DTYPE_t, ndim=2] alpha = self.alpha
        cdef np.ndarray[DTYPE_t, ndim=1] last = np.copy(self.P_S0.reshape(-1))
        cdef np.ndarray[DTYPE_t, ndim=1] Next = np.ones(self.N)
        cdef np.ndarray[DTYPE_t, ndim=2] pscs = self.P_ScS
        # iterate
        cdef int t,i,j
        cdef int N = self.N
        cdef int T = self.T
        cdef double *_gamma = <double *>gamma.data
        cdef double *_py = <double *>Py.data
        cdef int ystride0 = Py.strides[0]/8
        cdef int ystride1 = Py.strides[1]/8
        cdef double *_alpha = <double *>alpha.data
        cdef int astride0 = alpha.strides[0]/8
        cdef int astride1 = alpha.strides[1]/8
        cdef double *_last = <double *>last.data
        cdef double *_next = <double *>Next.data
        cdef double *_pscs = <double *>pscs.data
        cdef int pstride0 = pscs.strides[0]/8
        cdef int pstride1 = pscs.strides[1]/8
        cdef double *_tmp
        cdef double total
        for t in range(T):
            for i in range(N):
                _last[i] = _last[i]*_py[t*ystride0+i*ystride1]
            total = 0
            for i in range(N):
                total += _last[i]
            _gamma[t] = total
            for i in range(N):
                _last[i] /= total
                _alpha[astride0*t+astride1*i] = _last[i]
            for i in range(N):
                _next[i] = 0
                for j in range(N):
                    _next[i] += _last[j] *  _pscs[j*pstride0+i*pstride1]
            _tmp = _last
            _last = _next
            _next = _tmp
        return (np.log(self.gamma)).sum() # End of forward()
    @cython.boundscheck(False)
    def backward(self):
        """
        On entry:
        self    is an HMM
        y       is a sequence of observations
        exp(PyGhist[t]) = Pr{y(t)=y(t)|y_0^{t-1}}
        On return:
        for each state i, beta[t,i] = Pr{y_{t+1}^T|s(t)=i}/Pr{y_{t+1}^T}
        """
        # Ensure allocation and size of beta
        self.beta = Scalar.initialize(self.beta,(self.T,self.N))
        # Setup direct access to arrays
        cdef np.ndarray[DTYPE_t, ndim=1] gamma = self.gamma
        cdef np.ndarray[DTYPE_t, ndim=2] Py = self.Py
        cdef np.ndarray[DTYPE_t, ndim=2] beta = self.beta
        cdef np.ndarray[DTYPE_t, ndim=1] last = np.ones(self.N)
        cdef np.ndarray[DTYPE_t, ndim=1] _next = np.ones(self.N)
        cdef np.ndarray[DTYPE_t, ndim=2] pcs = self.P_ScS
        # iterate
        cdef int t,i,j
        cdef int N = self.N
        cdef int T = self.T
        for t in range(T-1,-1,-1):
            beta[t,:] = last
            last *= Py[t,:]
            last /= gamma[t]
            for i in range(N):
                _next[i] = 0
                for j in range(N):
                    _next[i] += pcs[i,j] * last[j]
            for i in range(N):
                last[i] = _next[i]
        return # End of backward()
    @cython.boundscheck(False)
    def reestimate_s(self):
        """ Reestimate state transition probabilities and initial
        state probabilities.  Given the observation probabilities, ie,
        self.state[s].Py[t], given alpha, beta, gamma, and Py, these
        calcuations are independent of the observation model
        calculations."""
        cdef np.ndarray[DTYPE_t, ndim=2] u_sum = np.zeros(
            (self.N,self.N),np.float64)
        cdef np.ndarray[DTYPE_t, ndim=1] gamma = self.gamma
        cdef np.ndarray[DTYPE_t, ndim=2] Py = self.Py
        cdef np.ndarray[DTYPE_t, ndim=2] alpha = self.alpha
        cdef np.ndarray[DTYPE_t, ndim=2] beta = self.beta
        cdef int t,i,j
        cdef int N = self.N
        cdef int T = self.T
        for t in range(T-1):
            for i in range(N):
                for j in range(N):
                    u_sum[i,j] += alpha[t,i]*beta[t+1,j]*Py[t+1,j]/gamma[t+1]
        alpha *= beta
        wsum = self.alpha.sum(axis=0)
        self.P_S0_ergodic = np.copy(wsum)
        self.P_S0 = np.copy(self.alpha[0])
        for x in (self.P_S0_ergodic, self.P_S0):
            x /= x.sum()
        ScST = self.P_ScS.T # To get element wise multiplication and correct /
        ScST *= u_sum.T
        ScST /= ScST.sum(axis=0)
        return (self.alpha,wsum) # End of reestimate_s()
#--------------------------------
# Local Variables:
# mode: python
# End:
