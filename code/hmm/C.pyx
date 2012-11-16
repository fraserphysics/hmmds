'''C.pyx: cython code for speed up.  Only significant improvment comes
from the loop over time in reestimate_s()
'''
import Scalar, numpy as np, scipy.sparse as SS
# Imitate http://docs.cython.org/src/tutorial/numpy.html
cimport cython, numpy as np
DTYPE = np.float64
ITYPE = np.int32
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t ITYPE_t
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

        # Setup direct access to numpy arrays
        cdef np.ndarray[DTYPE_t, ndim=1] gamma = self.gamma
        cdef double *_gamma = <double *>gamma.data

        cdef np.ndarray[DTYPE_t, ndim=1] last = np.copy(self.P_S0.reshape(-1))
        cdef double *_last = <double *>last.data

        cdef np.ndarray[DTYPE_t, ndim=2] Alpha = self.alpha
        cdef char *_alpha = Alpha.data
        cdef int astride = Alpha.strides[0]
        cdef double *a

        cdef np.ndarray[DTYPE_t, ndim=2] Pscs = self.P_ScS
        cdef char *_ps = Pscs.data
        cdef int pstride0 = Pscs.strides[0]
        cdef int pstride1 = Pscs.strides[1]
        cdef double *ps

        cdef np.ndarray[DTYPE_t, ndim=2] Py = self.Py
        cdef int pystride = Py.strides[0]
        cdef char *_py = Py.data
        cdef double *py

        # Allocate vector of length N
        cdef np.ndarray[DTYPE_t, ndim=1] Next = np.empty(self.N)
        cdef double *_next = <double *>Next.data

        cdef int t,i,j
        cdef int N = self.N
        cdef int T = self.T
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
            for i in range(N):
                _next[i] = 0
                for j in range(N):
                    ps = <double *>(_ps + j*pstride0+i*pstride1)
                    _next[i] += _last[j] *  ps[0]
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

        # Setup direct access to numpy arrays
        cdef np.ndarray[DTYPE_t, ndim=1] gamma = self.gamma
        cdef double *_gamma = <double *>gamma.data

        cdef np.ndarray[DTYPE_t, ndim=1] last = np.ones(self.N)
        cdef double *_last = <double *>last.data

        cdef np.ndarray[DTYPE_t, ndim=2] Beta = self.beta
        cdef char *_beta = Beta.data
        cdef int bstride = Beta.strides[0]
        cdef double *b

        cdef np.ndarray[DTYPE_t, ndim=2] Pscs = self.P_ScS
        cdef char *_ps = Pscs.data
        cdef int pstride = Pscs.strides[0]
        cdef double *ps

        cdef np.ndarray[DTYPE_t, ndim=2] Py = self.Py
        cdef int pystride = Py.strides[0]
        cdef char *_py = Py.data
        cdef double *py

        # Allocate vector of length N
        cdef np.ndarray[DTYPE_t, ndim=1] Next = np.empty(self.N)
        cdef double *_next = <double *>Next.data

        # iterate
        cdef int t,i,j
        cdef int N = self.N
        cdef int T = self.T
        cdef double *_tmp
        cdef double total
        for t in range(T-1,-1,-1):
            py = <double *>(_py+t*pystride)
            b = <double *>(_beta+t*bstride)
            for i in range(N):
                b[i] = _last[i]
                _last[i] *= py[i]/_gamma[t]
            for i in range(N):
                total = 0
                ps = <double *>(_ps + i*pstride)
                for j in range(N):
                    total += ps[j] * _last[j]
                _next[i] = total
            _tmp = _last
            _last = _next
            _next = _tmp
        return # End of backward()
    @cython.boundscheck(False)
    def reestimate_s(self):
        """ Reestimate state transition probabilities and initial
        state probabilities.  Given the observation probabilities, ie,
        self.state[s].Py[t], given alpha, beta, gamma, and Py, these
        calcuations are independent of the observation model
        calculations."""
        cdef np.ndarray[DTYPE_t, ndim=1] wsum = np.zeros(self.N,np.float64)
        cdef double *_w = <double *>wsum.data

        cdef np.ndarray[DTYPE_t, ndim=2] u_sum = np.zeros(
            (self.N,self.N),np.float64)
        cdef char *_u = u_sum.data
        cdef int ustride = u_sum.strides[0]
        cdef double *u

        cdef np.ndarray[DTYPE_t, ndim=1] gamma = self.gamma
        cdef double *_gamma = <double *>gamma.data

        cdef np.ndarray[DTYPE_t, ndim=2] Py = self.Py
        cdef int pystride = Py.strides[0]
        cdef char *_py = Py.data
        cdef double *py

        cdef np.ndarray[DTYPE_t, ndim=2] Alpha = self.alpha
        cdef char *_alpha = Alpha.data
        cdef int astride = Alpha.strides[0]
        cdef double *a

        cdef np.ndarray[DTYPE_t, ndim=2] Beta = self.beta
        cdef char *_beta = Beta.data
        cdef int bstride = Beta.strides[0]
        cdef double *b
        cdef double *bt

        cdef int t,i,j
        cdef int N = self.N
        cdef int T = self.T
        for t in range(T-1):
            py = <double *>(_py+(t+1)*pystride)
            a = <double *>(_alpha+t*astride)
            b = <double *>(_beta+(t+1)*bstride)
            bt = <double *>(_beta+t*bstride)
            for i in range(N):
                u = <double *>(_u+i*ustride)
                for j in range(N):
                    u[j] += a[i]*b[j]*py[j]/_gamma[t+1]
                a[i] *= bt[i]
                _w[i] += a[i]
        #Alpha[T-1,:] *= Beta[T-1,:] but Beta[T-1,:] = 1
        wsum += self.alpha[T-1]
        self.P_S0_ergodic = np.copy(wsum)
        self.P_S0 = np.copy(self.alpha[0])
        for x in (self.P_S0_ergodic, self.P_S0):
            x /= x.sum()
        self.P_ScS.inplace_elementwise_multiply(u_sum)
        self.P_ScS.normalize()
        return self.alpha # End of reestimate_s()

class PROB(np.ndarray):
    '''   Dying code to be replaced by SPARSE class
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
class SPARSE(PROB):
    '''Replacement for Scalar.PROB that stores data in sparse matrix
    format.  P[a,b] is the probability of b given a.
    '''
    threshold = 1.0e-15
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
          strides=None, order=None):
        obj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset, strides,
                         order)
        obj.tcol = np.empty(shape[0])
        obj.trow = np.empty(shape[1])
        obj.normalize()
        return obj
    def normalize(self):
        S = self.sum(axis=1)
        for i in range(self.shape[0]):
            self[i,:] /= S[i]
        mask = np.zeros(self.shape,dtype=np.int8)
        for i in range(self.shape[0]):
            row = self[i,:]
            mask[i,np.where(row>row.max()*SPARSE.threshold)] = 1
        for i in range(self.shape[1]):
            col = self[:,i]
            mask[np.where(col>col.max()*SPARSE.threshold),i] = 1
        self *= mask
        self.csc = SS.csc_matrix(self)
    def step_back(self,A):
        #A[:] = self.csc*A
        #return
        cdef np.ndarray[DTYPE_t, ndim=1] _A = A
        cdef double *a = <double *>_A.data
        cdef np.ndarray[DTYPE_t, ndim=1] data = self.csc.data
        cdef double *_data = <double *>data.data
        cdef np.ndarray[ITYPE_t, ndim=1] indices = self.csc.indices
        cdef int *_indices = <int *>indices.data
        cdef np.ndarray[ITYPE_t, ndim=1] indptr = self.csc.indptr
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
        #A[:] = A*self.csc
        #return
        cdef np.ndarray[DTYPE_t, ndim=1] _A = A
        cdef double *a = <double *>_A.data
        cdef np.ndarray[DTYPE_t, ndim=1] data = self.csc.data
        cdef double *_data = <double *>data.data
        cdef np.ndarray[ITYPE_t, ndim=1] indices = self.csc.indices
        cdef int *_indices = <int *>indices.data
        cdef np.ndarray[ITYPE_t, ndim=1] indptr = self.csc.indptr
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
    x = np.array(x)
    return SPARSE(x.shape,buffer=x.data)

class HMM_SPARSE(Scalar.HMM):
    def __init__(self, P_S0, P_S0_ergodic, P_ScS, P_YcS):
        Scalar.HMM.__init__(self, P_S0, P_S0_ergodic, P_ScS, P_YcS,
                        prob=make_prob)
#--------------------------------
# Local Variables:
# mode: python
# End:
