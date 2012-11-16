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

class PROB:
    '''Replacement for Scalar.PROB that stores data in sparse matrix
    format.  P[a,b] is the probability of b given a.
    '''
    threshold = 1.0e-15
    # For pruning.  Drop x[i,j] if x[i,j] < threshold*max(A[i,:]) and
    # x[i,j] < threshold*max(A[:,j])
    def __init__(self,x):
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
    def assign_col(self,i,col):
        '''Implements self[:,i]=col.  Very slow because finding each csc[j,i]
        is slow

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
        for i in range(M):    # Normalize the rows and find max of the
                              # resulting rows and columns
            for j in range(self.indptr[i],self.indptr[i+1]):
                J = self.indices[j]
                self.data[j] /= sum_col[J]
                x = self.data[j]
                if x > max_row[i]:
                    max_row[i] = x
                if x > max_col[J]:
                    max_col[J] = x
        for i in range(M):
            k = self.indptr[i]
            L = self.indptr[i+1]
            for j in range(self.indptr[i],L):
                J = self.indices[j]
                x = self.data[j]/PROB.threshold
                if x > max_row[i] or x > max_col[J]: # FixMe: Check this
                    self.indices[k] = J
                    self.data[k] = self.data[j]
                    k += 1
                else:
                    self.indptr[i+1] -= 1
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
        ''' Implements A[:] = self*A*self
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
    return PROB(x)

class HMM_SPARSE(Scalar.HMM):
    def __init__(self, P_S0, P_S0_ergodic, P_ScS, P_YcS):
        Scalar.HMM.__init__(self, P_S0, P_S0_ergodic, P_ScS, P_YcS,
                        prob=make_prob)
#--------------------------------
# Local Variables:
# mode: python
# End:
