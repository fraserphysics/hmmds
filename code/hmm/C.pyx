'''C.pyx: cython code for speed up.  Only significant improvment comes
from the loop over time in reestimate_s()
'''
import Scalar, numpy as np
# Imitate http://docs.cython.org/src/tutorial/numpy.html
cimport cython, numpy as np
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

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
        cdef np.ndarray[DTYPE_t, ndim=1] _next = np.ones(self.N)
        cdef np.ndarray[DTYPE_t, ndim=2] pcs = self.P_ScS
        # iterate
        cdef int t,i,j
        cdef int N = self.N
        cdef int T = self.T
        for t in range(T):
            last *= Py[t]              # Element-wise multiply
            gamma[t] = last.sum()
            last /= gamma[t]
            alpha[t,:] = last
            for i in range(N):
                _next[i] = 0
                for j in range(N):
                    _next[i] += last[j] *  pcs[j,i]
            for i in range(N):
                last[i] = _next[i]
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
