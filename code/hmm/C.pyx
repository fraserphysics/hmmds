import Scalar, numpy as np

def initialize(x,shape,dtype=np.float64):
    if x == None or x.shape != shape:
        return np.zeros(shape,dtype)
    return x*0

class HMM(Scalar.HMM):
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
       self.alpha = initialize(self.alpha,(self.T,self.N))
       self.gamma = initialize(self.gamma,(self.T,))
       last = np.copy(self.P_S0.reshape(-1)) # Copy
       for t in range(self.T):
           last *= self.Py[t]              # Element-wise multiply
           self.gamma[t] = last.sum()
           last /= self.gamma[t]
           self.alpha[t,:] = last
           last = np.dot(last,self.P_ScS)
       return (np.log(self.gamma)).sum() # End of forward()
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
        self.beta = initialize(self.beta,(self.T,self.N))
        last = np.array(np.ones(self.beta.shape[1]))
        pscst = self.P_ScS.T               # Transpose view
        # iterate
        for t in range(self.T-1,-1,-1):
            self.beta[t,:] = last
            last = np.dot((last*self.Py[t,:]/self.gamma[t]),pscst)
        return # End of backward()
    def reestimate_s(self):
        """ Reestimate state transition probabilities and initial
        state probabilities.  Given the observation probabilities, ie,
        self.state[s].Py[t], given alpha, beta, gamma, and Py, these
        calcuations are independent of the observation model
        calculations."""
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
        ScST = self.P_ScS.T # To get element wise multiplication and correct /
        ScST *= u_sum.T
        ScST /= ScST.sum(axis=0)
        return (self.alpha,wsum) # End of reestimate_s()
#--------------------------------
# Local Variables:
# mode: python
# End:
