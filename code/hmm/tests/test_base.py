# Copyright (c) 2013 Andrew M. Fraser
import numpy as np
from hmm.Scalar import initialize, Prob, Discrete_Observations, Class_y
from hmm.Scalar import make_prob
from hmm.base import HMM
from numpy.testing import assert_, assert_allclose, run_module_suite
from scipy.linalg import circulant

c2s = {
    0:[0,1],
    1:[2,3],
    2:[4,5],
    }
P_S0 = np.ones(6)/6.0
P_S0_ergodic = np.ones(6)/6.0
P_SS = circulant([0,  0, 0, 0, .5, .5])
P_YS = circulant([.4, 0, 0, 0, .3, .3])
class TestHMM:
    def __init__(self):
        self.mod = HMM(P_S0,P_S0_ergodic,P_YS,P_SS)
        self.S,Y = self.mod.simulate(1000)
        Y = (np.array(Y[0], np.int32),)
        self.Y = Y
    def test_decode(self):
        E = np.where(self.mod.decode(self.Y) != self.S)[0]
        assert_(len(E) < 300)
    def test_train(self):
        L = self.mod.train(self.Y,n_iter=10, display=False)
        for i in range(1,len(L)):
            assert_(L[i-1] < L[i])
        assert_allclose(self.mod.y_mod.P_YS, P_YS, atol=0.08)
        assert_allclose(self.mod.P_SS, P_SS, atol=0.2)
        return
class TestHMM_classy:
    def __init__(self):
        pars = (Discrete_Observations, P_YS, c2s)
        self.mod = HMM(P_S0, P_S0, pars, P_SS, Class_y, make_prob)
        self.S,CY = self.mod.simulate(1000)
        self.CY = [np.array(CY[0], np.int32), np.array(CY[1], np.int32)]
        p_s = 0.7*P_SS + 0.3/6
        p_y = 0.7*P_YS + 0.3/6
        pars = (Discrete_Observations, p_y, c2s)
        self.mod_t = HMM(P_S0, P_S0, pars, p_s, Class_y, make_prob)
        self.L = self.mod_t.train(CY, n_iter=20, display=False)
    def test_train(self):
        for i in range(1,len(self.L)):
            assert_(self.L[i-1] < self.L[i])
        assert_allclose(self.mod_t.y_mod.y_mod.P_YS, P_YS, atol=0.08)
        assert_allclose(self.mod_t.P_SS, P_SS, atol=0.16)
    def test_decode(self):
        D = self.mod.class_decode((self.CY[1],))
        E = np.where(D != self.CY[0])[0]
        assert_(len(E) < 150)
    # ToDo: Change globals to those used for doc_test class_decode.
    # Test  multi_train()

if __name__ == "__main__":
    run_module_suite()
