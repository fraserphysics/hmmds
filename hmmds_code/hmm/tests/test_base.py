# In root of project, run:
# conda create -n hmmds --file pip_req.txt
# source activate hmmds
# python setup.py develop
# python -m pytest hmmds_code/ or pytest hmmds_code/hmm/tests/test_base.py
# Copyright (c) 2013 2017 Andrew M. Fraser
import numpy as np
from numpy.testing import assert_, assert_allclose, run_module_suite
from scipy.linalg import circulant
import hmmds_code.hmm.Scalar
import hmmds_code.hmm.base
import hmmds_code.hmm.C
import unittest

c2s = {
    0:[0,1],
    1:[2,3],
    2:[4,5],
    }
P_S0 = np.ones(6)/6.0
P_S0_ergodic = np.ones(6)/6.0
P_SS = circulant([0,  0, 0, 0, .5, .5])
P_YS = circulant([.4, 0, 0, 0, .3, .3])
class TestHMM(unittest.TestCase):
    def setUp(self):
        self.mod = hmmds_code.hmm.base.HMM(
            P_S0.copy(),         # Initial distribution of states
            P_S0_ergodic.copy(), # Stationary distribution of states
            P_YS.copy(),         # Parameters of observation model
            P_SS.copy()          # State trasition proabilities
        )
        self.Cmod = hmmds_code.hmm.C.HMM(
            P_S0.copy(), P_S0_ergodic.copy(), P_YS.copy(), P_SS.copy())
        self.Smod = hmmds_code.hmm.C.HMM_SPARSE(
            P_S0.copy(), P_S0_ergodic.copy(), P_YS.copy(), P_SS.copy())
        self.mods = (self.mod, self.Cmod, self.Smod)
        self.S,Y = self.mod.simulate(1000)
        self.Y = (np.array(Y[0], np.int32),)
        return
    def test_decode(self):
        # Check that self.mod gets 70% of the states right
        states = self.mod.decode(self.Y)
        wrong = np.where(states != self.S)[0]
        assert_(len(wrong) < 300)
        # Check that other models get the same state sequence as self.mod
        for mod in self.mods[1:]:
            wrong = np.where(states != mod.decode(self.Y))[0]
            assert_(len(wrong) == 0)
        return
    def test_train(self):
        L = self.mod.train(self.Y,n_iter=10, display=False)
        # Check that log likelihood increases montonically
        for i in range(1,len(L)):
            assert_(L[i-1] < L[i])
        # Check that trained model is close to true model
        assert_allclose(self.mod.y_mod.P_YS.values(), P_YS, atol=0.08)
        assert_allclose(self.mod.P_SS.values(), P_SS, atol=0.2)
        # Check that other models give results close to self.mod
        for mod in self.mods[1:]:
            L_mod = mod.train(self.Y,n_iter=10, display=False)
            assert_allclose(L_mod, L)
            assert_allclose(
                mod.y_mod.P_YS.values(), self.mod.y_mod.P_YS.values())
            assert_allclose(mod.P_SS.values(), self.mod.P_SS.values())
        return
    def multi_train(self, mod):
        ys = []
        for i in [1,2,0,4,3]:
            ys.append([x[200*i:200*(i+1)] for x in self.Y])
        L = mod.multi_train(ys, n_iter=10, display=False)
        for i in range(1,len(L)):
            assert_(L[i-1] < L[i])
        assert_allclose(mod.y_mod.P_YS.values(), P_YS, atol=0.08)
        assert_allclose(mod.P_SS.values(),       P_SS, atol=0.2)
    def test_multi_train(self):
        for mod in self.mods:
            self.multi_train(mod)
class TestHMM_classy(unittest.TestCase):
    def setUp(self):
        pars = (hmmds_code.hmm.Scalar.Discrete_Observations, P_YS, c2s)
        self.mod = hmmds_code.hmm.base.HMM(P_S0, P_S0, pars, P_SS, hmmds_code.hmm.Scalar.Class_y, hmmds_code.hmm.Scalar.make_prob)
        self.S,CY = self.mod.simulate(1000)
        self.CY = [np.array(CY[0], np.int32), np.array(CY[1], np.int32)]
        p_s = 0.7*P_SS + 0.3/6
        p_y = 0.7*P_YS + 0.3/6
        pars = (hmmds_code.hmm.Scalar.Discrete_Observations, p_y, c2s)
        self.mod_t = hmmds_code.hmm.base.HMM(P_S0, P_S0, pars, p_s, hmmds_code.hmm.Scalar.Class_y, hmmds_code.hmm.Scalar.make_prob)
        self.L = self.mod_t.train(CY, n_iter=20, display=False)
    def test_train(self):
        for i in range(1,len(self.L)):
            assert_(self.L[i-1] < self.L[i])
        assert_allclose(self.mod_t.y_mod.y_mod.P_YS, P_YS, atol=0.08)
        assert_allclose(self.mod_t.P_SS, P_SS, atol=0.16)
    def test_decode(self):
        D = self.mod.class_decode((self.CY[1],))
        E = np.where(D != self.CY[0])[0]
        assert_(len(E) < 150),'from assert E={0}\nlen(E)={1}'.format(E,len(E))

if __name__ == "__main__":
    run_module_suite()

#--------------------------------
# Local Variables:
# mode: python
# End:
