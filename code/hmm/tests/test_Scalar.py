# export PYTHONPATH=/home/andy/projects/hmmds3/code/
# export PYTHONPATH=/home/andy/projects/hmmds3/code/hmm/:$PYTHONPATH
# Copyright (c) 2013 Andrew M. Fraser
import numpy as np
import Scalar
from numpy.testing import assert_, assert_allclose, assert_almost_equal
from numpy.testing import run_module_suite, assert_equal
from scipy.linalg import circulant
import C

A = [
    [0, 2, 2.0],
    [2, 2, 4.0],
    [6, 2, 2.0]]
B = [
    [0, 1],
    [1, 1],
    [1, 3.0]]
C = [
    [0, 0, 2.0],
    [0, 0, 1.0],
    [6, 0, 0.0]]
class TestScalar:
    def __init__(self):
        self.A = Scalar.make_prob(A)
        self.B = Scalar.make_prob(B)
        self.C = Scalar.make_prob(C)
        self.Ms = (self.A, self.B, self.C)
        for M in self.Ms:
            M.normalize()
        return
    def test_normalize(self):
        for M in self.Ms:
            m,n = M.shape
            for i in range(m):
                s = 0
                for j in range(n):
                    s += M[i,j]
                assert_almost_equal(1, s)
        return
    def test_assign(self):
        a = self.C.sum()
        self.C.assign_col(1, [1, 1, 1])
        assert_almost_equal(self.C.sum(), a+3)
        return
    def test_likelihoods(self):
        assert_allclose(self.C.likelihoods([0,1,2])[2], [1,1,0])
        return
    def test_cost(self):
        assert_almost_equal(self.C.cost(self.B.T[0], self.B.T[1]),
                            [[ 0, 0, 0], [0, 0, .375], [0.25, 0, 0]])
        return
    def test_inplace_elementwise_multiply(self):
        self.C.inplace_elementwise_multiply(self.A)
        assert_almost_equal(self.C, [[ 0, 0, .5], [0, 0, 0.5], [0.6, 0, 0]])
        return
    def test_step_forward(self):
        self.A.step_forward(self.B.T[1])
        assert_almost_equal(self.B.T[1], [ 0.575,  0.775,  0.9  ])
    def test_step_back(self):
        self.A.step_back(self.B.T[1])
        assert_almost_equal(self.B.T[1], [ 0.625,  0.75,  0.85  ])
    def test_values(self):
        assert_almost_equal(self.C.values(), [[0,0,1],[0,0,1],[1,0,0]])
class Test_Discrete_Observations:
    def __init__(self):
        P_YS = Scalar.make_prob(B)
        P_YS.normalize()
        self.y_mod = Scalar.Discrete_Observations(P_YS)
        N = 20
        Y = np.empty(N, dtype=np.int32)
        for i in range(N):
            Y[i] = (i + i%2 + i%3 + i%5)%2
        self.Y = [Y]
        self.w = np.array(20*[0,0,1.0]).reshape((N,3))
        self.w[0,:] = [1,0,0]
        self.w[3,:] = [0,1,0]
        self.Ys = [[Y[5:]],[Y[3:7]],[Y[:4]]]
    def test_calc(self):
        PY = self.y_mod.calc(self.Y)[2:4]
        assert_equal(PY, [[ 0, 0.5, 0.25],[ 1, 0.5, 0.75]])
    def test_join(self):
        n_seg, t_seg, y_all  = self.y_mod.join(self.Ys)
        assert_equal(n_seg, 3)
        assert_equal(t_seg, [0, 15, 19, 23])
    def test_reestimate(self):
        self.y_mod.reestimate(self.w, self.Y)
        assert_almost_equal([[1, 0],[0, 1],[5/9, 4/9]], self.y_mod.P_YS)
        print(self.y_mod)

if __name__ == "__main__":
    run_module_suite()

#--------------------------------
# Local Variables:
# mode: python
# End:
