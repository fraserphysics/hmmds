import numpy
import numpy.linalg

A = numpy.arange(12).reshape((3, 2, 2))
U, S, Vh = numpy.linalg.svd(A, full_matrices=True)
US = numpy.empty((3, 2, 2))
for i in range(3):
    for j in range(2):
        for k in range(2):
            US[i, j, k] = U[i, j, k] * S[i, k]
assert numpy.allclose(US, U * S[..., None, :])
A_ = numpy.matmul(U * S[..., None, :], Vh)
assert numpy.allclose(A, A_)
print(f'{U.shape=}, {S.shape=} {Vh.shape=}')
for i, a in enumerate(A):
    u, s, vh = numpy.linalg.svd(a, full_matrices=True)
    assert numpy.allclose(u, U[i])
    assert numpy.allclose(s, S[i])
    assert numpy.allclose(vh, Vh[i])
    print(f'{u.shape=}, {s.shape=} {vh.shape=}')
flat = -A.flatten()
flat.sort()
print(f'{flat=}')
