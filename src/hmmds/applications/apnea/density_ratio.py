"""simple.py Estimates a ratio of probability densities as a function

References:

See Eqn 14 on page 16 of 
http://www.ms.k.u-tokyo.ac.jp/sugi/2010/RIMS2010.pdf

Density Ratio Estimation: A Comprehensive Review
Masashi Sugiyama, Tokyo Institute of Technology (sugi@cs.titech.ac.jp)
Taiji Suzuki, The University of Tokyo (s-taiji@stat.t.u-tokyo.ac.jp)
Takafumi Kanamori, Nagoya University (kanamori@is.nagoya-u.ac.jp)

"""
import numpy

import numpy.random
import numpy.linalg


class DensityRatio:

    def __init__(self, sigma, centers, theta):
        self.sigma = sigma
        self.centers = centers
        self.theta = theta

    def __call__(self, x):
        return compute_kernel(x, self.centers, self.sigma) @ self.theta


def uLSIF(x, y, sigma, _lambda, kernel_num=100):
    """
    Estimate p(x)/q(x) by uLSIF
    (Unconstrained Least-Square Importance Fitting)

    Arguments:
        x (numpy.ndarray): Sample from p(x).
        y (numpy.ndarray): Sample from q(x).
        sigma (float): Gaussian kernel bandwidth.
        lambda (float): Regularization parameter.
        kernel_num (int): Number of kernels. (Default 100)

    Returns:
        DensityRatio instance
    """

    # Number of samples.
    nx = x.shape[0]
    ny = y.shape[0]

    kernel_num = min(kernel_num, nx)
    centers = x[numpy.random.randint(nx, size=kernel_num)]

    psi_x = compute_kernel(x, centers, sigma)
    psi_y = compute_kernel(y, centers, sigma)
    assert psi_x.shape == (nx, kernel_num)
    assert psi_y.shape == (ny, kernel_num)
    H = psi_y.T.dot(psi_y) / ny
    h = psi_x.mean(axis=0).T
    assert h.shape == (kernel_num,)
    assert H.shape == (kernel_num, kernel_num)
    theta = numpy.linalg.solve(
        H + numpy.diag(numpy.array(_lambda).repeat(kernel_num)), h)
    return DensityRatio(sigma, centers, theta)


def compute_kernel(x_array, y_array, sigma):
    """ Calculate Gaussian densities for pairs x,y

    Args:
    x_array: x_array.shape == (n_x, n_d)
    y_array: y_array.shape == (n_y, n_d)
    sigma: Scalar width

    Returns:
    result[i,j] = exp(-( (x[i]-y[j]) dot (x[i]-y[j]) ) / (2 * sigma**2)
    """

    n_x = x_array.shape[0]
    n_y = y_array.shape[0]
    result = numpy.empty((n_x, n_y))

    for j, y_row in enumerate(y_array):
        result[:, j] = numpy.exp(-numpy.power(x_array - y_row, 2).sum(axis=1) /
                                 (2 * sigma * sigma))

    return result
