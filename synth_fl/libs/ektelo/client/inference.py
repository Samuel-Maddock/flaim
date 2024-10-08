from __future__ import division

import math

import ektelo
import numpy as np
from ektelo import matrix, util
from ektelo.operators import InferenceOperator
from scipy import linalg, optimize, sparse
from scipy.sparse.linalg import lsmr, lsqr


def nnls(A, y, l1_reg=0, l2_reg=0, maxiter=15000):
    """
    Solves the NNLS problem min || Ax - y || s.t. x >= 0 using gradient-based optimization
    :param A: numpy matrix, scipy sparse matrix, or scipy Linear Operator
    :param y: numpy vector
    """

    def loss_and_grad(x):
        diff = A.dot(x) - y
        res = 0.5 * np.sum(diff**2)
        f = res + l1_reg * np.sum(x) + l2_reg * np.sum(x**2)
        grad = A.T.dot(diff) + l1_reg + l2_reg * x

        return f, grad

    xinit = np.zeros(A.shape[1])
    bnds = [(0, None)] * A.shape[1]
    xest, _, info = optimize.lbfgsb.fmin_l_bfgs_b(
        loss_and_grad, x0=xinit, pgtol=1e-4, bounds=bnds, maxiter=maxiter, m=1
    )
    xest[xest < 0] = 0.0
    return xest, info


def wnnls(W, A, y):
    xhat = lsmr(A, y)[0]
    yhat = W.dot(xhat)

    return nnls(W, yhat)


def multWeightsFast(hatx, M, y, updateRounds=1):
    """Multiplicative weights update
    hatx: starting estimate of database
    M: A query matrix representing measurements
    y: array of corresponding answers to query
    updateRounds: number of times to repeat the update of _all_ provided queries

    NOTE: this implementation is more efficient that multWeightsUpdate, and exploits
    possible matrix-free representation of measuremetns more effectively
    However, results may be slightly different than multWeightsUpdate
    """
    assert M.shape[0] == y.size
    total = np.sum(hatx)

    for i in range(updateRounds):
        error = y - M.dot(hatx)
        hatx *= np.exp(M.T.dot(error) / (2.0 * total))
        hatx *= total / np.sum(hatx)

    return hatx


def multWeightsUpdate(hatx, M, y, updateRounds=1):
    """Multiplicative weights update
    hatx: starting estimate of database
    M: A query matrix representing measurements
    y: array of corresponding answers to query
    updateRounds: number of times to repeat the update of _all_ provided queries
    """
    assert M.shape[0] == y.size

    total = np.sum(hatx)

    for i in range(updateRounds):
        for j in range(y.size):
            error = y[j] - M[j].dot(hatx)
            q = M[j].dense_matrix().flatten()
            update_vector = np.exp(q * error / (2.0 * total))
            hatx *= update_vector
            hatx *= total / np.sum(hatx)

    return hatx


def _apply_scales(Ms, ys, scale_factors):
    if scale_factors is None:
        if type(Ms) is list:
            return matrix.VStack(Ms), np.concatenate(ys)
        return Ms, ys
    assert type(Ms) == list and type(ys) == list
    assert len(Ms) > 0 and len(Ms) == len(ys) and len(ys) == len(scale_factors)

    A = matrix.VStack([M * (1.0 / w) for M, w in zip(Ms, scale_factors)])
    y = np.concatenate([y / w for y, w in zip(ys, scale_factors)])

    return A, y


class LeastSquares(InferenceOperator):
    def __init__(self, method="lsmr", l2_reg=0.0):
        super(LeastSquares, self).__init__()

        self.method = method
        self.l2_reg = l2_reg

    def infer(self, Ms, ys, scale_factors=None):
        """Either:
        1) Ms is a single M and ys is a single y
           (scale_factors ignored) or
        2) Ms and ys are lists of M matrices and y vectors
           and scale_factors is a list of the same length.
        """
        A, y = _apply_scales(Ms, ys, scale_factors)

        if self.method == "standard":
            assert self.l2_reg == 0, "l2 reg not supported with method=standard"
            (x_est, _, rank, _) = linalg.lstsq(
                A.dense_matrix(), y, lapack_driver="gelsy"
            )
        elif self.method == "lsmr":
            res = lsmr(A, y, atol=0, btol=0, damp=self.l2_reg)
            x_est = res[0]
        elif self.method == "lsqr":
            res = lsqr(A, y, atol=0, btol=0, damp=self.l2_reg)
            x_est = res[0]

        return x_est


class NonNegativeLeastSquares(InferenceOperator):
    """
    Non negative least squares (nnls)
    Note: undefined behavior when system is under-constrained
    """

    def __init__(self, l1_reg=0, l2_reg=0):
        super(NonNegativeLeastSquares, self).__init__()
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    def infer(self, Ms, ys, scale_factors=None):
        """Either:
        1) Ms is a single M and ys is a single y
           (scale_factors ignored) or
        2) Ms and ys are lists of M matrices and y vectors
           and scale_factors is a list of the same length.
        """
        A, y = _apply_scales(Ms, ys, scale_factors)

        x_est, info = nnls(A, y, self.l1_reg, self.l2_reg)

        return x_est


class WorkloadNonNegativeLeastSquares(InferenceOperator):
    def __init__(self, W):
        self.W = W
        super(WorkloadNonNegativeLeastSquares, self).__init__()

    def infer(self, Ms, ys, scale_factors=None):
        A, y = _apply_scales(Ms, ys, scale_factors)

        x_est, info = wnnls(self.W, A, y)

        return x_est


class MultiplicativeWeights(InferenceOperator):
    """
    Multiplicative weights update with multiple update rounds and optional history
    useHistory is no longer available inside the operator. To use history measurements,
    use M and ans with full history.
    """

    def __init__(self, updateRounds=50):
        super(MultiplicativeWeights, self).__init__()
        self.updateRounds = updateRounds

    def infer(self, Ms, ys, x_est, scale_factors=None):
        """Either:
        1) Ms is a single M and ys is a single y
           (scale_factors ignored) or
        2) Ms and ys are lists of M matrices and y vectors
           and scale_factors is a list of the same length.
        """
        M, y = _apply_scales(Ms, ys, scale_factors)

        """ mult_weights is an update method which works on the original domain"""
        assert (
            x_est is not None
        ), "Multiplicative Weights update needs a starting xest, but there is none."

        x_est = multWeightsUpdate(x_est, M, y, self.updateRounds)
        return x_est


class AHPThresholding(InferenceOperator):
    """
    Special update operator for AHP thresholding step.
    This operator assumes that the previous one is a Laplace measurement of the Identity workload.
    The xest is updated by answers from the Identity workload after thresholding.
    To calculate the threshold, the eps used for the measurement is assumed to be ratio*_eps_total
    """

    def __init__(self, eta, ratio):
        super(AHPThresholding, self).__init__()
        self.eta = eta
        self.ratio = ratio

    def infer(self, Ms, ys, eps_par, scale_factors=None):
        """Either:
        1) Ms is a single M and ys is a single y
           (scale_factors ignored) or
        2) Ms and ys are lists of M matrices and y vectors
           and scale_factors is a list of the same length.
        """
        A, y = _apply_scales(Ms, ys, scale_factors)

        eps = eps_par * self.ratio
        x_est = lsmr(A, y.flatten())[0]
        x_est = x_est.reshape(A.shape[1])
        n = len(x_est)
        cutoff = self.eta * math.log(n) / eps
        x_est = np.where(x_est <= cutoff, 0, x_est)

        return x_est
