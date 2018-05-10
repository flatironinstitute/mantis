from __future__ import print_function, division, absolute_import
import cvxpy as cp
from functools import partial
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from nomad.nmf import DecomposedArray, symnmf_admm
from nomad.utils import dot_matrix


def nomad(X, n_clusters, method='cvx'):
    if method == 'cvx':
        D = dot_matrix(X)
        Q = sdp_km(D, n_clusters)
    elif method == 'cgm':
        D = dot_matrix(X)
        Q = sdp_km_conditional_gradient(D, n_clusters)
    elif method == 'bm':
        Y = sdp_km_burer_monteiro(X, n_clusters)
        D = dot_matrix(X)
        Q = Y.dot(Y.T)
    else:
        raise ValueError('The method should be one of "cvx" and "bm"')

    return D, Q


def sdp_km(D, n_clusters):
    ones = np.ones((D.shape[0], 1))
    Z = cp.Semidef(D.shape[0])
    objective = cp.Maximize(cp.trace(D * Z))
    constraints = [Z >= 0,
                   Z * ones == ones,
                   cp.trace(Z) == n_clusters]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    Q = np.asarray(Z.value)
    rs = Q.sum(axis=1)
    print('Q', Q.min(), Q.max(), '|',
          rs.min(), rs.max(), '|',
          np.trace(Q), np.trace(D.dot(Q)))
    print('Final objective', np.trace(D.dot(Q)))

    return Q


def sdp_km_burer_monteiro(X, n_clusters, rank=None, maxiter=1e3, tol=1e-5,
                          verbose=False):
    if rank is None:
        rank = 8 * n_clusters

    X_norm = X - np.mean(X, axis=0)
    if X_norm.shape[0] > X_norm.shape[1]:
        cov = X_norm.T.dot(X_norm)
    else:
        XXt = X_norm.dot(X_norm.T)
        cov = XXt
    X_norm /= np.trace(cov.dot(cov)) ** 0.25

    if X_norm.shape[0] <= X_norm.shape[1]:
        XXt /= np.trace(cov.dot(cov)) ** 0.5

    Y_shape = (len(X), rank)
    ones = np.ones((len(X), 1))

    def lagrangian(x, lambda1, lambda2, sigma1, sigma2):
        Y = x.reshape(Y_shape)

        if X_norm.shape[0] > X_norm.shape[1]:
            YtX = Y.T.dot(X_norm)
            obj = -np.trace(YtX.dot(YtX.T))
        else:
            obj = -np.trace(Y.T.dot(XXt).dot(Y))

        trYtY_minus_nclusters = np.trace(Y.T.dot(Y)) - n_clusters
        obj -= lambda1 * trYtY_minus_nclusters
        obj += .5 * sigma1 * trYtY_minus_nclusters ** 2

        YYt1_minus_1 = Y.dot(Y.T.dot(ones)) - ones
        obj -= lambda2.T.dot(YYt1_minus_1)[0, 0]
        obj += .5 * sigma2 * np.sum(YYt1_minus_1 ** 2)

        return obj

    def grad(x, lambda1, lambda2, sigma1, sigma2):
        Y = x.reshape(Y_shape)

        if X_norm.shape[0] > X_norm.shape[1]:
            delta = -2 * X_norm.dot(X_norm.T.dot(Y))
        else:
            delta = -2 * XXt.dot(Y)

        YtY = Y.T.dot(Y)
        delta -= 2 * (lambda1
                      -sigma1 * (np.trace(YtY) - n_clusters)) * Y

        delta -= ones.dot(lambda2.T.dot(Y)) + lambda2.dot(ones.T.dot(Y))

        Yt1 = Y.T.dot(ones)
        delta += sigma2 * (Y.dot(Yt1).dot(Yt1.T) + ones.dot(Yt1.T).dot(YtY)
                           -2 * ones.dot(Yt1.T))

        return delta.flatten()

    if X_norm.shape[0] > X_norm.shape[1]:
        da = DecomposedArray(X_norm)
        Y = symnmf_admm(da, rank)
    else:
        Y = symnmf_admm(XXt, rank)

    Y /= np.linalg.norm(Y, axis=0, keepdims=True)
    Y *= n_clusters / len(Y)

    lambda1 = 0.
    lambda2 = np.zeros((len(X), 1))
    sigma1 = 1
    sigma2 = 1
    step = 1

    error = []
    for n_iter in range(int(maxiter)):
        fun = partial(lagrangian, lambda1=lambda1, lambda2=lambda2,
                      sigma1=sigma1, sigma2=sigma2)
        jac = partial(grad, lambda1=lambda1, lambda2=lambda2,
                      sigma1=sigma1, sigma2=sigma2)
        bounds = [(0, 1)] * np.prod(Y_shape)

        Y_old = Y.copy()
        res = minimize(fun, Y.flatten(), jac=jac, bounds=bounds,
                       method='L-BFGS-B',)
        Y = res.x.reshape(Y_shape)

        lambda1 -= step * sigma1 * (np.trace(Y.T.dot(Y)) - n_clusters)
        lambda2 -= step * sigma2 * (Y.dot(Y.T.dot(ones)) - ones)

        error.append(np.linalg.norm(Y - Y_old) / np.linalg.norm(Y_old))

        if verbose:
            YYt1_minus_1 = Y.dot(Y.T.dot(ones)) - ones
            fmt_str = '{: 4d} | Y bounds {:.5f} {:.5f} | ' \
                      'constraints {:.10f} {:.10f} | E {:.10f}'
            print(fmt_str.format(n_iter, Y.min(), Y.max(),
                                 np.trace(Y.T.dot(Y)) - n_clusters,
                                 np.abs(YYt1_minus_1).max(),
                                 error[-1]
                                 ))


        if error[-1] < tol:
            break

    return Y


def sdp_km_conditional_gradient(D, n_clusters, max_iter=2e3,
                                stop_tol_max=1e-1, stop_tol_rmse=1e-4,
                                verbose=False, track_stats=False):
    n = len(D)
    one_over_n = 1. / n

    def lagrangian(Q, lagrange_lower_bound, t):
        obj = -np.sum(D * Q)
        obj += np.sum(lagrange_lower_bound * (Q + one_over_n))
        penalty = (t + 1) ** 0.5
        obj += penalty * np.sum(np.minimum(Q + one_over_n, 0) ** 2)
        return obj

    def gradient(Q, lagrange_lower_bound, t):
        delta = -D

        delta += lagrange_lower_bound
        penalty = (t + 1) ** 0.5
        delta += penalty * np.minimum(Q + one_over_n, 0)
        return delta

    def solve_lp(grad, t):
        # The following commented 2 lines are the mathematically
        # friendly version of the 2 lines following them.
        # ortho_mat = np.eye(n) - np.ones_like(D) / n
        # A = ortho_mat.dot(grad).dot(ortho_mat)
        grad11 = np.broadcast_to(grad.sum(axis=1) / n, (n, n))
        A = grad - grad11 - grad11.T + grad.sum() / (n ** 2)

        tol = (t + 1) ** -1
        return sp.linalg.eigsh(A, k=1, which='SA', tol=tol)

    def line_search(update, current, lagrange_lower_bound, t):
        alpha_lower = 0
        alpha_upper = 1

        while np.abs(alpha_lower - alpha_upper) > 1e-4:
            lin_comb_lower = alpha_lower * update + (1 - alpha_lower) * current
            lin_comb_upper = alpha_upper * update + (1 - alpha_upper) * current

            obj_lower = lagrangian(lin_comb_lower, lagrange_lower_bound, t)
            obj_upper = lagrangian(lin_comb_upper, lagrange_lower_bound, t)

            if obj_upper < obj_lower:
                alpha_lower = 0.5 * (alpha_lower + alpha_upper)
            else:
                alpha_upper = 0.5 * (alpha_lower + alpha_upper)

        return alpha_lower

    if track_stats or verbose:
        rmse_list = []
        obj_value_list = []

    Q = np.zeros_like(D)
    lagrange_lower_bound = np.zeros(D.shape)
    step = 1
    n_inner_iter = 10

    for t in range(int(max_iter)):
        for inner_it in range(n_inner_iter):
            grad = gradient(Q, lagrange_lower_bound, t)
            s, v, = solve_lp(grad, t)

            if s < 0:
                update = (n_clusters - 1) * np.outer(v, v)
                eta_ls = line_search(update, Q, lagrange_lower_bound,
                                     t * n_inner_iter + inner_it)
                eta_fix = 2. / (t * n_inner_iter + inner_it + 2)
                eta = np.maximum(eta_ls, eta_fix)
                # eta = 2. / (t * n_inner_iter + inner_it + 2)
                Q = (1 - eta) * Q + eta * update

        Q_nneg = Q + one_over_n

        rmse = np.sqrt(np.mean(Q_nneg[Q_nneg < 0] ** 2))
        max_error = np.abs(np.min(Q_nneg[Q_nneg < 0])) / one_over_n

        if track_stats or verbose:
            obj_value_list.append(np.trace(D.dot(Q)))
            rmse_list.append(rmse)

        if max_error < stop_tol_max and rmse < stop_tol_rmse:
            break

        lagrange_lower_bound += step * Q_nneg
        np.minimum(lagrange_lower_bound, 0, out=lagrange_lower_bound)

        if verbose and t % 10 == 0:
            row_sum = Q.sum(axis=1)
            print('iteration', t, 'Q', -one_over_n, Q.min(), Q.max(), '|',
                  row_sum.min(), row_sum.max(), '|',
                  np.trace(Q), np.trace(D.dot(Q)), '|',
                  eta)

    Q += one_over_n

    if verbose:
        row_sum = np.mean(Q.sum(axis=1))
        print('sum constraint', row_sum.min(), row_sum.max())
        print('trace constraint', np.trace(Q))
        print('nonnegative constraint', np.min(Q), np.mean(np.minimum(Q, 0)))

    print('final objective', np.trace(D.dot(Q)))

    if track_stats:
        return Q, rmse_list, obj_value_list
    else:
        return Q


def copositive_burer_monteiro(X, alpha, beta, rank=None, maxiter=1e3, tol=1e-5,
                              constraint_tol=1e-5, verbose=True):
    if rank is None:
        rank = 8 * beta * len(X)

    X_norm = X - np.mean(X, axis=0)
    if X_norm.shape[0] > X_norm.shape[1]:
        cov = X_norm.T.dot(X_norm)
    else:
        XXt = X_norm.dot(X_norm.T)
        cov = XXt
    X_norm /= np.trace(cov.dot(cov)) ** 0.25

    if X_norm.shape[0] <= X_norm.shape[1]:
        XXt /= np.trace(cov.dot(cov)) ** 0.5

    Y_shape = (len(X), rank)
    ones = np.ones((len(X), 1))

    def lagrangian(x, lagrange_mul, sigma):
        Y = x.reshape(Y_shape)

        if X_norm.shape[0] > X_norm.shape[1]:
            YtX = Y.T.dot(X_norm)
            obj = -np.trace(YtX.dot(YtX.T))
        else:
            obj = -np.trace(Y.T.dot(XXt).dot(Y))

        Yt1 = Y.T.dot(ones)
        obj += alpha * Yt1.T.dot(Yt1)

        diagYtY_minus_beta = np.sum(Y ** 2, axis=1) - beta
        obj += lagrange_mul.dot(diagYtY_minus_beta)
        obj += .5 * sigma * np.sum(np.maximum(diagYtY_minus_beta, 0) ** 2)

        return obj[0, 0]

    def grad(x, lagrange_mul, sigma):
        Y = x.reshape(Y_shape)

        if X_norm.shape[0] > X_norm.shape[1]:
            delta = -2 * X_norm.dot(X_norm.T.dot(Y))
        else:
            delta = -2 * XXt.dot(Y)

        Yt1 = Y.T.dot(ones)
        delta += 2 * alpha * ones.dot(Yt1.T)

        diagYtY_minus_beta = np.maximum(np.sum(Y ** 2, axis=1) - beta, 0)
        delta += 2 * np.diag(lagrange_mul + sigma * diagYtY_minus_beta).dot(Y)

        return delta.flatten()

    if X_norm.shape[0] > X_norm.shape[1]:
        da = DecomposedArray(X_norm, c=-alpha)
        Y = symnmf_admm(da, rank)
    else:
        Y = symnmf_admm(XXt, rank)

    Y /= np.linalg.norm(Y, axis=0, keepdims=True)

    lagrange_mul = np.zeros((len(X),))
    sigma = 1
    step = 1

    error = []
    error_constraint = []
    obj_value = []
    for n_iter in range(int(maxiter)):
        fun = partial(lagrangian, lagrange_mul=lagrange_mul, sigma=sigma)
        jac = partial(grad, lagrange_mul=lagrange_mul, sigma=sigma)
        bounds = [(0, beta)] * np.prod(Y_shape)

        Y_old = Y.copy()
        res = minimize(fun, Y.flatten(), jac=jac, bounds=bounds,
                       method='L-BFGS-B',)
        Y = res.x.reshape(Y_shape)

        norm_diff = np.sum(Y ** 2, axis=1) - beta

        lagrange_mul += step * sigma * norm_diff
        np.maximum(lagrange_mul, 0, out=lagrange_mul)

        norm_const = 1e-16 + np.linalg.norm(np.linalg.norm(Y_old))
        error.append(np.linalg.norm(Y - Y_old) / norm_const)
        error_constraint.append(np.maximum(norm_diff, 0).mean())

        Yt1 = Y.T.dot(ones)
        obj = np.trace(Y.T.dot(X).dot(X.T.dot(Y))) - alpha * Yt1.T.dot(Yt1)
        obj_value.append(obj[0, 0])

        if verbose:
            fmt_str = '{: 4d} {:.5f} | ' \
                      'Y bounds {:.5f} {:.5f} | ' \
                      'beta {:.5f} | ' \
                      'Ynorm - beta {:.5f} {:.5f} | ' \
                      'E {:.10f} {:.10f}'
            print(fmt_str.format(n_iter, X_norm.dot(X_norm.T).max(),
                                 Y.min(), Y.max(),
                                 beta,
                                 norm_diff.min(), norm_diff.max(),
                                 error[-1], error_constraint[-1]
                                 ))

        if error[-1] < tol and error_constraint[-1] < constraint_tol:
            break

    # if verbose:
    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     plt.semilogy(obj_value)

    return Y
