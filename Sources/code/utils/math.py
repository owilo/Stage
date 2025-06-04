import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.optimize import brentq

def ros_transform(z, pis, mus, covs):
    # rosenblatt transform
    K, d = mus.shape
    u = np.zeros(d)

    # extract means and std for the 1st dim
    m1 = mus[:, 0]
    s1 = np.sqrt(covs[:, 0, 0])
    # mixture CDF at z[0]
    u[0] = np.sum(pis * norm.cdf(z[0], loc=m1, scale=s1))

    x_prev = np.zeros(d)
    x_prev[0] = z[0]
    for i in range(1, d):
        # compute posterior weights w_k proportional to pis * N(z[0:i] | mu[0:i], Sigma[0:i,0:i])
        log_weights = np.zeros(K)
        for k in range(K):
            mu_k = mus[k, :i]
            cov_k = covs[k, :i, :i]
            log_weights[k] = np.log(pis[k]) + multivariate_normal.logpdf(z[:i], mean=mu_k, cov=cov_k)
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= np.sum(weights)

        # conditional parameters for each component
        m_cond = np.zeros(K)
        s_cond = np.zeros(K)
        for k in range(K):
            mu_k = mus[k]
            Sigma_k = covs[k]
            mu_1 = mu_k[:i]
            mu_2 = mu_k[i]
            S11 = Sigma_k[:i, :i]
            S12 = Sigma_k[:i, i]
            S22 = Sigma_k[i, i]
            invS11 = np.linalg.inv(S11)
            m_cond[k] = mu_2 + S12.T.dot(invS11).dot(z[:i] - mu_1)
            s_cond[k] = np.sqrt(S22 - S12.T.dot(invS11).dot(S12))

        # mixture conditional CDF
        cdf_cond = lambda x: np.sum(weights * norm.cdf(x, loc=m_cond, scale=s_cond))
        u[i] = cdf_cond(z[i])
        x_prev[i] = z[i]

    return u

def ros_inverse(u, pis, mus, covs, bracket=(-10, 10), tol=1e-6):
    K, d = mus.shape
    z = np.zeros(d)

    m1 = mus[:, 0]
    s1 = np.sqrt(covs[:, 0, 0])
    mix_cdf1 = lambda x: np.sum(pis * norm.cdf(x, loc=m1, scale=s1)) - u[0]
    z[0] = brentq(mix_cdf1, bracket[0], bracket[1], xtol=tol)

    for i in range(1, d):
        # precompute posterior weights given z[:i]
        log_weights = np.zeros(K)
        for k in range(K):
            mu_k = mus[k, :i]
            cov_k = covs[k, :i, :i]
            log_weights[k] = np.log(pis[k]) + multivariate_normal.logpdf(z[:i], mean=mu_k, cov=cov_k)
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= np.sum(weights)

        # conditional params
        m_cond = np.zeros(K)
        s_cond = np.zeros(K)
        for k in range(K):
            mu_k = mus[k]
            Sigma_k = covs[k]
            mu_1 = mu_k[:i]
            mu_2 = mu_k[i]
            S11 = Sigma_k[:i, :i]
            S12 = Sigma_k[:i, i]
            S22 = Sigma_k[i, i]
            invS11 = np.linalg.inv(S11)
            m_cond[k] = mu_2 + S12.T.dot(invS11).dot(z[:i] - mu_1)
            s_cond[k] = np.sqrt(S22 - S12.T.dot(invS11).dot(S12))

        # define conditional CDF minus u[i]
        func = lambda x: np.sum(weights * norm.cdf(x, loc=m_cond, scale=s_cond)) - u[i]
        z[i] = brentq(func, bracket[0], bracket[1], xtol=tol)

    return z
