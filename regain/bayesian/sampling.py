"""Sampling module for Wishart processes."""
from functools import partial

import numpy as np
from scipy import stats

from regain.bayesian.stats import lognstat


def GWP_construct(umat, L, uut=None):
    """Build the sample from the GWP.

    Optimised with uut:
    uut = np.array([u.dot(u.T) for u in umat.T])
    """
    if uut is None:
        v, p, n = umat.shape
        M = np.zeros((p, p, n))
        for i in range(n):
            for j in range(v):
                Lu = L.dot(umat[j, :, i])
                LuuL = Lu[:, None].dot(Lu[None, :])
                M[..., i] += LuuL

    else:
        M = np.array([np.linalg.multi_dot((L, uu_i, L.T))
                      for uu_i in uut]).transpose()

    # assert np.allclose(N, M)
    return M


def elliptical_slice(
        current_state, prior, likelihood=None, angle_range=0, max_iter=20):
    """Markov chain update for a distribution with a Gaussian "prior" factored out.

    A Markov chain update is applied to the D-element array xx leaving a
    "posterior" distribution
        P(xx) \propto N(xx0,Sigma) \ell(xx)
    invariant. Where N(0,Sigma) is a zero-mean Gaussian distribution with
    covariance Sigma. Often \ell is a likelihood function.

    Parameters
    ----------
    current_state : Bunch object
        Current state.
    prior :  array-like, shape (D,)
        Single sample from N(0, Sigma)
    angle_range : float, default 0
        Explore whole ellipse with break point at first rejection.
        Set in (0,2*pi] to explore a bracket of the specified width
        centred uniformly at random.

    Returns:
    --------
    current_state : Bunch object, including np.ndarray, shape (D,)
        Perturbed vector plus other info (as log likelihood of the state).

    Originally written in matlab by Iain Murray
    http://homepages.inf.ed.ac.uk/imurray2/pub/10ess/elliptical_slice.m
    Iain Murray, September 2009
    Tweak to interface and documentation, September 2010

    Reference
    ---------
    Elliptical slice sampling
    Iain Murray, Ryan Prescott Adams and David J.C. MacKay.
    The Proceedings of the 13th International Conference on Artificial
    Intelligence and Statistics (AISTATS), JMLR W&CP 9:541-548, 2010.
    """
    if likelihood is None:
        raise ValueError(
            "`likelihood` parameter is None, should be a "
            "function to evaluate likelihood")
    initial_theta = current_state.xx
    v, p, N = initial_theta.shape
    D = v * p * N

    L = current_state.L

    start_logp = current_state.log_likelihood
    if start_logp is None:
        # cur_log_like = log_lik_frob(S, xx.V, variance)
        # cur_log_like = time_multivariate_normal_logpdf(initial_theta, xx.V)
        start_logp = likelihood(xx.V)
    current_logp = start_logp

    # Set up the ellipse and the slice threshold
    if prior.size == D:
        #  User provided a prior sample:
        nu = prior
    else:
        #  User specified Cholesky of prior covariance:
        if prior.shape != (D, D):
            raise ValueError(
                'Prior must be given by a D-element sample '
                'or DxD chol(Sigma)')
        nu = np.reshape(
            prior.T.dot(np.random.normal(size=D)), initial_theta.shape)

    hh = 0.001 * np.log(np.random.uniform()) + current_logp

    #  Set up a bracket of angles and pick a first proposal.
    #  "phi = (theta'-theta)" is a change in angle.
    if angle_range <= 0:
        #  Bracket whole ellipse with both edges at first proposed point
        phi = np.random.uniform() * 2 * np.pi
        phi_min = phi - 2 * np.pi
        phi_max = phi
    else:
        #  Randomly center bracket on current point
        phi_min = -angle_range * np.random.uniform()
        phi_max = phi_min + angle_range
        phi = np.random.uniform() * (phi_max - phi_min) + phi_min

    # Slice sampling loop
    update_state = True
    # LLt = L.dot(L.T)
    for iteration_ in range(max_iter):
        # Compute xx for proposed angle difference and check if on the slice
        proposal = np.real(initial_theta * np.cos(phi) + nu * np.sin(phi))
        uut = np.array([u.dot(u.T) for u in proposal.T])
        V = GWP_construct(proposal, L, uut=uut)
        # cur_log_like = log_lik_frob(S, V, variance)
        # cur_log_like = stats.wishart.logpdf(V, nu, LLt)
        # cur_log_like = time_multivariate_normal_logpdf(xx_proposal, V)
        current_logp = likelihood(V)

        if current_logp > hh:
            # New point is on slice, ** EXIT LOOP **
            break

        # Shrink slice to rejected point
        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            # error = True
            # break
            raise RuntimeError(
                'BUG DETECTED: Shrunk to current position '
                'and still not acceptable.')

        # Propose new angle difference
        phi = np.random.uniform() * (phi_max - phi_min) + phi_min
    else:
        update_state = False

    if update_state:
        # update with new point
        current_state['uut'] = uut
        current_state['xx'] = proposal
        current_state['V'] = V
        current_state['log_likelihood'] = current_logp

    return current_state


def sample_hyper_kernel(
        initial_theta, var_proposal, ustack, kern, prior_distr):
    """Metropolis-Hastings for sampling the posterior of the kernel
    hyperparameter.

    According to the paper, we use a lognormal distribution as the proposal.

    Parameters
    ----------
    initial_theta : type
        Initial kernel hyperparameter.
    var_proposal : type
        Variance for the proposal.

    """
    # Propose a sample
    mu, sigma = lognstat(initial_theta, var_proposal)
    proposal = np.random.lognormal(mu, sigma)
    # log_qzastztau = lognormal_logpdf(proposal, mu=mu, sigma=sigma)
    log_qzastztau = stats.lognorm.logpdf(
        proposal, loc=0, s=sigma, scale=np.exp(mu))

    # Criterion to choose whether to accept the proposed sample or not
    logp_post = partial(
        posterior_iw, ustack=ustack, kern=kern, prior_distr=prior_distr)
    logp_diff = logp_post(proposal) - logp_post(initial_theta)

    mu, sigma = lognstat(proposal, var_proposal)
    # log_qztauzast = lognormal_logpdf(initial_theta, mu=mu, sigma=sigma)
    log_qztauzast = stats.lognorm.logpdf(
        initial_theta, loc=0, s=sigma, scale=np.exp(mu))

    # Now we decide whether to accept zast or use the previous value
    log_acceptance_proba = min(0, logp_diff + log_qztauzast - log_qzastztau)
    accept = np.log(np.random.uniform()) < log_acceptance_proba

    sample = proposal if accept else initial_theta
    return sample, accept


def posterior_iw(inverse_width, ustack, kern, prior_distr):
    """Posterior probability of inverse_width.

    Parameters
    ----------
    inverse_width : float
        Kernel parameter.
    ustack : ndarray, shape (v*p, n)
        Sample tensor.
    kern : function
        Function for computing the kernel.
    prior_distr : rv_frozen
        Prior for inverse_width.

    Returns
    -------
    logprob
        Posterior probability.
    """
    K = kern(inverse_width=inverse_width)

    # k_inverse = linalg.pinvh(K)
    # v, p, _ = u.shape
    # F = np.tensordot(u, u, axes=([1, 0], [1, 0]))
    # logpugl = v * p * fast_logdet(k_inverse) - np.sum(F * k_inverse)
    # logpugl -= v * p * n * np.log(2 * np.pi)
    # logpugl /= 2.

    # creates a (vp * n) matrix, then compute centered empirical covariance
    # ustack = np.vstack(u)
    # logpugl = ustack.shape[0] * log_likelihood(
    #     empirical_covariance(ustack), k_inverse)
    # this is equivalent to
    logp = stats.multivariate_normal(
        ustack.mean(axis=0), cov=K, allow_singular=True).logpdf(ustack).sum()
    # with centering the samples (but there is no need for k_inverse)

    logp_prior = prior_distr.logpdf(inverse_width)
    return logp + logp_prior


def sample_ell(
        Ltau, var_proposal, umat, prior_distr, uut=None, likelihood=None):
    """Metropolis-Hastings for sampling the posterior of the elements in L.

    Use a spherical normal distribution as the proposal.
    """
    # Run the MH individually per component of L
    free_elements = Ltau.size
    L_proposal = np.zeros(free_elements)

    if not isinstance(var_proposal, np.ndarray):
        var_proposal = var_proposal * np.ones(free_elements)
    sigma_proposal = np.sqrt(var_proposal)

    def get_logp(ell_lower):
        v, p, _ = umat.shape
        ell = np.zeros((p, p))
        ell[np.tril_indices_from(ell)] = ell_lower
        D = GWP_construct(umat, ell, uut=uut)
        # logpS = log_lik_frob(S, D, var_err)
        logp = likelihood(D)
        return logp

    for i in range(free_elements):
        L_proposal[i] = _sample_ell_comp(
            Ltau, i, sigma_proposal=sigma_proposal[i], prior_distr=prior_distr,
            likelihood=get_logp)
        Ltau[i] = L_proposal[i]

    return L_proposal


def _sample_ell_comp(Ltaug, i, sigma_proposal, prior_distr, likelihood=None):
    """Sample a single element for L."""
    if likelihood is None:
        raise ValueError(
            "`likelihood` parameter is None, should be a "
            "function to evaluate likelihood")
    # Propose a sample
    Ltau = Ltaug[i]
    Last = np.random.normal(Ltau, sigma_proposal)
    Lastg = Ltaug.copy()
    Lastg[i] = Last

    # Criterion to choose whether to accept the proposed sample or not
    logp_post = partial(
        logp_ell_posterior, i=i, prior_distr=prior_distr,
        likelihood=likelihood)

    logp_diff = logp_post(Lastg) - logp_post(Ltaug)
    logq_ast_tau = stats.norm.logpdf(Last, Ltau, sigma_proposal)
    logq_tau_ast = stats.norm.logpdf(Ltau, Last, sigma_proposal)
    logq_diff = logq_tau_ast - logq_ast_tau

    # Now we decide whether to accept zast or use the previous value
    log_acceptance_proba = min(0, logp_diff + logq_diff)
    accept = np.log(np.random.uniform()) < log_acceptance_proba

    sample = Last if accept else Ltau
    return sample


def logp_ell_posterior(ell_lower, i, prior_distr, likelihood):
    """Log-probability of the posterior of L.

    Parameters
    ----------
    ell_lower : ndarray
        Lower Cholesky.
    i : type
        Index.
    prior_distr : rv_frozen
    likelihood : function
        Likelihood of ell_lower.

    Returns
    -------
    type
        Description of returned object.

    """
    # v, p, n = u.shape
    # ell = np.zeros((p, p))
    # ell[np.tril_indices_from(ell)] = ell_lower
    # D = GWP_construct(u, ell, uut=uut)
    # # logpS = log_lik_frob(S, D, var_err)
    # logp = likelihood(D)
    logp = likelihood(ell_lower)
    prior = prior_distr.logpdf(ell_lower[i])
    return logp + prior
