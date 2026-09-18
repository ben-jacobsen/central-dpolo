"""
    File: singular_approximation.py
    Author: Ben Jacobsen
    Purpose: implements methods for computing and bounding the
       finite-horizon sensitivity of Toeplitz factorizations
       in our family.
"""

import cmath
import math

import mpmath as mp
import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import minimize, minimize_scalar

import singular_approximation as sa


def full_norm_sq(gamma, alpha=-1 / 2, delta=0, tail_cutoff=2**10):
    """
    Compute the full, undamped squared l2 norm

        sum over n >= 0 ([z^n] f(z; gamma, delta))^2.

    This implementation is specialized to alpha = -1/2.  It uses the
    endpoint-stable Parseval calculation described in the paper appendix:

      1. integrate the smooth part corresponding to omega in [0, pi/3];
      2. on the singular part, set theta = exp(-s) and integrate up to
         s = tail_cutoff;
      3. replace the remaining tail by its leading asymptotic integral

             integral_S^infinity s^(2 gamma) log(s)^(2 delta) ds.

    The norm is finite exactly when

        gamma < -1/2,

    or when

        gamma = -1/2 and delta < -1/2.

    The tail approximation has relative correction O(tail_cutoff^-2)
    at the integrand level, so changing tail_cutoff is an easy numerical
    stability check.

    Returns mp.inf outside the square-integrable region.
    """
    gamma = mp.mpf(gamma)
    delta = mp.mpf(delta)
    alpha = mp.mpf(alpha)
    tail_cutoff = mp.mpf(tail_cutoff)

    if alpha != -mp.mpf("0.5"):
        raise NotImplementedError(
            "full_norm_sq currently implements the endpoint-stable "
            "formula only for alpha = -1/2")

    if gamma > -mp.mpf("0.5"):
        return mp.inf
    if gamma == -mp.mpf("0.5") and delta >= -mp.mpf("0.5"):
        return mp.inf
    if tail_cutoff <= -mp.log(mp.pi / 3):
        raise ValueError("tail_cutoff is too small")

    # Smooth part: omega in [0, pi/3].
    #
    # These are the I_1,...,I_4 quantities from the appendix.  atan2 is
    # used instead of atan(-omega/log(I1)) so that the endpoint omega=pi/3
    # is handled without division by zero.
    def smooth_integrand(omega):
        I1 = 2 * mp.cos(omega)
        I2 = mp.log(I1)**2 + omega**2
        I3 = mp.mpf("0.25") * mp.log(I2)**2
        I4 = mp.atan2(-omega, mp.log(I1)) + 2 * omega

        return (2 * I1**(-1) * I2**gamma * (I3 + I4**2)**delta)

    smooth = mp.quad(
        smooth_integrand,
        [0, mp.pi / 3],
    )

    # Singular part: theta in (0, pi/3], with theta = exp(-s).
    s0 = -mp.log(mp.pi / 3)

    def singular_integrand(s):
        theta = mp.exp(-s)
        lam = -mp.log(2 * mp.sin(theta / 2))
        omega = (mp.pi - theta) / 2
        L = lam**2 + omega**2

        angle = mp.atan2(omega, lam) - theta
        loglog_term = (mp.mpf("0.25") * mp.log(L)**2 + angle**2)

        return (theta / (2 * mp.sin(theta / 2)) * L**gamma *
                loglog_term**delta)

    # A few breakpoints keep the quadrature transparent and help mpmath
    # on large choices of tail_cutoff without changing the mathematics.
    points = [s0]
    point = mp.mpf(2)
    while point < tail_cutoff:
        if point > s0:
            points.append(point)
        point *= 2
    points.append(tail_cutoff)

    singular = mp.quad(singular_integrand, points)

    # Leading asymptotic tail:
    #
    #   integral_S^infinity s^(2 gamma) log(s)^(2 delta) ds.
    #
    # For gamma < -1/2, let rate = -(1 + 2 gamma) > 0 and substitute
    # u = log s.  For gamma = -1/2, integrate directly.
    if gamma < -mp.mpf("0.5"):
        rate = -(1 + 2 * gamma)
        shape = 1 + 2 * delta
        lower = rate * mp.log(tail_cutoff)

        tail = (mp.gammainc(shape, lower, mp.inf) / rate**shape)
    else:
        tail = (mp.log(tail_cutoff)**(1 + 2 * delta) / (-1 - 2 * delta))

    return mp.power(2, 2 * delta) / mp.pi * (smooth + singular + tail)


def damped_norm_sq(r, gamma, alpha=-1 / 2, delta=0):
    """
    Compute the exponentially damped squared coefficient sum

        sum over n >= 0 r^(2n) ([z^n] f(z; gamma, delta))^2

    for 0 < r < 1.

    By Parseval, this equals

        (1 / 2pi) int_0^(2pi) |f(r exp(i theta))|^2 dtheta.

    Since the Taylor coefficients are real, the integrand is symmetric and
    we integrate over [0, pi].  The breakpoint at pi/3 is not mathematically
    necessary, but is useful numerically when r is close to 1.
    """
    r = mp.mpf(r)
    if not 0 < r < 1:
        raise ValueError("r must lie strictly between 0 and 1")

    def integrand(theta):
        z = r * mp.exp(1j * theta)
        return mp.fabs(sa.singular_function(z, gamma, alpha=alpha,
                                            delta=delta))**2

    return mp.quad(integrand, [0, mp.pi / 3, mp.pi]) / mp.pi


def finite_upper_bound(N, r, gamma, alpha=-1 / 2, delta=0):
    """
    Compute the radial upper bound

        sum over 0 <= n <= N a_n^2
            <= r^(-2N) sum over n >= 0 r^(2n) a_n^2.

    The returned value is the upper bound on the *squared* l2 norm.
    """
    if N < 0:
        raise ValueError("N must be nonnegative")

    r = mp.mpf(r)
    return mp.power(r, -2 * N) * damped_norm_sq(
        r, gamma, alpha=alpha, delta=delta)


def partial_sum(N, gamma, alpha=-1 / 2, delta=0, degree=12):
    """
    Numerically recover the inclusive finite squared coefficient sum

        sum over 0 <= n <= N ([z^n] f(z; gamma, delta))^2.

    Let

        Z(q) = sum_n a_n^2 exp(-q n)
             = damped_norm_sq(exp(-q/2), gamma, delta).

    Then Z(q)/q is the Laplace transform of the cumulative step function

        P(x) = sum_{n <= x} a_n^2.

    We numerically invert this transform with the Stehfest algorithm.

    The inversion is evaluated at x = N + 1/2 rather than x = N because P
    jumps at each integer.  Inverting exactly at an integer returns the
    midpoint across the jump, whereas N + 1/2 lies on the constant plateau
    whose value is the inclusive sum through N.

    degree=12 has worked well empirically for this family.
    """
    if N < 0:
        raise ValueError("N must be nonnegative")
    if degree <= 0 or degree % 2:
        raise ValueError("Stehfest degree must be a positive even integer")

    x = mp.mpf(N) + mp.mpf("0.5")

    def transform(q):
        r = mp.exp(-q / 2)
        return damped_norm_sq(r, gamma, alpha=alpha, delta=delta) / q

    return mp.invertlaplace(
        transform,
        x,
        method="stehfest",
        degree=degree,
    )


def log_finite_upper_bound(N, s, gamma, alpha=-1 / 2, delta=0):
    """
    Compute the logarithm of the radial finite-horizon upper bound.

    Here s = -log(r), so

        U_N(s; gamma, delta)
          = exp(2 N s)
            * ||f(exp(-s) z; gamma, delta)||_2^2.

    Working with the logarithm avoids unnecessary overflow during
    optimization.
    """
    if N < 0:
        raise ValueError("N must be nonnegative")

    s = mp.mpf(s)
    if s <= 0:
        raise ValueError("s must be positive")

    r = mp.exp(-s)
    return (2 * N * s +
            mp.log(damped_norm_sq(r, gamma, alpha=alpha, delta=delta)))


def product_log_upper_bound(
    n_hat,
    N,
    s_L,
    s_R,
    gamma,
    alpha=-1 / 2,
    delta=0,
    tail_cutoff=100,
):
    """
    Logarithm of the product sensitivity bound.

    For finite N,

        log U_n_hat(s_L; -gamma, -delta)
        + log U_N(s_R; gamma, delta).

    For N = infinity,

        log U_n_hat(s_L; -gamma, -delta)
        + log ||f(gamma, delta)||_2^2,

    and s_R is ignored.
    """
    left = log_finite_upper_bound(
        n_hat,
        s_L,
        -gamma,
        alpha=alpha,
        delta=-delta,
    )

    if math.isinf(N):
        right = mp.log(
            full_norm_sq(
                gamma,
                alpha=alpha,
                delta=delta,
                tail_cutoff=tail_cutoff,
            ))
    else:
        right = log_finite_upper_bound(
            N,
            s_R,
            gamma,
            alpha=alpha,
            delta=delta,
        )

    return left + right


def damped_log_norm_moments(
    s,
    gamma,
    alpha=-1 / 2,
    delta=0,
    epsrel=1e-10,
):
    """
    Compute log Z and the first two moments of the exponent features.

    Let

        Z = ||f(exp(-s) z; gamma, delta)||_2^2,

        X(theta) = log |log(1/(1-z)) / z|,

        Y(theta) = log |(2/z) log(log(1/(1-z)) / z)|,

    with z = exp(-s + i theta).  Under the probability measure proportional
    to |f(z; gamma, delta)|^2 dtheta, this returns

        log Z,
        E[(X, Y)],
        Cov(X, Y).

    The six required integrals are evaluated together with scipy.quad_vec.
    This routine is intended for optimization, where ordinary double
    precision is sufficient; final reported sensitivities can still be
    recomputed with damped_norm_sq at arbitrary mpmath precision.
    """
    s = float(s)
    gamma = float(gamma)
    delta = float(delta)
    alpha = float(alpha)

    if s <= 0:
        raise ValueError("s must be positive")

    r = math.exp(-s)
    if r == 1.0:
        raise ValueError(
            "s is too small for the double-precision optimization routine")

    def integrand(theta):
        z = r * complex(math.cos(theta), math.sin(theta))
        one_minus_z = 1 - z
        L = -cmath.log(one_minus_z)
        A = L / z
        B = 2 * cmath.log(A) / z

        X = math.log(abs(A))
        Y = math.log(abs(B))

        log_weight = (2 * alpha * math.log(abs(one_minus_z)) + 2 * gamma * X +
                      2 * delta * Y)
        weight = math.exp(log_weight)

        return weight * np.array([
            1,
            X,
            Y,
            X * X,
            X * Y,
            Y * Y,
        ])

    values = (quad_vec(
        integrand,
        0,
        math.pi / 3,
        epsrel=epsrel,
        epsabs=1e-12,
    )[0] + quad_vec(
        integrand,
        math.pi / 3,
        math.pi,
        epsrel=epsrel,
        epsabs=1e-12,
    )[0]) / math.pi

    Z = values[0]
    mean = values[1:3] / Z

    second_moment = np.array([
        [values[3], values[4]],
        [values[4], values[5]],
    ]) / Z

    covariance = second_moment - np.outer(mean, mean)
    covariance = (covariance + covariance.T) / 2

    return math.log(Z), mean, covariance


def full_log_norm_moments(
    gamma,
    alpha=-1 / 2,
    delta=0,
    tail_cutoff=100,
    epsrel=1e-10,
):
    """
    Compute log Z and exponent-feature moments for the undamped norm.

    This is the optimization analogue of full_norm_sq.  It uses the same
    three-piece endpoint treatment:

      1. the smooth omega integral on [0, pi/3];
      2. theta = exp(-s) up to tail_cutoff;
      3. the leading asymptotic tail beyond tail_cutoff.

    It returns

        log Z,
        E[(X, Y)],
        Cov(X, Y),

    where

        X = log |log(1/(1-z)) / z|,
        Y = log |(2/z) log(log(1/(1-z)) / z)|.

    For Newton optimization we require gamma < -1/2 strictly.  At the
    boundary gamma=-1/2 the norm itself can be finite, but the gamma moments
    needed for derivatives need not be.
    """
    gamma = float(gamma)
    delta = float(delta)
    alpha = float(alpha)
    tail_cutoff = float(tail_cutoff)

    if alpha != -0.5:
        raise NotImplementedError(
            "full_log_norm_moments currently supports only alpha = -1/2")
    if gamma >= -0.5:
        raise ValueError("Newton optimization of the N=inf objective requires "
                         "gamma < -1/2 strictly")
    if tail_cutoff <= -math.log(math.pi / 3):
        raise ValueError("tail_cutoff is too small")

    log2 = math.log(2.0)

    def pack(weight, X, Y):
        return weight * np.array([
            1.0,
            X,
            Y,
            X * X,
            X * Y,
            Y * Y,
        ])

    # Smooth piece in the appendix's omega coordinate.
    def smooth_integrand(omega):
        I1 = 2 * math.cos(omega)
        log_I1 = math.log(I1)
        I2 = log_I1 * log_I1 + omega * omega
        I3 = 0.25 * math.log(I2)**2
        I4 = math.atan2(-omega, log_I1) + 2 * omega
        J = I3 + I4 * I4

        weight = 2 * I1**(-1) * I2**gamma * J**delta

        # |A|^2 = I2 and |B|^2 = 4 J.
        X = 0.5 * math.log(I2)
        Y = log2 + 0.5 * math.log(J)

        return pack(weight, X, Y)

    smooth = quad_vec(
        smooth_integrand,
        0,
        math.pi / 3,
        epsrel=epsrel,
        epsabs=1e-12,
    )[0]

    # Singular piece in theta=exp(-s).
    s0 = -math.log(math.pi / 3)

    def singular_integrand(s):
        theta = math.exp(-s)
        lam = -math.log(2 * math.sin(theta / 2))
        omega = (math.pi - theta) / 2
        L = lam * lam + omega * omega

        angle = math.atan2(omega, lam) - theta
        J = 0.25 * math.log(L)**2 + angle * angle

        weight = (theta / (2 * math.sin(theta / 2)) * L**gamma * J**delta)

        # Again |A|^2 = L and |B|^2 = 4 J.
        X = 0.5 * math.log(L)
        Y = log2 + 0.5 * math.log(J)

        return pack(weight, X, Y)

    singular = quad_vec(
        singular_integrand,
        s0,
        tail_cutoff,
        epsrel=epsrel,
        epsabs=1e-12,
    )[0]

    # Leading asymptotic tail.  Put u=log(s), so
    #
    #   s^(2 gamma) log(s)^(2 delta) ds
    #     = exp((2 gamma + 1)u) u^(2 delta) du.
    #
    # In the same approximation,
    #
    #   X ~ log(s) = u,
    #   Y ~ log(2) + log log(s) = log(2) + log(u).
    u0 = math.log(tail_cutoff)

    def tail_integrand(u):
        weight = math.exp((2 * gamma + 1) * u) * u**(2 * delta)
        X = u
        Y = log2 + math.log(u)
        return pack(weight, X, Y)

    tail = quad_vec(
        tail_integrand,
        u0,
        np.inf,
        epsrel=epsrel,
        epsabs=1e-12,
    )[0]

    values = smooth + singular + tail

    mass = values[0]
    mean = values[1:3] / mass

    second_moment = np.array([
        [values[3], values[4]],
        [values[4], values[5]],
    ]) / mass

    covariance = second_moment - np.outer(mean, mean)
    covariance = (covariance + covariance.T) / 2

    log_Z = 2 * delta * log2 - math.log(math.pi) + math.log(mass)

    return log_Z, mean, covariance


def _exponent_block_derivatives(
    n_hat,
    N,
    s_L,
    s_R,
    gamma,
    delta,
    alpha=-1 / 2,
    tail_cutoff=100,
):
    """
    Value, gradient, and Hessian of the fixed-radius exponent block.

    For finite N,

        J(p) = log U_n_hat(s_L; -p) + log U_N(s_R; p).

    For N = infinity,

        J(p) = log U_n_hat(s_L; -p) + log ||f_R(p)||_2^2,

    and there is no right damping parameter.

    In either case,

        grad J = 2 (E_R[(X,Y)] - E_L[(X,Y)]),

        Hess J = 4 (Cov_R(X,Y) + Cov_L(X,Y)).
    """
    log_Z_L, mean_L, covariance_L = damped_log_norm_moments(
        s_L,
        -gamma,
        alpha=alpha,
        delta=-delta,
    )

    if math.isinf(N):
        log_Z_R, mean_R, covariance_R = full_log_norm_moments(
            gamma,
            alpha=alpha,
            delta=delta,
            tail_cutoff=tail_cutoff,
        )
        value = (2 * n_hat * s_L + log_Z_L + log_Z_R)
    else:
        log_Z_R, mean_R, covariance_R = damped_log_norm_moments(
            s_R,
            gamma,
            alpha=alpha,
            delta=delta,
        )
        value = (2 * n_hat * s_L + log_Z_L + 2 * N * s_R + log_Z_R)

    gradient = 2 * (mean_R - mean_L)
    hessian = 4 * (covariance_R + covariance_L)

    return value, gradient, hessian


def _optimize_exponent_block(
    n_hat,
    N,
    s_L,
    s_R,
    gamma,
    delta,
    alpha,
    gamma_bounds,
    delta_bounds,
    tol,
    max_iter,
    tail_cutoff=100,
):
    """
    Damped Newton minimization of the convex (gamma, delta) block.
    """
    lower = np.array([
        float(gamma_bounds[0]),
        float(delta_bounds[0]),
    ])
    upper = np.array([
        float(gamma_bounds[1]),
        float(delta_bounds[1]),
    ])
    point = np.array([float(gamma), float(delta)])

    for _ in range(max_iter):
        value, gradient, hessian = _exponent_block_derivatives(
            n_hat,
            N,
            s_L,
            s_R,
            point[0],
            point[1],
            alpha=alpha,
            tail_cutoff=tail_cutoff,
        )

        if np.linalg.norm(gradient, ord=np.inf) <= tol:
            break

        # The Hessian is positive semidefinite.  pinv is a convenient,
        # transparent fallback for the nearly-collinear X,Y case.
        direction = -np.linalg.pinv(hessian) @ gradient

        # Numerical roundoff can occasionally spoil descent in a nearly
        # singular direction.  Fall back to steepest descent if necessary.
        if np.dot(gradient, direction) >= 0:
            direction = -gradient

        # Largest step that remains inside the box constraints.
        step = 1.0
        for i in range(2):
            if direction[i] > 0:
                step = min(
                    step,
                    0.99 * (upper[i] - point[i]) / direction[i],
                )
            elif direction[i] < 0:
                step = min(
                    step,
                    0.99 * (lower[i] - point[i]) / direction[i],
                )

        if step <= 0:
            break

        # Armijo backtracking.  A trial evaluation also computes moments,
        # but quad_vec evaluates all six moments in a single quadrature.
        directional_derivative = np.dot(gradient, direction)
        while step > 1e-8:
            trial = point + step * direction
            trial_value, _, _ = _exponent_block_derivatives(
                n_hat,
                N,
                s_L,
                s_R,
                trial[0],
                trial[1],
                alpha=alpha,
                tail_cutoff=tail_cutoff,
            )

            if trial_value <= (value + 1e-4 * step * directional_derivative):
                point = trial
                break

            step /= 2
        else:
            break

        if np.linalg.norm(step * direction, ord=np.inf) <= tol:
            break

    return float(point[0]), float(point[1])


def optimize_parameters(
    n_hat,
    N,
    gamma=-1 / 2,
    delta=0,
    s_L=None,
    s_R=None,
    alpha=-1 / 2,
    gamma_bounds=(-10, 5),
    delta_bounds=(-10, 10),
    log_s_bounds=(-40, 2),
    max_iter=20,
    tol=1e-6,
    scalar_tol=1e-5,
    exponent_tol=1e-7,
    exponent_max_iter=10,
    tail_cutoff=100,
    verbose=False,
):
    """
    Minimize the product radial bound by simple coordinate descent.

    For finite N the optimization variables are

        gamma, delta, s_L, s_R,

    with blocks s_L, s_R, and (gamma, delta).

    For N = infinity, the right factor is evaluated with full_norm_sq and
    there is no s_R block, leaving only s_L and (gamma, delta).

    Radius blocks are solved by bounded one-dimensional minimization.  The
    exponent block is solved jointly by damped Newton iterations using its
    analytic gradient and Hessian.

    The radius variables are optimized in log(s), since the useful scale of
    s can vary by many orders of magnitude when N is large.

    Parameters
    ----------
    n_hat:
        Finite horizon used for the left factor.
    N:
        Maximum horizon used for the right factor.
    gamma, delta:
        Initial exponents.
    s_L, s_R:
        Initial damping parameters.  For N=infinity, s_R is ignored and the
        returned value of s_R is None.  Otherwise both are reoptimized during
        each sweep.
    gamma_bounds, delta_bounds:
        Search intervals for the exponent coordinates.  If an optimum lands
        on a boundary, rerun with a wider interval.
    log_s_bounds:
        Search interval for log(s).  The default corresponds roughly to
        4e-18 <= s <= 7.4.
    max_iter:
        Maximum number of coordinate-descent sweeps.
    tol:
        Stop when the decrease in the log objective over a full sweep is at
        most this value.
    scalar_tol:
        Absolute tolerance passed to each bounded scalar minimization.
    exponent_tol:
        Infinity-norm stopping tolerance for the exponent gradient and step.
    exponent_max_iter:
        Maximum number of Newton iterations within each exponent block.
    tail_cutoff:
        Endpoint cutoff used by the undamped right norm when N=infinity.
    verbose:
        Print one line after each complete sweep.

    Returns
    -------
    dict
        Contains the optimized parameters, the final bound, convergence
        information, and the full iteration history.

    Notes
    -----
    For fixed gamma, delta the s_L and s_R subproblems are convex.  For
    fixed s_L, s_R the exponent block is jointly convex in gamma, delta, with
    gradient and Hessian given by feature means and covariances.
    """
    if n_hat < 0 or N < 0:
        raise ValueError("n_hat and N must be nonnegative")
    infinite_N = math.isinf(N)
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol < 0 or scalar_tol <= 0 or exponent_tol <= 0:
        raise ValueError("invalid optimization tolerance")
    if exponent_max_iter <= 0:
        raise ValueError("exponent_max_iter must be positive")

    gamma_lo, gamma_hi = map(float, gamma_bounds)
    delta_lo, delta_hi = map(float, delta_bounds)
    log_s_lo, log_s_hi = map(float, log_s_bounds)

    if not gamma_lo < gamma_hi:
        raise ValueError("gamma_bounds must be increasing")
    if not delta_lo < delta_hi:
        raise ValueError("delta_bounds must be increasing")
    if not log_s_lo < log_s_hi:
        raise ValueError("log_s_bounds must be increasing")

    if infinite_N:
        # The full right norm requires gamma <= -1/2.  The Newton derivatives
        # require a strict interior point, so keep the optimization a tiny
        # distance inside the square-integrable region.
        gamma_hi = min(gamma_hi, -0.500001)
        if not gamma_lo < gamma_hi:
            raise ValueError(
                "for N=infinity, gamma_bounds must include values below -1/2")

    gamma = float(gamma)
    delta = float(delta)

    if infinite_N and gamma >= gamma_hi:
        gamma = min(-0.51, (gamma_lo + gamma_hi) / 2)

    if s_L is None:
        s_L = 1 / (2 * (n_hat + 1))
    s_L = float(s_L)

    if infinite_N:
        s_R = None
    else:
        if s_R is None:
            s_R = 1 / (2 * (N + 1))
        s_R = float(s_R)

    if s_L <= 0:
        raise ValueError("s_L must be positive")
    if not infinite_N and s_R <= 0:
        raise ValueError("s_R must be positive")

    def objective(s_left, s_right, g, d):
        return float(
            product_log_upper_bound(
                n_hat,
                N,
                s_left,
                s_right,
                g,
                alpha=alpha,
                delta=d,
                tail_cutoff=tail_cutoff,
            ))

    def optimize_s(cutoff, g, d):

        def objective_log_s(log_s):
            s = math.exp(log_s)
            return float(
                log_finite_upper_bound(
                    cutoff,
                    s,
                    g,
                    alpha=alpha,
                    delta=d,
                ))

        result = minimize_scalar(
            objective_log_s,
            bounds=(log_s_lo, log_s_hi),
            method="bounded",
            options={"xatol": scalar_tol},
        )
        return math.exp(result.x)

    history = []

    current = objective(s_L, s_R, gamma, delta)
    history.append({
        "iteration": 0,
        "gamma": gamma,
        "delta": delta,
        "s_L": s_L,
        "s_R": s_R,
        "log_bound": current,
    })

    converged = False

    for iteration in range(1, max_iter + 1):
        previous = current

        # Left radius: f_L = f(-gamma, -delta).
        s_L = optimize_s(
            n_hat,
            -gamma,
            -delta,
        )

        # Right radius exists only for finite N.
        if not infinite_N:
            s_R = optimize_s(
                N,
                gamma,
                delta,
            )

        # Exponent block: analytic damped Newton.
        gamma, delta = _optimize_exponent_block(
            n_hat,
            N,
            s_L,
            s_R,
            gamma,
            delta,
            alpha,
            (gamma_lo, gamma_hi),
            (delta_lo, delta_hi),
            exponent_tol,
            exponent_max_iter,
            tail_cutoff=tail_cutoff,
        )

        current = objective(s_L, s_R, gamma, delta)

        history.append({
            "iteration": iteration,
            "gamma": gamma,
            "delta": delta,
            "s_L": s_L,
            "s_R": s_R,
            "log_bound": current,
        })

        if verbose:
            if infinite_N:
                print(f"{iteration:2d}: "
                      f"gamma={gamma:.8g}, "
                      f"delta={delta:.8g}, "
                      f"s_L={s_L:.8g}, "
                      f"log_bound={current:.12g}")
            else:
                print(f"{iteration:2d}: "
                      f"gamma={gamma:.8g}, "
                      f"delta={delta:.8g}, "
                      f"s_L={s_L:.8g}, "
                      f"s_R={s_R:.8g}, "
                      f"log_bound={current:.12g}")

        improvement = previous - current
        if 0 <= improvement <= tol:
            converged = True
            break

    return {
        "gamma": gamma,
        "delta": delta,
        "s_L": s_L,
        "s_R": s_R,
        "r_L": mp.exp(-s_L),
        "r_R": None if infinite_N else mp.exp(-s_R),
        "log_bound": mp.mpf(current),
        "bound": mp.exp(current),
        "iterations": len(history) - 1,
        "converged": converged,
        "history": history,
    }
