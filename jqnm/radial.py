"""Solve the radial Teukolsky equation via Leaver's method.

This module uses JAX for GPU acceleration and automatic differentiation.
"""

from __future__ import division, print_function, absolute_import

import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial

from .contfrac import lentz


def sing_pt_char_exps(omega, a, s, m):
    r"""Compute the three characteristic exponents of the singular points
    of the radial Teukolsky equation.
    """
    omega = jnp.asarray(omega, dtype=jnp.complex128)
    a = jnp.asarray(a, dtype=jnp.float64)
    s = jnp.asarray(s, dtype=jnp.float64)
    m = jnp.asarray(m, dtype=jnp.float64)

    root = jnp.sqrt(1.0 - a * a)
    r_p, r_m = 1.0 + root, 1.0 - root

    sigma_p = (2.0 * omega * r_p - m * a) / (2.0 * root)
    sigma_m = (2.0 * omega * r_m - m * a) / (2.0 * root)

    zeta = +1.0j * omega
    xi = -s - 1.0j * sigma_p
    eta = -1.0j * sigma_m

    return zeta, xi, eta


def D_coeffs(omega, a, s, m, A):
    """The D_0 through D_4 coefficients for the radial continued fraction."""
    omega = jnp.asarray(omega, dtype=jnp.complex128)
    A = jnp.asarray(A, dtype=jnp.complex128)

    zeta, xi, eta = sing_pt_char_exps(omega, a, s, m)

    root = jnp.sqrt(1.0 - a * a)

    p = root * zeta
    alpha = 1.0 + s + xi + eta - 2.0 * zeta + s
    gamma = 1.0 + s + 2.0 * eta
    delta = 1.0 + s + 2.0 * xi
    sigma = (
        A
        + a * a * omega * omega
        - 8.0 * omega * omega
        + p * (2.0 * alpha + gamma - delta)
        + (1.0 + s - 0.5 * (gamma + delta)) * (s + 0.5 * (gamma + delta))
    )

    D0 = delta
    D1 = 4.0 * p - 2.0 * alpha + gamma - delta - 2.0
    D2 = 2.0 * alpha - gamma + 2.0
    D3 = alpha * (4.0 * p - delta) - sigma
    D4 = alpha * (alpha - gamma + 1.0)

    return jnp.array([D0, D1, D2, D3, D4])


def leaver_cf_trunc_inversion(omega, a, s, m, A, n_inv, N=300, r_N=1.0):
    """Legacy function for truncated continued fraction."""
    n = jnp.arange(0, N + 1)

    D = D_coeffs(omega, a, s, m, A)

    alpha = n * n + (D[0] + 1.0) * n + D[0]
    beta = -2.0 * n * n + (D[1] + 2.0) * n + D[3]
    gamma = n * n + (D[2] - 3.0) * n + D[4] - D[2] + 2.0

    def forward_step(conv, i):
        return alpha[i] / (beta[i] - gamma[i] * conv), None

    conv1, _ = lax.scan(forward_step, 0.0 + 0.0j, jnp.arange(n_inv))

    def backward_step(conv, i):
        return gamma[i] / (beta[i] - alpha[i] * conv), None

    conv2, _ = lax.scan(backward_step, -r_N + 0.0j, jnp.arange(N, n_inv, -1))

    return beta[n_inv] - gamma[n_inv] * conv1 - alpha[n_inv] * conv2


def rad_a(i, n_inv, D):
    """Compute a_i for continued fraction."""
    n = i + n_inv - 1
    return -(n * n + (D[0] + 1.0) * n + D[0]) / (
        n * n + (D[2] - 3.0) * n + D[4] - D[2] + 2.0
    )


def rad_b(i, n_inv, D):
    """Compute b_i for continued fraction."""
    n = i + n_inv
    result = (-2.0 * n * n + (D[1] + 2.0) * n + D[3]) / (
        n * n + (D[2] - 3.0) * n + D[4] - D[2] + 2.0
    )
    return jnp.where(i == 0, 0.0 + 0.0j, result)


def leaver_cf_inv_lentz_old(
    omega, a, s, m, A, n_inv, tol=1.0e-10, N_min=0, N_max=10000
):
    """Legacy function using Python loops (not fully JIT-compatible)."""
    D = D_coeffs(omega, a, s, m, A)

    n = jnp.arange(0, n_inv + 1)
    alpha = n * n + (D[0] + 1.0) * n + D[0]
    beta = -2.0 * n * n + (D[1] + 2.0) * n + D[3]
    gamma = n * n + (D[2] - 3.0) * n + D[4] - D[2] + 2.0

    conv1 = 0.0 + 0.0j
    for i in range(n_inv):
        conv1 = alpha[i] / (beta[i] - gamma[i] * conv1)

    conv2, cf_err, n_frac = lentz(
        lambda i, n_inv, D: rad_a(i, n_inv, D),
        lambda i, n_inv, D: rad_b(i, n_inv, D),
        args=(n_inv, D),
        tol=tol,
        N_min=N_min,
        N_max=N_max,
    )

    return (beta[n_inv] - gamma[n_inv] * conv1 + gamma[n_inv] * conv2), cf_err, n_frac


@partial(jit, static_argnums=(5, 8))
def leaver_cf_inv_lentz(omega, a, s, m, A, n_inv, tol=1.0e-10, N_min=0, N_max=500):
    """Compute the n_inv inversion of the infinite continued fraction.

    This is a fully differentiable version using lax.scan with fixed iterations.
    Convergence is checked but we always run N_max iterations for differentiability.

    Parameters
    ----------
    omega: complex
      The complex frequency.
    a: float
      Spin parameter of the black hole.
    s: int
      Spin weight of the field.
    m: int
      Azimuthal number.
    A: complex
      Separation constant.
    n_inv: int
      Inversion number (static).
    tol: float
      Tolerance (used for error reporting).
    N_min: int
      Minimum iterations (unused, kept for compatibility).
    N_max: int
      Number of iterations to run (static for JIT).

    Returns
    -------
    (complex, float, int)
      (continued fraction value, error estimate, number of iterations)
    """
    omega = jnp.asarray(omega, dtype=jnp.complex128)
    A = jnp.asarray(A, dtype=jnp.complex128)

    D = D_coeffs(omega, a, s, m, A)

    # Compute alpha, beta, gamma for n = 0 to n_inv
    n_arr = jnp.arange(0, n_inv + 1)
    alpha_arr = n_arr * n_arr + (D[0] + 1.0) * n_arr + D[0]
    beta_arr = -2.0 * n_arr * n_arr + (D[1] + 2.0) * n_arr + D[3]
    gamma_arr = n_arr * n_arr + (D[2] - 3.0) * n_arr + D[4] - D[2] + 2.0

    # Forward recursion for conv1 using lax.fori_loop with static bounds
    def forward_body(i, conv):
        return alpha_arr[i] / (beta_arr[i] - gamma_arr[i] * conv)

    conv1 = lax.fori_loop(0, n_inv, forward_body, 0.0 + 0.0j)

    # Lentz's method using lax.scan (differentiable)
    tiny = 1.0e-30 + 0.0j

    def lentz_step(carry, _):
        f, C, D_val, n_val, Delta = carry

        # Compute a_n and b_n
        an = -(n_val * n_val + (D[0] + 1.0) * n_val + D[0]) / (
            n_val * n_val + (D[2] - 3.0) * n_val + D[4] - D[2] + 2.0
        )
        n_next = n_val + 1.0
        bn = (-2.0 * n_next * n_next + (D[1] + 2.0) * n_next + D[3]) / (
            n_next * n_next + (D[2] - 3.0) * n_next + D[4] - D[2] + 2.0
        )

        D_new = bn + an * D_val
        D_new = jnp.where(jnp.abs(D_new) < 1e-30, tiny, D_new)

        C_new = bn + an / C
        C_new = jnp.where(jnp.abs(C_new) < 1e-30, tiny, C_new)

        D_new = 1.0 / D_new
        Delta_new = C_new * D_new
        f_new = f * Delta_new

        return (f_new, C_new, D_new, n_next, Delta_new), Delta_new

    # Initial state
    init_carry = (
        tiny,  # f
        tiny,  # C
        0.0 + 0.0j,  # D
        float(n_inv),  # n_val
        1.0 + 0.0j,  # Delta
    )

    final_carry, deltas = lax.scan(lentz_step, init_carry, None, length=N_max)
    f_final, _, _, _, Delta_final = final_carry

    conv2 = f_final
    cf_err = jnp.abs(Delta_final - 1.0)

    result = beta_arr[n_inv] - gamma_arr[n_inv] * conv1 + gamma_arr[n_inv] * conv2

    return result, cf_err, N_max
