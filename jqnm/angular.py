# -*- coding: utf-8 -*-
"""Solve the angular Teukolsky equation via spectral decomposition.

For a given complex QNM frequency ω, the separation constant and
spherical-spheroidal decomposition are found as an eigenvalue and
eigenvector of an (infinite) matrix problem.  The interface to solving
this problem is :meth:`C_and_sep_const_closest`, which returns a
certain eigenvalue A and eigenvector C.  The eigenvector contains the
C coefficients in the equation

.. math:: {}_s Y_{\\ell m}(\\theta, \\phi; a\\omega) = {\\sum_{\\ell'=\\ell_{\\min} (s,m)}^{\\ell_\\max}} C_{\\ell' \\ell m}(a\\omega)\\ {}_s Y_{\\ell' m}(\\theta, \\phi) \\,.

Here ℓmin=max(\\|m\\|,\\|s\\|) (see :meth:`l_min`), and ℓmax can be chosen at
run time. The C coefficients are returned as a complex ndarray, with
the zeroth element corresponding to ℓmin.  You can get the associated
ℓ values by calling :meth:`ells`.

This module uses JAX for GPU acceleration and automatic differentiation.
"""

from __future__ import division, print_function, absolute_import

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

# TODO some documentation here, better documentation throughout


def _calF(s, l, m):
    """Eq. (52b)"""
    # Handle the case when s=0 and l+1=0
    l_float = jnp.asarray(l, dtype=jnp.float64)
    m_float = jnp.asarray(m, dtype=jnp.float64)
    s_float = jnp.asarray(s, dtype=jnp.float64)

    denom1 = (2.0 * l_float + 3.0) * (2.0 * l_float + 1.0)
    denom2 = (l_float + 1.0) ** 2

    # Avoid division by zero
    safe_denom1 = jnp.where(denom1 == 0.0, 1.0, denom1)
    safe_denom2 = jnp.where(denom2 == 0.0, 1.0, denom2)

    result = jnp.sqrt(
        ((l_float + 1.0) ** 2 - m_float * m_float) / safe_denom1
    ) * jnp.sqrt(((l_float + 1.0) ** 2 - s_float * s_float) / safe_denom2)

    # Return 0 when s=0 and l+1=0
    return jnp.where((s == 0) & (l + 1 == 0), 0.0, result)


def _calG(s, l, m):
    """Eq. (52c)"""
    l_float = jnp.asarray(l, dtype=jnp.float64)
    m_float = jnp.asarray(m, dtype=jnp.float64)
    s_float = jnp.asarray(s, dtype=jnp.float64)

    denom1 = 4.0 * l_float * l_float - 1.0
    denom2 = l_float * l_float

    safe_denom1 = jnp.where(denom1 == 0.0, 1.0, denom1)
    safe_denom2 = jnp.where(denom2 == 0.0, 1.0, denom2)

    result = jnp.sqrt((l_float * l_float - m_float * m_float) / safe_denom1) * jnp.sqrt(
        1.0 - s_float * s_float / safe_denom2
    )

    return jnp.where(l == 0, 0.0, result)


def _calH(s, l, m):
    """Eq. (52d)"""
    l_float = jnp.asarray(l, dtype=jnp.float64)
    m_float = jnp.asarray(m, dtype=jnp.float64)
    s_float = jnp.asarray(s, dtype=jnp.float64)

    denom = l_float * (l_float + 1.0)
    safe_denom = jnp.where(denom == 0.0, 1.0, denom)

    result = -m_float * s_float / safe_denom

    return jnp.where((l == 0) | (s == 0), 0.0, result)


def _calA(s, l, m):
    """Eq. (53a)"""
    return _calF(s, l, m) * _calF(s, l + 1, m)


def _calD(s, l, m):
    """Eq. (53b)"""
    return _calF(s, l, m) * (_calH(s, l + 1, m) + _calH(s, l, m))


def _calB(s, l, m):
    """Eq. (53c)"""
    return (
        _calF(s, l, m) * _calG(s, l + 1, m)
        + _calG(s, l, m) * _calF(s, l - 1, m)
        + _calH(s, l, m) ** 2
    )


def _calE(s, l, m):
    """Eq. (53d)"""
    return _calG(s, l, m) * (_calH(s, l - 1, m) + _calH(s, l, m))


def _calC(s, l, m):
    """Eq. (53e)"""
    return _calG(s, l, m) * _calG(s, l - 1, m)


def swsphericalh_A(s, l, m):
    """Angular separation constant at a=0.

    Eq. (50). Has no dependence on m. The formula is
      A_0 = l(l+1) - s(s+1)

    Parameters
    ----------
    s: int
      Spin-weight of interest

    l: int
      Angular quantum number of interest

    m: int
      Magnetic quantum number, ignored

    Returns
    -------
    int
      Value of A(a=0) = l(l+1) - s(s+1)
    """
    return l * (l + 1) - s * (s + 1)


def M_matrix_elem(s, c, m, l, lprime):
    """The (l, lprime) matrix element from the spherical-spheroidal
    decomposition matrix from Eq. (55).

    Parameters
    ----------
    s: int
      Spin-weight of interest

    c: complex
      Oblateness of the spheroidal harmonic

    m: int
      Magnetic quantum number

    l: int
      Angular quantum number of interest

    lprime: int
      Primed quantum number of interest

    Returns
    -------
    complex
      Matrix element M_{l, lprime}
    """
    c = jnp.asarray(c, dtype=jnp.complex128)

    # Compute all possible values
    val_lm2 = -c * c * _calA(s, lprime, m)
    val_lm1 = -c * c * _calD(s, lprime, m) + 2 * c * s * _calF(s, lprime, m)
    val_l0 = (
        swsphericalh_A(s, lprime, m)
        - c * c * _calB(s, lprime, m)
        + 2 * c * s * _calH(s, lprime, m)
    )
    val_lp1 = -c * c * _calE(s, lprime, m) + 2 * c * s * _calG(s, lprime, m)
    val_lp2 = -c * c * _calC(s, lprime, m)

    # Select the appropriate value based on lprime - l
    diff = lprime - l
    result = jnp.where(
        diff == -2,
        val_lm2,
        jnp.where(
            diff == -1,
            val_lm1,
            jnp.where(
                diff == 0,
                val_l0,
                jnp.where(
                    diff == 1, val_lp1, jnp.where(diff == 2, val_lp2, 0.0 + 0.0j)
                ),
            ),
        ),
    )

    return result


def l_min(s, m):
    """Minimum allowed value of l for a given s, m.

    The formula is l_min = max(|m|,|s|).

    Parameters
    ----------
    s: int
      Spin-weight of interest

    m: int
      Magnetic quantum number

    Returns
    -------
    int
      l_min
    """
    return max(abs(s), abs(m))


def ells(s, m, l_max):
    """Vector of ℓ values in C vector and M matrix.

    The format of the C vector and M matrix is that the 0th element
    corresponds to l_min(s,m) (see :meth:`l_min`).

    Parameters
    ----------
    s: int
      Spin-weight of interest

    m: int
      Magnetic quantum number

    l_max: int
      Maximum angular quantum number

    Returns
    -------
    int ndarray
      Vector of ℓ values, starting from l_min
    """
    return jnp.arange(l_min(s, m), l_max + 1)


def M_matrix(s, c, m, l_max):
    """Spherical-spheroidal decomposition matrix truncated at l_max.

    Parameters
    ----------
    s: int
      Spin-weight of interest

    c: complex
      Oblateness of the spheroidal harmonic

    m: int
      Magnetic quantum number

    l_max: int
      Maximum angular quantum number

    Returns
    -------
    complex ndarray
      Decomposition matrix
    """
    _ells = ells(s, m, l_max)
    n = len(_ells)
    c = jnp.asarray(c, dtype=jnp.complex128)

    # Create index arrays
    l_grid, lp_grid = jnp.meshgrid(_ells, _ells, indexing="ij")

    # Compute all the coefficient arrays for each l value
    # These are the building blocks for the matrix

    # Diagonal (diff = 0)
    A0 = swsphericalh_A(s, lp_grid, m).astype(jnp.complex128)
    calB_vals = jax.vmap(lambda l: _calB(s, l, m))(_ells)
    calH_vals = jax.vmap(lambda l: _calH(s, l, m))(_ells)

    diag_vals = jnp.diag(A0[0] - c * c * calB_vals + 2 * c * s * calH_vals)

    # Build each diagonal band
    M = jnp.zeros((n, n), dtype=jnp.complex128)

    # Diagonal (band 0)
    for i in range(n):
        l = _ells[i]
        val = (
            swsphericalh_A(s, l, m)
            - c * c * _calB(s, l, m)
            + 2 * c * s * _calH(s, l, m)
        )
        M = M.at[i, i].set(val)

    # Super-diagonal +1
    for i in range(n - 1):
        l = _ells[i]
        lprime = _ells[i + 1]
        val = -c * c * _calE(s, lprime, m) + 2 * c * s * _calG(s, lprime, m)
        M = M.at[i, i + 1].set(val)

    # Super-diagonal +2
    for i in range(n - 2):
        lprime = _ells[i + 2]
        val = -c * c * _calC(s, lprime, m)
        M = M.at[i, i + 2].set(val)

    # Sub-diagonal -1
    for i in range(1, n):
        lprime = _ells[i - 1]
        val = -c * c * _calD(s, lprime, m) + 2 * c * s * _calF(s, lprime, m)
        M = M.at[i, i - 1].set(val)

    # Sub-diagonal -2
    for i in range(2, n):
        lprime = _ells[i - 2]
        val = -c * c * _calA(s, lprime, m)
        M = M.at[i, i - 2].set(val)

    return M


@partial(jit, static_argnums=(0, 2, 3))
def sep_consts(s, c, m, l_max):
    """Finds eigenvalues of decomposition matrix, i.e. the separation
    constants, As.

    Parameters
    ----------
    s: int
      Spin-weight of interest

    c: complex
      Oblateness of spheroidal harmonic

    m: int
      Magnetic quantum number

    l_max: int
      Maximum angular quantum number

    Returns
    -------
    complex ndarray
      Eigenvalues of spherical-spheroidal decomposition matrix
    """
    return jnp.linalg.eigvals(M_matrix(s, c, m, l_max))


def sep_const_closest(A0, s, c, m, l_max, temperature=1e-3):
    """Gives the separation constant that is closest to A0.

    Uses a soft-min approach for differentiability.

    Parameters
    ----------
    A0: complex
      Value close to the desired separation constant.

    s: int
      Spin-weight of interest

    c: complex
      Oblateness of spheroidal harmonic

    m: int
      Magnetic quantum number

    l_max: int
      Maximum angular quantum number

    temperature: float
      Temperature for soft-min. Smaller = closer to hard argmin.

    Returns
    -------
    complex
      Separation constant that is the closest to A0.
    """
    As = sep_consts(s, c, m, l_max)
    distances = jnp.abs(As - A0)

    # Use softmin weights for differentiable selection
    # w_i = exp(-d_i / T) / sum(exp(-d_j / T))
    log_weights = -distances / temperature
    # Normalize for numerical stability
    log_weights = log_weights - jnp.max(log_weights)
    weights = jnp.exp(log_weights)
    weights = weights / jnp.sum(weights)

    # Weighted sum of eigenvalues
    A_selected = jnp.sum(weights * As)

    return A_selected


def C_and_sep_const_closest(A0, s, c, m, l_max):
    """Get a single eigenvalue and eigenvector of decomposition
    matrix, where the eigenvalue is closest to some guess A0.

    Parameters
    ----------
    A0: complex
      Value close to the desired separation constant.

    s: int
      Spin-weight of interest

    c: complex
      Oblateness of spheroidal harmonic

    m: int
      Magnetic quantum number

    l_max: int
      Maximum angular quantum number

    Returns
    -------
    complex, complex ndarray
      The first element of the tuple is the eigenvalue that is closest
      in value to A0. The second element of the tuple is the
      corresponding eigenvector.  The 0th element of this ndarray
      corresponds to :meth:`l_min`.
    """
    M = M_matrix(s, c, m, l_max)
    As, Cs = jnp.linalg.eig(M)
    i_closest = jnp.argmin(jnp.abs(As - A0))
    return As[i_closest], Cs[:, i_closest]
