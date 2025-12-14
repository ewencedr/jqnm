"""Analytic approximations for Schwarzschild QNMs.

The approximations implemented in this module can be used as initial
guesses when numerically searching for QNM frequencies.

This module uses JAX for GPU acceleration and automatic differentiation.
"""

from __future__ import division, print_function, absolute_import

import numpy as np


def dolan_ottewill_expansion(s, l, n):
    """High l asymptotic expansion of Schwarzschild QNM frequency.

    The result of [1]_ is an expansion in inverse powers of L =
    (l+1/2). Their paper stated this series out to L^{-4}, which is
    how many terms are implemented here. The coefficients in this
    series are themselves positive powers of N = (n+1/2). This means
    the expansion breaks down for large N.

    Parameters
    ----------
    s: int
      Spin weight of the field of interest.

    l: int
      Multipole number of interest.

    [The m parameter is omitted because this is just for Schwarzschild.]

    n: int
      Overtone number of interest.

    Returns
    -------
    complex
      Analytic approximation of QNM of interest.

    References
    ----------
    .. [1] SR Dolan, AC Ottewill, "On an expansion method for black
       hole quasinormal modes and Regge poles," CQG 26 225003 (2009),
       https://arxiv.org/abs/0908.0329 .
    """

    L = l + 0.5
    N = n + 0.5
    beta = 1.0 - s * s

    DO_coeffs = {
        -1: 1.0,
        0: -1.0j * N,
        +1: beta / 3.0 - 5.0 * N * N / 36.0 - 115.0 / 432.0,
        +2: -1.0j * N * (beta / 9.0 + 235.0 * N * N / 3888.0 - 1415.0 / 15552.0),
        +3: (
            -beta * beta / 27.0
            + (204.0 * N * N + 211.0) / 3888.0 * beta
            + (854160.0 * N**4 - 1664760.0 * N * N - 776939.0) / 40310784.0
        ),
        +4: -1.0j
        * N
        * (
            beta * beta / 27.0
            + (1100.0 * N * N - 2719.0) / 46656.0 * beta
            + (11273136.0 * N**4 - 52753800.0 * N * N + 66480535.0) / 2902376448.0
        ),
    }

    omega = 0.0 + 0.0j
    for k, c in DO_coeffs.items():
        omega = omega + c * np.power(L, -k)

    omega = omega / np.sqrt(27)

    return omega


def large_overtone_expansion(s, l, n):
    r"""The eikonal approximation for QNMs, valid for l >> n >> 1 .

    This is just the first two terms of the series in
    :meth:`dolan_ottewill_expansion`.
    The earliest work I know deriving this result is [1]_ but there
    may be others. In the eikonal approximation, valid when
    :math:`l \gg n \gg 1`, the QNM frequency is

    .. math:: \sqrt{27} M \omega \approx (l+\frac{1}{2}) - i (n+\frac{1}{2}) .

    Parameters
    ----------
    s: int
      Spin weight of the field of interest.

    l: int
      Multipole number of interest.

    [The m parameter is omitted because this is just for Schwarzschild.]

    n: int
      Overtone number of interest.

    Returns
    -------
    complex
      Analytic approximation of QNM of interest.

    References
    ----------
    .. [1] V Ferrari, B Mashhoon, "New approach to the quasinormal
       modes of a black hole," Phys. Rev. D 30, 295 (1984)

    """

    k = np.log(3.0) / (8.0 * np.pi)
    kappa = 0.25  # Surface gravity

    return k - 1.0j * kappa * (n + 0.5)


def Schw_QNM_estimate(s, l, n):
    """Give either :meth:`large_overtone_expansion` or :meth:`dolan_ottewill_expansion`.

    The Dolan-Ottewill expansion includes terms with higher powers of
    the overtone number n, so it breaks down faster at high n.

    Parameters
    ----------
    s: int
      Spin weight of the field of interest.

    l: int
      Multipole number of interest.

    [The m parameter is omitted because this is just for Schwarzschild.]

    n: int
      Overtone number of interest.

    Returns
    -------
    complex
      Analytic approximation of QNM of interest.

    """

    if (n > 3) and (n >= 2 * l):
        return large_overtone_expansion(s, l, n)
    else:
        return dolan_ottewill_expansion(s, l, n)
