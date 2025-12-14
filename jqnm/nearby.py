"""Find a nearby root of the coupled radial/angular Teukolsky equations.

This module uses JAX for GPU acceleration and automatic differentiation.
Uses optimistix for differentiable root-finding.
"""

from __future__ import division, print_function, absolute_import

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np
import optimistix as optx

from .angular import sep_const_closest, C_and_sep_const_closest
from . import radial


@partial(jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
def _qnm_residual(omega, params, s, m, l_max, n_inv, cf_tol, Nr_min, Nr_max):
    """Compute the QNM equation residual for root-finding.

    This is the core differentiable function that evaluates how close
    omega is to being a QNM frequency.

    Parameters
    ----------
    omega : complex
        Current guess for the QNM frequency (as 2-element real array)
    params : tuple
        (a, A0) - spin parameter and initial separation constant guess

    Returns
    -------
    array
        Real 2-element array containing [Re(residual), Im(residual)]
    """
    a, A0_real, A0_imag = params
    A0 = A0_real + 1j * A0_imag
    omega_complex = omega[0] + 1j * omega[1]

    # Oblateness parameter
    c = a * omega_complex

    # Separation constant at this a*omega
    A = sep_const_closest(A0, s, c, m, l_max)

    # Evaluate continued fraction
    inv_err, cf_err, n_frac = radial.leaver_cf_inv_lentz(
        omega_complex, a, s, m, A, n_inv, cf_tol, Nr_min, Nr_max
    )

    return jnp.array([jnp.real(inv_err), jnp.imag(inv_err)])


@partial(jit, static_argnums=(3, 4, 5, 6, 7, 8, 9))
def solve_qnm(
    omega_guess, a, A0, s, m, l_max, n_inv, cf_tol, Nr_min, Nr_max, tol=1e-10
):
    """Solve for QNM frequency using differentiable root-finding.

    This function is fully differentiable with respect to `a` and `A0`.

    Parameters
    ----------
    omega_guess : complex
        Initial guess for QNM frequency
    a : float
        Dimensionless spin parameter, 0 <= a < 1
    A0 : complex
        Initial guess for separation constant
    s : int
        Spin weight of field
    m : int
        Azimuthal quantum number
    l_max : int
        Maximum l for angular matrix truncation
    n_inv : int
        Inversion number (overtone selector)
    cf_tol : float
        Continued fraction tolerance
    Nr_min : int
        Minimum CF iterations
    Nr_max : int
        Maximum CF iterations
    tol : float
        Root-finding tolerance

    Returns
    -------
    complex
        The QNM frequency omega
    """
    # Initial guess as real array
    y0 = jnp.array([jnp.real(omega_guess), jnp.imag(omega_guess)])

    # Parameters
    params = (a, jnp.real(A0), jnp.imag(A0))

    # Define the residual function for optimistix
    def fn(y, args):
        return _qnm_residual(y, args, s, m, l_max, n_inv, cf_tol, Nr_min, Nr_max)

    # Use Newton solver from optimistix
    solver = optx.Newton(rtol=tol, atol=tol)
    sol = optx.root_find(fn, solver, y0, args=params, max_steps=100, throw=False)

    omega_real, omega_imag = sol.value
    return omega_real + 1j * omega_imag


@partial(jit, static_argnums=(3, 4, 5, 6, 7, 8, 9))
def solve_qnm_with_A(
    omega_guess, a, A0, s, m, l_max, n_inv, cf_tol, Nr_min, Nr_max, tol=1e-10
):
    """Solve for QNM frequency and return separation constant.

    Fully differentiable version that returns omega and A.
    The eigenvector C is computed separately (not differentiable through eigenvectors).

    Parameters
    ----------
    omega_guess : complex
        Initial guess for QNM frequency
    a : float
        Dimensionless spin parameter
    A0 : complex
        Initial guess for separation constant
    s, m, l_max, n_inv, cf_tol, Nr_min, Nr_max, tol :
        Same as solve_qnm

    Returns
    -------
    tuple
        (omega, A, C) - frequency, separation constant, mixing coefficients
        Note: gradients through C are not supported (eigenvector derivatives not implemented in JAX)
    """
    omega = solve_qnm(
        omega_guess, a, A0, s, m, l_max, n_inv, cf_tol, Nr_min, Nr_max, tol
    )

    c = a * omega
    # Use sep_const_closest for the eigenvalue (differentiable)
    A = sep_const_closest(A0, s, c, m, l_max)

    # Compute C separately using lax.stop_gradient to avoid eigenvector differentiation issues
    # This is fine because C is typically only used for output, not for differentiation
    c_stopped = jax.lax.stop_gradient(c)
    A_stopped = jax.lax.stop_gradient(A)
    _, C = C_and_sep_const_closest(A_stopped, s, c_stopped, m, l_max)

    return omega, A, C


class NearbyRootFinder(object):
    """Object to find and store results from simultaneous roots of
    radial and angular QNM equations, following the
    Leaver and Cook-Zalutskiy approach.

    This class provides both a differentiable JAX interface and
    a legacy scipy-based interface for compatibility.

    Parameters
    ----------
    a: float [default: 0.]
      Dimensionless spin of black hole, 0 <= a < 1.

    s: int [default: -2]
      Spin of field of interest

    m: int [default: 2]
      Azimuthal number of mode of interest

    A_closest_to: complex [default: 4.+0.j]
      Complex value close to desired separation constant.

    l_max: int [default: 20]
      Maximum value of l to include in the spherical-spheroidal
      matrix for finding separation constant.

    omega_guess: complex [default: .5-.5j]
      Initial guess of omega for root-finding

    tol: float [default: sqrt(double epsilon)]
      Tolerance for root-finding omega

    cf_tol: float [default: 1e-10]
      Tolerance for continued fraction calculation

    n_inv: int [default: 0]
      Inversion number of radial infinite continued fraction

    Nr_min: int [default: 300]
      Floor for Nr (for dynamic control of Nr)

    Nr_max: int [default: 4000]
      Ceiling for Nr (for dynamic control of Nr)
    """

    def __init__(self, *args, **kwargs):
        # Set defaults before using values in kwargs
        self.a = 0.0
        self.s = -2
        self.m = 2
        self.A0 = 4.0 + 0.0j
        self.l_max = 20
        self.omega_guess = 0.5 - 0.5j
        self.tol = np.sqrt(np.finfo(float).eps)
        self.cf_tol = 1e-10
        self.n_inv = 0
        self.Nr = 300
        self.Nr_min = 300
        self.Nr_max = 4000
        self.r_N = 1.0

        self.set_params(**kwargs)

    def set_params(self, *args, **kwargs):
        """Set the parameters for root finding."""
        self.a = kwargs.get("a", self.a)
        self.s = kwargs.get("s", self.s)
        self.m = kwargs.get("m", self.m)
        self.A0 = kwargs.get("A_closest_to", self.A0)
        self.l_max = kwargs.get("l_max", self.l_max)
        self.omega_guess = kwargs.get("omega_guess", self.omega_guess)
        self.tol = kwargs.get("tol", self.tol)
        self.cf_tol = kwargs.get("cf_tol", self.cf_tol)
        self.n_inv = kwargs.get("n_inv", self.n_inv)
        self.Nr = kwargs.get("Nr", self.Nr)
        self.Nr_min = kwargs.get("Nr_min", self.Nr_min)
        self.Nr_max = kwargs.get("Nr_max", self.Nr_max)
        self.r_N = kwargs.get("r_N", self.r_N)

        self.poles = np.array([])
        self.clear_results()

    def clear_results(self):
        """Clears the stored results from last call of do_solve"""
        self.solved = False
        self.opt_res = None
        self.omega = None
        self.A = None
        self.C = None
        self.cf_err = None
        self.n_frac = None
        self.poles = np.array([])

    def __call__(self, x):
        """Internal function for scipy compatibility."""
        omega = x[0] + 1.0j * x[1]
        c = self.a * omega
        A = sep_const_closest(self.A0, self.s, c, self.m, self.l_max)

        inv_err, self.cf_err, self.n_frac = radial.leaver_cf_inv_lentz(
            omega,
            self.a,
            self.s,
            self.m,
            A,
            self.n_inv,
            self.cf_tol,
            self.Nr_min,
            self.Nr_max,
        )

        pole_factors = np.prod(omega - self.poles) if len(self.poles) > 0 else 1.0
        supp_err = inv_err / pole_factors

        return [float(np.real(supp_err)), float(np.imag(supp_err))]

    def do_solve(self):
        """Try to find a root of the continued fraction equation.

        Uses the differentiable optimistix solver internally.
        """
        try:
            omega, A, C = solve_qnm_with_A(
                self.omega_guess,
                float(self.a),
                self.A0,
                self.s,
                self.m,
                self.l_max,
                self.n_inv,
                self.cf_tol,
                self.Nr_min,
                self.Nr_max,
                self.tol,
            )

            # Check if solution is valid
            if not jnp.isfinite(omega):
                self.clear_results()
                return None

            self.solved = True
            self.omega = complex(omega)
            self.A = np.complex128(A)
            self.C = np.array(C)

            # Compute cf_err for the solution
            c = self.a * omega
            A_check = sep_const_closest(self.A0, self.s, c, self.m, self.l_max)
            _, self.cf_err, self.n_frac = radial.leaver_cf_inv_lentz(
                omega,
                self.a,
                self.s,
                self.m,
                A_check,
                self.n_inv,
                self.cf_tol,
                self.Nr_min,
                self.Nr_max,
            )

            return self.omega

        except Exception:
            self.clear_results()
            return None

    def get_cf_err(self):
        """Return the continued fraction error and iterations."""
        return self.cf_err, self.n_frac

    def set_poles(self, poles=[]):
        """Set poles to multiply error function."""
        self.poles = np.array(poles).astype(complex)
