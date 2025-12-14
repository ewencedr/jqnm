"""Follow a QNM labeled by (s,l,m,n) as spin varies from a=0 upwards.

This module uses JAX for GPU acceleration and automatic differentiation.
"""

from __future__ import division, print_function, absolute_import

import logging
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy import optimize, interpolate

try:
    NoConvergence = optimize.NoConvergence
except AttributeError:  # scipy < 1.13.0
    NoConvergence = optimize.nonlin.NoConvergence

from .angular import l_min, swsphericalh_A, C_and_sep_const_closest
from .nearby import NearbyRootFinder, solve_qnm_with_A

from .schwarzschild.tabulated import QNMDict


def _jax_interp(x, xp, fp):
    """Simple linear interpolation in JAX (differentiable).

    Parameters
    ----------
    x : float
        Point at which to evaluate interpolation
    xp : array
        X coordinates of data points (must be sorted)
    fp : array
        Y coordinates of data points

    Returns
    -------
    float
        Interpolated value at x
    """
    # Find the interval containing x
    idx = jnp.searchsorted(xp, x, side="right") - 1
    idx = jnp.clip(idx, 0, len(xp) - 2)

    # Linear interpolation
    x0, x1 = xp[idx], xp[idx + 1]
    f0, f1 = fp[idx], fp[idx + 1]

    t = (x - x0) / (x1 - x0)
    return f0 + t * (f1 - f0)


@partial(jax.jit, static_argnums=(5, 6, 7, 8, 9, 10, 11))
def compute_qnm_at_spin(
    a,
    a_data,
    omega_data,
    A_data,
    A0_init,
    s,
    m,
    l_max,
    n_inv,
    cf_tol,
    Nr_min,
    Nr_max,
    tol=1e-10,
):
    """Compute QNM frequency at a given spin value (fully differentiable).

    This function interpolates from precomputed data to get an initial guess,
    then refines with the differentiable solver.

    Parameters
    ----------
    a : float
        Spin value at which to compute QNM
    a_data : array
        Array of spin values from precomputed sequence
    omega_data : array
        Array of complex omega values (as 2D array [real, imag])
    A_data : array
        Array of complex A values (as 2D array [real, imag])
    A0_init : complex
        Initial separation constant guess
    s, m, l_max, n_inv, cf_tol, Nr_min, Nr_max, tol :
        Solver parameters

    Returns
    -------
    tuple
        (omega, A, C) - QNM frequency, separation constant, mixing coefficients
    """
    # Interpolate omega guess
    omega_real_guess = _jax_interp(a, a_data, omega_data[0])
    omega_imag_guess = _jax_interp(a, a_data, omega_data[1])
    omega_guess = omega_real_guess + 1j * omega_imag_guess

    # Interpolate A guess
    A_real_guess = _jax_interp(a, a_data, A_data[0])
    A_imag_guess = _jax_interp(a, a_data, A_data[1])
    A0 = A_real_guess + 1j * A_imag_guess

    # Solve with differentiable solver
    omega, A, C = solve_qnm_with_A(
        omega_guess, a, A0, s, m, l_max, n_inv, cf_tol, Nr_min, Nr_max, tol
    )

    return omega, A, C


class KerrSpinSeq(object):
    """Object to follow a QNM up a sequence in a, starting from a=0.

    This class provides both:
    1. A caching interface for precomputed spin sequences
    2. A fully differentiable JAX interface via `__call__` and `compute_at`

    Values for omega and the separation constant from one value of a
    are used to seed the root finding for the next value of a.

    Parameters
    ----------
    a_max: float [default: .99]
      Maximum dimensionless spin of black hole for the sequence.

    delta_a: float [default: 0.005]
      Step size in a for following the sequence from a=0 to a_max

    s: int [default: -2]
      Spin of field of interest

    m: int [default: 2]
      Azimuthal number of mode of interest

    l: int [default: 2]
      The l-number of the mode

    l_max: int [default: 20]
      Maximum value of l in spherical-spheroidal matrix

    tol: float [default: sqrt(double epsilon)]
      Tolerance for root-finding omega

    cf_tol: float [default: 1e-10]
      Tolerance for continued fraction calculation

    n: int [default: 0]
      Overtone number of interest

    Nr_max: int [default: 4000]
      Maximum number of terms for evaluating continued fraction.
    """

    def __init__(self, *args, **kwargs):
        # Read args
        self.a_max = kwargs.get("a_max", 0.99)
        self.delta_a = kwargs.get("delta_a", 0.005)
        self.delta_a_min = kwargs.get("delta_a_min", 1.0e-5)
        self.delta_a_max = kwargs.get("delta_a_max", 4.0e-3)
        self.s = kwargs.get("s", -2)
        self.m = kwargs.get("m", 2)
        self.l = kwargs.get("l", 2)
        self.l_max = kwargs.get("l_max", 20)
        self.tol = kwargs.get("tol", np.sqrt(np.finfo(float).eps))
        self.cf_tol = kwargs.get("cf_tol", 1e-10)
        self.n = kwargs.get("n", 0)

        if "omega_guess" in kwargs.keys():
            self.omega_guess = kwargs.get("omega_guess")
        else:
            qnm_dict = QNMDict()
            self.omega_guess = qnm_dict(self.s, self.l, self.n)[0]

        self.Nr = kwargs.get("Nr", 300)
        self.Nr_min = self.Nr
        self.Nr_max = kwargs.get("Nr_max", 4000)
        self.r_N = kwargs.get("r_N", 0.0j)

        if not (self.a_max < 1.0):
            raise ValueError("a_max={} must be < 1.".format(self.a_max))
        if not (self.l >= l_min(self.s, self.m)):
            raise ValueError(
                "l={} must be >= l_min={}".format(self.l, l_min(self.s, self.m))
            )

        # Create arrays for storage
        self.a = []
        self.omega = []
        self.cf_err = []
        self.n_frac = []
        self.A = []
        self.C = []

        self.delta_a_prop = []

        # Interpolants (scipy, for legacy compatibility)
        self._interp_o_r = None
        self._interp_o_i = None
        self._interp_A_r = None
        self._interp_A_i = None

        # JAX-compatible data arrays
        self._a_jax = None
        self._omega_jax = None
        self._A_jax = None

        # Root finder instance
        self.solver = NearbyRootFinder(
            s=self.s,
            m=self.m,
            l_max=self.l_max,
            tol=self.tol,
            n_inv=self.n,
            Nr=self.Nr,
            Nr_min=self.Nr_min,
            Nr_max=self.Nr_max,
            r_N=self.r_N,
        )

    def do_find_sequence(self):
        """Solve for the spin sequence from a=0 to a_max."""
        logging.info("l={}, m={}, n={} starting".format(self.l, self.m, self.n))

        i = 0
        _a = 0.0

        warned_imag_axis = False

        A0 = swsphericalh_A(self.s, self.l, self.m)
        omega_guess = self.omega_guess

        while _a <= self.a_max:
            self.solver.set_params(a=_a, A_closest_to=A0, omega_guess=omega_guess)
            result = self.solver.do_solve()

            if result is None:
                raise NoConvergence(
                    "Failed to find QNM in sequence at a={}".format(_a),
                )

            if (i == 0) and (np.real(result) < 0):
                result = -np.conjugate(result)

            cf_err, n_frac = self.solver.get_cf_err()

            if (np.abs(np.real(result)) < self.tol) and not warned_imag_axis:
                logging.warn(
                    "Danger! At a={}, found Re[omega]={}, near imaginary axis.".format(
                        _a, result
                    )
                )
                warned_imag_axis = True

            self.a.append(_a)
            self.omega.append(result)
            self.A.append(self.solver.A)
            self.C.append(self.solver.C)
            self.cf_err.append(cf_err)
            self.n_frac.append(n_frac)

            if _a == self.a_max:
                break

            _a, omega_guess, A0 = self._propose_next_a_om_A()
            i = i + 1

        logging.info(
            "s={}, l={}, m={}, n={} completed with {} points".format(
                self.s, self.l, self.m, self.n, len(self.a)
            )
        )

        self.build_interps()

    def _propose_next_a_om_A(self):
        """Compute starting values for the next step along the spin sequence."""
        _a = self.a[-1]

        if len(self.a) < 3:
            omega_guess = self.omega[-1]
            A0 = self.A[-1]
            _a = _a + self.delta_a
        else:
            interp_o_r = interpolate.UnivariateSpline(
                self.a[-3:], np.real(self.omega[-3:]), s=0, k=2, ext=0
            )
            interp_o_i = interpolate.UnivariateSpline(
                self.a[-3:], np.imag(self.omega[-3:]), s=0, k=2, ext=0
            )
            interp_A_r = interpolate.UnivariateSpline(
                self.a[-3:], np.real(self.A[-3:]), s=0, k=2, ext=0
            )
            interp_A_i = interpolate.UnivariateSpline(
                self.a[-3:], np.imag(self.A[-3:]), s=0, k=2, ext=0
            )

            d2_o_r = interp_o_r.derivative(2)
            d2_o_i = interp_o_i.derivative(2)
            d2_A_r = interp_A_r.derivative(2)
            d2_A_i = interp_A_i.derivative(2)

            d2_o = np.abs(d2_o_r(_a) + 1.0j * d2_o_i(_a))
            d2_A = np.abs(d2_A_r(_a) + 1.0j * d2_A_i(_a))
            d2 = np.max([d2_o, d2_A])
            _delta_a = 0.05 / np.sqrt(d2)

            self.delta_a_prop.append(_delta_a)

            _delta_a = np.max([self.delta_a_min, _delta_a])
            _delta_a = np.min([self.delta_a_max, _delta_a])
            _a = _a + _delta_a

            if _a > self.a_max:
                _a = self.a_max

            omega_guess = interp_o_r(_a) + 1.0j * interp_o_i(_a)
            A0 = interp_A_r(_a) + 1.0j * interp_A_i(_a)

        return _a, omega_guess, A0

    def build_interps(self):
        """Build interpolating functions for omega(a) and A(a)."""
        k = 3  # cubic

        self._interp_o_r = interpolate.UnivariateSpline(
            self.a, np.real(self.omega), s=0, k=k, ext=0
        )
        self._interp_o_i = interpolate.UnivariateSpline(
            self.a, np.imag(self.omega), s=0, k=k, ext=0
        )
        self._interp_A_r = interpolate.UnivariateSpline(
            self.a, np.real(self.A), s=0, k=k, ext=0
        )
        self._interp_A_i = interpolate.UnivariateSpline(
            self.a, np.imag(self.A), s=0, k=k, ext=0
        )

        # Build JAX-compatible arrays for differentiable interpolation
        self._a_jax = jnp.array(self.a)
        self._omega_jax = jnp.array([np.real(self.omega), np.imag(self.omega)])
        self._A_jax = jnp.array([np.real(self.A), np.imag(self.A)])

    def compute_at(self, a):
        """Compute QNM at spin `a` (fully differentiable).

        This is the primary differentiable interface. Use this when you need
        gradients with respect to the spin parameter.

        Parameters
        ----------
        a : float or jax array
            Spin value, 0 <= a < 1

        Returns
        -------
        tuple
            (omega, A, C) - all as JAX arrays
        """
        # Build JAX arrays if not present (e.g., loaded from old pickle)
        if not hasattr(self, "_a_jax") or self._a_jax is None:
            self._a_jax = jnp.array(self.a)
            self._omega_jax = jnp.array([np.real(self.omega), np.imag(self.omega)])
            self._A_jax = jnp.array([np.real(self.A), np.imag(self.A)])

        A0_init = swsphericalh_A(self.s, self.l, self.m)

        return compute_qnm_at_spin(
            a,
            self._a_jax,
            self._omega_jax,
            self._A_jax,
            A0_init,
            self.s,
            self.m,
            self.l_max,
            self.n,
            self.cf_tol,
            self.Nr_min,
            self.Nr_max,
            self.tol,
        )

    def __call__(self, a, store=False, interp_only=False, resolve_if_found=False):
        """Solve for omega, A, and C[] at a given spin a.

        Parameters
        ----------
        a: float
          Value of spin, 0 <= a < 1.

        store: bool, optional [default: False]
          Whether or not to save newly solved data in sequence.

        interp_only: bool, optional [default: False]
          If True, just use the interpolated guess without solving.

        resolve_if_found: bool, optional [default: False]
          If True, re-solve even if a is already in the sequence.

        Returns
        -------
        complex, complex, complex ndarray
          (omega, A, C)
        """
        if not (
            (isinstance(a, float) or isinstance(a, int)) and (a >= 0.0) and (a < 1.0)
        ):
            raise ValueError("a={} is not a float in the range [0,1)".format(a))

        if interp_only:
            if store:
                logging.warn("store=True ignored when interp_only=True")
                store = False
            if resolve_if_found:
                logging.warn("resolve_if_found=True ignored when interp_only=True")
                resolve_if_found = False

        if (not resolve_if_found) and (a in self.a):
            a_ind = self.a.index(a)
            return self.omega[a_ind], self.A[a_ind], self.C[a_ind]

        o_r = self._interp_o_r(a)
        o_i = self._interp_o_i(a)
        A_r = self._interp_A_r(a)
        A_i = self._interp_A_i(a)

        omega_guess = complex(o_r, o_i)
        A_guess = complex(A_r, A_i)

        if interp_only:
            c = a * omega_guess
            A_guess_jax, C_guess_jax = C_and_sep_const_closest(
                A_guess, self.s, c, self.m, self.l_max
            )
            return omega_guess, complex(A_guess_jax), np.array(C_guess_jax)

        self.solver.set_params(a=a, omega_guess=omega_guess, A_closest_to=A_guess)
        result = self.solver.do_solve()

        if result is None:
            raise NoConvergence("Failed to find QNM in sequence at a={}".format(a))

        cf_err, n_frac = self.solver.get_cf_err()

        if store:
            if a in self.a:
                insert_ind = self.a.index(a)
            else:
                try:
                    insert_ind = next(i for i, _a in enumerate(self.a) if _a > a)
                except StopIteration:
                    insert_ind = len(self.a)

            self.a.insert(insert_ind, a)
            self.omega.insert(insert_ind, result)
            self.A.insert(insert_ind, self.solver.A)
            self.C.insert(insert_ind, self.solver.C)
            self.cf_err.insert(insert_ind, cf_err)
            self.n_frac.insert(insert_ind, n_frac)

        return result, self.solver.A, self.solver.C

    def __repr__(self):
        from textwrap import dedent

        rep = """<{} with s={}, l={}, m={}, n={},
             l_max={}, tol={},
             Nr={}, Nr_min={}, Nr_max={},
             with values at a=[{}, ... <{}> ..., {}]>"""
        rep = rep.format(
            type(self).__name__,
            str(self.s),
            str(self.l),
            str(self.m),
            str(self.n),
            str(self.l_max),
            str(self.tol),
            str(self.Nr),
            str(self.Nr_min),
            str(self.Nr_max),
            str(self.a[0]),
            str(len(self.a) - 2),
            str(self.a[-1]),
        )
        return dedent(rep)
