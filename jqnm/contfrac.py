""" Infinite continued fractions via Lentz's method.

This module uses JAX for GPU acceleration and automatic differentiation.
"""

from __future__ import division, print_function, absolute_import

import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial


def lentz(a, b, tol=1.e-10, N_min=0, N_max=10000, tiny=1.e-30, args=None):
    """Compute a continued fraction via modified Lentz's method.

    This implementation is by the book [1]_.  The value to compute is:
      b_0 + a_1/( b_1 + a_2/( b_2 + a_3/( b_3 + ...)))
    where a_n = a(n, *args) and b_n = b(n, *args).

    Parameters
    ----------
    a: callable returning numeric.
    b: callable returning numeric.

    tol: float [default: 1.e-10]
      Tolerance for termination of evaluation.

    N_min: int [default: 0]
      Minimum number of iterations to evaluate.

    N_max: int or comparable [default: 10000]
      Maximum number of iterations to evaluate.

    tiny: float [default: 1.e-30]
      Very small number to control convergence of Lentz's method when
      there is cancellation in a denominator.

    args: tuple [default: None]
      Additional arguments to pass to the user-defined functions a, b.
      If given, the additional arguments are passed to all
      user-defined functions.  So if, for example, `a` has the
      signature `a(n, x, y)`, then `b` must have the same signature,
      and args must be a tuple of length 2, `args=(x,y)`.

    Returns
    -------
    (float, float, int)
      The first element of the tuple is the value of the continued
      fraction. The second element is the estimated error. The third
      element is the number of iterations.

    References
    ----------
    .. [1] WH Press, SA Teukolsky, WT Vetterling, BP Flannery,
       "Numerical Recipes," 3rd Ed., Cambridge University Press 2007,
       ISBN 0521880688, 9780521880688 .
    """

    if args is None:
        args = ()

    if type(args) is not tuple:
        raise ValueError("args={} must be of type tuple".format(args))

    f_old = b(0, *args)
    f_old = jnp.where(f_old == 0, tiny, f_old)

    C_old = f_old
    D_old = 0. + 0.j

    conv = False
    j = 1
    Delta = 1. + 0.j

    while ((not conv) and (j < N_max)):
        aj, bj = a(j, *args), b(j, *args)

        D_new = bj + aj * D_old
        D_new = jnp.where(D_new == 0, tiny, D_new)

        C_new = bj + aj / C_old
        C_new = jnp.where(C_new == 0, tiny, C_new)

        D_new = 1. / D_new
        Delta = C_new * D_new
        f_new = f_old * Delta

        if ((j > N_min) and (jnp.abs(Delta - 1.) < tol)):
            conv = True

        j = j + 1
        D_old = D_new
        C_old = C_new
        f_old = f_new

    return f_new, jnp.abs(Delta - 1.), j - 1


def lentz_jax(a_func, b_func, n_inv, D, tol=1.e-10, N_min=0, N_max=10000, tiny=1.e-30):
    """JAX-compatible Lentz's method for continued fractions.
    
    This version is designed for the radial Teukolsky equation solver.
    
    Parameters
    ----------
    a_func: callable
      Function that computes a_n given n
    b_func: callable
      Function that computes b_n given n  
    n_inv: int
      Inversion number
    D: array
      D coefficients
    tol: float
      Convergence tolerance
    N_min: int
      Minimum iterations
    N_max: int
      Maximum iterations
    tiny: float
      Small number to avoid division by zero
        
    Returns
    -------
    tuple: (result, error, n_iterations)
    """
    
    def cond_fn(state):
        j, conv, f_old, C_old, D_old, Delta, n = state
        return (j < N_max) & (~conv)
    
    def body_fn(state):
        j, conv, f_old, C_old, D_old, Delta, n = state
        
        # Compute a_n and b_n
        an = a_func(n, D)
        n_next = n + 1
        bn = b_func(n_next, D)
        
        D_new = bn + an * D_old
        D_new = jnp.where(D_new == 0, tiny + 0.j, D_new)
        
        C_new = bn + an / C_old
        C_new = jnp.where(C_new == 0, tiny + 0.j, C_new)
        
        D_new = 1. / D_new
        Delta_new = C_new * D_new
        f_new = f_old * Delta_new
        
        conv_new = (j > N_min) & (jnp.abs(Delta_new - 1.) < tol)
        
        return (j + 1, conv_new, f_new, C_new, D_new, Delta_new, n_next)
    
    # Initialize
    f_init = tiny + 0.j
    C_init = f_init
    D_init = 0. + 0.j
    Delta_init = 1. + 0.j
    
    init_state = (1, False, f_init, C_init, D_init, Delta_init, n_inv)
    
    final_state = lax.while_loop(cond_fn, body_fn, init_state)
    j_final, _, f_final, _, _, Delta_final, _ = final_state
    
    return f_final, jnp.abs(Delta_final - 1.), j_final - 1
