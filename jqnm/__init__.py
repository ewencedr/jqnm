"""jqnm - JAX-accelerated Kerr quasinormal modes.

A JAX-based rewrite of the qnm package for computing Kerr black hole
quasinormal mode frequencies, angular separation constants, and
spherical-spheroidal mixing coefficients.

This package provides GPU acceleration and automatic differentiation
capabilities via JAX.

The highest-level interface is via :class:`jqnm.cached.KerrSeqCache`,
which will fetch instances of :class:`jqnm.spinsequence.KerrSpinSeq`.

Examples
--------

>>> import jqnm
>>> # jqnm.download_data() # Only need to do this once
>>> grav_220 = jqnm.modes_cache(s=-2, l=2, m=2, n=0)
>>> omega, A, C = grav_220(a=0.68)
>>> print(omega)
(0.5239751042900845-0.08151262363119986j)

**Members**

.. autosummary::

   jqnm.download_data
   jqnm.modes_cache

.. autofunction:: download_data

"""

from __future__ import print_function, division, absolute_import

# Enable 64-bit precision in JAX
import jax
jax.config.update("jax_enable_x64", True)

from ._version import __version__

__copyright__ = "Copyright (C) 2024 Cedric Ewen"
__email__ = ""  # Add your email here
__status__ = "beta"
__author__ = "Cedric Ewen"
__license__ = """
MIT License

Copyright (c) 2024 Cedric Ewen
Copyright (c) 2019 Leo C. Stein (original qnm package)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

__credits__ = """
This package is a JAX-based rewrite of the qnm package by Leo C. Stein.
Original package: https://github.com/duetosymmetry/qnm

If you use this package in academic work, please cite:
  Stein, Leo C. (2019). "qnm: A Python package for calculating Kerr 
  quasinormal modes, separation constants, and spherical-spheroidal 
  mixing coefficients." J. Open Source Softw., 4(42), 1683.
  doi:10.21105/joss.01683, arXiv:1908.10377
"""

from . import radial
from . import angular
from . import contfrac
from . import nearby

from . import spinsequence

from . import cached
from .cached import download_data

############################################################
## Package initialization

# Singleton for cache
modes_cache = cached.KerrSeqCache(init_schw=True)
"""Interface to the cache of QNMs.  This is a singleton instance of
:class:`jqnm.cached.KerrSeqCache`.  It can be called like a function
`jqnm.modes_cache(s,l,m,n)` to get a specific mode, the result being an
instance of :class:`jqnm.spinsequence.KerrSpinSeq`."""

# Ensure common versions of jitted functions are compiled
def _ensure_jitted():
    finder = nearby.NearbyRootFinder(a=0.3, s=-2, m=2, A_closest_to=3.6+0.1j,
                                     l_max=20, omega_guess=0.4-0.09j,
                                     tol=1e-10, n_inv=0, Nr_max=4000)
    finder.do_solve()
    finder.set_params(Nr_max=float("inf"))
    finder.do_solve()
    
    return

_ensure_jitted()
