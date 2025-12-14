"""Tests comparing jqnm (this package) against the original qnm package.

This module compares results from the JAX-based rewrite against the original
numpy/scipy qnm package to ensure numerical equivalence.

To run these tests, first install the original qnm package:
    pip install qnm

The tests use importlib to load the original package from site-packages
while the local development package can still be imported normally.
"""

import pytest
import numpy as np
import sys
import os
import importlib.util
import types


def _load_package_from_path(package_path, package_name):
    """Load a Python package from a specific filesystem path.
    
    Parameters
    ----------
    package_path : str
        Path to the package's __init__.py
    package_name : str
        Name to use for the imported package
        
    Returns
    -------
    module
        The loaded package
    """
    spec = importlib.util.spec_from_file_location(package_name, package_path)
    if spec is None:
        raise ImportError(f"Could not load spec from {package_path}")
    
    module = importlib.util.module_from_spec(spec)
    
    # We need to handle submodules - set up the parent package first
    parent_dir = os.path.dirname(package_path)
    
    # Temporarily add parent to path for submodule imports
    old_path = sys.path.copy()
    sys.path.insert(0, os.path.dirname(parent_dir))
    
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path = old_path
    
    return module


def _find_site_packages_qnm():
    """Find the original qnm package in site-packages.
    
    Returns the path to the qnm package in site-packages, or None if not found.
    """
    import site
    
    # Get all site-packages directories
    site_packages_dirs = site.getsitepackages()
    if hasattr(site, 'getusersitepackages'):
        user_site = site.getusersitepackages()
        if user_site and os.path.exists(user_site):
            site_packages_dirs.append(user_site)
    
    # Look for qnm in each site-packages
    for sp_dir in site_packages_dirs:
        qnm_path = os.path.join(sp_dir, 'qnm', '__init__.py')
        if os.path.exists(qnm_path):
            # Check it's the original (has numba imports, not JAX)
            with open(qnm_path, 'r') as f:
                content = f.read()
                if 'numba' in content or 'jax' not in content.lower():
                    return qnm_path
    
    return None


def _get_original_qnm():
    """Import the original qnm package from site-packages.
    
    Uses importlib to load the package with a different name to avoid
    conflicts with the local development version.
    """
    # Find the original qnm in site-packages
    qnm_init_path = _find_site_packages_qnm()
    
    if qnm_init_path is None:
        raise ImportError("Original qnm package not found in site-packages")
    
    # Get the package directory
    qnm_dir = os.path.dirname(qnm_init_path)
    parent_dir = os.path.dirname(qnm_dir)
    
    # Save current state
    old_path = sys.path.copy()
    old_modules = {}
    
    # Remove any existing qnm modules and store them
    for key in list(sys.modules.keys()):
        if key == 'qnm' or key.startswith('qnm.'):
            old_modules[key] = sys.modules.pop(key)
    
    try:
        # Add site-packages to front of path
        sys.path.insert(0, parent_dir)
        
        # Import the original qnm
        import qnm as qnm_orig
        
        # Import submodules we need
        from qnm import angular, radial, nearby, cached, spinsequence
        from qnm import schwarzschild
        from qnm.schwarzschild import approx
        
        # Download data if needed
        qnm_orig.download_data()
        
        # Create a copy of the module under a different name
        # so we can restore the local qnm later
        qnm_original = types.ModuleType('qnm_original')
        for attr in dir(qnm_orig):
            if not attr.startswith('_'):
                setattr(qnm_original, attr, getattr(qnm_orig, attr))
        
        # Also copy the __file__ and other special attrs
        qnm_original.__file__ = qnm_orig.__file__
        qnm_original.angular = qnm_orig.angular
        qnm_original.radial = qnm_orig.radial
        qnm_original.nearby = qnm_orig.nearby
        qnm_original.cached = qnm_orig.cached
        qnm_original.schwarzschild = qnm_orig.schwarzschild
        qnm_original.modes_cache = qnm_orig.modes_cache
        
        return qnm_original
        
    finally:
        # Restore original path
        sys.path = old_path
        
        # Clear any qnm modules we imported
        for key in list(sys.modules.keys()):
            if key == 'qnm' or key.startswith('qnm.'):
                del sys.modules[key]
        
        # Restore original qnm modules (the local ones)
        for key, mod in old_modules.items():
            sys.modules[key] = mod


# Try to get original qnm at module load time
_IMPORT_ERROR = ""
try:
    qnm_original = _get_original_qnm()
    HAS_ORIGINAL_QNM = True
except (ImportError, Exception) as e:
    HAS_ORIGINAL_QNM = False
    qnm_original = None
    _IMPORT_ERROR = str(e)

# Now import the local package (jqnm)
import jqnm


# Skip all tests in this module if original qnm is not available
pytestmark = pytest.mark.skipif(
    not HAS_ORIGINAL_QNM,
    reason=f"Original qnm package not available: {_IMPORT_ERROR if not HAS_ORIGINAL_QNM else ''}"
)


class TestAngularComparison:
    """Compare angular sector computations against original qnm."""

    @pytest.mark.parametrize("s,l,m", [
        (-2, 2, 2),
        (-2, 2, 1),
        (-2, 2, 0),
        (-2, 3, 2),
        (-1, 2, 2),
        (-1, 3, 1),
        (0, 2, 2),
    ])
    def test_swsphericalh_A(self, s, l, m):
        """Test angular separation constant at a=0."""
        result_jqnm = jqnm.angular.swsphericalh_A(s, l, m)
        result_orig = qnm_original.angular.swsphericalh_A(s, l, m)
        assert np.allclose(result_jqnm, result_orig), \
            f"swsphericalh_A mismatch: jqnm={result_jqnm}, original={result_orig}"

    @pytest.mark.parametrize("s,m", [
        (-2, 2),
        (-2, 0),
        (-2, -2),
        (-1, 1),
        (0, 0),
    ])
    def test_l_min(self, s, m):
        """Test minimum l value calculation."""
        result_jqnm = jqnm.angular.l_min(s, m)
        result_orig = qnm_original.angular.l_min(s, m)
        assert result_jqnm == result_orig, \
            f"l_min mismatch: jqnm={result_jqnm}, original={result_orig}"

    @pytest.mark.parametrize("s,c,m,l_max", [
        (-2, 0.1 + 0.01j, 2, 10),
        (-2, 0.5 + 0.05j, 2, 10),
        (-2, 0.3 - 0.02j, 1, 8),
        (-1, 0.2 + 0.01j, 1, 10),
    ])
    def test_M_matrix(self, s, c, m, l_max):
        """Test spherical-spheroidal decomposition matrix."""
        result_jqnm = np.array(jqnm.angular.M_matrix(s, c, m, l_max))
        result_orig = qnm_original.angular.M_matrix(s, c, m, l_max)
        assert np.allclose(result_jqnm, result_orig, rtol=1e-10), \
            f"M_matrix mismatch for s={s}, c={c}, m={m}, l_max={l_max}"

    @pytest.mark.parametrize("s,c,m,l_max", [
        (-2, 0.1 + 0.01j, 2, 10),
        (-2, 0.5 + 0.05j, 2, 12),
        (-2, 0.3 - 0.02j, 1, 8),
    ])
    def test_sep_consts(self, s, c, m, l_max):
        """Test separation constant eigenvalues."""
        result_jqnm = np.sort(np.array(jqnm.angular.sep_consts(s, c, m, l_max)))
        result_orig = np.sort(qnm_original.angular.sep_consts(s, c, m, l_max))
        # Eigenvalues can have different ordering, so we sort and compare
        assert np.allclose(result_jqnm, result_orig, rtol=1e-8), \
            f"sep_consts mismatch for s={s}, c={c}, m={m}, l_max={l_max}"

    @pytest.mark.parametrize("s,m,l,c", [
        (-2, 2, 2, 0.1 + 0.01j),
        (-2, 2, 3, 0.3 - 0.02j),
        (-2, 1, 2, 0.2 + 0.01j),
    ])
    def test_C_and_sep_const_closest(self, s, m, l, c):
        """Test C coefficients and separation constant computation."""
        l_max = l + 5
        # Use the Schwarzschild value as initial guess
        A0 = jqnm.angular.swsphericalh_A(s, l, m)
        
        A_jqnm, C_jqnm = jqnm.angular.C_and_sep_const_closest(A0, s, c, m, l_max)
        A_orig, C_orig = qnm_original.angular.C_and_sep_const_closest(A0, s, c, m, l_max)
        
        A_jqnm = np.complex128(A_jqnm)
        C_jqnm = np.array(C_jqnm)
        
        assert np.allclose(A_jqnm, A_orig, rtol=1e-8), \
            f"Separation constant mismatch: jqnm={A_jqnm}, original={A_orig}"
        
        # C vectors may differ by a global phase, so compare magnitudes
        # or normalize by first non-zero element
        C_jqnm_normalized = C_jqnm / C_jqnm[0] if abs(C_jqnm[0]) > 1e-10 else C_jqnm
        C_orig_normalized = C_orig / C_orig[0] if abs(C_orig[0]) > 1e-10 else C_orig
        
        assert np.allclose(C_jqnm_normalized, C_orig_normalized, rtol=1e-6), \
            f"C coefficients mismatch"


class TestRadialComparison:
    """Compare radial sector computations against original qnm."""

    @pytest.mark.parametrize("omega,a,s,m", [
        (0.4 - 0.1j, 0.5, -2, 2),
        (0.3 - 0.05j, 0.3, -2, 1),
        (0.5 - 0.2j, 0.7, -2, 2),
        (0.4 - 0.1j, 0.1, -1, 1),
    ])
    def test_sing_pt_char_exps(self, omega, a, s, m):
        """Test singular point characteristic exponents."""
        result_jqnm = jqnm.radial.sing_pt_char_exps(omega, a, s, m)
        result_orig = qnm_original.radial.sing_pt_char_exps(omega, a, s, m)
        
        for i, (val_jqnm, val_orig) in enumerate(zip(result_jqnm, result_orig)):
            assert np.allclose(val_jqnm, val_orig, rtol=1e-10), \
                f"sing_pt_char_exps component {i} mismatch"

    @pytest.mark.parametrize("omega,a,s,m,A", [
        (0.4 - 0.1j, 0.5, -2, 2, 4.0 + 0.1j),
        (0.3 - 0.05j, 0.3, -2, 1, 3.5 + 0.05j),
        (0.5 - 0.2j, 0.7, -2, 2, 4.5 - 0.2j),
    ])
    def test_D_coeffs(self, omega, a, s, m, A):
        """Test D coefficient computation."""
        result_jqnm = np.array(jqnm.radial.D_coeffs(omega, a, s, m, A))
        result_orig = qnm_original.radial.D_coeffs(omega, a, s, m, A)
        
        assert np.allclose(result_jqnm, result_orig, rtol=1e-10), \
            f"D_coeffs mismatch"

    @pytest.mark.parametrize("omega,a,s,m,A,n_inv", [
        (0.4 - 0.2j, 0.02, -2, 2, 4.0 + 0.0j, 0),
        (0.5 - 0.1j, 0.5, -2, 2, 4.0 + 0.5j, 0),
        (0.4 - 0.15j, 0.3, -2, 1, 3.5 + 0.2j, 0),
    ])
    def test_leaver_cf_inv(self, omega, a, s, m, A, n_inv):
        """Test Leaver continued fraction inversion."""
        # Use the new JIT-compatible function
        result_jqnm, cf_err_jqnm, n_frac_jqnm = jqnm.radial.leaver_cf_inv_lentz(
            omega, a, s, m, A, n_inv, tol=1e-10, N_min=0, N_max=500
        )
        
        # Use the original function
        result_orig, cf_err_orig, n_frac_orig = qnm_original.radial.leaver_cf_inv_lentz(
            omega, a, s, m, A, n_inv, tol=1e-10, N_min=0, N_max=np.inf
        )
        
        result_jqnm = np.complex128(result_jqnm)
        
        # Allow some tolerance since implementations may differ slightly
        assert np.allclose(result_jqnm, result_orig, rtol=1e-5), \
            f"leaver_cf_inv mismatch: jqnm={result_jqnm}, original={result_orig}"


class TestSchwarzschildComparison:
    """Compare Schwarzschild approximations against original qnm."""

    @pytest.mark.parametrize("s,l,n", [
        (-2, 2, 0),
        (-2, 2, 1),
        (-2, 3, 0),
        (-2, 4, 2),
        (-1, 2, 0),
        (0, 2, 0),
    ])
    def test_dolan_ottewill_expansion(self, s, l, n):
        """Test Dolan-Ottewill high-l expansion."""
        result_jqnm = jqnm.schwarzschild.approx.dolan_ottewill_expansion(s, l, n)
        result_orig = qnm_original.schwarzschild.approx.dolan_ottewill_expansion(s, l, n)
        
        assert np.allclose(result_jqnm, result_orig, rtol=1e-12), \
            f"dolan_ottewill_expansion mismatch: jqnm={result_jqnm}, original={result_orig}"

    @pytest.mark.parametrize("s,l,n", [
        (-2, 2, 5),
        (-2, 3, 7),
        (-2, 2, 10),
    ])
    def test_large_overtone_expansion(self, s, l, n):
        """Test large overtone (eikonal) expansion."""
        result_jqnm = jqnm.schwarzschild.approx.large_overtone_expansion(s, l, n)
        result_orig = qnm_original.schwarzschild.approx.large_overtone_expansion(s, l, n)
        
        assert np.allclose(result_jqnm, result_orig, rtol=1e-12), \
            f"large_overtone_expansion mismatch: jqnm={result_jqnm}, original={result_orig}"

    @pytest.mark.parametrize("s,l,n", [
        (-2, 2, 0),
        (-2, 2, 1),
        (-2, 2, 5),
        (-2, 3, 0),
        (-2, 3, 7),
    ])
    def test_Schw_QNM_estimate(self, s, l, n):
        """Test Schwarzschild QNM estimate."""
        result_jqnm = jqnm.schwarzschild.approx.Schw_QNM_estimate(s, l, n)
        result_orig = qnm_original.schwarzschild.approx.Schw_QNM_estimate(s, l, n)
        
        assert np.allclose(result_jqnm, result_orig, rtol=1e-12), \
            f"Schw_QNM_estimate mismatch: jqnm={result_jqnm}, original={result_orig}"


class TestNearbyRootFinderComparison:
    """Compare NearbyRootFinder results against original qnm."""

    @pytest.mark.parametrize("a,s,m,l,n", [
        (0.3, -2, 2, 2, 0),
        (0.5, -2, 2, 2, 0),
        (0.7, -2, 2, 2, 0),
        (0.3, -2, 1, 2, 0),
        (0.5, -2, 2, 3, 0),
    ])
    def test_do_solve(self, a, s, m, l, n):
        """Test QNM root finding."""
        l_max = 20
        A0 = jqnm.angular.swsphericalh_A(s, l, m)
        
        # Get a good initial guess from original package
        orig_seq = qnm_original.modes_cache(s=s, l=l, m=m, n=n)
        omega_guess_orig, A_guess_orig, _ = orig_seq(a=a)
        
        # Create solvers
        finder_jqnm = jqnm.nearby.NearbyRootFinder(
            a=a, s=s, m=m, A_closest_to=A0, l_max=l_max,
            omega_guess=omega_guess_orig, tol=1e-10, n_inv=n,
            Nr_max=500
        )
        
        finder_orig = qnm_original.nearby.NearbyRootFinder(
            a=a, s=s, m=m, A_closest_to=A0, l_max=l_max,
            omega_guess=omega_guess_orig, tol=1e-10, n_inv=n,
        )
        
        omega_jqnm = finder_jqnm.do_solve()
        omega_orig = finder_orig.do_solve()
        
        assert omega_jqnm is not None, "jqnm solver failed"
        assert omega_orig is not None, "original solver failed"
        
        assert np.allclose(omega_jqnm, omega_orig, rtol=1e-6), \
            f"omega mismatch: jqnm={omega_jqnm}, original={omega_orig}"
        
        assert np.allclose(finder_jqnm.A, finder_orig.A, rtol=1e-6), \
            f"A mismatch: jqnm={finder_jqnm.A}, original={finder_orig.A}"


class TestModesCacheComparison:
    """Compare modes_cache results against original qnm."""

    @pytest.mark.parametrize("s,l,m,n,a", [
        (-2, 2, 2, 0, 0.68),
        (-2, 2, 2, 0, 0.1),
        (-2, 2, 2, 0, 0.9),
        (-2, 2, 1, 0, 0.5),
        (-2, 3, 2, 0, 0.5),
        (-2, 2, 2, 1, 0.5),
        (-1, 2, 2, 0, 0.5),
    ])
    def test_modes_cache_values(self, s, l, m, n, a):
        """Test that modes_cache returns consistent values."""
        mode_jqnm = jqnm.modes_cache(s=s, l=l, m=m, n=n)
        mode_orig = qnm_original.modes_cache(s=s, l=l, m=m, n=n)
        
        omega_jqnm, A_jqnm, C_jqnm = mode_jqnm(a=a)
        omega_orig, A_orig, C_orig = mode_orig(a=a)
        
        # Convert to numpy for comparison
        omega_jqnm = complex(omega_jqnm)
        A_jqnm = np.complex128(A_jqnm)
        C_jqnm = np.array(C_jqnm)
        
        assert np.allclose(omega_jqnm, omega_orig, rtol=1e-6), \
            f"omega mismatch for (s={s}, l={l}, m={m}, n={n}, a={a}): " \
            f"jqnm={omega_jqnm}, original={omega_orig}"
        
        assert np.allclose(A_jqnm, A_orig, rtol=1e-6), \
            f"A mismatch for (s={s}, l={l}, m={m}, n={n}, a={a}): " \
            f"jqnm={A_jqnm}, original={A_orig}"
        
        # C vectors may differ by global phase
        if abs(C_jqnm[0]) > 1e-10 and abs(C_orig[0]) > 1e-10:
            C_jqnm_normalized = C_jqnm / C_jqnm[0]
            C_orig_normalized = C_orig / C_orig[0]
            assert np.allclose(C_jqnm_normalized, C_orig_normalized, rtol=1e-5), \
                f"C coefficients mismatch for (s={s}, l={l}, m={m}, n={n}, a={a})"

    def test_known_value_220(self):
        """Test the known (2,2,0) mode value from documentation."""
        mode_jqnm = jqnm.modes_cache(s=-2, l=2, m=2, n=0)
        omega_jqnm, _, _ = mode_jqnm(a=0.68)
        
        # Known value from qnm documentation
        expected_omega = 0.5239751042900845 - 0.08151262363119974j
        
        assert np.allclose(omega_jqnm, expected_omega, rtol=1e-6), \
            f"omega mismatch: jqnm={omega_jqnm}, expected={expected_omega}"


class TestSpinSequenceComparison:
    """Compare spin sequence behavior against original qnm."""

    def test_spin_sequence_a_values(self):
        """Test that spin sequences have consistent spin values."""
        seq_jqnm = jqnm.modes_cache(s=-2, l=2, m=2, n=0)
        seq_orig = qnm_original.modes_cache(s=-2, l=2, m=2, n=0)
        
        # Both should have same cached a values
        assert len(seq_jqnm.a) == len(seq_orig.a), \
            f"Different number of cached a values"
        
        assert np.allclose(seq_jqnm.a, seq_orig.a), \
            f"Cached a values differ"

    def test_spin_sequence_omega_values(self):
        """Test that spin sequences have consistent omega values."""
        seq_jqnm = jqnm.modes_cache(s=-2, l=2, m=2, n=0)
        seq_orig = qnm_original.modes_cache(s=-2, l=2, m=2, n=0)
        
        omega_jqnm = np.array(seq_jqnm.omega)
        omega_orig = np.array(seq_orig.omega)
        
        assert np.allclose(omega_jqnm, omega_orig, rtol=1e-6), \
            f"Cached omega values differ"

    @pytest.mark.parametrize("interp_only", [True, False])
    def test_interpolation_modes(self, interp_only):
        """Test interpolation vs solve modes."""
        a = 0.68
        seq_jqnm = jqnm.modes_cache(s=-2, l=2, m=2, n=0)
        seq_orig = qnm_original.modes_cache(s=-2, l=2, m=2, n=0)
        
        omega_jqnm, A_jqnm, C_jqnm = seq_jqnm(a=a, interp_only=interp_only)
        omega_orig, A_orig, C_orig = seq_orig(a=a, interp_only=interp_only)
        
        omega_jqnm = complex(omega_jqnm)
        
        assert np.allclose(omega_jqnm, omega_orig, rtol=1e-6), \
            f"omega mismatch with interp_only={interp_only}: " \
            f"jqnm={omega_jqnm}, original={omega_orig}"


class TestEdgeCasesComparison:
    """Test edge cases and boundary conditions."""

    def test_schwarzschild_limit(self):
        """Test a=0 (Schwarzschild) limit."""
        a = 0.0
        mode_jqnm = jqnm.modes_cache(s=-2, l=2, m=2, n=0)
        mode_orig = qnm_original.modes_cache(s=-2, l=2, m=2, n=0)
        
        omega_jqnm, A_jqnm, _ = mode_jqnm(a=a)
        omega_orig, A_orig, _ = mode_orig(a=a)
        
        omega_jqnm = complex(omega_jqnm)
        A_jqnm = np.complex128(A_jqnm)
        
        assert np.allclose(omega_jqnm, omega_orig, rtol=1e-6), \
            f"Schwarzschild omega mismatch"
        
        assert np.allclose(A_jqnm, A_orig, rtol=1e-6), \
            f"Schwarzschild A mismatch"

    def test_high_spin(self):
        """Test near-extremal spin."""
        a = 0.99
        mode_jqnm = jqnm.modes_cache(s=-2, l=2, m=2, n=0)
        mode_orig = qnm_original.modes_cache(s=-2, l=2, m=2, n=0)
        
        omega_jqnm, A_jqnm, _ = mode_jqnm(a=a)
        omega_orig, A_orig, _ = mode_orig(a=a)
        
        omega_jqnm = complex(omega_jqnm)
        
        assert np.allclose(omega_jqnm, omega_orig, rtol=1e-5), \
            f"High spin omega mismatch: jqnm={omega_jqnm}, original={omega_orig}"

    @pytest.mark.parametrize("m", [-2, -1, 0, 1, 2])
    def test_different_m_modes(self, m):
        """Test different azimuthal numbers."""
        a = 0.5
        l = 2
        mode_jqnm = jqnm.modes_cache(s=-2, l=l, m=m, n=0)
        mode_orig = qnm_original.modes_cache(s=-2, l=l, m=m, n=0)
        
        omega_jqnm, A_jqnm, _ = mode_jqnm(a=a)
        omega_orig, A_orig, _ = mode_orig(a=a)
        
        omega_jqnm = complex(omega_jqnm)
        
        assert np.allclose(omega_jqnm, omega_orig, rtol=1e-6), \
            f"m={m} omega mismatch: jqnm={omega_jqnm}, original={omega_orig}"

    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_overtones(self, n):
        """Test different overtone numbers."""
        a = 0.5
        mode_jqnm = jqnm.modes_cache(s=-2, l=2, m=2, n=n)
        mode_orig = qnm_original.modes_cache(s=-2, l=2, m=2, n=n)
        
        omega_jqnm, _, _ = mode_jqnm(a=a)
        omega_orig, _, _ = mode_orig(a=a)
        
        omega_jqnm = complex(omega_jqnm)
        
        assert np.allclose(omega_jqnm, omega_orig, rtol=1e-5), \
            f"n={n} omega mismatch: jqnm={omega_jqnm}, original={omega_orig}"

