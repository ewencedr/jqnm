"""Tests for JAX compatibility, differentiability, and GPU-readiness."""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
import jqnm

jax.config.update("jax_enable_x64", True)

class TestJaxAngular:
    def test_m_matrix_runs_on_jax(self):
        from jqnm.angular import M_matrix
        M = M_matrix(s=-2, c=0.5+0.1j, m=2, l_max=5)
        assert isinstance(M, jax.Array)
        assert M.shape == (4, 4)

    def test_sep_consts_differentiable(self):
        from jqnm.angular import sep_consts
        def loss(c_real, c_imag):
            return jnp.sum(jnp.abs(sep_consts(-2, c_real+1j*c_imag, 2, 5))**2)
        g = jax.grad(loss, argnums=(0, 1))(0.5, 0.1)
        assert np.all(np.isfinite(g))

    def test_sep_const_closest_differentiable(self):
        from jqnm.angular import sep_const_closest, swsphericalh_A
        def loss(c_real, c_imag):
            A0 = swsphericalh_A(-2, 2, 2)
            return jnp.abs(sep_const_closest(A0, -2, c_real+1j*c_imag, 2, 10))**2
        g = jax.grad(loss, argnums=(0, 1))(0.3, 0.05)
        assert np.all(np.isfinite(g))

class TestJaxRadial:
    def test_d_coeffs_differentiable(self):
        from jqnm.radial import D_coeffs
        def loss(o_r, o_i, a):
            return jnp.sum(jnp.abs(D_coeffs(o_r+1j*o_i, a, -2, 2, 4.0+0j))**2)
        g = jax.grad(loss, argnums=(0, 1, 2))(0.5, -0.1, 0.6)
        assert np.all(np.isfinite(g))

    def test_leaver_cf_differentiable(self):
        from jqnm.radial import leaver_cf_inv_lentz
        def loss(o_r, o_i):
            r, _, _ = leaver_cf_inv_lentz(o_r+1j*o_i, 0.02, -2, 2, 4.0+0j, 0)
            return jnp.abs(r)
        g = jax.grad(loss, argnums=(0, 1))(0.4, -0.2)
        assert np.all(np.isfinite(g))

class TestJaxRootFinding:
    def test_solve_qnm_with_A(self):
        from jqnm.nearby import solve_qnm_with_A
        omega, A, C = solve_qnm_with_A(
            0.52-0.08j, 0.68, 4.0+0j, -2, 2, 20, 0, 1e-10, 300, 500, 1e-10)
        assert np.allclose(omega, 0.5239751-0.0815126j, rtol=1e-5)

    def test_solve_qnm_differentiable_wrt_spin(self):
        from jqnm.nearby import solve_qnm_with_A
        def omega_real(a):
            omega, _, _ = solve_qnm_with_A(
                0.52-0.08j, a, 4.0+0j, -2, 2, 20, 0, 1e-10, 300, 500, 1e-10)
            return jnp.real(omega)
        g = jax.grad(omega_real)(0.68)
        assert np.isfinite(g) and g > 0

class TestJaxJitVmap:
    def test_d_coeffs_vmap(self):
        from jqnm.radial import D_coeffs
        omegas = jnp.array([0.5-0.1j, 0.6-0.2j])
        D = jax.vmap(lambda o: D_coeffs(o, 0.6, -2, 2, 4.0+0j))(omegas)
        assert D.shape == (2, 5)

class TestJaxNumericalAccuracy:
    def test_qnm_frequency_accuracy(self):
        grav_220 = jqnm.modes_cache(s=-2, l=2, m=2, n=0)
        omega, _, _ = grav_220(a=0.68)
        assert np.allclose(omega, 0.5239751042900845-0.08151262363119986j, atol=1e-10)

    def test_schwarzschild_limit(self):
        grav_220 = jqnm.modes_cache(s=-2, l=2, m=2, n=0)
        omega, _, _ = grav_220(a=0.0)
        assert np.allclose(omega, 0.373671671010979-0.08896216j, atol=1e-10)

class TestJaxDeviceCompatibility:
    def test_default_device(self):
        assert jax.default_backend() in ['cpu', 'gpu', 'tpu']

    def test_64bit_enabled(self):
        assert jax.config.jax_enable_x64

class TestGradientChain:
    def test_angular_to_radial_gradient(self):
        from jqnm.angular import sep_const_closest, swsphericalh_A
        from jqnm.radial import leaver_cf_inv_lentz
        def loss(o_r, o_i, a):
            omega = o_r + 1j*o_i
            c = a * omega
            A = sep_const_closest(swsphericalh_A(-2, 2, 2), -2, c, 2, 10)
            r, _, _ = leaver_cf_inv_lentz(omega, a, -2, 2, A, 0)
            return jnp.abs(r)**2
        g = jax.grad(loss, argnums=(0, 1, 2))(0.5, -0.1, 0.6)
        assert np.all(np.isfinite(g))

    def test_higher_order_derivatives(self):
        from jqnm.radial import D_coeffs
        def loss(o_r, o_i):
            return jnp.sum(jnp.abs(D_coeffs(o_r+1j*o_i, 0.6, -2, 2, 4.0+0j))**2)
        hess = jax.grad(jax.grad(loss, argnums=0), argnums=0)(0.5, -0.1)
        assert np.isfinite(hess)
