import pytest
import jqnm
import numpy as np

try:
    from pathlib import Path  # py 3
except ImportError:
    from pathlib2 import Path  # py 2


class JqnmTestDownload(object):
    """
    Base class so that each test will automatically download_data
    """

    @classmethod
    def setup_class(cls):
        """
        Download the data when setting up the test class.
        """
        jqnm.download_data()


class TestJqnmFileOps(JqnmTestDownload):
    def test_cache_file_operations(self):
        """Test file operations and downloading the on-disk cache."""

        print("Downloading with overwrite=True")
        jqnm.cached.download_data(overwrite=True)
        print("Clearing disk cache but not tarball")
        jqnm.cached._clear_disk_cache(delete_tarball=False)
        print("Decompressing tarball")
        jqnm.cached._decompress_data()


class TestJqnmOneMode(JqnmTestDownload):
    def test_one_mode(self):
        """
        An example of a test
        """
        grav_220 = jqnm.modes_cache(s=-2, l=2, m=2, n=0)
        omega, A, C = grav_220(a=0.68)
        assert np.allclose(omega, (0.5239751042900845 - 0.08151262363119974j))


class TestJqnmNewLeaverSolver(JqnmTestDownload):
    def test_compare_old_new_Leaver(self):
        """Check consistency between old and new Leaver solvers"""
        from jqnm.radial import leaver_cf_inv_lentz_old, leaver_cf_inv_lentz

        old = leaver_cf_inv_lentz_old(
            omega=0.4 - 0.2j, a=0.02, s=-2, m=2, A=4.0 + 0.0j, n_inv=0
        )
        new = leaver_cf_inv_lentz(
            omega=0.4 - 0.2j, a=0.02, s=-2, m=2, A=4.0 + 0.0j, n_inv=0
        )
        # Compare continued fraction values (allowing for numerical differences)
        assert np.allclose(old[0], new[0], rtol=1e-6)


class TestJqnmSolveInterface(JqnmTestDownload):
    """
    Test the various interface options for solving
    """

    def test_interp_only(self):
        """Check that we get reasonable values (but not identical!)
        with just interpolation.
        """

        grav_220 = jqnm.modes_cache(s=-2, l=2, m=2, n=0)
        a = 0.68
        assert a not in grav_220.a

        omega_int, A_int, C_int = grav_220(a=a, interp_only=True)
        omega_sol, A_sol, C_sol = grav_220(a=a, interp_only=False, store=False)

        assert np.allclose(omega_int, omega_sol) and not np.equal(omega_int, omega_sol)
        assert np.allclose(A_int, A_sol) and not np.equal(A_int, A_sol)
        assert np.allclose(C_int, C_sol) and not all(np.equal(C_int, C_sol))

    def test_store_a(self):
        """Check that the option store=True updates a spin sequence"""

        grav_220 = jqnm.modes_cache(s=-2, l=2, m=2, n=0)

        old_n = len(grav_220.a)
        k = int(old_n / 2)

        new_a = 0.5 * (grav_220.a[k] + grav_220.a[k + 1])

        assert new_a not in grav_220.a

        _, _, _ = grav_220(new_a, store=False)
        n_1 = len(grav_220.a)
        assert old_n == n_1

        _, _, _ = grav_220(new_a, store=True)
        n_2 = len(grav_220.a)
        assert n_2 == n_1 + 1

    def test_resolve(self):
        """Test that option resolve_if_found=True really does a new
        solve"""

        grav_220 = jqnm.modes_cache(s=-2, l=2, m=2, n=0)

        n = len(grav_220.a)
        k = int(n / 2)
        a = grav_220.a[k]

        grav_220.solver.solved = False
        omega_old, A_old, C_old = grav_220(a=a, resolve_if_found=False)
        solved_1 = grav_220.solver.solved

        omega_new, A_new, C_new = grav_220(a=a, resolve_if_found=True)
        solved_2 = grav_220.solver.solved

        assert (solved_1 is False) and (solved_2 is True)
        assert np.allclose(omega_new, omega_old)
        assert np.allclose(A_new, A_old)
        assert np.allclose(C_new, C_old)


class TestMirrorModeTransformation(JqnmTestDownload):
    @pytest.mark.parametrize(
        "s, l, m, n, a",
        [
            (-2, 2, 2, 0, 0.1),  # Low spin
            (-2, 2, 2, 0, 0.9),  # High spin
            (-2, 2, 2, 4, 0.7),  # Different overtone
            (-2, 3, 2, 0, 0.7),  # l odd
            (-2, 3, 1, 0, 0.7),  # l and m odd
            (-1, 3, 1, 0, 0.7),  # s, l, and m odd
        ],
    )
    def test_mirror_mode_transformation(self, s, l, m, n, a):
        import copy

        mode = jqnm.modes_cache(s=s, l=l, m=m, n=n)
        om, A, C = mode(a=a)

        # Convert to numpy for .conj() method compatibility
        A_np = np.complex128(A)
        C_np = np.array(C)

        solver = copy.deepcopy(
            mode.solver
        )  # need to import copy -- don't want to actually modify this mode's solver
        solver.clear_results()
        solver.set_params(
            a=a, m=-m, A_closest_to=np.conj(A_np), omega_guess=-np.conj(om)
        )
        om_prime = solver.do_solve()

        assert np.allclose(-np.conj(om), solver.omega)
        assert np.allclose(np.conj(A_np), solver.A)
        assert np.allclose(
            (-1) ** (l + jqnm.angular.ells(s, m, mode.l_max)) * np.conj(C_np), solver.C
        )


@pytest.mark.slow
class TestJqnmBuildCache(JqnmTestDownload):
    def test_build_cache(self):
        """Check the default cache-building functionality"""

        jqnm.cached._clear_disk_cache(delete_tarball=False)
        jqnm.modes_cache.seq_dict = {}
        jqnm.cached.build_package_default_cache(jqnm.modes_cache)
        assert 860 == len(jqnm.modes_cache.seq_dict.keys())
        jqnm.modes_cache.write_all()
        cache_data_dir = jqnm.cached.get_cachedir() / "data"

        # Magic number, default num modes is 860
        assert 860 == len(list(cache_data_dir.glob("*.pickle")))
