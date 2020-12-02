import pytest
import numpy as np
import numpy.testing as npt

from . import TestQC, get_codes_settings, get_codes_ids

CASE = 'heh'

codes_settings = get_codes_settings(CASE)
ids = get_codes_ids()


@pytest.mark.parametrize('code', codes_settings, indirect=True, ids=ids)
class TestHeH(TestQC):

    @pytest.mark.skip()
    def test_excitations(self, code):
        excitations = list(code.get_excitations())
        assert excitations == [
            (0, 1),
        ]

    def test_vec2mat(self, code):
        self.skip_if_not_implemented('vec2mat', code)
        vec = np.array([1., 2.])
        npt.assert_allclose(
            code.vec2mat(vec),
            [
                [0.0, 1.0],
                [2.0, 0.0],
            ]
        )

    def test_mat2vec(self, code):
        self.skip_if_not_implemented('mat2vec', code)
        mat = [
                [0.0, 1.0],
                [2.0, 0.0],
              ]
        npt.assert_allclose(code.mat2vec(mat), [1., 2.])

    def test_get_orbhess(self, code):
        """Get diagonal orbital hessian"""
        self.skip_open_shell(code)
        self.skip_if_not_implemented('get_orbital_diagonal', code)
        od = code.get_orbital_diagonal()
        npt.assert_allclose(
            od,
            [
               1.19562374
            ]*2,
            atol=1e-5
        )

    def test_get_number_of_electrons(self, code):
        assert code.get_number_of_electrons() == 3

    def test_get_s2_diagonal(self, code):
        """Get diagonal overlap hessian"""
        self.skip_open_shell(code)
        self.skip_if_not_implemented('get_overlap_diagonal', code)
        sd = code.get_overlap_diagonal()
        npt.assert_allclose(sd, [1, -1])

    def test_get_overlap(self, code):
        """
        Get overlap matrix

        Two conventions cartesian xyz/spherical -1,0,1
        """
        self.skip_if_not_implemented('get_overlap', code)
        S = code.get_overlap()
        Sref = [
            [1, .084253],
            [.084253, 1],
        ]
        npt.assert_allclose(S, Sref, atol=1e-6)

    def test_get_dipole(self, code):
        """Get dipole matrix"""
        self.skip_if_not_implemented('get_dipole', code)
        x, y, z = code.get_dipole()
        yref = [
            [-2.446859, -0.211067],
            [-0.211067, -2.611603],
        ]
        npt.assert_allclose(y, yref, atol=1e-6)

    def test_get_rhs(self, code):
        """Get property gradient right-hand side"""
        rhs,  = code.get_rhs('z',)
        expected = np.array([
            0.001333, -0.001333
        ])
        # absolute value because of different phase of mo:s
        npt.assert_allclose(abs(rhs), abs(expected), atol=1e-5)

    @pytest.mark.parametrize(
        'args',
        [
            (
               'z', (0.0,),
               {('z', 0.0):
                   np.array([
                       0.0005575,
                       -0.0005575,
                   ])
                }
            ),
        ],
        ids=['0.0']
    )
    def test_initial_guess(self, code, args):
        """form paired trialvectors from rhs/orbdiag"""
        self.skip_open_shell(code)
        ops, freqs, expected = args
        initial_guess = code.initial_guess(
            ops, freqs, hessian_diagonal_shift=0
        )
        for op, freq in zip(ops, freqs):
            npt.assert_allclose(
                abs(initial_guess[(op, freq)]),
                2*abs(expected[(op, freq)]),
                atol=1e-5
            )

    @pytest.mark.parametrize(
        'args',
        [
            (
                'xyz', 'xyz', (0,),
                {
                    ('x', 'x', 0): -7.079369443557E-02,
                    ('x', 'y', 0): -3.270003834531E-03,
                    ('x', 'z', 0): -5.239642983352E-04,
                    ('y', 'x', 0): -3.270003834531E-03,
                    ('y', 'y', 0): -1.510434674034E-04,
                    ('y', 'z', 0): -2.420222985075E-05,
                    ('z', 'x', 0): -5.239642983352E-04,
                    ('z', 'y', 0): -2.420222985075E-05,
                    ('z', 'z', 0): -3.878009024939E-06,
                }
            ),
            (
                'xyz', 'xyz', (0.5,),
                {
                    ('x', 'x', 0.5): -1.008512113896E-01,
                    ('x', 'y', 0.5): -4.658378837133E-03,
                    ('x', 'z', 0.5): -7.464285432950E-04,
                    ('y', 'x', 0.5): -4.658378837133E-03,
                    ('y', 'y', 0.5): -2.151733538075E-04,
                    ('y', 'z', 0.5): -3.447798872821E-05,
                    ('z', 'x', 0.5): -7.464285432950E-04,
                    ('z', 'y', 0.5): -3.447798872821E-05,
                    ('z', 'z', 0.5): -5.524530271561E-06,
                }
            ),
            (
                '', '', (),
                {
                }
            ),
        ],
        ids=['0', '0.5', 'none']
    )
    def test_lr(self, code, args):
        aops, bops, freqs, expected = args
        lr = code.lr(aops, bops, freqs)
        for k, v in lr.items():
            npt.assert_allclose(v, expected[k], atol=1e-4)

    @pytest.mark.skip()
    def test_initial_excitation(self, code):
        expected = [
            (
                0.35994304,
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] +
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            )
        ]
        calculated = code.initial_excitations(1)
        w1, X1 = expected[0]
        w2, X2 = calculated[0]
        assert w1 == pytest.approx(w2)
        npt.assert_allclose(X1, X2, atol=1e07)

    @pytest.mark.skip()
    def test_excitation_energies(self, code):
        self.skip_if_not_implemented('excitation_energies', code)

        w = code.excitation_energies(3)
        npt.assert_allclose(w, [0.16179567, 0.22270771, 0.22270771], atol=1e-5)

    @pytest.mark.skip()
    def test_eigenvectors(self, code):
        self.skip_if_not_implemented('eigenvectors', code)

        X = code.eigenvectors(1)
        npt.assert_allclose(abs(X[4, 0]), 0.7078520478, atol=1e-5)
        npt.assert_allclose(abs(X[4+8, 0]), 0.0254907063, atol=1e-5)

    @pytest.mark.skip()
    @pytest.mark.parametrize(
        'data',
        [
            (
                1,
                [
                    (0.16179567, [-0.7078520478, -0.0254907063])
                ],
            ),
            (
                0,
                [
                ],
            ),
        ],
        ids=['1', '0'],
    )
    def test_pp_solve(self, code, data):
        self.skip_if_not_implemented('pp_solve', code)
        n, expected = data

        eigensolutions = list(code.pp_solve(n))

        assert len(eigensolutions) == len(expected)

        for (w, X), (wref, Xref) in zip(eigensolutions, expected):

            assert w == pytest.approx(wref)
            try:
                npt.assert_allclose(X[[4, 12]], Xref, atol=1e-5)
            except AssertionError:
                npt.assert_allclose(-X[[4, 12]], Xref, atol=1e-5)

    @pytest.mark.skip()
    def test_transition_moments(self, code):
        self.skip_if_not_implemented('transition_moments', code)
        transition_moments = code.transition_moments('xyz', 3)
        npt.assert_allclose(
            np.abs(transition_moments['z']),
            [0.54692108, 0, 0],
            atol=1e-5
        )

    @pytest.mark.skip()
    def test_oscillator_strengths(self, code):
        self.skip_if_not_implemented('oscillator_strengths', code)
        oscillator_strengths = code.oscillator_strengths(3)['I']
        npt.assert_allclose(
            oscillator_strengths,
            [0.0322645017, 0.2543398498, 0.2543398498],
            atol=1e-5
        )

    def test_roothan_rohf_final(self, code):
        code.set_roothan_iterator(
            CASE,
            electrons=3,
            max_iterations=10,
            threshold=1e-5,
            tmpdir=code.get_workdir(),
            ms=1/2,
        )
        final_energy, final_norm = code.run_roothan_iterations()
        assert final_energy == pytest.approx(-3.269722925573)
        assert final_norm < 1e-5

    def test_scf_roothan_final(self, code):
        code.set_scf_iterator(
            'roothan',
            electrons=3,
            max_iterations=10,
            threshold=1e-5,
            tmpdir=code.get_workdir(),
            ms=1/2,
        )
        final_energy, final_norm = list(code.scf)[-1]
        assert final_energy == pytest.approx(-3.269722925573)
        assert final_norm < 1e-5

    def test_scf_diis_final(self, code):
        code.set_scf_iterator(
            'diis',
            electrons=3,
            max_iterations=10,
            threshold=1e-5,
            tmpdir=code.get_workdir(),
            ms=1/2,
        )
        final_energy, final_norm = list(code.scf)[-1]
        assert final_energy == pytest.approx(-3.269722925573)
        assert final_norm < 1e-5

    def test_roothan_rohf_initial_energy(self, code):
        code.set_roothan_iterator(
            CASE,
            electrons=3,
            max_iterations=10,
            threshold=1e-5,
            tmpdir=code.get_workdir(),
            ms=1/2,
        )
        initial_energy, _ = next(iter(code.roothan))
        assert initial_energy == pytest.approx(-3.26919092387)

    def test_roothan_scf_initial_energy(self, code):
        code.set_scf_iterator(
            'roothan',
            electrons=3,
            max_iterations=10,
            threshold=1e-5,
            tmpdir=code.get_workdir(),
            ms=1/2,
        )
        initial_energy, _ = next(iter(code.scf))
        assert initial_energy == pytest.approx(-3.26919092387)

    def test_diis_scf_initial_energy(self, code):
        code.set_scf_iterator(
            'diis',
            electrons=3,
            max_iterations=10,
            threshold=1e-5,
            tmpdir=code.get_workdir(),
            ms=1/2,
        )
        initial_energy, _ = next(iter(code.scf))
        assert initial_energy == pytest.approx(-3.26919092387)

    def test_roothan_rohf_initial_mo(self, code):
        code.set_roothan_iterator(
            CASE,
            electrons=3,
            max_iterations=10,
            threshold=1e-5,
            tmpdir=code.get_workdir(),
            ms=1/2,
        )
        next(iter(code.roothan))
        npt.assert_allclose(
            code.roothan.Ca,
            [[-1.000293534043742882e+00, 8.100760368342355133e-02],
             [3.558560208483147521e-03, -1.003562017392793715e+00]],
            atol=1e-8
        )

    @pytest.mark.skip()
    def test_roothan_uhf(self, code):
        final_energy, final_norm = code.run_uroothan_iterations(
            CASE,
            electrons=3,
            max_iterations=10,
            threshold=1e-5,
            tmpdir=code.get_workdir(),
            ms=1/2,
        )
        assert final_energy == pytest.approx(-3.2697229256)
        assert final_norm < 1e-5
