import pytest
import numpy as np
import numpy.testing as npt

from . import TestQC, get_codes_settings, get_codes_ids

CASE = 'lih'

codes_settings = get_codes_settings(CASE)
ids = get_codes_ids()


@pytest.mark.parametrize('code', codes_settings, indirect=True, ids=ids)
class TestLiH(TestQC):

    def test_excitations(self, code):
        excitations = list(code.get_excitations())
        assert excitations == [
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
        ]

    def test_vec2mat(self, code):
        self.skip_if_not_implemented('vec2mat', code)
        lz = list(range(8))
        ly = list(range(8, 16))
        vec = np.array(lz + ly)
        npt.assert_allclose(
            code.vec2mat(vec),
            [
                [0,  0, 0, 1, 2, 3],
                [0,  0, 4, 5, 6, 7],
                [8, 12, 0, 0, 0, 0],
                [9, 13, 0, 0, 0, 0],
                [10, 14, 0, 0, 0, 0],
                [11, 15, 0, 0, 0, 0],
            ]
        )

    def test_mat2vec(self, code):
        self.skip_if_not_implemented('mat2vec', code)
        mat = [
                [0,  0, 0, 1, 2, 3],
                [0,  0, 4, 5, 6, 7],
                [8, 12, 0, 0, 0, 0],
                [9, 13, 0, 0, 0, 0],
                [10, 14, 0, 0, 0, 0],
                [11, 15, 0, 0, 0, 0],
              ]
        npt.assert_allclose(code.mat2vec(mat), range(16))

    def test_get_orbhess(self, code):
        """Get diagonal orbital hessian"""
        self.skip_if_not_implemented('get_orbital_diagonal', code)
        od = code.get_orbital_diagonal()
        npt.assert_allclose(
            od*2,
            [
                9.71017135, 10.05445847, 10.05445847, 11.54743296,
                1.43977221,  1.78405933, 1.78405933,  3.27703381,
                9.71017135, 10.05445847, 10.05445847, 11.54743296,
                1.43977221,  1.78405933,  1.78405933,  3.27703381,
            ],
            atol=1e-6
        )

    def test_get_s2_diagonal(self, code):
        """Get diagonal overlap hessian"""
        self.skip_if_not_implemented('get_overlap_diagonal', code)
        sd = code.get_overlap_diagonal()
        lsd = len(sd)
        npt.assert_allclose(sd, [2.0]*(lsd//2) + [-2.0]*(lsd//2))

    def test_get_overlap(self, code):
        """
        Get overlap matrix

        Two conventions cartesian xyz/spherical -1,0,1
        """
        self.skip_if_not_implemented('get_overlap', code)
        S = code.get_overlap()
        Sref1 = [
          [1.00000000, 0.06142646, 0.38597576, 0.00000000, 0.00000000, 0.50572609],
          [0.06142646, 1.00000000, 0.24113665, 0.00000000, 0.00000000, 0.00000000],
          [0.38597576, 0.24113665, 1.00000000, 0.00000000, 0.00000000, 0.00000000],
          [0.00000000, 0.00000000, 0.00000000, 1.00000000, 0.00000000, 0.00000000],
          [0.00000000, 0.00000000, 0.00000000, 0.00000000, 1.00000000, 0.00000000],
          [0.50572609, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 1.00000000],
        ]
        Sref2 = [
          [1.00000000, 0.06142646, 0.38597576, 0.00000000, 0.50572609, 0.00000000],
          [0.06142646, 1.00000000, 0.24113665, 0.00000000, 0.00000000, 0.00000000],
          [0.38597576, 0.24113665, 1.00000000, 0.00000000, 0.00000000, 0.00000000],
          [0.00000000, 0.00000000, 0.00000000, 1.00000000, 0.00000000, 0.00000000],
          [0.50572609, 0.00000000, 0.00000000, 0.00000000, 1.00000000, 0.00000000],
          [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 1.00000000],
        ]
        try:
            npt.assert_allclose(S, Sref1, atol=1e-6)
        except AssertionError:
            npt.assert_allclose(S, Sref2, atol=1e-6)


    def test_get_dipole(self, code):
        """Get dipole matrix"""
        self.skip_if_not_implemented('get_dipole', code)
        x, y, z = code.get_dipole()
        xref = [
            [0.00000000, 0.00000000, 0.00000000, 0.31713290, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 0.14723998, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 1.80329832, 0.00000000, 0.00000000],
            [0.31713290, 0.14723998, 1.80329832, 0.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
        ]
        try:  # cover cartesian/spherical cases
            npt.assert_allclose(x, xref, atol=1e-6)
        except AssertionError:
            npt.assert_allclose(y, xref, atol=1e-6)



    def test_get_rhs(self, code):
        """Get property gradient right-hand side"""
        rhs,  = code.get_rhs('z',)
        expected = np.array([
            0.28808650,
            0,
            0,
           -0.24183442,
           -0.90722391,
            0,
            0,
           -1.48001444,
           -0.28808650,
            0,
            0,
            0.24183442,
            0.90722391,
            0,
            0,
            1.48001444,
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
                     2.96685289e-02,  8.20729443e-18,  8.34980544e-17,
                    -2.09426998e-02, -6.30116279e-01, -4.43025394e-17,
                    -1.02185072e-15, -4.51632338e-01, -2.96685289e-02,
                    -8.20729443e-18, -8.34980544e-17,  2.09426998e-02,
                     6.30116279e-01,  4.43025394e-17,  1.02185072e-15,
                     4.51632338e-01
                   ])
                }
            ),
        ],
        ids=['0.0']
    )
    def test_initial_guess(self, code, args):
        """form paired trialvectors from rhs/orbdiag"""
        ops, freqs, expected = args
        initial_guess = code.initial_guess(ops, freqs, hessian_diagonal_shift=0)
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
                    ('x', 'x', 0): -15.468383,
                    ('x', 'y', 0): 0,
                    ('x', 'z', 0): 0,
                    ('y', 'x', 0): 0,
                    ('y', 'y', 0): -15.468383,
                    ('y', 'z', 0): 0,
                    ('z', 'x', 0): 0,
                    ('z', 'y', 0): 0,
                    ('z', 'z', 0): -6.685486,
                }
            ),
            (
                'xyz', 'xyz', (0.5,),
                {
                    ('x', 'x', 0.5): 3.718038,
                    ('x', 'y', 0.5): 0,
                    ('x', 'z', 0.5): 0,
                    ('y', 'x', 0.5): 0,
                    ('y', 'y', 0.5): 3.718038,
                    ('y', 'z', 0.5): 0,
                    ('z', 'x', 0.5): 0,
                    ('z', 'y', 0.5): 0,
                    ('z', 'z', 0.5): -9.065174,
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

