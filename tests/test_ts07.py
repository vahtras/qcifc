import pytest

import numpy.testing as npt

from . import TestQC, get_codes_settings, get_codes_ids

CASE = 'ts07'

codes_settings = get_codes_settings(CASE)
ids = get_codes_ids()


@pytest.mark.parametrize('code', codes_settings, indirect=True, ids=ids)
class TestTS07(TestQC):
    @pytest.mark.skip()
    def test_get_orbhess(self, code):
        self.skip_if_not_implemented('get_orbital_diagonal', code)
        """Get diagonal orbital hessian"""
        od = code.get_orbital_diagonal()
        npt.assert_allclose(
            od,
            [
               19.58513690, 15.16979761, 0.91442880, 0.99810181, 0.37378880,
                0.52961076, 0.24252438, 0.11137636, 0.29099060, 40.44354049,
               41.50266802, 41.80152641, 41.45490650, 31.14954530, 30.62912178,
               30.50671567, 30.86750710, 2.91814149, 3.46933801, 3.69211856,
                3.55114767, 2.78607936, 2.75385386, 2.72556071, 2.94129552,
                1.84081782, 2.15051627, 2.29341203, 2.33285820, 1.81697055,
                1.73205161, 2.03064438, 1.99202475, 1.48606631, 1.83631374,
                1.99590938, 2.04727472, 1.20677249, 1.67572444, 1.86316594,
                1.87454393, 1.35059850, 1.20246862, 1.22058581, 1.43630087,
                0.57555841, 0.90149656, 1.01195131, 0.93229178,
            ]*2
        )

    def test_get_s2_diagonal(self, code):
        """Get diagonal overlap hessian"""
        self.skip_if_not_implemented('get_overlap_diagonal', code)
        sd = code.get_overlap_diagonal()
        expected = [1.]*9 + [2.]*36 + [1.]*4
        expected += [-1.]*9 + [-2.]*36 + [-1.]*4
        npt.assert_allclose(sd, expected)

    def test_get_rhs(self, code):
        """Get property gradient right-hand side"""
        self.skip_if_not_implemented('get_rhs', code)
        rhs,  = code.get_rhs('z',)
        expected = [
            0.02374939, -0.00507364, -0.21746464, -0.01346780, -0.06384615,
            0.04179650, -0.15581620, -0.13997110, 0.16153586, 0.00091128,
            0.00057740, -0.00024758, 0.00280781, 0.00084879, 0.05510469,
            0.09241263, -0.05708850, 0.01220453, 0.00694466, -0.00008657,
            0.04720088, 0.02193391, 0.15200688, 0.21916085, -0.10645213,
           -0.12268620, -0.22834393, -0.26967534, -0.28132094, 0.25433727,
            1.14003801, 1.11314992, -0.28249752, -0.09423784, -0.16437619,
           -0.21591672, -0.38589896, -0.26828002, -0.41129342, -0.49330389,
           -0.30350910, 0.05853996, -0.22214696, -0.55807650, 0.33904318,
           -0.03798739, -0.08858959, -0.06870476, -0.11945265, -0.02374939,
            0.00507364, 0.21746464, 0.01346780, 0.06384615, -0.04179650,
            0.15581620, 0.13997110, -0.16153586, -0.00091128, -0.00057740,
            0.00024758, -0.00280781, -0.00084879, -0.05510469, -0.09241263,
            0.05708850, -0.01220453, -0.00694466, 0.00008657, -0.04720088,
           -0.02193391, -0.15200688, -0.21916085, 0.10645213, 0.12268620,
            0.22834393, 0.26967534, 0.28132094, -0.25433727, -1.14003801,
           -1.11314992, 0.28249752, 0.09423784, 0.16437619, 0.21591672,
            0.38589896, 0.26828002, 0.41129342, 0.49330389, 0.30350910,
           -0.05853996, 0.22214696, 0.55807650, -0.33904318, 0.03798739,
            0.08858959, 0.06870476, 0.11945265,
        ]
        npt.assert_allclose(rhs, expected, atol=1e-8)

    @pytest.mark.skip()
    @pytest.mark.parametrize(
        'args',
        [
            (
                'z', (0.0,),
                {
                    ('z', 0.0):
                    [
                    ]
                }
            ),
            (
                'z', (0.5,),
                {
                    ('z', 0.5):
                    [
                    ],
                }
            ),
        ],
        ids=['0.0', '0.5']
    )
    def test_initial_guess(self, code, args):
        """form paired trialvectors from rhs/orbdiag"""
        self.skip_if_not_implemented('initial_guess', code)
        ops, freqs, expected = args
        initial_guess = code.initial_guess(ops, freqs)
        for op, freq in zip(ops, freqs):
            npt.assert_allclose(
                initial_guess[(op, freq)],
                expected[(op, freq)],
                rtol=1e-5,
            )

    @pytest.mark.parametrize(
        'args',
        [
            ('x', 'x', (0,), {('x', 'x', 0): -7.306549913351e+00}),
            ('y', 'y', (0,), {('y', 'y', 0): -2.144798212184e+01}),
            ('z', 'z', (0,), {('z', 'z', 0): -6.539944849631e+00}),
            ('x', 'x', (0.5,), {('x', 'x', 0.5): -1.008026113219e+01}),
            ('y', 'y', (0.5,), {('y', 'y', 0.5):  4.551977412440e+00}),
            ('z', 'z', (0.5,), {('z', 'z', 0.5): -6.650252800056e+00}),
        ],
        ids=['xx0', 'yy0', 'zz0', 'xx0.5', 'yy0.5', 'zz0.5']
    )
    def test_lr(self, code, args):
        self.skip_if_not_implemented('lr', code)
        aops, bops, freqs, expected = args
        lr = code.lr(aops, bops, freqs)
        for k, v in lr.items():
            npt.assert_allclose(v, expected[k], rtol=1e-4)
