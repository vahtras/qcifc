import pytest
import numpy as np
import numpy.testing as npt

from . import TestQC, get_codes_settings, get_codes_ids

CASE = 'ts02'

codes_settings = get_codes_settings(CASE)
ids = get_codes_ids()


@pytest.mark.parametrize('code', codes_settings, indirect=True, ids=ids)
class TestTS02(TestQC):

    def test_get_orbhess(self, code):
        """Get diagonal orbital hessian"""
        self.skip_if_not_implemented('get_orbital_diagonal', code)
        od = code.get_orbital_diagonal()
        npt.assert_allclose(
            od,
            [
                41.15162966,  3.05665099,  1.7610597,  1.53740381,  1.40776145,
                83.71247278, 84.45117178,  7.52251542,  8.26121442,  4.93133286,
                5.67003186,  4.48402107,  5.22272007,  4.22473636,  4.96343536,
                2.16508555,  2.53443505,
            ]*2
        )


    def test_get_s2_diagonal(self, code):
        """Get diagonal overlap hessian"""
        self.skip_if_not_implemented('get_overlap_diagonal', code)
        sd = code.get_overlap_diagonal()
        npt.assert_allclose(
            sd,
            [
                1., 1., 1., 1., 1., 2., 2., 2., 2., 2.,
                2., 2., 2., 2., 2., 1., 1.,
                -1., -1., -1., -1., -1., -2., -2., -2., -2.,
                -2., -2., -2., -2., -2., -2., -1., -1.
            ]
        )


    def test_get_rhs(self, code):
        """Get property gradient right-hand side"""
        self.skip_if_not_implemented('get_rhs', code)
        rhs,  = code.get_rhs('z',)
        expected = [
             +8.94435483e-17,  8.87286414e-16, -7.28540423e-17,
             +1.58013378e-16,  1.26794454e-02, -1.03948048e-17,
             -3.28434472e-18, -2.72446954e-17, -4.10615567e-18,
             +2.26716303e-16,  8.45837058e-17, -3.09199726e-16,
             -1.14952287e-16,  1.00618252e-01,  3.78492357e-02,
             +6.95956129e-17,  2.62418517e-17, -8.94435483e-17,
             -8.87286414e-16,  7.28540423e-17, -1.58013378e-16,
             -1.26794454e-02,  1.03948048e-17,  3.28434472e-18,
             +2.72446954e-17,  4.10615567e-18, -2.26716303e-16,
             -8.45837058e-17,  3.09199726e-16,  1.14952287e-16,
             -1.00618252e-01, -3.78492357e-02, -6.95956129e-17,
             -2.62418517e-17
        ]
        npt.assert_allclose(rhs, expected, atol=1e-8)


    @pytest.mark.parametrize(
        'args',
        [
            (
                'z', (0.0,),
                {('z', 0.0): [
                    +2.17351169e-18,  2.90280578e-16, -4.13694335e-17,
                    +1.02779359e-16,  9.00681388e-03, -1.24172712e-19,
                    -3.88904577e-20, -3.62175334e-18, -4.97040200e-19,
                    +4.59746502e-17,  1.49176773e-17, -6.89559039e-17,
                    -2.20100417e-17,  2.38164572e-02,  7.62561269e-03,
                    +3.21445095e-17,  1.03541228e-17, -2.17351169e-18,
                    -2.90280578e-16,  4.13694335e-17, -1.02779359e-16,
                    -9.00681388e-03,  1.24172712e-19,  3.88904577e-20,
                    +3.62175334e-18,  4.97040200e-19, -4.59746502e-17,
                    -1.49176773e-17,  6.89559039e-17,  2.20100417e-17,
                    -2.38164572e-02, -7.62561269e-03, -3.21445095e-17,
                    -1.03541228e-17
                ]}
            ),
            (
                'z', (0.5,),
                {
                    ('z', 0.5):
                    [
                        +2.20024508e-18,  3.47050269e-16, -5.77720802e-17,
                        +1.52316173e-16,  1.39678165e-02, -1.25673969e-19,
                        -3.93564842e-20, -4.17702276e-18, -5.65491588e-19,
                        +5.76690683e-17,  1.81120190e-17, -8.87479495e-17,
                        -2.72223318e-17,  3.12020089e-02,  9.54960337e-03,
                        +4.17970193e-17,  1.28988397e-17, -2.14742014e-18,
                        -2.49472444e-16,  3.22211936e-17, -7.75562394e-17,
                        -6.64624257e-03,  1.22706898e-19,  3.84353386e-20,
                        +3.19679039e-18,  4.43371191e-19, -3.82235003e-17,
                        -1.26811547e-17,  5.63819361e-17,  1.84729966e-17,
                        -1.92580535e-02, -6.34688454e-03, -2.61138382e-17,
                        -8.64801891e-18
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
                atol=1e-5,
            )


    @pytest.mark.parametrize(
        'args',
        [
            ('x', 'x', (0,), {('x', 'x', 0): -3.905850683643e+00}),
            ('y', 'y', (0,), {('y', 'y', 0): -1.142145071013e+01}),
            ('z', 'z', (0,), {('z', 'z', 0): -4.258206719769e-02}),
            ('x', 'x', (0.5,), {('x', 'x', 0.5): -5.509763897604e+00}),
            ('y', 'y', (0.5,), {('y', 'y', 0.5): 3.988082514830e+00}),
            ('z', 'z', (0.5,), {('z', 'z', 0.5): -2.554873449850e-01}),
        ],
        ids=['xx0', 'yy0', 'zz0', 'xx0.5', 'yy0.5', 'zz0.5']
    )
    def test_lr(self, code, args):
        self.skip_if_not_implemented('lr', code)
        aops, bops, freqs, expected = args
        lr = code.lr(aops, bops, freqs)
        for k, v in lr.items():
            npt.assert_allclose(v, expected[k], rtol=1e-4)
