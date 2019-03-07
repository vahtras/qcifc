import pytest
import numpy as np
import numpy.testing as npt

from . import TestQC, get_codes_settings, get_codes_ids

CASE = 'ts01'

codes_settings = get_codes_settings(CASE)
ids = get_codes_ids()


@pytest.mark.parametrize('code', codes_settings, indirect=True, ids=ids)
class TestTS01(TestQC):

    def test_get_orbhess(self, code):
        """Get diagonal orbital hessian"""
        self.skip_if_not_implemented('get_orbital_diagonal', code)
        od = code.get_orbital_diagonal()
        npt.assert_allclose(
            od,
            [
                207.63510426,  21.00522587,  15.89399134,  15.88200483,
                +15.88200483,   2.29411362,   1.41453738,   1.07335853,
                + 1.07335853, 416.9399669,   43.68021011,  33.45774106,
                +33.43376805,  33.43376805,   6.25798563,   4.49883315,
                + 3.81647545,   3.81647545,   2.0435971,
            ]*2
        )


    def test_get_s2_diagonal(self, code):
        """Get diagonal overlap hessian"""
        self.skip_if_not_implemented('get_overlap_diagonal', code)
        sd = code.get_overlap_diagonal()
        npt.assert_allclose(
            sd,
            [
                1., 1., 1., 1., 1., 1., 1., 1., 1.,
                2., 2., 2., 2., 2., 2., 2., 2., 2., 1.,
                -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                -2., -2., -2., -2., -2., -2., -2., -2., -2., -1.
            ]
        )


    def test_get_rhs(self, code):
        """Get property gradient right-hand side"""
        self.skip_if_not_implemented('get_rhs', code)
        rhs,  = code.get_rhs('z',)
        expected = np.array([
            -6.57989324e-19,  4.62112979e-18,  3.46230723e-20,
             2.51740485e-12, -3.41544066e-03,  -6.31689618e-17,
            -2.21892286e-17,  1.11583016e-16, -3.25372463e-02,
            +1.77519347e-19,  1.39030993e-17, -1.30975112e-18,
            -2.27296296e-11,  3.08379878e-02, -1.07341703e-16,
            -1.20667401e-17, -9.76877341e-16,  2.84863611e-01,
            +8.22187178e-18,  6.57989324e-19, -4.62112979e-18,
            -3.46230723e-20,  -2.51740485e-12, 3.41544066e-03,
            +6.31689618e-17,  2.21892286e-17, -1.11583016e-16,
             3.25372463e-02, -1.77519347e-19, -1.39030993e-17,
            +1.30975112e-18,  2.27296296e-11, -3.08379878e-02,
            +1.07341703e-16,  1.20667401e-17,  9.76877341e-16,
            -2.84863611e-01, -8.22187178e-18
           ])
        npt.assert_allclose(rhs, expected, atol=1e-8)


    @pytest.mark.parametrize(
        'args',
        [
            (
                'z', (0.0,),
                {('z', 0.0): [
                    -3.16896955e-21,  2.19999053e-19,  2.17837493e-21,
                     1.58506743e-13, -2.15050977e-04, -2.75352368e-17,
                    -1.56865621e-17,  1.03956891e-16, -3.03134929e-02,
                    +4.25767164e-22,  3.18292866e-19, -3.91464300e-20,
                     -6.79840502e-13,+9.22360524e-04, -1.71527564e-17,
                    -2.68219329e-18, -2.55963219e-16,  7.46404934e-02,
                    +4.02323520e-18,
                    +3.16896955e-21, -2.19999053e-19, -2.17837493e-21,
                    -1.58506743e-13,  2.15050977e-04,  2.75352368e-17,
                    +1.56865621e-17, -1.03956891e-16,  3.03134929e-02,
                    -4.25767164e-22, -3.18292866e-19,  3.91464300e-20,
                      6.79840502e-13,-9.22360524e-04,  1.71527564e-17,
                    +2.68219329e-18,  2.55963219e-16, -7.46404934e-02,
                    -4.02323520e-18
                ]}
            ),
            (
                'z', (0.5,),
                {
                    ('z', 0.5): [
                        -3.17661908e-21,  2.25363516e-19,  2.24912900e-21,
                         1.63659086e-13, -2.22041320e-04,  -3.52090085e-17,
                        -2.42627901e-17,  1.94612985e-16, -5.67485166e-02,
                        +4.26790791e-22,  3.25750488e-19, -4.03525038e-20,
                        -7.00801387e-13, +9.50798802e-04, -2.04149860e-17,
                        -3.44878981e-18, -3.46843904e-16,  1.01141876e-01,
                        +5.32643641e-18,
                        +3.16135678e-21, -2.14884039e-19, -2.11193672e-21,
                        -1.53668912e-13,  2.08487343e-04, 2.26078715e-17,
                        +1.15898643e-17, -7.09202724e-17, 2.06801219e-02,
                        -4.24748436e-22, -3.11169066e-19,  3.80103593e-20,
                        6.60097077e-13,  -8.95574012e-04,  1.47894620e-17,
                        +2.19441830e-18,  2.02819956e-16, -5.91435820e-02,
                        -3.23237976e-18
                         ],
                }
            ),
        ],
        ids=['0.0', '0.5']
    )
    def test_initial_guess(self, code, args):
        """form paired trialvectors from rhs/orbdiag"""
        self.skip_if_not_implemented('initial_guess', code)
        self.skip_if_not_implemented('get_orbital_diagonal', code)
        self.skip_if_not_implemented('get_overlap_diagonal', code)
        ops, freqs, expected = args
        initial_guess = code.initial_guess(ops, freqs)
        for op, freq in zip(ops, freqs):
            print(initial_guess[(op, freq)] - expected[(op, freq)])
            npt.assert_allclose(
                initial_guess[(op, freq)],
                expected[(op, freq)],
                atol=1e-5,
            )

    @pytest.mark.parametrize(
        'args',
        [
            ('x', 'x', (0,), {('x', 'x', 0): -1.545996633923e+01}),
            ('y', 'y', (0,), {('y', 'y', 0): -1.745339237129e-01}),
            ('z', 'z', (0,), {('z', 'z', 0): -1.745339237129e-01}),
            ('x', 'x', (0.5,), {('x', 'x', 0.5): -2.270755038893e+01}),
            ('y', 'y', (0.5,), {('y', 'y', 0.5): -2.072139155315e-01}),
            ('z', 'z', (0.5,), {('z', 'z', 0.5): -2.072139155315e-01}),
        ],
        ids=['xx0', 'yy0', 'zz0', 'xx0.5', 'yy0.5', 'zz0.5']
    )
    def test_lr(self, code, args):
        self.skip_if_not_implemented('lr', code)
        aops, bops, freqs, expected = args
        lr = code.lr(aops, bops, freqs)
        for k, v in lr.items():
            npt.assert_allclose(v, expected[k], rtol=1e-4)
