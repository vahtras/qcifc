import pytest
import numpy as np
import numpy.testing as npt

from . import TestQC, get_codes_settings, get_codes_ids

CASE = 'h2o'

codes_settings = get_codes_settings(CASE)
ids = get_codes_ids()


@pytest.mark.parametrize('code', codes_settings, indirect=True, ids=ids)
class TestH2O(TestQC):

    @pytest.mark.parametrize(
        'args',
        [
            (
                'xyz', 'xyz', (0,),
                {
                    ('x', 'x', 0): -3.046547105763,
                    ('x', 'y', 0): 0,
                    ('x', 'z', 0): 0,
                    ('y', 'x', 0): 0,
                    ('y', 'y', 0): -6.591935043775,
                    ('y', 'z', 0): 0,
                    ('z', 'x', 0): 0,
                    ('z', 'y', 0): 0,
                    ('z', 'z', 0): -4.980329678152,
                }
            ),
            (
                'xyz', 'xyz', (0.2,),
                {
                    ('x', 'x', 0.2): -3.495340131306,
                    ('x', 'y', 0.2): 0,
                    ('x', 'z', 0.2): 0,
                    ('y', 'x', 0.2): 0,
                    ('y', 'y', 0.2): -7.226826515191,
                    ('y', 'z', 0.2): 0,
                    ('z', 'x', 0.2): 0,
                    ('z', 'y', 0.2): 0,
                    ('z', 'z', 0.2): -5.518828539302,
                }
            ),
            (
                '', '', (),
                {
                }
            ),
        ],
        ids=['0', '0.2', 'none']
    )
    def test_lr(self, code, args):
        aops, bops, freqs, expected = args
        lr = code.lr(aops, bops, freqs)
        for k, v in lr.items():
            npt.assert_allclose(v, expected[k], atol=1e-4)

    def test_excitation_energies(self, code):
        self.skip_if_not_implemented('excitation_energies', code)

        w = code.excitation_energies(3)
        npt.assert_allclose(w, [0.34252829, 0.40843353, 0.43986599], atol=1e-5)

    @pytest.mark.parametrize(
        'args',
        [
            (
                'x', 1,
                {
                    'x': [0.37065028],
                }
            ),
            (
                'xyz', 1,
                {
                    'x': [0.37065028],
                    'y': [0.],
                    'z': [0.],
                }
            ),
            (
                'x', 3,
                {
                    'x': [0.37065028, 0, 0],
                }
            ),
            (
                'xyz', 3,
                {
                    'x': [0.37065028, 0, 0],
                    'y': [0, 0, 0],
                    'z': [0, 0, 0.58815138],
                }
            ),
            (
                '', 0,
                {
                }
            ),
        ],
        ids=['x1', 'xyz1', 'x3', 'xyz3', 'none']
    )
    def test_transition_moments(self, code, args):
        self.skip_if_not_implemented('transition_moments', code)

        ops, roots, expected = args
        transition_moments = code.transition_moments(ops, roots)
        # remove the frequency key for these tests
        del transition_moments['w']
        for op, moment in transition_moments.items():
            npt.assert_allclose(
                np.abs(moment),
                expected[op],
                rtol=1e-4,
                atol=1e-5,
            )

    def test_oscillator_strengths(self, code):
        self.skip_if_not_implemented('oscillator_strengths', code)
        oscillator_strengths = code.oscillator_strengths(3)['I']
        npt.assert_allclose(
            oscillator_strengths,
            [3.13713973E-02, 0, 0.10143956],
            rtol=1e-4,
            atol=1e-5
        )
