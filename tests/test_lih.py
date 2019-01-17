import pytest
import numpy as np
import numpy.testing as npt
from util import full
from .conftest import case_dir, case_fixture

CASE = 'lih'
tmpdir = case_dir(CASE)
mod = case_fixture(CASE)


def test_get_orbhess(mod, qcp):
    """Get diagonal orbital hessian"""
    od = qcp.get_orbital_diagonal()
    npt.assert_allclose(
        od,
        [
          +9.8642112,  10.07503561, 10.07503561, 10.46299801, 10.80965931,
          10.80965931, 10.82520425, 11.346842,   15.07835893,  1.24322897,
          +1.45405338,  1.45405338, 1.84201578,  2.18867709,  2.18867709,
          +2.20422202,  2.72585977,  6.4573767,
          +9.8642112,  10.07503561, 10.07503561, 10.46299801, 10.80965931,
          10.80965931, 10.82520425, 11.346842,   15.07835893,  1.24322897,
          +1.45405338,  1.45405338, 1.84201578,  2.18867709,  2.18867709,
          +2.20422202,  2.72585977, 6.4573767,
        ]
    )


def test_get_s2_diagonal(mod, qcp):
    """Get diagonal overlap hessian"""
    sd = qcp.get_overlap_diagonal()
    lsd = len(sd)
    npt.assert_allclose(sd, [2.0]*(lsd//2) + [-2.0]*(lsd//2))


def test_get_rhs(mod, qcp):
    """Get property gradient right-hand side"""
    rhs,  = qcp.get_rhs('z',)
    expected = [
          1.17073239e-01, -2.37864884e-16,  2.87393832e-16,
          1.65158629e-01,  2.09474735e-17,  8.48105006e-17,
         -2.60718081e-01,  2.34555790e-01, -4.08030858e-02,
         -1.25178721e+00,  2.00503702e-15, -2.48602195e-15,
         -1.15322545e-02, -1.33078270e-16, -6.65195683e-16,
         -3.49956863e-02,  2.07700781e+00,  3.44560225e-01,
         -1.17073239e-01,  2.37864884e-16, -2.87393832e-16,
         -1.65158629e-01, -2.09474735e-17, -8.48105006e-17,
          2.60718081e-01, -2.34555790e-01,  4.08030858e-02,
          1.25178721e+00, -2.00503702e-15,  2.48602195e-15,
          1.15322545e-02,  1.33078270e-16,  6.65195683e-16,
          3.49956863e-02, -2.07700781e+00, -3.44560225e-01
    ]
    npt.assert_allclose(rhs, expected, atol=1e08)


@pytest.mark.parametrize(
    'args',
    [
        (
           'z', (0.0,),
           {('z', 0.0):
            [1.18684846e-02, -2.36093343e-17,  2.85253416e-17,
             1.57850196e-02,  1.93784770e-18,  7.84580699e-18,
            -2.40843568e-02,  2.06714600e-02, -2.70606941e-03,
            -1.00688388e+00,  1.37892944e-15, -1.70971849e-15,
            -6.26067085e-03, -6.08030627e-17, -3.03925913e-16,
            -1.58766612e-02,  7.61964293e-01,  5.33591643e-02,
            -1.18684846e-02,  2.36093343e-17, -2.85253416e-17,
            -1.57850196e-02, -1.93784770e-18, -7.84580699e-18,
             2.40843568e-02, -2.06714600e-02,  2.70606941e-03,
             1.00688388e+00, -1.37892944e-15,  1.70971849e-15,
             6.26067085e-03,  6.08030627e-17,  3.03925913e-16,
             1.58766612e-02, -7.61964293e-01, -5.33591643e-02]}
        ),
    ],
    ids=['0.0']
)
def test_initial_guess(mod, qcp, args):
    """form paired trialvectors from rhs/orbdiag"""
    ops, freqs, expected = args
    initial_guess = qcp.initial_guess(ops, freqs)
    for op, freq in zip(ops, freqs):
        npt.assert_allclose(
            initial_guess[(op, freq)],
            expected[(op, freq)],
            atol=1e-8
        )


@pytest.mark.parametrize(
    'args',
    [
        (
            'xyz', 'xyz', (0,),
            {
                ('x', 'x', 0): -20.869910,
                ('x', 'y', 0): 0,
                ('x', 'z', 0): 0,
                ('y', 'x', 0): 0,
                ('y', 'y', 0): -20.869910,
                ('y', 'z', 0): 0,
                ('z', 'x', 0): 0,
                ('z', 'y', 0): 0,
                ('z', 'z', 0): -17.754933,
            }
        ),
        (
            'xyz', 'xyz', (0.03,),
            {
                ('x', 'x', 0.03): -21.3928977,
                ('x', 'y', 0.03): 0,
                ('x', 'z', 0.03): 0,
                ('y', 'x', 0.03): 0,
                ('y', 'y', 0.03): -21.3928977,
                ('y', 'z', 0.03): 0,
                ('z', 'x', 0.03): 0,
                ('z', 'y', 0.03): 0,
                ('z', 'z', 0.03): -18.183962,
            }
        ),
    ],
    ids=['0', '0.03']
)
def test_lr(mod, qcp, args):
    aops, bops, freqs, expected = args
    lr = qcp.lr(aops, bops, freqs)
    for k, v in lr.items():
        npt.assert_allclose(v, expected[k], atol=1e-8)


def test_new_trials1(mod, qcp):
    td = {0.5: np.array([2.5, 1.5,  1.5, 0.5])}
    residuals = {('op', 0.5): full.init([1, 1, 0, 0])}
    b = full.init([[1, 0, 0, 0], [0, 0, 1, 0]])
    new_trials = qcp.generate_new_trials(residuals, td, b)

    npt.assert_allclose(
        new_trials.T,
        [
            [0., 1, 0., 0.],
            [0., 0., 0., 1],
        ]
    )
