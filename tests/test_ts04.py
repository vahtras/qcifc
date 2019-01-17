import pytest
import numpy.testing as npt
from .conftest import case_dir, case_fixture

CASE = 'ts04'
tmpdir = case_dir(CASE)
mod = case_fixture(CASE)


@pytest.mark.skip()
def test_get_orbhess(mod, qcp):
    """Get diagonal orbital hessian"""
    od = qcp.get_orbital_diagonal()
    npt.assert_allclose(
        od,
        [
        ]*2
    )


def test_get_s2_diagonal(mod, qcp):
    """Get diagonal overlap hessian"""
    sd = qcp.get_overlap_diagonal()
    expected = [1.]*9 + [2.]*45 + [1.]*5
    expected += [-1.]*9 + [-2.]*45 + [-1.]*5
    npt.assert_allclose(sd, expected)


@pytest.mark.skip()
def test_get_rhs(mod, qcp):
    """Get property gradient right-hand side"""
    rhs,  = qcp.get_rhs('z',)
    expected = []
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
def test_initial_guess(mod, qcp, args):
    """form paired trialvectors from rhs/orbdiag"""
    ops, freqs, expected = args
    initial_guess = qcp.initial_guess(ops, freqs)
    for op, freq in zip(ops, freqs):
        npt.assert_allclose(
            initial_guess[(op, freq)],
            expected[(op, freq)],
            rtol=1e-5,
        )


@pytest.mark.parametrize(
    'args',
    [
        ('x', 'x', (0,), {('x', 'x', 0): -7.919300117806e+00}),
        ('y', 'y', (0,), {('y', 'y', 0): -9.417538523718e+00}),
        ('z', 'z', (0,), {('z', 'z', 0): -1.741701816106e+01}),
        ('x', 'x', (0.5,), {('x', 'x', 0.5): -9.067142511985e+00}),
        ('y', 'y', (0.5,), {('y', 'y', 0.5): -1.013645310651e+01}),
        ('z', 'z', (0.5,), {('z', 'z', 0.5): -3.232573463730e+00}),
    ],
    ids=['xx0', 'yy0', 'zz0', 'xx0.5', 'yy0.5', 'zz0.5']
)
def test_lr(mod, qcp, args):
    aops, bops, freqs, expected = args
    lr = qcp.lr(aops, bops, freqs)
    for k, v in lr.items():
        npt.assert_allclose(v, expected[k], rtol=1e-4)
