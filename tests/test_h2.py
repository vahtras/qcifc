import pytest
import os
import numpy
import numpy.testing as npt
from .conftest import case_dir, case_fixture

CASE = 'h2'
tmpdir = case_dir(CASE)
mod = case_fixture(CASE)


def test_get_wrkdir(qcp):
    """Get factory workdir"""
    assert qcp.get_workdir() ==  \
        os.path.join(os.path.dirname(__file__), 'test_h2.d')


def test_set_wrkdir(qcp):
    """Get factory workdir"""
    qcp.set_workdir('/tmp/123')
    assert qcp.get_workdir() == '/tmp/123'


def test_get_overlap(qcp):
    """Get overlap"""
    npt.assert_allclose(
        qcp.get_overlap(),
        [[1.0, 0.65987313], [0.65987313, 1.0]]
        )


def test_get_h1(qcp):
    """Get one-electron Hamiltonian"""
    npt.assert_allclose(
        qcp.get_one_el_hamiltonian(),
        [[-1.12095946, -0.95937577], [-0.95937577, -1.12095946]]
        )


def test_get_z(qcp):
    """Nuclear repulsion energy"""
    assert qcp.get_nuclear_repulsion() == pytest.approx(0.7151043)


def test_get_mo(qcp):
    """Read MO coefficients"""
    cmo = qcp.get_mo()
    npt.assert_allclose(
        cmo,
        [[.54884227, -1.212451936],
         [.54884227, 1.21245193]]
    )


def test_set_get_dens_a(qcp):
    """Set density test"""
    da = [[1, 0], [0, 1]]
    db = [[1, 0], [0, 0]]
    qcp.set_densities(da, db)
    npt.assert_allclose(
        qcp.get_densities()[0], da
    )
    npt.assert_allclose(
        qcp.get_densities()[1], db
    )


def test_get_two_fa(qcp):
    """Get alpha Fock matrix"""
    da = numpy.array([[1, 0], [0, 1]])
    db = numpy.array([[1, 0], [0, 0]])
    faref = numpy.array([
        [1.04701025, 0.44459112],
        [0.44459112, 0.8423992]
    ])

    fbref = numpy.array([
        [1.34460081, 0.88918225],
        [0.88918225, 1.61700513]
    ])
    qcp.set_densities(da, db)
    fa, fb = qcp.get_two_el_fock()
    npt.assert_allclose(fa, faref)
    npt.assert_allclose(fb, fbref)


def test_get_orbhess(qcp):
    """Get diagonal orbital hessian"""
    od = qcp.get_orbital_diagonal()
    npt.assert_allclose(od, [4.99878931, 4.99878931])


def test_get_rhs(qcp):
    """Get property gradient right-hand side"""
    x, y, z = qcp.get_rhs('x', 'y', 'z')
    npt.assert_allclose(x, [0, 0])
    npt.assert_allclose(y, [0, 0])
    npt.assert_allclose(z, [1.86111268, -1.86111268])
    npt.assert_allclose((x, y, z), ([0, 0], [0, 0], [1.86111268, -1.86111268]))


@pytest.mark.parametrize(
    'trials',
    [
        ([1, 0], [1.89681370, -0.36242092]),
        ([0, 1], [-0.36242092, 1.89681370]),
        ([[1, 0],
          [0, 1]],
         [[1.89681370, -0.36242092],
          [-0.36242092, 1.89681370]]),
    ]
)
def test_oli(qcp, trials):
    """Linear transformation E2*N"""
    n, e2n = trials
    numpy.testing.assert_allclose(qcp.e2n(n), e2n)


def test_sli(qcp):
    """Linear transformation E2*N"""
    absolute_tolerance = 1e-10
    s2n = qcp.s2n([1, 0])
    npt.assert_allclose(
        s2n, [2.00000000,  0.00000000],
        atol=absolute_tolerance
    )
    s2n = qcp.s2n([0, 1])
    npt.assert_allclose(
        s2n, [0.00000000, -2.00000000],
        atol=absolute_tolerance
    )


@pytest.mark.parametrize(
    'args',
    [
        ('x', (0,), {('x', 0): [0, 0]},),
        ('z', (0,), {('z', 0): [0.37231269, -0.37231269]}),
        ('z', (0.5,), {('z', 0.5): [0.46541904, -0.31024805]}),
        (
            'z', (0, 0.5),
            {
                ('z', 0): [0.37231269, -0.37231269],
                ('z', 0.5): [0.46541904, -0.31024805]
            }
        ),
        (
            'xz', (0,),
            {
                ('x', 0): [0.0, 0.0],
                ('z', 0): [0.37231269, -0.37231269],
            }
        ),
        (
            'xz', (0.5,),
            {
                ('x', 0.5): [0., 0.],
                ('z', 0.5): [-0.31024805, 0.46541904]
            }
        ),
        (
            'xz', (0, 0.5),
            {
                ('x', 0): [0., 0.],
                ('x', 0.5): [0., 0.],
                ('z', 0): [0.37231269, -0.37231269],
                ('z', 0.5): [0.46541904, -0.31024805],
            }
        )
    ],
    ids=['x-0', 'z-0', 'z-0.5', 'z-(0, 0.5)', 'xz-0', 'xz-0.5', 'xz-(0, 0.5)']
)
def test_initial_guess(qcp, args):
    """form paired trialvectors from rhs/orbdiag"""
    ops, freqs, expected = args
    initial_guess = qcp.initial_guess(ops=ops, freqs=freqs)
    for op, freq in zip(ops, freqs):
        npt.assert_allclose(
            initial_guess[(op, freq)],
            expected[(op, freq)],
            rtol=1e-5,
            )


@pytest.mark.parametrize(
    'args',
    [
        (
            {('x', 0): [0, 0]},
            []
        ),
        (
            {('z', 0): [0.37231269, -0.37231269]},
            [[0.37231269, -0.37231269]]
        ),
        (
            {('z', 0.5): [0.46541904, -0.31024805]},
            [[0.46541904, -0.31024805], [-0.31024805, 0.46541904]]
        ),
        (
            {
                ('z', 0): [0.37231269, -0.37231269],
                ('z', 0.5): [0.46541904, -0.31024805]
            },
            [
                [0.37231269, -0.37231269],
                [0.46541904, -0.31024805],
                [-0.31024805, 0.46541904],
            ]
        ),
        (
            {
                ('x', 0): [0.0, 0.0],
                ('z', 0): [0.37231269, -0.37231269],
            },
            [
                [0.37231269, -0.37231269],
            ]
        ),
        (
            {
                ('x', 0.5): [0., 0.],
                ('z', 0.5): [-0.31024805, 0.46541904]
            },
            [
                [-0.31024805, 0.46541904],
                [0.46541904, -0.31024805],
            ]
        ),
        (
            {
                ('x', 0): [0., 0.],
                ('x', 0.5): [0., 0.],
                ('z', 0): [0.37231269, -0.37231269],
                ('z', 0.5): [0.46541904, -0.31024805],
            },
            [
                [0.37231269, -0.37231269],
                [0.46541904, -0.31024805],
                [-0.31024805, 0.46541904],
            ]
        )
    ],
    ids=['x-0', 'z-0', 'z-0.5', 'z-(0, 0.5)', 'xz-0', 'xz-0.5', 'xz-(0, 0.5)']
)
def test_setup_trials(qcp, args):
    """
    Form paired trialvectors from initial guesses (rhs/diagonal)
    Parameterized input lists are reformatted to arrays.
    """
    initial_guesses, expected = args
    ig = {
        key: numpy.array(vector)
        for key, vector in initial_guesses.items()
    }
    b = qcp.setup_trials(ig, renormalize=False)
    npt.assert_allclose(b.T, expected, rtol=1e-5)


@pytest.mark.parametrize(
    'args',
    [
        ('x', (0,), {('x', 0): [0, 0]}),
        ('z', (0,), {('z', 0): [0.82378017, -0.82378017]}),
        ('z', (0.5,), {('z', 0.5): [1.91230027, -0.40322064]}),
        (
            'z', (0, 0.5),
            {
                ('z', 0): [0.82378017, -0.82378017],
                ('z', 0.5): [1.91230027, -0.40322064]
            }
        ),
    ],
    ids=['x-0', 'z-0', 'z-0.5', 'z-(0, 0.5)']
)
def test_solve(qcp, args):
    ops, freqs, expected = args
    solutions = qcp.lr_solve(ops=ops, freqs=freqs)
    for op, freq in solutions:
        npt.assert_allclose(
            solutions[(op, freq)],
            expected[(op, freq)]
        )


@pytest.mark.parametrize(
    'args',
    [
        ('z', 'z', (0,), {('z', 'z', 0): -3.066295447276}),
        ('z', 'z', (0.5,), {('z', 'z', 0.5): -4.309445328973108}),
    ],
    ids=['0', '0.5']
)
def test_lr(qcp, args):
    aops, bops, freqs, expected = args
    lr = qcp.lr(aops, bops, freqs)
    for k, v in lr.items():
        npt.assert_allclose(v, expected[k])


@pytest.mark.parametrize(
    'args',
    [
        ('z', 1, {('z', 0): 1.1946797}),
    ],
    ids=['z1', ]
)
def test_pp(qcp, args):
    if 'pp' not in dir(qcp):
        pytest.skip('not implemented')

    aops, nfreqs, expected = args
    pp = qcp.pp(aops, nfreqs)
    for k, v in pp.items():
        npt.assert_allclose(v, expected[k])


def test_excitation_energies(qcp):
    if 'excitation_energies' not in dir(qcp):
        pytest.skip('not implemented')

    w = qcp.excitation_energies(1)
    assert w == pytest.approx(0.93093411)


def test_eigenvectors(qcp):
    if 'eigenvectors' not in dir(qcp):
        pytest.skip('not implemented')
    X = qcp.eigenvectors(1)
    npt.assert_allclose(X.T, [[0.7104169615, 0.0685000673]])


def test_dim(qcp):
    assert qcp.response_dim() == 2
