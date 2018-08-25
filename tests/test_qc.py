import unittest
import pytest
import os
import numpy
import numpy.testing as npt

from qcifc.core import QuantumChemistry, DaltonFactory

@pytest.fixture
def qcp():
    tmp = os.path.join(os.path.dirname(__file__), 'test_h2.d')
    factory = QuantumChemistry.get_factory(
            'Dalton',
            tmpdir=tmp,
            )
    return factory

def test_create_dalton_factory(qcp):
    """Create 'concrete' factory'"""
    assert isinstance(qcp, DaltonFactory)

def test_unknown_raises_typeerror(qcp):
    """Unknown code raises TypeError"""
    with pytest.raises(TypeError):
        QuantumChemistry.get_factory('Gamess')

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
    npt.assert_allclose(cmo, [[.54884227, -1.212451936],
        [.54884227, 1.21245193]])

def test_set_get_dens_a(qcp):
    """Set density test"""
    da = [[1, 0], [0, 1]]; db = [[1, 0], [0, 0]]
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
        [1.04701025 , 0.44459112],
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
    rhs,  = qcp.get_rhs('z',) 
    npt.assert_allclose(rhs, [1.86111268, -1.86111268])

def test_oli(qcp):
    """Linear transformation E2*N"""
    e2n = qcp.e2n([1, 0])
    numpy.testing.assert_allclose(e2n, [1.89681370, -0.36242092])
    e2n = qcp.e2n([0, 1])
    npt.assert_allclose(e2n, [-0.36242092, 1.89681370])

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

def test_initial_guess(qcp):
    """form paired trialvectors from rhs/orbdiag"""
    npt.assert_allclose(
        qcp.initial_guess('z').T,
        [[0.37231269, -0.37231269]]
    )

def test_solve(qcp):
    npt.assert_allclose(
        qcp.lr_solve('z').T,
        [ 0.82378017, -0.82378017],
    )

@pytest.mark.parametrize('wlr', [(0, 3.066295447276)], ids=['0'])
def test_lr(qcp, wlr):
    w, lr = wlr
    n = qcp.lr_solve('z')
    v, = qcp.get_rhs('z')
    npt.assert_allclose((n&v), lr)
