import pytest
import os
import subprocess
import numpy.testing as npt

from qcifc.core import QuantumChemistry, DaltonFactory

@pytest.fixture(params=['DaltonDummy', 'Dalton'])
def qcp(request):
    tmp = os.path.join(os.path.dirname(__file__), 'test_lih.d')
    factory = QuantumChemistry.set_code(
            request.param,
            tmpdir=tmp,
            )
    return factory

@pytest.fixture(scope='module')
def mod():
    tmpdir = os.path.join(os.path.dirname(__file__), 'test_lih.d')
    os.chdir(tmpdir)
    subprocess.call(['dalton', '-get', 'AOPROPER AOONEINT AOTWOINT', 'rsp_polar', 'lih'])
    subprocess.call(['tar', 'xvfz', 'rsp_polar_lih.tar.gz'])
    yield
    subprocess.call('rm *.[0-9] DALTON.* *AO* *SIR* *RSP* molden.inp', shell=True)
    
def test_get_orbhess(mod, qcp):
    """Get diagonal orbital hessian"""
    od = qcp.get_orbital_diagonal() 
    npt.assert_allclose(od, [
 9.8642112,  10.07503561, 10.07503561, 10.46299801, 10.80965931, 10.80965931,
 10.82520425, 11.346842,   15.07835893,  1.24322897,  1.45405338,  1.45405338,
  1.84201578,  2.18867709,  2.18867709,  2.20422202,  2.72585977,  6.4573767, 
 9.8642112,  10.07503561, 10.07503561, 10.46299801, 10.80965931, 10.80965931,
 10.82520425, 11.346842,   15.07835893,  1.24322897,  1.45405338,  1.45405338,
  1.84201578,  2.18867709,  2.18867709,  2.20422202,  2.72585977,  6.4573767, 
    ])

def test_get_s2_diagonal(mod, qcp):
    """Get diagonal overlap hessian"""
    sd = qcp.get_overlap_diagonal() 
    lsd = len(sd)
    npt.assert_allclose(sd, [2.0]*(lsd//2) + [-2.0]*(lsd//2))
    

def test_get_rhs(mod, qcp):
    """Get property gradient right-hand side"""
    rhs,  = qcp.get_rhs('z',) 
    npt.assert_allclose(
       rhs,
       [ 1.17073239e-01, -2.37864884e-16,  2.87393832e-16,
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
         3.49956863e-02, -2.07700781e+00, -3.44560225e-01] 
    )

@pytest.mark.parametrize('wlr',
    [
        (
            (0.0,), 
           [[1.18684846e-02, -2.36093343e-17,  2.85253416e-17,
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
             1.58766612e-02, -7.61964293e-01, -5.33591643e-02]]
        ),
    ],
    ids=['0.0']
)
def test_initial_guess(mod, qcp, wlr):
    """form paired trialvectors from rhs/orbdiag"""
    w, lr = wlr
    npt.assert_allclose(
        qcp.initial_guess('z', w).T,
        lr,
    )

@pytest.mark.parametrize('wlr',
    [
        ((0,), ((-20.869910,), (-20.869910,), (-17.754933,))),
        ((0.03,), ((-21.3928977,), (-21.3928977,), (-18.183962,)))
    ],
    ids=['0', '0.03']
)
def test_lr(mod, qcp, wlr):
    freqs, aref = wlr
    a = tuple(qcp.lr(f'{c};{c}', freqs) for c in 'xyz')
    npt.assert_allclose(a, aref)
    #assert False
