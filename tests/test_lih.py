import pytest
import os
import subprocess
import numpy.testing as npt

from qcifc.core import QuantumChemistry, DaltonFactory

@pytest.fixture(params=['DaltonDummy'])
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
    
    

def test_1(mod, qcp):
    pass

def test_2(mod, qcp):
    pass

@pytest.mark.parametrize('wlr',
    [(0, (-20.869629, -20.869629, -17.754407,)),
     (0.03, (-21.39261900, -21.3926190, -18.183422))],
    ids=['0', '0.3']
)
def test_lr(mod, qcp, wlr):
    w, (x, y, z) = wlr
    npt.assert_allclose(qcp.lr('x;x', w), x)
    npt.assert_allclose(qcp.lr('y;y', w), y)
    npt.assert_allclose(qcp.lr('z;z', w), z)
