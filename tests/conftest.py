import os
import subprocess
import pytest
from . import codes

@pytest.fixture(scope='module')
def code(request):
    coder, settings = request.param
    code = coder()
    code.setup(**settings)
    code.run_scf()
    yield code
    code.cleanup_scf()


def _case_dir(case):
    @pytest.fixture(scope='module')
    def tmpdir():
        """Help fixture used by conftest.qcp"""
        return f'test_{case}.d'
    return tmpdir


@pytest.fixture(scope='module')
def dalton_setup(tmpdir):
    tmpdir = os.path.join(os.path.dirname(__file__), tmpdir)
    os.chdir(tmpdir)
    subprocess.call(
        ['dalton', '-get', 'AOPROPER AOONEINT AOTWOINT', 'hf', case]
    )
    subprocess.call(['tar', 'xvfz', f'hf_{case}.tar.gz'])
    yield
    subprocess.call(
        'rm *.[0-9] DALTON.* *AO* *SIR* *RSP* molden.inp', shell=True
    )

@pytest.fixture(scope='module')
def vlx_setup(tmpdir):
    tmpdir = os.path.join(os.path.dirname(__file__), tmpdir)
    os.chdir(tmpdir)
    subprocess.call(
        ['dalton', '-get', 'AOPROPER AOONEINT AOTWOINT', 'hf', case]
    )
    subprocess.call(['tar', 'xvfz', f'hf_{case}.tar.gz'])
    yield
    subprocess.call(
        'rm *.[0-9] DALTON.* *AO* *SIR* *RSP* molden.inp', shell=True
    )


setups = {
    'dalton': dalton_setup,
    'vlx': vlx_setup,
}

@pytest.fixture(params=codes)
def qcp(request, tmpdir):
    tmp_path = os.path.join(os.path.dirname(__file__), tmpdir)
    factory = request.param(tmpdir=tmp_path)
    return factory
