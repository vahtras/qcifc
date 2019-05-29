import pytest


@pytest.fixture(scope='function')
def code(request):
    cls, settings = request.param
    qcp = cls()
    qcp.setup(**settings)
    qcp.run_scf(settings['case'])
    yield qcp
    qcp.cleanup_scf()
