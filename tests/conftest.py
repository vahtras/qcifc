import pytest


@pytest.fixture(scope='function')
def code(request):
    coder, settings = request.param
    code = coder()
    code.setup(**settings)
    code.run_scf(settings['case'])
    yield code
    code.cleanup_scf()
