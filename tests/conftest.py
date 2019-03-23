import pytest


@pytest.fixture(scope='function')
def code(request):
    Coder, settings = request.param
    code = Coder()
    code.setup(**settings)
    code.run_scf(settings['case'])
    yield code
    code.cleanup_scf()
