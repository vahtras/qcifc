import pytest


@pytest.fixture(scope='function')
def code(request):
    coder, settings = request.param
    code = coder()
    code.setup(**settings)
    code.run_scf()
    yield code
    code.cleanup_scf()
