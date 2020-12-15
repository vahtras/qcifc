import pytest
import numpy as np
import numpy.testing as npt

from hypothesis import given
from hypothesis.strategies import integers

# from ortho import __version__
from qcifc.normalizer import Lowdin, GramSchmidt, QR


# def test_version():
#     assert __version__ == '0.1.0'


def _randomnk(m, n, k):
    """
    Create a random square matrix of dimension m*n, but rank k
    Replace last n-k columns with linear combinations of the first k
    """
    a = np.random.random((m, n))
    for i in range(k, n):
        pars = np.random.random(k)
        a[:, i] = a[:, :k] @ pars
    return a


@pytest.mark.parametrize(
    'data',
    [
        ([[1., 0., 0.]], [[1.]]),
        (
            [[1., 0., 0.], [0., 1., 0.]],
            np.eye(2)
        ),
        (
            [[1., 0., 0.], [1., 0., 0.]],
            np.eye(1)
        ),
    ]
)
def test_lowdin(data):
    normalizer = Lowdin()
    indata, expected = data
    basis = np.array(indata).T
    outdata = normalizer.normalize(basis)
    npt.assert_allclose(outdata.T@outdata, expected)


@pytest.mark.parametrize(
    'data',
    [
        ([[1., 0., 0.]], [[1.]]),
        ([[2., 0., 0.]], [[1.]]),
        (
            [[1., 0., 0.], [0., 1., 0.]],
            np.eye(2)
        ),
        (
            [[2., 0., 0.], [0., 2., 0.]],
            np.eye(2)
        ),
        (
            [[1., 0., 0.], [1., 0., 0.]],
            np.eye(1)
        ),
        (
            [[3., 1.], [2., 2.]],
            np.eye(2)
        ),
        (
            [[2., 0., 0.], [3., 0., 0.]],
            np.eye(1)
        ),
    ]
)
def test_gs(data):
    normalizer = GramSchmidt()
    indata, expected = data
    basis = np.array(indata).T
    outdata = normalizer.normalize(basis)
    npt.assert_allclose(outdata.T@outdata, expected, atol=1e-10)


@pytest.mark.parametrize(
    'data',
    [
        (
            [[3., 1.], [2., 2.]],
            np.array([[3., 1.], [-1., 3]])/np.sqrt(10)
        ),
    ]
)
def test_gs2(data):
    normalizer = GramSchmidt()
    indata, expected = data
    basis = np.array(indata).T
    outdata = normalizer.normalize(basis)
    npt.assert_allclose(outdata.T, expected)


@pytest.mark.parametrize(
    'data',
    [
        ([[1., 0., 0.]], [[1.]]),
        ([[2., 0., 0.]], [[1.]]),
        (
            [[1., 0., 0.], [0., 1., 0.]],
            np.eye(2)
        ),
        (
            [[2., 0., 0.], [0., 2., 0.]],
            np.eye(2)
        ),
        (
            [[1., 0., 0.], [1., 0., 0.]],
            np.eye(1)
        ),
        (
            [[3., 1.], [2., 2.]],
            np.eye(2)
        ),
        (
            [[2., 0., 0.], [3., 0., 0.]],
            np.eye(1)
        ),
    ]
)
def test_qr(data):
    normalizer = QR()
    indata, expected = data
    basis = np.array(indata).T
    outdata = normalizer.normalize(basis)
    npt.assert_allclose(outdata.T@outdata, expected, atol=1e-10)


@pytest.mark.parametrize(
    'data',
    [
        (
            (5, 5, 5),
            np.eye(5)
        ),
        (
            (5, 5, 4),
            np.eye(4)
        ),
        (
            (5, 5, 3),
            np.eye(3)
        ),
        (
            (5, 5, 2),
            np.eye(2)
        ),
    ]
)
def test_qrr(data):
    normalizer = QR()
    indata, expected = data
    basis = _randomnk(*indata)
    outdata = normalizer.normalize(basis)
    npt.assert_allclose(outdata.T@outdata, expected, atol=1e-10)


@given(
    integers(min_value=100, max_value=200),
    integers(min_value=1, max_value=100),
    integers(min_value=1, max_value=100)
)
def test_qrhyp(m, n, k):
    if n < k:
        n, k = k, n
    normalizer = QR()
    basis = _randomnk(m, n, k)
    outdata = normalizer.normalize(basis)
    npt.assert_allclose(outdata.T@outdata, np.eye(k), atol=1e-10)


@given(
    integers(min_value=100, max_value=200),
    integers(min_value=1, max_value=100),
    integers(min_value=1, max_value=100)
)
def test_gshyp(m, n, k):
    if n < k:
        n, k = k, n
    normalizer = GramSchmidt()
    basis = _randomnk(m, n, k)
    outdata = normalizer.normalize(basis)
    npt.assert_allclose(outdata.T@outdata, np.eye(k), atol=1e-10)


@given(
    integers(min_value=100, max_value=200),
    integers(min_value=1, max_value=100),
    integers(min_value=1, max_value=100)
)
def test_gspol(m, n, k):
    if n < k:
        n, k = k, n
    normalizer = Lowdin()
    basis = _randomnk(m, n, k)
    outdata = normalizer.normalize(basis)
    npt.assert_allclose(outdata.T@outdata, np.eye(k), atol=1e-10)
