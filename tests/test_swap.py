import numpy as np
import numpy.testing as npt
from qcifc.core import swap
from util.full import matrix, init

def test_np():
    x = np.array([1., 2., ])
    y = swap(x)
    assert type(x) == type(y)
    assert x.shape == y.shape
    npt.assert_allclose(y, [2., 1.])

def test_mat():
    x = init([1., 2., ])
    y = swap(x)
    assert type(x) == type(y)
    assert x.shape == y.shape
    npt.assert_allclose(y, [2., 1.])

def test_mat2():
    x = init([[1., 2.]])
    y = swap(x)
    assert type(x) == type(y)
    assert x.shape == y.shape
    npt.assert_allclose(y, [[2.], [1.]])
