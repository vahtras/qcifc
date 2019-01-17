"""Abstract interfact to QM codes"""
import abc
import numpy as np
from util import full

SMALL = 1e-10


class QuantumChemistry(abc.ABC):
    """Abstract factory"""

    def __init__(self, code, **kwargs):
        self.tmpdir = kwargs.get('tmpdir', '/tmp')

    @abc.abstractmethod
    def get_overlap(self):  # pragma: no cover
        """Abstract overlap getter"""

    @abc.abstractmethod
    def get_one_el_hamiltonian(self):  # pragma: no cover
        """Abstract h1 getter"""

    @abc.abstractmethod
    def get_nuclear_repulsion(self):  # pragma: no cover
        """Abstract Z getter"""


def swap(xy):
    """
    Swap X and Y parts of response vector
    """
    assert len(xy.shape) <= 2, 'Not implemented for dimensions > 2'
    yx = xy.copy()
    half_rows = xy.shape[0]//2
    if len(xy.shape) == 1:
        yx[:half_rows] = xy[half_rows:]
        yx[half_rows:] = xy[:half_rows]
    else:
        yx[:half_rows, :] = xy[half_rows:, :]
        yx[half_rows:, :] = xy[:half_rows, :]
    return yx


def bappend(b1, b2):
    """
    Merge arrays by appending column-wise
    """
    b12 = np.append(b1, b2, axis=1).view(full.matrix)
    return b12


def get_transform(basis, threshold=1e-10):
    Sb = basis.T*basis
    l, T = Sb.eigvec()
    b_norm = np.sqrt(Sb.diagonal())
    mask = l > threshold*b_norm
    return T[:, mask]
