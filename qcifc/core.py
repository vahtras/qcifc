"""Abstract interfact to QM codes"""
import abc
import numpy as np
from util import full

SMALL = 1e-10


class QuantumChemistry(abc.ABC):
    """Abstract factory"""

    def __init__(self, code, **kwargs):
        self.tmpdir = kwargs.get('tmpdir', '/tmp')

    def setup(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abc.abstractmethod
    def get_overlap(self):  # pragma: no cover
        """Abstract overlap getter"""

    @abc.abstractmethod
    def get_one_el_hamiltonian(self):  # pragma: no cover
        """Abstract h1 getter"""

    @abc.abstractmethod
    def get_nuclear_repulsion(self):  # pragma: no cover
        """Abstract Z getter"""

    def run_scf(self):
        pass

    def cleanup_scf(self):
        pass

    def set_densities(self, *das):
        """Set densities"""
        self._da, self._db = das

    def get_densities(self):
        """Get densities"""
        return self._da, self._db

    def initial_guess(self, ops="xyz", freqs=(0,)):
        od = self.get_orbital_diagonal()
        sd = self.get_overlap_diagonal()
        dim = od.shape[0]
        ig = {}
        for op, grad in zip(ops, self.get_rhs(*ops)):
            gn = np.linalg.norm(grad)
            for w in freqs:
                if gn < SMALL:
                    ig[(op, w)] = np.zeros(dim)
                else:
                    td = od - w*sd
                    ig[(op, w)] = grad/td
        return ig

    def setup_trials(self, vectors, td=None, b=None, renormalize=True):
        """
        Set up initial trial vectors from a set of intial guesses
        """
        trials = []
        for (op, freq), vec in vectors.items():
            if td is not None:
                v = vec/td[freq]
            else:
                v = vec
            if np.linalg.norm(v) > SMALL:
                trials.append(v)
                if freq > SMALL:
                    trials.append(swap(v))
        new_trials = full.init(trials)
        if b is not None:
            new_trials = new_trials - b*b.T*new_trials
        if trials and renormalize:
            t = get_transform(new_trials)
            truncated = new_trials*t
            S12 = (truncated.T*truncated).invsqrt()
            new_trials = truncated*S12
        return new_trials


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
