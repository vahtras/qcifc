import abc

import numpy as np


class Normalizer(abc.ABC):

    def __init__(self, threshold=1e-7):
        self.threshold = threshold

    @abc.abstractmethod
    def normalize(self, b):
        "Implements truncate and normalize"


class Lowdin(Normalizer):

    def normalize(self, basis):
        overlap = basis.T@basis
        l, T = np.linalg.eigh(overlap)
        mask = l > self.threshold
        inverse_sqrt = T[:, mask]*np.sqrt(1/l[mask])
        return basis @ inverse_sqrt


class GramSchmidt(Normalizer):

    def normalize(self, basis):
        """
        Return Gram-Schmidt normalized basis
        """

        new = basis[:, :1]/np.linalg.norm(basis[:, 0])

        for column in basis.T[1:]:
            column -= new @ (new.T @ column)
            norm = np.linalg.norm(column)
            if norm > self.threshold:
                column /= norm
                new = np.append(new, column.reshape((len(column), 1)), axis=1)
        return new


class QR(Normalizer):

    def normalize(self, basis):

        q, r = np.linalg.qr(basis)
        mask = np.max(np.abs(r), axis=1) > self.threshold
        return q[:, mask]
