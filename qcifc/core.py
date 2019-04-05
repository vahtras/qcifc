"""Abstract interfact to QM codes"""
import abc
import numpy as np
import pandas as pd
from util import full

SMALL = 1e-10

class Observer(abc.ABC):
    @abc.abstractmethod
    def update(self):
        pass

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

    def initial_guess(self, ops="xyz", freqs=(0,), hessian_diagonal_shift=0.0001):
        od = self.get_orbital_diagonal(shift=hessian_diagonal_shift)
        sd = self.get_overlap_diagonal()
        dim = od.shape[0]
        ig = pd.DataFrame()
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
        for (op, freq) in vectors:
            vec = vectors[(op, freq)].values
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

    def lr_solve(self, ops="xyz", freqs=(0,), maxit=25, threshold=1e-5):
        from util.full import matrix

        V1 = pd.DataFrame({op: v for op, v in zip(ops, self.get_rhs(*ops))})
        igs = pd.DataFrame(self.initial_guess(ops=ops, freqs=freqs))
        b = self.setup_trials(igs)
        # if the set of trial vectors is null we return the initial guess
        if not np.any(b):
            return igs
        e2b = self.e2n(b).view(matrix)
        s2b = self.s2n(b).view(matrix)

        od = self.get_orbital_diagonal(shift=.0001)
        sd = self.get_overlap_diagonal()
        td = {w: od - w*sd for w in freqs}

        solutions = pd.DataFrame()
        residuals = pd.DataFrame()
        e2nn = pd.DataFrame()
        relative_residual_norm = pd.Series(index=igs.columns)


        self.update("|".join(
            f"it  <<{op};{op}>>{freq}     rn      nn"
            for op, freq in igs
            )
        )

        for i in range(maxit):
            # next solution
            for op, freq in igs:
                v = V1[op].values.view(matrix)
                reduced_solution = (b.T*v)/(b.T*(e2b-freq*s2b))
                solutions[(op, freq)] = b*reduced_solution
                e2nn[(op, freq)] = e2b*reduced_solution

#           e2nn = pd.DataFrame(
#               self.e2n(solutions.values), columns=solutions.columns
#           )
            s2nn = pd.DataFrame(
                self.s2n(solutions.values), columns=solutions.columns
            )

            # next residual
            output = ""
            for op, freq in igs:
                v = V1[op].values.view(matrix)
                n = solutions[(op, freq)]
                r = e2nn[(op, freq)] - freq*s2nn[(op, freq)] - v
                residuals[(op, freq)] = r
                nv = np.dot(n, v)
                rn = np.linalg.norm(r)
                nn = np.linalg.norm(n)
                relative_residual_norm[(op, freq)] = rn / nn
                output += f"{i+1} {-nv:.6f} {rn:.5e} {nn:.5e}|"
                
            self.update(output)
            #print()
            max_residual = max(relative_residual_norm)
            if max_residual < threshold:
                print("Converged")
                break
            new_trials = self.setup_trials(residuals, td=td, b=b)
            b = bappend(b, new_trials)
            new_e2b = self.e2n(new_trials).view(matrix)
            new_s2b = self.s2n(new_trials).view(matrix)
            e2b = bappend(e2b, new_e2b)
            s2b = bappend(s2b, new_s2b)
        return solutions

    def _get_E2S2(self):
        dim = 2*len(list(self.get_excitations()))
        E2 = full.init(self.e2n(np.eye(dim)))
        S2 = full.init(self.s2n(np.eye(dim)))
        return E2, S2

    def lr(self, aops, bops, freqs=(0,), **kwargs):
        v1 = {op: v for op, v in zip(aops, self.get_rhs(*aops))}
        solutions = self.lr_solve(bops, freqs, **kwargs)
        lrs = {}
        for aop in aops:
            for bop, w in solutions:
                lrs[(aop, bop, w)] = -np.dot(v1[aop], solutions[(bop, w)])
        return lrs


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
