"""Abstract interfact to QM codes"""
import abc
import numpy as np

SMALL = 1e-10


class Observer(abc.ABC):
    @abc.abstractmethod
    def update(self):
        """update method"""


class OutputStream(Observer):
    def __init__(self, stream):
        self.stream = stream

    def update(self, text):
        self.stream(text)


class QuantumChemistry(abc.ABC):
    """Abstract factory"""

    def setup(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def is_master(self):
        return True

    def set_observer(self, observer):
        self.observers.append(observer)

    def update(self, text):
        for observer in self.observers:
            observer.update(text)

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
        """Initialize scf state"""

    def cleanup_scf(self):
        """Initialize scf calculation"""

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
        for (op, freq) in vectors:
            vec = vectors[(op, freq)]
            if td is not None:
                v = vec/td[freq]
            else:
                v = vec
            if np.linalg.norm(v) > SMALL:
                trials.append(v)
                if freq > SMALL:
                    trials.append(swap(v))
        new_trials = np.array(trials).T
        if b is not None:
            new_trials = new_trials - b@b.T@new_trials
        if trials and renormalize:
            truncated = truncate(new_trials)
            new_trials = lowdin_normalize(truncated)
        return new_trials

    def lr_solve(self, ops="xyz", freqs=(0,), maxit=25, threshold=1e-5, roots=0):

        V1 = {op: v for op, v in zip(ops, self.get_rhs(*ops))}
        guess = self.initial_guess(ops=ops, freqs=freqs)
        b = self.setup_trials(guess)
        # if the set of trial vectors is null we return the initial guess
        if not np.any(b):
            return guess
        e2b = self.e2n(b)
        s2b = self.s2n(b)

        od = self.get_orbital_diagonal(shift=.0001)
        sd = self.get_overlap_diagonal()
        td = {w: od - w*sd for w in freqs}

        solutions = {}
        residuals = {}
        e2nn = {}
        s2nn = {}
        relative_residual_norm = {}

        self.update("|".join(
            f"it  <<{op};{op}>>{freq}     rn      nn"
            for op, freq in guess
            )
        )

        for i in range(maxit):
            # next solution
            for op, freq in guess:
                v = V1[op]
                reduced_solution = np.linalg.solve(b.T@(e2b-freq*s2b), b.T@v)
                solutions[(op, freq)] = b@reduced_solution
                e2nn[(op, freq)] = e2b@reduced_solution
                s2nn[(op, freq)] = s2b@reduced_solution

            # next residual
            output = ""
            for op, freq in guess:
                v = V1[op]
                n = solutions[(op, freq)]
                r = e2nn[(op, freq)] - freq*s2nn[(op, freq)] - v
                residuals[(op, freq)] = r
                nv = np.dot(n, v)
                rn = np.linalg.norm(r)
                nn = np.linalg.norm(n)
                relative_residual_norm[(op, freq)] = rn / nn
                output += f"{i+1} {-nv:.6f} {rn:.5e} {nn:.5e}|"

            self.update(output)

            if max(relative_residual_norm.values()) < threshold:
                print("Converged")
                break
            new_trials = self.setup_trials(residuals, td=td, b=b)
            b = bappend(b, new_trials)
            new_e2b = self.e2n(new_trials)
            new_s2b = self.s2n(new_trials)
            e2b = bappend(e2b, new_e2b)
            s2b = bappend(s2b, new_s2b)
        return solutions

    def direct_solver(self, ops="xyz", freqs=(0.), **kwargs):
        V1 = {op: v for op, v in zip(ops, self.get_rhs(*ops))}
        E2, S2 = self._get_E2S2()
        solutions = {
            (op, freq): np.linalg.solve((E2-freq*S2), V1[op])
            for freq in freqs for op in ops
        }
        return solutions

    def _get_E2S2(self):
        dim = 2*len(list(self.get_excitations()))
        E2 = self.e2n(np.eye(dim))
        S2 = self.s2n(np.eye(dim))
        return E2, S2

    def lr(self, aops, bops, freqs=(0,), **kwargs):
        v1 = {op: v for op, v in zip(aops, self.get_rhs(*aops))}
        solutions = self.lr_solve(bops, freqs, **kwargs)
        lrs = {}
        for aop in aops:
            for bop, w in solutions:
                lrs[(aop, bop, w)] = -np.dot(v1[aop], solutions[(bop, w)])
        return lrs

    def initial_excitations(self, n):
        excitations = list(self.get_excitations())
        excitation_energies = 0.5*self.get_orbital_diagonal()
        w = {ia: w for ia, w in zip(excitations, excitation_energies)}
        ordered_excitations = sorted(w, key=w.get)[:n]
        final = {}
        for (i, a) in ordered_excitations:
            ia = excitations.index((i, a))
            Xn = np.zeros(2*len(excitations))
            Xn[ia] = 1.0
            final[(i, a)] = (w[(i, a)], Xn)
        return final


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
    b12 = np.append(b1, b2, axis=1)
    return b12


def truncate(basis, threshold=1e-10):
    """
    Remove linear dependency in basis
    - skip eigenvectors of small eigenvalues of overlap matrix
    Returns truncated transformed basis
    """
    Sb = basis.T@basis
    l, T = np.linalg.eig(Sb)
    b_norm = np.sqrt(Sb.diagonal())
    mask = l > threshold*b_norm
    return basis @ T[:, mask]


def lowdin_normalize(basis):
    l, T = np.linalg.eig(basis.T@basis)
    inverse_sqrt = (T*np.sqrt(1/l)) @ T.T
    return basis@inverse_sqrt
