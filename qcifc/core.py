"""Abstract interfact to QM codes"""
import abc
import numpy as np
from . import normalizer

SMALL = 1e-10


class Observer(abc.ABC):
    @abc.abstractmethod
    def update(self):
        """update method"""

    @abc.abstractmethod
    def reset(self):
        """reset method"""


class OutputStream(Observer):
    def __init__(self, stream, width, d=6):
        self.stream = stream
        self.width = width
        self.d = d

    def update(self, items, **kwargs):
        converged = kwargs.get('converged')
        info = kwargs.get('info')
        for item in items:
            if not converged:
                if isinstance(item, float):
                    self.stream(f'{item:{self.width}.{self.d}f}')
                else:
                    self.stream(f'{item:^{self.width}s}')
            else:
                self.stream(self.green(f'{item:{self.width}.{self.d}f}'))
        if info:
            self.stream(f'{info:^{self.width}s}')

    def reset(self):
        self.stream('\n')

    def green(self, text):
        return f"\033[32m{text}\033[00m"


class QuantumChemistry(abc.ABC):
    """Abstract factory"""

    def setup(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.set_normalizer(normalizer.Lowdin())

    def set_normalizer(self, nzr):
        self.normalizer = nzr

    def is_master(self):
        return True

    def set_observer(self, observer):
        self.observers.append(observer)

    def reset_observers(self):
        for o in self.observers:
            o.reset()

    def update_observers(self, items, **kwargs):
        for observer in self.observers:
            observer.update(items, **kwargs)

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
        """Clean-up after scf calculation"""

    def _set_densities(self, *das):
        """Set densities"""
        self._da, self._db = das

    def get_densities(self):
        """Get densities"""
        return self._da, self._db

    def initial_guess(
            self, ops="xyz", freqs=(0,), roots=0, hessian_diagonal_shift=0.0001
        ):
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

    def init_trials(self, vectors, excitations=[], td=None, b=None, renormalize=True):
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

        for w, X in excitations:
            trials.append(X)
            trials.append(swap(X))

        new_trials = np.array(trials).T
        if b is not None:
            new_trials = new_trials - b@b.T@new_trials
        if trials and renormalize:
            #truncated = truncate(new_trials)
            new_trials = self.normalizer.normalize(new_trials)
        return new_trials

    def setup_trials(self, vectors, excitations=[], converged={}, td=None, tdx=None, b=None, renormalize=True):
        """
        Set up initial trial vectors from a set of intial guesses
        """
        trials = []
        for (op, freq) in vectors:
            if converged[(op, freq)]:
                continue
            vec = vectors[(op, freq)]
            if td is not None:
                v = vec/td[freq]
            else:
                v = vec
            if np.linalg.norm(v) > SMALL:
                trials.append(v)
                if freq > SMALL:
                    trials.append(swap(v))

        for k, (w, X) in enumerate(excitations):
            if converged[k]:
                continue
            if tdx:
                trials.append(X/tdx[w])
            else:
                trials.append(X)
            trials.append(swap(X))

        new_trials = np.array(trials).T
        if b is not None:
            new_trials = new_trials - b@b.T@new_trials
        if trials and renormalize:
            truncated = truncate(new_trials)
            new_trials = lowdin_normalize(truncated)
        return new_trials

    def pp_solve(self, roots, threshold=1e-5):
        _, excitations =  self.lr_solve(ops=(), freqs=(), roots=roots, threshold=threshold)
        return excitations

    def lr_solve(
        self, ops="xyz", freqs=(0,), maxit=25, threshold=1e-5, roots=0
    ):

        V1 = {op: v for op, v in zip(ops, self.get_rhs(*ops))}
        solutions = self.initial_guess(ops=ops, freqs=freqs)
        excitations = self.initial_excitations(roots)

        b = self.init_trials(solutions, excitations)
        # if the set of trial vectors is null we return the initial guess
        if not np.any(b):
            return solutions, excitations
        e2b = self.e2n(b)
        s2b = self.s2n(b)

        od = self.get_orbital_diagonal(shift=.0001)
        sd = self.get_overlap_diagonal()
        td = {w: od - w*sd for w in freqs}

        residuals = {}
        exresiduals = [None]*roots
        relative_residual_norm = {}
        converged = {}

        self.update_observers([], info='It')
        for op, freq in solutions:
            self.update_observers([f"<<{op};{op}>>{freq}", "rn", "nn"])
        for k in range(roots):
            self.update_observers([f"w_{k+1}", 'rn', 'nn'])
        self.update_observers([], info='dim')
        self.reset_observers()

        for i in range(maxit):
            # next solution
            self.update_observers([], info=f'{i+1}')
            output = ""
            for op, freq in solutions:
                v = V1[op]
                reduced_solution = np.linalg.solve(b.T@(e2b-freq*s2b), b.T@v)
                solutions[(op, freq)] = b@reduced_solution
                residuals[(op, freq)] = (e2b - freq*s2b)@reduced_solution - v

                r = residuals[(op, freq)]
                n = solutions[(op, freq)]

                nv = np.dot(n, v)
                rn = np.linalg.norm(r)
                nn = np.linalg.norm(n)

                relative_residual_norm[(op, freq)] = rn / nn
                converged[(op, freq)] = rn / nn < threshold
                self.update_observers([nv, rn, nn], converged=converged[(op, freq)])

            if roots > 0:
                reduced_ev = self.direct_ev_solver2(roots, b.T@e2b, b.T@s2b)
                for k, (w, reduced_X) in enumerate(reduced_ev):
                    r = (e2b - w*s2b)@reduced_X
                    X = b@reduced_X
                    rn = np.linalg.norm(r)
                    xn = np.linalg.norm(X)
                    exresiduals[k] = (w, r)
                    excitations[k] = (w, X)
                    relative_residual_norm[k] = rn/xn
                    converged[k] = rn/xn < threshold
                    self.update_observers([w, rn, xn], converged=converged[k])

            self.update_observers([], info=f'({len(b.T)})')
            self.reset_observers()

            if max(relative_residual_norm.values()) < threshold:
                print("Converged")
                break

            tdx = {w: od-w*sd for w, _ in excitations}
            new_trials = self.setup_trials(residuals, exresiduals, converged, td=td, tdx=tdx, b=b)
            b = bappend(b, new_trials)
            e2b = bappend(e2b, self.e2n(new_trials))
            s2b = bappend(s2b, self.s2n(new_trials))

        return solutions, excitations

    def direct_lr_solver(self, ops="xyz", freqs=(0.), **kwargs):
        V1 = {op: v for op, v in zip(ops, self.get_rhs(*ops))}
        E2, S2 = self._get_E2S2()
        solutions = {
            (op, freq): np.linalg.solve((E2-freq*S2), V1[op])
            for freq in freqs for op in ops
        }
        return solutions

    def direct_ev_solver(self, n_states, E2=None, S2=None):
        if E2 is None or S2 is None:
            E2, S2 = self._get_E2S2()
        wn, Xn = np.linalg.eigh((np.linalg.solve(S2, E2)))
        p = wn.argsort()
        wn = wn[p]
        Xn = Xn[:, p]
        dim = len(E2)
        lo = dim//2
        hi = dim//2 + n_states
        for i in range(lo, hi):
            norm = np.sqrt(Xn[:, i].T@S2@Xn[:, i])
            Xn[:, i] /= norm
        return zip(wn[lo: hi], Xn[:, lo: hi].T)

    def direct_ev_solver2(self, n_states, E2=None, S2=None):
        if E2 is None or S2 is None:
            E2, S2 = self._get_E2S2()
        E2 = (E2 + E2.T)/2
        S2 = (S2 + S2.T)/2
        T = np.linalg.solve(E2, S2)
        wn, Xn = np.linalg.eig(T)
        p = list(reversed(wn.argsort()))
        wn = wn[p]
        Xn = Xn[:, p]
        for i in range(n_states):
            norm = np.sqrt(Xn[:, i].T@S2@Xn[:, i])
            Xn[:, i] /= norm
        return zip(1/wn[:n_states], Xn[:, :n_states].T)

    def _get_E2S2(self):
        dim = 2*len(list(self.get_excitations()))
        E2 = self.e2n(np.eye(dim))
        S2 = self.s2n(np.eye(dim))
        return E2, S2

    def lr(self, aops, bops, freqs=(0,), **kwargs):
        v1 = {op: v for op, v in zip(aops, self.get_rhs(*aops))}
        solutions, _ = self.lr_solve(bops, freqs, **kwargs)
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
        final = []
        for (i, a) in ordered_excitations:
            ia = excitations.index((i, a))
            Xn = np.zeros(2*len(excitations))
            Xn[ia] = 1.0
            final.append((w[(i, a)], Xn))
        return final

    def transition_moments(self, ops, roots, **kwargs):
        solutions = list(self.pp_solve(roots, **kwargs))
        V1 = {op: V for op, V in zip(ops, self.get_rhs(*ops))}
        tms = {op: np.array([np.dot(V1[op], s[1]) for s in solutions]) for op in ops}
        tms['w'] = np.array([s[0] for s in solutions])
        return tms

    def oscillator_strengths(self, roots, **kwargs):
        tms = self.transition_moments('xyz', roots, **kwargs)
        osc = 2/3*tms['w']*(tms['x']**2 + tms['y']**2 + tms['z']**2)
        return {'w': tms['w'], 'I': osc}

    def excitation_energies(self, n_states):
        return np.array([w for w, _ in self.pp_solve(n_states)])

    def eigenvectors(self, n_states):
        return np.array([X for _, X in self.pp_solve(n_states)]).T


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
    l, T = np.linalg.eigh(Sb)
    b_norm = np.sqrt(Sb.diagonal())
    mask = l > threshold*b_norm
    return basis @ T[:, mask]


def lowdin_normalize(basis):
    l, T = np.linalg.eigh(basis.T@basis)
    inverse_sqrt = (T*np.sqrt(1/l)) @ T.T
    return basis@inverse_sqrt
