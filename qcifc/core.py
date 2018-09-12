"""Abstrace interfact to QM codes"""
import abc
import os
from util import full
import numpy as np

SMALL = 1e-10

class QuantumChemistry(object):
    """Abstract factory"""
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def set_code(code, **kwargs):
        """Return concrete factory"""
        if code == 'Dalton':
            return DaltonFactory(**kwargs)
        elif code == 'DaltonDummy':
            return DaltonFactoryDummy(**kwargs)
        else:
            raise TypeError("QM %s not implemented" % code)

    @abc.abstractmethod
    def get_overlap(self):#pragma: no cover
        """Abstract overlap getter"""
        pass

    @abc.abstractmethod
    def get_one_el_hamiltonian(self):#pragma: no cover
        """Abstract h1 getter"""
        pass

    @abc.abstractmethod
    def get_nuclear_repulsion(self):#pragma: no cover
        """Abstract Z getter"""
        pass



import numpy
from daltools import one, sirrst, sirifc, prop
from dalmisc import oli
import two.core
import two.vb

class DaltonFactory(QuantumChemistry):
    """Concrete 'factory', Dalton"""

    labels = {'x': 'XDIPLEN', 'y': 'YDIPLEN', 'z': 'ZDIPLEN'}

    def __init__(self, **kwargs):
        self._tmpdir = kwargs.get('tmpdir', '/tmp')
        self._da = None
        self._db = None

    def get_workdir(self):
        """Return work directory"""
        return self._tmpdir

    def set_workdir(self, tmpdir):
        """Set work directory"""
        self._tmpdir = tmpdir


    def get_overlap(self):
        return one.read(
            "OVERLAP",
            os.path.join(self.get_workdir(), "AOONEINT")
            ).unpack().unblock()

    def get_one_el_hamiltonian(self):
        """Get one-electron Hamiltonian"""
        return one.read(
            "ONEHAMIL",
            os.path.join(self.get_workdir(), "AOONEINT")
            ).unpack().unblock()

    def get_nuclear_repulsion(self):
        """Get nuclear repulsion energy"""
        return one.readhead(
            os.path.join(self.get_workdir(), "AOONEINT")
            )["potnuc"]

    def get_mo(self):
        """Get molecular orbitals from restart file"""
        rst = sirrst.SiriusRestart(
            os.path.join(self.get_workdir(), "SIRIUS.RST")
            )
        return rst.cmo.unblock()

    def set_densities(self, *das):
        """Set densities"""
        self._da, self._db = das

    def get_densities(self):
        """Get densities"""
        return self._da, self._db

    def get_two_el_fock(self):
        """Get focks"""
        return two.core.fockab(
            self.get_densities(),
            filename=os.path.join(self.get_workdir(), "AOTWOINT"),
            f2py=False
            )

    def get_two_el_right_hessian(self, d_am, delta, filename=None):
        """Calculate <K|H|d2L>"""
        if filename is None:#pragma: no cover
            filename=os.path.join(self.get_workdir(), "AOTWOINT")
        return two.vb.vb_transform(
            d_am, delta,
            filename=filename
            )

    def get_two_el_leftright_hessian(self, d_ma, d_am, delta1, delta2, filename=None):
        """Calculate <K|H|d2L>"""
        if filename is None:#pragma: no cover
            filename=os.path.join(self.get_workdir(), "AOTWOINT")
        return two.vb.vb_transform2(
            d_ma, d_am, delta1, delta2,
            filename=filename
            )

    def get_orbital_diagonal(self, filename=None):
        if filename is None:
            filename = os.path.join(self.get_workdir(), "SIRIFC")
        od = sirifc.SirIfc(filename).orbdiag
        return numpy.append(od, od)

    def get_overlap_diagonal(self, filename=None):
        if filename is None:
            filename = os.path.join(self.get_workdir(), "SIRIFC")
        ifc = sirifc.SirIfc(filename)
        lsd = ifc.nwopt
        sd = numpy.ones((2, lsd))
        for i in (0, 1):
            sd[i, ifc.nisht*ifc.nasht: ifc.nisht*(ifc.norbt - ifc.nisht)] *= 2
        sd[1, :] *= -1
        return sd.flatten()

    def get_rhs(self, *labels):
        return prop.grad(
            *(self.labels[label] for label in labels),
            tmpdir=self.get_workdir()
        )

    def e2n(self, trial):
        b = numpy.array(trial)
        u = numpy.ndarray(b.shape)
        if len(b.shape)== 1:
            u = oli.e2n(b, tmpdir=self.get_workdir())#.reshape(b.shape)
        elif len(b.shape) == 2:
            rows, columns = b.shape
            for c in range(columns):
                u[:, c] = oli.e2n(b[:, c], tmpdir=self.get_workdir())
        else:
            raise TypeError
        return u

    def s2n(self, trial):
        b = numpy.array(trial)
        u = numpy.ndarray(b.shape)
        if len(b.shape)== 1:
            u = oli.s2n(b, tmpdir=self.get_workdir())
        elif len(b.shape) == 2:
            rows, columns = b.shape
            for c in range(columns):
                u[:, c] = oli.s2n(b[:, c], tmpdir=self.get_workdir())
        else:
            raise TypeError
        return u

    def initial_guess(self, ops="xyz", freqs=(0,)):
        od = self.get_orbital_diagonal()
        sd = self.get_overlap_diagonal()
        dim = od.shape[0]
        ig = {}
        for op, grad in zip(ops, self.get_rhs(*ops)):
            gn = grad.norm2()
            for w in freqs:
                if gn < SMALL:
                    ig[(op, w)] = numpy.zeros(dim)
                else:
                    td = od - w*sd
                    ig[(op, w)] = grad/td
        return ig

    def setup_trials(self, vectors):
        """
        Set up initial trial vectors from a set of intial guesses
        """
        b = []
        for (op, freq), v in vectors.items():
            if numpy.linalg.norm(v) > SMALL:
                b.append(v)
                if freq > SMALL:
                    b.append(swap(v))
        return full.init(b)
                
            


    def lr_solve(self, ops="xyz", freqs=(0,), maxit=20, threshold=1e-5):
        from util.full import matrix

        V1 = {op: v for op, v in zip(ops, self.get_rhs(*ops))}
        igs = self.initial_guess(ops=ops, freqs=freqs)
        b = self.setup_trials(igs)
        # if the set of trial vectors is null we return the initial guess
        if not numpy.any(b):
            return igs

        td = {
            w: self.get_orbital_diagonal() - w*self.get_overlap_diagonal()
            for w in freqs
        }

        solutions = {}
        residuals = {}
        for i in range(maxit):
            e2b = self.e2n(b).view(matrix)
            s2b = self.s2n(b).view(matrix)
            for op, freq in igs:
                v = V1[op]
                n = b*((b.T*v)/(b.T*(e2b-freq*s2b)))
                r = self.e2n(n)-freq*self.s2n(n) - v
                solutions[(op, freq)] = n
                residuals[(op, freq)] = r
                print(f"{i+1} <<{op};{op}>>({freq})={-n&v:.6f} rn={r.norm2():.1e}", end='')
            print()
            max_residual = max(r.norm2() for r in residuals.values())
            if max_residual < threshold:
                break
            new_trials = []
            for (op, freq), r in residuals.items():
                rt = r/td[freq]
                new_trials.append(rt)
                if freq != 0:
                    new_trials.append(swap(rt))
            b = bappend(b, full.init(new_trials))
        return solutions

    def lr(self, aops, bops, freqs=(0,)):
        v1 = {op: v for op, v in zip(aops, self.get_rhs(*aops))}
        solutions = self.lr_solve(bops, freqs)
        lrs = {}
        for aop in aops:
            for bop, w in solutions:
                lrs[(aop, bop, w)] = -v1[aop]&solutions[(bop, w)]
        return lrs

def swap(xy):
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
    return numpy.append(b1, b2, axis=1).view(full.matrix)

class DaltonFactoryDummy(DaltonFactory):
    """Concrete dummy 'factory', Dalton"""

    def lr_solve(self, ops="xyz", freqs=(0.)):
        V1 = {op: v for op, v in zip(ops, self.get_rhs(*ops))}
        row_dim = V1[ops[0]].shape[0]
        E2 = full.init([self.e2n(i) for i in numpy.eye(row_dim)])
        S2 = full.init([self.s2n(i) for i in numpy.eye(row_dim)])
        solutions = {
            (op, freq): (V1[op]/(E2-freq*S2)) for freq in freqs for op in ops
        }
        return solutions

    def get_overlap_diagonal(self, filename=None):
        if filename is None:
            filename = os.path.join(self.get_workdir(), "SIRIFC")
        ifc = sirifc.SirIfc(filename)
        n = 2*ifc.nwopt
        sd = numpy.array(
            [
                oli.s2n(c, tmpdir=self.get_workdir())
                    for c in numpy.eye(n)
            ]
        ).diagonal()
        return sd

    def get_orbital_hessian(self, filename=None):
        if filename is None:
            filename = os.path.join(self.get_workdir(), "SIRIFC")
        ifc = sirifc.SirIfc(filename)
        n = 2*ifc.nwopt
        e2_diagonal = numpy.array(
            [
                oli.e2n(c, tmpdir=self.get_workdir())
                    for c in numpy.eye(n)
            ]
        ).diagonal()
        return e2_diagonal
