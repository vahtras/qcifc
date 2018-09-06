"""Abstrace interfact to QM codes"""
import abc
import os
from util import full
import numpy as np

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
        #fix
        ig = []
        for v in self.get_rhs(*ops):
            for w  in freqs:
                td = od - w*sd
                ig.append(v/td)
                if w != 0:
                    ig.append(swap(v/td))
        return numpy.array(ig).T

    def lr_solve(self, ops="xyz", freqs=(0,), maxit=20, threshold=1e-5):
        from util.full import matrix
        v = self.get_rhs(*ops)[0].view(matrix)
        b  = self.initial_guess(ops=ops, freqs=freqs).view(matrix)
        td = [self.get_orbital_diagonal() - w*self.get_overlap_diagonal()
            for w in freqs]
        for i in range(maxit):
            e2b = self.e2n(b).view(matrix)
            s2b = self.s2n(b).view(matrix)
            solutions=[]
            for w in freqs:
                n = b*((b.T*v)/(b.T*(e2b-w*s2b)))
                solutions.append(n)
            residuals = []
            for w, n in zip(freqs, solutions):
                r = self.e2n(n)-w*self.s2n(n) - v
                residuals.append(r)
                print(i+1, w, -n&v, r.norm2())
            max_residual = max(r.norm2() for r in residuals)
            if max_residual < threshold:
                break
            new_trials = []
            for r, t in zip(residuals, td):
                rt = r/t
                new_trials.append(rt)
                if w != 0:
                    new_trials.append(swap(rt))
            b = bappend(b, full.init(new_trials))
        return numpy.array(solutions).view(matrix)

    def lr(self, label, freqs=(0,)):
        a, b = label.split(';')
        n = self.lr_solve(ops=b, freqs=freqs)
        v, = self.get_rhs(a)
        return tuple(-(k&v) for k in n)

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
        #import pdb; pdb.set_trace()
        V = self.get_rhs(*ops)
        row_dim = V[0].shape[0]
        E2 = full.init([self.e2n(i) for i in numpy.eye(row_dim)])
        S2 = full.init([self.s2n(i) for i in numpy.eye(row_dim)])
        return [v/(E2-w*S2) for w in freqs for v in V]

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
