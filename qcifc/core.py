"""Abstrace interfact to QM codes"""
import abc
import os
from util import full

class QuantumChemistry(object):
    """Abstract factory"""
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def get_factory(code, **kwargs):
        """Return concrete factory"""
        if code == 'Dalton':
            return DaltonFactory(**kwargs)
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

    labels = {'z': 'ZDIPLEN'}

    def __init__(self, **kwargs):
        self.__tmpdir = kwargs.get('tmpdir', '/tmp')
        self._da = None
        self._db = None

    def get_workdir(self):
        """Return work directory"""
        return self.__tmpdir

    def set_workdir(self, tmpdir):
        """Set work directory"""
        self.__tmpdir = tmpdir


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

    def get_rhs(self, label):
        return prop.grad(
            self.labels[label],
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

    def initial_guess(self, label, w=0):
        V = self.get_rhs(label)
        od = self.get_orbital_diagonal()
        #fix
        sd = 2*numpy.ones(len(od))
        sd[len(od)//2:] = -2
        td = od - w*sd
        ig = (V/td).reshape((len(od), 1))
        if w != 0:
            ig = bappend(ig, swap(ig))
        return ig

    def lr_solve(self, label, w=0):
        from util.full import matrix
        b  = self.initial_guess(label, w).view(matrix)
        maxit = 10
        for i in range(maxit):
            e2b = self.e2n(b).view(matrix)
            s2b = self.s2n(b).view(matrix)
            t2r = b.T*(e2b-w*s2b)
            print(t2r)
            v = self.get_rhs(label)[0].view(matrix)
            vr = b.T*v
            nr = vr/t2r
            n = b*nr
            resid = (self.e2n(n) - v).norm2()
            print(i, -n*v, resid)
            if resid < 1e-8:
                break
        return n

def swap(t):
    from util import full
    r, c = t.shape
    print(r, c)
    assert c == 1
    rh = r // 2
    new = full.init([numpy.append(t[rh:, :], t[:rh, :])])
    return new

def bappend(b1, b2):
    return numpy.append(b1, b2, axis=1).view(full.matrix)
