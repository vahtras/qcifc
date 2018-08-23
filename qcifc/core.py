"""Abstrace interfact to QM codes"""
import abc
import os

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



from daltools import one, sirrst, sirifc
import two.core
import two.vb

class DaltonFactory(QuantumChemistry):
    """Concrete 'factory', Dalton"""

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
        return sirifc.SirIfc(filename).orbdiag
