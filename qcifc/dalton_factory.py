import os
import subprocess

import numpy as np

from daltools import one, sirrst, sirifc, prop, rspvec, dens
from dalmisc import oli, rohf
from dalmisc.scf_iter import URoothanIterator
import two.core

from .core import QuantumChemistry, RoothanIterator, DiisIterator


class DaltonFactory(QuantumChemistry):
    """Concrete 'factory', Dalton"""

    labels = {'x': 'XDIPLEN', 'y': 'YDIPLEN', 'z': 'ZDIPLEN'}

    def __init__(self, **kwargs):
        self._tmpdir = kwargs.get('tmpdir', '/tmp')
        self._da = None
        self._db = None
        self.observers = []

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

    def get_densities(self, ifcfile="SIRIFC"):
        return oli.get_densities(ifcfile)

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

    def get_dipole(self):
        """Get dipole integrals"""
        return prop.read(
            "XDIPLEN",
            "YDIPLEN",
            "ZDIPLEN",
            filename=os.path.join(self.get_workdir(), "AOPROPER")
            )

    def get_mo(self):
        """Get molecular orbitals from restart file"""
        rst = sirrst.SiriusRestart(
            os.path.join(self.get_workdir(), "SIRIUS.RST")
            )
        return rst.cmo.unblock()

    def get_two_el_fock(self, *dab):
        """Get focks"""
        fab = two.core.fockab(
            *dab,
            filename=os.path.join(self.get_workdir(), "AOTWOINT"),
            f2py=False
            )
        return fab

    def get_number_of_electrons(self):
        ifc = self._sirifc()
        electrons = 2*ifc.nisht + ifc.nasht
        return electrons

    def _sirifc(self, filename=None):
        if filename is None:
            filename = os.path.join(self.get_workdir(), "SIRIFC")
        return sirifc.SirIfc(filename)

    def get_orbital_diagonal(self, filename=None, shift=0.):
        ifc = self._sirifc()
        try:
            od = ifc.orbdiag * .5
        except sirifc.LabelNotFound:
            print("Fix me")
            od = []
            fc = ifc.fc.unblock()
            for i in range(ifc.nisht):
                for a in range(ifc.nisht, ifc.norbt):
                    od.append(2*(fc[a, a] - fc[i, i]))
        return np.append(od, od) + shift

    def get_overlap_diagonal(self, filename=None):
        ifc = self._sirifc(filename)
        lsd = ifc.nwopt
        sd = np.ones((2, lsd))
        for i in (0, 1):
            sd[i, ifc.nisht*ifc.nasht: ifc.nisht*(ifc.norbt - ifc.nisht)] *= 2
        sd[1, :] *= -1
        return sd.flatten()

    def get_rhs(self, *labels):
        return prop.grad(
            *(self.labels[label] for label in labels),
            tmpdir=self.get_workdir()
        )

    def vec2mat(self, vec):
        ifc = self._sirifc()
        mat = rspvec.tomat(vec, ifc)
        return mat

    def mat2vec(self, mat):
        ifc = self._sirifc()
        mat = rspvec.tovec(np.array(mat), ifc)
        return mat

    def e2n(self, trial):
        b = np.array(trial)
        u = np.ndarray(b.shape)
        if len(b.shape) == 1:
            u = oli.e2n(b, tmpdir=self.get_workdir())  # .reshape(b.shape)
        elif len(b.shape) == 2:
            rows, columns = b.shape
            for c in range(columns):
                u[:, c] = oli.e2n(b[:, c], tmpdir=self.get_workdir())
        else:
            raise TypeError
        return u

    def s2n(self, trial):
        b = np.array(trial)
        u = np.ndarray(b.shape)

        if len(b.shape) == 1:
            u = oli.s2n(b, tmpdir=self.get_workdir())
        elif len(b.shape) == 2:
            rows, columns = b.shape
            for c in range(columns):
                u[:, c] = oli.s2n(b[:, c], tmpdir=self.get_workdir())
        else:
            raise TypeError
        return u

    def response_dim(self):
        filename = os.path.join(self.get_workdir(), "SIRIFC")
        ifc = sirifc.SirIfc(filename)
        return 2*ifc.nwopt

    def get_excitations(self):
        ifc = self._sirifc()
        excitations = list(rspvec.jwop(ifc))
        return excitations

    def run_scf(self, mol):
        cwd = os.getcwd()
        os.chdir(self.get_workdir())
        subprocess.call(
            ['dalton', '-get', 'AOPROPER AOONEINT AOTWOINT', 'hf', mol]
        )
        subprocess.call(['tar', 'xvfz', f'hf_{mol}.tar.gz'])
        os.chdir(cwd)

    def set_scf_iterator(self, algorithm, *args, **kwargs):
        iterator = iterators[algorithm]
        self.scf = iterator(self, **kwargs)

    def set_roothan_iterator(self, *args, **kwargs):
        self.roothan = DaltonRoothanIterator(self, **kwargs)

    def set_diis_iterator(self, mol, **kwargs):
        self.diis = DiisIterator(**kwargs)

    def run_uroothan_iterations(self, mol, **kwargs):
        roothan = URoothanIterator(**kwargs)
        for i, (e, gn) in enumerate(roothan):
            print(f'{i:2d}: {e:14.10f} {gn:.3e}')
        return e, gn

    def run_diis_iterations(self, mol, **kwargs):
        diis = DiisIterator(**kwargs)
        for i, (e, gn) in enumerate(diis):
            print(f'{i:2d}: {e:14.10f} {gn:.3e}')
        return e, gn

    def cleanup_scf(self):
        subprocess.call(
            'rm *.[0-9] DALTON.* *AO* *SIR* *RSP* molden.inp', shell=True
        )




class DaltonFactoryDummy(DaltonFactory):
    """Concrete dummy 'factory', Dalton"""

    def get_overlap_diagonal(self, filename=None):
        n = self.response_dim()
        sd = np.array(
            [
                oli.s2n(c, tmpdir=self.get_workdir())
                for c in np.eye(n)
            ]
        ).diagonal()
        return sd

    def lr_solve(self, ops="xyz", freqs=(0.), **kwargs):
        return self.direct_lr_solver(ops, freqs, **kwargs), []

    def pp_solve(self, n_states):
        return self.direct_ev_solver2(n_states)


class DaltonRoothanIterator(RoothanIterator):

    def update_mo(self):

        F = self.Feff()
        Ca = dens.cmo(F, self.S)
        Cb = Ca
        self.C = Ca, Cb

    def Feff2(self):
        Fa = self.S.I@(self.h1 + self.Fa)
        Fb = self.S.I@(self.h1 + self.Fb)
        Da = self.Da@self.S
        Db = self.Db@self.S
        return self.S@rohf.Feff(Da, Db, Fa, Fb)


class DaltonDiisIterator(DiisIterator):
    pass


iterators = {
    'roothan': DaltonRoothanIterator,
    'diis': DaltonDiisIterator,
}
