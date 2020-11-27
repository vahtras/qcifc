from collections import deque
import os
import subprocess

import numpy as np

from daltools import one, sirrst, sirifc, prop, rspvec, dens
from dalmisc import oli, rohf
from dalmisc.scf_iter import URoothanIterator
import two.core

from .core import QuantumChemistry, RoothanIterator


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

    def set_roothan_iterator(self, mol, **kwargs):
        self.roothan = DaltonRoothanIterator(**kwargs)

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

    def __iter__(self):
        """
        Initial setup for SCF iterations
        """
        self.na = (self.nel + 2*self.ms)//2
        self.nb = (self.nel - 2*self.ms)//2

        AOONEINT = os.path.join(self.tmpdir, 'AOONEINT')
        self.Z = one.readhead(AOONEINT)['potnuc']
        self.h1 = one.read(
            label='ONEHAMIL', filename=AOONEINT
        ).unpack().unblock()
        self.S = one.read(
            label='OVERLAP', filename=AOONEINT
        ).unpack().unblock()
        if self.C is None:
            self.Ca = dens.cmo(self.h1, self.S)
            self.Cb = self.Ca
            self.C = (self.Ca, self.Cb)

        return self

    def set_densities(self):
        Ca, Cb = self.C
        self.Da = dens.C1D(Ca, self.na)
        self.Db = dens.C1D(Cb, self.nb)

    def set_focks(self):
        AOTWOINT = os.path.join(self.tmpdir, 'AOTWOINT')
        (self.Fa, self.Fb), = two.core.fockab(
            (self.Da, self.Db),
            filename=AOTWOINT
        )

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


class DiisIterator(RoothanIterator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evecs = deque()
        self.vecs = deque()
        self.max_vecs = kwargs.get('max_vecs', 2)

    def __next__(self):
        """
        Updates for in a SCF iteration
        """
        if not self.converged() and self.it < self.max_iterations:
            self.it += 1
            self.set_densities()
            self.set_focks()
            e = self.energy()
            gn = self.gn()

            self.vecs.append(self.Feff())
            self.evecs.append(self.ga + self.gb)
            if len(self.vecs) > self.max_vecs:
                self.vecs.popleft()
                self.evecs.popleft()

            self.update_mo()
            self.energies.append(e)
            self.gradient_norms.append(gn)
            return (e, gn)
        else:
            raise StopIteration

    def B(self):
        dim = len(self.evecs) + 1
        Bmat = np.ones((dim, dim))
        for i, vi in enumerate(self.evecs):
            for j, vj in enumerate(self.evecs):
                Bmat[i, j] = 4*(vi & (self.S.I@vj@self.S.I))

        Bmat[-1, -1] = 0
        return Bmat

    def c(self):
        rhs = np.zeros(len(self.evecs) + 1)
        rhs[-1] = 1.0
        return np.linalg.solve(self.B(), rhs)[:-1]

    def Fopt(self):
        return sum(
            c*e
            for c, e in zip(self.c(), self.vecs)
        )

    def update_mo(self):

        F = self.Fopt()
        Ca = dens.cmo(F, self.S)
        Cb = Ca
        self.C = Ca, Cb
