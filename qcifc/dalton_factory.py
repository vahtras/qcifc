import os
import subprocess

import numpy as np

from daltools import one, sirrst, sirifc, prop, rspvec, oli
import two.core
import two.vb
from util import full

from .core import QuantumChemistry, SMALL, swap, get_transform, bappend


class DaltonFactory(QuantumChemistry):
    """Concrete 'factory', Dalton"""

    labels = {'x': 'XDIPLEN', 'y': 'YDIPLEN', 'z': 'ZDIPLEN'}

    def __init__(self, **kwargs):
        self._tmpdir = kwargs.get('tmpdir', '/tmp')
        self._da = None
        self._db = None

    def update(self, text):
        print(text)

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
        fab =  two.core.fockab(
            *dab,
            filename=os.path.join(self.get_workdir(), "AOTWOINT"),
            f2py=False
            )
        return fab

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

    def _sirifc(self, filename=None):
        if filename is None:
            filename = os.path.join(self.get_workdir(), "SIRIFC")
        return sirifc.SirIfc(filename)

    def get_orbital_diagonal(self, filename=None, shift=0.):
        ifc = self._sirifc()
        fc = ifc.fc.unblock()

        od = []
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

    def generate_new_trials(self, residuals, td, b):
        return self.setup_trials(
            vectors=residuals, td=td, b=b, renormalize=True
        )


    def response_dim(self):
        filename = os.path.join(self.get_workdir(), "SIRIFC")
        ifc = sirifc.SirIfc(filename)
        return 2*ifc.nwopt

    #def pp(args, **kwargs):
    #    pass

    #def excitation_energies(*args, **kwargs):
        #pass

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

    def cleanup_scf(self):
        subprocess.call(
            'rm *.[0-9] DALTON.* *AO* *SIR* *RSP* molden.inp', shell=True
        )



class DaltonFactoryDummy(DaltonFactory):
    """Concrete dummy 'factory', Dalton"""

    def lr_solve(self, ops="xyz", freqs=(0.), **kwargs):
        V1 = {op: v for op, v in zip(ops, self.get_rhs(*ops))}
        E2, S2 = self._get_E2S2()
        solutions = {
            (op, freq): (V1[op]/(E2-freq*S2)) for freq in freqs for op in ops
        }
        return solutions

    def pp(self, ops="xyz", nfreqs=1, **kwargs):
        V1 = {op: v for op, v in zip(ops, self.get_rhs(*ops))}
        E2, S2 = self._get_E2S2()
        Xn = self.eigenvectors(nfreqs)
        solutions = {
            (op, i): (Xn[:, i] & V1[op]) for i in range(nfreqs) for op in ops
        }
        return solutions

    def get_overlap_diagonal(self, filename=None):
        n = self.response_dim()
        sd = np.array(
            [
                oli.s2n(c, tmpdir=self.get_workdir())
                for c in np.eye(n)
            ]
        ).diagonal()
        return sd

    def get_orbital_hessian(self, filename=None):
        if filename is None:
            filename = os.path.join(self.get_workdir(), "SIRIFC")
        ifc = sirifc.SirIfc(filename)
        n = 2*ifc.nwopt
        e2_diagonal = np.array(
            [
                oli.e2n(c, tmpdir=self.get_workdir())
                for c in np.eye(n)
            ]
        ).diagonal()
        return e2_diagonal

    def _get_E2S2(self):
        dim = 2*len(self.get_excitations())
        E2 = full.init([self.e2n(i) for i in np.eye(dim)])
        S2 = full.init([self.s2n(i) for i in np.eye(dim)])
        return E2, S2

    def excitation_energies(self, n_states):
        E2, S2 = self._get_E2S2()
        wn = (E2/S2).eig()
        return wn[len(wn)//2: len(wn)//2 + n_states]

    def eigenvectors(self, n_states):
        E2, S2 = self._get_E2S2()
        _, Xn = (E2/S2).eigvec()
        dim = len(E2)
        for i in range(dim//2, dim//2 + n_states):
            norm = np.sqrt(Xn[:, i].T*S2*Xn[:, i])
            Xn[:, i] /= norm
        return Xn[:, dim//2: dim//2 + n_states]
