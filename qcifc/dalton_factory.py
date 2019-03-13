import os
import subprocess

import numpy

from daltools import one, sirrst, sirifc, prop, rspvec
from dalmisc import oli
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

    def get_two_el_fock(self, da, db):
        """Get focks"""
        return two.core.fockab(
            (da, db),
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

    def _sirifc(self, filename=None):
        if filename is None:
            filename = os.path.join(self.get_workdir(), "SIRIFC")
        return sirifc.SirIfc(filename)

    def get_orbital_diagonal(self, filename=None):
        od = self._sirifc(filename).orbdiag
        return numpy.append(od, od)

    def get_overlap_diagonal(self, filename=None):
        ifc = self._sirifc(filename)
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

    def vec2mat(self, vec):
        ifc = self._sirifc()
        mat = rspvec.tomat(vec, ifc)
        return mat

    def mat2vec(self, mat):
        ifc = self._sirifc()
        mat = rspvec.tovec(numpy.array(mat), ifc)
        return mat

    def e2n(self, trial):
        b = numpy.array(trial)
        u = numpy.ndarray(b.shape)
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
        b = numpy.array(trial)
        u = numpy.ndarray(b.shape)

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

    def lr_solve(self, ops="xyz", freqs=(0,), maxit=25, threshold=1e-5):
        from util.full import matrix

        V1 = {op: v for op, v in zip(ops, self.get_rhs(*ops))}
        igs = self.initial_guess(ops=ops, freqs=freqs)
        b = self.setup_trials(igs)
        # if the set of trial vectors is null we return the initial guess
        if not numpy.any(b):
            return {k: v.view(matrix) for k, v in igs.items()}
        e2b = self.e2n(b).view(matrix)
        s2b = self.s2n(b).view(matrix)

        od = self.get_orbital_diagonal()
        sd = self.get_overlap_diagonal()
        td = {w: od - w*sd for w in freqs}

        solutions = {}
        residuals = {}
        for i in range(maxit):
            for op, freq in igs:
                v = V1[op]
                n = b*((b.T*v)/(b.T*(e2b-freq*s2b)))
                r = self.e2n(n)-freq*self.s2n(n) - v
                solutions[(op, freq)] = n
                residuals[(op, freq)] = r
                print(f"{i+1} <<{op};{op}>>({freq})={-n&v:.6f} rn={r.norm2():.2e} ", end='')
            print()
            max_residual = max(r.norm2() for r in residuals.values())
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

    def lr(self, aops, bops, freqs=(0,), **kwargs):
        v1 = {op: v for op, v in zip(aops, self.get_rhs(*aops))}
        solutions = self.lr_solve(bops, freqs, **kwargs)
        lrs = {}
        for aop in aops:
            for bop, w in solutions:
                lrs[(aop, bop, w)] = -v1[aop]&solutions[(bop, w)]
        return lrs

    def response_dim(self):
        filename = os.path.join(self.get_workdir(), "SIRIFC")
        ifc = sirifc.SirIfc(filename)
        return 2*ifc.nwopt

    #def pp(args, **kwargs):
    #    pass

    #def excitation_energies(*args, **kwargs):
        #pass

    def run_scf(self):
        os.chdir(self.get_workdir())
        subprocess.call(
            ['dalton', '-get', 'AOPROPER AOONEINT AOTWOINT', 'hf', self.case]
        )
        subprocess.call(['tar', 'xvfz', f'hf_{self.case}.tar.gz'])

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

    def _get_E2S2(self):
        filename = os.path.join(self.get_workdir(), "SIRIFC")
        ifc = sirifc.SirIfc(filename)
        dim = 2*ifc.nwopt
        E2 = full.init([self.e2n(i) for i in numpy.eye(dim)])
        S2 = full.init([self.s2n(i) for i in numpy.eye(dim)])
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
            norm = numpy.sqrt(Xn[:, i].T*S2*Xn[:, i])
            Xn[:, i] /= norm
        return Xn[:, dim//2: dim//2 + n_states]
