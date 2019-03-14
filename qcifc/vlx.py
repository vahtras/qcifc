import os
import itertools

from mpi4py import MPI
import numpy as np
import veloxchem as vlx
from veloxchem.veloxchemlib import szblock

from .core import QuantumChemistry


class VeloxChem(QuantumChemistry):

    def __init__(self, **kwargs):
        self._tmpdir = kwargs.get('tmpdir', '/tmp')
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.basis = kwargs.get('basis')
        self.xyz = kwargs.get('xyz')
        self.inp = kwargs.get('inp')
        self.out = kwargs.get('out')

    def is_master(self):
        return self.rank == vlx.mpi_master()

    def get_workdir(self):
        return self._tmpdir

    def set_workdir(self, tmpdir):
        self._tmpdir = tmpdir

    def np2vlx(self, vec):
        nocc = self.task.molecule.number_of_electrons() // 2
        norb = self.scf_driver.mol_orbs.number_mos()
        xv = vlx.ExcitationVector(szblock.aa, 0, nocc, nocc, norb, True)
        zlen = len(vec) // 2
        z, y = vec[:zlen], vec[zlen:]
        xv.set_yzcoefficients(z, y)
        return xv

    def vec2mat(self, vec):
        xv = self.np2vlx(vec)
        kz = xv.get_zmatrix()
        ky = xv.get_ymatrix()

        rows = kz.number_of_rows() + ky.number_of_rows()
        cols = kz.number_of_columns() + ky.number_of_columns()

        kzy = np.zeros((rows, cols))
        kzy[:kz.number_of_rows(), ky.number_of_columns():] = kz.to_numpy()
        kzy[kz.number_of_rows():, :ky.number_of_columns()] = ky.to_numpy()

        return kzy

    def get_excitations(self):
        nocc = self.task.molecule.number_of_electrons() // 2
        norb = self.scf_driver.mol_orbs.number_mos()
        xv = vlx.ExcitationVector(szblock.aa, 0, nocc, nocc, norb, True)
        cre = xv.bra_unique_indexes()
        ann = xv.ket_unique_indexes()
        excitations = itertools.product(cre, ann)
        return excitations

    def mat2vec(self, mat):
        excitations = list(self.get_excitations())

        m = np.array(mat)
        z = [m[i, j] for i, j in excitations]
        y = [m[j, i] for i, j in excitations]

        return np.array(z + y)

    def get_overlap(self):
        overlap_driver = vlx.OverlapIntegralsDriver(
            self.rank, self.size, self.comm
        )
        master_node = (self.rank == vlx.mpi_master())
        if master_node:
            mol = vlx.Molecule.read_xyz(self.xyz)
            bas = vlx.MolecularBasis.read(mol, self.basis)
        else:
            mol = vlx.Molecule()
            bas = vlx.MolecularBasis()

        mol.broadcast(self.rank, self.comm)
        bas.broadcast(self.rank, self.comm)

        S = overlap_driver.compute(mol, bas, self.comm)
        return S.to_numpy()

    def get_dipole(self):
        dipole_driver = vlx.ElectricDipoleIntegralsDriver(
            self.rank, self.size, self.comm
        )
        master_node = (self.rank == vlx.mpi_master())
        if master_node:
            mol = vlx.Molecule.read_xyz(self.xyz)
            bas = vlx.MolecularBasis.read(mol, self.basis)
        else:
            mol = vlx.Molecule()
            bas = vlx.MolecularBasis()

        mol.broadcast(self.rank, self.comm)
        bas.broadcast(self.rank, self.comm)

        D = dipole_driver.compute(mol, bas, self.comm)
        return D.x_to_numpy(), D.y_to_numpy(), D.z_to_numpy()

    def get_one_el_hamiltonian(self):
        kinetic_driver = vlx.KineticEnergyIntegralsDriver(
            self.rank, self.size, self.comm
        )
        potential_driver = vlx.NuclearPotentialIntegralsDriver(
            self.rank, self.size, self.comm
        )
        master_node = (self.rank == vlx.mpi_master())
        if master_node:
            mol = vlx.Molecule.read_xyz(self.xyz)
            bas = vlx.MolecularBasis.read(mol, self.basis)
        else:
            mol = vlx.Molecule()
            bas = vlx.MolecularBasis()

        mol.broadcast(self.rank, self.comm)
        bas.broadcast(self.rank, self.comm)

        T = kinetic_driver.compute(mol, bas, self.comm).to_numpy()
        V = potential_driver.compute(mol, bas, self.comm).to_numpy()

        return T-V

    def get_nuclear_repulsion(self):
        mol = vlx.Molecule.read_xyz(self.xyz)
        return mol.nuclear_repulsion_energy()

    def run_scf(self):
        self.scf_driver = vlx.ScfRestrictedDriver()
        inp = str(self.inp)
        out = str(self.out)
        os.chdir(self.get_workdir())
        self.task = vlx.MpiTask((inp, out), self.comm)
        self.scf_driver.compute_task(self.task)

    def get_mo(self):
        mos = self.scf_driver.mol_orbs.alpha_to_numpy()
        return mos

    def set_densities(self, da, db):
        from veloxchem import denmat
        self._da = vlx.AODensityMatrix(
            [np.array(da), np.array(db)], denmat.rest
        )

    def get_densities(self):
        try:
            da = self._da.alpha_to_numpy(0)
            db = self._da.beta_to_numpy(1)
        except AttributeError:
            da = self.scf_driver.density.alpha_to_numpy(0)
            db = self.scf_driver.density.beta_to_numpy(0)
        return da, db

    def get_two_el_fock(self, da, db):
        from veloxchem.veloxchemlib import denmat, fockmat
        master_node = (self.rank == vlx.mpi_master())
        if master_node:
            mol = vlx.Molecule.read_xyz(self.xyz)
            bas = vlx.MolecularBasis.read(mol, self.basis)
        else:
            mol = vlx.Molecule()
            bas = vlx.MolecularBasis()
        mol.broadcast(self.rank, self.comm)
        bas.broadcast(self.rank, self.comm)

        dt = da + db
        ds = da - db
        dens = vlx.AODensityMatrix([dt, ds], denmat.rest)
        fock = vlx.AOFockMatrix(dens)
        fock.set_fock_type(fockmat.rgenjk, 0)
        fock.set_fock_type(fockmat.rgenk, 1)

        eri_driver = vlx.ElectronRepulsionIntegralsDriver(
            self.rank, self.size, self.comm
        )
        screening = eri_driver.compute(vlx.ericut.qqden, 1.0e-12, mol, bas)
        eri_driver.compute(vlx.ericut.qqden, 1.0e-12, mol, bas)
        eri_driver.compute(fock, dens, mol, bas, screening, self.comm)
        fock.reduce_sum(self.rank, self.size, self.comm)

        ft = fock.to_numpy(0).T
        fs = -fock.to_numpy(1).T

        fa = (ft + fs)/2
        fb = (ft - fs)/2

        return fa, fb

    def get_orbital_diagonal(self):

        orben = self.scf_driver.mol_orbs.ea_to_numpy()
        z = [4*(orben[j] - orben[i]) for i, j in self.get_excitations()]
        e2c = np.array(z + z)
        return e2c

    def get_overlap_diagonal(self):
        lz = len(list(self.get_excitations()))
        s2d = 2.0*np.ones(2*lz)
        s2d[lz:] = -2.0
        return s2d


    def get_rhs(self, *args):
        """
        Create right-hand sides of linear response equations

        Input: args, string labels of operators
               currently supported:
                    electric dipoles: x, y, z
        Output:
               operator gradients V(p, q)[1] = <0|[E(p, q), V]|0>
               in same order as args
        """
        if 'x' in args or 'y' in args or 'z' in args:
            props = {k: v for k, v in zip('xyz', self.get_dipole())}
        Da, Db = self.get_densities()
        D = Da + Db
        S = self.get_overlap()
        mo = self.get_mo()

        matrices = tuple(
            mo.T@(S@D@props[p].T - props[p].T@D@S)@mo for p in args
        )
        gradients = tuple(self.mat2vec(m) for m in matrices)
        return gradients

    def s2n(self, vecs):

        b = np.array(vecs)

        S = self.get_overlap()
        da, db = self.get_densities()
        D = da + db
        mo = self.get_mo()

        if len(b.shape) == 1:
            kappa = self.vec2mat(vecs).T
            kappa_ao = mo @ kappa @ mo.T

            s2n_ao = kappa_ao.T@S@D - D@S@kappa_ao.T
            s2n_mo = mo.T @ S @ s2n_ao @ S@mo
            s2n_vecs = - self.mat2vec(s2n_mo)
        elif len(b.shape) == 2:
            s2n_vecs = np.ndarray(b.shape)
            rows, columns = b.shape
            for c in range(columns):
                kappa = self.vec2mat(b[:, c]).T
                kappa_ao = mo @ kappa @ mo.T

                s2n_ao = kappa_ao.T@S@D - D@S@kappa_ao.T
                s2n_mo = mo.T @ S @ s2n_ao @ S@mo
                s2n_vecs[:, c] = - self.mat2vec(s2n_mo)
        return s2n_vecs

    def e2n(self, vecs):
        vecs = np.array(vecs)

        S = self.get_overlap()
        da, db = self.get_densities()
        fa, fb = self.get_two_el_fock(da, db)
        h = self.get_one_el_hamiltonian()
        fa += h
        fb += h
        mo = self.get_mo()

        if len(vecs.shape) == 1:


            kN = self.vec2mat(vecs).T
            kn = mo @ kN @ mo.T

            dak = kn.T@S@da - da@S@kn.T
            dbk = kn.T@S@db - db@S@kn.T

            fak, fbk = self.get_two_el_fock(dak, dbk)

            kfa = S@kn@fa - fa@kn@S
            kfb = S@kn@fa - fa@kn@S

            fat = fak + kfa
            fbt = fbk + kfb

            gao = S@(da@fat.T + db@fbt.T) - (fat.T@da + fbt.T@db)@S
            gmo = mo.T @ gao @ mo

            gv = - self.mat2vec(gmo)
        else:
            gv = np.zeros(vecs.shape)
            for col in range(vecs.shape[1]):
                vec = vecs[:, col]

                kN = self.vec2mat(vec).T
                kn = mo @ kN @ mo.T

                dak = kn.T@S@da - da@S@kn.T
                dbk = kn.T@S@db - db@S@kn.T

                fak, fbk = self.get_two_el_fock(dak, dbk)

                kfa = S@kn@fa - fa@kn@S
                kfb = S@kn@fa - fa@kn@S

                fat = fak + kfa
                fbt = fbk + kfb

                gao = S@(da@fat.T + db@fbt.T) - (fat.T@da + fbt.T@db)@S
                gmo = mo.T @ gao @ mo

                gv[:, col] = -self.mat2vec(gmo)

        return gv

