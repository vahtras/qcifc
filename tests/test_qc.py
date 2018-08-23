import unittest
import os
import numpy
from qcifc.core import QuantumChemistry, DaltonFactory


class AFTest(unittest.TestCase):
    """Abstract Factory test"""

    def setUp(self):
        self.tmp = os.path.join(os.path.dirname(__file__), 'test_h2.d')
        self.factory = QuantumChemistry.get_factory(
            'Dalton',
            tmpdir=self.tmp
            )

        self.daref = numpy.array([
            [1., 0],
            [0., 1.]
            ])

        self.dbref = numpy.array([
            [1., 0],
            [0., 0]
            ])

        self.faref = numpy.array([
            [1.04701025 , 0.44459112],
            [0.44459112, 0.8423992]
            ])

        self.fbref = numpy.array([
            [1.34460081, 0.88918225],
            [0.88918225, 1.61700513]
            ])


    def tearDown(self):
        pass

    def test_create_dalton_factory(self):
        """Create 'concrete' factory'"""
        self.assertIsInstance(self.factory, DaltonFactory)

    def test_unknown_raises_typeerror(self):
        """Unknown code raises TypeError"""
        with self.assertRaises(TypeError):
            QuantumChemistry.get_factory('Gamess')

    def test_get_wrkdir(self):
        """Get factory workdir"""
        self.assertEqual(self.factory.get_workdir(), self.tmp)

    def test_set_wrkdir(self):
        """Get factory workdir"""
        self.factory.set_workdir('/tmp/123')
        self.assertEqual(self.factory.get_workdir(), '/tmp/123')

    def test_get_overlap(self):
        """Get overlap"""
        numpy.testing.assert_allclose(
            self.factory.get_overlap(),
            [[1.0, 0.65987313], [0.65987313, 1.0]]
            )

    def test_get_h1(self):
        """Get one-electron Hamiltonian"""
        numpy.testing.assert_allclose(
            self.factory.get_one_el_hamiltonian(),
            [[-1.12095946, -0.95937577], [-0.95937577, -1.12095946]]
            )

    def test_get_z(self):
        """Nuclear repulsion energy"""
        self.assertAlmostEqual(
            self.factory.get_nuclear_repulsion(),
            0.7151043
            )

    def test_get_mo(self):
        """Read MO coefficients"""
        cmo = self.factory.get_mo()
        numpy.testing.assert_allclose(cmo, [[.54914538, -1.20920006],
            [.54914538, 1.20920006]])

    def test_set__get_dens_a(self):
        """Set density test"""
        self.factory.set_densities(self.daref, self.dbref)
        d_a, _ = self.factory.get_densities()
        numpy.testing.assert_allclose(d_a, self.daref)

    def test_set__get_dens_b(self):
        """Set density test"""
        self.factory.set_densities(self.daref, self.dbref)
        _, d_b = self.factory.get_densities()
        numpy.testing.assert_allclose(d_b, self.dbref)
       
    def test_get_two_fa(self):
        """Get alpha Fock matrix"""
        self.factory.set_densities(self.daref, self.dbref)
        f_a, _ = self.factory.get_two_el_fock()
        numpy.testing.assert_allclose(f_a, self.faref)

    def test_get_two_fb(self):
        """Get beta Fock matrix"""
        self.factory.set_densities(self.daref, self.dbref)
        _, f_b = self.factory.get_two_el_fock()
        numpy.testing.assert_allclose(f_b, self.fbref)

    def test_get_orbhess(self):
        """Get diagonal orbital hessian"""
        od = self.factory.get_orbital_diagonal() 
        numpy.testing.assert_allclose(od, [4.998789307784559])

if __name__ == "__main__":
    unittest.main()
