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
            [1., 0., 0., 0., 0., 0],
            [0., 1., 0., 0., 0., 0],
            [0., 0., 0., 0., 0., 0],
            [0., 0., 0., 0., 0., 0],
            [0., 0., 0., 0., 0., 0],
            [0., 0., 0., 0., 0., 0]
            ])

        self.dbref = numpy.array([
            [1., 0., 0., 0., 0., 0],
            [0., 0., 0., 0., 0., 0],
            [0., 0., 0., 0., 0., 0],
            [0., 0., 0., 0., 0., 0],
            [0., 0., 0., 0., 0., 0],
            [0., 0., 0., 0., 0., 0]
            ])

        self.faref = numpy.array([
            [2.02818057, 0.26542036, 0.00000000, 0.06037429, 0.00000000, 0.00000000],
            [0.26542036, 0.74551226, 0.00000000, 0.28646605, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 1.01061061, -0.47343699, 0.00000000, 0.00000000],
            [0.06037429, 0.28646605, -0.47343699, 0.86039561, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 0.00000000, 1.01061061, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 1.01061061],
            ])

        self.fbref = numpy.array([
            [2.07812203, 0.35828051, 0.00000000, 0.09571548, 0.00000000, 0.00000000],
            [0.35828051, 1.03607456, 0.00000000, 0.40151695, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 1.07479494, -0.50700370, 0.00000000, 0.00000000],
            [0.09571548, 0.40151695, -0.50700370, 0.93374109, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 0.00000000, 1.07479494, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 1.07479494],
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

if __name__ == "__main__":
    unittest.main()
