import unittest
import os
import numpy
from qcifc.qm import QuantumChemistry, DaltonFactory


class AFTest(unittest.TestCase):

    def setUp(self):
        self.tmp = os.path.join(os.path.dirname(__file__), 'test_h2.d')
        self.factory = QuantumChemistry.get_factory('Dalton', 
            tmpdir=self.tmp
            )

    def tearDown(self):
        pass

    def test_create_dalton_factory(self):
        self.assertIsInstance(self.factory, DaltonFactory)

    def test_unknown_raises_typeerror(self):
        with self.assertRaises(TypeError):
            factory = QuantumChemistry.get_factory('Gamess')

    def test_wrkdir(self):
        self.assertEqual(self.factory.get_workdir(), self.tmp)

    def test_get_overlap(self):
        S = self.factory.get_overlap()
        numpy.testing.assert_allclose(S, [[1.0, 0.65987313], [0.65987313, 1.0]])

    def test_get_h1(self):
        h1 = self.factory.get_one_el_hamiltonian()
        numpy.testing.assert_allclose(h1, 
            [[-1.12095946, -0.95937577], [-0.95937577, -1.12095946]]
            )

    def test_get_z(self):
        Z = self.factory.get_nuclear_repulsion()
        self.assertAlmostEqual(Z, 0.7151043)


if __name__ == "__main__":
    unittest.main()
