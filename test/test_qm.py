import unittest
import os
import numpy
from ..qm import QuantumChemistry, DaltonFactory


class AFTest(unittest.TestCase):

    def setUp(self):
        self.tmp = os.path.join(os.path.dirname(__file__), 'test_h2.d')
        self.factory = QuantumChemistry.get_factory('Dalton', self.tmp)
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

if __name__ == "__main__":
    unittest.main()
