import  unittest
import os
import sys
import subprocess
import numpy
from util.full import matrix, unit
from . import qcifc


class TestBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        root, ext = n, e = os.path.splitext(__file__)
        cls.base_dir = root + ".d"
        
class TestH2O(TestBase):

    @classmethod
    def setUpClass(cls):
        super(TestH2O, cls).setUpClass()
        cls.subdir = "H2O"
        cls.dal = "hf"
        cls.mol = "H2O_STO3G"
        cls.filename = "%s_%s.AOTWOINT" % (cls.dal, cls.mol)
        cls.tmpdir = os.path.join(cls.base_dir, cls.subdir)
        cls.aotwoint = os.path.join(cls.tmpdir, cls.filename)
        if not os.path.exists(cls.aotwoint):
            os.chdir(cls.tmpdir)
            args = ['dalton', '-get', 'AOTWOINT', cls.dal, cls.mol]
            #subprocess.call(args)

    def setUp(self):
        numpy.random.seed(0)
        da = numpy.random.random((7, 2)).view(matrix)
        db = numpy.random.random((7, 2)).view(matrix)
        Da = numpy.random.random((7, 7)).view(matrix)
        Db = numpy.random.random((7, 7)).view(matrix)
        
        self.d = (da, db)
        self.dT = (da.T, db.T)
        self.D = (Da, Db)
        self.H1 = numpy.load(os.path.join(self.tmpdir, 'H1.npy'))
        self.H2 = numpy.load(os.path.join(self.tmpdir, 'H2.npy'))

        self.factory = qcifc.core.QuantumChemistry.set_code(
            'Dalton',
            tmpdir=self.tmpdir
            )



    def test_vb_transform(self):
        H = self.factory.get_two_el_right_hessian(
            self.d,
            self.D,
            filename=self.aotwoint
            )
        numpy.testing.assert_almost_equal(H, self.H1)

    #@unittest.skip('not yet')
    def test_vb_transform2(self):
        H = self.factory.get_two_el_leftright_hessian(
            self.dT,
            self.d,
            self.D,
            self.D,
            filename=self.aotwoint
            )
        #H = vb.vb_transform2(self.dT, self.d, self.D, self.D, filename=self.aotwoint)
        numpy.testing.assert_almost_equal(H, self.H2)


if __name__ == "__main__":
    unittest.main()

    
