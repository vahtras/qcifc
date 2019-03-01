import os
import sys

import pytest

sys.path.insert(0, os.path.abspath('..'))
from qcifc.dalton_factory import DaltonFactoryDummy, DaltonFactory
from qcifc.vlx import VeloxChem

codes = {
    'dummy': DaltonFactoryDummy,
    'dalton': DaltonFactory,
    'vlx': VeloxChem,
}


class TestQC:
    @staticmethod
    def skip_if_not_implemented(method, code):
        if method not in dir(code):
            pytest.skip('not implemented')
