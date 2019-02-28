import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from qcifc.dalton_factory import DaltonFactoryDummy, DaltonFactory

from qcifc.vlx import VeloxChem

codes = [
    DaltonFactoryDummy,
    DaltonFactory,
    VeloxChem,
]
