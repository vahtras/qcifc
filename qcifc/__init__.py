from . import core
from .dalton_factory import DaltonFactory, DaltonFactoryDummy
from .vlx import VeloxChem

program = {
    'dummy': DaltonFactoryDummy,
    'dalton': DaltonFactory,
    'veloxchem': VeloxChem,
}
