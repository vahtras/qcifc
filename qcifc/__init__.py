from . import core
from .dalton_factory import DaltonFactory
from .vlx import VeloxChem

program = {
    'dalton': DaltonFactory,
    'veloxchem': VeloxChem,
}
