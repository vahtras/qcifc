from . import core
from .dalton_factory import DaltonFactory, DaltonFactoryDummy
from .vlx import VeloxChem, VeloxChemDummy

_program = {
    'dummy': DaltonFactoryDummy,
    'dalton': DaltonFactory,
    'veloxchem': VeloxChem,
    'veloxchemdummy': VeloxChemDummy,
}


def program(name, **kwargs):
    _prog = _program[name]()
    _prog.setup(**kwargs)
    return _prog
