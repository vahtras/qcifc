import os
import sys
import pathlib
import itertools

import pytest

sys.path.insert(0, os.path.abspath('..'))
from qcifc.dalton_factory import (
    # DaltonFactoryDummy,
    DaltonFactory,
)

from qcifc.vlx import (
    VeloxChem,
    # VeloxChemDummy,
)

codes = {
    # 'dummy': DaltonFactoryDummy,
    'dalton': DaltonFactory,
    # 'vlx': VeloxChem,
    # 'vlxdummy': VeloxChemDummy,
}


class TestQC:
    @staticmethod
    def skip_if_not_implemented(method, code):
        if method not in dir(code):
            pytest.skip('not implemented')

    @staticmethod
    def skip_open_shell(code):
        if isinstance(code, codes['vlx']):
            pytest.skip('open shell not implemented')

    @staticmethod
    def skip_if_dummy(code):
        if 'dummy' in str(code).lower():
            pytest.skip('skip dummy version')


def get_settings(case):
    test_root = pathlib.Path(__file__).parent
    test_dir = test_root/f'test_{case}.d'
    settings = [dict(
        case=case,
        xyz=test_dir/f'{case}.xyz',
        inp=test_dir/f'{case}.inp',
        out=test_dir/f'{case}.out',
        basis='STO-3G',
        _tmpdir=test_dir,
    )]
    return settings


def get_codes_settings(case):
    codes_settings = list(
        itertools.product(
            codes.values(), get_settings(case)
        )
    )
    return codes_settings


def get_codes_ids():
    return list(codes.keys())
