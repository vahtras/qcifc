import sys
import pytest
import numpy as np
import pandas as pd
import numpy.testing as npt

from qcifc.core import OutputStream

from . import TestQC, get_codes_settings, get_codes_ids

CASE = '3h2'

codes_settings = get_codes_settings(CASE)
ids = get_codes_ids()


@pytest.mark.parametrize('code', codes_settings, indirect=True, ids=ids)
class TestSCF3H2(TestQC):

    def test_first_roothan_rohf(self, code):
        code.set_roothan_iterator(
            'h2',
            electrons=2,
            max_iterations=10,
            threshold=1e-5,
            tmpdir=code.get_workdir(),
            ms=1,
        )
        it = iter(code.roothan)

        assert it.na == 2
        assert it.nb == 0

        assert it.occa == [0, 1]
        assert it.occb == []

        assert it.ms == 1

        initial_energy, initial_norm = next(it)
        assert initial_energy == pytest.approx(-0.530773372830)

    @pytest.mark.skip
    def test_roothan_uhf(self, code):
        code.set_uroothan_iterator(
            'h2',
            electrons=2,
            max_iterations=10,
            threshold=1e-5,
            tmpdir=code.get_workdir(),
            ms=1,
        )
        final_energy, final_norm = code.run_uroothan_iterations()
        assert final_norm < 1e-5
        assert final_energy == pytest.approx(-0.530773372830)

    def test_diis_rohf(self, code):
        final_energy, final_norm = code.run_diis_iterations(
            'h2',
            electrons=2,
            max_iterations=10,
            threshold=1e-5,
            tmpdir=code.get_workdir(),
            ms=1,
        )
        assert final_norm < 1e-5
        assert final_energy == pytest.approx(-0.530773372830)

    @pytest.mark.skip()
    def test_diis_uhf(self, code):
        final_energy, final_norm = code.run_udiis_iterations(
            'h2',
            electrons=1,
            max_iterations=10,
            threshold=1e-5,
            tmpdir=code.get_workdir(),
            ms=1/2,
        )
        assert final_norm < 1e-5
        assert final_energy == pytest.approx(-0.5382054446)
