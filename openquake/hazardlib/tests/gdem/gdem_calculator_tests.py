# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2015-2018 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.
import os
import unittest
import numpy as np
from openquake.hazardlib.gdem.hazus import (HAZUSLiquefaction,
                                            HAZUSLandsliding,
                                            HAZUSLandslidingKy)
from openquake.hazardlib.gdem.zhu_2015 import ZhuEtAl2015Global
from openquake.hazardlib.gdem.scalar import (FotopoulouPitilakis2015PGA,
                                             FotopoulouPitilakis2015PGV,
                                             FotopoulouPitilakis2015Ky,
                                             Jibson2007Ia, Jibson2007PGA,
                                             Jibson2007PGAMag,
                                             RathjeSaygili2009PGA)
from openquake.hazardlib.imt import PGA, PGV, IA
from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014
from openquake.hazardlib.gsim.travasarou_2003 import TravasarouEtAl2003
from openquake.hazardlib.calc.hazard_curve import classical
from openquake.hazardlib import valid
from openquake.hazardlib.calc.filters import SourceFilter
from openquake.baselib.general import DictArray, groupby, AccumDict
from openquake.hazardlib.probability_map import ProbabilityMap
import openquake.hazardlib.tests.gdem.gdem_test_utils as gtu

LIQ_IMTLS = {"PGDfLatSpread": [1.0E-15, 0.001, 0.1, 1., 5., 10.],
             "PGDfSettle": [1.0E-15, 0.001, 0.1, 1., 5., 10.]}

LS_IMTLS = {"PGDfSlope": [1.0E-15, 0.001, 0.1, 1., 5., 10.]}

BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

class GDEMClassicalTestCases(unittest.TestCase):
    """
    Test cases for each of the geotechnical calculators - verifies
    execution and generates reproducible curves
    """
    def setUp(self):
        self.source = [gtu._setup_fault_source()]

    def _run_calculator(self, gdem, imtls, sites):
        """
        Executes the classical calculator
        """
        param = {"imtls": DictArray(imtls),
                 "truncation_level": 3.,
                 "filter_distance": "rrup"}
        curves = classical(self.source,
                           SourceFilter(sites, valid.floatdict("200")),
                           [gdem], param)
        pmap = ProbabilityMap(param["imtls"].array, 1)
        for res in [curves]:
            for grp_id in res:
                pmap |= res[grp_id]
        return pmap.convert(param["imtls"], len(sites))

    def _check_curves(self, results, test_file, precision=5):
        """
        Verifies the curves agree with those in the specific file
        """
        test_file = os.path.join(BASE_DATA_PATH, test_file)
        expected = np.genfromtxt(test_file, delimiter=",")
        for i in range(results.shape[0]):
            print(results[i, :], expected[i, :])
        np.testing.assert_array_almost_equal(results, expected, precision)

    def test_hazus_liquefaction_classical(self):
        sites = gtu._setup_sites_liquefaction()
        gdem = HAZUSLiquefaction(gsim_by_imt={"PGA": BooreEtAl2014()},
                                 truncation=6., nsample=100)
        curves = self._run_calculator(gdem, LIQ_IMTLS, sites)
        self._check_curves(curves["PGDfLatSpread"],
                           "hazus_liquefaction_lateral_spread.csv")
        self._check_curves(curves["PGDfSettle"],
                           "hazus_liquefaction_settlement.csv")

    def test_zhu_2015_global(self):
        sites = gtu._setup_sites_liquefaction_zhu()
        gdem = ZhuEtAl2015Global(gsim_by_imt={"PGA": BooreEtAl2014()},
                                 truncation=6., nsample=100)
        curves = self._run_calculator(gdem, LIQ_IMTLS, sites)
        self._check_curves(curves["PGDfLatSpread"],
                           "zhu_liquefaction_lateral_spread.csv")
        self._check_curves(curves["PGDfSettle"],
                           "zhu_liquefaction_settlement.csv")

    def test_hazus_landsliding(self):
        sites = gtu._setup_sites_landsliding()
        gdem = HAZUSLandsliding(gsim_by_imt={"PGA": BooreEtAl2014()},
                                truncation=6., nsample=100)
        curves = self._run_calculator(gdem, LS_IMTLS, sites)
        self._check_curves(curves["PGDfSlope"], "hazus_landsliding.csv")

    def test_hazus_landsliding_ky(self):
        sites = gtu._setup_sites_yield_accel()
        gdem = HAZUSLandslidingKy(gsim_by_imt={"PGA": BooreEtAl2014()},
                                  truncation=6., nsample=100)
        curves = self._run_calculator(gdem, LS_IMTLS, sites)
        self._check_curves(curves["PGDfSlope"], "hazus_landsliding_ky.csv")

    def test_fotopoulou_pitilakis_2015_pga(self):
        sites = gtu._setup_sites_yield_accel()
        gdem = FotopoulouPitilakis2015PGA(gsim_by_imt={"PGA": BooreEtAl2014()},
                                          truncation=6., nsample=100)
        curves = self._run_calculator(gdem, LS_IMTLS, sites)
        self._check_curves(curves["PGDfSlope"], "landsliding_fp15_pga.csv")

    def test_fotopoulou_pitilakis_2015_ky(self):
        sites = gtu._setup_sites_yield_accel()
        gdem = FotopoulouPitilakis2015Ky(gsim_by_imt={"PGA": BooreEtAl2014()},
                                          truncation=6., nsample=100)
        curves = self._run_calculator(gdem, LS_IMTLS, sites)
        self._check_curves(curves["PGDfSlope"], "landsliding_fp15_ky.csv")

    def test_fotopoulou_pitilakis_2015_pgv(self):
        sites = gtu._setup_sites_yield_accel()
        gdem = FotopoulouPitilakis2015PGV(gsim_by_imt={"PGV": BooreEtAl2014()},
                                          truncation=6., nsample=100)
        curves = self._run_calculator(gdem, LS_IMTLS, sites)
        self._check_curves(curves["PGDfSlope"], "landsliding_fp15_pgv.csv")

    def test_jibson_2007_pga(self):
        sites = gtu._setup_sites_yield_accel()
        gdem = Jibson2007PGA(gsim_by_imt={"PGA": BooreEtAl2014()},
                             truncation=6., nsample=100)
        curves = self._run_calculator(gdem, LS_IMTLS, sites)
        self._check_curves(curves["PGDfSlope"],
                           "landsliding_jibson2007_pga.csv")

    def test_jibson_2007_pga_mag(self):
        sites = gtu._setup_sites_yield_accel()
        gdem = Jibson2007PGAMag(gsim_by_imt={"PGA": BooreEtAl2014()},
                                          truncation=6., nsample=100)
        curves = self._run_calculator(gdem, LS_IMTLS, sites)
        self._check_curves(curves["PGDfSlope"],
                           "landsliding_jibson2007_pga_mag.csv")

    def test_jibson_2007_ia(self):
        sites = gtu._setup_sites_yield_accel()
        gdem = Jibson2007Ia(gsim_by_imt={"IA": TravasarouEtAl2003()},
                                          truncation=6., nsample=100)
        curves = self._run_calculator(gdem, LS_IMTLS, sites)
        self._check_curves(curves["PGDfSlope"],
                           "landsliding_jibson2007_ia.csv")

    def test_rathje_saygili_2009_pga(self):
        sites = gtu._setup_sites_yield_accel()
        gdem = RathjeSaygili2009PGA(gsim_by_imt={"PGA": BooreEtAl2014()},
                                    truncation=6., nsample=100)
        curves = self._run_calculator(gdem, LS_IMTLS, sites)
        self._check_curves(curves["PGDfSlope"],
                           "landsliding_rathje_saygili09_pga.csv")
