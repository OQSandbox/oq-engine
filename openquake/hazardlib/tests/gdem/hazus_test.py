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
import unittest
import numpy as np
from openquake.hazardlib.gdem.hazus import *


class PeripheralFunctionsTestCase(unittest.TestCase):
    """
    """
    def test_get_probability_liquefaction(self):
        """
        """

        categories = np.array([0., 1., 2., 3., 4., 5.])

        # 0.05 g
        pgas = 0.05 * np.ones(6)
        pliq = get_probability_liquefaction(pgas, categories)
        np.testing.assert_array_almost_equal(pliq, np.zeros(pliq.shape))
        # 0.1 g
        pgas = 0.1 * np.ones(6)
        pliq = get_probability_liquefaction(pgas, categories)
        np.testing.assert_array_almost_equal(
            pliq, np.array([0., 0., 0., 0., 0., 0.089]))
        # 0.2 g
        pgas = 0.2 * np.ones(6)
        pliq = get_probability_liquefaction(pgas, categories)
        np.testing.assert_array_almost_equal(
            pliq, np.array([0., 0., 0., 0.334, 0.614, 0.998]))
        # 0.5 g
        pgas = 0.2 * np.ones(6)
        pliq = get_probability_liquefaction(pgas, categories)
        np.testing.assert_array_almost_equal(
        pliq, np.array([0., 1., 1., 1., 1., 1.]))

    def test_get_pga_t_ratio(self):

        categories = np.array([0., 1., 2., 3., 4., 5.])
        # 0.05 g
        pgas = 0.05 * np.ones(6)
        pga_ratio = get_pga_t_ratio(pgas, categories)
        np.testing.assert_array_almost_equal(pga_ratio,
            np.array([0., 0.19230769, 0.23809524,
                      0.33333333, 0.41666667, 0.55555556]))
        # 0.1 g
        pgas = 0.1 * np.ones(6)
        pga_ratio = get_pga_t_ratio(pgas, categories)
        np.testing.assert_array_almost_equal(pga_ratio,
            np.array([0., 0.38461538, 0.47619048,
                      0.66666667, 0.83333333, 1.11111111]))
        # 0.2 g
        pgas = 0.2 * np.ones(6)
        pga_ratio = get_pga_t_ratio(pgas, categories)
        np.testing.assert_array_almost_equal(pga_ratio,
            np.array([0., 0.76923077, 0.95238095,
                      1.33333333, 1.66666667, 2.22222222]))
        # 0.5 g
        pgas = 0.2 * np.ones(6)
        pga_ratio = get_pga_t_ratio(pgas, categories)
        np.testing.assert_array_almost_equal(pga_ratio,
            np.array([0., 1.92307692, 2.38095238,
                      3.33333333, 4.16666667, 5.55555556]))


