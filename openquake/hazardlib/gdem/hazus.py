# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2012-2017 GEM Foundation
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

"""

"""

import numpy as np


# Limits of Vs for each susceptibility category - where does this come from
SUS_LIMS = [(620., np.inf),
            (500., 620.),
            (360., 500.),
            (300., 360.),
            (180., 300.),
            (-np.inf, 180.)]

M_PER_INCH = 0.0254

def _get_liquefaction_susceptibility_category(vs):
    """
    Returns the susceptibility category corresponding to the Vs at the site
    """
    sus = np.zeros(vs.shape, dtype=int)
    for i, (low, high) in enumerate(SUS_LIMS):
        idx = np.logical_and(vs >= low, vs < high)
        if np.any(idx):
            sus[idx] = i
    return sus

# Probability of liquefaction given PGA = a (Table 4.12)
P_PGA_LIQ = {
    0: lambda pga: 0.0,
    1: lambda pga: ((4.16 * pga) - 1.08),
    2: lambda pga: ((5.57 * pga) - 1.18),
    3: lambda pga: ((6.67 * pga) - 1.0),
    4: lambda pga: ((7.67 * pga) - 0.92),
    5: lambda pga: ((9.09 * pga) - 0.82),
}

# Proportion of map unit susceptible to liquefaction (Table 4.11)
PMU_LIQ = {0: 0.0, 1: 0.02, 2: 0.05, 3: 0.1, 4: 0.2, 5: 0.25}

# PGA / PGA_Threshold (PGA_Threshold taken from Table 4.13)
PGA_T_RATIO_LIQ = {
    0: 0.0, 
    1: lambda pga: pga / 0.26,
    2: lambda pga: pga / 0.21,
    3: lambda pga: pga / 0.15,
    4: lambda pga: pga / 0.12,
    5: lambda pga: pga / 0.09
}

# Settlement displacements (Table 4.14)
D_SETTLEMENT = {
    0: 0.0,
    1: 0.0,
    2: 1.0 * M_PER_INCH,
    3: 2.0 * M_PER_INCH,
    4: 6.0 * M_PER_INCH,
    5: 12.0 * M_PER_INCH,
}

# List of tuples of lateral spread bounds and formulae
# ((lower limit, upper limit), (m, c))
# where lower limit and upper limit are the bounding limits of the
# piecewise linear segment in terms of PGA / PGA_T, and m and c
# are the segment model in terms of D = m * (PGA / PGA_T) + c
D_LATERAL_SPREAD = [((1., 2.), (12., -12.)),
                    ((2., 3.), (18., -24.)),
                    # Currently assuming this extrapolates into > 4.
                    # - should this be capped?
                    ((3., np.inf), (70., -80.))]

def get_lateral_spread(pga, sus):
    """
    Returns the displacement due to lateral spreading
    for the given PGA and susceptibility category
    """
    pga_pgat = np.zeros_like(pga)
    disp = np.zeros_like(pga)
    for i in range(1, 6):
        idx = sus == i
        if np.any(idx):
            pga_pgat[idx] = PGA_T_RATIO_LIQ(pga[idx])
    for (low, high), (m, c) in D_LATERAL_SPREAD:
        idx = np.logical_and(pga_pgat >= low,
                             pga_pgat < high)
        if np.any(idx):
            disp[idx] = (m * pga_pgat[idx]) + c
    # Return displacements in m
    return disp * M_PER_INCH
