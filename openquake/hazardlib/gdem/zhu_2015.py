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
Liquefaction Hazard Assessment using the formulation of Zhu et al (2015) for
the probability of failure, with HAZUS estimates of settlement and spreading
"""

import numpy as np
from openquake.hazardlib.gdem.hazus import HAZUSLiquefaction
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGDfSettle, PGDfLatSpread, PGA


pgdf_settle = PGDfSettle()
pgdf_spread = PGDfLatSpread()


class ZhuEtAl2015Global(HAZUSLiquefaction):
    """
    Implements an adaptation of the HAZUS Liquefaction methodology in which
    the probability of liquefaction is determined via the empirical model of
    Zhu et al. (2015)

    Zhu, J., Daley, D., Baise, L. G., Thompson, E., Wald, D. J. and
    Knudsen, K. L. (2015) "A Geospatial Liquefaction Model for Rapid Response
    and Loss Estimation". Earthquake Spectra, 31(3): 1813 - 1837

    Implemented herein is the "Global" model (from pooled data from the 1995
    Kobe and 2011 Christchurch earthquakes), which is dependent upon
    the ground shaking (PGA), Vs30 and the compound topographic index (CTI)

    The Zhu et al. (2015) methods describe only the probability of failure.
    Therefore it remains necessary to adopt the HAZUS approach to describe
    the expected lateral spread and settlement; hence the HAZUS liquefaction
    susceptibility category is still required for this methodology.
    """
    DEFINED_FOR_DEFORMATION_TYPES = set((PGDfSettle, PGDfLatSpread))
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set((PGA,))

    # Needs all HAZUS attributes (liquefaction susceptibility and dw),
    # as well as vs30 and compound topographic index (CTI)
    REQUIRES_SITES_PARAMETERS = set(("liquefaction_susceptibility",
                                     "dw", "cti", "vs30"))

    def get_failure_model(self, sctx, gmv, properties):
        """
        Returns the probability of failure given the ground motion values and
        the site properties from Table 4 ("Global model") and equation 2.
        """
        model = 24.10 + 2.067 * np.log(gmv) + 0.355 * sctx.cti +\
            -4.784 * np.log(sctx.vs30)
        return 1. / (1.0 + np.exp(model))
