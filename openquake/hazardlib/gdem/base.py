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
Base module for Ground Deformation Estimation Models
"""

import abc
import numpy as np
from openquake.hazardlib.gsim.base import GMPE

class GDEM(GMPE):
    """
    General base class for a Ground Deformation Estimation Model (GDEM)
    
    :param dict gmpe_set:
        GMPE sets organised by intensity measure type, i.e.
        {IMT1: GMPE1, IMT2: GMPE2, ...}
    """
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set((PGDf,))
    def __init__(self, gmpe_set={}):
        super().__init__()
        self.gmpe_set = {}
        if not len(gmpe):
            raise ValueError("Dictionary of Ground Motion Models Must be Defined")
        for imt in gmpe_set:
            self.gmpe_set[imt] = gmpe_set[imt]()
            self.DEFINED_FOR_INTENSITY_MEASURE_TYPES = \
                self.DEFINED_FOR_INTENSITY_MEASURE_TYPES.union(
                    self.gmpe_set[imt].DEFINED_FOR_INTENSITY_MEASURE_TYPES)
            self.REQUIRES_DISTANCES = self.REQUIRES_DISTANCES.union(
                self.gmpe_set[imt].REQUIRES_DISTANCES)
            self.REQUIRES_RUPTURE_PARAMETERS = self.REQUIRES_RUPTURE_PARAMETERS.union(
                self.gmpe_set[imt].REQUIRES_RUPTURE_PARAMETERS)
            self.REQUIRES_SITES_PARAMETERS = self.REQUIRES_SITES_PARAMETERS.union(
                self.gmpe_set[imt].REQUIRES_SITES_PARAMETERS)
        
    #def _update_required_parameters(self, old, new)
    
    abc.abstractmethod
    def get_poes(sctx, rctx, dctx, iml, truncation_level):
        """
        Returns the probability of exceeding a given level of ground
        deformation 
        """
    
    abc.abstractmethod
    def get_probability_failure(self, sctx, rctx, dctx, iml):
        """
        Determines the probability of occurrence of geotechnical failure
        """

