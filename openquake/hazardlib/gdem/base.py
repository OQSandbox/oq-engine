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
from collections import OrderedDict
from scipy.stats import truncnorm
from openquake.hazardlib.imt import from_string
from openquake.hazardlib.gsim.multi import MultiGMPE
from openquake.hazardlib import const


class GDEM(MultiGMPE):
    """
    General base class for a Ground Deformation Estimation Model (GDEM)
    
    :param dict gmpe_set:
        GMPE sets organised by intensity measure type, i.e.
        {IMT1: GMPE1, IMT2: GMPE2, ...}
    """
    # Overwrite the abstract methods in GMPE class
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set(())
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = None
    DEFINED_FOR_DEFORMATION_TYPES = set(())
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set((const.StdDev.TOTAL,))
    REQUIRES_DISTANCES = set()
    REQUIRES_RUPTURE_PARAMETERS = set(("mag",))
    REQUIRES_SITES_PARAMETERS = set()
    IMT_ORDER = []

    def __init__(self, gsim_by_imt, truncation=6.0, nsample=100):
        super().__init__(gsim_by_imt)
        self.imts = []
        for imt in self.gsim_by_imt:
            self.REQUIRES_SITES_PARAMETERS = (
                self.REQUIRES_SITES_PARAMETERS |
                self.gsim_by_imt[imt].REQUIRES_SITES_PARAMETERS)
            self.REQUIRES_RUPTURE_PARAMETERS = (
                self.REQUIRES_RUPTURE_PARAMETERS |
                self.gsim_by_imt[imt].REQUIRES_RUPTURE_PARAMETERS)
            self.REQUIRES_DISTANCES = (self.REQUIRES_DISTANCES |
                self.gsim_by_imt[imt].REQUIRES_DISTANCES)
            self.imts.append(imt)
        # Geotechnical hazard requires the definition of an integral of
        # the expected displacement conditional upon the shaking at the surface
        # As this is integrated numerically we can pre-calculate the integral
        # bins according to the specific level of truncation and the number
        # of samples. In the geotechnical modules the bins and their
        # probabilities are integrated upon.
        xvals = np.linspace(-truncation, truncation, nsample + 1)
        self.truncnorm_probs = truncnorm.sf(xvals, -truncation, truncation,
                                            loc=0., scale=1.)
        self.truncnorm_probs = self.truncnorm_probs[:-1] -\
            self.truncnorm_probs[1:]
        self.epsilons = (xvals[:-1] + xvals[1:]) / 2.

    @abc.abstractmethod
    def get_probability_failure(self, sctx, rctx, dctx):
        """
        Returns the probability of failure
        """

    @abc.abstractmethod
    def get_poes(self, sctx, rctx, dctx, imt, imtls, truncation_level):
        """
        Returns the probabilities of exceeding the given level of ground
        motion
        """

    @abc.abstractmethod
    def get_displacement_field(self, rupture, sitecol, cmaker, num_events=1,
                               truncation_level=None, correlation_model=None):
        """
        Returns the field of displacements
        """

    def get_shaking_mean_and_stddevs(self, sctx, rctx, dctx):
        """
        Returns a dictionary containing the expected mean and standard
        deviations of ground motion for all of the parameters required by the
        model
        """
        gsimtls = []
        stddev_types = list(self.DEFINED_FOR_STANDARD_DEVIATION_TYPES)
        for imt in self.gsim_by_imt:
            mean, stddevs = self.get_mean_and_stddevs(sctx, rctx, dctx, 
                                                      from_string(imt),
                                                      stddev_types)
            gsimls = {"mean": mean}
            for i, stddev_type in enumerate(stddev_types):
                if not stddev_type in\
                    self.gsim_by_imt[imt].DEFINED_FOR_STANDARD_DEVIATION_TYPES:
                    continue
                gsimls[stddev_type] = stddevs[i]
            gsimtls.append((str(imt), gsimls))
        return OrderedDict(gsimtls)
