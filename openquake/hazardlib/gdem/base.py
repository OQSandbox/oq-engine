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
from openquake.hazardlib.gsim.base import GMPE
from openquake.hazardlib.calc.gmf import GmfComputer


class GSIMFComputer(GmfComputer):
    """
    Adaptation of the `class`:openquake.hazardlib.calc.gmf.GmfComputer
    for the case when multiple GSIMs are input, each associated with only
    a single intensity measure type
    """
    def compute(self, gsim_set, num_events, seed=None):
        """
        :param gsim_set: a dictionary of imts and their required GSIMs 
        :param num_events: the number of seismic events
        :param seed: a random seed or None
        :returns: a 32 bit array of shape (num_imts, num_sites, num_events)
        """
        try:  # read the seed from self.rupture.rupture if possible
            seed = seed or self.rupture.rupture.seed
        except AttributeError:
            pass
        if seed is not None:
            np.random.seed(seed)
        result = np.zeros(
            (len(gsim_set), len(self.sids), num_events), np.float32)
        for imti, (imt, gsim) in enumerate(gsim_set.items()):
            result[imti] = self._compute(None, gsim, num_events,
                                         from_string(imt))
        return result


class GDEM(GMPE):
    """
    General base class for a Ground Deformation Estimation Model (GDEM)
    
    :param dict gmpe_set:
        GMPE sets organised by intensity measure type, i.e.
        {IMT1: GMPE1, IMT2: GMPE2, ...}
    """
    # Overwide the abstract methods in GMPE class
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set(())
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = set(())
    DEFINED_FOR_DEFORMATION_TYPES = set(())
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set(())
    DEFINED_FOR_TECTONIC_REGION_TYPE = set(())
    REQUIRES_DISTANCES = set(())
    REQUIRES_RUPTURE_PARAMETERS = set(())
    REQUIRES_SITES_PARAMETERS = set(())

    def __init__(self, gmpe_set={}, truncation=3.0, nsample=1000):
        super().__init__()
        self.gmpe_set = OrderedDict([])
        self.stddev_types = OrderedDict([])
        if not len(gmpe_set):
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
            self.stddev_types[imt] = []
            for stddev_type in\
                self.gmpe_set[imt].DEFINED_FOR_STANDARD_DEVIATION_TYPES:
                self.stddev_types[imt].append(stddev_type)
        # Get discrete truncated normal probability distribution
        xvals = np.linspace(-truncation, truncation, nsample + 1)
        self.truncnorm_probs = truncnorm.sf(xvals, -truncation, truncation,
                                            loc=0., scale=1.)
        self.truncnorm_probs = self.truncnorm_probs[:-1] -\
            self.truncnorm_probs[1:]
        self.epsilons = (xvals[:-1] + xvals[1:]) / 2.

    @abc.abstractmethod
    def get_probability_failure(self, sctx, rctx, dctx, gsimtls):
        """
        Returns the probability of failure
        """

    @abc.abstractmethod
    def get_poes(self, sctx, rctx, dctx, imtls, truncation_level):
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

    def get_mean_and_stddevs(self, sctx, rctx, dctx):
        """
        Returns a dictionary containing the expected mean and standard
        deviations of ground motion for all of the parameters required by the
        model
        """
        gsimtls = []
        for imt, gmpe in self.gmpe_set.items():
            mean, stddevs = gmpe.get_mean_and_stddevs(sctx, rctx, dctx,
                                                      from_string(imt),
                                                      self.stddev_types[imt])
            gsimls = {"mean": mean}
            for i, stddev_type in enumerate(self.stddev_types[imt]):
                if not stddev_type in\
                    gmpe.DEFINED_FOR_STANDARD_DEVIATION_TYPES:
                    continue
                gsimls[stddev_type] = stddevs[i]
            gsimtls.append((imt, gsimls))
        return OrderedDict(gsimtls)

