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


import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
from collections import OrderedDict
from openquake.hazardlib.gdem.base import GDEM
from openquake.hazardlib import const
from openquake.hazardlib.imt import (PGDfSettle, PGDfLatSpread, PGDfSlope,
                                     PGA, PGV, IA)


# Some models return displacement in cm - need to convert to m
CM2M = -np.log(100.0)


class SlopeDisplacementScalar(GDEM):
    """
    Common class for all forms of slipe displacement models that are
    conditioned upon a single intensity measure (or on a single IM plus
    a property of the rupture or site)

    In this case the mean and standard deviation of the ground motion are
    numerically integrated over, with each IML then used as the conditioning
    value of the empirical displacement model. The probability of exceeding
    a specific level of displacement conditional upon the shaking IML is
    determined from the mean and standard deviation of the empirical
    displacement model
    """
    DEFINED_FOR_DEFORMATION_TYPES = set((PGDfSlope,))
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set((PGA,))
    REQUIRES_SITES_PARAMETERS = set(("yield_acceleration", "vs30"))
    IMT_ORDER = []

    def _setup_properties(self, sctx, rctx):
        """
        For the site category returns the critical accelerations and
        landsliding area
        """
      
        k_y = np.copy(sctx.yield_acceleration)
        # If yield acceleration is less than or equal to zero then assume
        # no displacement is possible
        k_y[k_y <= 0.0] = np.inf
        return {"a_c": k_y}

    def _get_failure_gmvs(self, gsimtls):
        """
        Retrieve from gsimtls the ground motion values needed for the
        failure probability
        In the HAZUS case this is only PGA
        """
        means = []
        stddevs = []
        for imt in self.IMT_ORDER:
            means.append(gsimtls[imt]["mean"])
            stddevs.append(gsimtls[imt][const.StdDev.TOTAL])
        return means, stddevs

    def get_probability_failure(self, sctx, rctx, dctx):
        """
        Returns the probability of failure - in this case any ground motions
        exceeding yield acceleration have a probability of failure of 1.0
        """
        gsimtls = self.get_shaking_mean_and_stddevs(sctx, rctx, dctx)
        properties = self._setup_properties(sctx, rctx)
        gmv_mean, gmv_sigma = self._get_failure_gmvs(gsimtls)
        p_failure = np.zeros_like(properties["a_c"])
        for j, epsilon in enumerate(self.epsilons):
            # Get the PGA corresponding to a given epislon
            gmv = np.exp(gmv_mean[0] + epsilon * gmv_sigma[0])
            idx = gmv >= properties["a_c"]
            # Where PGA exceeds the yield acceleration multiply the
            # epsilon probability by the map area
            if np.any(idx):
                p_failure[idx] += self.truncnorm_probs[j]
        return p_failure

    def get_poes(self, sctx, rctx, dctx, imtls, truncation_level):
        """
        Returns the probabilities of exceeding the given level of ground
        motion
        """
        gsimtls = self.get_shaking_mean_and_stddevs(sctx, rctx, dctx)
        properties = self._setup_properties(sctx, rctx)
        gmv_mean, gmv_sigma = self._get_failure_gmvs(gsimtls)
        poes = np.zeros([len(properties["a_c"]),
                         len(imtls["PGDfSlope"])])
        for j, epsilon in enumerate(self.epsilons):
            gmv = np.exp(gmv_mean[0] + epsilon * gmv_sigma[0])
            # Get probability of failure
            idx = gmv >= properties["a_c"]
            if not np.any(idx):
                # Nothing exceeding critical acceleration
                continue
            # Get displacement
            mean_disp, [stddevs] = self.get_displacement(sctx, rctx, dctx,
                                                         gmv, properties, idx)
            for i, iml in enumerate(imtls["PGDfSlope"]):
                poes[idx, i] += (
                    norm.sf(np.log(iml), loc=mean_disp, scale=stddevs) * 
                    self.truncnorm_probs[j])
        return [poes]

    def get_displacement(self, sctx, rctx, dctx, gmv, properties, idx):
        """
        Returns the expected displacement and standard deviation of
        displacement
        """
        raise NotImplementedError("Not implemented for base class")

    def _get_gmv_field_location(self):
        """
        Get the ground motion values needed for the field.
        """
        req_imt = self.IMT_ORDER[0]
        if not req_imt in self.imts:
            raise ValueError("%s requires calculation of %s "
                             "but not found in imts" %
                             (self.__class__.__name__, self.IMT_ORDER[0]))

        return [self.imts.index(req_imt)]

    def get_displacement_field(self, rupture, sitecol, cmaker, num_events=1,
                               truncation_level=None, correlation_model=None):
        """
        Returns a field of expected displacements and the accompanying
        ground motions. Samples a GMF initially and for those sites for which
        the GMF exceeds the yield acceleration then the displacement is
        sampled from the empricial model
        """
        # Gets the ground motion fields
        gmf_loc = self._get_gmv_field_location()
        gmf_computer = GmfComputer(rupture, sitecol,
                                   [str(imt) for IMT in self.IMT_ORDER],
                                   cmaker, truncation_level,
                                   correlation_model)
        gmfs = gmf_computer.compute(self.gmpe, num_events, seed=None)
        # Get the ground motion values field
        gmvs = gmfs[gmf_loc[0]]
        properties = self._setup_properties(gmf_computer.ctx[0],
                                            gmf_computer.ctx[1])
        
        n = properties["a_c"].shape
        properties["a_c"] = np.tile(
            np.reshape(properties["a_c"], [n, 1]), [1, num_events]) 
        mask = gmv >= properties["a_c"]
        displacement = np.zeros([1, len(gmf_computer.sctx.sids), num_events],
                                 dtype=np.float32)
        if not np.any(mask):
            # No displacement at any site - return zeros and the ground motion
            # fields
            return displacement, gmfs, mask.astype(float)

        mean_disp, [stddevs] = self.get_displacement(sctx, rctx, dctx, gmv,
                                                     properties, mask)
        displacement[0][mask] = np.exp(mean_disp +
                                       np.random.normal(0., 1.) * stddevs)
        return displacement, gmfs, mask.astype(float)
        

class Jibson2007PGA(SlopeDisplacementScalar):
    """
    Implements the scalar empirical slope displacement model of Jibson (2007)
    for the case that only the peak ground acceleration (PGA) is required.

    Jibson R.W. (2007) "Regression models for estimating coseismic landslide
    displacement", Engineering Geology 91: 209 - 218
    """
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set((PGA,))
    IMT_ORDER = ["PGA",]
    
    def get_displacement(self, sctx, rctx, dctx, gmv, properties, idx):
        """
        Returns the expected displacement and standard deviation of Jibson
        (2007) equation (6)
        """
        ac_ratio = properties["a_c"][idx] / gmv[idx]
        mean = 0.215 + np.log(((1.0 - ac_ratio) ** 2.341) *
                              (ac_ratio ** -1.438))
        stddevs = 0.51 * np.ones_like(ac_ratio)
        # Convert mean from cm to m
        return mean + CM2M, [stddevs]


class Jibson2007PGAMag(SlopeDisplacementScalar):
    """
    Implements the scalar empirical slope displacement model of Jibson (2007)
    for the case that both peak ground acceleration (PGA) and magnitude are
    required.
    """
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set((PGA,))
    IMT_ORDER = ["PGA",]

    def get_displacement(self, sctx, rctx, dctx, gmv, properties, idx):
        """
        Returns the expected displacement and standard deviation of Jibson
        (2007) equation (7)
        """
        ac_ratio = properties["a_c"][idx] / gmv[idx]
        mean = -2.710 + np.log(((1.0 - ac_ratio) ** 2.335) *
                              (ac_ratio ** -1.478)) + 0.424 * rctx.mag
        stddevs = 0.454 * np.ones_like(ac_ratio)
        # Convert mean from cm to m
        return mean + CM2M, [stddevs]


class Jibson2007Ia(SlopeDisplacementScalar):
    """
    Implements the scalar empirical slope displacement model of Jibson (2007)
    for the case that only the Arias Intensity (IA) is required.
    """
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set((IA,))
    IMT_ORDER = ["IA",]

    def get_displacement(self, sctx, rctx, dctx, gmv, properties, idx):
        """
        Returns the expected displacement and standard deviation of Jibson
        (2007) equation (9)
        """
        mean = 2.401 * np.log(gmv[idx]) -\
            3.481 * np.log(properties["a_c"][idx]) - 3.230
        stddevs = 0.616 * np.ones_like(properties["a_c"][idx])
        # Convert mean from cm to m
        return mean + CM2M, [stddevs]


class FotopoulouPitilakis2015PGV(SlopeDisplacementScalar):
    """
    Implements the scalar form of the empirical predictive relations for
    seismically induced slope displacement of Fotopoulou & Pitilakis (2015)
    for the case that only PGV and yield acceleration is required.

    Fotopoulou S. D. & Pitilakis K. D. (2015) "Predictive relationships for
    seismically induced slope displacements using numerical analysis results".
    Bulletin of Earthquake Engineering, 13: 3207 - 3238
    """
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set((PGV,))
    IMT_ORDER = ["PGV",]

    def get_displacement(self, sctx, rctx, dctx, gmv, properties, idx):
        """
        Returns the expected displacement and standard deviation of Fotopoulou
        & Pitilakis (2015) equation 8
        """
        a_c = properties["a_c"][idx]
        mean = -9.891 + 1.873 * np.log(gmv[idx]) - 5.964 * a_c +\
            0.285 * rctx.mag
        stddevs = 0.65 * np.ones_like(a_c)
        return mean, [stddevs]


class FotopoulouPitilakis2015PGA(SlopeDisplacementScalar):
    """
    Implements the scalar form of the empirical predictive relations for
    seismically induced slope displacement of Fotopoulou & Pitilakis (2015)
    for the case that only PGA is required.
    """
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set((PGA,))
    IMT_ORDER = ["PGA",]

    def get_displacement(self, sctx, rctx, dctx, gmv, properties, idx):
        """
        Returns the expected displacement and standard deviation of Fotopoulou
        & Pitilakis (2015) equation 9
        """
        a_c = properties["a_c"][idx]
        mean = -2.965 + 2.127 * np.log(gmv[idx]) - 6.583 * a_c +\
            0.535 * rctx.mag
        stddevs = 0.72 * np.ones_like(a_c)
        return mean, [stddevs]


class FotopoulouPitilakis2015Ky(SlopeDisplacementScalar):
    """
    Implements the scalar form of the empirical predictive relations for
    seismically induced slope displacement of Fotopoulou & Pitilakis (2015)
    for the case that only PGA and the ratio of yield acceleration to PGA is
    required
    """
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set((PGA,))
    IMT_ORDER = ["PGA",]

    def get_displacement(self, sctx, rctx, dctx, gmv, properties, idx):
        """
        Returns the expected displacement and standard deviation of Fotopoulou
        & Pitilakis (2015) equation 10
        """
        a_c = properties["a_c"][idx]
        mean = -10.246 - 2.165 * np.log(a_c / gmv[idx]) + 7.844 * a_c +\
            0.654 * rctx.mag
        stddevs = 0.75 * np.ones_like(a_c)
        return mean, [stddevs]


class RathjeSaygili2009PGA(SlopeDisplacementScalar):
    """
    Implements the scalar empirical predictive model for earthquake-induced
    sliding displacements presented by Rathje & Saygili (2009) dependent
    only upon PGA

    Rathje, E. M. and Saygili, G. (2009) Probabilistic Assessment of
    Earthquake-Induced Sliding Displacements of Natural Slopes, Bulletin of the
    New Zealand Society for Earthquake Engineering, 42(1): 18 - 27
    """
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set((PGA,))
    IMT_ORDER = ["PGA",]

    def get_displacement(self, sctx, rctx, dctx, gmv, properties, idx):
        """
        Returns the expected displacement and standard deviation of Rathje &
        Saygili (2009)
        """

        ac_ratio = properties["a_c"][idx] / gmv[idx]
        mean = 4.89 - 4.85 * ac_ratio - 19.64 * (ac_ratio ** 2.) +\
            42.49 * (ac_ratio ** 3.) - 29.06 * (ac_ratio ** 4.) +\
            0.72 * np.log(gmv[idx]) + 0.89 * (rctx.mag - 6.0)
    
        stddevs = 0.732 + 0.789 * ac_ratio - 0.539 * (ac_ratio ** 2.)
        # Convert mean from cm to m
        return mean + CM2M, [stddevs]
