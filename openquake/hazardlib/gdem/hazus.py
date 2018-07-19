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
Liquefaction and Landsliding Hazard Assessment using the formulation of
HAZUS
"""

import numpy as np
from scipy.interpolate import interp1d
from collections import OrderedDict
from openquake.hazardlib.gdem.base import GDEM # GSIMFComputer
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGDfSettle, PGDfLatSpread, PGDfSlope, PGA
from openquake.hazardlib.calc.gmf import GmfComputer


# Limits of Vs for each susceptibility category - where does this come from
SUS_LIMS = [(620., np.inf),
            (500., 620.),
            (360., 500.),
            (300., 360.),
            (180., 300.),
            (-np.inf, 180.)]


# HAZUS gives displacements in inches - convert them to m
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


def get_probability_liquefaction(pga, category):
    """
    Returns the probability of liquefaction, within the range 0 <= P(Liq) <= 1
    """
    pliq = np.zeros_like(pga)
    for i in range(1, 6):
        idx = category == i
        pliq[idx] = P_PGA_LIQ[i](pga[idx])
    pliq[pliq < 0.] = 0.
    pliq[pliq > 1.] = 1.
    return pliq


# Proportion of map unit susceptible to liquefaction (Table 4.11)
PMU_LIQ = {0: 0.0, 1: 0.02, 2: 0.05, 3: 0.1, 4: 0.2, 5: 0.25}


# PGA / PGA_Threshold (PGA_Threshold taken from Table 4.13)
PGA_THRESHOLD_CLASS = {0: 0.0, 1: 0.26, 2: 0.21, 3: 0.15, 4: 0.12, 5: 0.09}

PGA_T_RATIO_LIQ = {
    0: lambda pga: 0.0, 
    1: lambda pga: pga / 0.26,
    2: lambda pga: pga / 0.21,
    3: lambda pga: pga / 0.15,
    4: lambda pga: pga / 0.12,
    5: lambda pga: pga / 0.09
}

def get_pga_t_ratio(pga, category):
    """
    Get PGA / PGA Threshold for each site clas
    """
    pga_t_ratio = np.zeros_like(pga)
    for i in range(1, 6):
        idx = category == i
        if np.any(idx):
            pga_t_ratio[idx] = PGA_T_RATIO_LIQ[i](pga[idx])
    return pga_t_ratio


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
                    ((3., np.inf), (70., -180.))]


pgdf_settle = PGDfSettle()
pgdf_latspread = PGDfLatSpread()
pgdf_slope = PGDfSlope()


class HAZUSLiquefaction(GDEM):
    """
    Implements a probabilistic version of the HAZUS Liquefaction calculator
    as described in Chapter 4.2.2.1.4 of the HAZUS Technical Manual
    """
    DEFINED_FOR_DEFORMATION_TYPES = set((PGDfSettle, PGDfLatSpread))
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set((PGA,))
    REQUIRES_SITES_PARAMETERS = set(("liquefaction_susceptibility",
                                     "dw", "vs30"))
    
    def get_probability_failure(self, sctx, rctx, dctx):
        """
        Returns the probability of failure
        """
        gsimtls = self.get_shaking_mean_and_stddevs(sctx, rctx, dctx)
            
        # Get properties that are dependent on rupture and site
        properties = self._setup_properties(sctx, rctx)
        # Get the mean and standard deviation of the ground motion values
        gmv_mean, gmv_sigma = self._get_failure_gmvs(gsimtls)
        # For each epsilon value determine the PGA and the probability of
        # failure for that PGA
        p_failure = np.zeros(properties["n"], dtype=np.float64)
        for j, epsilon in enumerate(self.epsilons):
            gmv = np.exp(gmv_mean + epsilon * gmv_sigma)
            # The probability of failure is the adjusted liquefaction
            # probability for that epsilon multiplied by the probability
            # of the given epsilon
            p_failure += (self.truncnorm_probs[j] *
                          self.get_failure_model(sctx, gmv, properties))
        return p_failure

    def _setup_properties(self, sctx, rctx):
        """
        Returns a dictionary of rupture and site properties
        """
        # Calculate the magnitude and ground water depth correction factors
        properties = {
            "kmw": self._get_magnitude_correction_factor(rctx.mag),
            "kdw": self._get_groundwater_depth_correction_factor(sctx),
            "kdelta": self._get_displacement_correction_factor(rctx.mag),
            "n": sctx.liquefaction_susceptibility.shape[0]}

        # All scalar quantities per site category can be pre-computed
        properties["p_fact"] = np.zeros(properties["n"])
        properties["pga_threshold"] = np.zeros(properties["n"])
        properties["settlement"] = np.zeros(properties["n"])
        for i in range(1, 6):
            idx = sctx.liquefaction_susceptibility == i
            if not np.any(idx):
                continue
            properties["p_fact"][idx] = PMU_LIQ[i] /\
                (properties["kmw"] * properties["kdw"][idx])
            properties["settlement"][idx] = D_SETTLEMENT[i]
            properties["pga_threshold"][idx] = PGA_THRESHOLD_CLASS[i]
        return properties

    def _get_failure_gmvs(self, gsimtls):
        """
        Retrieve from gsimtls the ground motion values needed for the
        failure probability
        In the HAZUS case this is only PGA
        """
        return [gsimtls["PGA"]["mean"], gsimtls["PGA"][const.StdDev.TOTAL]]

    def get_failure_model(self, sctx, gmv, properties):
        """
        Returns the probability of failure given the ground motion values,
        the site properties and the epsilon
        """
        # Get the probability of liquefaction for that PGA
        p_liq = get_probability_liquefaction(gmv,
                                             sctx.liquefaction_susceptibility)
        return p_liq * properties["p_fact"]

    def get_poes(self, sctx, rctx, dctx, imtls, truncation_level):
        """
        Returns the probabilities of exceeding the given level of ground
        motion
        """
        # Get the mean and standard deviation ground motions
        gsimtls = self.get_shaking_mean_and_stddevs(sctx, rctx, dctx)
        properties = self._setup_properties(sctx, rctx)
        # Get the magnitude and depth correction factors
        # Setup probabilities with zeros
        poes = OrderedDict([(imt, np.zeros([len(sctx.vs30), len(imtls[imt])]))
                            for imt in imtls])
        gmv_mean, gmv_sigma = self._get_failure_gmvs(gsimtls)
        for j, epsilon in enumerate(self.epsilons):
            # Get the ground motion value for the given epsilon
            gmv = np.exp(gmv_mean + epsilon * gmv_sigma)
            # Get probability of liquefaction for given PGA
            # Determine the probability of failure
            p_failure = self.truncnorm_probs[j] * self.get_failure_model(
                sctx, gmv, properties)
             
            if not np.any(p_failure > 0.):
                # No liquefaction triggered
                continue
            # Deal with settlement first
            if "PGDfSettle" in imtls:
                # As no uncertainty is given in the settlement model then
                # probability of exceeding displacement level is the
                # product of the probability of the epsilon, the probability
                # of failure and a heaviside function taking the value of 1.0
                # if the level is exceeded or 0.0 otherwise
                for i, iml in enumerate(imtls["PGDfSettle"]):
                    poes["PGDfSettle"][:, i] += (
                        p_failure * (properties["settlement"] >= iml))
            # Deal with lateral spread
            if "PGDfLatSpread" in imtls:
                # Get PGA Threshold
                pga_pgat = np.zeros_like(gmv)
                idx = gmv > 0.0
                pga_pgat[idx] = gmv[idx] / properties["pga_threshold"][idx]
                displacement = np.zeros_like(pga_pgat)
                # Calculate lateral spread - based on a set of linear functions
                # defined by m, c for each pga/pgat value
                for (low, high), (m, c) in D_LATERAL_SPREAD:
                    dls_idx = np.logical_and(pga_pgat >= low,
                                             pga_pgat < high)
                    if np.any(dls_idx):
                        # Calculate predicted displacement
                        displacement[dls_idx] = M_PER_INCH *(
                            properties["kdelta"] * (m * pga_pgat[dls_idx] + c))
                for i, iml in enumerate(imtls["PGDfLatSpread"]):
                    # As with settlement, no uncertainty is given for the
                    # lateral spread prediction model to the probability is
                    # defined by a Heaviside function
                    poes["PGDfLatSpread"][:, i] += (p_failure *
                                                   (displacement >= iml))
        # Returns PoEs as a list
        return [poes[imt] for imt in imtls]

    def _get_gmv_field_location(self):
        """
        Finds the location of the necessary IMTs within the IMT list
        In this case only PGA is needed

        """
        if not PGA() in self.imts:
            raise ValueError("HAZUS method requires calculation of PGA "
                             "but not found in imts")

        return [self.imts.index(PGA())]
        #imt_locs = [imts.index(PGA())]
        #return [gmfs[self.IMT_ORDER.index("PGA")]]
        
    def get_displacement_field(self, rupture, sitecol, cmaker, num_events=1,
                               truncation_level=None, correlation_model=None):
        """
        Returns the field of displacements
        """
        # Calculate the ground motion fields
        gmf_loc = self._get_gmv_field_location()
        gmf_computer = GmfComputer(rupture, sitecol,
                                   [str(imt) for imt in self.imts],
                                   cmaker, truncation_level,
                                   correlation_model)
        gmfs = gmf_computer.compute(self.gmpe, num_events, seed=None)
        # Get the PGA field - should have the dimension [nsites, num_events]
        gmvs = gmfs[gmf_loc[0]]
        # Get site and rupture related properties
        properties = self._setup_properties(sitecol,
                                            gmf_computer.rupture)

        # Determine the probability of failure
        # Both the pga threshold and the settlement need to take the same shape
        # as the ground motion values - in this case a 2-D form is needed
        for key in ["pga_threshold", "settlement"]:
            # Tile
            properties[key] = np.tile(
                # Re-shape to 1-D vector then repeat for num. events
                np.reshape(properties[key], [properties["n"], 1]),
                num_events)
         
        p_failure = np.zeros_like(gmvs)
        for j in range(p_failure.shape[1]):
            p_failure[:, j] = self.get_failure_model(sitecol,
                                                     gmvs[:, j],
                                                     properties) 
        # Setup displacement field
        displacement = np.zeros([2, len(gmf_computer.sids), num_events],
                                dtype=np.float32)
        # Sample the field
        mask = np.random.uniform(0., 1., p_failure.shape) <= p_failure
        if not np.any(mask):
            # No liquefaction is observed - return a field of zeros!
            # Note that there are two intensity measures, so to speak, which
            # represent lateral spread and settlement respectively
            # Return the zero displacement fields and the GMFs
            return displacement, gmfs
                    
        # Some sites observe liquefaction - now to calculate lateral
        # spread and settlement
        # Calculate PGA to threshold PGA ratio
        pga_pgat = gmvs[mask] / properties["pga_threshold"][mask]

        # Calculate lateral spread
        lateral_spread = np.zeros_like(pga_pgat)
        # Get the displacement correction factor - which is a function of mag
        for (low, high), (m, c) in D_LATERAL_SPREAD:
            dls_idx = np.logical_and(pga_pgat >= low,
                                     pga_pgat < high)
            if np.any(dls_idx):
                lateral_spread[dls_idx] = M_PER_INCH * (
                    properties["kdelta"] * (m * pga_pgat[dls_idx] + c))
        displacement[0][mask] = np.copy(lateral_spread)
        # Settlement is a simple scalar function
        displacement[1][mask] = properties["settlement"][mask]
        return displacement, gmfs

    def _check_susceptibility(self, sctx):
        """
        In cases where susceptibility category is missing, get the category
        according to the Vs30
        """
        idx = np.isnan(sctx.liquefaction_susceptibility)
        if np.any(idx):
            sctx.liquefaction_susceptibility[idx] = \
                _get_liquefaction_susceptibility_category(vs30[idx])
        return sctx

    def _get_magnitude_correction_factor(self, mag):
        """
        Returns the magnitude correction factor
        (Equation 4.21)
        """
        return 0.0027 * (mag ** 3.) - 0.0267 * (mag ** 2.) -\
            0.2055 * mag + 2.9188
    
    def _get_groundwater_depth_correction_factor(self, sctx):
        """
        Returns the groundwater depth correction factor
        (Equation 4.22)
        """
        return 0.022 * sctx.dw + 0.93
    
    def _get_displacement_correction_factor(self, mag):
        """
        Returns the displacement correction Kdelta
        """
        return 0.0086 * (mag ** 3.) - 0.0914 * (mag ** 2.) +\
            0.4698 * mag - 0.9835


SLOPE_CRITICAL_ACCELERATIONS = {
    0: np.inf, 1: 0.6, 2: 0.5, 3: 0.4, 4: 0.35, 5: 0.3,
    6: 0.25, 7: 0.2, 8: 0.15, 9: 0.1, 10: 0.05
}


LANDSLIDING_AREA = {
    0: 0.0, 1: 0.01, 2: 0.02, 3: 0.03, 4: 0.05, 5: 0.08,
    6: 0.1, 7: 0.15, 8: 0.2, 9: 0.25, 10: 0.3
}


UPPER_BOUND = np.array([[0.10, 40.11421158253171],
                        [0.20, 18.73304150352677],
                        [0.30, 10.659256441146821],
                        [0.40, 5.289741049710613],
                        [0.50, 2.585511221751352],
                        [0.60, 1.283080571304289],
                        [0.70, 0.5901441824802593],
                        [0.80, 0.20962887837169294],
                        [0.90, 0.04865594982811198]])

F_UB = interp1d(UPPER_BOUND[:, 0], np.log10(UPPER_BOUND[:, 1]),
                kind='linear', fill_value="extrapolate")


LOWER_BOUND = np.array([[0.10, 22.515331794940273],
                        [0.20, 11.518267840625688],
                        [0.30, 6.357764958836381],
                        [0.40, 3.1550915838131592],
                        [0.50, 1.5897370830822521],
                        [0.60, 0.7889197105874919],
                        [0.70, 0.3116954104184578],
                        [0.80, 0.0995460290224862],
                        [0.90, 0.019846861460734253]])


F_LB = interp1d(LOWER_BOUND[:, 0], np.log10(LOWER_BOUND[:, 1]),
                kind='linear', fill_value="extrapolate")


class HAZUSLandsliding(GDEM):
    """
    """
    DEFINED_FOR_DEFORMATION_TYPES = set((PGDfSlope,))
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set((PGA,))
    REQUIRES_SITES_PARAMETERS = set(("landsliding_susceptibility", "vs30"))

    def get_probability_failure(self, sctx, rctx, dctx):
        """
        Returns the probability of failure
        """
        gsimtls = self.get_shaking_mean_and_stddevs(sctx, rctx, dctx)
        # Get the yield acceleration and map area (i.e. probability of
        # observing slope displacement per unit)
        properties = self._setup_properties(sctx, rctx)
        #a_c, pma = self._get_ac_map_area(sctx)
        gmv_mean, gmv_sigma = self._get_failure_gmvs(gsimtls)
        p_failure = np.zeros_like(properties["a_c"])
        for j, epsilon in enumerate(self.epsilons):
            # Get the PGA corresponding to a given epislon
            gmv = np.exp(gmv_mean + epsilon * gmv_sigma)
            idx = gmv >= properties["a_c"]
            # Where PGA exceeds the yield acceleration multiply the
            # epsilon probability by the map area
            if np.any(idx):
                p_failure[idx] += (self.truncnorm_probs[j] *
                                   properties["pma"][idx])
        return p_failure

    def _get_failure_gmvs(self, gsimtls):
        """
        Retrieve from gsimtls the ground motion values needed for the
        failure probability
        In the HAZUS case this is only PGA
        """
        return [gsimtls["PGA"]["mean"], gsimtls["PGA"][const.StdDev.TOTAL]]

    def get_poes(self, sctx, rctx, dctx, imtls, truncation_level):
        """
        Returns the probabilities of exceeding the given level of ground
        motion
        """
        gsimtls = self.get_shaking_mean_and_stddevs(sctx, rctx, dctx)
        #a_c, pma = self._get_ac_map_area(sctx)
        properties = self._setup_properties(sctx, rctx)
        gmv_mean, gmv_sigma = self._get_failure_gmvs(gsimtls)
        poes = np.zeros([len(sctx.landsliding_susceptibility),
                         len(imtls["PGDfSlope"])])
        n_cycles = self._get_number_cycles(rctx.mag)
        for j, epsilon in enumerate(self.epsilons):
            gmv = np.exp(gmv_mean + epsilon * gmv_sigma)
            # Get probability of failure
            idx = gmv >= properties["a_c"]
            if not np.any(idx):
                # Nothing exceeding critical acceleration
                continue
            p_failure = properties["pma"] * idx
            # Get expected ground motion
            e_d_ub, e_d_lb = self._get_expected_displacement_factor(
                properties["a_c"][idx] / gmv[idx])

            lb_displacement = M_PER_INCH * n_cycles * e_d_lb * gmv[idx]
            ub_displacement = M_PER_INCH * n_cycles * e_d_ub * gmv[idx]
            diff_displacement = ub_displacement - lb_displacement
            for i, iml in enumerate(imtls["PGDfSlope"]):
                # A Uniform distribution is assumed between the upper
                # and lower bound displacements
                displacement_prob = (iml - lb_displacement) / diff_displacement
                displacement_prob[displacement_prob < 0.] = 0.
                displacement_prob[displacement_prob > 1.] = 1.
                poes[idx, i] += (self.truncnorm_probs[j] *
                                 p_failure[idx] *
                                 (1.0 - displacement_prob))
        return [poes]

    def _get_gmv_field_location(self):
        """
        Get the ground motion values needed for the field - in this case PGA
        """
        if not PGA() in self.imts:
            raise ValueError("HAZUS Method requires calculation of PGA "
                             "but not found in imts")

        return [self.imts.index(PGA())]

    def get_displacement_field(self, rupture, sitecol, cmaker, num_events=1,
                               truncation_level=None, correlation_model=None):
        """
        Returns the field of displacements
        """
        gmf_loc = self._get_gmv_field_location()
        # Gets the ground motion fields
        gmf_computer = GmfComputer(rupture, sitecol, 
                                   [str(imt) for imt in self.imts],
                                   cmaker, truncation_level, correlation_model)
        gmfs = gmf_computer.compute(self.gmpe, num_events, seed=None)
        # Get the PGA field
        gmv = gmfs[gmf_loc[0]]
        # Return the critical acceleration and proportion of mapped area
        properties = self._setup_properties(sitecol,
                                            gmf_computer.rupture)
        # Get probability of failure. If PGA < a_c, no failure is possible,
        # whilst for PGA > a_c the probability of any individual location
        # observing displacement is equal to the proportion of mapped area
        p_failure = np.zeros_like(gmv)
        for i in range(num_events):
            p_failure[:, i] = properties["pma"] * (gmv[:, i] >=
                                                   properties["a_c"])
        # Sample the occurence of landsliding.
        mask = np.random.uniform(0., 1., p_failure.shape) <= p_failure
        displacement = np.zeros([1, len(gmf_computer.sids), num_events],
                                 dtype=np.float32)
        if not np.any(mask):
            # No displacement at any site - return zeros and the ground motion
            # fields
            return displacement, gmfs
        # If any displacement is registered then need to turn a_c into
        # array of shape [nsites, num_events]
        properties["a_c"] = np.tile(
                # Re-shape to 1-D vector then repeat for num. events
                np.reshape(properties["a_c"], [properties["n"], 1]),
                num_events)
        # Calculate number of cycles factor
        n_cycles = self._get_number_cycles(gmf_computer.rupture.mag)
        # Calculate ratio of acceleration to critical acceleration
        a_c_ais = properties["a_c"][mask] / gmv[mask]
        # Expected displacement factor returns lower and upper bounds, assumed
        # to be uniformly distributed
        e_d_ub, e_d_lb = self._get_expected_displacement_factor(a_c_ais)
        # Final displacement sampling uniformly between the upper and
        # lower bound displacement factors
        displacement[0][mask] = M_PER_INCH * n_cycles * gmv[mask] *\
            np.random.uniform(e_d_lb, e_d_ub, a_c_ais.shape)
        return displacement, gmfs
  
    def get_yield_acceleration(self, sctx):
        """
        Gets the yield acceleration according to the HAZUS susceptibility
        categories
        """
        a_c = np.zeros(sctx.landsliding_susceptibility.shape)
        for i in range(1, 11):
            idx = sctx.landsliding_susceptibility == i
            a_c[idx] = SLOPE_CRITICAL_ACCELERATIONS[i]
        return a_c

    def _setup_properties(self, sctx, rctx):
        """
        For the site category returns the critical accelerations and
        landsliding area
        """
        properties = {"a_c": self.get_yield_acceleration(sctx),
                      "n": sctx.landsliding_susceptibility.shape[0]}
        properties["pma"] = np.zeros_like(properties["a_c"])
        for i in range(1, 11):
            idx = sctx.landsliding_susceptibility == i
            properties["pma"][idx] = LANDSLIDING_AREA[i]
        return properties

    def _get_expected_displacement_factor(self, ac_ais):
        """
        Returns the expected displacement factor, this describes two curves
        (an upper and lower bound curve) with a uniform distribution assumed
        betweem them
        """
        return (10.0 ** F_UB(ac_ais), 10.0 ** F_LB(ac_ais))
    
    def _get_number_cycles(self, mag):
        """
        Returns the number of cycles
        """
        return 0.3419 * (mag ** 3.) - 5.5214 * (mag ** 2.) + 33.6154 * mag -\
            70.7692
