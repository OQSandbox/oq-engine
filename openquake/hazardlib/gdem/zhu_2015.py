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
Liquefaction Hazard Assessment using the formulation of Zhu et al (2015)
and  Zhu et al. (2017) for the probability of failure, with HAZUS estimates
of settlement and spreading
"""

import numpy as np
from scipy.interpolate import interp1d
from collections import OrderedDict
from openquake.hazardlib.gdem.base import GDEM, GSIMFComputer
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGDfSettle, PGDfLatSpread, PGA, PGV
from openquake.hazardlib.gdem.hazus import (M_PER_INCH, PMU_LIQ,
                                            PGA_THRESHOLD_CLASS,
                                            PGA_T_RATIO_LIQ,
                                            get_pga_t_ratio,
                                            D_SETTLEMENT, D_LATERAL_SPREAD)

pgdf_settle = PGDfSettle()
pgdf_spread = PGDfLatSpread()

class ZhuEtAl2015Global(GDEM):
    """
    Implements a probabilistic version of the HAZUS Liquefaction calculator
    as described in Chapter 4.2.2.1.4 of the HAZUS Technical Manual
    """
    DEFINED_FOR_DEFORMATION_TYPES = set((PGDfSettle, PGDfLatSpread))
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set((PGA,))
    REQUIRES_RUPTURE_PARAMETERS = set(())
    REQUIRES_SITES_PARAMETERS = set(("liquefaction_susceptibility",
                                     "cti", "vs30"))

    def get_probability_failure(self, sctx, rctx, dctx, gsimtls=None):
        """
        Returns the probability of failure
        """
        if not gsimtls:
            gsimtls = self.get_mean_and_stddevs(sctx, rctx, dctx)

        p_failure = np.zeros(sctx.vs30.shape,
                             dtype=np.float64)
        # Get the mean and standard deviation PGA
        pga_mean, pga_sigma = gsimtls["PGA"]["mean"],\
            gsimtls["PGA"][const.StdDev.TOTAL]
        # For each epsilon value determine the PGA and the probability of
        # failure for that PGA
        for j, epsilon in enumerate(self.epsilons):
            # Get the PGA for the given epsilon
            pga = np.exp(pga_mean + epsilon * pga_sigma)
            # Get the probability of failure
            model = self.failure_model(pga, sctx)
            p_failure += (self.truncnorm_probs[j] *
                          (1. / (1. + np.exp(model))))
        return p_failure
    
    def failure_model(self, iml, sctx):
        """
        """
        return 24.10 + 2.067 * np.log(iml) + 0.355 * sctx.cti +\
            -4.784 * np.log(sctx.vs30)

    def get_poes(self, sctx, rctx, dctx, imtls, truncation_level):
        """
        Returns the probabilities of exceeding the given level of ground
        motion
        """
        # Get the mean and standard deviation ground motions
        gsimtls = self.get_mean_and_stddevs(sctx, rctx, dctx)
        pga_threshold = np.zeros_like(p_fact)
        settlement = np.zeros_like(p_fact)
        pga_threshold = np.zeros_like(p_fact)
        settlement = np.zeros_like(p_fact)
        for i in range(1, 6):
            idx = sctx.liquefaction_susceptibility == i
            if not np.any(idx):
                continue
            # Calculates the adjustment factors for the probability of
            # liquefaction
            settlement[idx] = D_SETTLEMENT[i]
            pga_threshold[idx] = PGA_THRESHOLD_CLASS[i]
        # Get the displacement correction factor
        kdelta = self._get_displacement_correction_factor(rctx.mag)

        # Setup probabilities with zeros
        poes = OrderedDict([(imt, np.zeros([len(sctx.vs30), len(imtls[imt])]))
                            for imt in imtls])

        for j, epsilon in enumerate(self.epsilons):
            # Get the PGA for the given epsilon
            pga = np.exp(gsimtls["PGA"]["mean"] +
                         epsilon * gsimtls["PGA"][const.StdDev.TOTAL])
            p_failure = (1. / (1. + np.exp(self.failure_model(pga, sctx))))
            if not np.any(p_failure > 0.):
                # No liquefaction observed at this epsilon
                continue
            if pgdf_settle in imtls:
                for i, iml in enumerate(imtls[pgdf_settle]):
                    poes[pgdf_settle][:, i] += (self.truncnorm_probs[j] *
                                                p_failure *
                                                (settlement >= iml))
            if pgdf_spread in imtls:
                # Get PGA Threshold
                pga_pgat = np.zeros_like(pga)
                idx = pga > 0.0
                pga_pgat[idx] = pga[idx] / pga_threshold[idx]
                displacement = np.zeros_like(pga_pgat)
                # Calculate lateral spread - based on a set of linear functions
                # defined by m, c for each pga/pgat value
                for (low, high), (m, c) in D_LATERAL_SPREAD:
                    dls_idx = np.logical_and(pga_pgat >= low,
                                             pga_pgat < high)
                    if np.any(dls_idx):
                        # Calculate predicted displacement
                        displacement[dls_idx] = M_PER_INCH * (
                            kdelta * (m * pga_pgat[dls_idx] + c))
                for i, iml in enumerate(imtls[pgdf_latspread]):
                    # As with settlement, no uncertainty is given for the
                    # lateral spread prediction model to the probability is
                    # defined by a Heaviside function
                    poes[pgdf_spread][:, i] += (self.truncnorm_probs[j] *
                                                p_failure *
                                                (displacement >= iml))
        # Returns PoEs as a list
        return [poes[imt] for imt in imtls]

    @staticmethod
    def _get_displacement_correction_factor(mag):
        """
        Returns the displacement correction Kdelta
        """
        return 0.0086 * (mag ** 3.) - 0.0914 * (mag ** 2.) +\
            0.4698 * mag - 0.9835

    def get_displacement_field(self, rupture, sitecol, cmaker, num_events=1,
                               truncation_level=None, correlation_model=None):
        """
        Returns the field of displacements
        """
        # Calculate the ground motion fields
        gmf_computer = GSIMFComputer(rupture, sitecol, list(self.gmpe_set),
                                     cmaker, truncation_level,
                                     correlation_model)
        gmfs = gmf_computer.compute(self.gmpe_set, num_events, seed=None)
        # Get the PGA field - should have the dimension [nsites, num_events]
        pga = gmfs[list(self.gmpe_set).index("PGA")]
        # Determine the probability of failure
        # Get correction factors
        # Get the site properties
        p_failure = np.zeros_like(pga)
        settlement = np.zeros_like(pga)
        pga_threshold = np.zeros_like(pga)
        for i in range(1, 6):
            idx = gmf_computer.ctx[0].liquefaction_susceptibility == i
            if not np.any(idx):
                continue
            p_fact[idx] = PMU_LIQ[i] / (kmw * kdw[idx])
            pga_threshold[idx, :] = PGA_THRESHOLD_CLASS[i]
            settlement[idx, :] = D_SETTLEMENT[i]
        for j in range(p_failure.shape[1]):
            p_failure[:, j] = 1.0 / (1. + np.exp(self.failure_model(pga[:, j],
                                                                    sctx)))

        # Setup displacement field
        displacement = np.zeros([2, len(gmf_computer.sids), num_events],
                                dtype=np.float32)
        # Sample the field
        sampler = np.random.uniform(0., 1., p_failure.shape)
        mask = sampler <= p_failure
        if not np.any(mask):
            # No liquefaction is observed - return a field of zeros!
            # Note that there are two intensity measures, so to speak, which
            # represent lateral spread and settlement respectively
            # Return the zero displacement fields and the GMFs
            return displacement, gmfs
                    
        # Some sites observe liquefaction - now to calculate lateral
        # spread and settlement

        # Calculate PGA to threshold PGA ratio
        pga_pgat = pga[mask] / pga_threshold[mask]

        # Calculate lateral spread
        lateral_spread = np.zeros_like(pga_pgat)
        # Get the displacement correction factor - which is a function of mag
        kdelta = self._get_displacement_correction_factor(
            gmf_computer.ctx[1].mag)
        for (low, high), (m, c) in D_LATERAL_SPREAD:
            dls_idx = np.logical_and(pga_pgat >= low,
                                     pga_pgat < high)
            if np.any(dls_idx):
                lateral_spread[dls_idx] = M_PER_INCH * (
                    kdelta * (m * pga_pgat[dls_idx] + c))
        displacement[0][mask] = np.copy(lateral_spread)
        # Settlement is a simple scalar function
        displacement[1][mask] = settlement[mask]
        return displacement, gmfs          
            






