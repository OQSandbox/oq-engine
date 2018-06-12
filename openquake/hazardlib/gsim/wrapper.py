# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2012-2018 GEM Foundation
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
Wrapper GMPE to be passed a dictionary of ground motion models organised
by IMT type
"""

import re
from openquake.hazardlib.gsim import get_available_gsims
from openquake.hazardlib.gsim.base import CoeffsTable, GMPE
from openquake.hazardlib import const
from openquake.hazardlib.imt import from_string

GSIM_LIST = get_available_gsims()

class WrapperGMPE(GMPE):
    """
    GMPE can call ground motions for various IMTs when instantiated with
    a dictionary of ground motion models organised by IMT
    """
    #: Supported tectonic region type is undefined
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.UNDEFINED

    #: Supported intensity measure types are not set
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set(())

    #: Supported intensity measure component is horizontal
    #: :attr:`~openquake.hazardlib.const.IMC.HORIZONTAL`,
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.HORIZONTAL

    #: Supported standard deviation type
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL
    ])

    #: Required site parameters will be set be selected GMPES
    REQUIRES_SITES_PARAMETERS = set(())

    #: Required rupture parameter is magnitude, others will be set by the GMPEs
    REQUIRES_RUPTURE_PARAMETERS = set(('mag', ))

    #: Required distance metrics will be set by the GMPEs
    REQUIRES_DISTANCES = set(())

    def __init__(self, gmpes_by_imt):
        """
        Instantiate with a dictionary of GMPEs organised by IMT. In this case 
        """
        super().__init__()
        self.gmpes = {}
        if isinstance(gmpes_by_imt, str):
            # Convert string into dictionary
            gmpes_by_imt = self._gmpe_str_to_dict(gmpes_by_imt)

        for imt, gmpe in list(gmpes_by_imt.items()):
            # IMT should be a string
            gmpe_imt = from_string(imt)
            self.gmpes[gmpe_imt] = GSIM_LIST[gmpe]()
            if not gmpe_imt.__class__ in\
                self.gmpes[gmpe_imt].DEFINED_FOR_INTENSITY_MEASURE_TYPES:
                raise ValueError("IMT %s not supported by %s" % (imt, gmpe))
            self.DEFINED_FOR_INTENSITY_MEASURE_TYPES = (
                self.DEFINED_FOR_INTENSITY_MEASURE_TYPES |
                self.gmpes[gmpe_imt].DEFINED_FOR_INTENSITY_MEASURE_TYPES)
            self.DEFINED_FOR_STANDARD_DEVIATION_TYPES = (
                self.DEFINED_FOR_STANDARD_DEVIATION_TYPES |
                self.gmpes[gmpe_imt].DEFINED_FOR_STANDARD_DEVIATION_TYPES)
            self.REQUIRES_SITES_PARAMETERS = (
                self.REQUIRES_SITES_PARAMETERS |
                self.gmpes[gmpe_imt].REQUIRES_SITES_PARAMETERS)
            self.REQUIRES_RUPTURE_PARAMETERS = (
                self.REQUIRES_RUPTURE_PARAMETERS |
                self.gmpes[gmpe_imt].REQUIRES_RUPTURE_PARAMETERS)
            self.REQUIRES_DISTANCES = (self.REQUIRES_DISTANCES |
                self.gmpes[gmpe_imt].REQUIRES_DISTANCES)

    @staticmethod
    def _gmpe_str_to_dict(gmpes_by_imt):
        """
        """
        content = re.search(r"\{(.*)\}", gmpes_by_imt).group(1)
        gmpe_dict = []
        for keyval in content.split(","):
            imt_str, gmpe_str = keyval.split(":")
            gmpe_dict.append((imt_str.strip(), gmpe_str.strip()))
        return dict(gmpe_dict)


    def get_mean_and_stddevs(self, sctx, rctx, dctx, imt, stddev_types):
        """
        Call the get mean and stddevs of the GMPE for the respective IMT
        """
        return self.gmpes[imt].get_mean_and_stddevs(sctx, rctx, dctx, imt,
                                                    stddev_types)
