import unittest
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


import numpy as np

from openquake.hazardlib.source import SimpleFaultSource
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.geo import Line, Point, SimpleFaultSurface, Mesh
from openquake.hazardlib.mfd import EvenlyDiscretizedMFD
from openquake.hazardlib.scalerel import PeerMSR
from openquake.hazardlib import const
from openquake.hazardlib.imt import (PGA, SA, PGV,PGDfLatSpread,
                                     PGDfSettle, PGDfSlope)
from openquake.hazardlib.site import Site, SiteCollection, site_param_dt
from openquake.baselib.general import DictArray, groupby, AccumDict
from openquake.hazardlib.calc.filters import SourceFilter
from openquake.hazardlib import valid


def _setup_fault_source():
    """
    Builds a fault source using the PEER Bending Fault case.
    """
    point_order_dipping_east = [Point(-64.78365, -0.45236),
                                Point(-64.80164, -0.45236),
                                Point(-64.90498,-0.36564),
                                Point(-65.0000, -0.16188),
                                Point(-65.0000, 0.0000)]
    trace_dip_east = Line(point_order_dipping_east)
    fault_surface1 = SimpleFaultSurface.from_fault_data(trace_dip_east,
                                                        0.0, 12.0, 60., 1.0)
    # Activity Rates
    # cm per km2
    area = 60. * 12.0 / np.sin(np.radians(60.))
    cm2perkm2 = (100. * 1000.) ** 2.
    mo1 = 3.0E11 * 0.2 * (area * cm2perkm2)
    mo_m6p75 = 10.0 ** (16.05 + 1.5 * 6.75)
    rate1 = mo1 / mo_m6p75
    mfd1 = EvenlyDiscretizedMFD(6.75, 0.01, [rate1])
    tom = PoissonTOM(1.0)
    aspect = 2.0
    rake = 90.0
    src = SimpleFaultSource("PEER_FLT_EAST",
                             "PEER Bending Fault Dipping East",
                             "Active Shallow Crust", mfd1, 1.0, PeerMSR(), 2.0,
                             tom, 0.0, 12.0, trace_dip_east, 60.0, rake)
    src.num_ruptures = src.count_ruptures()
    return src

def _setup_sites_liquefaction():
    """
    Returns a collection of sites for liquefaction hazard analysis using the
    HAZUS method
    """
    
    point_1 = Point(-64.98651, -0.15738)
    point_2 = Point(-64.77466, -0.45686)
    point_3 = Point(-64.92747, -0.38363)
    point_4 = Point(-65.05396, -0.17088)
    #         Vs30  HAZUS   dw
    params = [[800., 0, 100.],
              [600., 1, 50.],
              [400., 2, 50.],
              [300., 3, 20.],
              [200., 4, 20.],
              [150., 5, 20.],
              [150., 5, 5.]]
    sites = []
    for locn in [point_1, point_2, point_3, point_4]:
        for vs30, hazus_cat, dw in params:
            sites.append({"lon": locn.longitude,
                          "lat": locn.latitude, "vs30": vs30,
                          "vs30measured":False, "z1pt0":48.0, "z2pt5":0.607,
                          "liquefaction_susceptibility":hazus_cat, "dw":dw})
    site_model_dt = np.dtype([(p, site_param_dt[p])
                              for p in sorted(sites[0])])
    tuples = [tuple(site[name] for name in site_model_dt.names)
              for site in sites]
    sites = np.array(tuples, site_model_dt)
    req_site_params = ("vs30", "liquefaction_susceptibility", "dw")
    return SiteCollection.from_points(sites["lon"], sites["lat"],
                                      sitemodel=sites,
                                      req_site_params=req_site_params)


def _setup_sites_liquefaction_zhu():
    """
    Returns a collection of sites for liquefaction hazard analysis using the
    HAZUS method
    """
    
    point_1 = Point(-64.98651, -0.15738)
    point_2 = Point(-64.77466, -0.45686)
    point_3 = Point(-64.92747, -0.38363)
    point_4 = Point(-65.05396, -0.17088)
    params = [[800., 0, 100., 0.0],
              [600., 1, 50., 1.0],
              [400., 2, 50., 2.0],
              [300., 3, 20., 5.0],
              [200., 4, 20., 7.0],
              [150., 5, 20., 10.0],
              [150., 5, 5., 15.0]]
    sites = []
    for locn in [point_1, point_2, point_3, point_4]:
        for vs30, hazus_cat, dw, cti in params:
            sites.append({"lon": locn.longitude,
                          "lat": locn.latitude, "vs30": vs30,
                          "vs30measured":False, "z1pt0": 48.0, "z2pt5": 0.607,
                          "liquefaction_susceptibility": hazus_cat, "dw": dw,
                          "cti": cti})
    site_model_dt = np.dtype([(p, site_param_dt[p])
                              for p in sorted(sites[0])])
    tuples = [tuple(site[name] for name in site_model_dt.names)
              for site in sites]
    sites = np.array(tuples, site_model_dt)
    req_site_params = ("vs30", "liquefaction_susceptibility", "dw", "cti")
    return SiteCollection.from_points(sites["lon"], sites["lat"],
                                      sitemodel=sites,
                                      req_site_params=req_site_params)

def _setup_sites_landsliding():
    """
    Returns a collections of sites at four different locations with different
    landsliding properties for the HAZUS method
    """
    ls_params = [[800., 10],
                 [800., 9],
                 [600., 8],
                 [600., 7],
                 [400., 6],
                 [400., 5],
                 [300., 4],
                 [300., 3],
                 [200., 2],
                 [200., 1],
                 [150., 0]]
    point_1 = Point(-64.98651, -0.15738)
    point_2 = Point(-64.77466, -0.45686)
    point_3 = Point(-64.92747, -0.38363)
    point_4 = Point(-65.05396, -0.17088)
    sites = []
    for locn in [point_1, point_2, point_3, point_4]:
        for vs30, hazus_ls_cat in ls_params:
            sites.append({"lon": locn.longitude,
                          "lat": locn.latitude, "vs30": vs30,
                          "vs30measured":False, "z1pt0":48.0, "z2pt5":0.607,
                          "landsliding_susceptibility": hazus_ls_cat}
                          )
    site_model_dt = np.dtype([(p, site_param_dt[p])
                              for p in sorted(sites[0])])
    tuples = [tuple(site[name] for name in site_model_dt.names)
              for site in sites]
    sites = np.array(tuples, site_model_dt)
    req_site_params = ("vs30", "landsliding_susceptibility")
    return SiteCollection.from_points(sites["lon"], sites["lat"],
                                      sitemodel=sites,
                                      req_site_params=req_site_params)

def _setup_sites_yield_accel():
    """
    Returns a collection of sites (at four different locations) with
    different yield acceleration properties
    """
    ls_params = [[800., 1.0],
                 [600., 0.9],
                 [400., 0.8],
                 [300., 0.6],
                 [200., 0.4],
                 [150., 0.0]]
    point_1 = Point(-64.98651, -0.15738)
    point_2 = Point(-64.77466, -0.45686)
    point_3 = Point(-64.92747, -0.38363)
    point_4 = Point(-65.05396, -0.17088)
    sites = []
    for locn in [point_1, point_2, point_3, point_4]:
        for vs30, k_y in ls_params:
            sites.append({"lon": locn.longitude,
                          "lat": locn.latitude, "vs30": vs30,
                          "vs30measured":False, "z1pt0": 48.0, "z2pt5": 0.607,
                          "yield_acceleration": k_y})
    site_model_dt = np.dtype([(p, site_param_dt[p])
                              for p in sorted(sites[0])])
    tuples = [tuple(site[name] for name in site_model_dt.names)
              for site in sites]
    sites = np.array(tuples, site_model_dt)
    req_site_params = ("vs30", "yield_acceleration")
    return SiteCollection.from_points(sites["lon"], sites["lat"],
                                      sitemodel=sites,
                                      req_site_params=req_site_params)
