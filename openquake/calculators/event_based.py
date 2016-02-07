#  -*- coding: utf-8 -*-
#  vim: tabstop=4 shiftwidth=4 softtabstop=4

#  Copyright (c) 2015, GEM Foundation

#  OpenQuake is free software: you can redistribute it and/or modify it
#  under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

#  OpenQuake is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU Affero General Public License
#  along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.
import time
import os.path
import random
import operator
import logging
import itertools
import collections

import numpy

from openquake.baselib.general import AccumDict, humansize
from openquake.hazardlib.calc.filters import \
    filter_sites_by_distance_to_rupture
from openquake.hazardlib.calc.hazard_curve import zero_curves
from openquake.hazardlib import geo, site, calc
from openquake.hazardlib.gsim.base import gsim_imt_dt
from openquake.commonlib import readinput, parallel, datastore
from openquake.commonlib.util import max_rel_diff_index

from openquake.calculators import base, views
from openquake.commonlib.oqvalidation import OqParam
from openquake.calculators.calc import MAX_INT, gmvs_to_haz_curve
from openquake.calculators.classical import ClassicalCalculator

# ######################## rupture calculator ############################ #

# a numpy record storing the number of ruptures and ground motion fields
# for each realization
counts_dt = numpy.dtype([('rup', int), ('gmf', int)])


def num_affected_sites(rupture, num_sites):
    """
    :param rupture: a SESRupture object
    :param num_sites: the total number of sites
    :returns: the number of sites affected by the rupture
    """
    return (len(rupture.indices) if rupture.indices is not None
            else num_sites)


def get_site_ids(rupture, num_sites):
    """
    :param rupture: a SESRupture object
    :param num_sites: the total number of sites
    :returns: the indices of the sites affected by the rupture
    """
    if rupture.indices is None:
        return list(range(num_sites))
    return rupture.indices


def counts_per_rlz(num_sites, rlzs_assoc, sescollection):
    """
    :param num_sites: the number of sites
    :param rlzs_assoc: an instance of RlzsAssoc
    :param sescollection: a list of dictionaries tag -> SESRupture
    :returns: the numbers of nonzero GMFs, for each realization
    """
    rlzs = rlzs_assoc.realizations
    counts = numpy.zeros(len(rlzs), counts_dt)
    for rlz in rlzs:
        col_ids = list(rlzs_assoc.get_col_ids(rlz))
        for sc in sescollection[col_ids]:
            i = rlz.ordinal

            # ruptures per realization
            counts['rup'][i] += len(sc)

            # gmvs per realization
            for rup in sc.values():
                counts['gmf'][i] += num_affected_sites(rup, num_sites)
    return counts


# this is used in the sanity check in the execute method
def get_gmfs_nbytes(num_sites, num_imts, rlzs_assoc, sescollection):
    """
    :param num_sites: the number of sites
    :param num_imts: the number of IMTs
    :param rlzs_assoc: an instance of RlzsAssoc
    :param sescollection: a list of dictionaries tag -> SESRupture
    :returns: the number of bytes required to store the GMFs
    """
    nbytes = 0
    # iterating over the ses collections;
    # the bytes needed to store the GMF generated by a singlerupture are
    # num_affected_sites * bytes_per_record; the size of a record is
    # 4 bytes for the idx + 8 bytes * number_of_gsims * number_of_imts
    for sescol, gsims in zip(sescollection, rlzs_assoc.get_gsims_by_col()):
        bytes_per_record = 4 + 8 * len(gsims) * num_imts
        for tag, rup in sescol.items():
            nbytes += bytes_per_record * num_affected_sites(rup, num_sites)
    return nbytes


@datastore.view.add('gmfs_total_size')
def view_gmfs_total_size(name, dstore):
    """
    :returns:
        the total size of the GMFs as human readable string; it assumes
        4 bytes for the rupture index, 4 bytes for the realization index
        and 8 bytes for each float (there are num_imts floats per gmf)
    """
    nbytes = 0
    num_imts = len(OqParam.from_(dstore.attrs).imtls)
    for counts in dstore['counts_per_rlz']:
        nbytes += 8 * counts['gmf'] * (num_imts + 1)
    return humansize(nbytes)


@datastore.view.add('col_rlz_assocs')
def view_col_rlz_assocs(name, dstore):
    """
    :returns: an array with the association array col_ids -> rlz_ids
    """
    rlzs_assoc = dstore['rlzs_assoc']
    num_ruptures = dstore['num_ruptures']
    num_rlzs = len(rlzs_assoc.realizations)
    col_ids_list = [[] for _ in range(num_rlzs)]
    for rlz in rlzs_assoc.realizations:
        for col_id in sorted(rlzs_assoc.get_col_ids(rlz)):
            if num_ruptures[col_id]:
                col_ids_list[rlz.ordinal].append(col_id)
    assocs = collections.defaultdict(list)
    for i, col_ids in enumerate(col_ids_list):
        assocs[tuple(col_ids)].append(i)
    tbl = [['Collections', 'Realizations']] + sorted(assocs.items())
    return views.rst_table(tbl)


# #################################################################### #


def get_geom(surface, is_from_fault_source, is_multi_surface):
    """
    The following fields can be interpreted different ways,
    depending on the value of `is_from_fault_source`. If
    `is_from_fault_source` is True, each of these fields should
    contain a 2D numpy array (all of the same shape). Each triple
    of (lon, lat, depth) for a given index represents the node of
    a rectangular mesh. If `is_from_fault_source` is False, each
    of these fields should contain a sequence (tuple, list, or
    numpy array, for example) of 4 values. In order, the triples
    of (lon, lat, depth) represent top left, top right, bottom
    left, and bottom right corners of the the rupture's planar
    surface. Update: There is now a third case. If the rupture
    originated from a characteristic fault source with a
    multi-planar-surface geometry, `lons`, `lats`, and `depths`
    will contain one or more sets of 4 points, similar to how
    planar surface geometry is stored (see above).

    :param rupture: an instance of :class:`openquake.hazardlib.source.rupture.BaseProbabilisticRupture`
    :param is_from_fault_source: a boolean
    :param is_multi_surface: a boolean
    """
    if is_from_fault_source:
        # for simple and complex fault sources,
        # rupture surface geometry is represented by a mesh
        surf_mesh = surface.get_mesh()
        lons = surf_mesh.lons
        lats = surf_mesh.lats
        depths = surf_mesh.depths
    else:
        if is_multi_surface:
            # `list` of
            # openquake.hazardlib.geo.surface.planar.PlanarSurface
            # objects:
            surfaces = surface.surfaces

            # lons, lats, and depths are arrays with len == 4*N,
            # where N is the number of surfaces in the
            # multisurface for each `corner_*`, the ordering is:
            #   - top left
            #   - top right
            #   - bottom left
            #   - bottom right
            lons = numpy.concatenate([x.corner_lons for x in surfaces])
            lats = numpy.concatenate([x.corner_lats for x in surfaces])
            depths = numpy.concatenate([x.corner_depths for x in surfaces])
        else:
            # For area or point source,
            # rupture geometry is represented by a planar surface,
            # defined by 3D corner points
            lons = numpy.zeros((4))
            lats = numpy.zeros((4))
            depths = numpy.zeros((4))

            # NOTE: It is important to maintain the order of these
            # corner points. TODO: check the ordering
            for i, corner in enumerate((surface.top_left,
                                        surface.top_right,
                                        surface.bottom_left,
                                        surface.bottom_right)):
                lons[i] = corner.longitude
                lats[i] = corner.latitude
                depths[i] = corner.depth
    return lons, lats, depths


# this is very ugly for compatibility with the Django API in the engine
# TODO: simplify this, now that the old calculators have been removed
class SESRupture(object):
    def __init__(self, rupture, indices, seed, tag, col_id):
        self.rupture = rupture
        self.indices = indices
        self.seed = seed
        self.tag = tag
        self.col_id = col_id
        # extract the SES ordinal (>=1) from the rupture tag
        # for instance 'col=00~ses=0001~src=1~rup=001-01' => 1
        pieces = tag.split('~')
        self.ses_idx = int(pieces[1].split('=')[1])
        self.ordinal = None  # to be set

    def export(self):
        """
        Return a new SESRupture object, with all the attributes set
        suitable to export in XML format.
        """
        rupture = self.rupture
        new = self.__class__(
            rupture, self.indices, self.seed, self.tag, self.col_id)
        new.rupture = new
        new.is_from_fault_source = iffs = isinstance(
            rupture.surface, (geo.ComplexFaultSurface, geo.SimpleFaultSurface))
        new.is_multi_surface = ims = isinstance(
            rupture.surface, geo.MultiSurface)
        new.lons, new.lats, new.depths = get_geom(
            rupture.surface, iffs, ims)
        new.surface = rupture.surface
        new.strike = rupture.surface.get_strike()
        new.dip = rupture.surface.get_dip()
        new.rake = rupture.rake
        new.hypocenter = rupture.hypocenter
        new.tectonic_region_type = rupture.tectonic_region_type
        new.magnitude = new.mag = rupture.mag
        new.top_left_corner = None if iffs or ims else (
            new.lons[0], new.lats[0], new.depths[0])
        new.top_right_corner = None if iffs or ims else (
            new.lons[1], new.lats[1], new.depths[1])
        new.bottom_left_corner = None if iffs or ims else (
            new.lons[2], new.lats[2], new.depths[2])
        new.bottom_right_corner = None if iffs or ims else (
            new.lons[3], new.lats[3], new.depths[3])
        return new

    def __lt__(self, other):
        return self.tag < other.tag


@parallel.litetask
def compute_ruptures(sources, sitecol, siteidx, rlzs_assoc, monitor):
    """
    :param sources:
        List of commonlib.source.Source tuples
    :param sitecol:
        a :class:`openquake.hazardlib.site.SiteCollection` instance
    :param siteidx:
        always equal to 0
    :param rlzs_assoc:
        a :class:`openquake.commonlib.source.RlzsAssoc` instance
    :param monitor:
        monitor instance
    :returns:
        a dictionary trt_model_id -> [Rupture instances]
    """
    assert siteidx == 0, (
        'siteidx can be nonzero only for the classical_tiling calculations: '
        'tiling with the EventBasedRuptureCalculator is an error')
    # NB: by construction each block is a non-empty list with
    # sources of the same trt_model_id
    trt_model_id = sources[0].trt_model_id
    oq = monitor.oqparam
    sesruptures = []
    calc_times = []

    # Compute and save stochastic event sets
    for src in sources:
        t0 = time.time()
        s_sites = src.filter_sites_by_distance_to_source(
            oq.maximum_distance, sitecol)
        if s_sites is None:
            continue

        num_occ_by_rup = sample_ruptures(
            src, oq.ses_per_logic_tree_path, rlzs_assoc.csm_info)
        # NB: the number of occurrences is very low, << 1, so it is
        # more efficient to filter only the ruptures that occur, i.e.
        # to call sample_ruptures *before* the filtering
        for rup, rups in build_ses_ruptures(
                src, num_occ_by_rup, s_sites, oq.maximum_distance, sitecol):
            sesruptures.extend(rups)
        dt = time.time() - t0
        calc_times.append((src.id, dt))
    res = AccumDict({trt_model_id: sesruptures})
    res.calc_times = calc_times
    return res


def sample_ruptures(src, num_ses, info):
    """
    Sample the ruptures contained in the given source.

    :param src: a hazardlib source object
    :param num_ses: the number of Stochastic Event Sets to generate
    :param info: a :class:`openquake.commonlib.source.CompositionInfo` instance
    :returns: a dictionary of dictionaries rupture ->
              {(col_id, ses_id): num_occurrences}
    """
    col_ids = info.col_ids_by_trt_id[src.trt_model_id]
    # the dictionary `num_occ_by_rup` contains a dictionary
    # (col_id, ses_id) -> num_occurrences
    # for each occurring rupture
    num_occ_by_rup = collections.defaultdict(AccumDict)
    # generating ruptures for the given source
    for rup_no, rup in enumerate(src.iter_ruptures()):
        rup.seed = seed = src.seed[rup_no]
        numpy.random.seed(seed)
        for col_id in col_ids:
            for ses_idx in range(1, num_ses + 1):
                num_occurrences = rup.sample_number_of_occurrences()
                if num_occurrences:
                    num_occ_by_rup[rup] += {
                        (col_id, ses_idx): num_occurrences}
        rup.rup_no = rup_no + 1
    return num_occ_by_rup


def build_ses_ruptures(
        src, num_occ_by_rup, s_sites, maximum_distance, sitecol):
    """
    Filter the ruptures stored in the dictionary num_occ_by_rup and
    yield pairs (rupture, <list of associated SESRuptures>)
    """
    for rup in sorted(num_occ_by_rup, key=operator.attrgetter('rup_no')):
        # filtering ruptures
        r_sites = filter_sites_by_distance_to_rupture(
            rup, maximum_distance, s_sites)
        if r_sites is None:
            # ignore ruptures which are far away
            del num_occ_by_rup[rup]  # save memory
            continue
        indices = r_sites.indices if len(r_sites) < len(sitecol) \
            else None  # None means that nothing was filtered

        # creating SESRuptures
        sesruptures = []
        rnd = random.Random(rup.seed)
        for (col_idx, ses_idx), num_occ in sorted(
                num_occ_by_rup[rup].items()):
            for occ_no in range(1, num_occ + 1):
                tag = 'col=%02d~ses=%04d~src=%s~rup=%d-%02d' % (
                    col_idx, ses_idx, src.source_id, rup.seed, occ_no)
                sesruptures.append(
                    SESRupture(rup, indices, rnd.randint(0, MAX_INT),
                               tag, col_idx))
        if sesruptures:
            yield rup, sesruptures


@base.calculators.add('event_based_rupture')
class EventBasedRuptureCalculator(ClassicalCalculator):
    """
    Event based PSHA calculator generating the ruptures only
    """
    core_task = compute_ruptures
    tags = datastore.persistent_attribute('tags')
    num_ruptures = datastore.persistent_attribute('num_ruptures')
    counts_per_rlz = datastore.persistent_attribute('counts_per_rlz')
    is_stochastic = True

    @staticmethod
    def count_ruptures(ruptures_by_trt_id, trt_model):
        """
        Returns the number of ruptures sampled in the given trt_model.

        :param ruptures_by_trt_id: a dictionary with key trt_id
        :param trt_model: a TrtModel instance
        """
        return sum(
            len(ruptures) for trt_id, ruptures in ruptures_by_trt_id.items()
            if trt_model.id == trt_id)

    def agg_curves(self, acc, val):
        """
        For the rupture calculator, just increment the AccumDict
        trt_id -> ruptures
        """
        acc += val

    def zerodict(self):
        """
        Initial accumulator, a dictionary trt_model_id -> list of ruptures
        """
        smodels = self.rlzs_assoc.csm_info.source_models
        zd = AccumDict((tm.id, []) for smodel in smodels
                       for tm in smodel.trt_models)
        zd.calc_times = []
        return zd

    def send_sources(self):
        """
        Filter, split and set the seed array for each source, then send it the
        workers
        """
        oq = self.oqparam
        self.manager = self.SourceManager(
            self.csm, self.core_task.__func__,
            oq.maximum_distance, self.datastore,
            self.monitor.new(oqparam=oq), oq.random_seed, oq.filter_sources)
        self.manager.submit_sources(self.sitecol)

    def post_execute(self, result):
        """
        Save the SES collection and the array counts_per_rlz
        """
        logging.info('Generated %d SESRuptures',
                     sum(len(v) for v in result.values()))
        nc = self.rlzs_assoc.csm_info.num_collections
        cols = self.rlzs_assoc.csm_info.cols  # pairs (trt_id, idx)
        sescollection = numpy.array([{} for col_id in range(nc)])
        tags = []
        ordinal = 0
        for trt_id in sorted(result):
            for sr in sorted(result[trt_id]):
                sr.ordinal = ordinal
                ordinal += 1
                sescollection[sr.col_id][sr.tag] = sr
                tags.append(sr.tag)
                if len(sr.tag) > 100:
                    logging.error(
                        'The tag %s is long %d characters, it will be '
                        'truncated to 100 characters in the /tags array',
                        sr.tag, len(sr.tag))
        with self.monitor('saving ruptures', autoflush=True):
            self.tags = numpy.array(tags, (bytes, 100))
            for i, (sescol, col) in enumerate(zip(sescollection, cols)):
                nr = len(sescol)
                logging.info('Saving the SES collection #%d with %d ruptures',
                             i, nr)
                key = 'sescollection/trtmod=%s-%s' % tuple(col)
                self.datastore[key] = sescol
                self.datastore.set_attrs(key, num_ruptures=nr)
        with self.monitor('counts_per_rlz'):
            self.num_ruptures = numpy.array(list(map(len, sescollection)))
            self.counts_per_rlz = counts_per_rlz(
                len(self.sitecol), self.rlzs_assoc, sescollection)
            self.datastore['counts_per_rlz'].attrs[
                'gmfs_nbytes'] = get_gmfs_nbytes(
                len(self.sitecol), len(self.oqparam.imtls),
                self.rlzs_assoc, sescollection)


# ######################## GMF calculator ############################ #

def make_gmfs(ses_ruptures, sitecol, imts, gsims,
              trunc_level, correl_model, monitor):
    """
    :param ses_ruptures: a list of SESRuptures
    :param sitecol: a SiteCollection instance
    :param imts: an ordered list of intensity measure type strings
    :param gsims: an order list of GSIM instance
    :param trunc_level: truncation level
    :param correl_model: correlation model instance
    :param monitor: a monitor instance
    :returns: a list of arrays, one for each rupture
    """
    gmfs = []
    ctx_mon = monitor('make contexts')
    gmf_mon = monitor('compute poes')
    for rupture, group in itertools.groupby(
            ses_ruptures, operator.attrgetter('rupture')):
        sesruptures = list(group)
        indices = sesruptures[0].indices
        r_sites = (sitecol if indices is None else
                   site.FilteredSiteCollection(indices, sitecol.complete))
        with ctx_mon:
            computer = calc.gmf.GmfComputer(
                rupture, r_sites, imts, gsims, trunc_level, correl_model)
        with gmf_mon:
            for sr in sesruptures:
                gmf = computer.compute([sr.seed])[0]
                # TODO: change idx to rup_idx, also in hazardlib
                gmf['idx'] = sr.ordinal
                gmfs.append(gmf)
    return gmfs


@parallel.litetask
def compute_gmfs_and_curves(ses_ruptures, sitecol, rlzs_assoc, monitor):
    """
    :param ses_ruptures:
        a list of blocks of SESRuptures of the same SESCollection
    :param sitecol:
        a :class:`openquake.hazardlib.site.SiteCollection` instance
    :param rlzs_assoc:
        a RlzsAssoc instance
    :param monitor:
        a Monitor instance
    :returns:
        a dictionary (trt_model_id, gsim) -> haz_curves and/or
        (trt_model_id, col_id) -> gmfs
   """
    oq = monitor.oqparam
    # NB: by construction each block is a non-empty list with
    # ruptures of the same col_id and therefore trt_model_id
    col_id = ses_ruptures[0].col_id
    trt_id = rlzs_assoc.csm_info.get_trt_id(col_id)
    gsims = rlzs_assoc.get_gsims_by_col()[col_id]
    trunc_level = oq.truncation_level
    correl_model = readinput.get_correl_model(oq)
    tot_sites = len(sitecol.complete)
    num_sites = len(sitecol)
    gmfs = make_gmfs(ses_ruptures, sitecol, oq.imtls, gsims,
                     trunc_level, correl_model, monitor)
    result = {(trt_id, col_id): numpy.concatenate(gmfs)
              if oq.ground_motion_fields else None}
    if oq.hazard_curves_from_gmfs:
        with monitor('bulding hazard curves', measuremem=False):
            duration = oq.investigation_time * oq.ses_per_logic_tree_path * (
                oq.number_of_logic_tree_samples or 1)
            # collect the gmvs by site
            gmvs_by_sid = collections.defaultdict(list)
            for sr, gmf in zip(ses_ruptures, gmfs):
                site_ids = get_site_ids(sr, num_sites)
                for sid, gmv in zip(site_ids, gmf):
                    gmvs_by_sid[sid].append(gmv)
            # build the hazard curves for each GSIM
            for gsim in gsims:
                gs = str(gsim)
                result[trt_id, gs] = to_haz_curves(
                    tot_sites, gmvs_by_sid, gs, oq.imtls,
                    oq.investigation_time, duration)
    return result


def to_haz_curves(num_sites, gmvs_by_sid, gs, imtls,
                  investigation_time, duration):
    """
    :param num_sites: length of the full site collection
    :param gmvs_by_sid: dictionary site_id -> gmvs
    :param gs: GSIM string
    :param imtls: ordered dictionary {IMT: intensity measure levels}
    :param investigation_time: investigation time
    :param duration: investigation_time * number of Stochastic Event Sets
    """
    curves = zero_curves(num_sites, imtls)
    for imt in imtls:
        curves[imt] = numpy.array([
            gmvs_to_haz_curve(
                [gmv[gs][imt] for gmv in gmvs_by_sid[sid]],
                imtls[imt], investigation_time, duration)
            for sid in range(num_sites)])
    return curves


@base.calculators.add('event_based')
class EventBasedCalculator(ClassicalCalculator):
    """
    Event based PSHA calculator generating the ground motion fields and
    the hazard curves from the ruptures, depending on the configuration
    parameters.
    """
    pre_calculator = 'event_based_rupture'
    core_task = compute_gmfs_and_curves
    is_stochastic = True

    def pre_execute(self):
        """
        Read the precomputed ruptures (or compute them on the fly) and
        prepare some empty files in the export directory to store the gmfs
        (if any). If there were pre-existing files, they will be erased.
        """
        super(EventBasedCalculator, self).pre_execute()
        self.sesruptures = []
        gsims_by_col = self.rlzs_assoc.get_gsims_by_col()
        self.datasets = {}
        for col_id, col in enumerate(self.rlzs_assoc.csm_info.cols):
            sescol = self.datastore['sescollection/trtmod=%s-%s' % tuple(col)]
            gmf_dt = gsim_imt_dt(gsims_by_col[col_id], self.oqparam.imtls)
            for tag, sesrup in sorted(sescol.items()):
                sesrup = sescol[tag]
                self.sesruptures.append(sesrup)
            if self.oqparam.ground_motion_fields and sescol:
                self.datasets[col_id] = self.datastore.create_dset(
                    'gmfs/col%02d' % col_id, gmf_dt)

    def combine_curves_and_save_gmfs(self, acc, res):
        """
        Combine the hazard curves (if any) and save the gmfs (if any)
        sequentially; notice that the gmfs may come from
        different tasks in any order.

        :param acc: an accumulator for the hazard curves
        :param res: a dictionary trt_id, gsim -> gmf_array or curves_by_imt
        :returns: a new accumulator
        """
        sav_mon = self.monitor('saving gmfs')
        agg_mon = self.monitor('aggregating hcurves')
        save_gmfs = self.oqparam.ground_motion_fields
        for trt_id, gsim_or_col in res:
            if isinstance(gsim_or_col, int) and save_gmfs:
                with sav_mon:
                    gmfa = res[trt_id, gsim_or_col]
                    dataset = self.datasets[gsim_or_col]
                    dataset.attrs['trt_model_id'] = trt_id
                    dataset.extend(gmfa)
                    self.nbytes += gmfa.nbytes
                    self.datastore.hdf5.flush()
            elif isinstance(gsim_or_col, str):  # aggregate hcurves
                with agg_mon:
                    curves_by_imt = res[trt_id, gsim_or_col]
                    self.agg_dicts(
                        acc, AccumDict({(trt_id, gsim_or_col): curves_by_imt}))
        sav_mon.flush()
        agg_mon.flush()
        return acc

    def execute(self):
        """
        Run in parallel `core_task(sources, sitecol, monitor)`, by
        parallelizing on the ruptures according to their weight and
        tectonic region type.
        """
        oq = self.oqparam
        if not oq.hazard_curves_from_gmfs and not oq.ground_motion_fields:
            return
        monitor = self.monitor(self.core_task.__name__)
        monitor.oqparam = oq
        zc = zero_curves(len(self.sitecol.complete), self.oqparam.imtls)
        zerodict = AccumDict((key, zc) for key in self.rlzs_assoc)
        self.nbytes = 0
        curves_by_trt_gsim = parallel.apply_reduce(
            self.core_task.__func__,
            (self.sesruptures, self.sitecol, self.rlzs_assoc, monitor),
            concurrent_tasks=self.oqparam.concurrent_tasks,
            acc=zerodict, agg=self.combine_curves_and_save_gmfs,
            key=operator.attrgetter('col_id'))
        if oq.ground_motion_fields:
            # sanity check on the saved gmfs size
            expected_nbytes = self.datastore[
                'counts_per_rlz'].attrs['gmfs_nbytes']
            self.datastore['gmfs'].attrs['nbytes'] = self.nbytes
            assert self.nbytes == expected_nbytes, (
                self.nbytes, expected_nbytes)
        return curves_by_trt_gsim

    def post_execute(self, result):
        """
        :param result:
            a dictionary (trt_model_id, gsim) -> haz_curves or an empty
            dictionary if hazard_curves_from_gmfs is false
        """
        oq = self.oqparam
        if not oq.hazard_curves_from_gmfs and not oq.ground_motion_fields:
            return
        if oq.hazard_curves_from_gmfs:
            ClassicalCalculator.post_execute.__func__(self, result)
        if oq.compare_with_classical:  # compute classical curves
            export_dir = os.path.join(oq.export_dir, 'cl')
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
            oq.export_dir = export_dir
            # use a different datastore
            self.cl = ClassicalCalculator(oq, self.monitor)
            self.cl.datastore.parent = self.datastore
            # TODO: perhaps it is possible to avoid reprocessing the source
            # model, however usually this is quite fast and do not dominate
            # the computation
            result = self.cl.run()
            for imt in self.mean_curves.dtype.fields:
                rdiff, index = max_rel_diff_index(
                    self.cl.mean_curves[imt], self.mean_curves[imt])
                logging.warn('Relative difference with the classical '
                             'mean curves for IMT=%s: %d%% at site index %d',
                             imt, rdiff * 100, index)
