#!/usr/bin/env python

import argparse
import copy
import cPickle as pickle
from dipy.viz.colormap import orient2rgb
import os
import time

import nibabel as nib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KDTree
import sys

from superset import compute_kdt_and_dr, compute_superset_with_k, compute_superset_with_r


### Taken from tractome repo
def streamline2rgb(streamline):
    """Compute orientation of a streamline and retrieve and appropriate RGB
    color to represent it.
    """
    # simplest implementation:
    tmp = orient2rgb(streamline[0] - streamline[-1])
    return tmp


### Taken from tractome repo
def compute_colors(streamlines, alpha):
    """Compute colors for a list of streamlines.
    """
    # assert(type(streamlines) == type([]))
    tot_vertices = np.sum([len(curve) for curve in streamlines])
    color = np.empty((tot_vertices, 4), dtype='f4')
    counter = 0
    for curve in streamlines:
        color[counter:counter +
              len(curve), :3] = streamline2rgb(curve).astype('f4')
        counter += len(curve)
    color[:, 3] = alpha
    return color


### Taken from tractome repo
def compute_buffers(streamlines, alpha, save=False, filename=None):
    """Compute buffers for GL.
    """
    tmp = streamlines
    if type(tmp) is not type([]):
        tmp = streamlines.tolist()
    streamlines_buffer = np.ascontiguousarray(np.concatenate(tmp).astype('f4'))
    streamlines_colors = np.ascontiguousarray(
        compute_colors(streamlines, alpha))
    streamlines_count = np.ascontiguousarray(
        np.array([len(curve) for curve in streamlines], dtype='i4'))
    streamlines_first = np.ascontiguousarray(
        np.concatenate([[0], np.cumsum(streamlines_count)[:-1]]).astype('i4'))
    tmp = {
        'buffer': streamlines_buffer,
        'colors': streamlines_colors,
        'count': streamlines_count,
        'first': streamlines_first
    }
    if save:
        print "saving buffers to", filename
        np.savez_compressed(filename, **tmp)
    return tmp


### Modified from tractome repo
def save_info(clusters, buffers, full_dissimilarity_matrix, num_prototypes, kdt,
              filepath):
    """
    Saves all the information from the tractography required for
    the whole segmentation procedure.
    """
    info = {
        'initclusters': clusters,
        'buff': buffers,
        'dismatrix': full_dissimilarity_matrix,
        'nprot': num_prototypes,
        'kdtree': kdt
    }
    print "Saving information of the tractography for the segmentation"
    print filepath
    filedir = os.path.dirname(filepath)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    pickle.dump(info, open(filepath, 'w'), protocol=pickle.HIGHEST_PROTOCOL)


### Taken from tractome repo
def mbkm_wrapper(full_dissimilarity_matrix, n_clusters, streamlines_ids):
    """Wrapper of MBKM with API compatible to the Manipulator.

    streamlines_ids can be set or list.
    """
    sids = np.array(list(streamlines_ids))
    dissimilarity_matrix = full_dissimilarity_matrix[sids]

    print "MBKM clustering time:",
    init = 'random'
    mbkm = MiniBatchKMeans(
        init=init,
        n_clusters=n_clusters,
        batch_size=1000,
        n_init=10,
        max_no_improvement=5,
        verbose=0)
    t0 = time.time()
    mbkm.fit(dissimilarity_matrix)
    t_mini_batch = time.time() - t0
    print t_mini_batch

    print "exhaustive smarter search of the medoids:",
    medoids_exhs = np.zeros(n_clusters, dtype=np.int)
    t0 = time.time()
    idxs = []
    for i, centroid in enumerate(mbkm.cluster_centers_):
        idx_i = np.where(mbkm.labels_ == i)[0]
        if idx_i.size == 0: idx_i = [0]
        tmp = full_dissimilarity_matrix[idx_i] - centroid
        medoids_exhs[i] = sids[idx_i[(tmp * tmp).sum(1).argmin()]]
        idxs.append(set(sids[idx_i].tolist()))

    t_exhs_query = time.time() - t0
    print t_exhs_query, "sec"
    clusters = dict(zip(medoids_exhs, idxs))
    return clusters


def generate_seg(struct_path, tract_path, bundle_path, out_path, **kwargs):
    """Saves the information of the segmentation result from the
    current session.
    """
    out_dir = os.path.dirname(out_path)
    out_fn = os.path.basename(out_path).rsplit('.', 1)[0]

    print "Loading tract"
    T = nib.streamlines.load(tract_path)
    T_sl = T.streamlines

    print "Loading segmented bundle"
    b = nib.streamlines.load(bundle_path)
    b_sl = b.streamlines

    print "Computing superset"
    kdt_T, prototypes, dr_T = compute_kdt_and_dr(T_sl)
    if 'k' in kwargs.keys():
        additional_ids, b_ids = compute_superset_with_k(
            b_sl, kdt_T, prototypes, k=kwargs['k'])
    else:
        additional_ids, b_ids = compute_superset_with_r(
            b_sl, kdt_T, prototypes, r=kwargs['r'])
    ss_ids = b_ids + additional_ids

    print "Saving superset.trk in the current directory"
    ss_path = os.path.join(out_dir, '%s.trk' % out_fn)
    ss = nib.streamlines.Tractogram(T_sl[ss_ids], affine_to_rasmm=np.eye(4))
    dr_ss = dr_T[np.array(ss_ids)]
    kdt_ss = KDTree(dr_ss)
    nib.streamlines.save(ss, ss_path, header=b.header.copy())

    print "Creating initial clusters"
    clusters = mbkm_wrapper(dr_ss, 50, range(len(ss_ids)))

    print "generating simple history"
    print "step 0: cluster initializtion"
    simple_history = [copy.deepcopy(clusters)]
    simple_history_pointer = 0

    print "step 1: reset clusters with the ones of the bundle"
    clusters = mbkm_wrapper(dr_ss, 50, range(len(b_ids)))
    simple_history.append(copy.deepcopy(clusters))
    simple_history_pointer += 1

    expand = False
    selected = set()

    state = {
        'clusters': clusters,
        'selected': selected,
        'expand': expand,
        'simple_history': simple_history,
        'simple_history_pointer': simple_history_pointer
    }

    seg_info = {
        'structfilename': struct_path,
        'tractfilename': ss_path,
        'segmsession': state
    }

    print "Saving segmentation"
    pickle.dump(seg_info, open(out_path, 'w'), protocol=pickle.HIGHEST_PROTOCOL)

    streams = ss.apply_affine(np.linalg.inv(b.affine)).streamlines
    streams = np.array(streams, dtype=np.object)
    buffers = compute_buffers(streams, alpha=1.0, save=False)
    save_info(clusters, buffers, dr_ss, len(prototypes), kdt_ss,
              '%s/.temp/%s.spa' % (out_dir, out_fn))


if __name__ == '__main__':

    #### ARGUMENT PARSING
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-struct', nargs='?', required=True, help='structural image path')
    parser.add_argument('-T', nargs='?', required=True, help='tract path')
    parser.add_argument(
        '-b', nargs='?', required=True, help='segmented bundle path')
    parser.add_argument(
        '-o', nargs='?', const=0, default='seg.seg', help='out file name')
    parser.add_argument(
        '-r',
        nargs='?',
        const=10,
        help='radius(in mm) of neightborhood in nn, def: 10mm')
    parser.add_argument(
        '-k',
        nargs='?',
        const=300,
        help='k for k-nn in superset computation, def: 300')
    args = parser.parse_args()

    if args.r and args.k:
        sys.exit('options -r and -k are exclusive. Choose one')

    struct_path = os.path.abspath(args.struct)
    T_path = os.path.abspath(args.T)
    b_path = os.path.abspath(args.b)
    o_path = os.path.abspath(args.o)
    import ipdb; ipdb.set_trace()
    if args.k:
        generate_seg(struct_path, T_path, b_path, o_path, k=int(args.k))
    elif args.r:
        generate_seg(struct_path, T_path, b_path, o_path, r=float(args.r))
    else:
        generate_seg(struct_path, T_path, b_path, o_path, r=10)
