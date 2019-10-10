#!/usr/bin/env python
import sys
import os
import argparse
import ConfigParser
import numpy as np
import nibabel as nib
import glob
from time import time
import pickle

from functools import partial
from dissimilarity import compute_dissimilarity
from dipy.tracking.distances import bundles_distances_mam
from distances import parallel_distance_computation
from sklearn.neighbors import KDTree


def compute_superset_with_k(true_bundle, kdt, prototypes, k=1000, distance_func=bundles_distances_mam):
    """Compute a superset of the true target tract with k-NN.
    """
    true_bundle = np.array(true_bundle, dtype=np.object)
    dm_true_bundle = distance_func(true_bundle, prototypes)
    D, I = kdt.query(dm_true_bundle, k=k)
    superset_idx = np.unique(I.flat)
    # recompute the gt streamlines idx
    I = kdt.query_radius(dm_true_bundle, 10e-4)
    gt_idx = []
    for arr in I:
        for e in arr:
            gt_idx.append(e)
    gt_idx = list(set(gt_idx))
    superset_idx = list(set(superset_idx))
    print('gt length: %d' % len(gt_idx))
    additional_idx = list(set(superset_idx) - set(gt_idx))

    return additional_idx, gt_idx

def compute_superset_with_r(true_bundle, kdt, prototypes, r=1, distance_func=bundles_distances_mam):
    """Compute a superset of the true target tract with k-NN.
    """
    true_bundle = np.array(true_bundle, dtype=np.object)
    dm_true_bundle = distance_func(true_bundle, prototypes)
    I = kdt.query_radius(dm_true_bundle, r)
    superset_idx = []
    for arr in I:
        for e in arr:
            superset_idx.append(e)
    superset_idx = list(set(superset_idx))
    # recompute the gt streamlines idx
    I = kdt.query_radius(dm_true_bundle, 10e-4)
    gt_idx = []
    for arr in I:
        for e in arr:
            gt_idx.append(e)
    gt_idx = list(set(gt_idx))
    print('gt length: %d' % len(gt_idx))
    additional_idx = list(set(superset_idx) - set(gt_idx))

    return additional_idx, gt_idx

def compute_kdt_and_dr(tract, num_prototypes=None):
    """Compute the dissimilarity representation of the tract and
    build the kd-tree.
    """
    tract = np.array(tract, dtype=np.object)
    print("Computing dissimilarity matrices...")
    if num_prototypes is None:
        num_prototypes = 40
        print("Using %s prototypes as in Olivetti et al. 2012."
              % num_prototypes)
    else:
        print("Using %s prototypes" % num_prototypes)
    t0 = time()
    distance = partial(parallel_distance_computation,
                            distance=bundles_distances_mam)
    dm_tract, prototype_idx = compute_dissimilarity(tract,
                                                    distance,
                                                    num_prototypes,
                                                    prototype_policy='sff',
                                                    verbose=False)
    print("%s sec." % (time() - t0))
    prototypes = tract[prototype_idx]
    print("Building the KD-tree of tract.")
    kdt = KDTree(dm_tract)
    return kdt, prototypes, dm_tract
