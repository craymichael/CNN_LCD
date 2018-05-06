# =====================================================================
# cnn_lcd.py - CNNs for loop-closure detection in vSLAM systems.
# Copyright (C) 2018  Zach Carmichael
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# =====================================================================
from __future__ import (print_function, division, unicode_literals,
                        absolute_import)

import argparse
import os
import sys

import numpy as np
from scipy.signal import medfilt
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             precision_score, recall_score)
from sklearn.cluster import KMeans

# local imports
from cnn_models import get_model_features, is_valid_model
from dataset import get_dataset

import matplotlib.pyplot as plt
if 'DISPLAY' not in os.environ.keys():
    import matplotlib as mpl

    mpl.use('Agg')
    del mpl

plt.rcParams.update({'font.size': 12,
                     'font.family': 'Times New Roman'})


def get_descriptors(imgs, feat_func, pca=True, pca_dim=500, eps=1e-5,
                    cache=None, name=''):
    """ Returns feature descriptor vector for given image(s). Method follows
    procedure adapted from Zhang et al.:
        'Loop Closure Detection for Visual SLAM Systems Using Convolutional
         Neural Network'

    Args:
        imgs:      Iterable of images each of shape (h,w,c)
        feat_func: Function that takes imgs and cache as arguments, and
                   returns CNN features and cache
        pca:       Whether to perform PCA (and whitening)
        pca_dim:   Dimension to reduce vectors to
        eps:       Small value to prevent division by 0
        cache:     Dict containing descs/other cached values
        name:      Name used for caching
    """
    if cache is None:
        cache = {}
    if name + 'pdescs' not in cache:
        # Get features from network
        descs, cache = feat_func(imgs, cache)
        # Ensure features as vectors
        descs = descs.reshape(len(imgs), -1)
        print(descs.shape)
        # L2 Norm
        descs = descs / np.linalg.norm(descs, axis=1)[:, None]
        cache.update({name + 'pdescs': descs})
    else:
        descs = cache[name + 'pdescs']
    if pca:
        print('Performing PCA with pca_dim={}'.format(pca_dim))
        descs, cache = pca_red(descs, pca_dim, eps=eps, whiten=True,
                               cache=cache, name=name)
        print('PCA done.')
    return descs, cache


def pca_red(descs, dim, eps=1e-5, whiten=True, cache=None, name=''):
    """ Performs PCA + whitening on image descriptors

    Args:
        descs:  input matrix of image descriptors
        dim:    the number of principal components to reduce descs to
        eps:    small epsilon to avoid 0-division
        whiten: whether to whiten the principal components
        cache:  PCA cache (see name parameter)
        name:   used to differentiate different cached value between models

    Returns:
        descs:  the descs post-reduction
        cache:  the (updated) cache
    """
    if cache is None:
        cache = {}
    # Zero-center data
    dmean = descs.mean(axis=0)
    descs = descs - dmean
    if name + 'S' not in cache or name + 'U' not in cache:
        # Compute covariance matrix
        cov = descs.T.dot(descs) / (descs.shape[0] - 1)
        # Apply SVD
        U, S, W = np.linalg.svd(cov)
        cache.update({name + 'U': U, name + 'S': S})
    else:
        U, S = cache[name + 'U'], cache[name + 'S']
    # Project onto principal axes
    descs = descs.dot(U[:, :dim])
    # Whiten
    if whiten:
        descs = descs / np.sqrt(S[:dim] + eps)
    return descs, cache


def cluster_kmeans(sim):
    """Run k-means on similarity matrix and segment"""
    sim_dim = sim.shape[0]
    sim = sim.reshape(-1, 1)

    # Augment with spatial coordinates
    sim_aug = np.concatenate(
        [sim,
         np.mgrid[:sim_dim, :sim_dim].reshape(-1, sim_dim ** 2).T],
        axis=1
    )

    # Empirical metric for number of loop-closures given number of images
    # in sequence (assumption: equally-spaced samples):
    n_clusters = int(np.sqrt(sim_dim))
    print('Performing clustering via KMeans(n={}).'.format(n_clusters))

    km = KMeans(n_clusters=n_clusters, n_jobs=2,
                max_iter=300)
    labels = km.fit_predict(sim_aug)
    print('Got cluster labels')

    for i in range(n_clusters):
        lab_idx = (labels == i)
        if lab_idx.size:
            cc = sim[lab_idx].mean()
            # cc = sim[lab_idx].max()
            sim[lab_idx] = cc

    # Re-normalize and reshape
    sim = sim.reshape(sim_dim, sim_dim) / sim.max()
    return sim


def median_filter(sim, gt, k_size=None):
    """ Apply median filtering and tune kernel size if applicable.

    Args:
        sim:    The similarity matrix
        gt:     The ground truth matrix
        k_size: The square kernel size

    Returns:
        sim: filtered similarity matrix
    """
    # NOTE: only lower triangular part of matrix actually requires filtering
    tri_idx = np.tril_indices(gt.shape[0], -1)

    if k_size is None:
        print('Sweeping median kernel sizes.')
        best_ks = None
        # Compute baseline AP (no median filtering)
        best_ap = average_precision_score(gt[tri_idx], sim[tri_idx])
        best_sim = sim
        k_sizes = list(range(1, 61, 2))
        # Compute similarity matrix
        for ks in k_sizes:
            sim_filtered = medfilt(sim, kernel_size=ks)
            # Re-normalize
            sim_filtered = sim_filtered / sim_filtered.max()
            ks_ap = average_precision_score(gt[tri_idx], sim_filtered[tri_idx])
            if ks_ap > best_ap:
                best_ks = ks
                best_ap = ks_ap
                best_sim = sim_filtered
            print('Finished with ks={} (AP={}). Best so far={}'.format(ks, ks_ap, best_ks))
        print('Best ks={} yielded an AP of {}%.'.format(best_ks, best_ap * 100))
        sim = best_sim
    else:
        print('Filtering with median kernel with kernel size {}.'.format(k_size))
        sim = medfilt(sim, kernel_size=k_size)

        # Re-normalize
        sim = sim / sim.max()
        ap = average_precision_score(gt[tri_idx], sim[tri_idx])
        print('Median filter with ks={} on sim yielded an AP of {}%.'.format(k_size, ap * 100))

    return sim


def similarity_matrix(descs, gt, median=True, cluster=False, plot=False,
                      k_size=None, name=''):
    """ Compute pairwise similarity between descriptors. Using provided gt to find best
    parameters given function args.

    Args:
        descs:   feature descriptors of shape (n, d)
        gt:      the ground truth
        median:  whether to use median filtering (chooses median value that obtains
                 highest avg precision...
        cluster: whether to cluster
        plot:    whether to plot matrix
        k_size:  specify None to sweep, otherwise the value to use
        name:    name for plot file+cache
    """
    print('Computing similarity matrix...')
    n = descs.shape[0]
    diffs = np.zeros((n, n))

    # Compute L2 norm of each vector
    norms = np.linalg.norm(descs, axis=1)
    descs_norm = descs / norms[:, None]

    # Compute similarity of every vector with every vector
    for i, desc in enumerate(descs):
        # Compute difference
        diff = np.linalg.norm(descs_norm - descs_norm[i], axis=1)
        diffs[i] = diff

    # Compute max difference
    dmax = diffs.max()

    # Normalize difference and create sim matrix
    sim = 1. - (diffs / dmax)
    assert gt.shape[0] == sim.shape[0]

    if cluster:
        sim = cluster_kmeans(sim)

    if median:
        sim = median_filter(sim, gt, k_size=k_size)

    if plot:
        f, ax = plt.subplots()
        cax = ax.imshow(sim, cmap='coolwarm', interpolation='nearest',
                        vmin=0., vmax=1.)
        cbar = f.colorbar(cax, ticks=[0, 0.5, 1])
        cbar.ax.set_yticklabels(['0', '0.5', '1'])
        plt.savefig('simplot_{}.png'.format(name), format='png', dpi=150)
        plt.show()

        # Preprocess gt...
        gt = gt.copy()
        gt += gt.T  # add transpose
        gt += np.eye(gt.shape[0], dtype=gt.dtype)

        # Plot
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(sim, cmap='coolwarm', interpolation='nearest',
                     vmin=0., vmax=1.)
        ax[0].set_axis_off()
        ax[0].set_title('Similarity Matrix')
        ax[1].imshow(gt, cmap='gray', interpolation='nearest',
                     vmin=0., vmax=1.)
        ax[1].set_axis_off()
        ax[1].set_title('Ground Truth')
        plt.savefig('simplot_w_gt_{}.png'.format(name), format='png', dpi=150)
        plt.show()

    return sim


def mean_per_class_accuracy(y_true, y_pred, n_classes=None, labels=None):
    """ Computes mean per-class accuracy

    Args:
        y_true:    the true labels
        y_pred:    the predicted labels
        n_classes: the number of classes, optional. If not provided, the number of
                   unique classes or length of `labels` if provided.
        labels:    the unique labels, optional. If not provided, unique labels are used
                   if `n_classes` not provided, otherwise range(n_classes).

    Returns:
        mean per-class accuracy
    """
    if n_classes is None:
        if labels is None:
            labels = np.unique(y_true)
        n_classes = len(labels)
    elif labels is None:
        labels = np.arange(n_classes)
    elif len(labels) != n_classes:
        raise ValueError('Number of classes specified ({}) differs from '
                         'number of labels ({}).'.format(n_classes, len(labels)))
    acc = 0.
    for c in labels:
        c_mask = (y_true == c)
        c_count = c_mask.sum()
        if c_count:  # Avoid division by 0
            # Add accuracy for class c
            acc += np.logical_and(c_mask, (y_pred == c)).sum() / c_count
    # Mean accuracy per class
    return acc / n_classes


def compute_and_plot_scores(sim, gt, model_name):
    """ Computes relevant metrics and plots results.

    Args:
        sim:        Similarity matrix
        gt:         Ground truth matrix
        model_name: Name of the model for logging
    """
    # Modify sim matrix to get "real" vector of loop-closures
    # symmetric matrix, take either diagonal matrix, rid diagonal
    sim = sim[np.tril_indices(sim.shape[0], -1)]

    # Ground truth only present in lower diagonal for Oxford datasets
    gt = gt[np.tril_indices(gt.shape[0], -1)]

    # Compute PR-curve
    precision, recall, thresholds = precision_recall_curve(gt, sim)
    average_precision = average_precision_score(gt, sim)
    print('Average Precision: {}'.format(average_precision))

    best_macc = 0.
    best_mthresh = None
    # Compute the best MPC-accuracy at hard-coded thresholds
    thresholds = np.arange(0, 1.02, 0.02)
    for thresh in thresholds:
        sim_thresh = np.zeros_like(sim)
        sim_thresh[sim >= thresh] = 1
        macc = mean_per_class_accuracy(gt, sim_thresh, n_classes=2)
        if macc > best_macc:
            best_macc = macc
            best_mthresh = thresh

    sim_mthresh = np.zeros_like(sim)
    sim_mthresh[sim >= best_mthresh] = 1
    precision_at_mthresh = precision_score(gt, sim_mthresh)
    recall_at_mthresh = recall_score(gt, sim_mthresh)
    print('Best MPC-ACC (thresh={}): {}'.format(best_mthresh, best_macc))
    print('Precision (thresh={}): {}'.format(best_mthresh, precision_at_mthresh))
    print('Recall (thresh={}): {}'.format(best_mthresh, recall_at_mthresh))

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.3f}'.format(
        average_precision))
    plt.savefig('precision-recall_curve_{}.png'.format(model_name),
                format='png', dpi=150)
    plt.show()


def main(args):
    model_name = args.model

    # Check specified model
    if is_valid_model(model_name):
        # Create weights path
        weights_path = os.path.join(args.weights_dir, args.weights_base)
        weights_path = weights_path.format(args.overfeat)
        # Create feature function
        feat_func = lambda _imgs, _cache: get_model_features(_imgs, model_name,
                                                             overfeat_weights_path=weights_path,
                                                             overfeat_typ=args.overfeat, layer=args.layer,
                                                             cache=_cache)
    else:
        print('Unknown model type: {}'.format(model_name))
        sys.exit(1)

    # Load dataset
    imgs, gt = get_dataset(args.dataset, args.debug)
    if args.plot_gt:
        plt.figure()
        plt.imshow(gt, cmap='gray', interpolation='nearest')
        plt.savefig('{}_gt_plot.png'.format(args.dataset), format='png', dpi=150)
        plt.show()
        sys.exit(0)

    # Compute feature descriptors
    descs, cache = get_descriptors(imgs, feat_func, pca=True, pca_dim=500,
                                   eps=1e-5, cache=None, name=model_name)
    # Kernel sizes for median filter
    if args.sweep_median:
        k_size = None
    elif args.dataset.lower() == 'city':
        k_size = 17  # BEST HARD-CODED PARAMETER FROM SWEEP: ```range(1,61,2)```
    elif args.dataset.lower() == 'college':
        k_size = 11  # BEST HARD-CODED PARAMETER FROM SWEEP: ```range(1,61,2)```
    else:
        k_size = None  # SWEEP

    # Compute similarity matrix
    sim = similarity_matrix(descs, gt, plot=True, cluster=args.cluster, median=True,
                            k_size=k_size, name='_'.join([args.dataset, model_name]))

    assert sim.shape == gt.shape, 'sim and gt not the same shape: {} != {}'.format(sim.shape, gt.shape)

    compute_and_plot_scores(sim, gt, model_name)


if __name__ == '__main__':
    # Parse CLI args
    parser = argparse.ArgumentParser(description='CNNs for loop-closure '
                                                 'detection.')
    parser.add_argument('model', type=str,
                        help='Model name: [overfeat, inception_v{1,2,3,4}, nasnet, resnet_v2_152]')
    parser.add_argument('--dataset', type=str,
                        help='Either "city" or "college".', default='city')
    parser.add_argument('--overfeat', type=int,
                        help='0 for small network, 1 for large', default=1)
    parser.add_argument('--weights_dir', type=str, default='OverFeat/data/default',
                        help='Weights directory.')
    parser.add_argument('--weights_base', type=str, default='net_weight_{}',
                        help='Basename of weights file.')
    parser.add_argument('--layer', type=int, default=None,
                        help='Layer number to extract features from.')
    parser.add_argument('--plot_gt', action='store_true',
                        help='Plots heat-map of ground truth and exits')
    parser.add_argument('--cluster', action='store_true',
                        help='Additionally performs clustering on sim matrix.')
    parser.add_argument('--sweep_median', action='store_true',
                        help='Sweep median filter size values.')
    parser.add_argument('--debug', action='store_true',
                        help='Use small number of images to debug code')
    args = parser.parse_args()

    # Start program
    main(args)
