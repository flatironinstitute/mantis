from matplotlib.collections import LineCollection
import matplotlib.colors as mpl_colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import scipy.io
import seaborn.apionly as sns
from nomad import sdp_km_burer_monteiro, copositive_burer_monteiro
from data import sphere
from experiments.utils import plot_matrix, plot_data_embedded

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'sphere_bm/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def plot_bumps_on_data(X, bumps, palette='Set1', ax=None):
    if ax is None:
        ax = plt.gca()

    # Plot the sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    ax.plot_surface(0.99 * np.outer(np.cos(u), np.sin(v)),
                    0.99 * np.outer(np.sin(u), np.sin(v)),
                    0.99 * np.outer(np.ones(np.size(u)), np.cos(v)),
                    color='w', edgecolors='#AAAAAA', alpha=1)

    plot_data_embedded(X, palette='w', edgecolors='k', ax=ax)

    colors = sns.color_palette(palette, n_colors=len(bumps))
    colors = [mpl_colors.to_hex(c) for c in colors]
    for i, (b, c) in enumerate(zip(bumps, colors)):
        alpha = np.maximum(b, 0) / b.max()
        plot_data_embedded(X, palette=c, alpha=alpha, edgecolors='none', ax=ax)


def plot_bumps_1d(Y, subsampling=20, labels=None, labels_palette='hls',
                  ax=None):
    if ax is None:
        ax = plt.gca()

    Y_subsampled = Y[:, ::subsampling]

    ax.plot(Y_subsampled)
    ax.set_xticks([])

    if labels is not None:
        labels = np.sort(labels)
        unique_labels = np.unique(labels)

        segments = []
        for lab in unique_labels:
            subset = np.where(labels == lab)[0]
            segments.append((subset[0] - 0.5, subset[-1] + 0.5))

        offset = -0.1 * Y_subsampled.max()
        h_segments = [((s[0], offset), (s[1], offset)) for s in segments]

        colors = sns.color_palette(labels_palette, n_colors=len(unique_labels))

        hlc = LineCollection(h_segments, colors=colors)
        hlc.set_linewidth(5)
        hlc.set_clip_on(False)
        ax.add_collection(hlc)


def align_bumps(Y, ref_idx):
    Y_ref = Y[:, ref_idx]

    idx_best = np.zeros((Y.shape[1],), dtype=np.int)
    corr_best = np.zeros((Y.shape[1],))
    for i in range(Y.shape[0]):
        Y_cshift = np.roll(Y, i, axis=0)
        corr = Y_ref.dot(Y_cshift)

        mask = corr > corr_best
        idx_best[mask] = i
        corr_best[mask] = corr[mask]

    Y_aligned = np.zeros_like(Y)
    for j in range(Y.shape[1]):
        Y_aligned[:, j] = np.roll(Y[:, j], idx_best[j], axis=0)

    return Y_aligned


def test_one_circle(n_clusters=25, use_copositive=False):
    X = sphere.generate_sphere_grid()
    print(X.shape)
    labels = np.arange(len(X))

    rank = 4 * n_clusters
    print(rank)
    if use_copositive:
        beta = n_clusters / len(X)
        Y = copositive_burer_monteiro(X, alpha=0.0048, beta=beta, rank=rank,
                                      tol=1e-6, constraint_tol=1e-6,
                                      verbose=True)
        name = 'sphere_copositive_bm'
    else:
        Y = sdp_km_burer_monteiro(X, n_clusters, rank=rank, tol=1e-5,
                                  verbose=True)
        name = 'sphere_sdpkm_bm'

    Q = Y.dot(Y.T)

    idx = np.argsort(np.argmax(Y, axis=0))
    Y = Y[:, idx]

    Y_aligned = align_bumps(Y, Y.shape[1] // 2)

    scipy.io.savemat(name + '.mat', {'X': X, 'Y': Y})

    sns.set_style('white')

    plt.figure(figsize=(12, 5), tight_layout=True)
    gs = gridspec.GridSpec(1, 3)

    ax = plt.subplot(gs[0], projection='3d')
    plot_data_embedded(X, palette='hls', ax=ax)
    ax.set_title('Input dataset', fontsize='xx-large')

    ax = plt.subplot(gs[1])
    plot_matrix(Q, ax=ax, labels=labels, which_labels='both',
                labels_palette='hls')
    plt_title = ax.set_title(r'$\mathbf{Q}$', fontsize='xx-large')
    plt_title.set_position((0.5, 1.07))

    ax = plt.subplot(gs[2])
    plot_matrix(Y, ax=ax, labels=labels, which_labels='vertical',
                labels_palette='hls')
    plt_title = ax.set_title(r'$\mathbf{Y}^\top$', fontsize='xx-large')
    plt_title.set_position((0.5, 1.07))

    plt.savefig('{}{}.pdf'.format(dir_name, name), dpi=300)


if __name__ == '__main__':
    test_one_circle(use_copositive=True)
    test_one_circle(use_copositive=False)
    plt.show()
