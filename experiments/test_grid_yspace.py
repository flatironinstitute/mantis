import matplotlib.colors as mpl_colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn.apionly as sns
from nomad import sdp_km_burer_monteiro, copositive_burer_monteiro
from experiments.utils import plot_matrix, plot_data_embedded

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'grid_bm/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def plot_bumps_on_data(X, bumps, palette='Set1'):
    plot_data_embedded(X, palette='w')

    colors = sns.color_palette(palette, n_colors=len(bumps))
    colors = [mpl_colors.to_hex(c) for c in colors]
    np.random.shuffle(colors)
    for i, (b, c) in enumerate(zip(bumps, colors)):
        alpha = np.maximum(b, 0) / b.max()
        plot_data_embedded(X, palette=c, alpha=alpha)


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


def test_grid(n_clusters=16, use_copositive=False):
    X = np.mgrid[0:16, 0:16]
    X = X.reshape((len(X), -1)).T
    labels = np.arange(len(X))

    # X_norm = X - np.mean(X, axis=0)
    # cov = X_norm.T.dot(X_norm)
    # X_norm /= np.trace(cov.dot(cov)) ** 0.25
    #
    # alpha = 0.001
    # plt.matshow(np.maximum(X_norm.dot(X_norm.T) - alpha, 0), cmap='gray_r')
    #
    # from scipy.spatial.distance import pdist, squareform
    # plt.matshow(squareform(pdist(X)), cmap='gray_r')
    #
    # return

    rank = len(X)
    print(rank)
    if use_copositive:
        beta = n_clusters / len(X)
        Y = copositive_burer_monteiro(X, alpha=0.003, beta=beta, rank=rank,
                                      tol=1e-5, constraint_tol=1e-5,
                                      verbose=True)
        name = 'grid_copositive_bm'
    else:
        Y = sdp_km_burer_monteiro(X, n_clusters, rank=rank, tol=1e-6,
                                  verbose=True)
        name = 'grid_sdpkm_bm'

    Q = Y.dot(Y.T)

    idx = np.argsort(np.argmax(Y, axis=0))
    Y = Y[:, idx]

    sns.set_style('white')

    plt.figure(figsize=(12, 4.7), tight_layout=True)
    gs = gridspec.GridSpec(1, 3)

    ax = plt.subplot(gs[0])
    plot_data_embedded(X, palette='hls', ax=ax)
    plt_title = ax.set_title('Input dataset', fontsize='xx-large')
    # plt_title.set_position((0.5, 1.07))

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

    plt.savefig('{}{}.pdf'.format(dir_name, name))

    pdf_file_name = '{}{}_plot_{}_on_data_{}{}'
    for i in range(Y.shape[1]):
        plt.figure()
        plot_bumps_on_data(X, [Y[:, i]])
        plt.savefig(pdf_file_name.format(dir_name, name, 'Y', i, '.png'))
        plt.close()

    pdf_file_name = '{}{}_plot_{}_on_data_{}'
    plt.figure()
    bumps_locs = np.random.random_integers(Y.shape[1], size=6)
    plot_bumps_on_data(X, [Y[:, i] for i in bumps_locs], palette='Set1')
    plt.savefig(pdf_file_name.format(dir_name, name, 'Y', 'multiple.png'),
                dpi=300)

    Y_aligned = align_bumps(Y, Y.shape[1] // 2)

    _, ax = plt.subplots(1, 1)
    plot_matrix(Y_aligned, ax=ax)
    plt_title = ax.set_title(r'Aligned $\mathbf{Y}^\top$', fontsize='xx-large')
    plt_title.set_position((0.5, 1.07))
    plt.savefig('{}{}_Y_aligned_2d.pdf'.format(dir_name, name))

    _, ax = plt.subplots(1, 1)
    ax.plot(Y_aligned)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(r'Aligned $\mathbf{Y}$ rows', fontsize='xx-large')
    plt.savefig('{}{}Y_aligned_1d.pdf'.format(dir_name, name))

    pos = np.arange(len(Y))
    median = np.median(Y_aligned, axis=1)
    mu = np.mean(Y_aligned, axis=1)
    sigma = np.std(Y_aligned, axis=1)

    _, ax = plt.subplots(1, 1)
    plt_mean = ax.plot(pos, mu, color='#377eb8')
    ax.fill_between(pos, np.maximum(mu - sigma, 0), mu + sigma, alpha=0.3,
                    color='#377eb8')
    plt_median = ax.plot(pos, median, '-.', color='#e41a1c')
    ax.set_xticks([])
    ax.set_yticks([])
    plt_aux = ax.fill(np.NaN, np.NaN, '#377eb8', alpha=0.3, linewidth=0)
    ax.legend([(plt_mean[0], plt_aux[0]), plt_median[0]],
              [r'Mean $\pm1$ STD', 'Median'],
              loc='upper left', fontsize='xx-large')
    ax.set_title(r'Aligned $\mathbf{Y}$ columns', fontsize='xx-large')
    plt.savefig('{}{}Y_aligned_1d_summary.pdf'.format(dir_name, name))


if __name__ == '__main__':
    # test_grid(use_copositive=True)
    test_grid(use_copositive=False)
    plt.show()
