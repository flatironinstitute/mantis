import matplotlib.colors as mpl_colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn.apionly as sns
from mantis import sdp_km_burer_monteiro, copositive_burer_monteiro
from data import toy
from experiments.utils import plot_bumps_1d, plot_matrix, plot_data_embedded

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'circle_bm/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def compute_yspace(n_clusters=8, use_copositive=True):
    np.random.seed(1)

    X, gt = toy.circles(n_samples=200)
    X = X[gt == 0, :]

    rank = len(X)
    print(rank)
    if use_copositive:
        beta = n_clusters / len(X)
        Y = copositive_burer_monteiro(X, alpha=0.01, beta=beta, rank=rank,
                                      tol=1e-5, constraint_tol=1e-5,
                                      verbose=True)
        name = 'circle_copositive_bm'
    else:
        Y = sdp_km_burer_monteiro(X, n_clusters, rank=rank, tol=1e-5,
                                  verbose=True)
        name = 'circle_sdpkm_bm'

    return X, Y, name


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


def test_one_circle(X, Y, name, bump_subsampling=20, cos_freq=0.13,
                    cos_loc=None):
    labels = np.arange(len(X))
    Q = Y.dot(Y.T)

    idx = np.argsort(np.argmax(Y, axis=0))
    Y = Y[:, idx]

    Y_aligned = align_bumps(Y, Y.shape[1] // 2)

    sns.set_style('white')

    plt.figure(figsize=(12, 5), tight_layout=True)
    gs = gridspec.GridSpec(1, 3)

    ax = plt.subplot(gs[0])
    plot_data_embedded(X, palette='hls', ax=ax)
    ax.set_title('Input dataset', fontsize=30)

    ax = plt.subplot(gs[1])
    plot_matrix(Q, ax=ax, labels=labels, which_labels='both',
                labels_palette='hls', colorbar_labelsize=15)
    plt_title = ax.set_title(r'$\mathbf{Y}^\top \mathbf{Y}$', fontsize=25)
    plt_title.set_position((0.5, 1.07))

    ax = plt.subplot(gs[2])
    plot_matrix(Y, ax=ax, labels=labels, which_labels='vertical',
                labels_palette='hls', colorbar_labelsize=15)
    plt_title = ax.set_title(r'$\mathbf{Y}^\top$', fontsize=25)
    plt_title.set_position((0.5, 1.07))

    plt.savefig('{}{}.pdf'.format(dir_name, name), dpi=300)

    # pdf_file_name = '{}{}_plot_{}_on_data_{}{}'
    # for i in range(Y.shape[1]):
    #     plt.figure()
    #     plot_bumps_on_data(X, [Y[:, i]])
    #     plt.savefig(pdf_file_name.format(dir_name, name, 'Y', i, '.png'),
    #                 dpi=300)
    #     plt.close()

    pdf_file_name = '{}{}_plot_{}_on_data_{}'
    plt.figure()
    bumps_locs = np.linspace(0, Y.shape[1], num=6, endpoint=False, dtype=np.int)
    plot_bumps_on_data(X, [Y[:, i] for i in bumps_locs], palette='Set1')
    plt.savefig(pdf_file_name.format(dir_name, name, 'Y', 'multiple.png'),
                dpi=300)

    _, ax = plt.subplots(1, 1)
    plot_matrix(Y_aligned, ax=ax)
    plt_title = ax.set_title('Aligned receptive fields', fontsize=25)
    plt_title.set_position((0.5, 1.07))
    plt.savefig('{}{}_Y_aligned_2d.pdf'.format(dir_name, name), dpi=300)

    _, ax = plt.subplots(1, 1)
    plot_bumps_1d(Y, labels=labels, ax=ax, subsampling=bump_subsampling)
    ax.set_title('Receptive fields', fontsize=25)
    plt.savefig('{}{}Y_1d.pdf'.format(dir_name, name), dpi=300)

    _, ax = plt.subplots(1, 1)
    ax.plot(Y_aligned)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Aligned receptive fields', fontsize=25)
    plt.savefig('{}{}Y_aligned_1d.pdf'.format(dir_name, name), dpi=300)

    pos = np.arange(len(Y))
    # median = np.median(Y_aligned, axis=1)
    mu = np.mean(Y_aligned, axis=1)
    sigma = np.std(Y_aligned, axis=1)

    def cosine(x):
        peak = np.argmax(mu)
        if cos_loc is None:
            cloc = np.mean(np.where(mu == mu[peak])[0])
        else:
            cloc = cos_loc
        cos_bump = mu[peak] * np.cos(cos_freq * (x - cloc))
        cos_bump[mu == 0] = 0
        cos_bump[cos_bump < 0] = 0
        return cos_bump

    _, ax = plt.subplots(1, 1)
    plt_mean = ax.plot(pos, mu, color='#377eb8')
    ax.fill_between(pos, np.maximum(mu - 3 * sigma, 0), mu + 3 * sigma,
                    alpha=0.3, color='#377eb8')
    plt_cosine = ax.plot(pos, cosine(pos), '-.', color='#e41a1c')
    ax.set_xticks([])
    ax.set_yticks([])
    plt_aux = ax.fill(np.NaN, np.NaN, '#377eb8', alpha=0.3, linewidth=0)
    ax.legend([(plt_mean[0], plt_aux[0]), plt_cosine[0]],
              [r'Mean $\pm$ 3 STD', 'Truncated cosine'],
              loc='center left', fontsize=25)
    ax.set_title(r'Receptive fields summary', fontsize=25)
    plt.savefig('{}{}Y_aligned_1d_summary.pdf'.format(dir_name, name), dpi=300)


if __name__ == '__main__':
    X, Y, name = compute_yspace()
    test_one_circle(X, Y, name)

    import scipy.io
    mat = scipy.io.savemat(name + '.mat', dict(X=X, Y=Y))

    # import scipy.io
    # mat = scipy.io.loadmat('circle_online_copos_bumps1000_XY.mat')
    # X = mat['Xord']
    # Y = mat['Yord']
    # print(X.shape, Y.shape)
    #
    # test_one_circle(X, Y, 'circle_copositive_online', bump_subsampling=200,
    #                 cos_freq=0.014)

    plt.show()
