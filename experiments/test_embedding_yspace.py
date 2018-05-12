from matplotlib.collections import LineCollection
import matplotlib.colors as mpl_colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import seaborn.apionly as sns
from nomad import spectral_embedding, sdp_km_burer_monteiro,\
    copositive_burer_monteiro, dot_matrix
from data import toy, real
from experiments.utils import plot_matrix, plot_data_embedded, plot_images_embedded

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'embedding_bm/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def compute_yspace(X, n_clusters, use_copositive=False, alpha=0):
    rank = len(X)
    if use_copositive:
        beta = n_clusters / len(X)
        Y = copositive_burer_monteiro(X, alpha=alpha, beta=beta, rank=rank,
                                      tol=1e-6, constraint_tol=1e-6,
                                      verbose=True)
        name = 'copositive_bm'
    else:
        Y = sdp_km_burer_monteiro(X, n_clusters, rank=rank, tol=1e-6,
                                  verbose=True)
        name = 'sdpkm_bm'

    return Y, name


def plot_bumps_on_data(X, bumps, palette='Set1'):
    plot_data_embedded(X, palette='w')

    colors = sns.color_palette(palette, n_colors=len(bumps))
    colors = [mpl_colors.to_hex(c) for c in colors]
    np.random.shuffle(colors)
    for i, (b, c) in enumerate(zip(bumps, colors)):
        alpha = np.maximum(b, 0) / b.max()
        plot_data_embedded(X, palette=c, alpha=alpha)


def plot_bumps_1d(Y, subsampling=20, labels=None, labels_palette='hls',
                  ax=None):
    if ax is None:
        ax = plt.gca()

    ax.plot(Y[:, ::subsampling])
    ax.set_xticks([])

    if labels is not None:
        labels = np.sort(labels)
        unique_labels = np.unique(labels)

        segments = []
        for lab in unique_labels:
            subset = np.where(labels == lab)[0]
            segments.append((subset[0] - 0.5, subset[-1] + 0.5))

        offset = -0.008
        h_segments = [((s[0], offset), (s[1], offset)) for s in segments]

        colors = sns.color_palette(labels_palette, n_colors=len(unique_labels))

        hlc = LineCollection(h_segments, colors=colors)
        hlc.set_linewidth(5)
        hlc.set_clip_on(False)
        ax.add_collection(hlc)


def test_toy_embedding(X, Y, n_clusters, target_dim, filename,
                       palette='hls', elev_azim=None):
    print('--------\n', filename)

    embedding = spectral_embedding(Y, target_dim=target_dim, gramian=False,
                                   discard_first=False)

    D = dot_matrix(X)
    Q = Y.dot(Y.T)
    labels = np.arange(len(X))

    sns.set_style('white')

    plt.figure(figsize=(15, 4), tight_layout=True)
    gs = gridspec.GridSpec(1, 5)

    if X.shape[1] == 3:
        ax = plt.subplot(gs[0], projection='3d')
    else:
        ax = plt.subplot(gs[0])
    plot_data_embedded(X, ax=ax, palette=palette, elev_azim=elev_azim)
    ax.set_title('Input dataset', fontsize='xx-large')

    titles = ['Input Gramian $\mathbf{{D}}$',
              '$\mathbf{{Q}}$ ($K={0}, r={1}$)'.format(n_clusters, Y.shape[1])]
    for i, (M, t) in enumerate(zip([D, Q], titles)):
        ax = plt.subplot(gs[i + 1])
        plot_matrix(M, ax=ax, labels=labels, which_labels='both',
                    labels_palette=palette)
        plt_title = ax.set_title(t, fontsize='xx-large')
        plt_title.set_position((0.5, 1.07))

    ax = plt.subplot(gs[3])
    plot_matrix(Y, ax=ax, labels=labels, which_labels='vertical',
                labels_palette=palette)
    title_y = '$\mathbf{{Y}}$ ($K={0}, r={1}$)'.format(n_clusters, Y.shape[1])
    plt_title = ax.set_title(title_y, fontsize='xx-large')
    plt_title.set_position((0.5, 1.07))

    if target_dim == 2:
        ax = plt.subplot(gs[4])
    if target_dim == 3:
        ax = plt.subplot(gs[4], projection='3d')
    plot_data_embedded(embedding, ax=ax, palette=palette)
    ax.set_title('2D embedding', fontsize='xx-large')

    plt.savefig('{}{}.pdf'.format(dir_name, filename), dpi=300)

    pdf_file_name = '{}{}_plot_{}_on_data_{}{}'
    for i in range(Y.shape[1]):
        plt.figure()
        plot_bumps_on_data(embedding, [Y[:, i]])
        plt.savefig(pdf_file_name.format(dir_name, filename, 'Y', i, '.png'),
                    dpi=300)
        plt.close()

    pdf_file_name = '{}{}_plot_{}_on_data_{}'
    plt.figure()
    plot_bumps_on_data(embedding, [Y[:, i] for i in range(0, Y.shape[1], 10)])
    plt.savefig(pdf_file_name.format(dir_name, filename, 'Y', 'multiple.png'),
                dpi=300)

    pdf_file_name = '{}{}_plot_{}_1d_{}'
    _, ax = plt.subplots(1, 1)
    plot_bumps_1d(Y, subsampling=10, labels=labels, ax=ax)
    plt.savefig(pdf_file_name.format(dir_name, filename, 'Y', '.pdf'),
                dpi=300)


def test_real_embedding(X, Y, n_clusters, target_dim, img_getter,
                        filename, subsampling=10, zoom=.5, labels=None,
                        palette='hls'):
    print('--------\n', filename)

    embedding = spectral_embedding(Y, target_dim=target_dim, gramian=False,
                                   discard_first=False)

    D = dot_matrix(X)
    Q = Y.dot(Y.T)
    point_labels = np.arange(len(X))

    sns.set_style('white')

    plt.figure(figsize=(15, 4), tight_layout=True)
    gs = gridspec.GridSpec(1, 4, wspace=0.)

    titles = ['Input Gramian $\mathbf{{D}}$',
              '$\mathbf{{Q}}$ ($K={0}, r={1}$)'.format(n_clusters, Y.shape[1])]
    for i, (M, t) in enumerate(zip([D, Q], titles)):
        ax = plt.subplot(gs[i])
        plot_matrix(M, ax=ax, labels=point_labels, which_labels='both',
                    labels_palette=palette)
        plt_title = ax.set_title(t, fontsize='xx-large')
        plt_title.set_position((0.5, 1.07))

    ax = plt.subplot(gs[2])
    plot_matrix(Y, ax=ax, labels=point_labels, which_labels='vertical',
                labels_palette=palette)
    title_y = '$\mathbf{{Y}}$ ($K={0}, r={1}$)'.format(n_clusters, Y.shape[1])
    plt_title = ax.set_title(title_y, fontsize='xx-large')
    plt_title.set_position((0.5, 1.07))

    ax = plt.subplot(gs[3])
    plot_images_embedded(embedding, img_getter, labels=labels,
                         subsampling=subsampling,
                         zoom=zoom, palette=palette, ax=ax)

    plt.savefig('{}{}.pdf'.format(dir_name, filename), dpi=300)

    plt.figure()
    plot_images_embedded(embedding, img_getter, labels=labels,
                         subsampling=subsampling,
                         zoom=zoom, palette=palette)
    plt.savefig('{}{}_embedding.pdf'.format(dir_name, filename), dpi=300)

    pdf_file_name = '{}{}_plot_{}_on_data_{}{}'
    for i in range(Y.shape[1]):
        plt.figure()
        plot_bumps_on_data(embedding, [Y[:, i]])
        plt.savefig(pdf_file_name.format(dir_name, filename, 'Y', i, '.png'),
                    dpi=300)
        plt.close()

    pdf_file_name = '{}{}_plot_{}_on_data_{}'
    plt.figure()
    bumps_locs = np.linspace(0, len(X), num=6, endpoint=False, dtype=np.int)
    plot_bumps_on_data(embedding, [Y[:, i] for i in bumps_locs], palette='Set1')
    plt.savefig(pdf_file_name.format(dir_name, filename, 'Y', 'multiple.png'),
                dpi=300)

    pdf_file_name = '{}{}_plot_{}_1d_{}'
    plt.figure()
    _, ax = plt.subplots(1, 1)
    plot_bumps_1d(Y, subsampling=10, labels=point_labels, ax=ax)
    plt.savefig(pdf_file_name.format(dir_name, filename, 'Y', '.pdf'),
                dpi=300)


def test_trefoil():
    X = toy.trefoil_knot(n_samples=200)

    n_clusters = 16

    Y, name = compute_yspace(X, n_clusters, use_copositive=False)
    idx = np.argsort(np.argmax(Y, axis=0))
    Y = Y[:, idx]

    test_toy_embedding(X, Y, n_clusters, 2, 'trefoil_knot_' + name)

    # Y, name = compute_yspace(X, n_clusters, use_copositive=True, alpha=0.005)
    # idx = np.argsort(np.argmax(Y, axis=0))
    # Y = Y[:, idx]
    #
    # test_toy_embedding(X, Y, n_clusters, 2, 'trefoil_knot_' + name)


def test_teapot():
    X = real.teapot()

    def teapot_img(k):
        return X[k, :].reshape((3, 101, 76)).T

    n_clusters = 20

    Y, name = compute_yspace(X, n_clusters, use_copositive=False)
    idx = np.argsort(np.argmax(Y, axis=0))
    Y = Y[:, idx]

    test_real_embedding(X, Y, n_clusters, 2, teapot_img, 'teapot_knot_' + name)

    # X_norm = X.astype(np.float) / np.linalg.norm(X, axis=1, keepdims=True)
    #
    # Y, name = compute_yspace(X_norm, n_clusters, use_copositive=True, alpha=0.02)
    # idx = np.argsort(np.argmax(Y, axis=0))
    # Y = Y[:, idx]
    #
    # test_real_embedding(X, Y, n_clusters, 2, teapot_img, 'teapot_knot_' + name)


def test_mnist(digit=1, n_samples=500, n_clusters=16, subsampling=5):
    X = real.mnist(digit=digit, n_samples=n_samples)
    print('Number of samples:', X.shape[0])

    def mnist_img(k):
        return 255. - X[k, :].reshape((28, 28))

    filename = 'mnist{}_n{}_k{}'.format(digit, n_samples, n_clusters)
    # test_real_embedding(X, n_clusters, 2, mnist_img, filename,
    #                     subsampling=subsampling, zoom=0.3, palette='none')
    Y, name = compute_yspace(X, n_clusters, use_copositive=False)
    idx = np.argsort(np.argmax(Y, axis=0))
    Y = Y[:, idx]

    test_real_embedding(X, Y, n_clusters, 2, mnist_img, filename + name,
                        subsampling=subsampling, zoom=0.3, palette='hsv')


def main():
    test_trefoil()
    test_teapot()
    test_mnist(digit=0)

    plt.show()


if __name__ == '__main__':
    main()