from matplotlib.collections import LineCollection
import matplotlib.colors as mpl_colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pickle
import seaborn.apionly as sns
from mantis import spectral_embedding, sdp_km_burer_monteiro,\
    copositive_burer_monteiro, dot_matrix
from data import toy, real
from experiments.utils import plot_matrix, plot_data_embedded,\
    plot_images_embedded

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'embedding_bm/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def compute_yspace(X, n_clusters, rank=0, use_copositive=False, alpha=0):
    if rank == 0:
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


def plot_bumps_on_data(X, bumps, palette='Set1', ax=None):
    if ax is None:
        ax = plt.gca()

    plot_data_embedded(X, palette='w', edgecolors='k', ax=ax)

    colors = sns.color_palette(palette, n_colors=len(bumps))
    colors = [mpl_colors.to_hex(c) for c in colors]
    np.random.shuffle(colors)
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


def test_toy_embedding(X, Y, target_dim, filename, palette='hls',
                       elev_azim=None):
    print('--------\n', filename)

    embedding = spectral_embedding(Y, target_dim=target_dim, gramian=False,
                                   discard_first=False)

    D = dot_matrix(X)
    Q = Y.dot(Y.T)
    labels = np.arange(len(X))

    sns.set_style('white')

    plt.figure(figsize=(20, 5), tight_layout=True)
    gs = gridspec.GridSpec(1, 5)

    if X.shape[1] == 3:
        ax = plt.subplot(gs[0], projection='3d')
    else:
        ax = plt.subplot(gs[0])
    plot_data_embedded(X, ax=ax, palette=palette, elev_azim=elev_azim)
    ax.set_title('Input dataset', fontsize='xx-large')

    titles = [r'Input Gramian $\mathbf{X}^\top \mathbf{X}$',
              r'$\mathbf{Y}^\top \mathbf{Y}$']
    for i, (M, t) in enumerate(zip([D, Q], titles)):
        ax = plt.subplot(gs[i + 1])
        plot_matrix(M, ax=ax, labels=labels, which_labels='both',
                    labels_palette=palette, colorbar_labelsize=15)
        plt_title = ax.set_title(t, fontsize=25)
        plt_title.set_position((0.5, 1.07))

    ax = plt.subplot(gs[3])
    plot_matrix(Y, ax=ax, labels=labels, which_labels='vertical',
                labels_palette=palette, colorbar_labelsize=15)
    title_y = r'$\mathbf{Y}^\top$ ($'
    plt_title = ax.set_title(title_y, fontsize=25)
    plt_title.set_position((0.5, 1.07))

    if target_dim == 2:
        ax = plt.subplot(gs[4])
    if target_dim == 3:
        ax = plt.subplot(gs[4], projection='3d')
    plot_data_embedded(embedding, ax=ax, palette=palette)
    ax.set_title('2D embedding', fontsize=25)
    plt.savefig('{}{}.pdf'.format(dir_name, filename), dpi=300)

    fig = plt.figure()
    if target_dim == 2:
        ax = fig.add_subplot(111)
    if target_dim == 3:
        ax = fig.add_subplot(111, projection='3d')
    plot_data_embedded(embedding, ax=ax, palette=palette)
    plt.savefig('{}{}_embedding.pdf'.format(dir_name, filename), dpi=300)

    # pdf_file_name = '{}{}_plot_{}_on_data_{}{}'
    # for i in range(Y.shape[1]):
    #     plt.figure()
    #     plot_bumps_on_data(embedding, [Y[:, i]])
    #     plt.savefig(pdf_file_name.format(dir_name, filename, 'Y', i, '.png'),
    #                 dpi=300)
    #     plt.close()

    pdf_file_name = '{}{}_plot_{}_on_data_{}'
    plt.figure()
    if target_dim == 2:
        ax = fig.add_subplot(111)
    if target_dim == 3:
        ax = fig.add_subplot(111, projection='3d')
    plot_bumps_on_data(embedding, [Y[:, i] for i in range(0, Y.shape[1], 10)],
                       ax=ax)
    plt.savefig(pdf_file_name.format(dir_name, filename, 'Y', 'multiple.pdf'),
                dpi=300)

    pdf_file_name = '{}{}_plot_{}_1d{}'
    _, ax = plt.subplots(1, 1)
    plot_bumps_1d(Y, subsampling=10, labels=labels, ax=ax)
    ax.set_yticks([])
    ax.set_title(r'Rows of $\mathbf{Y}$', fontsize=25)
    plt.savefig(pdf_file_name.format(dir_name, filename, 'Y', '.pdf'),
                dpi=300)


def test_real_embedding(X, Y, target_dim, img_getter, filename, subsampling=10,
                        zoom=.5, labels=None, palette='hls'):
    print('--------\n', filename)

    embedding = spectral_embedding(Y, target_dim=target_dim, gramian=False,
                                   discard_first=False)

    D = dot_matrix(X)
    Q = Y.dot(Y.T)

    if palette == 'none':
        point_labels = None
    else:
        point_labels = np.arange(len(X))

    sns.set_style('white')

    plt.figure(figsize=(16, 5.1), tight_layout=True)
    gs = gridspec.GridSpec(1, 4)

    titles = [r'$\mathbf{X}^\top \mathbf{X}$',
              r'$\mathbf{Y}^\top \mathbf{Y}$']
    for i, (M, t) in enumerate(zip([D, Q], titles)):
        ax = plt.subplot(gs[i])
        plot_matrix(M, ax=ax, labels=point_labels, which_labels='both',
                    labels_palette=palette, colorbar_labelsize=15)
        plt_title = ax.set_title(t, fontsize=25)
        plt_title.set_position((0.5, 1.07))

    ax = plt.subplot(gs[2])
    plot_matrix(Y, ax=ax, labels=point_labels, which_labels='vertical',
                labels_palette=palette, colorbar_labelsize=15)
    title_y = r'$\mathbf{Y}^\top$'
    plt_title = ax.set_title(title_y, fontsize=25)
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

    # pdf_file_name = '{}{}_plot_{}_on_data_{}{}'
    # for i in range(Y.shape[1]):
    #     plt.figure()
    #     plot_bumps_on_data(embedding, [Y[:, i]])
    #     plt.savefig(pdf_file_name.format(dir_name, filename, 'Y', i, '.png'),
    #                 dpi=300)
    #     plt.close()

    pdf_file_name = '{}{}_plot_{}_on_data_{}'
    plt.figure()
    # bump_locs = np.linspace(0, Y.shape[1], num=6, endpoint=False, dtype=np.int)
    bump_locs = [1, 6, 31, 40, 43, 47, 55]
    plot_bumps_on_data(embedding, [Y[:, i] for i in bump_locs], palette='Set1')
    # plt.title('Receptive fields', fontsize=25)
    plt.savefig(pdf_file_name.format(dir_name, filename, 'Y', 'multiple.png'),
                dpi=300)

    pdf_file_name = '{}{}_plot_{}_1d{}'
    plt.figure()
    _, ax = plt.subplots(1, 1)
    plot_bumps_1d(Y, subsampling=5, labels=point_labels, ax=ax)
    ax.set_yticks([])
    ax.set_title('Receptive fields', fontsize=25)
    plt.savefig(pdf_file_name.format(dir_name, filename, 'Y', '.pdf'),
                dpi=300)


def test_trefoil():
    X = toy.trefoil_knot(n_samples=200)

    n_clusters = 16

    Y, name = compute_yspace(X, n_clusters, use_copositive=False)
    idx = np.argsort(np.argmax(Y, axis=0))
    Y = Y[:, idx]

    test_toy_embedding(X, Y, n_clusters, 'trefoil_knot_' + name)


def test_teapot():
    X = real.teapot()

    def teapot_img(k):
        return X[k, :].reshape((3, 101, 76)).T

    n_clusters = 20

    Y, name = compute_yspace(X, n_clusters, use_copositive=False)
    idx = np.argsort(np.argmax(Y, axis=0))
    Y = Y[:, idx]

    test_real_embedding(X, Y, n_clusters, teapot_img, 'teapot_' + name)


def test_mnist(digit=1, n_samples=1000, n_clusters=32, subsampling=10,
               from_file=False):
    X = real.mnist(digit=digit, n_samples=n_samples)
    print('Number of samples:', X.shape[0])

    def mnist_img(k):
        im = X[k, :].reshape((28, 28))
        return np.repeat(im[:, :, np.newaxis], 3, axis=-1)

    filename = 'mnist{}_n{}_k{}'.format(digit, n_samples, n_clusters)

    if not from_file:
        Y, name = compute_yspace(X, n_clusters, rank=4 * n_clusters,
                                 use_copositive=False)
        idx = np.argsort(np.argmax(Y, axis=0))
        Y = Y[:, idx]

        with open('mnist_Y.pickle', 'wb') as f:
            pickle.dump(Y, f)
            pickle.dump(name, f)
    else:
        with open('mnist_Y.pickle', 'rb') as f:
            Y = pickle.load(f)
            name = pickle.load(f)

    test_real_embedding(X, Y, 2, mnist_img, filename + name,
                        subsampling=subsampling, zoom=0.5, palette='none')


def test_yale_faces(subjects=[1], n_clusters=16):
    X, gt = real.yale_faces(subjects=subjects)

    def yale_img(k):
        return X[k, :].reshape((192, 168))

    Y, name = compute_yspace(X, n_clusters, rank=0,
                             use_copositive=False)

    filename = 'yale' + '-'.join([str(s) for s in subjects])
    test_real_embedding(X, Y, 2, yale_img, filename + '_' + name,
                        subsampling=3, zoom=0.1, labels=gt, palette='none')


def main():
    test_trefoil()
    test_teapot()
    test_mnist(digit=0, from_file=False)

    test_yale_faces(subjects=[1])
    test_yale_faces(subjects=[1, 4])
    test_yale_faces(subjects=[1, 4, 5])
    test_yale_faces(subjects=[1, 4, 37])
    test_yale_faces(subjects=[1, 4, 5, 27])


    plt.show()


if __name__ == '__main__':
    main()