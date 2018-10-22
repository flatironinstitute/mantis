import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
import seaborn as sns
from mantis import sdp_km_burer_monteiro
from data import toy
from experiments.utils import plot_matrix, plot_data_clustered

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'clustering/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def test_clustering(X, gt, n_clusters, filename, palette='Set1'):
    Y = sdp_km_burer_monteiro(X, n_clusters, rank=len(X), tol=1e-6,
                              verbose=True)
    D = X.dot(X.T)
    Q = Y.dot(Y.T)

    sns.set_style('white')
    plt.figure(figsize=(12, 3.5), tight_layout=True)
    gs = gridspec.GridSpec(1, 4, wspace=0.15)

    ax = plt.subplot(gs[0])
    plot_data_clustered(X, gt, ax=ax, palette=palette)
    ax.set_title('Input dataset', fontsize='xx-large')

    titles = [r'$\mathbf{X}^\top \mathbf{X}$',
              r'$\mathbf{Y}^\top \mathbf{Y}$',
              r'$\mathbf{Y}^\top$'
    ]
    for i, (M, t, wl) in enumerate(zip([D, Q, Y], titles,
                                       ['both', 'both', 'vertical'])):
        ax = plt.subplot(gs[i + 1])
        plot_matrix(M, ax=ax, labels=gt, which_labels=wl,
                    labels_palette=palette, colorbar_labelsize=15)
        ax.set_title(t, fontsize='xx-large')

    plt.tight_layout()
    plt.savefig('{}{}.pdf'.format(dir_name, filename))


if __name__ == '__main__':
    X, gt = toy.gaussian_blobs()
    test_clustering(X, gt, 6, 'gaussian_blobs')

    X, gt = toy.circles()
    test_clustering(X, gt, 16, 'circles')

    X, gt = toy.moons()
    test_clustering(X, gt, 16, 'moons')

    X, gt = toy.double_swiss_roll()
    test_clustering(X, gt, 48, 'double_swiss_roll')

    plt.show()
