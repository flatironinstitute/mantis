from __future__ import absolute_import, print_function
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn.apionly as sns
from nomad import sdp_km_burer_monteiro, copositive_burer_monteiro
from data import toy
from tests.utils import plot_matrix, plot_data_clustered, plot_data_embedded

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'circle_bm/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def plot_bumps_on_data(X, bump):
    plot_data_embedded(X, palette='w')
    alpha = np.maximum(bump, 0) / bump.max()
    plot_data_embedded(X, palette='#FF0000', alpha=alpha)


def test_one_circle(n_clusters=16, use_copositive=False):
    X, gt = toy.circles(n_samples=200)
    X = X[gt == 0, :]
    gt = gt[gt == 0]

    rank = n_clusters * 8
    if use_copositive:
        beta = n_clusters / len(X)
        Y = copositive_burer_monteiro(X, alpha=0.0138, beta=beta, rank=rank)
        name = 'circle_sdpkm_bm'
    else:
        Y = sdp_km_burer_monteiro(X, n_clusters, rank=rank)
        name = 'circle_copositive_bm'

    Q = Y.dot(Y.T)

    idx = np.argsort(np.argmax(Y, axis=0))
    Y = Y[:, idx]

    sns.set_style('white')
    sns.set_color_codes()

    plt.figure(figsize=(12, 6), tight_layout=True)
    gs = gridspec.GridSpec(1, 3, width_ratios=(0.78, 0.78, 1))

    ax = plt.subplot(gs[0])
    plot_data_clustered(X, gt, ax=ax)
    ax.set_title('Input dataset', fontsize='xx-large')

    ax = plt.subplot(gs[1])
    plot_matrix(Q, ax=ax)
    ax.set_title('$\mathbf{{Q}}$', fontsize='xx-large')

    ax = plt.subplot(gs[2])
    plot_matrix(Y, ax=ax)
    ax.set_title('$\mathbf{{Y}}$', fontsize='xx-large')

    plt.savefig('{}{}.pdf'.format(dir_name, name))

    pdf_file_name = '{}{}_plot_{}_on_data_{}{}'

    for i in range(len(X)):
        plt.figure()
        plot_bumps_on_data(X, Y[:, i])
        plt.savefig(pdf_file_name.format(dir_name, name, 'Y', i, '.png'))
        plt.close()


if __name__ == '__main__':
    test_one_circle(use_copositive=True)
    test_one_circle(use_copositive=False)
    plt.show()
