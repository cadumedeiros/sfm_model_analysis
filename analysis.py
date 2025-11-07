# analysis.py
import numpy as np
import matplotlib.pyplot as plt

from load_data import facies, nx, ny, nz
from derived_fields import ensure_reservoir, ensure_clusters

def compute_global_metrics(reservoir_facies):
    res_arr = ensure_reservoir(reservoir_facies)
    clusters_arr, _ = ensure_clusters(reservoir_facies)

    total_cells = facies.size
    res_cells = int(res_arr.sum())
    ntg = res_cells / total_cells if total_cells else 0.0

    counts = np.bincount(clusters_arr)
    if counts.size > 0:
        counts[0] = 0

    n_clusters = (counts > 0).sum()
    largest_label = counts.argmax() if counts.size > 0 else 0
    largest_size = int(counts[largest_label]) if counts.size > 0 else 0
    connected_fraction = largest_size / res_cells if res_cells > 0 else 0.0

    return {
        "total_cells": total_cells,
        "res_cells": res_cells,
        "ntg": ntg,
        "n_clusters": n_clusters,
        "largest_label": largest_label,
        "largest_size": largest_size,
        "connected_fraction": connected_fraction,
    }

def compute_directional_percolation(reservoir_facies):
    # usa o mesmo array de clusters que já está no grid
    clusters_arr, _ = ensure_clusters(reservoir_facies)
    clusters_xyz = clusters_arr.reshape((nx, ny, nz), order="F")

    # X
    left = clusters_xyz[0, :, :]
    right = clusters_xyz[-1, :, :]
    left_ids = set(np.unique(left)); left_ids.discard(0)
    right_ids = set(np.unique(right)); right_ids.discard(0)
    x_common = left_ids.intersection(right_ids)

    # Y
    front = clusters_xyz[:, 0, :]
    back = clusters_xyz[:, -1, :]
    f_ids = set(np.unique(front)); f_ids.discard(0)
    b_ids = set(np.unique(back)); b_ids.discard(0)
    y_common = f_ids.intersection(b_ids)

    # Z
    top = clusters_xyz[:, :, 0]
    bottom = clusters_xyz[:, :, -1]
    t_ids = set(np.unique(top)); t_ids.discard(0)
    bo_ids = set(np.unique(bottom)); bo_ids.discard(0)
    z_common = t_ids.intersection(bo_ids)

    return {
        "x_perc": bool(x_common),
        "x_clusters": x_common,
        "y_perc": bool(y_common),
        "y_clusters": y_common,
        "z_perc": bool(z_common),
        "z_clusters": z_common,
    }

def get_cluster_sizes(reservoir_facies):
    """
    Devolve um array 1D só com os tamanhos dos clusters de reservatório.
    """
    _, _ = ensure_reservoir(reservoir_facies), ensure_clusters(reservoir_facies)
    clusters_arr, _ = ensure_clusters(reservoir_facies)

    counts = np.bincount(clusters_arr)
    if counts.size > 0:
        counts[0] = 0  # tira o fundo
    cluster_sizes = counts[counts > 0]
    return cluster_sizes

def plot_cluster_histogram(reservoir_facies, bins=30):
    """
    Plota o histograma reaproveitando o que tinha no hist_clusters.py
    """
    cluster_sizes = get_cluster_sizes(reservoir_facies)

    print(f"Nº de clusters: {cluster_sizes.size}")
    if cluster_sizes.size:
        print(f"Tamanho mínimo: {cluster_sizes.min()}")
        print(f"Tamanho máximo: {cluster_sizes.max()}")

    plt.figure()
    plt.hist(cluster_sizes, bins=bins)
    plt.xlabel("Tamanho do cluster (nº de células)")
    plt.ylabel("Frequência")
    plt.title("Histograma de tamanhos de cluster")
    plt.tight_layout()
    plt.show()