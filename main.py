# main.py

import visualize

from analysis import compute_global_metrics, compute_directional_percolation, plot_cluster_histogram
from derived_fields import ensure_reservoir
from local_windows import compute_local_ntg

def main():
    # 1) define aqui quais fácies são reservatório
    RESERVOIR_FACIES = {23}

    # 2) escolhe o modo de visualização e parâmetros
    MODE = "clusters"  # "reservoir", "clusters", "largest", "facies", "ntg_local"
    Z_EXAG = 15.0
    SHOW_SCALAR_BAR = True
    
    # 3) executa as análises e visualização
    metrics = compute_global_metrics(RESERVOIR_FACIES)
    perc = compute_directional_percolation(RESERVOIR_FACIES)

    print("=== MÉTRICAS GLOBAIS ===")
    print(f"Total de células        : {metrics['total_cells']}")
    print(f"Células de reservatório : {metrics['res_cells']}")
    print(f"NTG                     : {metrics['ntg']:.3f}")
    print(f"Nº de clusters (res)    : {metrics['n_clusters']}")
    print(f"Maior cluster (rótulo)  : {metrics['largest_label']}")
    print(f"Tamanho maior cluster   : {metrics['largest_size']} células")
    print(f"Fração conectada        : {metrics['connected_fraction']:.3f}")
    print()

    print("=== PERCOLAÇÃO DIRECIONAL ===")
    print(f"Xmin→Xmax: {perc['x_perc']}  clusters = {perc['x_clusters']}")
    print(f"Ymin→Ymax: {perc['y_perc']}  clusters = {perc['y_clusters']}")
    print(f"Topo→Base: {perc['z_perc']}  clusters = {perc['z_clusters']}")
    print("==============================")

    res_mask = ensure_reservoir(RESERVOIR_FACIES)
    compute_local_ntg(res_mask, window=(3, 3, 3)) # (5, 5, 3)

    visualize.run(mode=MODE, z_exag=Z_EXAG, show_scalar_bar=SHOW_SCALAR_BAR)
    # plot_cluster_histogram(RESERVOIR_FACIES, bins=30)

    metrics = compute_global_metrics(RESERVOIR_FACIES)

if __name__ == "__main__":
    main()
