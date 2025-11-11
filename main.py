# main.py

import visualize
import pyvista as pv

from analysis import (compute_global_metrics, 
                      compute_directional_percolation, 
                      plot_cluster_histogram, 
                      print_facies_metrics, 
                      export_facies_metrics_to_excel, 
                      add_local_thickness_of_facies, 
                      make_thickness_2d_from_grid,
                      add_local_thickness_of_facies_all_clusters
)
from derived_fields import ensure_reservoir
from local_windows import compute_local_ntg
from visualize import show_thickness_2d

def main():
    
    RESERVOIR_FACIES = {23}
    
    MODE = "thickness_local"  # "facies", "reservoir", "clusters", "largest", "ntg_local", "thickness_local"

    # Se MODE = "thickness_local"
    THICKNESS_MODE = "largest"  # "all_clusters" ou "largest"

    Z_EXAG = 15.0
    SHOW_SCALAR_BAR = True
    
    
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

    metrics = compute_global_metrics(RESERVOIR_FACIES)

    # print_facies_metrics()
    # export_facies_metrics_to_excel()

    if MODE == "thickness_local":
        if THICKNESS_MODE == "all_clusters":
            add_local_thickness_of_facies_all_clusters(RESERVOIR_FACIES)
        else:
            add_local_thickness_of_facies(RESERVOIR_FACIES)
    
        surf = make_thickness_2d_from_grid("thickness_local", "thickness_2d")
        show_thickness_2d(surf, "thickness_2d")

    visualize.run(mode=MODE, z_exag=Z_EXAG, show_scalar_bar=SHOW_SCALAR_BAR)
    # plot_cluster_histogram(RESERVOIR_FACIES, bins=30)
    

if __name__ == "__main__":
    main()
