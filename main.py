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
                      add_local_thickness_of_facies_all_clusters,
                      add_vertical_thickness_basic,
                      add_vertical_facies_metrics,
)
from derived_fields import ensure_reservoir
from local_windows import compute_local_ntg
from visualize import show_thickness_2d, set_thickness_scalar

def main():
    
    RESERVOIR_FACIES = {0}
    FACIES_ANALISE = 23
    
    MODE = "reservoir"  # "facies", "reservoir", "clusters", "largest", "ntg_local", "thickness_local"


    if MODE == "thickness_local":
        visualizar = "NTG envelope" # "Espessura", "NTG coluna", "NTG envelope"

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
    compute_local_ntg(res_mask, window=(1, 1, 5)) # (5, 5, 3)

    metrics = compute_global_metrics(RESERVOIR_FACIES)

    # print_facies_metrics()
    export_facies_metrics_to_excel()

    # if MODE == "thickness_local":
    #     if THICKNESS_MODE == "all_clusters":
    #         add_local_thickness_of_facies_all_clusters(RESERVOIR_FACIES)
    #     else:
    #         add_local_thickness_of_facies(RESERVOIR_FACIES)
    
    #     surf = make_thickness_2d_from_grid("thickness_local", "thickness_2d")
    #     show_thickness_2d(surf, "thickness_2d")
    
# --------------------------------------------------------------------------
    add_vertical_facies_metrics(FACIES_ANALISE)

    
    
    # show_thickness_2d(surf_Ttot, scalar_name=f"{scalar}_2d")
# --------------------------------------------------------------------------
    
    if MODE == "thickness_local":
    
        # 1) ESPESSURA TOTAL
        if visualizar == "Espessura":
            scalar = f"vert_Ttot_f{FACIES_ANALISE}"
            surf_Ttot = make_thickness_2d_from_grid(
            array_name_3d=f"vert_Ttot_f{FACIES_ANALISE}",
            array_name_2d=f"vert_Ttot_f{FACIES_ANALISE}_2d",
        )
            show_thickness_2d(surf_Ttot, scalar_name=f"{scalar}_2d")
            set_thickness_scalar(scalar, title=f"Espessura total fácies {FACIES_ANALISE} (m)")

        elif visualizar == "NTG coluna":
            scalar = f"vert_NTG_col_f{FACIES_ANALISE}"
            surf_NTG_col = make_thickness_2d_from_grid(
            array_name_3d=f"vert_NTG_col_f{FACIES_ANALISE}",
            array_name_2d=f"vert_NTG_col_f{FACIES_ANALISE}_2d",
        )
            show_thickness_2d(surf_NTG_col, scalar_name=f"{scalar}_2d")
            set_thickness_scalar(scalar, title=f"NTG coluna fácies {FACIES_ANALISE}")   

        elif visualizar == "NTG envelope":
            scalar = f"vert_NTG_env_f{FACIES_ANALISE}"
            surf_NTG_env = make_thickness_2d_from_grid(
            array_name_3d=f"vert_NTG_env_f{FACIES_ANALISE}",
            array_name_2d=f"vert_NTG_env_f{FACIES_ANALISE}_2d",
        )
            show_thickness_2d(surf_NTG_env, scalar_name=f"{scalar}_2d")
            set_thickness_scalar(scalar, title=f"NTG envelope fácies {FACIES_ANALISE}")


    visualize.run(mode=MODE, z_exag=Z_EXAG, show_scalar_bar=SHOW_SCALAR_BAR)
    # plot_cluster_histogram(RESERVOIR_FACIES, bins=30)
    

if __name__ == "__main__":
    main()
