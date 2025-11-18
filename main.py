# main.py
import pandas as pd

from analysis import (compute_global_metrics, 
                      compute_directional_percolation, 
                      export_facies_metrics_to_excel,
                      make_thickness_2d_from_grid,
                      add_vertical_facies_metrics,
)
from derived_fields import ensure_reservoir
from local_windows import compute_local_ntg
from visualize import show_thickness_2d, set_thickness_scalar

from ui.window import MainWindow
import sys
from PyQt5 import QtWidgets

def main():
    
    RESERVOIR_FACIES = {0}
    
    MODE = "facies"  # "facies", "reservoir", "clusters", "largest", "ntg_local", "thickness_local"


    if MODE == "thickness_local":
        visualizar = "Espessura" # "Espessura", "NTG coluna", "NTG envelope"

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
    export_facies_metrics_to_excel()
    facies_df = pd.read_excel("results/facies_metrics.xlsx")

# --------------------------------------------------------------------------
    RESERVOIR_FACIES = RESERVOIR_FACIES.pop()
    add_vertical_facies_metrics(RESERVOIR_FACIES)

    scalar = f"vert_Ttot_f{RESERVOIR_FACIES}"
    set_thickness_scalar(scalar, title=f"Espessura total fácies {RESERVOIR_FACIES} (m)")
    
    if MODE == "thickness_local":
    
        # 1) ESPESSURA TOTAL
        if visualizar == "Espessura":
            scalar = f"vert_Ttot_f{RESERVOIR_FACIES}"
            surf_Ttot = make_thickness_2d_from_grid(
            array_name_3d=f"vert_Ttot_f{RESERVOIR_FACIES}",
            array_name_2d=f"vert_Ttot_f{RESERVOIR_FACIES}_2d",
        )
            show_thickness_2d(surf_Ttot, scalar_name=f"{scalar}_2d")
            set_thickness_scalar(scalar, title=f"Espessura total fácies {RESERVOIR_FACIES} (m)")

        elif visualizar == "NTG coluna":
            scalar = f"vert_NTG_col_f{RESERVOIR_FACIES}"
            surf_NTG_col = make_thickness_2d_from_grid(
            array_name_3d=f"vert_NTG_col_f{RESERVOIR_FACIES}",
            array_name_2d=f"vert_NTG_col_f{RESERVOIR_FACIES}_2d",
        )
            show_thickness_2d(surf_NTG_col, scalar_name=f"{scalar}_2d")
            set_thickness_scalar(scalar, title=f"NTG coluna fácies {RESERVOIR_FACIES}")   

        elif visualizar == "NTG envelope":
            scalar = f"vert_NTG_env_f{RESERVOIR_FACIES}"
            surf_NTG_env = make_thickness_2d_from_grid(
            array_name_3d=f"vert_NTG_env_f{RESERVOIR_FACIES}",
            array_name_2d=f"vert_NTG_env_f{RESERVOIR_FACIES}_2d",
        )
            show_thickness_2d(surf_NTG_env, scalar_name=f"{scalar}_2d")
            set_thickness_scalar(scalar, title=f"NTG envelope fácies {RESERVOIR_FACIES}")


    # 1) Cria a aplicação Qt
    app = QtWidgets.QApplication(sys.argv)

    # 2) Cria a janela principal
    win = MainWindow(
        mode=MODE,
        z_exag=Z_EXAG,
        show_scalar_bar=SHOW_SCALAR_BAR,
        reservoir_facies=RESERVOIR_FACIES,
    )

    win.set_metrics(metrics, perc)
    win.set_facies_metrics(facies_df)

    win.show()

    # 3) Inicia o loop de eventos
    sys.exit(app.exec_())
    

if __name__ == "__main__":
    main()
