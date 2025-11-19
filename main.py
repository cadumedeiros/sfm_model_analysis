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

from window import MainWindow
import sys
from PyQt5 import QtWidgets, QtCore

def main():
    
    RESERVOIR_FACIES = {13}
    
    MODE = "facies"  # "facies", "reservoir", "clusters", "largest", "ntg_local", "thickness_local"


    if MODE == "thickness_local":
        visualizar = "Espessura" # "Espessura", "NTG coluna", "NTG envelope", "Maior pacote", "Nº pacotes", "ICV", "Qv"

    Z_EXAG = 10.0
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
    add_vertical_facies_metrics(RESERVOIR_FACIES)

    if MODE == "thickness_local":
        scalar_map = {
            "Espessura": (
                "vert_Ttot_reservoir",
                "Espessura total reservatório (m)",
            ),
            "NTG coluna": (
                "vert_NTG_col_reservoir",
                "NTG coluna (reservatório)",
            ),
            "NTG envelope": (
                "vert_NTG_env_reservoir",
                "NTG envelope (reservatório)",
            ),
            "Maior pacote": (
                "vert_Tpack_max_reservoir",
                "Maior pacote vertical (m)",
            ),
            "Nº pacotes": (
                "vert_n_packages_reservoir",
                "Número de pacotes verticais",
            ),
            "ICV": (
                "vert_ICV_reservoir",
                "Índice de continuidade vertical (ICV)",
            ),
            "Qv": (
                "vert_Qv_reservoir",
                "Índice combinado Qv",
            ),
        }

        scalar_name, title = scalar_map[visualizar]

        # gera o mapa 2D (superfície estruturada) a partir do array 3D
        surf = make_thickness_2d_from_grid(
            array_name_3d=scalar_name,
            array_name_2d=scalar_name + "_2d",
        )
        show_thickness_2d(surf, scalar_name=scalar_name + "_2d")

        # define esse scalar como padrão pro modo "Espessura local" no 3D
        set_thickness_scalar(scalar_name, title=title)


    # 1) Cria a aplicação Qt
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
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
