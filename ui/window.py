from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QColor, QBrush
from pyvistaqt import BackgroundPlotter
import pyvista as pv
from PyQt5.QtGui import QPixmap
import numpy as np
          
from visualize import run
from load_data import facies
from config import load_facies_colors

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, mode, z_exag, show_scalar_bar, reservoir_facies):
        super().__init__()
        self.setWindowTitle("Suite de Análise e Visualização para Modelos SFM")

        self.resize(1100, 700)

        # Layout base
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        self.setCentralWidget(central)

        # Painel lateral --------------------------------------------------
        self.panel = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(self.panel)

        logo_label = QtWidgets.QLabel()
        pix = QPixmap("assets/forward_PNG.png")   # caminho do seu logo
        pix = pix.scaledToWidth(100, QtCore.Qt.SmoothTransformation)  # ajusta o tamanho
        logo_label.setPixmap(pix)
        logo_label.setAlignment(QtCore.Qt.AlignCenter)

        panel_layout.addWidget(logo_label)

        panel_layout.addSpacing(10)

        # ---- Botões de modos ----
        modes = {
            "Fácies": "facies",
            "Reservatório": "reservoir",
            "Clusters": "clusters",
            "Maior Cluster": "largest",
            "Espessura local": "thickness_local",
            "NTG local": "ntg_local",
        }

        group = QtWidgets.QGroupBox("Modo de Visualização")
        group_layout = QtWidgets.QVBoxLayout(group)

        for label, mode_code in modes.items():
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(lambda _, mm=mode_code: self.change_mode(mm))
            group_layout.addWidget(btn)

        panel_layout.addWidget(group)

        # ---- Submodos thickness_local ----
        self.thick_combo = QtWidgets.QComboBox()
        self.thick_combo.addItems(["Espessura", "NTG coluna", "NTG envelope"])
        self.thick_combo.currentTextChanged.connect(self.change_thickness_mode)

        sub_group = QtWidgets.QGroupBox("Thickness Local")
        sub_layout = QtWidgets.QVBoxLayout(sub_group)
        sub_layout.addWidget(self.thick_combo)
        panel_layout.addWidget(sub_group)

        legend_group = QtWidgets.QGroupBox("Legenda de Fácies")
        legend_layout = QtWidgets.QVBoxLayout(legend_group)

        self.facies_legend_table = QtWidgets.QTableWidget()
        self.facies_legend_table.setColumnCount(2)
        self.facies_legend_table.setHorizontalHeaderLabels(["Cor", "Fácies"])
        self.facies_legend_table.verticalHeader().setVisible(False)
        self.facies_legend_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.facies_legend_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.facies_legend_table.setShowGrid(False)

        # deixa as colunas compactas
        header = self.facies_legend_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)

        self.facies_legend_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.facies_legend_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        legend_layout.addWidget(self.facies_legend_table)

        panel_layout.addWidget(legend_group)

        panel_layout.addStretch()

        # # Plotter PyVista --------------------------------------------------
        # self.plotter = BackgroundPlotter(show=False)
        # layout.addWidget(self.panel, 0)
        # layout.addWidget(self.plotter.interactor, 1)

        layout.addWidget(self.panel, 0)

        # --------- plotter + abas à direita ----------
        self.plotter = BackgroundPlotter(show=False)

        # cria o QTabWidget
        self.tabs = QtWidgets.QTabWidget()

        # Aba 1: Visualização 3D
        self.viz_tab = QtWidgets.QWidget()
        viz_layout = QtWidgets.QVBoxLayout(self.viz_tab)
        viz_layout.addWidget(self.plotter.interactor)
        self.tabs.addTab(self.viz_tab, "Visualização 3D")


        # Aba 2: Métricas
        self.metrics_tab = QtWidgets.QWidget()
        self.metrics_layout = QtWidgets.QVBoxLayout(self.metrics_tab)

        # ---- métricas globais ----
        metrics_group = QtWidgets.QGroupBox("Análises do Modelo")
        mg_layout = QtWidgets.QVBoxLayout(metrics_group)

        self.metrics_text = QtWidgets.QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMinimumHeight(120)

        mg_layout.addWidget(self.metrics_text)
        self.metrics_layout.addWidget(metrics_group)

        # ---- métricas por fácies (Excel) ----
        facies_group = QtWidgets.QGroupBox("Análises por Fácies")
        fg_layout = QtWidgets.QVBoxLayout(facies_group)

        self.facies_table = QtWidgets.QTableWidget()
        fg_layout.addWidget(self.facies_table)

        self.metrics_layout.addWidget(facies_group)
        self.metrics_layout.addStretch()

        self.tabs.addTab(self.metrics_tab, "Métricas Globais")

        # adiciona o conjunto de abas no lado direito
        layout.addWidget(self.tabs, 1)

        self.populate_facies_legend()

        # Carrega visualização inicial
        self.state = dict()
        self.state["reservoir_facies"] = reservoir_facies
        run(mode=mode, z_exag=z_exag, show_scalar_bar=show_scalar_bar, 
            external_plotter=self.plotter, external_state=self.state)
        
    
    def populate_facies_legend(self):
        colors_dict = load_facies_colors()
        present = sorted(set(int(v) for v in np.unique(facies)))

        self.facies_legend_table.setRowCount(len(present))

        for row, fac in enumerate(present):
            rgba = colors_dict.get(fac, (200, 200, 200, 255))
            r, g, b, a = rgba
            if r <= 1 and g <= 1 and b <= 1:
                r, g, b = int(r * 255), int(g * 255), int(b * 255)

            color_item = QtWidgets.QTableWidgetItem()
            color = QColor(r, g, b)
            color_item.setBackground(QBrush(color))
            color_item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.facies_legend_table.setItem(row, 0, color_item)

            text_item = QtWidgets.QTableWidgetItem(str(fac))
            text_item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.facies_legend_table.setItem(row, 1, text_item)

        self.facies_legend_table.resizeColumnsToContents()
        self.facies_legend_table.setColumnWidth(0, 30)

        # --- ajusta altura pra caber tudo sem scroll ---
        header_h = self.facies_legend_table.horizontalHeader().height()
        row_h = self.facies_legend_table.verticalHeader().defaultSectionSize()
        n_rows = self.facies_legend_table.rowCount()
        frame = 2 * self.facies_legend_table.frameWidth()

        total_height = header_h + n_rows * row_h + frame + 4
        self.facies_legend_table.setFixedHeight(total_height)

        # --- ajusta largura pra caber só as duas colunas ---
        self.facies_legend_table.resizeColumnsToContents()
        self.facies_legend_table.setColumnWidth(0, 26)

        header_h = self.facies_legend_table.horizontalHeader().height()
        row_h = self.facies_legend_table.verticalHeader().defaultSectionSize()
        n_rows = self.facies_legend_table.rowCount()
        frame = 2 * self.facies_legend_table.frameWidth()

        # quantas linhas queremos mostrar sem scroll?
        max_visible_rows = 14

        if n_rows <= max_visible_rows:
            # cabe tudo: sem scroll, altura exata
            total_height = header_h + n_rows * row_h + frame + 4
            self.facies_legend_table.setFixedHeight(total_height)
            self.facies_legend_table.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarAlwaysOff
            )
        else:
            # muitas fácies: limita altura e ativa scroll
            total_height = header_h + max_visible_rows * row_h + frame + 4
            self.facies_legend_table.setFixedHeight(total_height)
            self.facies_legend_table.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarAsNeeded
            )

        # largura compacta (como já tínhamos)
        vh = self.facies_legend_table.verticalHeader().width()
        col_w = (
            self.facies_legend_table.columnWidth(0)
            + self.facies_legend_table.columnWidth(1)
        )
        padding = 10
        total_width = col_w + vh + frame + 4 + padding
        self.facies_legend_table.setFixedWidth(total_width)


    # ----------------------------------------------------------------------
    def change_mode(self, new_mode):
        print("Modo:", new_mode)
        self.state["mode"] = new_mode
        self.state["refresh"]()   # função que você já tem no visualize.py


    def change_thickness_mode(self, label):
        print("Thickness:", label)
        self.state["thickness_mode"] = label

        # avisa o visualize.py para atualizar o scalar/clim
        if "update_thickness" in self.state:
            self.state["update_thickness"]()

        # se já estiver nesse modo, redesenha
        if self.state.get("mode") == "thickness_local":
            self.state["refresh"]()

    def set_metrics(self, metrics, perc):
        if metrics is None:
            self.metrics_text.setPlainText("Nenhuma análise calculada.")
            return
        
        def fmt_clusters(arr):
            if arr is None:
                return "[]"
            try:
                return "[" + ", ".join(str(int(c)) for c in arr) + "]"
            except TypeError:
                return "[" + str(int(arr)) + "]"

        lines = []
        lines.append("=== Métricas Globais ===")
        lines.append(f"NTG global           : {metrics['ntg']:.3f}")
        lines.append(f"Total de células     : {metrics['total_cells']}")
        lines.append(f"Células reservatório : {metrics['res_cells']}")
        lines.append(f"Nº de clusters       : {metrics['n_clusters']}")
        lines.append(f"Maior cluster (id)   : {metrics['largest_label']}")
        lines.append(f"Tamanho maior cluster: {metrics['largest_size']} células")
        lines.append(f"Fração conectada     : {metrics['connected_fraction']:.3f}")
        lines.append("")

        if perc is not None:
            x_ok = "Sim" if perc["x_perc"] else "Não"
            y_ok = "Sim" if perc["y_perc"] else "Não"
            z_ok = "Sim" if perc["z_perc"] else "Não"

            lines.append("=== Percolação Direcional ===")
            lines.append(
                f"Xmin→Xmax: {x_ok}   clusters conectados = {fmt_clusters(perc['x_clusters'])}"
            )
            lines.append(
                f"Ymin→Ymax: {y_ok}   clusters conectados = {fmt_clusters(perc['y_clusters'])}"
            )
            lines.append(
                f"Topo→Base: {z_ok}   clusters conectados = {fmt_clusters(perc['z_clusters'])}"
            )

        self.metrics_text.setPlainText("\n".join(lines))

    def set_facies_metrics(self, df):
        """
        Preenche a tabela de 'Análises por Fácies' com o DataFrame facies_metrics.xlsx
        deixando cabeçalhos em português e formatando números.
        """
        # mapeia nomes das colunas -> rótulos bonitos
        pretty = {
            "facies": "Fácies",
            "cells": "Células",
            "fraction": "Fração no grid",
            "n_clusters": "Nº de clusters",
            "largest_label": "ID maior cluster",
            "largest_size": "Tamanho maior cluster (células)",
            "connected_fraction": "Fração conectada",
            "volume_total": "Volume total",
            "volume_largest_cluster": "Volume maior cluster",
            "thickness_largest_cluster": "Espessura maior cluster",
            "Perc_X": "Percolação X",
            "Perc_Y": "Percolação Y",
            "Perc_Z": "Percolação Z",
        }

        self.facies_table.setRowCount(len(df))
        self.facies_table.setColumnCount(len(df.columns))

        headers = [pretty.get(col, col) for col in df.columns]
        self.facies_table.setHorizontalHeaderLabels(headers)

        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                val = df.iloc[i][col]

                # formatação numérica básica
                if isinstance(val, (float, np.floating)):
                    if col in ("fraction", "connected_fraction", "Perc_X", "Perc_Y", "Perc_Z"):
                        text = f"{val:.3f}"
                    elif "thickness" in col:
                        text = f"{val:.2f}"
                    else:
                        text = f"{val:.1f}"
                elif isinstance(val, (int, np.integer)):
                    text = str(int(val))
                else:
                    text = str(val)

                item = QtWidgets.QTableWidgetItem(text)
                self.facies_table.setItem(i, j, item)

        self.facies_table.resizeColumnsToContents()
