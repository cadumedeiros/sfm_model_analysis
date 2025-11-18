from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QColor, QBrush
from pyvistaqt import BackgroundPlotter
from PyQt5.QtGui import QPixmap
import numpy as np
          
from visualize import run, update_2d_plot
from load_data import facies
from config import load_facies_colors
from analysis import compute_global_metrics, compute_directional_percolation


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, mode, z_exag, show_scalar_bar, reservoir_facies):
        super().__init__()
        self.setWindowTitle("SFM View Analysis")

        if isinstance(reservoir_facies, (int, np.integer)):
            initial_reservoir = {int(reservoir_facies)}
        else:
            # já é iterável (lista, set, etc.)
            initial_reservoir = {int(f) for f in reservoir_facies}

        self.resize(1100, 700)
        self.setMinimumSize(800, 600)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )

        # Layout base
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        self.setCentralWidget(central)

        # Painel lateral --------------------------------------------------
        self.panel = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(self.panel)

        self.panel.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Expanding
        )

        logo_label = QtWidgets.QLabel()
        pix = QPixmap("assets/forward_PNG.png")   # caminho do seu logo
        pix = pix.scaledToWidth(140, QtCore.Qt.SmoothTransformation)  # ajusta o tamanho
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
        self.thick_combo.addItems([
            "Espessura",
            "NTG coluna",
            "NTG envelope",
            "Maior pacote",
            "Nº pacotes",
            "ICV",
            "Qv",
            "Qv absoluto",
        ])
        self.thick_combo.currentTextChanged.connect(self.change_thickness_mode)

        sub_group = QtWidgets.QGroupBox("Thickness Local")
        sub_layout = QtWidgets.QVBoxLayout(sub_group)
        sub_layout.addWidget(self.thick_combo)
        panel_layout.addWidget(sub_group)

        # ---- Seleção de fácies reservatório ----
        self.res_group = QtWidgets.QGroupBox("Fácies do Reservatório")
        res_layout = QtWidgets.QVBoxLayout(self.res_group)

        self.reservoir_list = QtWidgets.QListWidget()
        self.reservoir_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)

        self.reservoir_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        # preencher com fácies existentes no modelo
        present = sorted(set(int(v) for v in np.unique(facies)))
        for fac in present:
            item = QtWidgets.QListWidgetItem(str(fac))
            item.setCheckState(QtCore.Qt.Unchecked)
            item.setData(QtCore.Qt.UserRole, fac)
            self.reservoir_list.addItem(item)

        self.reservoir_list.itemChanged.connect(self.change_reservoir_facies)

        res_layout.addWidget(self.reservoir_list)
        panel_layout.addWidget(self.res_group)

        # --- grupo da legenda ---
        self.legend_group = QtWidgets.QGroupBox("Legenda de Fácies")
        self.legend_group.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Expanding
        )
        legend_layout = QtWidgets.QVBoxLayout(self.legend_group)

        # tabela da legenda
        self.facies_legend_table = QtWidgets.QTableWidget()
        self.facies_legend_table.setColumnCount(2)
        self.facies_legend_table.setHorizontalHeaderLabels(["Cor", "Fácies"])
        self.facies_legend_table.verticalHeader().setVisible(False)
        self.facies_legend_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.facies_legend_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.facies_legend_table.setShowGrid(False)

        header = self.facies_legend_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)

        self.facies_legend_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.facies_legend_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        # --- scroll só para a legenda ---
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)

        scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )

        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        # container interno do scroll
        legend_container = QtWidgets.QWidget()
        legend_container_layout = QtWidgets.QVBoxLayout(legend_container)
        legend_container_layout.setContentsMargins(0, 0, 0, 0)
        legend_container_layout.addWidget(self.facies_legend_table)

        scroll.setWidget(legend_container)

        # adiciona scroll dentro do groupbox (e NUNCA no panel_layout!)
        legend_layout.addWidget(scroll)

        # adiciona o groupbox ao painel
        panel_layout.addWidget(self.legend_group)


        # --------- plotter + abas à direita ----------
        self.plotter = BackgroundPlotter(show=False)

        self.plotter.interactor.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )

        # Plotter 2D para o mapa (aba "Mapa 2D")
        self.plotter_2d = BackgroundPlotter(show=False)
        self.plotter_2d.interactor.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )

        # cria o QTabWidget
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )

        # Aba 1: Visualização 3D
        self.viz_tab = QtWidgets.QWidget()
        viz_layout = QtWidgets.QVBoxLayout(self.viz_tab)
        viz_layout.addWidget(self.plotter.interactor)
        self.tabs.addTab(self.viz_tab, "Visualização 3D")

        # Aba 2: Mapa 2D
        self.map2d_tab = QtWidgets.QWidget()
        map2d_layout = QtWidgets.QVBoxLayout(self.map2d_tab)
        map2d_layout.addWidget(self.plotter_2d.interactor)
        self.tabs.addTab(self.map2d_tab, "Mapa 2D")



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
        layout.addWidget(self.panel)
        layout.addWidget(self.tabs, 1)

        self.populate_facies_legend()

        # Carrega visualização inicial
        self.state = dict()
        self.state["reservoir_facies"] = initial_reservoir

        run(
            mode=mode,
            z_exag=z_exag,
            show_scalar_bar=show_scalar_bar,
            external_plotter=self.plotter,
            external_state=self.state,
        )

        self.update_2d_map()

        for i in range(self.reservoir_list.count()):
            fac = self.reservoir_list.item(i).data(QtCore.Qt.UserRole)
            if fac in initial_reservoir:
                self.reservoir_list.item(i).setCheckState(QtCore.Qt.Checked)
        
    
    def populate_facies_legend(self):
        colors_dict = load_facies_colors()

        # facies presentes no modelo
        present = sorted(set(int(v) for v in np.unique(facies)))

        # conta número de células por fácies
        vals, counts = np.unique(facies.astype(int), return_counts=True)
        count_dict = {int(v): int(c) for v, c in zip(vals, counts)}

        # agora a tabela tem 3 colunas: Cor | Fácies | Células
        self.facies_legend_table.clear()
        self.facies_legend_table.setRowCount(len(present))
        self.facies_legend_table.setColumnCount(3)
        self.facies_legend_table.setHorizontalHeaderLabels(["Cor", "Fácies", "Células"])

        for row, fac in enumerate(present):
            rgba = colors_dict.get(fac, (200, 200, 200, 255))
            r, g, b, a = rgba
            # se vier em 0–1, converte pra 0–255
            if r <= 1 and g <= 1 and b <= 1:
                r, g, b = int(r * 255), int(g * 255), int(b * 255)

            # coluna 0: cor
            color_item = QtWidgets.QTableWidgetItem()
            color = QColor(r, g, b)
            color_item.setBackground(QBrush(color))
            color_item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.facies_legend_table.setItem(row, 0, color_item)

            # coluna 1: id da fácies
            text_item = QtWidgets.QTableWidgetItem(str(fac))
            text_item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.facies_legend_table.setItem(row, 1, text_item)

            # coluna 2: número de células da fácies
            n_cells = count_dict.get(fac, 0)
            cells_item = QtWidgets.QTableWidgetItem(str(n_cells))
            cells_item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.facies_legend_table.setItem(row, 2, cells_item)

        # ajustar larguras das colunas
        self.facies_legend_table.resizeColumnsToContents()
        self.facies_legend_table.setColumnWidth(0, 26)

        # largura compacta
        frame = 2 * self.facies_legend_table.frameWidth()
        vh = self.facies_legend_table.verticalHeader().width()
        col_w = (
            self.facies_legend_table.columnWidth(0)
            + self.facies_legend_table.columnWidth(1)
            + self.facies_legend_table.columnWidth(2)
        )
        padding = 16
        total_width = col_w + vh + frame + 4 + padding
        group_width = total_width + 16  # pequeno extra pros frames/margens

        self.legend_group.setMaximumWidth(group_width)
        if hasattr(self, "res_group"):
            self.res_group.setMaximumWidth(group_width)
        self.facies_legend_table.setFixedWidth(total_width)

    def populate_clusters_legend(self):
        """
        Preenche a mesma tabela de legenda, mas com:
        Cor | Cluster | Células
        usando as infos pré-computadas em visualize.run
        (state['clusters_lut'] e state['clusters_sizes']).
        """
        if not hasattr(self, "state"):
            return

        sizes_dict = self.state.get("clusters_sizes")
        lut = self.state.get("clusters_lut")

        if not sizes_dict or lut is None:
            # se não tiver nada calculado ainda, limpa a tabela
            self.facies_legend_table.clear()
            self.facies_legend_table.setRowCount(0)
            self.facies_legend_table.setColumnCount(0)
            return

        # ordena clusters do maior para o menor (igual ao visualize)
        labels = sorted(sizes_dict.keys(), key=lambda k: sizes_dict[k], reverse=True)

        self.facies_legend_table.clear()
        self.facies_legend_table.setColumnCount(3)
        self.facies_legend_table.setHorizontalHeaderLabels(["Cor", "Cluster", "Células"])
        self.facies_legend_table.setRowCount(len(labels))

        for row, lab in enumerate(labels):
            # pega a cor do LUT (0–1)
            r, g, b, a = lut.GetTableValue(int(lab))
            if r <= 1 and g <= 1 and b <= 1:
                r_i, g_i, b_i = int(r * 255), int(g * 255), int(b * 255)
            else:
                r_i, g_i, b_i = int(r), int(g), int(b)

            # coluna 0: quadradinho de cor
            color_item = QtWidgets.QTableWidgetItem()
            color = QColor(r_i, g_i, b_i)
            color_item.setBackground(QBrush(color))
            color_item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.facies_legend_table.setItem(row, 0, color_item)

            # coluna 1: id do cluster
            id_item = QtWidgets.QTableWidgetItem(str(lab))
            id_item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.facies_legend_table.setItem(row, 1, id_item)

            # coluna 2: número de células
            size_item = QtWidgets.QTableWidgetItem(str(sizes_dict[lab]))
            size_item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.facies_legend_table.setItem(row, 2, size_item)

        # ajustar larguras das colunas
        self.facies_legend_table.resizeColumnsToContents()
        self.facies_legend_table.setColumnWidth(0, 26)

        frame = 2 * self.facies_legend_table.frameWidth()
        vh = self.facies_legend_table.verticalHeader().width()
        col_w = (
            self.facies_legend_table.columnWidth(0)
            + self.facies_legend_table.columnWidth(1)
            + self.facies_legend_table.columnWidth(2)
        )
        padding = 10
        total_width = col_w + vh + frame + 4 + padding
        self.facies_legend_table.setFixedWidth(total_width)



    # ----------------------------------------------------------------------
    def change_mode(self, new_mode):
        print("Modo:", new_mode)
        self.state["mode"] = new_mode

        # Atualiza o tipo de legenda no painel esquerdo
        if new_mode == "clusters":
            self.legend_group.setTitle("Legenda de Clusters")
            self.populate_clusters_legend()
            
        else:
            self.legend_group.setTitle("Legenda de Fácies")
            self.populate_facies_legend()

        # redesenha a visualização 3D
        self.state["refresh"]()


    def change_thickness_mode(self, label):
        print("Thickness:", label)
        self.state["thickness_mode"] = label

        # avisa o visualize.py para atualizar o scalar/clim
        if "update_thickness" in self.state:
            self.state["update_thickness"]()

        # se já estiver nesse modo, redesenha
        if self.state.get("mode") == "thickness_local":
            self.state["refresh"]()

        if self.state.get("mode") in ("ntg_local", "thickness_local"):
            self.state["refresh"]()

        # Atualiza o mapa 2D com as novas métricas
        self.update_2d_map()

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

        # facies atualmente selecionadas (a partir do state)
        selected = None
        if hasattr(self, "state"):
            selected = self.state.get("reservoir_facies")

        if selected:
            facies_str = ", ".join(str(int(f)) for f in sorted(selected))
        else:
            facies_str = "(nenhuma selecionada)"

        lines = []
        lines.append("=== Métricas Globais ===")
        lines.append(f"Fácies de reservatório: {facies_str}")
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

    def change_reservoir_facies(self, item):
        selected = []
        for i in range(self.reservoir_list.count()):
            it = self.reservoir_list.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                selected.append(it.data(QtCore.Qt.UserRole))

        # atualiza o estado global
        self.state["reservoir_facies"] = set(selected)

        print("Reservatório atualizado:", self.state["reservoir_facies"])

        # 1) recalcula arrays de reservatório/clusters no visualize.py
        if "update_reservoir" in self.state:
            self.state["update_reservoir"]()

        # 2) recalcula métricas globais para as fácies selecionadas
        if self.state["reservoir_facies"]:
            metrics = compute_global_metrics(self.state["reservoir_facies"])
            perc = compute_directional_percolation(self.state["reservoir_facies"])
        else:
            metrics = None
            perc = None

        self.set_metrics(metrics, perc)

        # 3) se o modo atual for clusters, atualiza também a legenda de clusters
        if self.state.get("mode") == "clusters":
            self.legend_group.setTitle("Legenda de Clusters")
            self.populate_clusters_legend()

        # 4) redesenha o 3D com os arrays novos
        if "refresh" in self.state:
            self.state["refresh"]()
            
        # 3) Atualiza NTG_local e thickness_local no visualize
        if "update_reservoir_fields" in self.state:
            self.state["update_reservoir_fields"](selected)

        # 4) Se estiver em NTG local ou Espessura local, redesenha o 3D
        if self.state.get("mode") in ("ntg_local", "thickness_local"):
            self.state["refresh"]()

        self.update_2d_map()

    def update_2d_map(self):
        """
        Atualiza o plotter 2D com a métrica vertical selecionada
        no combo "Thickness Local".
        """
        if not hasattr(self, "plotter_2d"):
            return

        presets = self.state.get("thickness_presets") or {}
        mode_label = self.state.get("thickness_mode", "Espessura")

        if mode_label not in presets:
            # tenta cair para Espessura como fallback
            if "Espessura" in presets:
                mode_label = "Espessura"
            else:
                return

        scalar_name, title = presets[mode_label]

        try:
            update_2d_plot(self.plotter_2d, scalar_name, title)
        except Exception as e:
            print("Erro ao atualizar mapa 2D:", e)



