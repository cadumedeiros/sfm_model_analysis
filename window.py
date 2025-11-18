from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QColor, QBrush, QPixmap
from pyvistaqt import BackgroundPlotter
import numpy as np
import os

from visualize import run, update_2d_plot
from load_data import facies, nx, ny, nz, load_facies_from_grdecl
from config import load_facies_colors
from analysis import (
    compute_global_metrics,
    compute_directional_percolation,
    facies_distribution_array,
    reservoir_facies_distribution_array,
    compute_global_metrics_for_array,
)

def make_facies_table():
    table = QtWidgets.QTableWidget()
    table.setColumnCount(4)
    table.setHorizontalHeaderLabels(["Cor", "Fácies", "Células", "Sel."])
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
    table.setMaximumHeight(150)
    table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
    return table


def fill_facies_table(table, facies_array, reservoir_set):
    colors = load_facies_colors()

    vals, counts = np.unique(facies_array.astype(int), return_counts=True)
    count_dict = {int(v): int(c) for v, c in zip(vals, counts)}

    present = sorted(count_dict.keys())

    table.setRowCount(len(present))

    for row, fac in enumerate(present):
        rgba = colors.get(fac, (200/255, 200/255, 200/255, 1.0))
        r, g, b, a = [int(255*c) for c in rgba]

        # coluna 0 — cor
        item_color = QtWidgets.QTableWidgetItem()
        item_color.setBackground(QBrush(QColor(r, g, b)))
        item_color.setFlags(QtCore.Qt.ItemIsEnabled)
        table.setItem(row, 0, item_color)

        # coluna 1 — fácies
        item_f = QtWidgets.QTableWidgetItem(str(fac))
        item_f.setFlags(QtCore.Qt.ItemIsEnabled)
        table.setItem(row, 1, item_f)

        # coluna 2 — células
        item_c = QtWidgets.QTableWidgetItem(str(count_dict[fac]))
        item_c.setFlags(QtCore.Qt.ItemIsEnabled)
        table.setItem(row, 2, item_c)

        # coluna 3 — checkbox
        check = QtWidgets.QTableWidgetItem()
        check.setFlags(check.flags() | QtCore.Qt.ItemIsUserCheckable)
        check.setCheckState(QtCore.Qt.Checked if fac in reservoir_set else QtCore.Qt.Unchecked)
        check.setData(QtCore.Qt.UserRole, fac)
        table.setItem(row, 3, check)

def make_grid_with_exag(facies_array, z_exag):
    from load_data import grid as base_grid
    g = base_grid.copy()
    g.cell_data["Facies"] = facies_array.astype(int)

    pts = g.points.copy()
    pts[:, 2] *= z_exag
    g.points = pts
    return g




class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, mode, z_exag, show_scalar_bar, reservoir_facies):
        super().__init__()
        self.setWindowTitle("SFM View Analysis")

        if isinstance(reservoir_facies, (int, np.integer)):
            initial_reservoir = {int(reservoir_facies)}
        else:
            # já é iterável (lista, set, etc.)
            initial_reservoir = {int(f) for f in reservoir_facies}

        # ---------- MODELOS (BASE e COMPARADO) ----------
        # Base: é o modelo carregado por default (load_data.facies)
        # Compare: será preenchido quando você escolher um GRDECL externo
        self.models = {
            "base": {
                "name": "BaseModel",
                "facies": facies,                    # do load_data
                "reservoir_facies": set(initial_reservoir),
            },
            "compare": {
                "name": None,                        # caminho ou rótulo do modelo comparado
                "facies": None,                      # array 1D de fácies do modelo comparado
                "reservoir_facies": set(),           # fácies de reservatório do modelo comparado
            },
        }


        self.resize(1600, 900)
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

                # ------------------------------------------------------------------
        # TABELAS PARA A ABA DE COMPARAÇÃO (MÉTRICAS)
        # ------------------------------------------------------------------
        # 1) Tabela de métricas globais (base x comparado)
        self.global_compare_table = QtWidgets.QTableWidget()
        self.global_compare_table.setColumnCount(4)
        self.global_compare_table.setHorizontalHeaderLabels(
            ["Métrica", "Base", "Comparado", "Diferença (Comp - Base)"]
        )
        self.global_compare_table.verticalHeader().setVisible(False)
        self.global_compare_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        # 2) Tabela de distribuição de fácies no grid inteiro
        self.facies_compare_table = QtWidgets.QTableWidget()
        self.facies_compare_table.setColumnCount(5)
        self.facies_compare_table.setHorizontalHeaderLabels(
            ["Fácies", "Células Base", "% Base", "Células Comp", "% Comp"]
        )
        self.facies_compare_table.verticalHeader().setVisible(False)
        self.facies_compare_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        # 3) Tabela de distribuição de fácies dentro do reservatório
        self.reservoir_facies_compare_table = QtWidgets.QTableWidget()
        self.reservoir_facies_compare_table.setColumnCount(5)
        self.reservoir_facies_compare_table.setHorizontalHeaderLabels(
            ["Fácies (res)", "Células Base", "% do res Base", "Células Comp", "% do res Comp"]
        )
        self.reservoir_facies_compare_table.verticalHeader().setVisible(False)
        self.reservoir_facies_compare_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)


        # ------------------------------------------------------------------
        # ABA: COMPARAÇÃO (contém duas sub-abas: Métricas e Visualização 3D)
        # ------------------------------------------------------------------
        self.compare_tab = QtWidgets.QWidget()
        comp_layout = QtWidgets.QVBoxLayout(self.compare_tab)

        # ----- Sub-abas internas -----
        self.compare_tabs = QtWidgets.QTabWidget()
        comp_layout.addWidget(self.compare_tabs)

        # ==========================
        # 1) SUB-ABA: MÉTRICAS
        # ==========================
        self.compare_metrics_widget = QtWidgets.QWidget()
        cmp_metrics_layout = QtWidgets.QVBoxLayout(self.compare_metrics_widget)

        # --- HEADERS: Base e Comparado ---
        self.base_model_label = QtWidgets.QLabel("Modelo base: (carregado automaticamente)")
        self.comp_model_label = QtWidgets.QLabel("Modelo comparado: (nenhum)")

        header_layout = QtWidgets.QHBoxLayout()
        header_layout.addWidget(self.base_model_label)
        header_layout.addWidget(self.comp_model_label)
        cmp_metrics_layout.addLayout(header_layout)

        # --- Botão para selecionar modelo comparado ---
        self.select_compare_btn = QtWidgets.QPushButton("Selecionar modelo para comparar...")
        self.select_compare_btn.clicked.connect(self.open_compare_dialog)
        cmp_metrics_layout.addWidget(self.select_compare_btn)

        # --- tabelas existentes (que você já tinha) ---
        # Basta mover aqui as suas três tabelas:
        #   self.global_compare_table
        #   self.facies_compare_table
        #   self.reservoir_facies_compare_table

        # Exemplo:
        cmp_metrics_layout.addWidget(self.global_compare_table)
        cmp_metrics_layout.addWidget(self.facies_compare_table)
        cmp_metrics_layout.addWidget(self.reservoir_facies_compare_table)

        cmp_metrics_layout.addStretch()

        self.compare_tabs.addTab(self.compare_metrics_widget, "Métricas")

        # ==========================
        # 2) SUB-ABA: VISUALIZAÇÃO 3D
        # ==========================
        self.compare_3d_widget = QtWidgets.QWidget()
        cmp3d_layout = QtWidgets.QHBoxLayout(self.compare_3d_widget)

        # criaremos os plotters aqui depois
        # -------------------------------
        # COLUNA ESQUERDA: MODELO BASE
        # -------------------------------
        left_col = QtWidgets.QVBoxLayout()

        title_left = QtWidgets.QLabel("Modelo base")
        title_left.setAlignment(QtCore.Qt.AlignCenter)
        left_col.addWidget(title_left)

        # plotter base
        self.comp_plotter_base = BackgroundPlotter(show=False)
        left_col.addWidget(self.comp_plotter_base.interactor)
        # deixa o estilo padrão por enquanto; depois ajustamos interação se precisar


        # tabela de fácies do BASE
        left_col.addWidget(QtWidgets.QLabel("Reservatório (base):"))
        self.res_table_base_cmp = make_facies_table()
        self.res_table_base_cmp.itemChanged.connect(self.update_base_reservoir_compare)
        left_col.addWidget(self.res_table_base_cmp)

        # -------------------------------
        # COLUNA DIREITA: MODELO COMPARADO
        # -------------------------------
        right_col = QtWidgets.QVBoxLayout()

        title_right = QtWidgets.QLabel("Modelo comparado")
        title_right.setAlignment(QtCore.Qt.AlignCenter)
        right_col.addWidget(title_right)

        # plotter comparado
        self.comp_plotter_comp = BackgroundPlotter(show=False)
        right_col.addWidget(self.comp_plotter_comp.interactor)
        # idem para o modelo comparado


        # tabela de fácies do COMPARADO
        right_col.addWidget(QtWidgets.QLabel("Reservatório (comparado):"))
        self.res_table_comp_cmp = make_facies_table()
        self.res_table_comp_cmp.itemChanged.connect(self.update_compare_reservoir_compare)
        right_col.addWidget(self.res_table_comp_cmp)

        # adiciona lado a lado
        cmp3d_layout.addLayout(left_col, stretch=1)
        cmp3d_layout.addLayout(right_col, stretch=1)


        # por enquanto deixa vazio
        self.compare_tabs.addTab(self.compare_3d_widget, "Visualização 3D")

        # adiciona aba Comparação no tab principal
        self.tabs.addTab(self.compare_tab, "Comparação")


        # adiciona o conjunto de abas no lado direito
        layout.addWidget(self.panel)
        layout.addWidget(self.tabs, 1)

        self.populate_facies_legend()

        # Carrega visualização inicial
        self.state = dict()
        self.state["reservoir_facies"] = initial_reservoir

        # ------ estado para comparação entre modelos ------
        # Base (modelo carregado em load_data)
        self.base_facies_stats, self.base_total_cells = facies_distribution_array(facies)
        self.base_res_stats = {}
        self.base_res_total = 0
        self.base_metrics = None
        self.base_perc = None

        # Modelo comparado (carregado via diálogo)
        self.compare_path = None
        self.compare_facies = None
        self.compare_facies_stats = None
        self.compare_total_cells = 0
        self.comp_res_stats = {}
        self.comp_res_total = 0
        self.compare_metrics = None
        self.compare_perc = None
        self.compare_states = {
            "base": {},
            "compare": {},
        }

        run(
            mode=mode,
            z_exag=z_exag,
            show_scalar_bar=show_scalar_bar,
            external_plotter=self.plotter,
            external_state=self.state,
        )

        self.update_2d_map()
        self.update_comparison_tables()
        self.init_compare_3d()

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
        self.update_compare_3d_mode()


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
        
        self.base_metrics = metrics
        self.base_perc = perc
        
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
        
        self.init_compare_3d()
        self.update_comparison_tables()

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

                # ----- distribuições de fácies dentro do reservatório -----
        rf = self.state["reservoir_facies"]

        if rf:
            # base
            self.base_res_stats, self.base_res_total = reservoir_facies_distribution_array(facies, rf)

            # modelo comparado (se existir)
            if self.compare_facies is not None:
                self.comp_res_stats, self.comp_res_total = reservoir_facies_distribution_array(
                    self.compare_facies, rf
                )
                # também atualiza métricas globais do comparado
                self.compare_metrics, self.compare_perc = compute_global_metrics_for_array(
                    self.compare_facies, rf
                )
            else:
                self.comp_res_stats, self.comp_res_total = {}, 0
                self.compare_metrics, self.compare_perc = None, None
        else:
            self.base_res_stats, self.base_res_total = {}, 0
            self.comp_res_stats, self.comp_res_total = {}, 0
            self.compare_metrics, self.compare_perc = None, None

        # atualiza as tabelas da aba de comparação
        self.init_compare_3d()
        self.update_comparison_tables()


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

        # ------------------------------------------------------------------
    # MODELO COMPARADO
    # ------------------------------------------------------------------
    def load_compare_model(self, grdecl_path: str):
        """
        Carrega um GRDECL externo só para COMPARAÇÃO (mesmo tamanho de grid).
        Salva as fácies no self.models["compare"]["facies"].
        """
        # lê fácies do GRDECL externo
        fac_compare = load_facies_from_grdecl(grdecl_path)

        # checa se tem o mesmo número de células do grid base
        expected = nx * ny * nz
        if fac_compare.size != expected:
            QtWidgets.QMessageBox.warning(
                self,
                "Erro ao carregar modelo comparado",
                f"O modelo '{grdecl_path}' tem {fac_compare.size} células, "
                f"mas o grid base tem {expected}.",
            )
            return

        # salva dentro da estrutura models
        self.models["compare"]["name"] = grdecl_path
        self.models["compare"]["facies"] = fac_compare
        # por padrão, usa as mesmas fácies de reservatório da base
        self.models["compare"]["reservoir_facies"] = set(
            self.models["base"]["reservoir_facies"]
        )

        print("\n[COMPARE] Modelo carregado para comparação:")
        print("  path:", grdecl_path)
        print("  células:", fac_compare.size)
        print("  fácies únicas:", np.unique(fac_compare))

        # aqui, mais pra frente, vamos:
        # - atualizar métricas de comparação
        # - atualizar visualização 3D de comparação
        # - atualizar mapas 2D de diferença etc.


    
    def open_compare_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Selecionar modelo para comparar",
            "assets",
            "GRDECL (*.grdecl)"
        )
        if not path:
            return

        self.load_compare_model(path)

        # atualiza label
        self.comp_model_label.setText(f"Modelo comparado: {path}")

        # atualiza listas de fácies nas UIs
        self.populate_compare_facies_lists()

        # faz a renderização inicial
        self.init_compare_3d()



    def update_comparison_tables(self):
        """
        Atualiza todas as tabelas da aba 'Comparação' com base nas
        estatísticas e métricas disponíveis.
        """

        # --------- 1) Tabela de métricas globais do reservatório ---------
        m0 = getattr(self, "base_metrics", None)
        m1 = getattr(self, "compare_metrics", None)
        p0 = getattr(self, "base_perc", None)
        p1 = getattr(self, "compare_perc", None)

        rows = []

        def get(m, key):
            return None if m is None else m.get(key)

        rows.append(("NTG global", get(m0, "ntg"), get(m1, "ntg")))
        rows.append(("Total de células", get(m0, "total_cells"), get(m1, "total_cells")))
        rows.append(("Células de reservatório", get(m0, "res_cells"), get(m1, "res_cells")))
        rows.append(("Nº de clusters", get(m0, "n_clusters"), get(m1, "n_clusters")))
        rows.append(("Tamanho do maior cluster", get(m0, "largest_size"), get(m1, "largest_size")))
        rows.append(("Fração conectada", get(m0, "connected_fraction"), get(m1, "connected_fraction")))

        def flag(per, key):
            if per is None:
                return None
            return "Sim" if per.get(key) else "Não"

        if p0 is not None:
            rows.append(("Percolação X (Sim/Não)", flag(p0, "x_perc"), flag(p1, "x_perc")))
            rows.append(("Percolação Y (Sim/Não)", flag(p0, "y_perc"), flag(p1, "y_perc")))
            rows.append(("Percolação Z (Sim/Não)", flag(p0, "z_perc"), flag(p1, "z_perc")))

        self.global_compare_table.setRowCount(len(rows))

        def fmt_num(val):
            if val is None:
                return "-"
            if isinstance(val, (float, np.floating)):
                return f"{val:.3f}"
            return str(val)

        for i, (label, a, b) in enumerate(rows):
            self.global_compare_table.setItem(i, 0, QtWidgets.QTableWidgetItem(label))
            self.global_compare_table.setItem(i, 1, QtWidgets.QTableWidgetItem(fmt_num(a)))
            self.global_compare_table.setItem(i, 2, QtWidgets.QTableWidgetItem(fmt_num(b)))

            if isinstance(a, (int, np.integer, float, np.floating)) and isinstance(
                b, (int, np.integer, float, np.floating)
            ):
                diff = float(b) - float(a)
                self.global_compare_table.setItem(
                    i, 3, QtWidgets.QTableWidgetItem(f"{diff:.3f}")
                )
            else:
                self.global_compare_table.setItem(i, 3, QtWidgets.QTableWidgetItem("-"))

        self.global_compare_table.resizeColumnsToContents()

        # --------- 2) Distribuição de fácies no grid inteiro ---------
        stats0 = getattr(self, "base_facies_stats", None)
        stats1 = getattr(self, "compare_facies_stats", None)
        if stats0 is None:
            return

        facs_union = sorted(set(stats0.keys()) | (set(stats1.keys()) if stats1 else set()))
        self.facies_compare_table.setRowCount(len(facs_union))

        for row, fac in enumerate(facs_union):
            s0 = stats0.get(fac, {"cells": 0, "fraction": 0.0})
            s1 = stats1.get(fac, {"cells": 0, "fraction": 0.0}) if stats1 else {"cells": 0, "fraction": 0.0}

            self.facies_compare_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(fac)))
            self.facies_compare_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(s0["cells"])))
            self.facies_compare_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{100*s0['fraction']:.2f}"))
            self.facies_compare_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(s1["cells"])))
            self.facies_compare_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{100*s1['fraction']:.2f}"))

        self.facies_compare_table.resizeColumnsToContents()

        # --------- 3) Distribuição de fácies dentro do reservatório ---------
        stats0r = getattr(self, "base_res_stats", None) or {}
        stats1r = getattr(self, "comp_res_stats", None) or {}
        facs_res_union = sorted(set(stats0r.keys()) | set(stats1r.keys()))
        self.reservoir_facies_compare_table.setRowCount(len(facs_res_union))

        for row, fac in enumerate(facs_res_union):
            s0 = stats0r.get(fac, {"cells": 0, "fraction": 0.0})
            s1 = stats1r.get(fac, {"cells": 0, "fraction": 0.0})

            self.reservoir_facies_compare_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(fac)))
            self.reservoir_facies_compare_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(s0["cells"])))
            self.reservoir_facies_compare_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{100*s0['fraction']:.2f}"))
            self.reservoir_facies_compare_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(s1["cells"])))
            self.reservoir_facies_compare_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{100*s1['fraction']:.2f}"))

        self.reservoir_facies_compare_table.resizeColumnsToContents()

    def populate_compare_facies_lists(self):
        # BASE
        fill_facies_table(
            self.res_table_base_cmp,
            self.models["base"]["facies"],
            self.models["base"]["reservoir_facies"]
        )

        # COMPARADO
        if self.models["compare"]["facies"] is not None:
            fill_facies_table(
                self.res_table_comp_cmp,
                self.models["compare"]["facies"],
                self.models["compare"]["reservoir_facies"]
            )


    
    def update_base_reservoir_compare(self, item):
        # só reage à coluna "Sel." (índice 3)
        if item.column() != 3:
            return

        f = int(item.data(QtCore.Qt.UserRole))
        if item.checkState() == QtCore.Qt.Checked:
            self.models["base"]["reservoir_facies"].add(f)
        else:
            self.models["base"]["reservoir_facies"].discard(f)
        self.update_compare_3d_mode()

    def update_compare_reservoir_compare(self, item):
        if item.column() != 3:
            return

        f = int(item.data(QtCore.Qt.UserRole))
        if item.checkState() == QtCore.Qt.Checked:
            self.models["compare"]["reservoir_facies"].add(f)
        else:
            self.models["compare"]["reservoir_facies"].discard(f)
        self.update_compare_3d_mode()


    def init_compare_3d(self):
        """
        Cria DOIS visualizadores 3D completos (visualize.run) lado a lado,
        um para o modelo base e outro para o modelo comparado.
        """
        if self.models["base"]["facies"] is None:
            return

        from visualize import run

        mode = self.state.get("mode", "facies")
        z_exag = self.state.get("z_exag", 15.0)

        # ---------------- BASE ----------------
        self.comp_plotter_base.clear()
        self.compare_states["base"] = {}

        _, state_base = run(
            mode=mode,
            z_exag=z_exag,
            show_scalar_bar=False,
            external_plotter=self.comp_plotter_base,
            external_state=self.compare_states["base"],
            facies_array=self.models["base"]["facies"],
        )

        # força o mesmo conjunto de fácies de reservatório do modelo base
        state_base["reservoir_facies"] = set(self.models["base"]["reservoir_facies"])
        if "update_reservoir_fields" in state_base:
            state_base["update_reservoir_fields"](state_base["reservoir_facies"])
            state_base["refresh"]()

        # ---------------- COMPARADO ----------------
        self.comp_plotter_comp.clear()
        self.compare_states["compare"] = {}

        if self.models["compare"]["facies"] is not None:
            _, state_comp = run(
                mode=mode,
                z_exag=z_exag,
                show_scalar_bar=False,
                external_plotter=self.comp_plotter_comp,
                external_state=self.compare_states["compare"],
                facies_array=self.models["compare"]["facies"],
            )

            # aplica o conjunto de fácies de reservatório DO MODELO COMPARADO
            state_comp["reservoir_facies"] = set(self.models["compare"]["reservoir_facies"])
            if "update_reservoir_fields" in state_comp:
                state_comp["update_reservoir_fields"](state_comp["reservoir_facies"])
                state_comp["refresh"]()
        else:
            state_comp = None

        # registra callbacks de sincronização de k e box
        self.install_compare_sync_callbacks()

        # sincroniza a câmera logo na inicialização
        self.sync_compare_cameras()


    def install_compare_sync_callbacks(self):
        """
        Conecta callbacks para que:
        - mudar k_min em QUALQUER visualizador (principal ou comparação)
          atualize os três;
        - mover o box_widget em QUALQUER visualizador atualize os três.
        """
        main_state = getattr(self, "state", None)
        base_state = self.compare_states.get("base")
        comp_state = self.compare_states.get("compare")

        states = [s for s in (main_state, base_state, comp_state) if s]

        if not states:
            return

        def on_k_changed(new_k):
            for st in states:
                st["k_min"] = int(new_k)
                if "refresh" in st:
                    st["refresh"]()

        def on_box_changed(bounds):
            for st in states:
                st["box_bounds"] = bounds
                bw = st.get("box_widget")
                if bw is not None:
                    try:
                        bw.PlaceWidget(bounds)
                    except Exception:
                        pass
                if "refresh" in st:
                    st["refresh"]()

        # registra os callbacks em todos os estados
        for st in states:
            st["on_k_changed"] = on_k_changed
            st["on_box_changed"] = on_box_changed




    def sync_compare_cameras(self):
        """
        Mantém as câmeras da comparação alinhadas com a câmera principal.
        """
        if not hasattr(self, "plotter"):
            return
        if not hasattr(self, "comp_plotter_base") or not hasattr(self, "comp_plotter_comp"):
            return

        main_cam = self.plotter.camera

        def copy_cam(src, dst):
            dst.position = src.position
            dst.focal_point = src.focal_point
            dst.view_up = src.view_up
            dst.parallel_scale = src.parallel_scale

        def on_end_interaction_caller(obj, ev):
            # copiou do main para os outros
            copy_cam(main_cam, self.comp_plotter_base.camera)
            copy_cam(main_cam, self.comp_plotter_comp.camera)
            self.comp_plotter_base.render()
            self.comp_plotter_comp.render()

        # Observa a câmera do visualizador principal
        if hasattr(self.plotter, "iren") and self.plotter.iren is not None:
            interactor = self.plotter.iren
            interactor.add_observer("EndInteractionEvent", on_end_interaction_caller)






    def update_compare_3d_mode(self):
        mode = self.state.get("mode", "facies")

        for key in ("base", "compare"):
            st = self.compare_states.get(key)
            if not st:
                continue
            st["mode"] = mode
            # thickness_mode já está dentro do state, então o run se vira
            if "refresh" in st:
                st["refresh"]()









