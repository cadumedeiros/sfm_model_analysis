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
    """
    Cria a tabela de Fácies compacta (largura ajustada ao conteúdo).
    """
    table = QtWidgets.QTableWidget()
    table.setColumnCount(4)
    table.setHorizontalHeaderLabels(["Cor", "Fácies", "Células", "Sel."])
    
    # Configuração Visual
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
    
    # Permite crescer verticalmente, mas TRAVA horizontalmente
    table.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
    table.setMinimumWidth(250)
    table.setMaximumWidth(320) # [AJUSTE]: Largura máxima para ficar compacta
    
    # [AJUSTE]: Colunas se ajustam ao texto, não esticam para preencher vazio
    header = table.horizontalHeader()
    header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
    header.setStretchLastSection(False) 
    
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
            initial_reservoir = {int(f) for f in reservoir_facies}

        # ---------- MODELOS ----------
        self.models = {
            "base": {
                "name": "BaseModel",
                "facies": facies,
                "reservoir_facies": set(initial_reservoir),
            },
            "compare": {
                "name": None,
                "facies": None,
                "reservoir_facies": set(),
            },
        }

        self.resize(1600, 900)
        self.setMinimumSize(800, 600)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        self.setCentralWidget(central)

        # --- PAINEL LATERAL ---
        self.panel = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(self.panel)
        self.panel.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.panel.setMaximumWidth(300) 

        # Logo
        logo_label = QtWidgets.QLabel()
        pix = QPixmap("assets/forward_PNG.png")
        if not pix.isNull():
            pix = pix.scaledToWidth(140, QtCore.Qt.SmoothTransformation)
            logo_label.setPixmap(pix)
        logo_label.setAlignment(QtCore.Qt.AlignCenter)
        panel_layout.addWidget(logo_label)
        panel_layout.addSpacing(10)

        # Botões
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

        # Thickness
        self.thick_combo = QtWidgets.QComboBox()
        self.thick_combo.addItems([
            "Espessura", "NTG coluna", "NTG envelope", "Maior pacote",
            "Nº pacotes", "ICV", "Qv", "Qv absoluto",
        ])
        self.thick_combo.currentTextChanged.connect(self.change_thickness_mode)
        sub_group = QtWidgets.QGroupBox("Thickness Local")
        sub_layout = QtWidgets.QVBoxLayout(sub_group)
        sub_layout.addWidget(self.thick_combo)
        panel_layout.addWidget(sub_group)

        # Lista Reservatório
        self.res_group = QtWidgets.QGroupBox("Fácies do Reservatório")
        res_layout = QtWidgets.QVBoxLayout(self.res_group)
        self.reservoir_list = QtWidgets.QListWidget()
        self.reservoir_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        present = sorted(set(int(v) for v in np.unique(facies)))
        for fac in present:
            item = QtWidgets.QListWidgetItem(str(fac))
            item.setCheckState(QtCore.Qt.Unchecked)
            item.setData(QtCore.Qt.UserRole, fac)
            self.reservoir_list.addItem(item)
        self.reservoir_list.itemChanged.connect(self.change_reservoir_facies)
        res_layout.addWidget(self.reservoir_list)
        panel_layout.addWidget(self.res_group)

        # Legenda Lateral
        self.legend_group = QtWidgets.QGroupBox("Legenda de Fácies")
        legend_layout = QtWidgets.QVBoxLayout(self.legend_group)
        self.facies_legend_table = self._create_legend_table()
        legend_layout.addWidget(self.facies_legend_table)
        panel_layout.addWidget(self.legend_group)

        panel_layout.addStretch()

        # --- ÁREA DIREITA (TABs) ---
        self.tabs = QtWidgets.QTabWidget()
        
        # 1. Aba 3D Principal
        self.plotter = BackgroundPlotter(show=False)
        self.viz_tab = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(self.viz_tab)
        vl.setContentsMargins(0,0,0,0)
        vl.addWidget(self.plotter.interactor)
        self.tabs.addTab(self.viz_tab, "Visualização 3D")
        
        # 2. Aba 2D
        self.plotter_2d = BackgroundPlotter(show=False)
        self.map2d_tab = QtWidgets.QWidget()
        ml = QtWidgets.QVBoxLayout(self.map2d_tab)
        ml.setContentsMargins(0,0,0,0)
        ml.addWidget(self.plotter_2d.interactor)
        self.tabs.addTab(self.map2d_tab, "Mapa 2D")

        # 3. Aba Métricas Globais
        self.metrics_tab = QtWidgets.QWidget()
        self.metrics_layout = QtWidgets.QVBoxLayout(self.metrics_tab)
        metrics_group = QtWidgets.QGroupBox("Análises do Modelo")
        mg_layout = QtWidgets.QVBoxLayout(metrics_group)
        self.metrics_text = QtWidgets.QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMinimumHeight(120)
        mg_layout.addWidget(self.metrics_text)
        self.metrics_layout.addWidget(metrics_group)
        facies_group = QtWidgets.QGroupBox("Análises por Fácies")
        fg_layout = QtWidgets.QVBoxLayout(facies_group)
        self.facies_table = QtWidgets.QTableWidget()
        fg_layout.addWidget(self.facies_table)
        self.metrics_layout.addWidget(facies_group)
        self.metrics_layout.addStretch()
        self.tabs.addTab(self.metrics_tab, "Métricas Globais")

        # --- 4. ABA COMPARAÇÃO ---
        self.compare_tab = QtWidgets.QWidget()
        comp_layout = QtWidgets.QVBoxLayout(self.compare_tab)
        self.compare_tabs = QtWidgets.QTabWidget()
        comp_layout.addWidget(self.compare_tabs)
        
        # 4.1 Sub-aba Métricas (MANTIDA IGUAL)
        self.compare_metrics_widget = QtWidgets.QWidget()
        cmp_metrics_layout = QtWidgets.QVBoxLayout(self.compare_metrics_widget)
        
        hl = QtWidgets.QHBoxLayout()
        self.base_model_label = QtWidgets.QLabel("Modelo base: (carregado)")
        self.comp_model_label = QtWidgets.QLabel("Modelo comparado: (nenhum)")
        hl.addWidget(self.base_model_label)
        hl.addWidget(self.comp_model_label)
        cmp_metrics_layout.addLayout(hl)
        
        self.select_compare_btn = QtWidgets.QPushButton("Selecionar modelo...")
        self.select_compare_btn.clicked.connect(self.open_compare_dialog)
        cmp_metrics_layout.addWidget(self.select_compare_btn)
        
        self.global_compare_table = QtWidgets.QTableWidget()
        self.global_compare_table.setColumnCount(4)
        self.global_compare_table.setHorizontalHeaderLabels(["Métrica", "Base", "Comparado", "Dif"])
        self.global_compare_table.verticalHeader().setVisible(False)
        
        self.facies_compare_table = QtWidgets.QTableWidget()
        self.facies_compare_table.setColumnCount(5)
        self.facies_compare_table.setHorizontalHeaderLabels(["Fácies", "Cél. Base", "% Base", "Cél. Comp", "% Comp"])
        self.facies_compare_table.verticalHeader().setVisible(False)
        
        self.reservoir_facies_compare_table = QtWidgets.QTableWidget()
        self.reservoir_facies_compare_table.setColumnCount(5)
        self.reservoir_facies_compare_table.setHorizontalHeaderLabels(["Fácies (Res)", "Cél. Base", "% Base", "Cél. Comp", "% Comp"])
        self.reservoir_facies_compare_table.verticalHeader().setVisible(False)

        cmp_metrics_layout.addWidget(QtWidgets.QLabel("Métricas Globais"))
        cmp_metrics_layout.addWidget(self.global_compare_table)
        cmp_metrics_layout.addWidget(QtWidgets.QLabel("Fácies (Grid Inteiro)"))
        cmp_metrics_layout.addWidget(self.facies_compare_table)
        cmp_metrics_layout.addWidget(QtWidgets.QLabel("Fácies (Reservatório)"))
        cmp_metrics_layout.addWidget(self.reservoir_facies_compare_table)
        
        self.compare_tabs.addTab(self.compare_metrics_widget, "Métricas")

        # =====================================================================
        # 4.2 Sub-aba MAPAS 2D (NOVA)
        # =====================================================================
        self.compare_2d_widget = QtWidgets.QWidget()
        c2d_layout = QtWidgets.QHBoxLayout(self.compare_2d_widget)
        
        # Plotter Base 2D
        l_col_2d = QtWidgets.QVBoxLayout()
        self.comp_plotter_base_2d = BackgroundPlotter(show=False)
        l_col_2d.addWidget(QtWidgets.QLabel("Mapa Base"))
        l_col_2d.addWidget(self.comp_plotter_base_2d.interactor)
        c2d_layout.addLayout(l_col_2d)

        # Plotter Comparado 2D
        r_col_2d = QtWidgets.QVBoxLayout()
        self.comp_plotter_comp_2d = BackgroundPlotter(show=False)
        r_col_2d.addWidget(QtWidgets.QLabel("Mapa Comparado"))
        r_col_2d.addWidget(self.comp_plotter_comp_2d.interactor)
        c2d_layout.addLayout(r_col_2d)

        self.compare_tabs.addTab(self.compare_2d_widget, "Mapas 2D")

        # =====================================================================
        # 4.3 Sub-aba Visualização 3D (MANTIDA IGUAL)
        # =====================================================================
        self.compare_3d_widget = QtWidgets.QWidget()
        c3d_main_layout = QtWidgets.QVBoxLayout(self.compare_3d_widget)
        c3d_main_layout.setContentsMargins(0,0,0,0)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        c3d_main_layout.addWidget(splitter)

        # Topo 3D
        top_widget = QtWidgets.QWidget()
        top_layout = QtWidgets.QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0,0,0,0)
        top_layout.setSpacing(2)

        self.comp_plotter_base = BackgroundPlotter(show=False)
        l_container = QtWidgets.QWidget()
        l_layout = QtWidgets.QVBoxLayout(l_container)
        l_layout.setContentsMargins(0,0,0,0)
        l_layout.addWidget(QtWidgets.QLabel("Modelo Base"))
        l_layout.addWidget(self.comp_plotter_base.interactor)
        top_layout.addWidget(l_container)

        self.comp_plotter_comp = BackgroundPlotter(show=False)
        r_container = QtWidgets.QWidget()
        r_layout = QtWidgets.QVBoxLayout(r_container)
        r_layout.setContentsMargins(0,0,0,0)
        r_layout.addWidget(QtWidgets.QLabel("Modelo Comparado"))
        r_layout.addWidget(self.comp_plotter_comp.interactor)
        top_layout.addWidget(r_container)
        splitter.addWidget(top_widget)

        # Baixo Tabelas
        bottom_widget = QtWidgets.QWidget()
        bottom_widget.setMinimumHeight(200) 
        bottom_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        bottom_layout = QtWidgets.QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(10, 10, 10, 10)

        base_group = QtWidgets.QGroupBox("Dados Base")
        bg_layout = QtWidgets.QHBoxLayout(base_group)
        self.res_table_base_cmp = make_facies_table()
        self.res_table_base_cmp.itemChanged.connect(self.update_base_reservoir_compare)
        bg_layout.addWidget(self.res_table_base_cmp)
        self.clus_table_base_cmp = self._create_legend_table(["Cor", "Cluster", "Células"])
        self.clus_table_base_cmp.setVisible(False)
        bg_layout.addWidget(self.clus_table_base_cmp)
        bottom_layout.addWidget(base_group)

        comp_group = QtWidgets.QGroupBox("Dados Comparado")
        cg_layout = QtWidgets.QHBoxLayout(comp_group)
        self.res_table_comp_cmp = make_facies_table()
        self.res_table_comp_cmp.itemChanged.connect(self.update_compare_reservoir_compare)
        cg_layout.addWidget(self.res_table_comp_cmp)
        self.clus_table_comp_cmp = self._create_legend_table(["Cor", "Cluster", "Células"])
        self.clus_table_comp_cmp.setVisible(False)
        cg_layout.addWidget(self.clus_table_comp_cmp)
        bottom_layout.addWidget(comp_group)
        
        splitter.addWidget(bottom_widget)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setSizes([800, 400])

        self.compare_tabs.addTab(self.compare_3d_widget, "Visualização 3D")
        self.tabs.addTab(self.compare_tab, "Comparação")

        layout.addWidget(self.panel, 1)
        layout.addWidget(self.tabs, 5)

        # ... (Restante do __init__ continua igual)
        self.populate_facies_legend()
        self.state = dict()
        self.state["reservoir_facies"] = initial_reservoir
        self.compare_states = {"base": {}, "compare": {}}
        self.base_facies_stats, self.base_total_cells = facies_distribution_array(facies)
        self.base_res_stats = {}
        self.compare_path = None
        self.compare_facies = None
        self.compare_metrics = None

        _, self.state = run(
            mode=mode,
            z_exag=z_exag,
            show_scalar_bar=show_scalar_bar,
            external_plotter=self.plotter,
            external_state=self.state,
        )
        
        self.reservoir_list.blockSignals(True)
        for i in range(self.reservoir_list.count()):
            item = self.reservoir_list.item(i)
            fac = item.data(QtCore.Qt.UserRole)
            if fac in initial_reservoir:
                item.setCheckState(QtCore.Qt.Checked)
        self.reservoir_list.blockSignals(False)

        self.change_reservoir_facies(None) 
        self.update_2d_map()
        self.update_comparison_tables()
        self.init_compare_3d()
        # [NOVO]: Chama atualização inicial dos mapas 2D
        self.update_compare_2d_maps()

    # Helper para criar tabela
    def _create_legend_table(self, headers=["Cor", "Fácies", "Células", "Sel."]):
        """Cria tabela compacta para legendas extras."""
        table = QtWidgets.QTableWidget()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        table.setShowGrid(False)
        
        # Compacta horizontalmente, expande verticalmente
        table.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        table.setMaximumWidth(320)
        
        header = table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        
        return table

    # Helper para UI de comparação (caso você não tenha separado ainda)
    def _init_comparison_ui(self):
        self.compare_tab = QtWidgets.QWidget()
        comp_layout = QtWidgets.QVBoxLayout(self.compare_tab)
        self.compare_tabs = QtWidgets.QTabWidget()
        comp_layout.addWidget(self.compare_tabs)
        
        # Sub-aba Métricas
        self.compare_metrics_widget = QtWidgets.QWidget()
        cmp_metrics_layout = QtWidgets.QVBoxLayout(self.compare_metrics_widget)
        
        # Headers
        self.base_model_label = QtWidgets.QLabel("Modelo base: (carregado)")
        self.comp_model_label = QtWidgets.QLabel("Modelo comparado: (nenhum)")
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(self.base_model_label)
        hl.addWidget(self.comp_model_label)
        cmp_metrics_layout.addLayout(hl)
        
        self.select_compare_btn = QtWidgets.QPushButton("Selecionar modelo...")
        self.select_compare_btn.clicked.connect(self.open_compare_dialog)
        cmp_metrics_layout.addWidget(self.select_compare_btn)
        
        # Tabelas
        self.global_compare_table = QtWidgets.QTableWidget()
        self.facies_compare_table = QtWidgets.QTableWidget()
        self.reservoir_facies_compare_table = QtWidgets.QTableWidget()
        
        cmp_metrics_layout.addWidget(QtWidgets.QLabel("Métricas Globais"))
        cmp_metrics_layout.addWidget(self.global_compare_table)
        cmp_metrics_layout.addWidget(QtWidgets.QLabel("Fácies (Grid Inteiro)"))
        cmp_metrics_layout.addWidget(self.facies_compare_table)
        cmp_metrics_layout.addWidget(QtWidgets.QLabel("Fácies (Reservatório)"))
        cmp_metrics_layout.addWidget(self.reservoir_facies_compare_table)
        
        self.compare_tabs.addTab(self.compare_metrics_widget, "Métricas")
        
        # Sub-aba 3D
        self.compare_3d_widget = QtWidgets.QWidget()
        c3d_layout = QtWidgets.QHBoxLayout(self.compare_3d_widget)
        
        # Base
        l_col = QtWidgets.QVBoxLayout()
        l_col.addWidget(QtWidgets.QLabel("Modelo Base"))
        self.comp_plotter_base = BackgroundPlotter(show=False)
        l_col.addWidget(self.comp_plotter_base.interactor)
        self.res_table_base_cmp = make_facies_table()
        self.res_table_base_cmp.itemChanged.connect(self.update_base_reservoir_compare)
        l_col.addWidget(self.res_table_base_cmp)
        
        # Compare
        r_col = QtWidgets.QVBoxLayout()
        r_col.addWidget(QtWidgets.QLabel("Modelo Comparado"))
        self.comp_plotter_comp = BackgroundPlotter(show=False)
        r_col.addWidget(self.comp_plotter_comp.interactor)
        self.res_table_comp_cmp = make_facies_table()
        self.res_table_comp_cmp.itemChanged.connect(self.update_compare_reservoir_compare)
        r_col.addWidget(self.res_table_comp_cmp)
        
        c3d_layout.addLayout(l_col)
        c3d_layout.addLayout(r_col)
        
        self.compare_tabs.addTab(self.compare_3d_widget, "Visualização 3D")
        self.tabs.addTab(self.compare_tab, "Comparação")
        
    
    def populate_facies_legend(self):
        """Restaura a tabela LATERAL para o modo Fácies."""
        colors_dict = load_facies_colors()
        present = sorted(set(int(v) for v in np.unique(facies)))
        vals, counts = np.unique(facies.astype(int), return_counts=True)
        count_dict = {int(v): int(c) for v, c in zip(vals, counts)}

        self.facies_legend_table.clear()
        self.facies_legend_table.setRowCount(len(present))
        self.facies_legend_table.setColumnCount(3)
        self.facies_legend_table.setHorizontalHeaderLabels(["Cor", "Fácies", "Células"])

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

            cells_item = QtWidgets.QTableWidgetItem(str(count_dict.get(fac, 0)))
            cells_item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.facies_legend_table.setItem(row, 2, cells_item)

        self.facies_legend_table.resizeColumnsToContents()
        self.facies_legend_table.setColumnWidth(0, 26)

    def populate_clusters_legend(self):
        """
        Preenche a tabela LATERAL (esquerda) com a legenda de Clusters.
        Reutiliza self.facies_legend_table.
        """
        if not hasattr(self, "state"): return

        sizes_dict = self.state.get("clusters_sizes")
        lut = self.state.get("clusters_lut")
        
        table = self.facies_legend_table
        
        if not sizes_dict or lut is None:
            table.setRowCount(0)
            return
            
        # Reconfigura colunas para Cluster
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Cor", "Cluster", "Células"])

        labels = sorted(sizes_dict.keys(), key=lambda k: sizes_dict[k], reverse=True)
        table.setRowCount(len(labels))
        
        for row, lab in enumerate(labels):
            r, g, b, a = lut.GetTableValue(int(lab))
            color = QColor(int(r*255), int(g*255), int(b*255))
            
            item_c = QtWidgets.QTableWidgetItem()
            item_c.setBackground(QBrush(color))
            item_c.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(row, 0, item_c)

            item_id = QtWidgets.QTableWidgetItem(str(lab))
            item_id.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(row, 1, item_id)

            item_s = QtWidgets.QTableWidgetItem(str(sizes_dict[lab]))
            item_s.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(row, 2, item_s)
            
        table.resizeColumnsToContents()

    def populate_compare_clusters_tables(self):
        """Preenche as tabelas de clusters específicas da aba de COMPARAÇÃO."""
        
        # Helper interno para preencher uma tabela dado um 'state'
        def fill_one_table(table, st):
            sizes = st.get("clusters_sizes")
            lut = st.get("clusters_lut")
            
            if not sizes or not lut:
                table.setRowCount(0)
                return
            
            labels = sorted(sizes.keys(), key=lambda k: sizes[k], reverse=True)
            table.setRowCount(len(labels))
            
            for row, lab in enumerate(labels):
                r, g, b, a = lut.GetTableValue(int(lab))
                color = QColor(int(r*255), int(g*255), int(b*255))
                
                item_c = QtWidgets.QTableWidgetItem()
                item_c.setBackground(QBrush(color))
                item_c.setFlags(QtCore.Qt.ItemIsEnabled)
                table.setItem(row, 0, item_c)

                item_id = QtWidgets.QTableWidgetItem(str(lab))
                item_id.setFlags(QtCore.Qt.ItemIsEnabled)
                table.setItem(row, 1, item_id)

                item_s = QtWidgets.QTableWidgetItem(str(sizes[lab]))
                item_s.setFlags(QtCore.Qt.ItemIsEnabled)
                table.setItem(row, 2, item_s)
            
            table.resizeColumnsToContents()

        # Preenche tabela Base
        if self.compare_states.get("base"):
            fill_one_table(self.clus_table_base_cmp, self.compare_states["base"])
            
        # Preenche tabela Comparado
        if self.compare_states.get("compare"):
            fill_one_table(self.clus_table_comp_cmp, self.compare_states["compare"])

    # ----------------------------------------------------------------------
    def change_mode(self, new_mode):
        print("Modo:", new_mode)
        self.state["mode"] = new_mode

        # --- 1. Painel Principal (Sidebar) ---
        if new_mode == "clusters":
            self.legend_group.setTitle("Legenda de Clusters")
            self.populate_clusters_legend()
        else:
            self.legend_group.setTitle("Legenda de Fácies")
            self.populate_facies_legend()

        # --- 2. Aba Comparação (Bottom Panel) ---
        # Mostra tabelas de clusters APENAS se o modo for "clusters"
        is_cluster = (new_mode == "clusters")
        
        self.clus_table_base_cmp.setVisible(is_cluster)
        self.clus_table_comp_cmp.setVisible(is_cluster)
        
        # Se não for cluster (ex: facies, reservoir), a tabela de Fácies (res_table)
        # deve continuar visível e ocupar o espaço deixado pela tabela de clusters oculta.
        self.res_table_base_cmp.setVisible(True) # Sempre visível
        self.res_table_comp_cmp.setVisible(True) # Sempre visível

        if is_cluster:
            self.populate_compare_clusters_tables()

        # Redesenha 3D
        self.state["refresh"]()
        self.update_compare_3d_mode()


    def change_thickness_mode(self, label):
        print("Thickness:", label)
        # Atualiza visualizador principal
        self.state["thickness_mode"] = label
        if "update_thickness" in self.state:
            self.state["update_thickness"]()
        self.state["refresh"]()

        # [CORREÇÃO 2]: Atualiza visualizadores de COMPARAÇÃO
        for key in ["base", "compare"]:
            st = self.compare_states.get(key)
            if st:
                st["thickness_mode"] = label
                if "update_thickness" in st: 
                    st["update_thickness"]()
                if "refresh" in st: 
                    st["refresh"]()

        # Atualiza mapa 2D
        self.update_2d_map()

        # [NOVO] Atualiza mapas 2D da comparação
        self.update_compare_2d_maps()

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
        """
        Atualiza o Visualizador Principal e RECALCULA as Métricas (inclusive as de Comparação).
        NÃO afeta a visualização 3D da aba Comparação.
        """
        # 1. Coleta a nova seleção do painel esquerdo
        selected = []
        for i in range(self.reservoir_list.count()):
            it = self.reservoir_list.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                selected.append(int(it.data(QtCore.Qt.UserRole)))
        
        selected_set = set(selected)

        # 2. Atualiza o Estado Principal (Visualizador 1)
        self.state["reservoir_facies"] = selected_set
        
        # Atualiza visualização 3D Principal
        if "update_reservoir_fields" in self.state:
            self.state["update_reservoir_fields"](selected_set)
        
        if "refresh" in self.state:
            self.state["refresh"]()
            
        self.update_2d_map()

        # 3. RECALCULO DE MÉTRICAS (Para as Tabelas)
        # Isso afeta a aba "Métricas Globais" e a aba "Comparação > Métricas"
        
        # A) Métricas do Modelo Base
        # Recalcula métricas globais
        base_metrics = compute_global_metrics(selected_set)
        base_perc = compute_directional_percolation(selected_set)
        
        # Recalcula distribuição de fácies DENTRO do reservatório (Base)
        base_res_stats, base_res_total = reservoir_facies_distribution_array(facies, selected_set)
        
        # Armazena nos atributos da classe para as tabelas usarem
        self.base_metrics = base_metrics
        self.base_perc = base_perc
        self.base_res_stats = base_res_stats
        self.base_res_total = base_res_total
        
        # Atualiza texto da aba "Métricas Globais"
        self.set_metrics(base_metrics, base_perc)

        # B) Métricas do Modelo Comparado (Se houver um carregado)
        # Quando mudamos a definição de "reservatório" na esquerda, 
        # faz sentido atualizar as métricas do comparado para usar a mesma definição.
        if self.compare_facies is not None:
            # Recalcula globais do comparado com a nova seleção
            c_metrics, c_perc = compute_global_metrics_for_array(self.compare_facies, selected_set)
            
            # Recalcula distribuição do reservatório do comparado
            c_res_stats, c_res_total = reservoir_facies_distribution_array(self.compare_facies, selected_set)
            
            self.compare_metrics = c_metrics
            self.compare_perc = c_perc
            self.comp_res_stats = c_res_stats
            self.comp_res_total = c_res_total

        # 4. Atualiza as Tabelas da Aba Comparação
        # (Isso preenche os números novos sem mexer no 3D)
        self.update_comparison_tables()

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
        Carrega um GRDECL externo, salva no state e CALCULA AS MÉTRICAS imediatamente.
        """
        # 1. Carrega o array de fácies do arquivo
        try:
            fac_compare = load_facies_from_grdecl(grdecl_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erro", f"Falha ao ler arquivo:\n{e}")
            return

        # 2. Verificação de compatibilidade de grid
        expected = nx * ny * nz
        if fac_compare.size != expected:
            QtWidgets.QMessageBox.warning(
                self,
                "Erro de Dimensão",
                f"O modelo carregado tem {fac_compare.size} células, "
                f"mas o grid base espera {expected} ({nx}x{ny}x{nz}).",
            )
            return

        # 3. Atualiza Estrutura de Dados
        self.models["compare"]["name"] = grdecl_path
        self.models["compare"]["facies"] = fac_compare
        # Garante que o set de reservatório seja o mesmo da base inicialmente
        self.models["compare"]["reservoir_facies"] = set(self.models["base"]["reservoir_facies"])

        # 4. Atualiza Variáveis de Estado para as Tabelas (CRÍTICO)
        self.compare_path = grdecl_path
        self.compare_facies = fac_compare
        
        # --- CÁLCULOS ESTATÍSTICOS ---
        
        # A) Distribuição de Fácies (Grid Inteiro)
        self.compare_facies_stats, self.compare_total_cells = facies_distribution_array(fac_compare)
        
        # B) Métricas de Reservatório (Baseado na seleção ATUAL)
        rf = self.state.get("reservoir_facies", set())
        
        if rf:
            # Métricas Globais (NTG, Clusters, Conectividade)
            self.compare_metrics, self.compare_perc = compute_global_metrics_for_array(fac_compare, rf)
            # Distribuição dentro do Reservatório
            self.comp_res_stats, self.comp_res_total = reservoir_facies_distribution_array(fac_compare, rf)
        else:
            self.compare_metrics, self.compare_perc = None, None
            self.comp_res_stats, self.comp_res_total = {}, 0

        print(f"\n[COMPARE] Modelo '{os.path.basename(grdecl_path)}' carregado e calculado.")

        # 5. Atualiza a UI
        self.comp_model_label.setText(f"Modelo comparado: {os.path.basename(grdecl_path)}")
        
        # Atualiza listas de seleção nas abas de comparação
        self.populate_compare_facies_lists()
        
        # Atualiza as tabelas de métricas
        self.update_comparison_tables()

        # Inicializa a visualização 3D Comparada
        self.init_compare_3d()

        # [NOVO] Atualiza mapas 2D
        self.update_compare_2d_maps()

    def open_compare_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Selecionar modelo para comparar",
            "assets", # ou self.last_dir se tiver
            "GRDECL (*.grdecl);;Todos os arquivos (*)"
        )
        if not path:
            return
            
        self.load_compare_model(path)



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
        # [NOVO] Recalcula mapa 2D pois a definição de reservatório mudou
        self.update_compare_2d_maps()

    def update_compare_reservoir_compare(self, item):
        if item.column() != 3:
            return

        f = int(item.data(QtCore.Qt.UserRole))
        if item.checkState() == QtCore.Qt.Checked:
            self.models["compare"]["reservoir_facies"].add(f)
        else:
            self.models["compare"]["reservoir_facies"].discard(f)
        self.update_compare_3d_mode()
        # [NOVO]
        self.update_compare_2d_maps()


    # =========================================================================
    #  COLE ISTO NO FINAL DO SEU window.py (Substituindo as anteriores)
    # =========================================================================

    def init_compare_3d(self):
        """
        Inicializa os visualizadores de comparação com correção para evitar
        duplicação de dados e garantir independência.
        """
        if self.models["base"]["facies"] is None:
            return

        from visualize import run
        from load_data import grid as global_grid

        mode = self.state.get("mode", "facies")
        z_exag = self.state.get("z_exag", 15.0)
        show_sb = self.state.get("show_scalar_bar", False)

        # --- 1. Lado ESQUERDO: Modelo Base ---
        self.comp_plotter_base.clear()
        self.compare_states["base"] = {}
        
        # O modelo base usa o grid global original e as fácies originais
        _, state_base = run(
            mode=mode,
            z_exag=z_exag,
            show_scalar_bar=show_sb,
            external_plotter=self.comp_plotter_base,
            external_state=self.compare_states["base"],
            target_grid=global_grid, 
            target_facies=self.models["base"]["facies"]
        )

        # Aplica seleção de reservatório da base
        rf_base = self.models["base"]["reservoir_facies"]
        state_base["reservoir_facies"] = set(rf_base)
        if "update_reservoir_fields" in state_base:
            state_base["update_reservoir_fields"](rf_base)
            state_base["refresh"]()

        # --- 2. Lado DIREITO: Modelo Comparado ---
        self.comp_plotter_comp.clear()
        self.compare_states["compare"] = {}

        if self.models["compare"]["facies"] is not None:
            # [CORREÇÃO CRÍTICA] Clona o grid E injeta as fácies novas nele
            # Se não fizermos isso, o visualize pode ler as fácies antigas do grid.
            comp_grid = global_grid.copy(deep=True)
            comp_facies = self.models["compare"]["facies"]
            comp_grid.cell_data["Facies"] = comp_facies  # <--- Injeção explícita

            _, state_comp = run(
                mode=mode,
                z_exag=z_exag,
                show_scalar_bar=show_sb,
                external_plotter=self.comp_plotter_comp,
                external_state=self.compare_states["compare"],
                target_grid=comp_grid,
                target_facies=comp_facies
            )

            # Aplica seleção de reservatório do comparado
            rf_comp = self.models["compare"]["reservoir_facies"]
            state_comp["reservoir_facies"] = set(rf_comp)
            if "update_reservoir_fields" in state_comp:
                state_comp["update_reservoir_fields"](rf_comp)
                state_comp["refresh"]()
        
        # Ativa a sincronização direta entre as duas janelas
        self.install_compare_sync_callbacks()
        self.sync_compare_cameras()

    def sync_compare_cameras(self):
        """
        Sincroniza câmeras de forma robusta, verificando qual janela está ativa
        para evitar travamento dos widgets.
        """
        if not hasattr(self, "comp_plotter_base") or not hasattr(self, "comp_plotter_comp"):
            return

        plotter_base = self.comp_plotter_base
        plotter_comp = self.comp_plotter_comp
        
        # Referências
        cam_base = plotter_base.camera
        cam_comp = plotter_comp.camera
        
        # Interactors (para checar se o mouse está clicado)
        iren_base = plotter_base.iren
        iren_comp = plotter_comp.iren

        self._camera_syncing = False

        def copy_props(src, dst, dst_plotter):
            # Copia propriedades físicas
            dst.position = src.position
            dst.focal_point = src.focal_point
            dst.up = src.up
            dst.parallel_scale = src.parallel_scale
            dst.view_angle = src.view_angle
            
            # Copia Clipping Range com margem para não cortar bolinhas
            n, f = src.clipping_range
            dst.clipping_range = (n, f) 
            
            dst_plotter.render()

        def on_base_modified(obj, event):
            if self._camera_syncing: return
            
            # [CORREÇÃO CRÍTICA]: Só sincroniza se o mouse estiver ativo nesta janela
            # State 0 = Idle (parado). Se for != 0, usuário está interagindo.
            # Isso impede que a Base mande sinal se ela estiver apenas recebendo update.
            is_active = iren_base.get_interactor_style().GetState() != 0
            
            # Ou se for um evento de zoom (scroll não muda state as vezes, então deixamos passar)
            # Mas a proteção principal é a flag _camera_syncing
            
            self._camera_syncing = True
            try:
                copy_props(cam_base, cam_comp, plotter_comp)
            finally:
                self._camera_syncing = False

        def on_comp_modified(obj, event):
            if self._camera_syncing: return
            
            # Mesmo teste para o lado direito
            is_active = iren_comp.get_interactor_style().GetState() != 0
            
            self._camera_syncing = True
            try:
                copy_props(cam_comp, cam_base, plotter_base)
            finally:
                self._camera_syncing = False

        # Limpeza
        cam_base.RemoveAllObservers()
        cam_comp.RemoveAllObservers()

        # Conexão
        cam_base.AddObserver("ModifiedEvent", on_base_modified)
        cam_comp.AddObserver("ModifiedEvent", on_comp_modified)

        # Sync inicial
        self._camera_syncing = True
        copy_props(cam_base, cam_comp, plotter_comp)
        self._camera_syncing = False

    def update_compare_3d_mode(self):
        """Atualiza o modo (Fácies, Clusters, Thickness) nos dois lados."""
        mode = self.state.get("mode", "facies")
        
        for key in ["base", "compare"]:
            st = self.compare_states.get(key)
            if not st: continue
            
            st["mode"] = mode
            # Pega o estado da checkbox de scalar bar da UI principal ou do state principal
            st["show_scalar_bar"] = self.state.get("show_scalar_bar", False)
            
            # Garante consistência dos sub-modos
            if mode == "thickness_local":
                st["thickness_mode"] = self.state.get("thickness_mode", "Espessura")
                if "update_thickness" in st: st["update_thickness"]()
            
            # Se mudou para clusters ou reservatório, garante cálculo
            if mode in ["clusters", "reservoir", "largest"]:
                if "update_reservoir_fields" in st:
                    # Usa o conjunto de fácies específico DESTE modelo (base ou compare)
                    if key == "base":
                        rf = self.models["base"]["reservoir_facies"]
                    else:
                        rf = self.models["compare"]["reservoir_facies"]
                    st["update_reservoir_fields"](rf)

            if "refresh" in st:
                st["refresh"]()

    def install_compare_sync_callbacks(self):
        """
        Sincroniza o CORTE (Geometria) entre as janelas, mas NÃO move 
        os widgets passivos para evitar travamento de interação.
        """
        # Coleta todos os estados ativos
        states = []
        if self.state: states.append(self.state)
        if self.compare_states.get("base"): states.append(self.compare_states["base"])
        if self.compare_states.get("compare"): states.append(self.compare_states["compare"])

        # Coleta todos os plotters
        plotters = [self.plotter, self.comp_plotter_base, self.comp_plotter_comp]

        self._updating_cut = False

        def on_k_changed(new_k):
            # Sincronia de Camadas (Z)
            for st in states:
                st["k_min"] = int(new_k)
                if "refresh" in st: st["refresh"]()

        def on_box_changed(bounds):
            # Sincronia de Corte (Box)
            if self._updating_cut: return
            self._updating_cut = True
            
            try:
                # 1. Aplica o limite de corte (bounds) em TODOS os estados
                for st in states:
                    st["box_bounds"] = bounds
                    
                    # Chama o refresh() que recorta o grid internamente
                    if "refresh" in st: 
                        st["refresh"]()
                        
                    # NOTA: Removemos o 'PlaceWidget' daqui propositalmente.
                    # Isso faz com que o corte aconteça em todos, mas as 
                    # 'bolinhas' só se movam na janela ativa, destravando o mouse.

                # 2. Renderiza todas as telas para mostrar o corte novo
                for p in plotters: 
                    p.render()

            finally:
                self._updating_cut = False

        # Instala os callbacks
        for st in states:
            st["on_k_changed"] = on_k_changed
            st["on_box_changed"] = on_box_changed

    def update_compare_2d_maps(self):
        """
        Gera os mapas 2D lado a lado para a aba de comparação.
        Baseado na métrica selecionada no combo 'Thickness Local'.
        """
        # Pega a métrica atual
        if not hasattr(self, "state"): return
        
        # Mapeia o nome legível (Combo) para o nome interno do array
        presets = self.state.get("thickness_presets", {})
        mode_label = self.state.get("thickness_mode", "Espessura")
        
        if mode_label not in presets:
            mode_label = "Espessura"
        
        if mode_label not in presets: return # Segurança

        scalar_name, title = presets[mode_label]

        # 1. Mapa BASE
        # Precisamos acessar o grid que está sendo usado no plotter 3D da base
        if self.compare_states.get("base"):
            grid_base = self.compare_states["base"].get("current_grid_source")
            # Se não achar no state, tenta o global importado
            if not grid_base:
                from load_data import grid as grid_base
            
            # Só desenha se o array existir no grid
            if grid_base and scalar_name in grid_base.cell_data:
                # Truque: update_2d_plot espera um grid GLOBAL no analysis.py.
                # Vamos usar uma versão modificada ou injetar temporariamente?
                # Melhor: Vamos usar a lógica do visualize.py mas passando o grid explicitamente.
                # Como update_2d_plot do visualize.py usa 'make_thickness_2d_from_grid' que usa 'grid' global,
                # precisamos de uma versão que aceite grid como argumento
                
                # Vamos fazer a lógica local aqui mesmo para não mexer em outros arquivos
                self._draw_2d_map_local(self.comp_plotter_base_2d, grid_base, scalar_name, title)
            else:
                self.comp_plotter_base_2d.clear()

        # 2. Mapa COMPARADO
        if self.compare_states.get("compare"):
            grid_comp = self.compare_states["compare"].get("current_grid_source")
            if grid_comp and scalar_name in grid_comp.cell_data:
                self._draw_2d_map_local(self.comp_plotter_comp_2d, grid_comp, scalar_name, title)
            else:
                self.comp_plotter_comp_2d.clear()

    def _draw_2d_map_local(self, plotter, grid_source, scalar_name_3d, title):
        """
        Versão local de desenho 2D que aceita qualquer grid de entrada.
        """
        # Requer imports locais
        from load_data import nx, ny, nz
        import pyvista as pv
        import numpy as np

        if scalar_name_3d not in grid_source.cell_data:
            plotter.clear()
            return

        arr3d = grid_source.cell_data[scalar_name_3d].reshape((nx, ny, nz), order="F")

        # Calcula espessura 2D (max da coluna)
        thickness_2d = np.full((nx, ny), np.nan, dtype=float)
        for ix in range(nx):
            for iy in range(ny):
                col_vals = arr3d[ix, iy, :]
                col_vals = col_vals[col_vals > 0]
                if col_vals.size > 0:
                    thickness_2d[ix, iy] = col_vals.max()

        # Cria grid estruturado 2D
        x_min, x_max, y_min, y_max, _, z_max = grid_source.bounds
        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)
        xs, ys = np.meshgrid(xs, ys, indexing="ij")
        zs = np.full_like(xs, z_max)
        surf = pv.StructuredGrid(xs, ys, zs)

        # Injeta dados
        scalar_2d_name = scalar_name_3d + "_2d"
        surf.cell_data[scalar_2d_name] = thickness_2d[:nx-1, :ny-1].ravel(order="F")

        # Renderiza
        plotter.clear()
        plotter.add_mesh(surf, scalars=scalar_2d_name, cmap="plasma", show_edges=True, 
                         edge_color="black", line_width=0.5, nan_color="white", show_scalar_bar=False)
        plotter.view_xy()
        plotter.enable_parallel_projection()
        plotter.set_background("white")
        plotter.add_scalar_bar(title=title)
        plotter.reset_camera()









