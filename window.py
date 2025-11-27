from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QColor, QBrush, QPixmap
from pyvistaqt import BackgroundPlotter
import numpy as np
import os

from visualize import run, update_2d_plot, get_2d_clim
from load_data import facies, nx, ny, nz, load_facies_from_grdecl
from config import load_facies_colors
from analysis import (
    compute_global_metrics,
    compute_directional_percolation,
    facies_distribution_array,
    reservoir_facies_distribution_array,
    compute_global_metrics_for_array,
)

# --- WIDGET CUSTOMIZADO PARA OS SLIDERS (Grid Explorer) ---
class GridSlicerWidget(QtWidgets.QGroupBox):
    def __init__(self, nx, ny, nz, callback):
        super().__init__("Grid Explorer (Cortes)")
        self.callback = callback # function(axis, mode, value)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # I Axis
        layout.addWidget(QtWidgets.QLabel(f"I (0 - {nx-1}):"))
        self.i_widgets = self._create_axis_control("i", nx)
        layout.addLayout(self.i_widgets['layout'])
        
        # J Axis
        layout.addWidget(QtWidgets.QLabel(f"J (0 - {ny-1}):"))
        self.j_widgets = self._create_axis_control("j", ny)
        layout.addLayout(self.j_widgets['layout'])
        
        # K Axis
        layout.addWidget(QtWidgets.QLabel(f"K (0 - {nz-1}):"))
        self.k_widgets = self._create_axis_control("k", nz)
        layout.addLayout(self.k_widgets['layout'])
        
        self.is_updating = False

    def _create_axis_control(self, axis, limit):
        h_layout = QtWidgets.QHBoxLayout()
        
        # --- Controles Mínimo ---
        # lbl_min = QtWidgets.QLabel("Min:")
        spin_min = QtWidgets.QSpinBox()
        spin_min.setRange(0, limit-1)
        spin_min.setValue(0)
        spin_min.setFixedWidth(50)
        
        slider_min = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider_min.setRange(0, limit-1)
        slider_min.setValue(0)
        # Estilo opcional: deixa o slider min levemente diferente se quiser
        # slider_min.setStyleSheet("QSlider::handle:horizontal {background-color: #999;}") 
        
        # --- Controles Máximo ---
        # lbl_max = QtWidgets.QLabel("Max:")
        spin_max = QtWidgets.QSpinBox()
        spin_max.setRange(0, limit-1)
        spin_max.setValue(limit-1)
        spin_max.setFixedWidth(50)
        
        slider_max = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider_max.setRange(0, limit-1)
        slider_max.setValue(limit-1)
        slider_max.setInvertedAppearance(True) # Inverte visualmente para indicar 'fim'

        # --- Lógica de Conexão e Proteção ---
        def update_min(val):
            if self.is_updating: return
            self.is_updating = True
            
            # Proteção: Min não pode ser maior que Max
            current_max = spin_max.value()
            if val > current_max:
                val = current_max
            
            # Sincroniza spin e slider
            spin_min.setValue(val)
            slider_min.setValue(val)
            
            self.callback(axis, "min", val)
            self.is_updating = False

        def update_max(val):
            if self.is_updating: return
            self.is_updating = True
            
            # Proteção: Max não pode ser menor que Min
            current_min = spin_min.value()
            if val < current_min:
                val = current_min
            
            # Sincroniza spin e slider
            spin_max.setValue(val)
            slider_max.setValue(val)
            
            self.callback(axis, "max", val)
            self.is_updating = False

        # Conecta sinais
        spin_min.valueChanged.connect(update_min)
        slider_min.valueChanged.connect(update_min)
        
        spin_max.valueChanged.connect(update_max)
        slider_max.valueChanged.connect(update_max)
        
        # Monta o layout horizontal
        # h_layout.addWidget(lbl_min)
        h_layout.addWidget(spin_min)
        h_layout.addWidget(slider_min)
        h_layout.addSpacing(10)
        h_layout.addWidget(slider_max)
        h_layout.addWidget(spin_max)
        # h_layout.addWidget(lbl_max)
        
        return {
            'layout': h_layout,
            'spin_min': spin_min, 'slider_min': slider_min,
            'spin_max': spin_max, 'slider_max': slider_max
        }

    def external_update(self, axis, mode, value):
        """Chamado quando o teclado altera o slice, para atualizar a UI"""
        if self.is_updating: return
        self.is_updating = True
        
        widgets = getattr(self, f"{axis}_widgets")
        val = int(value)
        
        if mode == "min":
            widgets['spin_min'].setValue(val)
            widgets['slider_min'].setValue(val)
        else:
            widgets['spin_max'].setValue(val)
            widgets['slider_max'].setValue(val)
            
        self.is_updating = False


def make_facies_table():
    table = QtWidgets.QTableWidget()
    table.setColumnCount(4)
    table.setHorizontalHeaderLabels(["Cor", "Fácies", "Células", "Sel."])
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
    table.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
    table.setMinimumWidth(250)
    table.setMaximumWidth(320)
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
        item_color = QtWidgets.QTableWidgetItem()
        item_color.setBackground(QBrush(QColor(r, g, b)))
        item_color.setFlags(QtCore.Qt.ItemIsEnabled)
        table.setItem(row, 0, item_color)
        item_f = QtWidgets.QTableWidgetItem(str(fac))
        item_f.setFlags(QtCore.Qt.ItemIsEnabled)
        table.setItem(row, 1, item_f)
        item_c = QtWidgets.QTableWidgetItem(str(count_dict[fac]))
        item_c.setFlags(QtCore.Qt.ItemIsEnabled)
        table.setItem(row, 2, item_c)
        check = QtWidgets.QTableWidgetItem()
        check.setFlags(check.flags() | QtCore.Qt.ItemIsUserCheckable)
        check.setCheckState(QtCore.Qt.Checked if fac in reservoir_set else QtCore.Qt.Unchecked)
        check.setData(QtCore.Qt.UserRole, fac)
        table.setItem(row, 3, check)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, mode, z_exag, show_scalar_bar, reservoir_facies):
        super().__init__()
        self.setWindowTitle("SFM View Analysis")

        if isinstance(reservoir_facies, (int, np.integer)):
            initial_reservoir = {int(reservoir_facies)}
        else:
            initial_reservoir = {int(f) for f in reservoir_facies}

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
        self.panel.setMaximumWidth(320) 

        logo_label = QtWidgets.QLabel()
        pix = QPixmap("assets/forward_PNG.png")
        if not pix.isNull():
            pix = pix.scaledToWidth(140, QtCore.Qt.SmoothTransformation)
            logo_label.setPixmap(pix)
        logo_label.setAlignment(QtCore.Qt.AlignCenter)
        panel_layout.addWidget(logo_label)
        panel_layout.addSpacing(10)

        # NOVO: Grid Explorer Widget
        self.slicer_widget = GridSlicerWidget(nx, ny, nz, self.on_ui_slice_changed)
        panel_layout.addWidget(self.slicer_widget)

        modes = {
            "Fácies": "facies", "Reservatório": "reservoir", "Clusters": "clusters",
            "Maior Cluster": "largest", "Espessura local": "thickness_local", "NTG local": "ntg_local",
        }
        group = QtWidgets.QGroupBox("Modo de Visualização")
        group_layout = QtWidgets.QVBoxLayout(group)
        for label, mode_code in modes.items():
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(lambda _, mm=mode_code: self.change_mode(mm))
            group_layout.addWidget(btn)
        panel_layout.addWidget(group)

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

        self.legend_group = QtWidgets.QGroupBox("Legenda de Fácies")
        legend_layout = QtWidgets.QVBoxLayout(self.legend_group)
        self.facies_legend_table = self._create_legend_table()
        legend_layout.addWidget(self.facies_legend_table)
        panel_layout.addWidget(self.legend_group)
        panel_layout.addStretch()

        # --- ÁREA DIREITA ---
        self.tabs = QtWidgets.QTabWidget()
        self.plotter = BackgroundPlotter(show=False)
        self.viz_tab = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(self.viz_tab)
        vl.setContentsMargins(0,0,0,0)
        vl.addWidget(self.plotter.interactor)
        self.tabs.addTab(self.viz_tab, "Visualização 3D")
        
        self.plotter_2d = BackgroundPlotter(show=False)
        self.map2d_tab = QtWidgets.QWidget()
        ml = QtWidgets.QVBoxLayout(self.map2d_tab)
        ml.setContentsMargins(0,0,0,0)
        ml.addWidget(self.plotter_2d.interactor)
        self.tabs.addTab(self.map2d_tab, "Mapa 2D")

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

        # ABA COMPARAÇÃO
        self.compare_tab = QtWidgets.QWidget()
        comp_layout = QtWidgets.QVBoxLayout(self.compare_tab)
        self.compare_tabs = QtWidgets.QTabWidget()
        comp_layout.addWidget(self.compare_tabs)
        
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
        self.facies_compare_table.setColumnCount(9)
        self.facies_compare_table.setHorizontalHeaderLabels([
            "Fácies", "Cél. Base", "% Base", "Cél. Comp", "% Comp",
            "Vol. Base", "Vol. Comp", "Esp. Base", "Esp. Comp"
        ])
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

        self.compare_2d_widget = QtWidgets.QWidget()
        c2d_layout = QtWidgets.QHBoxLayout(self.compare_2d_widget)
        l_col_2d = QtWidgets.QVBoxLayout()
        self.comp_plotter_base_2d = BackgroundPlotter(show=False)
        l_col_2d.addWidget(QtWidgets.QLabel("Mapa Base"))
        l_col_2d.addWidget(self.comp_plotter_base_2d.interactor)
        c2d_layout.addLayout(l_col_2d)
        r_col_2d = QtWidgets.QVBoxLayout()
        self.comp_plotter_comp_2d = BackgroundPlotter(show=False)
        r_col_2d.addWidget(QtWidgets.QLabel("Mapa Comparado"))
        r_col_2d.addWidget(self.comp_plotter_comp_2d.interactor)
        c2d_layout.addLayout(r_col_2d)
        self.compare_tabs.addTab(self.compare_2d_widget, "Mapas 2D")

        self.compare_3d_widget = QtWidgets.QWidget()
        c3d_main_layout = QtWidgets.QVBoxLayout(self.compare_3d_widget)
        c3d_main_layout.setContentsMargins(0,0,0,0)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        c3d_main_layout.addWidget(splitter)

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

        # Sincronização UI <-> Plotter
        self.state["on_slice_changed"] = self.on_plotter_slice_changed

        self.change_reservoir_facies(None) 
        self.update_2d_map()
        self.update_comparison_tables()
        self.init_compare_3d()
        self.update_compare_2d_maps()

    def on_ui_slice_changed(self, axis, mode, value):
        """Chamado quando o usuário mexe nos sliders."""
        if "set_slice" in self.state:
            self.state["set_slice"](axis, mode, value)
            
        # Atualiza também as views de comparação se existirem
        self.sync_slices_to_compare(axis, mode, value)

    def on_plotter_slice_changed(self, axis, mode, value):
        """Chamado quando o usuário usa o teclado."""
        self.slicer_widget.external_update(axis, mode, value)
        self.sync_slices_to_compare(axis, mode, value)

    def sync_slices_to_compare(self, axis, mode, value):
        # Replica o corte para as janelas de comparação
        if self.compare_states.get("base") and "set_slice" in self.compare_states["base"]:
            self.compare_states["base"]["set_slice"](axis, mode, value)
            
        if self.compare_states.get("compare") and "set_slice" in self.compare_states["compare"]:
            self.compare_states["compare"]["set_slice"](axis, mode, value)

    def _create_legend_table(self, headers=["Cor", "Fácies", "Células", "Sel."]):
        table = QtWidgets.QTableWidget()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        table.setShowGrid(False)
        table.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        table.setMaximumWidth(320)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        return table

    def populate_facies_legend(self):
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
        if not hasattr(self, "state"): return
        sizes_dict = self.state.get("clusters_sizes")
        lut = self.state.get("clusters_lut")
        table = self.facies_legend_table
        if not sizes_dict or lut is None:
            table.setRowCount(0)
            return
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
        if self.compare_states.get("base"):
            fill_one_table(self.clus_table_base_cmp, self.compare_states["base"])
        if self.compare_states.get("compare"):
            fill_one_table(self.clus_table_comp_cmp, self.compare_states["compare"])

    def change_mode(self, new_mode):
        print("Modo:", new_mode)
        self.state["mode"] = new_mode
        if new_mode == "clusters":
            self.legend_group.setTitle("Legenda de Clusters")
            self.populate_clusters_legend()
        else:
            self.legend_group.setTitle("Legenda de Fácies")
            self.populate_facies_legend()

        is_cluster = (new_mode == "clusters")
        self.clus_table_base_cmp.setVisible(is_cluster)
        self.clus_table_comp_cmp.setVisible(is_cluster)
        self.res_table_base_cmp.setVisible(True)
        self.res_table_comp_cmp.setVisible(True)
        if is_cluster:
            self.populate_compare_clusters_tables()
        self.state["refresh"]()
        self.update_compare_3d_mode()

    def change_thickness_mode(self, label):
        print("Thickness:", label)
        self.state["thickness_mode"] = label
        if "update_thickness" in self.state:
            self.state["update_thickness"]()
        self.state["refresh"]()
        for key in ["base", "compare"]:
            st = self.compare_states.get(key)
            if st:
                st["thickness_mode"] = label
                if "update_thickness" in st: 
                    st["update_thickness"]()
                if "refresh" in st: 
                    st["refresh"]()
        self.update_2d_map()
        self.update_compare_2d_maps()

    def set_metrics(self, metrics, perc):
        if metrics is None:
            self.metrics_text.setPlainText("Nenhuma análise calculada.")
            return
        self.base_metrics = metrics
        self.base_perc = perc
        def fmt_clusters(arr):
            if arr is None: return "[]"
            try: return "[" + ", ".join(str(int(c)) for c in arr) + "]"
            except TypeError: return "[" + str(int(arr)) + "]"
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
        vol_grid = metrics.get('grid_volume', 0.0)
        vol_res = metrics.get('reservoir_volume', 0.0)
        lines.append(f"Volume Grid Total    : {vol_grid:.2e} m³")
        lines.append(f"Volume Reservatório   : {vol_res:.2e} m³")
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
            lines.append(f"Xmin→Xmax: {x_ok}   clusters conectados = {fmt_clusters(perc['x_clusters'])}")
            lines.append(f"Ymin→Ymax: {y_ok}   clusters conectados = {fmt_clusters(perc['y_clusters'])}")
            lines.append(f"Topo→Base: {z_ok}   clusters conectados = {fmt_clusters(perc['z_clusters'])}")
        self.metrics_text.setPlainText("\n".join(lines))
        self.init_compare_3d()
        self.update_comparison_tables()

    def set_facies_metrics(self, df):
        pretty = {
            "facies": "Fácies", "cells": "Células", "fraction": "Fração no grid",
            "n_clusters": "Nº de clusters", "largest_label": "ID maior cluster",
            "largest_size": "Tamanho maior cluster (células)", "connected_fraction": "Fração conectada",
            "volume_total": "Volume total", "volume_largest_cluster": "Volume maior cluster",
            "thickness_largest_cluster": "Espessura maior cluster",
            "Perc_X": "Percolação X", "Perc_Y": "Percolação Y", "Perc_Z": "Percolação Z",
        }
        self.facies_table.setRowCount(len(df))
        self.facies_table.setColumnCount(len(df.columns))
        headers = [pretty.get(col, col) for col in df.columns]
        self.facies_table.setHorizontalHeaderLabels(headers)
        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                val = df.iloc[i][col]
                if isinstance(val, (float, np.floating)):
                    if col in ("fraction", "connected_fraction", "Perc_X", "Perc_Y", "Perc_Z"): text = f"{val:.3f}"
                    elif "thickness" in col: text = f"{val:.2f}"
                    else: text = f"{val:.1f}"
                elif isinstance(val, (int, np.integer)): text = str(int(val))
                else: text = str(val)
                item = QtWidgets.QTableWidgetItem(text)
                self.facies_table.setItem(i, j, item)
        self.facies_table.resizeColumnsToContents()

    def change_reservoir_facies(self, item):
        selected = []
        for i in range(self.reservoir_list.count()):
            it = self.reservoir_list.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                selected.append(int(it.data(QtCore.Qt.UserRole)))
        selected_set = set(selected)
        self.state["reservoir_facies"] = selected_set
        if "update_reservoir_fields" in self.state:
            self.state["update_reservoir_fields"](selected_set)
        if "refresh" in self.state:
            self.state["refresh"]()
        self.update_2d_map()
        base_metrics = compute_global_metrics(selected_set)
        base_perc = compute_directional_percolation(selected_set)
        base_res_stats, base_res_total = reservoir_facies_distribution_array(facies, selected_set)
        self.base_metrics = base_metrics
        self.base_perc = base_perc
        self.base_res_stats = base_res_stats
        self.base_res_total = base_res_total
        self.set_metrics(base_metrics, base_perc)
        if self.compare_facies is not None:
            c_metrics, c_perc = compute_global_metrics_for_array(self.compare_facies, selected_set)
            c_res_stats, c_res_total = reservoir_facies_distribution_array(self.compare_facies, selected_set)
            self.compare_metrics = c_metrics
            self.compare_perc = c_perc
            self.comp_res_stats = c_res_stats
            self.comp_res_total = c_res_total
        self.update_comparison_tables()

    def update_2d_map(self):
        if not hasattr(self, "plotter_2d"): return
        presets = self.state.get("thickness_presets") or {}
        mode_label = self.state.get("thickness_mode", "Espessura")
        if mode_label not in presets:
            if "Espessura" in presets: mode_label = "Espessura"
            else: return
        scalar_name, title = presets[mode_label]
        try: update_2d_plot(self.plotter_2d, scalar_name, title)
        except Exception as e: print("Erro ao atualizar mapa 2D:", e)

    def load_compare_model(self, grdecl_path: str):
        try:
            fac_compare = load_facies_from_grdecl(grdecl_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erro", f"Falha ao ler arquivo:\n{e}")
            return
        expected = nx * ny * nz
        if fac_compare.size != expected:
            QtWidgets.QMessageBox.warning(self, "Erro de Dimensão", f"O modelo carregado tem {fac_compare.size} células, mas o grid base espera {expected} ({nx}x{ny}x{nz}).")
            return
        self.models["compare"]["name"] = grdecl_path
        self.models["compare"]["facies"] = fac_compare
        self.models["compare"]["reservoir_facies"] = set(self.models["base"]["reservoir_facies"])
        self.compare_path = grdecl_path
        self.compare_facies = fac_compare
        
        self.compare_facies_stats, self.compare_total_cells = facies_distribution_array(fac_compare)
        
        rf = self.state.get("reservoir_facies", set())
        if rf:
            self.compare_metrics, self.compare_perc = compute_global_metrics_for_array(fac_compare, rf)
            self.comp_res_stats, self.comp_res_total = reservoir_facies_distribution_array(fac_compare, rf)
        else:
            self.compare_metrics, self.compare_perc = None, None
            self.comp_res_stats, self.comp_res_total = {}, 0
        print(f"\n[COMPARE] Modelo '{os.path.basename(grdecl_path)}' carregado e calculado.")
        self.comp_model_label.setText(f"Modelo comparado: {os.path.basename(grdecl_path)}")
        self.populate_compare_facies_lists()
        self.update_comparison_tables()
        self.init_compare_3d()
        self.update_compare_2d_maps()

    def open_compare_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Selecionar modelo para comparar", "assets", "GRDECL (*.grdecl);;Todos os arquivos (*)")
        if not path: return
        self.load_compare_model(path)

    def update_comparison_tables(self):
        # 1) Métricas Globais
        m0 = getattr(self, "base_metrics", None)
        m1 = getattr(self, "compare_metrics", None)
        p0 = getattr(self, "base_perc", None)
        p1 = getattr(self, "compare_perc", None)
        rows = []
        def get(m, key): return None if m is None else m.get(key)
        rows.append(("NTG global", get(m0, "ntg"), get(m1, "ntg")))
        rows.append(("Total de células", get(m0, "total_cells"), get(m1, "total_cells")))
        rows.append(("Células reservatório", get(m0, "res_cells"), get(m1, "res_cells")))
        rows.append(("Nº de clusters", get(m0, "n_clusters"), get(m1, "n_clusters")))
        rows.append(("Tamanho do maior cluster", get(m0, "largest_size"), get(m1, "largest_size")))
        rows.append(("Fração conectada", get(m0, "connected_fraction"), get(m1, "connected_fraction")))
        def flag(per, key):
            if per is None: return None
            return "Sim" if per.get(key) else "Não"
        if p0 is not None:
            rows.append(("Percolação X (Sim/Não)", flag(p0, "x_perc"), flag(p1, "x_perc")))
            rows.append(("Percolação Y (Sim/Não)", flag(p0, "y_perc"), flag(p1, "y_perc")))
            rows.append(("Percolação Z (Sim/Não)", flag(p0, "z_perc"), flag(p1, "z_perc")))
        self.global_compare_table.setRowCount(len(rows))
        def fmt_num(val):
            if val is None: return "-"
            if isinstance(val, (float, np.floating)): return f"{val:.3f}"
            return str(val)
        for i, (label, a, b) in enumerate(rows):
            self.global_compare_table.setItem(i, 0, QtWidgets.QTableWidgetItem(label))
            self.global_compare_table.setItem(i, 1, QtWidgets.QTableWidgetItem(fmt_num(a)))
            self.global_compare_table.setItem(i, 2, QtWidgets.QTableWidgetItem(fmt_num(b)))
            if isinstance(a, (int, np.integer, float, np.floating)) and isinstance(b, (int, np.integer, float, np.floating)):
                try:
                    diff = float(b) - float(a)
                    self.global_compare_table.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{diff:.3f}"))
                except: self.global_compare_table.setItem(i, 3, QtWidgets.QTableWidgetItem("-"))
            else: self.global_compare_table.setItem(i, 3, QtWidgets.QTableWidgetItem("-"))
        self.global_compare_table.resizeColumnsToContents()

        # 2) Distribuição de Fácies Grid Inteiro
        stats0 = getattr(self, "base_facies_stats", None)
        stats1 = getattr(self, "compare_facies_stats", None)
        if stats0 is None: return
        facs_union = sorted(set(stats0.keys()) | (set(stats1.keys()) if stats1 else set()))
        self.facies_compare_table.setRowCount(len(facs_union))
        
        for row, fac in enumerate(facs_union):
            s0 = stats0.get(fac, {"cells": 0, "fraction": 0.0, "volume": 0.0, "thickness_gross": 0.0})
            s1 = stats1.get(fac, {"cells": 0, "fraction": 0.0, "volume": 0.0, "thickness_gross": 0.0}) if stats1 else {"cells": 0, "fraction": 0.0, "volume": 0.0, "thickness_gross": 0.0}

            self.facies_compare_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(fac)))
            self.facies_compare_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(s0["cells"])))
            self.facies_compare_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{100*s0['fraction']:.2f}"))
            self.facies_compare_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(s1["cells"])))
            self.facies_compare_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{100*s1['fraction']:.2f}"))
            
            vol0 = s0.get("volume", 0.0)
            vol1 = s1.get("volume", 0.0)
            esp0 = s0.get("thickness_gross", 0.0)
            esp1 = s1.get("thickness_gross", 0.0)

            self.facies_compare_table.setItem(row, 5, QtWidgets.QTableWidgetItem(f"{vol0:,.0f}"))
            self.facies_compare_table.setItem(row, 6, QtWidgets.QTableWidgetItem(f"{vol1:,.0f}"))
            self.facies_compare_table.setItem(row, 7, QtWidgets.QTableWidgetItem(f"{esp0:.1f}"))
            self.facies_compare_table.setItem(row, 8, QtWidgets.QTableWidgetItem(f"{esp1:.1f}"))

        self.facies_compare_table.resizeColumnsToContents()

        # 3) Distribuição Fácies Reservatório
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
        fill_facies_table(self.res_table_base_cmp, self.models["base"]["facies"], self.models["base"]["reservoir_facies"])
        if self.models["compare"]["facies"] is not None:
            fill_facies_table(self.res_table_comp_cmp, self.models["compare"]["facies"], self.models["compare"]["reservoir_facies"])

    def update_base_reservoir_compare(self, item):
        if item.column() != 3: return
        f = int(item.data(QtCore.Qt.UserRole))
        if item.checkState() == QtCore.Qt.Checked: self.models["base"]["reservoir_facies"].add(f)
        else: self.models["base"]["reservoir_facies"].discard(f)
        self.update_compare_3d_mode()
        self.update_compare_2d_maps()

    def update_compare_reservoir_compare(self, item):
        if item.column() != 3: return
        f = int(item.data(QtCore.Qt.UserRole))
        if item.checkState() == QtCore.Qt.Checked: self.models["compare"]["reservoir_facies"].add(f)
        else: self.models["compare"]["reservoir_facies"].discard(f)
        self.update_compare_3d_mode()
        self.update_compare_2d_maps()

    def init_compare_3d(self):
        if self.models["base"]["facies"] is None: return
        from visualize import run
        from load_data import grid as global_grid
        mode = self.state.get("mode", "facies")
        z_exag = self.state.get("z_exag", 15.0)
        show_sb = self.state.get("show_scalar_bar", False)
        
        self.comp_plotter_base.clear()
        self.compare_states["base"] = {}
        _, state_base = run(
            mode=mode, z_exag=z_exag, show_scalar_bar=show_sb,
            external_plotter=self.comp_plotter_base, external_state=self.compare_states["base"],
            target_grid=global_grid, target_facies=self.models["base"]["facies"]
        )
        rf_base = self.models["base"]["reservoir_facies"]
        state_base["reservoir_facies"] = set(rf_base)
        if "update_reservoir_fields" in state_base:
            state_base["update_reservoir_fields"](rf_base)
            state_base["refresh"]()

        self.comp_plotter_comp.clear()
        self.compare_states["compare"] = {}
        if self.models["compare"]["facies"] is not None:
            comp_grid = global_grid.copy(deep=True)
            comp_facies = self.models["compare"]["facies"]
            comp_grid.cell_data["Facies"] = comp_facies
            _, state_comp = run(
                mode=mode, z_exag=z_exag, show_scalar_bar=show_sb,
                external_plotter=self.comp_plotter_comp, external_state=self.compare_states["compare"],
                target_grid=comp_grid, target_facies=comp_facies
            )
            rf_comp = self.models["compare"]["reservoir_facies"]
            state_comp["reservoir_facies"] = set(rf_comp)
            if "update_reservoir_fields" in state_comp:
                state_comp["update_reservoir_fields"](rf_comp)
                state_comp["refresh"]()
        
        self.install_compare_sync_callbacks()
        self.sync_compare_cameras()

    def sync_compare_cameras(self):
        if not hasattr(self, "comp_plotter_base") or not hasattr(self, "comp_plotter_comp"): return
        plotter_base = self.comp_plotter_base
        plotter_comp = self.comp_plotter_comp
        cam_base = plotter_base.camera
        cam_comp = plotter_comp.camera
        iren_base = plotter_base.iren
        iren_comp = plotter_comp.iren
        self._camera_syncing = False

        def copy_props(src, dst, dst_plotter):
            dst.position = src.position
            dst.focal_point = src.focal_point
            dst.up = src.up
            dst.parallel_scale = src.parallel_scale
            dst.view_angle = src.view_angle
            n, f = src.clipping_range
            dst.clipping_range = (n, f) 
            dst_plotter.render()

        def on_base_modified(obj, event):
            if self._camera_syncing: return
            is_active = iren_base.get_interactor_style().GetState() != 0
            self._camera_syncing = True
            try: copy_props(cam_base, cam_comp, plotter_comp)
            finally: self._camera_syncing = False

        def on_comp_modified(obj, event):
            if self._camera_syncing: return
            is_active = iren_comp.get_interactor_style().GetState() != 0
            self._camera_syncing = True
            try: copy_props(cam_comp, cam_base, plotter_base)
            finally: self._camera_syncing = False

        cam_base.RemoveAllObservers()
        cam_comp.RemoveAllObservers()
        cam_base.AddObserver("ModifiedEvent", on_base_modified)
        cam_comp.AddObserver("ModifiedEvent", on_comp_modified)
        self._camera_syncing = True
        copy_props(cam_base, cam_comp, plotter_comp)
        self._camera_syncing = False

    def update_compare_3d_mode(self):
        mode = self.state.get("mode", "facies")
        for key in ["base", "compare"]:
            st = self.compare_states.get(key)
            if not st: continue
            st["mode"] = mode
            st["show_scalar_bar"] = self.state.get("show_scalar_bar", False)
            if mode == "thickness_local":
                st["thickness_mode"] = self.state.get("thickness_mode", "Espessura")
                if "update_thickness" in st: st["update_thickness"]()
            if mode in ["clusters", "reservoir", "largest"]:
                if "update_reservoir_fields" in st:
                    if key == "base": rf = self.models["base"]["reservoir_facies"]
                    else: rf = self.models["compare"]["reservoir_facies"]
                    st["update_reservoir_fields"](rf)
            if "refresh" in st: st["refresh"]()

    def install_compare_sync_callbacks(self):
        states = []
        if self.state: states.append(self.state)
        if self.compare_states.get("base"): states.append(self.compare_states["base"])
        if self.compare_states.get("compare"): states.append(self.compare_states["compare"])
        plotters = [self.plotter, self.comp_plotter_base, self.comp_plotter_comp]
        self._updating_cut = False

        # CALLBACK UNIFICADO PARA CORTES (I, J, K) - Sincroniza UI e visualizadores
        def on_slice_changed(axis, mode, value):
            if self._updating_cut: return
            self._updating_cut = True
            try:
                for st in states:
                    key = f"{axis}_{mode}"
                    st[key] = int(value)
                    
                    if "refresh" in st: st["refresh"]()
                for p in plotters: p.render()
            finally:
                self._updating_cut = False

        for st in states:
            st["on_slice_changed"] = on_slice_changed

    def update_compare_2d_maps(self):
        if not hasattr(self, "state"): return
        presets = self.state.get("thickness_presets", {})
        mode_label = self.state.get("thickness_mode", "Espessura")
        if mode_label not in presets: mode_label = "Espessura"
        if mode_label not in presets: return
        scalar_name, title = presets[mode_label]

        if self.compare_states.get("base"):
            grid_base = self.compare_states["base"].get("current_grid_source")
            if not grid_base: from load_data import grid as grid_base
            if grid_base and scalar_name in grid_base.cell_data:
                self._draw_2d_map_local(self.comp_plotter_base_2d, grid_base, scalar_name, title)
            else: self.comp_plotter_base_2d.clear()

        if self.compare_states.get("compare"):
            grid_comp = self.compare_states["compare"].get("current_grid_source")
            if grid_comp and scalar_name in grid_comp.cell_data:
                self._draw_2d_map_local(self.comp_plotter_comp_2d, grid_comp, scalar_name, title)
            else: self.comp_plotter_comp_2d.clear()

    def _draw_2d_map_local(self, plotter, grid_source, scalar_name_3d, title):
        from load_data import nx, ny, nz
        import pyvista as pv
        import numpy as np

        if scalar_name_3d not in grid_source.cell_data:
            plotter.clear()
            return

        # 3D -> projeção 2D (máximo na coluna)
        arr3d = grid_source.cell_data[scalar_name_3d].reshape((nx, ny, nz), order="F")
        thickness_2d = np.full((nx, ny), np.nan, dtype=float)
        for ix in range(nx):
            for iy in range(ny):
                col_vals = arr3d[ix, iy, :]
                col_vals = col_vals[col_vals > 0]
                if col_vals.size > 0:
                    thickness_2d[ix, iy] = col_vals.max()

        x_min, x_max, y_min, y_max, _, z_max = grid_source.bounds
        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)
        xs, ys = np.meshgrid(xs, ys, indexing="ij")
        zs = np.full_like(xs, z_max)

        surf = pv.StructuredGrid(xs, ys, zs)
        scalar_2d_name = scalar_name_3d + "_2d"
        surf.cell_data[scalar_2d_name] = thickness_2d[:nx-1, :ny-1].ravel(order="F")

        # limpa negativos e aplica faixa de cores fixa
        arr = surf.cell_data[scalar_2d_name]
        arr = np.where(arr < 0, np.nan, arr)
        surf.cell_data[scalar_2d_name] = arr
        clim = get_2d_clim(scalar_name_3d, arr)

        plotter.clear()
        plotter.add_mesh(
            surf,
            scalars=scalar_2d_name,
            cmap="plasma",
            show_edges=True,
            edge_color="black",
            line_width=0.5,
            nan_color="white",
            show_scalar_bar=False,
            clim=clim,
        )
        plotter.view_xy()
        plotter.enable_parallel_projection()
        plotter.set_background("white")
        plotter.add_scalar_bar(title=title)
        plotter.reset_camera()
