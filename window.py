# window.py
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QColor, QBrush
from pyvistaqt import BackgroundPlotter
import numpy as np
import os
import pandas as pd
from scipy.ndimage import label, generate_binary_structure

from visualize import run, get_2d_clim, make_clusters_lut, compute_cluster_sizes
from load_data import facies, nx, ny, nz, load_facies_from_grdecl
from config import load_facies_colors
from analysis import (
    facies_distribution_array,
    reservoir_facies_distribution_array,
    compute_global_metrics_for_array,
    _get_cell_volumes,   # <--- Adicionado
    _get_cell_z_coords   # <--- Adicionado
)

# --- WIDGET CUSTOMIZADO PARA OS SLIDERS (Grid Explorer) ---
class GridSlicerWidget(QtWidgets.QGroupBox):
    def __init__(self, nx, ny, nz, callback, initial_z=15.0):
        super().__init__("Geometria (Cortes & Escala)")
        self.callback = callback 
        self.is_updating = False
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Eixo I
        layout.addWidget(QtWidgets.QLabel(f"Inline (I): 0 - {nx-1}"))
        self.i_widgets = self._create_axis_control("i", nx)
        layout.addLayout(self.i_widgets['layout'])
        
        # Eixo J
        layout.addWidget(QtWidgets.QLabel(f"Crossline (J): 0 - {ny-1}"))
        self.j_widgets = self._create_axis_control("j", ny)
        layout.addLayout(self.j_widgets['layout'])
        
        # Eixo K
        layout.addWidget(QtWidgets.QLabel(f"Layer (K): 0 - {nz-1}"))
        self.k_widgets = self._create_axis_control("k", nz)
        layout.addLayout(self.k_widgets['layout'])
        
        layout.addSpacing(10)
        
        # --- NOVO: EXAGERO VERTICAL (Z) ---
        layout.addWidget(QtWidgets.QLabel("Exagero Vertical (Z):"))
        h_z = QtWidgets.QHBoxLayout()
        
        self.spin_z = QtWidgets.QDoubleSpinBox()
        self.spin_z.setRange(1.0, 100.0)
        self.spin_z.setSingleStep(1.0)
        self.spin_z.setValue(initial_z)
        self.spin_z.setFixedWidth(60)
        
        self.slider_z = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_z.setRange(1, 100)
        self.slider_z.setValue(int(initial_z))
        
        # Conexões Z
        self.spin_z.valueChanged.connect(self._on_z_spin_change)
        self.slider_z.valueChanged.connect(self._on_z_slider_change)
        
        h_z.addWidget(self.spin_z)
        h_z.addWidget(self.slider_z)
        layout.addLayout(h_z)

    def _on_z_spin_change(self, val):
        if self.is_updating: return
        self.is_updating = True
        self.slider_z.setValue(int(val))
        self.callback("z", "scale", val)
        self.is_updating = False

    def _on_z_slider_change(self, val):
        if self.is_updating: return
        self.is_updating = True
        self.spin_z.setValue(float(val))
        self.callback("z", "scale", float(val))
        self.is_updating = False

    def _create_axis_control(self, axis, limit):
        # (Este método continua idêntico ao anterior)
        h_layout = QtWidgets.QHBoxLayout()
        spin_min = QtWidgets.QSpinBox(); spin_min.setRange(0, limit-1); spin_min.setValue(0); spin_min.setFixedWidth(50)
        slider_min = QtWidgets.QSlider(QtCore.Qt.Horizontal); slider_min.setRange(0, limit-1); slider_min.setValue(0)
        spin_max = QtWidgets.QSpinBox(); spin_max.setRange(0, limit-1); spin_max.setValue(limit-1); spin_max.setFixedWidth(50)
        slider_max = QtWidgets.QSlider(QtCore.Qt.Horizontal); slider_max.setRange(0, limit-1); slider_max.setValue(limit-1); slider_max.setInvertedAppearance(True)

        def update_min(val):
            if self.is_updating: return
            self.is_updating = True
            if val > spin_max.value(): val = spin_max.value()
            spin_min.setValue(val); slider_min.setValue(val)
            self.callback(axis, "min", val)
            self.is_updating = False

        def update_max(val):
            if self.is_updating: return
            self.is_updating = True
            if val < spin_min.value(): val = spin_min.value()
            spin_max.setValue(val); slider_max.setValue(val)
            self.callback(axis, "max", val)
            self.is_updating = False

        spin_min.valueChanged.connect(update_min); slider_min.valueChanged.connect(update_min)
        spin_max.valueChanged.connect(update_max); slider_max.valueChanged.connect(update_max)
        h_layout.addWidget(spin_min); h_layout.addWidget(slider_min); h_layout.addSpacing(5)
        h_layout.addWidget(slider_max); h_layout.addWidget(spin_max)
        return {'layout': h_layout, 'spin_min': spin_min, 'slider_min': slider_min, 'spin_max': spin_max, 'slider_max': slider_max}

    def external_update(self, axis, mode, value):
        if self.is_updating: return
        self.is_updating = True
        
        if axis == "z" and mode == "scale":
            self.spin_z.setValue(float(value))
            self.slider_z.setValue(int(value))
        else:
            widgets = getattr(self, f"{axis}_widgets")
            val = int(value)
            if mode == "min": widgets['spin_min'].setValue(val); widgets['slider_min'].setValue(val)
            else: widgets['spin_max'].setValue(val); widgets['slider_max'].setValue(val)
            
        self.is_updating = False

    def _create_axis_control(self, axis, limit):
        h_layout = QtWidgets.QHBoxLayout()
        
        spin_min = QtWidgets.QSpinBox()
        spin_min.setRange(0, limit-1)
        spin_min.setValue(0)
        spin_min.setFixedWidth(50)
        
        slider_min = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider_min.setRange(0, limit-1)
        slider_min.setValue(0)
        
        spin_max = QtWidgets.QSpinBox()
        spin_max.setRange(0, limit-1)
        spin_max.setValue(limit-1)
        spin_max.setFixedWidth(50)
        
        slider_max = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider_max.setRange(0, limit-1)
        slider_max.setValue(limit-1)
        slider_max.setInvertedAppearance(True)

        def update_min(val):
            if self.is_updating: return
            self.is_updating = True
            current_max = spin_max.value()
            if val > current_max: val = current_max
            spin_min.setValue(val)
            slider_min.setValue(val)
            self.callback(axis, "min", val)
            self.is_updating = False

        def update_max(val):
            if self.is_updating: return
            self.is_updating = True
            current_min = spin_min.value()
            if val < current_min: val = current_min
            spin_max.setValue(val)
            slider_max.setValue(val)
            self.callback(axis, "max", val)
            self.is_updating = False

        spin_min.valueChanged.connect(update_min)
        slider_min.valueChanged.connect(update_min)
        spin_max.valueChanged.connect(update_max)
        slider_max.valueChanged.connect(update_max)
        
        h_layout.addWidget(spin_min)
        h_layout.addWidget(slider_min)
        h_layout.addSpacing(5)
        h_layout.addWidget(slider_max)
        h_layout.addWidget(spin_max)
        
        return {
            'layout': h_layout,
            'spin_min': spin_min, 'slider_min': slider_min,
            'spin_max': spin_max, 'slider_max': slider_max
        }

    def external_update(self, axis, mode, value):
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

    def _create_axis_control(self, axis, limit):
        h_layout = QtWidgets.QHBoxLayout()
        
        spin_min = QtWidgets.QSpinBox()
        spin_min.setRange(0, limit-1)
        spin_min.setValue(0)
        spin_min.setFixedWidth(50)
        
        slider_min = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider_min.setRange(0, limit-1)
        slider_min.setValue(0)
        
        spin_max = QtWidgets.QSpinBox()
        spin_max.setRange(0, limit-1)
        spin_max.setValue(limit-1)
        spin_max.setFixedWidth(50)
        
        slider_max = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider_max.setRange(0, limit-1)
        slider_max.setValue(limit-1)
        slider_max.setInvertedAppearance(True)

        def update_min(val):
            if self.is_updating: return
            self.is_updating = True
            current_max = spin_max.value()
            if val > current_max: val = current_max
            spin_min.setValue(val)
            slider_min.setValue(val)
            self.callback(axis, "min", val)
            self.is_updating = False

        def update_max(val):
            if self.is_updating: return
            self.is_updating = True
            current_min = spin_min.value()
            if val < current_min: val = current_min
            spin_max.setValue(val)
            slider_max.setValue(val)
            self.callback(axis, "max", val)
            self.is_updating = False

        spin_min.valueChanged.connect(update_min)
        slider_min.valueChanged.connect(update_min)
        spin_max.valueChanged.connect(update_max)
        slider_max.valueChanged.connect(update_max)
        
        h_layout.addWidget(spin_min)
        h_layout.addWidget(slider_min)
        h_layout.addSpacing(5)
        h_layout.addWidget(slider_max)
        h_layout.addWidget(spin_max)
        
        return {
            'layout': h_layout,
            'spin_min': spin_min, 'slider_min': slider_min,
            'spin_max': spin_max, 'slider_max': slider_max
        }

    def external_update(self, axis, mode, value):
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

# --- HELPER FUNCTIONS ---
def make_facies_table():
    table = QtWidgets.QTableWidget()
    table.setColumnCount(4)
    table.setHorizontalHeaderLabels(["Cor", "Fácies", "Células", "Sel."])
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
    table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
    header = table.horizontalHeader()
    header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
    return table

def fill_facies_table(table, facies_array, reservoir_set):
    colors = load_facies_colors()
    vals, counts = np.unique(facies_array.astype(int), return_counts=True)
    count_dict = {int(v): int(c) for v, c in zip(vals, counts)}
    present = sorted(count_dict.keys())
    table.setRowCount(len(present))
    for row, fac in enumerate(present):
        rgba = colors.get(fac, (0.8, 0.8, 0.8, 1.0))
        r, g, b, a = [int(255*c) for c in rgba]
        item_color = QtWidgets.QTableWidgetItem()
        item_color.setBackground(QBrush(QColor(r, g, b)))
        table.setItem(row, 0, item_color)
        table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(fac)))
        table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(count_dict[fac])))
        check = QtWidgets.QTableWidgetItem()
        check.setFlags(check.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        check.setCheckState(QtCore.Qt.Checked if fac in reservoir_set else QtCore.Qt.Unchecked)
        check.setData(QtCore.Qt.UserRole, fac)
        table.setItem(row, 3, check)

# --- CLASSE PRINCIPAL ---
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, mode, z_exag, show_scalar_bar, reservoir_facies):
        super().__init__()
        self.setWindowTitle("SFM View Analysis - Professional Edition")
        
        # --- 1. DADOS E ESTADO INICIAL ---
        if isinstance(reservoir_facies, (int, np.integer)):
            initial_reservoir = {int(reservoir_facies)}
        else:
            initial_reservoir = {int(f) for f in reservoir_facies}

        self.models = {
            "base": {"name": "Modelo Base", "facies": facies, "reservoir_facies": set(initial_reservoir)},
            "compare": {"name": None, "facies": None, "reservoir_facies": set()},
        }
        
        # Cache de métricas (inclui dataframe da tabela detalhada)
        self.cached_metrics = {
            "base": {"metrics": None, "perc": None, "df": None},
            "compare": {"metrics": None, "perc": None, "df": None}
        }

        self.state = {"reservoir_facies": initial_reservoir}
        self.compare_states = {"base": {}, "compare": {}}
        self.base_facies_stats, self.base_total_cells = facies_distribution_array(facies)
        self.compare_path = None
        self.compare_facies = None
        self.compare_metrics = None
        
        # --- 2. CONSTRUÇÃO DA INTERFACE ---
        self.setup_ui(nx, ny, nz)
        
        # --- 3. INICIALIZAÇÃO PYVISTA (3D) ---
        _, self.state = run(
            mode=mode,
            z_exag=z_exag,
            show_scalar_bar=show_scalar_bar,
            external_plotter=self.plotter,
            external_state=self.state,
        )
        
        self.state["on_slice_changed"] = self.on_plotter_slice_changed
        
        # --- 4. CONFIGURAÇÃO FINAL ---
        self.update_2d_map()
        self.populate_facies_legend()
        self.fill_unified_facies_table()
        
        # Calcula métricas iniciais para o modelo base
        self.change_reservoir_facies(initial_reservoir)

        # Seleciona o primeiro item da árvore (Base) para inicializar a UI lateral
        top_item = self.project_tree.topLevelItem(0)
        if top_item: 
            top_item.setExpanded(True)
            self.project_tree.setCurrentItem(top_item)

    def setup_ui(self, nx, ny, nz):
        self.resize(1600, 900)
        
        # Menu Bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Arquivo")
        action_load = QtWidgets.QAction("Carregar Modelo Adicional...", self)
        action_load.triggered.connect(self.open_compare_dialog)
        file_menu.addAction(action_load)
        file_menu.addSeparator()
        action_exit = QtWidgets.QAction("Sair", self)
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)
        
        self.act_persp_viz = QtWidgets.QAction("Visualização", self); self.act_persp_viz.setCheckable(True); self.act_persp_viz.setChecked(True)
        self.act_persp_viz.triggered.connect(lambda: self.switch_perspective("visualization"))
        menubar.addAction(self.act_persp_viz)
        
        self.act_persp_comp = QtWidgets.QAction("Comparação", self); self.act_persp_comp.setCheckable(True)
        self.act_persp_comp.triggered.connect(lambda: self.switch_perspective("comparison"))
        menubar.addAction(self.act_persp_comp)

        # Toolbar
        self.setup_toolbar_controls()

        # Docks
        self.setup_docks(nx, ny, nz)
        
        # Central
        self.central_stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.central_stack)
        
        # Vis Container
        self.viz_container = QtWidgets.QStackedWidget() 
        self.tabs = self.viz_container 
        
        # 0: 3D
        self.plotter = BackgroundPlotter(show=False)
        self.viz_tab = QtWidgets.QWidget(); vl = QtWidgets.QVBoxLayout(self.viz_tab); vl.setContentsMargins(0,0,0,0)
        vl.addWidget(self.plotter.interactor); self.viz_container.addWidget(self.viz_tab)
        
        # 1: 2D
        self.plotter_2d = BackgroundPlotter(show=False)
        self.map2d_tab = QtWidgets.QWidget(); ml = QtWidgets.QVBoxLayout(self.map2d_tab); ml.setContentsMargins(0,0,0,0)
        ml.addWidget(self.plotter_2d.interactor); self.viz_container.addWidget(self.map2d_tab)
        
        # 2: Detalhes Central (CORRIGIDO NOME DA VARIÁVEL)
        self.details_tab = QtWidgets.QWidget(); l_det = QtWidgets.QVBoxLayout(self.details_tab)
        
        # Parte superior: Texto Global Central
        self.central_metrics_text = QtWidgets.QTextEdit() # <--- NOME ÚNICO
        self.central_metrics_text.setReadOnly(True)
        self.central_metrics_text.setMaximumHeight(150)
        l_det.addWidget(QtWidgets.QLabel("Resumo Global"))
        l_det.addWidget(self.central_metrics_text)
        
        # Parte inferior: Tabela
        self.facies_table = QtWidgets.QTableWidget()
        l_det.addWidget(QtWidgets.QLabel("Detalhamento por Fácies"))
        l_det.addWidget(self.facies_table)
        
        self.viz_container.addWidget(self.details_tab)
        self.central_stack.addWidget(self.viz_container)
        
        # Comparação
        self.compare_3d_container = QtWidgets.QWidget()
        self.setup_comparison_3d_view(self.compare_3d_container)
        self.central_stack.addWidget(self.compare_3d_container)

        self.resizeDocks([self.dock_explorer, self.dock_props], [280, 280], QtCore.Qt.Horizontal)
        self.resizeDocks([self.dock_explorer, self.dock_props], [400, 400], QtCore.Qt.Vertical)

    def switch_perspective(self, mode):
        if mode == "visualization":
            # 1. Configura Menu
            self.act_persp_viz.setChecked(True)
            self.act_persp_comp.setChecked(False)
            
            # 2. Dock Esquerdo -> Mostra Árvore
            self.dock_left_container.setCurrentIndex(0)
            self.dock_explorer.setWindowTitle("Project Explorer")
            self.dock_props.setVisible(True) # Mostra propriedades
            
            # 3. Central -> Mostra Visualizador Padrão
            self.central_stack.setCurrentIndex(0)
            
        elif mode == "comparison":
            # 1. Configura Menu
            self.act_persp_viz.setChecked(False)
            self.act_persp_comp.setChecked(True)
            
            # 2. Dock Esquerdo -> Mostra Painel Comparação
            self.dock_left_container.setCurrentIndex(1)
            self.dock_explorer.setWindowTitle("Painel de Comparação")
            self.dock_props.setVisible(False) # Esconde propriedades (não usadas aqui)
            
            # 3. Central -> Mostra 3D Comparado
            self.central_stack.setCurrentIndex(1)
            
            # Força atualização
            self.update_comparison_tables()
            self.update_compare_2d_maps()

    def setup_comparison_3d_view(self, container):
        """Configura a área central: 3D em cima, 2D embaixo (opcional), divididos por Splitters."""
        # Layout principal do container
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Splitter Vertical Principal (Separa 3D do 2D)
        self.main_split_compare = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        
        # --- PARTE SUPERIOR (3D) ---
        self.split_3d = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        if not hasattr(self, 'comp_plotter_base'): self.comp_plotter_base = BackgroundPlotter(show=False)
        if not hasattr(self, 'comp_plotter_comp'): self.comp_plotter_comp = BackgroundPlotter(show=False)
        
        self.split_3d.addWidget(self.comp_plotter_base.interactor)
        self.split_3d.addWidget(self.comp_plotter_comp.interactor)
        self.main_split_compare.addWidget(self.split_3d)
        
        # --- PARTE INFERIOR (2D) ---
        # Container para os mapas 2D (inicialmente oculto)
        self.maps_2d_container = QtWidgets.QWidget()
        l_2d = QtWidgets.QHBoxLayout(self.maps_2d_container)
        l_2d.setContentsMargins(0, 0, 0, 0)
        
        self.split_2d = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        if not hasattr(self, 'comp_plotter_base_2d'): self.comp_plotter_base_2d = BackgroundPlotter(show=False)
        if not hasattr(self, 'comp_plotter_comp_2d'): self.comp_plotter_comp_2d = BackgroundPlotter(show=False)
        
        self.split_2d.addWidget(self.comp_plotter_base_2d.interactor)
        self.split_2d.addWidget(self.comp_plotter_comp_2d.interactor)
        
        l_2d.addWidget(self.split_2d)
        self.main_split_compare.addWidget(self.maps_2d_container)
        
        # Oculta o 2D por padrão
        self.maps_2d_container.setVisible(False)
        
        layout.addWidget(self.main_split_compare)

    def setup_toolbar_controls(self):
        toolbar = self.addToolBar("Controles")
        toolbar.setMovable(False)
        toolbar.clear() # <--- CORREÇÃO: Limpa tudo antes de adicionar para não duplicar
        
        toolbar.setIconSize(QtCore.QSize(16, 16))
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        
        # --- 1. MODO VISUALIZAÇÃO (ÚNICO PARA TUDO) ---
        self.btn_mode = QtWidgets.QToolButton(self)
        self.btn_mode.setText("Modo: Fácies") 
        self.btn_mode.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogListView))
        self.btn_mode.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.btn_mode.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.btn_mode.setAutoRaise(True)
        
        menu_mode = QtWidgets.QMenu(self.btn_mode)
        modes = [
            ("Fácies", "facies"), 
            ("Reservatório", "reservoir"), 
            ("Clusters", "clusters"), 
            ("Maior Cluster", "largest"), 
            ("Espessura Local", "thickness_local")
        ]
        
        for text, data in modes:
            action = menu_mode.addAction(text)
            # Usa lambda para capturar o valor correto
            action.triggered.connect(lambda ch, t=text, d=data: self._update_mode_btn(t, d))
            
        self.btn_mode.setMenu(menu_mode)
        toolbar.addWidget(self.btn_mode)
        
        toolbar.addSeparator()
        
        # --- 2. ESPESSURA (ÚNICO PARA TUDO) ---
        self.btn_thick = QtWidgets.QToolButton(self)
        self.btn_thick.setText("Espessura: Espessura")
        self.btn_thick.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView))
        self.btn_thick.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.btn_thick.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.btn_thick.setAutoRaise(True)
        
        menu_thick = QtWidgets.QMenu(self.btn_thick)
        thickness_opts = ["Espessura", "NTG coluna", "NTG envelope", "Maior pacote", "Nº pacotes", "ICV", "Qv", "Qv absoluto"]
        
        for label in thickness_opts:
            action = menu_thick.addAction(label)
            action.triggered.connect(lambda ch, l=label: self._update_thick_btn(l))
            
        self.btn_thick.setMenu(menu_thick)
        toolbar.addWidget(self.btn_thick)
        
        toolbar.addSeparator()
        
        # --- 3. MAPAS 2D (Apenas visível na Comparação) ---
        self.act_toggle_2d = QtWidgets.QAction("Mapas 2D", self)
        self.act_toggle_2d.setCheckable(True)
        self.act_toggle_2d.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogContentsView))
        self.act_toggle_2d.triggered.connect(self.toggle_compare_2d_view)
        self.act_toggle_2d.setVisible(False) 
        toolbar.addAction(self.act_toggle_2d)
        
        toolbar.addSeparator()
        
        # --- 4. SNAPSHOT ---
        btn_ss = QtWidgets.QAction("Snapshot", self)
        btn_ss.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        btn_ss.triggered.connect(self.take_snapshot)
        toolbar.addAction(btn_ss)

    

    # Helpers para atualizar o texto do botão (Mantidos)
    def _update_mode_btn(self, text, data):
        self.btn_mode.setText(f"Modo: {text}")
        self.change_mode(data)

    def _update_thick_btn(self, label):
        self.btn_thick.setText(f"Espessura: {label}")
        self.change_thickness_mode(label)

    # Helpers para atualizar o texto do botão quando clica no menu
    def _update_mode_btn(self, text, data):
        self.btn_mode.setText(f"Modo: {text}")
        self.change_mode(data)

    def _update_thick_btn(self, label):
        self.btn_thick.setText(f"Espessura: {label}")
        self.change_thickness_mode(label)

    def setup_docks(self, nx, ny, nz):
        # --- 1. DOCK EXPLORER (Topo Esquerda) ---
        self.dock_explorer = QtWidgets.QDockWidget("Project Explorer", self)
        self.dock_explorer.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        
        self.dock_left_container = QtWidgets.QStackedWidget()
        # O segredo: Dizer ao container que ele pode ser esmagado verticalmente (Ignored)
        self.dock_left_container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Ignored)
        
        # Widget 0: Árvore
        self.project_tree = QtWidgets.QTreeWidget()
        self.project_tree.setHeaderLabel("Hierarquia do Projeto")
        self.project_tree.itemDoubleClicked.connect(self.on_tree_double_clicked)
        self.project_tree.itemSelectionChanged.connect(self.on_tree_selection_changed)
        
        # DESTRAVA TOTAL: Altura mínima zero e política Ignored (encolha o quanto quiser)
        self.project_tree.setMinimumHeight(0)
        self.project_tree.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Ignored)
        
        self.dock_left_container.addWidget(self.project_tree)
        
        # Widget 1: Painel Comparação
        self.compare_panel = self.setup_comparison_dock_content()
        self.dock_left_container.addWidget(self.compare_panel)
        
        self.dock_explorer.setWidget(self.dock_left_container)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dock_explorer)
        
        self.add_model_to_tree("base", "Modelo Base")

        # --- 2. DOCK PROPRIEDADES (Baixo Esquerda) ---
        self.dock_props = QtWidgets.QDockWidget("Propriedades & Filtros", self)
        self.dock_props.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        
        self.props_stack = QtWidgets.QStackedWidget()
        # Política Expanding aqui para "empurrar" para cima contra o Explorer
        self.props_stack.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.props_stack.addWidget(QtWidgets.QLabel("Selecione um item."))
        
        # Pág 1: Grid Slicing
        self.page_grid = QtWidgets.QWidget()
        pg_layout = QtWidgets.QVBoxLayout(self.page_grid)
        self.slicer_widget = GridSlicerWidget(nx, ny, nz, self.on_ui_slice_changed)
        pg_layout.addWidget(self.slicer_widget)
        pg_layout.addStretch()
        self.props_stack.addWidget(self.page_grid)
        
        # --- PÁG PROPRIEDADES (COM SPLITTER) ---
        self.page_props = QtWidgets.QWidget()
        layout_props = QtWidgets.QVBoxLayout(self.page_props)
        layout_props.setContentsMargins(0, 0, 0, 0)
        
        self.props_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        
        # Parte de Cima: Legenda
        self.legend_group = QtWidgets.QGroupBox("Legenda & Filtro")
        lgl = QtWidgets.QVBoxLayout(self.legend_group)
        lgl.setContentsMargins(2, 5, 2, 2)
        
        self.facies_legend_table = QtWidgets.QTableWidget()
        self.facies_legend_table.setColumnCount(4)
        self.facies_legend_table.setHorizontalHeaderLabels(["Cor", "ID", "N", "Res"])
        self.facies_legend_table.verticalHeader().setVisible(False)
        self.facies_legend_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.facies_legend_table.itemChanged.connect(self.on_legend_item_changed)
        lgl.addWidget(self.facies_legend_table)
        
        self.props_splitter.addWidget(self.legend_group)
        
        # Parte de Baixo: Métricas Laterais
        self.metrics_group = QtWidgets.QGroupBox("Métricas (Resumo)")
        mgl = QtWidgets.QVBoxLayout(self.metrics_group)
        mgl.setContentsMargins(2, 5, 2, 2)
        
        self.sidebar_metrics_text = QtWidgets.QTextEdit()
        self.sidebar_metrics_text.setReadOnly(True)
        # O texto precisa querer crescer para ocupar o espaço liberado
        self.sidebar_metrics_text.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        mgl.addWidget(self.sidebar_metrics_text)
        
        self.props_splitter.addWidget(self.metrics_group)
        
        layout_props.addWidget(self.props_splitter)
        
        self.props_stack.addWidget(self.page_props)
        self.dock_props.setWidget(self.props_stack)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dock_props)

    def add_model_to_tree(self, model_key, model_name):
        """Adiciona a estrutura de pastas do modelo na árvore."""
        root_item = QtWidgets.QTreeWidgetItem(self.project_tree, [model_name])
        root_item.setData(0, QtCore.Qt.UserRole, "model_root")
        root_item.setData(0, QtCore.Qt.UserRole + 1, model_key)
        root_item.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_DirHomeIcon))
        root_item.setExpanded(True)
        
        # 1. Geometria
        item_grid = QtWidgets.QTreeWidgetItem(root_item, ["Geometria (Grid)"])
        item_grid.setData(0, QtCore.Qt.UserRole, "grid_settings")
        item_grid.setData(0, QtCore.Qt.UserRole + 1, model_key)
        item_grid.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView))
        
        # 2. Propriedades
        item_props = QtWidgets.QTreeWidgetItem(root_item, ["Propriedades & Filtros"])
        item_props.setData(0, QtCore.Qt.UserRole, "prop_settings")
        item_props.setData(0, QtCore.Qt.UserRole + 1, model_key)
        item_props.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogListView))

        # 3. Métricas (NOVO ITEM)
        item_metrics = QtWidgets.QTreeWidgetItem(root_item, ["Métricas & Estatísticas"])
        item_metrics.setData(0, QtCore.Qt.UserRole, "metrics_view")
        item_metrics.setData(0, QtCore.Qt.UserRole + 1, model_key)
        item_metrics.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogInfoView))

        # 4. Mapas 2D (NOVO ITEM)
        item_2d = QtWidgets.QTreeWidgetItem(root_item, ["Mapas 2D"])
        item_2d.setData(0, QtCore.Qt.UserRole, "map2d_view")
        item_2d.setData(0, QtCore.Qt.UserRole + 1, model_key)
        item_2d.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogContentsView))

    # --- LÓGICA DE INTERAÇÃO TREE ---

    def on_tree_double_clicked(self, item, col):
        """Duplo clique em Geometria força a troca para aba 3D."""
        role = item.data(0, QtCore.Qt.UserRole)
        model_key = item.data(0, QtCore.Qt.UserRole + 1)
        
        if role == "grid_settings" and model_key:
            self.switch_main_view_to_model(model_key)
            self.tabs.setCurrentIndex(0) # Força ir para 3D

    def switch_main_view_to_model(self, model_key):
        """Carrega grid, restaura filtros e modo de visualização específicos do modelo."""
        target_facies = self.models[model_key]["facies"]
        if target_facies is None: return

        from load_data import grid as global_grid
        from scipy.ndimage import label, generate_binary_structure
        
        # 1. Recupera Estado Específico do Modelo
        # Se não tiver modo salvo, usa 'facies' como padrão
        saved_mode = self.models[model_key].get("view_mode", "facies")
        current_res_set = self.models[model_key]["reservoir_facies"]
        
        # Atualiza o estado global com os dados deste modelo
        self.state["current_facies"] = target_facies
        self.state["reservoir_facies"] = current_res_set
        self.state["mode"] = saved_mode 

        # 2. Atualiza o Botão da Barra Superior para refletir o modo do modelo
        if hasattr(self, "btn_mode"):
            # Mapeia código para texto bonito
            labels = {"facies": "Fácies", "reservoir": "Reservatório", "clusters": "Clusters", 
                      "largest": "Maior Cluster", "thickness_local": "Espessura Local"}
            self.btn_mode.setText(f"Modo: {labels.get(saved_mode, saved_mode)}")

        # 3. Prepara o Grid Físico
        if model_key == "compare":
            active_grid = global_grid.copy(deep=True)
            active_grid.cell_data["Facies"] = target_facies
        else:
            active_grid = global_grid
            active_grid.cell_data["Facies"] = facies # Garante original
            
        # 4. CÁLCULO CRÍTICO: Recalcula a Máscara 'Reservoir' e Clusters
        # Isso garante que o que você vê no 3D bate com os checkboxes laterais
        is_res = np.isin(target_facies, list(current_res_set)).astype(np.uint8)
        active_grid.cell_data["Reservoir"] = is_res
        
        # Recalcula Clusters (necessário para modo Clusters e Maior Cluster)
        arr_3d = is_res.reshape((nx, ny, nz), order="F")
        structure = generate_binary_structure(3, 1)
        labeled, _ = label(arr_3d.transpose(2, 1, 0), structure=structure)
        clusters_1d = labeled.transpose(2, 1, 0).reshape(-1, order="F").astype(np.int32)
        active_grid.cell_data["Clusters"] = clusters_1d
        
        # Recalcula Maior Cluster
        counts = np.bincount(clusters_1d.ravel())
        if counts.size > 0: counts[0] = 0
        largest_lbl = counts.argmax() if counts.size > 0 else 0
        active_grid.cell_data["LargestCluster"] = (clusters_1d == largest_lbl).astype(np.uint8)

        # 5. Atualiza Metadados de Cores (LUT) para Clusters
        lut, rng = make_clusters_lut(clusters_1d)
        self.state["clusters_lut"] = lut
        self.state["clusters_rng"] = rng
        self.state["clusters_sizes"] = compute_cluster_sizes(clusters_1d)

        # 6. Finaliza a Troca
        self.state["current_grid_source"] = active_grid
        self.state["refresh"]() # Redesenha o 3D
        
        # Atualiza painéis laterais
        if saved_mode == "clusters":
            self.populate_clusters_legend()
        else:
            self.populate_facies_legend()
            
        self.update_sidebar_metrics_text(model_key)
        self.update_2d_map()

    def on_tree_selection_changed(self):
        items = self.project_tree.selectedItems()
        if not items:
            self.props_stack.setCurrentIndex(0)
            return
            
        item = items[0]
        role = item.data(0, QtCore.Qt.UserRole)
        model_key = item.data(0, QtCore.Qt.UserRole + 1)
        
        # 1. Configura Dock
        if role == "grid_settings":
            self.props_stack.setCurrentWidget(self.page_grid)
        else:
            self.props_stack.setCurrentWidget(self.page_props)
            
        # 2. Ações de Troca
        if model_key:
            # Sempre atualiza o texto lateral com as métricas do modelo clicado
            self.update_sidebar_metrics_text(model_key)
            
            if role == "metrics_view":
                self.viz_container.setCurrentIndex(2)
                self.update_metrics_view_content(model_key)
                
            elif role == "map2d_view":
                self.switch_main_view_to_model(model_key)
                self.viz_container.setCurrentIndex(1)
                
            elif role in ["grid_settings", "prop_settings", "model_root"]:
                # Aqui a mágica acontece: troca o grid e atualiza as cores
                self.switch_main_view_to_model(model_key)
                self.viz_container.setCurrentIndex(0)
    
    def update_sidebar_metrics_text(self, model_key):
        """Preenche a caixa de texto lateral com o resumo do modelo."""
        # Tenta pegar o widget correto. Se setup_docks não rodou ou tem erro, aborta.
        target = getattr(self, "sidebar_metrics_text", None)
        if not target: return

        data = self.cached_metrics.get(model_key)
        if not data or not data["metrics"]:
            target.setPlainText("Sem dados calculados.")
            return
            
        m = data["metrics"]
        p = data["perc"]
        
        lines = [
            f"=== {self.models[model_key]['name']} ===",
            f"NTG: {m['ntg']:.3f}",
            f"Células Res.: {m['res_cells']}",
            f"Conectividade: {m['connected_fraction']:.3f}",
            f"Clusters: {m['n_clusters']}",
            f"Vol. Grid: {m.get('grid_volume',0):.2e}",
            f"Vol. Res.: {m.get('reservoir_volume',0):.2e}"
        ]
        
        if p:
            def f(a): return str(list(a)) if a else "[]"
            lines.append("")
            lines.append("Percolação:")
            lines.append(f"X: {'Sim' if p['x_perc'] else 'N'} {f(p['x_clusters'])}")
            lines.append(f"Y: {'Sim' if p['y_perc'] else 'N'} {f(p['y_clusters'])}")
            lines.append(f"Z: {'Sim' if p['z_perc'] else 'N'} {f(p['z_clusters'])}")
            
        target.setPlainText("\n".join(lines))

    def update_metrics_view_content(self, model_key):
        data = self.cached_metrics.get(model_key)
        if not data: return
        
        # Atualiza Texto Central
        target_central = getattr(self, "central_metrics_text", None)
        metrics = data["metrics"]
        perc = data["perc"]
        
        if target_central:
            if metrics:
                # Mesmo texto da lateral, ou mais detalhado se preferir
                lines = [
                    f"=== {self.models[model_key]['name']} ===",
                    f"NTG Global: {metrics['ntg']:.3f}",
                    f"Células Reservatório: {metrics['res_cells']}",
                    f"Fração Conectada: {metrics['connected_fraction']:.3f}",
                    f"Número de Clusters: {metrics['n_clusters']}",
                    f"Maior Cluster: {metrics['largest_size']}",
                    f"Volume Grid: {metrics.get('grid_volume',0):.2e} m3",
                    f"Volume Reservatório: {metrics.get('reservoir_volume',0):.2e} m3",
                    "",
                    "--- Análise de Percolação ---"
                ]
                if perc:
                    def f(a): return str(list(a)) if a else "[]"
                    lines.append(f"X (In-Line): {'Conectado' if perc['x_perc'] else 'Não Conectado'} | Clusters: {f(perc['x_clusters'])}")
                    lines.append(f"Y (X-Line): {'Conectado' if perc['y_perc'] else 'Não Conectado'} | Clusters: {f(perc['y_clusters'])}")
                    lines.append(f"Z (Vertical): {'Conectado' if perc['z_perc'] else 'Não Conectado'} | Clusters: {f(perc['z_clusters'])}")
                
                target_central.setPlainText("\n".join(lines))
            else:
                target_central.setPlainText("Métricas não calculadas.")

        # Atualiza Tabela
        df = data.get("df")
        if df is not None:
             self.set_facies_metrics(df)
        else:
             self.facies_table.setRowCount(0)

    # --- LÓGICA DE CÁLCULO E DADOS ---

    def set_metrics(self, metrics, perc):
        """Salva métricas globais no cache do modelo Base."""
        self.cached_metrics["base"]["metrics"] = metrics
        self.cached_metrics["base"]["perc"] = perc
        
        # Se estivermos vendo as métricas do base, atualiza a tela agora
        if self.tabs.currentIndex() == 2:
            sel = self.project_tree.selectedItems()
            if sel and sel[0].data(0, QtCore.Qt.UserRole + 1) == "base":
                self.update_metrics_view_content("base")

    def set_facies_metrics(self, df):
        """Salva o DataFrame no cache Base e preenche tabela se necessário."""
        # Se chamado externamente (pelo main.py), assume-se que é do modelo BASE
        self.cached_metrics["base"]["df"] = df
        
        # Preenche a tabela visualmente
        pretty = {
            "facies": "Fácies", "cells": "Células", "fraction": "Fração",
            "n_clusters": "Nº Clusters", "largest_label": "Maior Cluster ID",
            "largest_size": "Tam. Maior Cluster", "connected_fraction": "Fração Conect.",
            "volume_total": "Vol Total", "volume_largest_cluster": "Vol Maior Cluster",
            "thickness_largest_cluster": "Espessura Maior",
            "Perc_X": "Perc X", "Perc_Y": "Perc Y", "Perc_Z": "Perc Z"
        }
        self.facies_table.setRowCount(len(df))
        self.facies_table.setColumnCount(len(df.columns))
        self.facies_table.setHorizontalHeaderLabels([pretty.get(c,c) for c in df.columns])
        
        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                val = df.iloc[i][col]
                # Formatação
                if isinstance(val, (float, np.floating)):
                    if col in ["fraction", "connected_fraction", "Perc_X", "Perc_Y", "Perc_Z"]: txt = f"{val:.3f}"
                    elif "volume" in col: txt = f"{val:.2e}"
                    else: txt = f"{val:.2f}"
                else: txt = str(val)
                self.facies_table.setItem(i, j, QtWidgets.QTableWidgetItem(txt))
        self.facies_table.resizeColumnsToContents()

    def change_reservoir_facies(self, reservoir_set):
        if not isinstance(reservoir_set, set): return
        
        # 1. Identifica modelo ativo
        current_model_key = "base"
        sel = self.project_tree.selectedItems()
        if sel:
            key = sel[0].data(0, QtCore.Qt.UserRole + 1)
            if key in ["base", "compare"]:
                current_model_key = key

        # 2. Atualiza Dados
        self.models[current_model_key]["reservoir_facies"] = reservoir_set
        self.state["reservoir_facies"] = reservoir_set
        
        # 3. Atualiza Visualização 3D (Recálculo pesado)
        self.switch_main_view_to_model(current_model_key)
        
        # 4. Recalcula Métricas (Cache)
        target_facies = self.models[current_model_key]["facies"]
        m, p = compute_global_metrics_for_array(target_facies, reservoir_set)
        self.cached_metrics[current_model_key]["metrics"] = m
        self.cached_metrics[current_model_key]["perc"] = p
        
        # 5. ATUALIZAÇÃO DA INTERFACE (CORRIGIDO)
        
        # Atualiza a Lateral (Métricas Resumo)
        self.update_sidebar_metrics_text(current_model_key)
        
        # Atualiza a Central (Resumo Global + Tabela)
        # Removemos o "if tab == 2" para garantir que o texto esteja sempre atualizado
        self.update_metrics_view_content(current_model_key)
        
        # 6. Sincroniza Comparação (se houver)
        if self.compare_facies is not None:
            self.update_comparison_tables()
            if hasattr(self, 'update_compare_3d_mode_single'):
                self.update_compare_3d_mode_single("base")
                self.update_compare_3d_mode_single("compare")

    def load_compare_model(self, grdecl_path):
        try: fac_compare = load_facies_from_grdecl(grdecl_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erro", str(e)); return
        
        if fac_compare.size != nx * ny * nz:
             QtWidgets.QMessageBox.warning(self, "Erro", "Grid incompatível"); return
        
        # 1. Configura Dados
        self.models["compare"]["name"] = os.path.basename(grdecl_path)
        self.models["compare"]["facies"] = fac_compare
        # Herda o filtro do base inicialmente
        rf = self.models["base"]["reservoir_facies"]
        self.models["compare"]["reservoir_facies"] = set(rf)
        self.compare_facies = fac_compare
        
        # 2. Calcula Métricas Globais (para o texto lateral e tabela comparativa)
        self.compare_facies_stats, _ = facies_distribution_array(fac_compare)
        cm, cp = compute_global_metrics_for_array(fac_compare, rf)
        self.comp_res_stats, _ = reservoir_facies_distribution_array(fac_compare, rf)
        
        # 3. Calcula Tabela Detalhada (NOVO - Resolve o problema da tabela vazia)
        df_detail = self.generate_detailed_metrics_df(fac_compare)
        
        # Salva tudo no cache
        self.cached_metrics["compare"] = {"metrics": cm, "perc": cp, "df": df_detail}
        
        # 4. Atualiza Interface
        self.add_model_to_tree("compare", f"Comparado: {self.models['compare']['name']}")
        self.fill_unified_facies_table()
        
        self.update_comparison_tables()
        self.init_compare_3d()
        self.update_compare_2d_maps()

    # --- FUNÇÕES VISUAIS (MAPS, 3D, ETC) ---

    def update_2d_map(self):
        """Atualiza o plotter 2D principal usando o Grid Ativo (com filtros aplicados)."""
        if not hasattr(self, "plotter_2d"): return
        
        # 1. Descobre qual o modo de espessura escolhido
        presets = self.state.get("thickness_presets") or {}
        mode = self.state.get("thickness_mode", "Espessura")
        if mode not in presets: 
            if "Espessura" in presets: mode = "Espessura"
            else: return
        scalar_name, title = presets[mode]
        
        active_grid = self.state.get("current_grid_source")
        if active_grid is None:
            from load_data import grid as active_grid
            
        try:
            self._draw_2d_map_local(self.plotter_2d, active_grid, scalar_name, title)
        except Exception as e:
            print(f"Erro ao atualizar mapa 2D: {e}")

    def update_compare_2d_maps(self):
        if not hasattr(self, "state"): return
        presets = self.state.get("thickness_presets", {})
        mode = self.state.get("thickness_mode", "Espessura")
        if mode not in presets: mode = "Espessura"
        if mode not in presets: return
        s, t = presets[mode]
        
        if self.compare_states.get("base"):
            gb = self.compare_states["base"].get("current_grid_source")
            if not gb: from load_data import grid as gb
            if gb: self._draw_2d_map_local(self.comp_plotter_base_2d, gb, s, t)
            
        if self.compare_states.get("compare"):
            gc = self.compare_states["compare"].get("current_grid_source")
            if gc: self._draw_2d_map_local(self.comp_plotter_comp_2d, gc, s, t)

    def _draw_2d_map_local(self, plotter, grid_source, scalar_name_3d, title):
        from load_data import nx, ny, nz
        import pyvista as pv
        import numpy as np
        
        if scalar_name_3d not in grid_source.cell_data:
            plotter.clear(); return
            
        arr3d = grid_source.cell_data[scalar_name_3d].reshape((nx, ny, nz), order="F")
        thickness_2d = np.full((nx, ny), np.nan, dtype=float)
        for ix in range(nx):
            for iy in range(ny):
                col = arr3d[ix, iy, :]; col = col[col > 0]
                if col.size > 0: thickness_2d[ix, iy] = col.max()
                
        x_min, x_max, y_min, y_max, _, z_max = grid_source.bounds
        xs = np.linspace(x_min, x_max, nx); ys = np.linspace(y_min, y_max, ny)
        xs, ys = np.meshgrid(xs, ys, indexing="ij"); zs = np.full_like(xs, z_max)
        surf = pv.StructuredGrid(xs, ys, zs)
        name2d = scalar_name_3d + "_2d"
        surf.cell_data[name2d] = thickness_2d[:nx-1, :ny-1].ravel(order="F")
        arr = surf.cell_data[name2d]; arr = np.where(arr < 0, np.nan, arr)
        surf.cell_data[name2d] = arr
        clim = get_2d_clim(scalar_name_3d, arr)
        
        plotter.clear()
        plotter.add_mesh(surf, scalars=name2d, cmap="plasma", show_edges=True, edge_color="black", line_width=0.5, nan_color="white", show_scalar_bar=False, clim=clim)
        plotter.view_xy(); plotter.enable_parallel_projection(); plotter.set_background("white")
        plotter.add_scalar_bar(title=title); plotter.reset_camera()

    # --- ABA COMPARAÇÃO ---

    def setup_comparison_tab(self):
        self.compare_tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.compare_tab)
        self.compare_tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.compare_tabs)
        
        # 1. Métricas
        w_met = QtWidgets.QWidget(); l_met = QtWidgets.QVBoxLayout(w_met)
        hl = QtWidgets.QHBoxLayout()
        self.base_model_label = QtWidgets.QLabel("Modelo Base"); self.comp_model_label = QtWidgets.QLabel("Modelo Comparado")
        hl.addWidget(self.base_model_label); hl.addWidget(self.comp_model_label); l_met.addLayout(hl)
        
        self.global_compare_table = QtWidgets.QTableWidget()
        self.global_compare_table.setColumnCount(4)
        self.global_compare_table.setHorizontalHeaderLabels(["Métrica", "Base", "Comp", "Dif"])
        l_met.addWidget(self.global_compare_table)
        
        self.facies_compare_table = QtWidgets.QTableWidget()
        self.facies_compare_table.setColumnCount(9)
        self.facies_compare_table.setHorizontalHeaderLabels(["Fácies", "Cel Base", "%", "Cel Comp", "%", "Vol Base", "Vol Comp", "Esp Base", "Esp Comp"])
        l_met.addWidget(self.facies_compare_table)
        
        self.reservoir_facies_compare_table = QtWidgets.QTableWidget()
        self.reservoir_facies_compare_table.setColumnCount(5)
        self.reservoir_facies_compare_table.setHorizontalHeaderLabels(["Fácies (Res)", "Cel Base", "%", "Cel Comp", "%"])
        l_met.addWidget(self.reservoir_facies_compare_table)
        
        self.compare_tabs.addTab(w_met, "Métricas Comparadas")
        
        # 2. Mapas 2D
        w_2d = QtWidgets.QWidget(); l_2d = QtWidgets.QHBoxLayout(w_2d)
        self.comp_plotter_base_2d = BackgroundPlotter(show=False); self.comp_plotter_comp_2d = BackgroundPlotter(show=False)
        l_2d.addWidget(self.comp_plotter_base_2d.interactor); l_2d.addWidget(self.comp_plotter_comp_2d.interactor)
        self.compare_tabs.addTab(w_2d, "Mapas 2D Comparados")
        
        # 3. 3D
        w_3d = QtWidgets.QWidget(); l_3d = QtWidgets.QVBoxLayout(w_3d)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        
        wt = QtWidgets.QWidget(); lt = QtWidgets.QHBoxLayout(wt)
        self.comp_plotter_base = BackgroundPlotter(show=False); self.comp_plotter_comp = BackgroundPlotter(show=False)
        lt.addWidget(self.comp_plotter_base.interactor); lt.addWidget(self.comp_plotter_comp.interactor); split.addWidget(wt)
        
        wb = QtWidgets.QWidget(); lb = QtWidgets.QHBoxLayout(wb)
        self.res_table_base_cmp = make_facies_table(); self.res_table_base_cmp.itemChanged.connect(self.update_base_reservoir_compare)
        self.res_table_comp_cmp = make_facies_table(); self.res_table_comp_cmp.itemChanged.connect(self.update_compare_reservoir_compare)
        self.clus_table_base_cmp = self._create_legend_table(["Cor","Cluster","Cel"]); self.clus_table_comp_cmp = self._create_legend_table(["Cor","Cluster","Cel"])
        lb.addWidget(self.res_table_base_cmp); lb.addWidget(self.clus_table_base_cmp)
        lb.addWidget(self.res_table_comp_cmp); lb.addWidget(self.clus_table_comp_cmp)
        split.addWidget(wb); l_3d.addWidget(split)
        
        self.compare_tabs.addTab(w_3d, "3D Comparado")

    def update_comparison_tables(self):
        # 1. Recupera as métricas do cache
        m0 = self.cached_metrics["base"]["metrics"]
        m1 = self.cached_metrics["compare"]["metrics"]
        
        # --- TABELA 1: GLOBAL ---
        rows = []
        def get(m, k): return m.get(k) if m else None
        
        rows.append(("NTG", get(m0, "ntg"), get(m1, "ntg")))
        rows.append(("Total Cel", get(m0, "total_cells"), get(m1, "total_cells")))
        rows.append(("Res Cel", get(m0, "res_cells"), get(m1, "res_cells")))
        rows.append(("Conectividade", get(m0, "connected_fraction"), get(m1, "connected_fraction")))
        rows.append(("Clusters", get(m0, "n_clusters"), get(m1, "n_clusters")))
        rows.append(("Maior Cluster", get(m0, "largest_size"), get(m1, "largest_size")))

        self.global_compare_table.setRowCount(len(rows))
        for i, (l, a, b) in enumerate(rows):
            self.global_compare_table.setItem(i, 0, QtWidgets.QTableWidgetItem(l))
            
            val_a = f"{a:.3f}" if isinstance(a, float) else str(a) if a is not None else "-"
            val_b = f"{b:.3f}" if isinstance(b, float) else str(b) if b is not None else "-"
            
            self.global_compare_table.setItem(i, 1, QtWidgets.QTableWidgetItem(val_a))
            self.global_compare_table.setItem(i, 2, QtWidgets.QTableWidgetItem(val_b))
            
            # Coluna Diferença
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                diff = b - a
                # Formatação condicional simples para diferença
                item_diff = QtWidgets.QTableWidgetItem(f"{diff:.3f}")
                if diff > 0: item_diff.setForeground(QColor("green"))
                elif diff < 0: item_diff.setForeground(QColor("red"))
                self.global_compare_table.setItem(i, 3, item_diff)
            else:
                self.global_compare_table.setItem(i, 3, QtWidgets.QTableWidgetItem("-"))
        self.global_compare_table.resizeColumnsToContents()

        # --- TABELA 2: DISTRIBUIÇÃO DE FÁCIES (GRID INTEIRO) ---
        # Usa self.base_facies_stats e self.compare_facies_stats calculados no init/load
        stats0 = getattr(self, "base_facies_stats", {})
        stats1 = getattr(self, "compare_facies_stats", {})
        
        if stats0:
            all_facies = sorted(set(stats0.keys()) | (set(stats1.keys()) if stats1 else set()))
            self.facies_compare_table.setRowCount(len(all_facies))
            
            for row, fac in enumerate(all_facies):
                s0 = stats0.get(fac, {"cells": 0, "fraction": 0.0, "volume": 0.0, "thickness_gross": 0.0})
                s1 = stats1.get(fac, {"cells": 0, "fraction": 0.0, "volume": 0.0, "thickness_gross": 0.0}) if stats1 else {"cells": 0, "fraction": 0.0, "volume": 0.0, "thickness_gross": 0.0}
                
                self.facies_compare_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(fac)))
                self.facies_compare_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(s0["cells"])))
                self.facies_compare_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{100*s0['fraction']:.1f}%"))
                self.facies_compare_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(s1["cells"])))
                self.facies_compare_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{100*s1['fraction']:.1f}%"))
                self.facies_compare_table.setItem(row, 5, QtWidgets.QTableWidgetItem(f"{s0.get('volume',0):.2e}"))
                self.facies_compare_table.setItem(row, 6, QtWidgets.QTableWidgetItem(f"{s1.get('volume',0):.2e}"))
                self.facies_compare_table.setItem(row, 7, QtWidgets.QTableWidgetItem(f"{s0.get('thickness_gross',0):.1f}"))
                self.facies_compare_table.setItem(row, 8, QtWidgets.QTableWidgetItem(f"{s1.get('thickness_gross',0):.1f}"))
            self.facies_compare_table.resizeColumnsToContents()

        # --- TABELA 3: RESERVATÓRIO ---
        # Usa reservoir_facies_distribution_array
        stats0r, _ = reservoir_facies_distribution_array(facies, self.models["base"]["reservoir_facies"])
        stats1r = getattr(self, "comp_res_stats", {})
        
        if stats0r:
            res_union = sorted(set(stats0r.keys()) | set(stats1r.keys() if stats1r else []))
            self.reservoir_facies_compare_table.setRowCount(len(res_union))
            for row, fac in enumerate(res_union):
                s0 = stats0r.get(fac, {"cells": 0, "fraction": 0.0})
                s1 = stats1r.get(fac, {"cells": 0, "fraction": 0.0}) if stats1r else {"cells": 0, "fraction": 0.0}
                
                self.reservoir_facies_compare_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(fac)))
                self.reservoir_facies_compare_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(s0["cells"])))
                self.reservoir_facies_compare_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{100*s0['fraction']:.1f}%"))
                self.reservoir_facies_compare_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(s1["cells"])))
                self.reservoir_facies_compare_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{100*s1['fraction']:.1f}%"))
            self.reservoir_facies_compare_table.resizeColumnsToContents()

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

    def open_compare_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Selecionar Modelo", "assets", "GRDECL (*.grdecl)")
        if path: self.load_compare_model(path)

    def _create_legend_table(self, h):
        t = QtWidgets.QTableWidget(); t.setColumnCount(len(h)); t.setHorizontalHeaderLabels(h)
        return t

    def init_compare_3d(self):
        if self.models["base"]["facies"] is None: return
        from visualize import run; from load_data import grid as gg
        
        self.comp_plotter_base.clear()
        self.compare_states["base"] = {}
        run(mode="facies", external_plotter=self.comp_plotter_base, external_state=self.compare_states["base"], 
            target_grid=gg, target_facies=self.models["base"]["facies"])
        
        self.comp_plotter_comp.clear()
        self.compare_states["compare"] = {}
        if self.models["compare"]["facies"] is not None:
             g2 = gg.copy(deep=True); g2.cell_data["Facies"] = self.models["compare"]["facies"]
             run(mode="facies", external_plotter=self.comp_plotter_comp, external_state=self.compare_states["compare"], 
                 target_grid=g2, target_facies=self.models["compare"]["facies"])
                 
        self.install_compare_sync_callbacks()
        self.sync_compare_cameras()

    def sync_compare_cameras(self):
        pb = self.comp_plotter_base
        pc = self.comp_plotter_comp
        
        # Flag para evitar loop infinito de atualização
        self._is_syncing = False

        def sync(src, dst):
            if self._is_syncing: return
            self._is_syncing = True
            try:
                # Copia propriedades da câmera
                dst.camera.position = src.camera.position
                dst.camera.focal_point = src.camera.focal_point
                dst.camera.view_angle = src.camera.view_angle
                dst.camera.up = src.camera.up
                dst.camera.clipping_range = src.camera.clipping_range
                dst.render()
            finally:
                self._is_syncing = False
            
        # CORREÇÃO: AddObserver deve ser com letras maiúsculas
        pb.camera.AddObserver("ModifiedEvent", lambda *args: sync(pb, pc))
        pc.camera.AddObserver("ModifiedEvent", lambda *args: sync(pc, pb))

    def install_compare_sync_callbacks(self):
        states = [self.compare_states.get("base"), self.compare_states.get("compare")]
        plotters = [self.comp_plotter_base, self.comp_plotter_comp]
        
        def on_slice(axis, mode, value):
            for st in states: 
                if st: 
                    st[f"{axis}_{mode}"] = int(value)
                    if "refresh" in st: st["refresh"]()
            for p in plotters: p.render()
            
        for st in states: 
            if st: st["on_slice_changed"] = on_slice

    def update_compare_3d_mode(self):
        mode = self.state.get("mode", "facies")
        for k in ["base", "compare"]:
            st = self.compare_states.get(k)
            if st:
                st["mode"] = mode
                if k == "base": rf = self.models["base"]["reservoir_facies"]
                else: rf = self.models["compare"]["reservoir_facies"]
                if "update_reservoir_fields" in st: st["update_reservoir_fields"](rf)
                if "refresh" in st: st["refresh"]()

    def populate_compare_clusters_tables(self):
        # Função auxiliar para preencher uma tabela de legenda de cluster
        def fill_table(table, state):
            sizes = state.get("clusters_sizes")
            lut = state.get("clusters_lut")
            if not sizes or not lut:
                table.setRowCount(0)
                return
            
            # Ordena clusters por tamanho (maior primeiro)
            labels = sorted(sizes.keys(), key=lambda k: sizes[k], reverse=True)
            table.setRowCount(len(labels))
            
            for row, lab in enumerate(labels):
                # Obtém cor da LUT do PyVista
                r, g, b, a = lut.GetTableValue(int(lab))
                c = QColor(int(r*255), int(g*255), int(b*255))
                
                # Coluna Cor
                item_c = QtWidgets.QTableWidgetItem()
                item_c.setBackground(QBrush(c))
                item_c.setFlags(QtCore.Qt.ItemIsEnabled)
                table.setItem(row, 0, item_c)
                
                # Coluna ID
                table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(lab)))
                
                # Coluna Células
                table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(sizes[lab])))

        # Preenche tabela Base
        if self.compare_states.get("base"):
            fill_table(self.clus_table_base_cmp, self.compare_states["base"])
            
        # Preenche tabela Compare
        if self.compare_states.get("compare"):
            fill_table(self.clus_table_comp_cmp, self.compare_states["compare"])

    def on_plotter_slice_changed(self, axis, mode, value):
        self.slicer_widget.external_update(axis, mode, value)
        if self.tabs.currentIndex() == 3:
             if self.compare_states.get("base"): 
                 self.compare_states["base"][f"{axis}_{mode}"] = int(value)
                 self.compare_states["base"]["refresh"]()

    def on_ui_slice_changed(self, axis, mode, value):
        """Recebe alterações do Widget de Geometria (Cortes e Z-Exag)."""
        if axis == "z" and mode == "scale":
            # Atualiza Exagero Z
            self.state["z_exag"] = float(value)
            self.state["refresh"]() # Redesenha
        else:
            # Atualiza Cortes (I, J, K)
            if "set_slice" in self.state: 
                self.state["set_slice"](axis, mode, value)
        
        # Sincroniza com a Comparação (Cortes e Z)
        self.sync_slices_to_compare(axis, mode, value)

    def sync_slices_to_compare(self, axis, mode, value):
        """Replica cortes e exagero Z para os estados de comparação."""
        targets = []
        if self.compare_states.get("base"): targets.append(self.compare_states["base"])
        if self.compare_states.get("compare"): targets.append(self.compare_states["compare"])
        
        for st in targets:
            if axis == "z" and mode == "scale":
                st["z_exag"] = float(value)
            elif "set_slice" in st:
                st["set_slice"](axis, mode, value)
            
            if "refresh" in st: st["refresh"]()
            
        # Força renderização dos plotters secundários
        if axis == "z" and mode == "scale":
            if hasattr(self, 'comp_plotter_base'): self.comp_plotter_base.render()
            if hasattr(self, 'comp_plotter_comp'): self.comp_plotter_comp.render()

    def take_snapshot(self):
        file, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Snapshot", "snap.png", "PNG (*.png)")
        if file: self.plotter.screenshot(file)

    def on_legend_item_changed(self, item):
        if item.column() != 3: return
        fac = item.data(QtCore.Qt.UserRole)
        if self.state.get("mode") == "clusters": return
        current = self.state["reservoir_facies"]
        if item.checkState() == QtCore.Qt.Checked: current.add(fac)
        else: current.discard(fac)
        self.change_reservoir_facies(current)

    def populate_facies_legend(self):
        """Preenche a legenda lateral com as estatísticas do GRID ATIVO na visualização."""
        self.facies_legend_table.blockSignals(True)
        colors_dict = load_facies_colors()
        
        # CORREÇÃO: Usa as fácies do estado atual (Base ou Compare), não a global
        current_f = self.state.get("current_facies")
        if current_f is None:
            # Fallback para o global se o estado estiver vazio
            from load_data import facies as current_f
            
        present = sorted(set(int(v) for v in np.unique(current_f)))
        vals, counts = np.unique(current_f.astype(int), return_counts=True)
        cd = dict(zip(vals, counts))
        
        self.facies_legend_table.setRowCount(len(present))
        
        # Recupera o set de reservatório ativo no estado
        active_res_set = self.state.get("reservoir_facies", set())
        
        for row, fac in enumerate(present):
            # Cor
            rgba = colors_dict.get(fac, (0.8, 0.8, 0.8, 1.0))
            c = QColor(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
            item_c = QtWidgets.QTableWidgetItem()
            item_c.setBackground(QBrush(c))
            item_c.setFlags(QtCore.Qt.ItemIsEnabled)
            self.facies_legend_table.setItem(row, 0, item_c)
            
            # ID
            item_id = QtWidgets.QTableWidgetItem(str(fac))
            item_id.setFlags(QtCore.Qt.ItemIsEnabled)
            self.facies_legend_table.setItem(row, 1, item_id)
            
            # Count
            item_n = QtWidgets.QTableWidgetItem(str(cd.get(fac, 0)))
            item_n.setFlags(QtCore.Qt.ItemIsEnabled)
            self.facies_legend_table.setItem(row, 2, item_n)
            
            # Checkbox (Baseado no estado atual)
            chk = QtWidgets.QTableWidgetItem()
            chk.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            chk.setCheckState(QtCore.Qt.Checked if fac in active_res_set else QtCore.Qt.Unchecked)
            chk.setData(QtCore.Qt.UserRole, fac)
            self.facies_legend_table.setItem(row, 3, chk)
            
        self.facies_legend_table.resizeColumnsToContents()
        self.facies_legend_table.blockSignals(False)

    def populate_clusters_legend(self):
        """Preenche a legenda lateral com os clusters do modelo ativo."""
        self.facies_legend_table.blockSignals(True)
        
        sizes = self.state.get("clusters_sizes")
        lut = self.state.get("clusters_lut")
        
        if not sizes or not lut:
            self.facies_legend_table.setRowCount(0)
            self.facies_legend_table.blockSignals(False)
            return
            
        # Ordena por tamanho (maior primeiro)
        labels = sorted(sizes.keys(), key=lambda k: sizes[k], reverse=True)
        self.facies_legend_table.setRowCount(len(labels))
        self.facies_legend_table.setColumnCount(4) # Mantém 4 colunas para consistência
        self.facies_legend_table.setHorizontalHeaderLabels(["Cor", "Cluster ID", "Células", ""])
        
        for row, lab in enumerate(labels):
            # Cor
            r, g, b, a = lut.GetTableValue(int(lab))
            c = QColor(int(r*255), int(g*255), int(b*255))
            item_c = QtWidgets.QTableWidgetItem()
            item_c.setBackground(QBrush(c))
            item_c.setFlags(QtCore.Qt.ItemIsEnabled)
            self.facies_legend_table.setItem(row, 0, item_c)
            
            # ID
            item_id = QtWidgets.QTableWidgetItem(str(lab))
            item_id.setFlags(QtCore.Qt.ItemIsEnabled)
            self.facies_legend_table.setItem(row, 1, item_id)
            
            # Count
            item_n = QtWidgets.QTableWidgetItem(str(sizes[lab]))
            item_n.setFlags(QtCore.Qt.ItemIsEnabled)
            self.facies_legend_table.setItem(row, 2, item_n)
            
            # Checkbox (Vazio/Desabilitado em Clusters)
            item_blk = QtWidgets.QTableWidgetItem()
            item_blk.setFlags(QtCore.Qt.NoItemFlags)
            self.facies_legend_table.setItem(row, 3, item_blk)
            
        self.facies_legend_table.resizeColumnsToContents()
        self.facies_legend_table.blockSignals(False)

    def change_mode(self, new_mode):
        # 1. Identifica qual modelo está ativo na visualização principal
        current_model_key = "base"
        sel = self.project_tree.selectedItems()
        if sel:
            key = sel[0].data(0, QtCore.Qt.UserRole + 1)
            if key in ["base", "compare"]:
                current_model_key = key

        # 2. Salva a preferência NESTE modelo
        self.models[current_model_key]["view_mode"] = new_mode
        
        # 3. Atualiza estado global e redesenha
        self.state["mode"] = new_mode
        self.state["refresh"]()
        
        # 4. Atualiza também a comparação (se aberta)
        for k in ["base", "compare"]:
            st = self.compare_states.get(k)
            if st:
                st["mode"] = new_mode
                if new_mode == "clusters":
                    rf = self.models[k]["reservoir_facies"]
                    if "update_reservoir_fields" in st: st["update_reservoir_fields"](rf)
                if "refresh" in st: st["refresh"]()
        
        # 5. Atualiza Interface Lateral
        self.legend_group.setTitle("Legenda & Filtro" if new_mode != "clusters" else "Legenda Clusters")
        if new_mode == "clusters": 
            self.populate_clusters_legend()
            self.populate_compare_clusters_tables()
        else: 
            self.populate_facies_legend()
            
        if hasattr(self, 'comp_plotter_base'): self.comp_plotter_base.render()
        if hasattr(self, 'comp_plotter_comp'): self.comp_plotter_comp.render()

    def change_thickness_mode(self, label):
        self.state["thickness_mode"] = label
        if "update_thickness" in self.state: self.state["update_thickness"]()
        self.state["refresh"]()
        self.update_2d_map()
        self.update_compare_2d_maps()

    def setup_comparison_dock_content(self):
        """Painel esquerdo de Comparação: Métricas e Filtros Unificados."""
        self.dock_compare_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.dock_compare_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        
        # --- 1. MÉTRICAS (Topo - Mantido igual) ---
        gb_metrics = QtWidgets.QGroupBox("Métricas Comparadas")
        l_met = QtWidgets.QVBoxLayout(gb_metrics)
        l_met.setContentsMargins(2, 5, 2, 2)
        tabs_metrics = QtWidgets.QTabWidget()
        
        # Aba Global
        tab_glob = QtWidgets.QWidget(); l_glob = QtWidgets.QVBoxLayout(tab_glob)
        self.global_compare_table = QtWidgets.QTableWidget()
        self.global_compare_table.setColumnCount(4)
        self.global_compare_table.setHorizontalHeaderLabels(["Métrica", "Base", "Comp", "Dif"])
        l_glob.addWidget(self.global_compare_table)
        tabs_metrics.addTab(tab_glob, "Global")
        
        # Aba Fácies
        tab_fac = QtWidgets.QWidget(); l_fac = QtWidgets.QVBoxLayout(tab_fac)
        self.facies_compare_table = QtWidgets.QTableWidget()
        self.facies_compare_table.setColumnCount(9)
        self.facies_compare_table.setHorizontalHeaderLabels(["Fác", "Cel B", "%", "Cel C", "%", "Vol B", "Vol C", "Esp B", "Esp C"])
        l_fac.addWidget(self.facies_compare_table)
        tabs_metrics.addTab(tab_fac, "Fácies")
        
        # Aba Reservatório
        tab_res = QtWidgets.QWidget(); l_res = QtWidgets.QVBoxLayout(tab_res)
        self.reservoir_facies_compare_table = QtWidgets.QTableWidget()
        self.reservoir_facies_compare_table.setColumnCount(5)
        self.reservoir_facies_compare_table.setHorizontalHeaderLabels(["Fác", "Cel B", "%", "Cel C", "%"])
        l_res.addWidget(self.reservoir_facies_compare_table)
        tabs_metrics.addTab(tab_res, "Reserv.")
        
        l_met.addWidget(tabs_metrics)
        splitter.addWidget(gb_metrics)

        # --- 2. FILTROS 3D UNIFICADOS (NOVO) ---
        gb_ctrl = QtWidgets.QGroupBox("Filtros Visualização 3D")
        l_ctrl = QtWidgets.QVBoxLayout(gb_ctrl)
        l_ctrl.setContentsMargins(2, 5, 2, 2)
        
        # Tabela Unificada
        self.unified_filter_table = QtWidgets.QTableWidget()
        self.unified_filter_table.setColumnCount(6)
        self.unified_filter_table.setHorizontalHeaderLabels(["Cor", "Fácies", "Cel B", "Cel C", "Sel B", "Sel C"])
        self.unified_filter_table.verticalHeader().setVisible(False)
        # Ajuste fino das larguras para caber tudo
        header = self.unified_filter_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents) # Cor
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents) # ID
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)          # Cel B
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)          # Cel C
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents) # Sel B
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeToContents) # Sel C
        
        self.unified_filter_table.itemChanged.connect(self.on_unified_filter_changed)
        
        l_ctrl.addWidget(QtWidgets.QLabel("Seleção Unificada:"))
        l_ctrl.addWidget(self.unified_filter_table)
        
        # Legenda Clusters (Separado pois depende do clique em cada visualizador)
        # Vamos usar abas SÓ para a legenda de clusters, pois elas podem ser muito diferentes
        tabs_cluster = QtWidgets.QTabWidget()
        self.clus_table_base_cmp = self._create_legend_table(["Cor","ID","Cel"])
        tabs_cluster.addTab(self.clus_table_base_cmp, "Clusters Base")
        
        self.clus_table_comp_cmp = self._create_legend_table(["Cor","ID","Cel"])
        tabs_cluster.addTab(self.clus_table_comp_cmp, "Clusters Comp")
        
        l_ctrl.addWidget(tabs_cluster)
        
        splitter.addWidget(gb_ctrl)
        layout.addWidget(splitter)
        
        self.dock_compare_widget.setVisible(False)
        return self.dock_compare_widget
    
    def fill_unified_facies_table(self):
        """Preenche a tabela unificada com a união das fácies dos dois modelos."""
        self.unified_filter_table.blockSignals(True)
        colors = load_facies_colors()
        
        # 1. Obtém contagens Base
        f0 = self.models["base"]["facies"]
        if f0 is not None:
            v0, c0 = np.unique(f0.astype(int), return_counts=True)
            dict0 = dict(zip(v0, c0))
        else: dict0 = {}
            
        # 2. Obtém contagens Comparado
        f1 = self.models["compare"]["facies"]
        if f1 is not None:
            v1, c1 = np.unique(f1.astype(int), return_counts=True)
            dict1 = dict(zip(v1, c1))
        else: dict1 = {}
        
        # 3. União das chaves ordenadas
        all_facies = sorted(set(dict0.keys()) | set(dict1.keys()))
        
        self.unified_filter_table.setRowCount(len(all_facies))
        
        # Recupera sets de reservatório atuais
        res0 = self.models["base"]["reservoir_facies"]
        res1 = self.models["compare"]["reservoir_facies"]
        
        for row, fac in enumerate(all_facies):
            # Col 0: Cor
            rgba = colors.get(fac, (0.8, 0.8, 0.8, 1.0))
            brush = QBrush(QColor(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)))
            item_c = QtWidgets.QTableWidgetItem()
            item_c.setBackground(brush)
            item_c.setFlags(QtCore.Qt.ItemIsEnabled)
            self.unified_filter_table.setItem(row, 0, item_c)
            
            # Col 1: ID Fácies
            item_id = QtWidgets.QTableWidgetItem(str(fac))
            item_id.setFlags(QtCore.Qt.ItemIsEnabled)
            self.unified_filter_table.setItem(row, 1, item_id)
            
            # Col 2: Células Base (Vazio se não existir)
            txt0 = str(dict0[fac]) if fac in dict0 else ""
            item_b = QtWidgets.QTableWidgetItem(txt0)
            item_b.setFlags(QtCore.Qt.ItemIsEnabled)
            self.unified_filter_table.setItem(row, 2, item_b)
            
            # Col 3: Células Comp (Vazio se não existir)
            txt1 = str(dict1[fac]) if fac in dict1 else ""
            item_cp = QtWidgets.QTableWidgetItem(txt1)
            item_cp.setFlags(QtCore.Qt.ItemIsEnabled)
            self.unified_filter_table.setItem(row, 3, item_cp)
            
            # Col 4: Checkbox Base
            check0 = QtWidgets.QTableWidgetItem()
            if fac in dict0: # Só habilita se existir no modelo
                check0.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                check0.setCheckState(QtCore.Qt.Checked if fac in res0 else QtCore.Qt.Unchecked)
            else:
                check0.setFlags(QtCore.Qt.NoItemFlags) # Desabilita
            check0.setData(QtCore.Qt.UserRole, fac) # Guarda ID
            self.unified_filter_table.setItem(row, 4, check0)
            
            # Col 5: Checkbox Comp
            check1 = QtWidgets.QTableWidgetItem()
            if fac in dict1: # Só habilita se existir no modelo
                check1.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                check1.setCheckState(QtCore.Qt.Checked if fac in res1 else QtCore.Qt.Unchecked)
            else:
                check1.setFlags(QtCore.Qt.NoItemFlags) # Desabilita
            check1.setData(QtCore.Qt.UserRole, fac) # Guarda ID
            self.unified_filter_table.setItem(row, 5, check1)
            
        self.unified_filter_table.blockSignals(False)

    def on_unified_filter_changed(self, item):
        """Lida com cliques na tabela unificada."""
        col = item.column()
        if col not in [4, 5]: return # Só processa cliques nos checkboxes
        
        fac = item.data(QtCore.Qt.UserRole)
        is_checked = (item.checkState() == QtCore.Qt.Checked)
        
        if col == 4: # Base
            target_set = self.models["base"]["reservoir_facies"]
            if is_checked: target_set.add(fac)
            else: target_set.discard(fac)
            # Atualiza visualização Base
            self.update_compare_3d_mode_single("base")
            
        elif col == 5: # Compare
            target_set = self.models["compare"]["reservoir_facies"]
            if is_checked: target_set.add(fac)
            else: target_set.discard(fac)
            # Atualiza visualização Compare
            self.update_compare_3d_mode_single("compare")

    def update_compare_3d_mode_single(self, key):
        """Atualiza a visualização (3D e 2D) de um lado da comparação após mudança de filtro."""
        st = self.compare_states.get(key)
        if st:
            rf = self.models[key]["reservoir_facies"]
            
            # 1. Recalcula as propriedades no grid 3D (Isso afeta o cálculo de espessura)
            if "update_reservoir_fields" in st: 
                st["update_reservoir_fields"](rf)
            
            # 2. Redesenha a cena 3D
            if "refresh" in st: 
                st["refresh"]()
            
        # 3. CRÍTICO: Manda atualizar também os Mapas 2D
        # (Agora que o grid 3D foi recalculado no passo 1, o 2D vai ler os dados novos)
        self.update_compare_2d_maps()
    
    def switch_perspective(self, mode):
        # Alterna visual (Menus e Docks)
        if mode == "visualization":
            self.act_persp_viz.setChecked(True); self.act_persp_comp.setChecked(False)
            self.dock_left_container.setCurrentIndex(0) # Arvore
            self.dock_explorer.setWindowTitle("Project Explorer")
            self.dock_props.setVisible(True)
            self.central_stack.setCurrentIndex(0) # Abas normais
            
            # Esconde botão 2D Compare
            self.act_toggle_2d.setVisible(False)
            
        elif mode == "comparison":
            self.act_persp_viz.setChecked(False); self.act_persp_comp.setChecked(True)
            self.dock_left_container.setCurrentIndex(1) # Painel Compare
            self.dock_explorer.setWindowTitle("Painel de Comparação")
            self.dock_props.setVisible(False)
            self.central_stack.setCurrentIndex(1) # 3D/2D Compare
            
            # Mostra botão 2D Compare
            self.act_toggle_2d.setVisible(True)
            
            self.update_comparison_tables()
            self.update_compare_2d_maps()

    def toggle_compare_2d_view(self):
        """Mostra/Esconde os mapas 2D na aba de comparação."""
        show = self.act_toggle_2d.isChecked()
        self.maps_2d_container.setVisible(show)
        # Ajusta o tamanho se abrir (70% 3D, 30% 2D)
        if show:
            self.main_split_compare.setSizes([700, 300])

    def generate_detailed_metrics_df(self, facies_array):
        """Gera o DataFrame detalhado para a tabela de métricas."""
        # Garante array numpy
        arr = np.asarray(facies_array, dtype=int)
        total_cells = arr.size
        
        # Reutiliza volumes e Z do grid base (assumindo mesma geometria)
        # Se os grids tiverem geometrias diferentes, isso precisaria ser ajustado.
        vols = _get_cell_volumes() 
        z_vals = _get_cell_z_coords()
        
        unique_f = np.unique(arr)
        data_list = []
        
        for f in unique_f:
            mask = (arr == f)
            count = int(mask.sum())
            if count == 0: continue
            
            # Estatísticas Básicas
            frac = count / total_cells
            vol_tot = float(vols[mask].sum())
            
            # Análise de Clusters (Labeling)
            mask_3d = mask.reshape((nx, ny, nz), order="F")
            struct = generate_binary_structure(3, 1)
            # Transpose para ordem (z,y,x) do scipy
            lbl_3d, n_clus = label(mask_3d.transpose(2,1,0), structure=struct)
            
            largest_size = 0
            vol_largest = 0.0
            thick = 0.0
            
            if n_clus > 0:
                # Flatten de volta para contar
                lbl_flat = lbl_3d.transpose(2,1,0).reshape(-1, order="F")
                counts = np.bincount(lbl_flat)
                counts[0] = 0 # ignora fundo
                
                largest_idx = counts.argmax()
                largest_size = counts[largest_idx]
                
                # Propriedades do Maior Cluster
                mask_largest = (lbl_flat == largest_idx)
                vol_largest = float(vols[mask_largest].sum())
                
                zs = z_vals[mask_largest]
                if zs.size > 0:
                    thick = float(zs.max() - zs.min())
            
            conn = largest_size / count if count > 0 else 0
            
            data_list.append({
                "facies": int(f),
                "cells": count,
                "fraction": frac,
                "n_clusters": n_clus,
                "largest_size": largest_size,
                "connected_fraction": conn,
                "volume_total": vol_tot,
                "volume_largest_cluster": vol_largest,
                "thickness_largest_cluster": thick
                # Adicione percolaçao aqui se desejar (requer mais calculo)
            })
            
        return pd.DataFrame(data_list)

# --- LÓGICA DE FECHAMENTO (LIMPEZA) ---
    def closeEvent(self, event):
        """Garante que os processos do VTK sejam encerrados antes de matar a janela."""
        
        # Fecha os plotters principais para parar o loop de renderização
        if hasattr(self, 'plotter'): 
            self.plotter.close()
        if hasattr(self, 'plotter_2d'): 
            self.plotter_2d.close()
        
        # Fecha os plotters da aba de comparação (se existirem)
        if hasattr(self, 'comp_plotter_base'): self.comp_plotter_base.close()
        if hasattr(self, 'comp_plotter_comp'): self.comp_plotter_comp.close()
        if hasattr(self, 'comp_plotter_base_2d'): self.comp_plotter_base_2d.close()
        if hasattr(self, 'comp_plotter_comp_2d'): self.comp_plotter_comp_2d.close()
        
        # Aceita o evento de fechamento
        event.accept()