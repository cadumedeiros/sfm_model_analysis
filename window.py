# window.py
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QColor, QBrush
from pyvistaqt import BackgroundPlotter
import numpy as np
import os
import pandas as pd
from scipy.ndimage import label, generate_binary_structure
from matplotlib.colors import ListedColormap

from visualize import run, get_2d_clim, make_clusters_lut, compute_cluster_sizes, prepare_grid_indices
from load_data import facies, nx, ny, nz
from config import load_facies_colors, load_markers

from analysis import (
    facies_distribution_array,
    reservoir_facies_distribution_array,
    compute_global_metrics_for_array,
    _get_cell_volumes,
    _get_cell_z_coords,
    _get_cell_thickness,
    sample_well_from_grid, 
    calculate_well_accuracy,
    print_layers
)
from wells import Well

# --- WIDGET CUSTOMIZADO PARA OS SLIDERS (Grid Explorer) ---
class GridSlicerWidget(QtWidgets.QGroupBox):
    def __init__(self, nx, ny, nz, callback, initial_z=15.0):
        super().__init__("Geometria (Cortes & Escala)")
        self.callback = callback 
        self.is_updating = False
        
        layout = QtWidgets.QVBoxLayout(self)
        
        layout.addWidget(QtWidgets.QLabel(f"Inline (I): 0 - {nx-1}"))
        self.i_widgets = self._create_axis_control("i", nx)
        layout.addLayout(self.i_widgets['layout'])
        
        layout.addWidget(QtWidgets.QLabel(f"Crossline (J): 0 - {ny-1}"))
        self.j_widgets = self._create_axis_control("j", ny)
        layout.addLayout(self.j_widgets['layout'])
        
        layout.addWidget(QtWidgets.QLabel(f"Layer (K): 0 - {nz-1}"))
        self.k_widgets = self._create_axis_control("k", nz)
        layout.addLayout(self.k_widgets['layout'])
        
        layout.addSpacing(10)
        
        # Exagero Z
        layout.addWidget(QtWidgets.QLabel("Exagero Vertical (Z):"))
        h_z = QtWidgets.QHBoxLayout()
        self.spin_z = QtWidgets.QDoubleSpinBox(); self.spin_z.setRange(1.0, 100.0); self.spin_z.setSingleStep(1.0); self.spin_z.setValue(initial_z); self.spin_z.setFixedWidth(60)
        self.slider_z = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slider_z.setRange(1, 100); self.slider_z.setValue(int(initial_z))
        
        self.spin_z.valueChanged.connect(self._on_z_spin_change)
        self.slider_z.valueChanged.connect(self._on_z_slider_change)
        h_z.addWidget(self.spin_z); h_z.addWidget(self.slider_z)
        layout.addLayout(h_z)

    def _on_z_spin_change(self, val):
        if self.is_updating: return
        self.is_updating = True; self.slider_z.setValue(int(val)); self.callback("z", "scale", val); self.is_updating = False

    def _on_z_slider_change(self, val):
        if self.is_updating: return
        self.is_updating = True; self.spin_z.setValue(float(val)); self.callback("z", "scale", float(val)); self.is_updating = False

    def _create_axis_control(self, axis, limit):
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
            self.callback(axis, "min", val) # Chamada DIRETA (Sem Timer)
            self.is_updating = False

        def update_max(val):
            if self.is_updating: return
            self.is_updating = True
            if val < spin_min.value(): val = spin_min.value()
            spin_max.setValue(val); slider_max.setValue(val)
            self.callback(axis, "max", val) # Chamada DIRETA
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
            self.spin_z.setValue(float(value)); self.slider_z.setValue(int(value))
        else:
            widgets = getattr(self, f"{axis}_widgets")
            val = int(value)
            if mode == "min": widgets['spin_min'].setValue(val); widgets['slider_min'].setValue(val)
            else: widgets['spin_max'].setValue(val); widgets['slider_max'].setValue(val)
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

# --- CLASSE PRINCIPAL ---
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, mode, z_exag, show_scalar_bar, reservoir_facies):
        super().__init__()
        self.setWindowTitle("Grid View Analysis")

        self.current_mode = mode

        self.open_reports = []
        
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

        self.wells = {}
        
        self.facies_colors = load_facies_colors() # Sua função
        self.markers_db = load_markers("assets/wellMarkers.txt")
        
        # Criação do Colormap
        self.pv_cmap = None
        self.clim = None
        if self.facies_colors:
            # Ordena IDs: 11, 12, 13...
            ids = sorted(self.facies_colors.keys())
            colors = [self.facies_colors[i] for i in ids]
            
            # Colormap DISCRETO
            self.pv_cmap = ListedColormap(colors)
            # Limites exatos para forçar o PyVista a não interpolar errado
            # Ex: se vai de 11 a 22, clim=[11, 22]
            self.clim = [ids[0], ids[-1]]

        self.state = {"reservoir_facies": initial_reservoir, "mode": mode}
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
        # self.fill_unified_facies_table()
        
        # Calcula métricas iniciais para o modelo base
        self.change_reservoir_facies(initial_reservoir)

        # Seleciona o primeiro item da árvore (Base) para inicializar a UI lateral
        top_item = self.project_tree.topLevelItem(0)
        if top_item: 
            top_item.setExpanded(True)
            self.project_tree.setCurrentItem(top_item)
    
    def open_selected_well_reports(self):
        from PyQt5 import QtCore

        items = self.project_tree.selectedItems()
        if not items:
            return

        # 1) tenta inferir o modelo a partir de qualquer item de modelo selecionado
        selected_model_key = None
        for it in items:
            role = it.data(0, QtCore.Qt.UserRole)
            if role in ("model_root", "grid_settings", "prop_settings", "metrics_view", "map2d_view"):
                mk = it.data(0, QtCore.Qt.UserRole + 1)
                if mk:
                    selected_model_key = mk
                    break

        # 2) fallback: último modelo “ativo”
        if not selected_model_key:
            selected_model_key = self.state.get("active_model_key", "base")

        # ✅ sanitize: se por algum motivo vier lixo (ex: nome de poço), cai pro base
        if selected_model_key not in self.models:
            selected_model_key = "base"

        # 3) pega poços selecionados
        well_names = []
        for it in items:
            if it.data(0, QtCore.Qt.UserRole) == "well_item":
                wn = it.data(0, QtCore.Qt.UserRole + 1)
                if wn:
                    well_names.append(wn)

        for w in well_names:
            self.show_well_comparison_report(w, selected_model_key)

 
    def setup_ui(self, nx, ny, nz):
        self.resize(1600, 900)

        menubar = self.menuBar()

        # --- Arquivo ---
        file_menu = menubar.addMenu("Arquivo")

        action_load = QtWidgets.QAction("Carregar Modelo Adicional...", self)
        action_load.triggered.connect(self.open_compare_dialog)

        action_load_well = QtWidgets.QAction("Carregar Poço (.las + .dev)...", self)
        action_load_well.triggered.connect(self.load_well_dialog)

        file_menu.addAction(action_load)
        file_menu.addAction(action_load_well)
        file_menu.addSeparator()

        action_exit = QtWidgets.QAction("Sair", self)
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)

        # --- Exibir ---
        self.view_menu = menubar.addMenu("Exibir")

        # Perspectivas
        self.act_persp_viz = QtWidgets.QAction("Visualização", self)
        self.act_persp_viz.setCheckable(True)
        self.act_persp_viz.setChecked(True)
        self.act_persp_viz.triggered.connect(lambda: self.switch_perspective("visualization"))

        self.act_persp_comp = QtWidgets.QAction("Comparação", self)
        self.act_persp_comp.setCheckable(True)
        self.act_persp_comp.triggered.connect(lambda: self.switch_perspective("comparison"))

        # Ribbon
        self.setup_toolbar_controls()

        self.ribbon_toolbar = QtWidgets.QToolBar("Ribbon")
        self.ribbon_toolbar.setMovable(False)
        self.ribbon_toolbar.setFloatable(False)
        self.ribbon_toolbar.setAllowedAreas(QtCore.Qt.TopToolBarArea)
        self.ribbon_toolbar.setStyleSheet("QToolBar { border: 0px; }")
        self.ribbon_toolbar.addWidget(self.ribbon)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.ribbon_toolbar)

        # Central
        self.central_stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.central_stack)

        # Docks
        self.setup_docks(nx, ny, nz)

        self.view_menu.addAction(self.dock_explorer.toggleViewAction())
        self.view_menu.addAction(self.dock_props.toggleViewAction())
        self.view_menu.addSeparator()
        self.view_menu.addAction(self.ribbon_toolbar.toggleViewAction())

        # --- PERSPECTIVA 1: VISUALIZAÇÃO ---
        self.viz_container = QtWidgets.QStackedWidget()
        self.tabs = self.viz_container

        # Pag 0: 3D
        self.viz_tab = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(self.viz_tab)
        vl.setContentsMargins(0, 0, 0, 0)
        self.plotter, plotter_widget = self._make_embedded_plotter(parent=self.viz_tab)
        vl.addWidget(plotter_widget)
        self.viz_container.addWidget(self.viz_tab)

        # Pag 1: Mapas 2D
        self.map2d_tab = QtWidgets.QWidget()
        ml = QtWidgets.QVBoxLayout(self.map2d_tab)
        ml.setContentsMargins(0, 0, 0, 0)
        self.plotter_2d, plotter_2d_widget = self._make_embedded_plotter(parent=self.map2d_tab)
        ml.addWidget(plotter_2d_widget)
        self.viz_container.addWidget(self.map2d_tab)

        # Pag 2: Métricas
        self.details_tab = QtWidgets.QWidget()
        l_det = QtWidgets.QVBoxLayout(self.details_tab)
        l_det.setContentsMargins(8, 8, 8, 8)
        l_det.setSpacing(8)

        self.central_metrics_text = QtWidgets.QTextEdit()
        self.central_metrics_text.setReadOnly(True)
        self.central_metrics_text.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        l_det.addWidget(QtWidgets.QLabel("Resumo Global"))
        l_det.addWidget(self.central_metrics_text, 2)

        self.facies_table = QtWidgets.QTableWidget()
        self.facies_table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        l_det.addWidget(QtWidgets.QLabel("Detalhamento por Fácies"))
        l_det.addWidget(self.facies_table, 3)

        self.viz_container.addWidget(self.details_tab)
        
        # Pag 3: Ranking (ATUALIZADO COM BOTÕES DE COPIAR e COLUNA STUDY)
        self.ranking_tab = QtWidgets.QWidget()
        l_rank = QtWidgets.QVBoxLayout(self.ranking_tab)
        l_rank.setContentsMargins(8, 8, 8, 8)
        
        self.ranking_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        
        # --- Container Superior: Tabela de Modelos ---
        w_top = QtWidgets.QWidget()
        l_top = QtWidgets.QVBoxLayout(w_top)
        l_top.setContentsMargins(0, 0, 0, 0)
        
        h_top_bar = QtWidgets.QHBoxLayout()
        h_top_bar.addWidget(QtWidgets.QLabel("Ranking Global de Modelos"))
        h_top_bar.addStretch(1)
        btn_copy_models = QtWidgets.QPushButton("Copiar Tabela")
        btn_copy_models.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        btn_copy_models.clicked.connect(lambda: self._copy_table_to_clipboard(self.tbl_models))
        h_top_bar.addWidget(btn_copy_models)
        l_top.addLayout(h_top_bar)

        self.tbl_models = QtWidgets.QTableWidget()
        # Colunas: Rank, Study, Modelo, Score, Fácies(acc), Fácies(kappa), Poços
        self.tbl_models.setColumnCount(7)
        self.tbl_models.setHorizontalHeaderLabels(["Rank", "Study", "Modelo", "Score", "Fácies (acc)", "Fácies (kappa)", "Poços"])
        self.tbl_models.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_models.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tbl_models.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_models.setSortingEnabled(True)
        self.tbl_models.itemSelectionChanged.connect(self._on_models_table_selection_changed)
        l_top.addWidget(self.tbl_models)
        
        # --- Container Inferior: Tabela de Poços ---
        w_bot = QtWidgets.QWidget()
        l_bot = QtWidgets.QVBoxLayout(w_bot)
        l_bot.setContentsMargins(0, 0, 0, 0)
        
        h_bot_bar = QtWidgets.QHBoxLayout()
        h_bot_bar.addWidget(QtWidgets.QLabel("Detalhamento por Poço (Modelo Selecionado)"))
        h_bot_bar.addStretch(1)
        btn_copy_wells = QtWidgets.QPushButton("Copiar Tabela")
        btn_copy_wells.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        btn_copy_wells.clicked.connect(lambda: self._copy_table_to_clipboard(self.tbl_wells))
        h_bot_bar.addWidget(btn_copy_wells)
        l_bot.addLayout(h_bot_bar)

        self.tbl_wells = QtWidgets.QTableWidget()
        self.tbl_wells.setColumnCount(7) 
        self.tbl_wells.setHorizontalHeaderLabels(["Poço", "Score", "Fácies (acc)", "Fácies (kappa)", "Espessura", "T_real", "T_sim", "Ações"])
        self.tbl_wells.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_wells.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tbl_wells.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_wells.setSortingEnabled(True)
        l_bot.addWidget(self.tbl_wells)
        
        self.ranking_splitter.addWidget(w_top)
        self.ranking_splitter.addWidget(w_bot)
        self.ranking_splitter.setStretchFactor(0, 1)
        self.ranking_splitter.setStretchFactor(1, 2)
        
        l_rank.addWidget(self.ranking_splitter)
        self.viz_container.addWidget(self.ranking_tab)

        self.central_stack.addWidget(self.viz_container)

        # --- PERSPECTIVA 2: COMPARAÇÃO (Mantida igual) ---
        self.compare_stack = QtWidgets.QStackedWidget()
        self.comp_page_3d = QtWidgets.QWidget()
        self.comp_layout_3d = QtWidgets.QVBoxLayout(self.comp_page_3d)
        self.comp_layout_3d.setContentsMargins(0, 0, 0, 0)
        self.compare_stack.addWidget(self.comp_page_3d)

        self.comp_page_metrics = QtWidgets.QWidget()
        self.comp_metrics_layout = QtWidgets.QVBoxLayout(self.comp_page_metrics)
        self.comp_metrics_layout.setContentsMargins(6, 6, 6, 6)
        self.tabs_compare_metrics = QtWidgets.QTabWidget()
        t_fa = QtWidgets.QWidget(); l_fa = QtWidgets.QVBoxLayout(t_fa)
        self.facies_compare_table = QtWidgets.QTableWidget(); l_fa.addWidget(self.facies_compare_table)
        self.tabs_compare_metrics.addTab(t_fa, "Fácies")
        t_res = QtWidgets.QWidget(); l_res = QtWidgets.QVBoxLayout(t_res)
        self.reservoir_facies_compare_table = QtWidgets.QTableWidget(); l_res.addWidget(self.reservoir_facies_compare_table)
        self.tabs_compare_metrics.addTab(t_res, "Reservatório")
        self.comp_metrics_layout.addWidget(self.tabs_compare_metrics)
        self.compare_stack.addWidget(self.comp_page_metrics)

        self.comp_page_2d = QtWidgets.QWidget()
        self.comp_2d_layout = QtWidgets.QVBoxLayout(self.comp_page_2d)
        self.comp_2d_layout.setContentsMargins(0, 0, 0, 0)
        self.compare_stack.addWidget(self.comp_page_2d)
        self.central_stack.addWidget(self.compare_stack)

    def _copy_table_to_clipboard(self, table_widget):
        """Copia o conteúdo de uma QTableWidget para o clipboard (formato CSV/Excel)."""
        if table_widget.rowCount() == 0:
            return

        cols = table_widget.columnCount()
        rows = table_widget.rowCount()

        # 1. Cabeçalhos
        headers = []
        for c in range(cols):
            it = table_widget.horizontalHeaderItem(c)
            if it:
                headers.append(it.text())
            else:
                headers.append("")
        
        # Junta com TABs
        clipboard_text = "\t".join(headers) + "\n"

        # 2. Linhas
        for r in range(rows):
            row_data = []
            for c in range(cols):
                # Se tiver widget na célula (ex: botões na última coluna), ignora ou põe placeholder
                if table_widget.cellWidget(r, c):
                    row_data.append("") # Deixa vazio no excel
                else:
                    it = table_widget.item(r, c)
                    if it:
                        # Substitui quebras de linha por espaço pra não quebrar o CSV
                        txt = it.text().replace("\n", " ").replace("\t", " ")
                        row_data.append(txt)
                    else:
                        row_data.append("")
            
            clipboard_text += "\t".join(row_data) + "\n"

        # Envia para o Clipboard do SO
        QtWidgets.QApplication.clipboard().setText(clipboard_text)
        
        # Feedback visual rápido na barra de status (se existir) ou print
        print("Tabela copiada para a área de transferência.")

    def _make_embedded_plotter(self, parent=None):
        """Cria um plotter do PyVista adequado para EMBED dentro de layouts Qt.

        No macOS, o BackgroundPlotter pode falhar em embedar (janela preta/vazia).
        O QtInteractor costuma ser mais estável para widgets embutidos.
        Retorna (plotter, widget_para_layout).
        """
        try:
            from pyvistaqt import QtInteractor
            p = QtInteractor(parent or self)
            # Melhor foco/atalhos quando embedado
            p.setFocusPolicy(QtCore.Qt.StrongFocus)
            return p, p
        except Exception:
            p = BackgroundPlotter(show=False)
            return p, p.interactor


    def load_well_dialog(self):
        """Diálogo para selecionar VÁRIOS poços (.las + _dev) de uma vez."""
        # 1) Seleciona múltiplos LAS
        las_paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Selecione 1 ou mais arquivos .LAS",
            "",
            "LAS Files (*.las)"
        )
        if not las_paths:
            return

        def _guess_dev_path(las_path: str) -> str | None:
            """Tenta achar o arquivo de trajetória baseado no nome do LAS."""
            base_name = os.path.splitext(las_path)[0]
            candidates = [
                base_name + "_dev",        # seu padrão atual (sem extensão)
                base_name + "_dev.dev",
                base_name + "_dev.txt",
                base_name + ".dev",
            ]
            for p in candidates:
                if os.path.exists(p):
                    return p
            return None

        loaded = []
        skipped = []

        # 2) Para cada LAS, tenta encontrar DEV e carregar
        for las_path in las_paths:
            base_name = os.path.splitext(las_path)[0]
            well_name = os.path.basename(base_name)

            dev_path = _guess_dev_path(las_path)

            # Se não achou automaticamente, pergunta (um por poço “faltante”)
            if dev_path is None:
                dev_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self,
                    f"Selecione o arquivo de Trajetória (_dev) para o poço: {well_name}",
                    os.path.dirname(las_path),
                    "All Files (*)"
                )
                if not dev_path:
                    skipped.append((well_name, "DEV não selecionado"))
                    continue

            try:
                new_well = Well(well_name, dev_path, las_path)

                if new_well.data is None or new_well.data.empty:
                    raise ValueError("Falha ao sincronizar LAS e DEV.")

                self.wells[well_name] = new_well
                loaded.append(well_name)
                print(f"Poço {well_name} carregado.")

                if hasattr(self, "wells_root_item") and self.wells_root_item is not None:
                    w_item = QtWidgets.QTreeWidgetItem(self.wells_root_item, [well_name])
                    w_item.setData(0, QtCore.Qt.UserRole, "well_item")
                    w_item.setData(0, QtCore.Qt.UserRole + 1, well_name)

                    # ✅ Ícone + checkbox (visível no 3D)
                    w_item.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_ArrowRight))
                    w_item.setFlags(w_item.flags() | QtCore.Qt.ItemIsUserCheckable)
                    w_item.setCheckState(0, QtCore.Qt.Checked)

            except Exception as e:
                skipped.append((well_name, str(e)))

        # 3) Atualiza 3D uma vez no final (bem mais rápido)
        if loaded:
            self.update_wells_3d()

        # 4) Resumo
        if skipped:
            msg = "Alguns poços não foram carregados:\n\n" + "\n".join(
                [f"- {n}: {err}" for n, err in skipped]
            )
            QtWidgets.QMessageBox.warning(self, "Carregar Poços", msg)


    def update_z_exaggeration(self):
        val = self.slider_z.value()
        new_z = val / 10.0
        self.lbl_z_val.setText(f"{new_z:.1f}x")

        self.state["z_exag"] = new_z

        # Refresh do 3D (actor scale)
        refresh = self.state.get("refresh")
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass

        # poços precisam ser redesenhados porque a trajetória depende do z_exag
        self._schedule_wells_update()


    def update_wells_3d(self):
        if not hasattr(self, "plotter") or self.plotter is None:
            return
        if not getattr(self, "wells", None):
            return

        z_exag = float(self.state.get("z_exag", 1.0))

        # 1. Identifica quais poços devem estar visíveis
        checked = set()
        it = QtWidgets.QTreeWidgetItemIterator(self.project_tree)
        while it.value():
            item = it.value()
            role = item.data(0, QtCore.Qt.UserRole)
            if role == "well_item" and item.checkState(0) == QtCore.Qt.Checked:
                wn = item.data(0, QtCore.Qt.UserRole + 1) or item.text(0)
                if wn:
                    checked.add(str(wn))
            it += 1

        # Otimização de estado
        state_key = (tuple(sorted(checked)), z_exag)
        if getattr(self, "_wells_draw_state", None) == state_key:
            return
        self._wells_draw_state = state_key

        if not hasattr(self, "_well_actors"):
            self._well_actors = {}

        # 2. LIMPEZA: Remove atores antigos
        for wn in list(self._well_actors.keys()):
            if wn not in checked:
                actors_list = self._well_actors.pop(wn)
                if not isinstance(actors_list, list):
                    actors_list = [actors_list]
                
                for actor in actors_list:
                    try:
                        self.plotter.remove_actor(actor)
                    except Exception:
                        pass

        # 3. DESENHO
        markers_db = getattr(self, "markers_db", {})

        for well_name in sorted(checked):
            if well_name in self._well_actors:
                continue

            well = self.wells.get(well_name)
            if well is None:
                continue

            current_well_actors = []

            try:
                # --- A) TUBO DO POÇO ---
                mesh = well.get_vtk_polydata(z_exag)
                
                if mesh is not None and mesh.n_points > 0:
                    # A.1 Tubo
                    if "Facies_Real" in mesh.point_data and self.pv_cmap is not None:
                        actor_tube = self.plotter.add_mesh(
                            mesh,
                            scalars="Facies_Real",
                            cmap=self.pv_cmap,
                            clim=self.clim,
                            line_width=3,
                            name=f"well_{well_name}",
                            reset_camera=False,
                            show_scalar_bar=False
                        )
                    else:
                        actor_tube = self.plotter.add_mesh(
                            mesh,
                            color="black",
                            line_width=3,
                            name=f"well_{well_name}",
                            reset_camera=False,
                        )
                    current_well_actors.append(actor_tube)

                    # --- A.2 RÓTULO DO NOME DO POÇO ---
                    # Pega o primeiro ponto (topo do poço)
                    top_point = mesh.points[0]
                    
                    actor_name = self.plotter.add_point_labels(
                        [top_point],           
                        [well_name],           
                        font_size=10,          
                        text_color="black",
                        shape="rounded_rect",
                        shape_color="white",
                        shape_opacity=0.3,     
                        show_points=False,     
                        reset_camera=False,
                        always_visible=True,   # <--- CORREÇÃO AQUI: Garante que não pisca/some
                        name=f"name_{well_name}"
                    )
                    current_well_actors.append(actor_name)

                # --- B) MARCADORES (Se houver) ---
                m_list = markers_db.get(well_name, [])
                if m_list:
                    m_mesh, m_labels = well.get_markers_mesh(m_list, z_exag)
                    
                    if m_mesh is not None and m_mesh.n_points > 0:
                        # B.1 Esferas dos Marcadores
                        actor_markers = self.plotter.add_mesh(
                            m_mesh,
                            color="red",
                            render_points_as_spheres=False,
                            reset_camera=False,
                            name=f"markers_{well_name}"
                        )
                        current_well_actors.append(actor_markers)

                        # B.2 Texto dos Marcadores
                        # Nota: Se quiser que os textos dos marcadores também nunca sumam
                        # (mesmo dentro da terra), adicione always_visible=True aqui também.
                        # Por padrão, deixei sem para dar noção de profundidade.
                        actor_labels = self.plotter.add_point_labels(
                            m_mesh.points,
                            m_labels,
                            font_size=12,
                            point_color="red",
                            text_color="black",
                            show_points=False,
                            reset_camera=False,
                            shape_opacity=0.5,
                            # always_visible=True, # Descomente se quiser ver labels através do grid
                            name=f"labels_{well_name}"
                        )
                        current_well_actors.append(actor_labels)

                self._well_actors[well_name] = current_well_actors

            except Exception as e:
                print(f"[WARN] Falha ao desenhar poço {well_name}: {e}")

        try:
            self.plotter.render()
        except Exception:
            pass


    def _pick_reference_xy_for_well_report(self, well, markers):
        """
        Escolhe (X,Y) de referência do poço para comparar com a coluna do grid.
        Preferência:
        1) ponto no meio do intervalo [top_marker, base_marker] (em DEPT/md)
        2) primeiro ponto do well.data
        """
        import numpy as np

        if well is None or well.data is None or well.data.empty:
            return None

        df = well.data

        # tenta usar marcador (md) se existir
        dept_mid = None
        if markers:
            mds = sorted([m.get("md") for m in markers if m.get("md") is not None])
            if len(mds) >= 2:
                dept_mid = 0.5 * (float(mds[0]) + float(mds[-1]))

        if dept_mid is not None and "DEPT" in df.columns:
            dept = df["DEPT"].to_numpy(dtype=float)
            i = int(np.argmin(np.abs(dept - dept_mid)))
            x = float(df.iloc[i]["X"])
            y = float(df.iloc[i]["Y"])
            return x, y

        # fallback: primeiro ponto
        try:
            x = float(df.iloc[0]["X"])
            y = float(df.iloc[0]["Y"])
            return x, y
        except Exception:
            return None


    def _get_ij_from_xy(self, grid, x, y):
        """
        Retorna (i0, j0) da célula mais próxima ao ponto (x,y), usando i_index/j_index.
        """
        import numpy as np
        import pyvista as pv
        from visualize import prepare_grid_indices

        if grid is None:
            return None

        g = grid
        prepare_grid_indices(g)
        if "i_index" not in g.cell_data or "j_index" not in g.cell_data:
            return None

        i_idx = np.asarray(g.cell_data["i_index"]).astype(int)
        j_idx = np.asarray(g.cell_data["j_index"]).astype(int)

        try:
            # z médio só pra achar a célula mais próxima (não importa muito)
            zmid = float(np.mean(g.bounds[4:6]))
            cid0 = int(g.find_closest_cell((float(x), float(y), zmid)))
        except Exception:
            # fallback com probe
            try:
                pt = pv.PolyData(np.array([[float(x), float(y), float(np.mean(g.bounds[4:6]))]]))
                samp = pt.sample(g)
                if "vtkOriginalCellIds" in samp.point_data:
                    cid0 = int(np.asarray(samp.point_data["vtkOriginalCellIds"])[0])
                else:
                    cid0 = 0
            except Exception:
                return None

        return int(i_idx[cid0]), int(j_idx[cid0])


    def _column_profile_from_grid_ij(self, grid, i0, j0):
        """
        Retorna (depth_out, fac_out, ttot_active) para a coluna exata (i0,j0).
        Mesma lógica do seu método atual: ordena topo->base, corta última célula,
        usa StratigraphicThickness (ou fallback) e Facies.
        """
        import numpy as np
        from visualize import prepare_grid_indices

        if grid is None:
            return np.array([]), np.array([]), 0.0

        g = grid
        prepare_grid_indices(g)

        if "i_index" not in g.cell_data or "j_index" not in g.cell_data:
            return np.array([]), np.array([]), 0.0

        i_idx = np.asarray(g.cell_data["i_index"]).astype(int)
        j_idx = np.asarray(g.cell_data["j_index"]).astype(int)

        # Facies
        if "Facies" in g.cell_data:
            fac = np.asarray(g.cell_data["Facies"]).astype(int)
        elif "facies" in g.cell_data:
            fac = np.asarray(g.cell_data["facies"]).astype(int)
        else:
            return np.array([]), np.array([]), 0.0

        # thickness
        if "StratigraphicThickness" in g.cell_data:
            th = np.asarray(g.cell_data["StratigraphicThickness"]).astype(float)
        elif "cell_thickness" in g.cell_data:
            th = np.asarray(g.cell_data["cell_thickness"]).astype(float)
        else:
            th = np.zeros_like(fac, dtype=float)

        ids = np.where((i_idx == int(i0)) & (j_idx == int(j0)))[0]
        if ids.size == 0:
            return np.array([]), np.array([]), 0.0

        # topo -> base pelo Z do centro
        zc = g.cell_centers().points[:, 2].astype(float)
        ids = ids[np.argsort(zc[ids])[::-1]]

        # corta última célula (camada extra)
        if ids.size >= 2:
            ids = ids[:-1]

        depth_out = []
        fac_out = []
        cum = 0.0

        for cid in ids:
            t = float(th[cid]) if np.isfinite(th[cid]) else 0.0
            f = int(fac[cid])
            depth_out.extend([cum, cum + t])
            fac_out.extend([f, f])
            cum += t

        depth_out = np.asarray(depth_out, dtype=float)
        fac_out = np.asarray(fac_out, dtype=int)

        mask_active = (fac[ids] != 0) & np.isfinite(th[ids])
        ttot_active = float(np.sum(th[ids][mask_active])) if ids.size else 0.0

        return depth_out, fac_out, ttot_active


    def _column_profile_from_grid(self, grid, x, y, *, i0=None, j0=None, return_ij=False):
        """
        Retorna um perfil vertical topo->base na coluna (i,j) do grid.

        Se i0/j0 forem None:
        - escolhe a coluna (i,j) mais próxima do ponto (x,y).

        Se i0/j0 forem fornecidos:
        - usa exatamente essa coluna (i0,j0) (útil para janelas 3x3 etc.)

        Saídas:
        depth_profile: array (m), começando em 0 no topo do grid
        fac_profile:   array (mesmo comprimento, facies em degraus)
        ttot_active:   espessura total ativa (exclui facies == 0)

        Se return_ij=True:
        retorna também (i0, j0).
        """
        import numpy as np
        import pyvista as pv
        from visualize import prepare_grid_indices

        if grid is None:
            if return_ij:
                return np.array([]), np.array([]), 0.0, None, None
            return np.array([]), np.array([]), 0.0

        g = grid.copy()

        # garante índices estruturais (i_index/j_index/k_index)
        try:
            prepare_grid_indices(g)
        except Exception:
            pass

        # facies
        fac = g.cell_data.get("Facies", None)
        if fac is None:
            fac = np.zeros(g.n_cells, dtype=int)
        else:
            fac = np.asarray(fac).astype(int)

        # thickness: prioridade StratigraphicThickness
        th = None
        if "StratigraphicThickness" in g.cell_data:
            th = np.asarray(g.cell_data["StratigraphicThickness"], dtype=float)
        elif "cell_thickness" in g.cell_data:
            th = np.asarray(g.cell_data["cell_thickness"], dtype=float)

        if th is None or len(th) != g.n_cells:
            th = np.zeros(g.n_cells, dtype=float)

        # precisa de i/j para selecionar coluna
        i_idx = g.cell_data.get("i_index", None)
        j_idx = g.cell_data.get("j_index", None)
        if i_idx is None or j_idx is None:
            if return_ij:
                return np.array([]), np.array([]), 0.0, None, None
            return np.array([]), np.array([]), 0.0

        i_idx = np.asarray(i_idx).astype(int)
        j_idx = np.asarray(j_idx).astype(int)

        # se i0/j0 não vierem, acha célula mais próxima em XY
        if i0 is None or j0 is None:
            b = g.bounds
            z_mid = 0.5 * (float(b[4]) + float(b[5]))
            try:
                cid0 = int(g.find_closest_cell((float(x), float(y), float(z_mid))))
            except Exception:
                try:
                    p = pv.PolyData(np.array([[float(x), float(y), float(z_mid)]]))
                    samp = p.sample(g, tolerance=1e9)
                    if "vtkOriginalCellIds" in samp.point_data:
                        cid0 = int(np.asarray(samp.point_data["vtkOriginalCellIds"])[0])
                    else:
                        cid0 = 0
                except Exception:
                    if return_ij:
                        return np.array([]), np.array([]), 0.0, None, None
                    return np.array([]), np.array([]), 0.0

            i0 = int(i_idx[cid0])
            j0 = int(j_idx[cid0])
        else:
            i0 = int(i0)
            j0 = int(j0)

        # pega todos os cells da coluna (i0,j0)
        ids = np.where((i_idx == i0) & (j_idx == j0))[0]
        if ids.size == 0:
            if return_ij:
                return np.array([]), np.array([]), 0.0, i0, j0
            return np.array([]), np.array([]), 0.0

        # ordena topo -> base pelo Z do centro da célula (robusto contra flip)
        zc = g.cell_centers().points[:, 2].astype(float)
        ids = ids[np.argsort(zc[ids])[::-1]]  # topo primeiro

        # REMOVE SEMPRE A ÚLTIMA CÉLULA (a mais profunda) — mantém sua regra atual
        if ids.size >= 2:
            ids = ids[:-1]

        # monta perfil em degraus (0 no topo)
        depth_out = []
        fac_out = []
        cum = 0.0

        for cid in ids:
            t = float(th[cid]) if np.isfinite(th[cid]) else 0.0
            f = int(fac[cid])
            depth_out.extend([cum, cum + t])
            fac_out.extend([f, f])
            cum += t

        depth_out = np.asarray(depth_out, dtype=float)
        fac_out = np.asarray(fac_out, dtype=int)

        # espessura total ativa (exclui facies 0)
        mask_active = (fac[ids] != 0) & np.isfinite(th[ids])
        ttot_active = float(np.sum(th[ids][mask_active])) if ids.size else 0.0

        if return_ij:
            return depth_out, fac_out, ttot_active, i0, j0
        return depth_out, fac_out, ttot_active



    def show_well_comparison_report(self, well_name, model_key="base"):
        """
        Relatório BASE vs SIM vs REAL.
        Calcula o 'Melhor da Janela' baseado na seleção da Ribbon e passa para o relatório.
        """
        import numpy as np
        from PyQt5 import QtWidgets, QtCore

        well = self.wells.get(well_name)
        if not well or well.data is None or well.data.empty:
            return

        from load_data import grid as base_grid
        if base_grid is None:
            QtWidgets.QMessageBox.warning(self, "Aviso", "Grid BASE não carregado.")
            return

        # --- resolve grid SIM (modelo selecionado) ---
        if model_key == "base":
            grid_sim_source = base_grid
            sim_model_name = self.models.get("base", {}).get("name", "Base")
        else:
            model_data = self.models.get(model_key, {})
            grid_sim_source = model_data.get("grid", None)
            sim_model_name = model_data.get("name", str(model_key))

            if grid_sim_source is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Aviso",
                    f"Modelo '{sim_model_name}' não tem grid carregado.\n"
                    "Carregue o modelo adicional antes de abrir o relatório."
                )
                return

        # --- marcadores e REAL ---
        key = str(well_name).strip()
        markers = self.markers_db.get(key, [])

        full_depth = well.data["DEPT"].to_numpy(dtype=float) if "DEPT" in well.data.columns else None
        if full_depth is None or full_depth.size == 0:
            QtWidgets.QMessageBox.warning(self, "Aviso", "Poço sem coluna DEPT para relatório.")
            return

        col_real = None
        if "fac" in well.data.columns:
            col_real = "fac"
        elif "lito_upscaled" in well.data.columns:
            col_real = "lito_upscaled"

        full_real = well.data[col_real].to_numpy(dtype=float) if col_real is not None else np.zeros_like(full_depth, dtype=float)

        real_depth0 = full_depth
        real_facies0 = full_real

        if markers:
            mds = sorted([m.get("md") for m in markers if m.get("md") is not None])
            if len(mds) >= 2:
                top_md, base_md = float(mds[0]), float(mds[-1])
                dmin, dmax = float(full_depth.min()), float(full_depth.max())
                if (top_md <= dmax + 1e-6) and (base_md >= dmin - 1e-6) and (base_md > top_md):
                    mask_r = (full_depth >= top_md) & (full_depth <= base_md)
                    if np.any(mask_r):
                        real_depth0 = full_depth[mask_r]
                        real_facies0 = full_real[mask_r]

        # --- BASE e SIM por coluna (i,j) ---
        xy = self._pick_reference_xy_for_well_report(well, markers)
        if xy is None:
            QtWidgets.QMessageBox.warning(self, "Aviso", "Não consegui obter (X,Y) do poço para comparação.")
            return

        xref, yref = xy
        
        # Base (sempre 1x1 no local)
        base_depth, base_facies, _ = self._column_profile_from_grid(base_grid, xref, yref)
        
        # --- NOVO: Pega o tamanho da janela da Ribbon (View -> Inspeção) ---
        try:
            txt = self.cmb_debug_win.currentText()
            w_size = int(txt.split("x")[0])
        except:
            w_size = 1

        # 1. Simulado Original (1x1 no local exato - usado na Correlação Padrão)
        sim_depth, sim_facies, _, i_orig, j_orig, _ = self._best_profile_score_in_window(
            grid_sim_source,
            xref, yref,
            real_depth=real_depth0,
            real_fac=np.where(np.isfinite(real_facies0), real_facies0, 0.0).astype(int),
            window_size=1, # Força 1x1
            n_bins=200,
            w_strat=0.7,
            w_thick=0.3,
            ignore_real_zeros=True,
            use_kappa=True,
        )

        # 2. Simulado Melhor (Na Janela selecionada - usado no Ranking Detail)
        best_depth, best_facies, _, i_best, j_best, fit_best = self._best_profile_score_in_window(
            grid_sim_source,
            xref, yref,
            real_depth=real_depth0,
            real_fac=np.where(np.isfinite(real_facies0), real_facies0, 0.0).astype(int),
            window_size=w_size, # Usa a janela da UI
            n_bins=200,
            w_strat=0.7,
            w_thick=0.3,
            ignore_real_zeros=True,
            use_kappa=True,
        )

        # REAL: não deixar NaN virar lixo
        real_depth = real_depth0
        real_facies = np.where(np.isfinite(real_facies0), real_facies0, 0.0).astype(int)

        # --- Cria dialog ---
        report_dialog = self._open_matplotlib_report(
            well_name=well_name,
            sim_model_name=sim_model_name,
            real_depth=real_depth, real_fac=real_facies,
            base_depth=base_depth, base_fac=base_facies,
            sim_depth=sim_depth, sim_fac=sim_facies,
            best_depth=best_depth, best_fac=best_facies, # Dados do melhor
            window_size_str=f"{w_size}x{w_size}"         # Info visual
        )

        report_dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        report_dialog.show()

        self.open_reports.append(report_dialog)

        def _cleanup():
            try:
                self.open_reports = [d for d in self.open_reports if d is not report_dialog]
            except Exception:
                pass

        report_dialog.destroyed.connect(_cleanup)


    def setup_comparison_3d_view(self, container):
        """Prepara o container para receber o grid dinâmico."""
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Placeholder inicial (opcional)
        label = QtWidgets.QLabel("Selecione os modelos na árvore para comparar.")
        label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(label)

    def setup_toolbar_controls(self):
        """
        Cria o Ribbon simplificado com a nova estrutura solicitada.
        """
        # Remove toolbar antiga
        for tb in self.findChildren(QtWidgets.QToolBar):
            self.removeToolBar(tb)

        # ---------- helpers ----------
        def make_tool_btn(text, icon, *, checkable=False):
            btn = QtWidgets.QToolButton()
            btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
            btn.setIcon(icon)
            btn.setIconSize(QtCore.QSize(28, 28))
            btn.setText(text)
            btn.setAutoRaise(True)
            btn.setCheckable(checkable)
            btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            return btn

        def make_group(title):
            frame = QtWidgets.QFrame()
            frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
            frame.setFrameShadow(QtWidgets.QFrame.Plain)
            v = QtWidgets.QVBoxLayout(frame)
            v.setContentsMargins(8, 6, 8, 6)
            v.setSpacing(4)
            h = QtWidgets.QHBoxLayout()
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            v.addLayout(h)
            lbl = QtWidgets.QLabel(title)
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            f = lbl.font(); f.setBold(False); lbl.setFont(f)
            lbl.setStyleSheet("color: rgba(0,0,0,160);")
            v.addWidget(lbl)
            return frame, h

        def make_tab():
            w = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(w)
            lay.setContentsMargins(8, 6, 8, 6)
            lay.setSpacing(10)
            return w, lay

        # ---------- widget principal ----------
        self.ribbon_tabs = QtWidgets.QTabWidget()
        self.ribbon_tabs.setDocumentMode(True)
        self.ribbon_tabs.setMovable(False)
        self.ribbon_tabs.setUsesScrollButtons(True)
        self.ribbon = self.ribbon_tabs

        self.ribbon_container = QtWidgets.QWidget()
        vroot = QtWidgets.QVBoxLayout(self.ribbon_container)
        vroot.setContentsMargins(0, 0, 0, 0)
        vroot.setSpacing(0)
        vroot.addWidget(self.ribbon_tabs)
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        sep.setFixedHeight(1)
        vroot.addWidget(sep)
        self.setMenuWidget(self.ribbon_container)

        # ==================== ABA HOME ====================
        tab_home, home_lay = make_tab()

        # Dados
        g_dados, g_dados_row = make_group("Dados")
        btn_modelo = make_tool_btn("Modelo", self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        btn_modelo.clicked.connect(self.open_compare_dialog)
        btn_pocos = make_tool_btn("Poços", self.style().standardIcon(QtWidgets.QStyle.SP_DirIcon))
        btn_pocos.clicked.connect(self.load_well_dialog)
        g_dados_row.addWidget(btn_modelo)
        g_dados_row.addWidget(btn_pocos)

        # Perspectiva
        g_persp, g_persp_row = make_group("Perspectiva")
        self.act_persp_viz.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon))
        self.act_persp_comp.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView))
        
        btn_viz = QtWidgets.QToolButton(); btn_viz.setDefaultAction(self.act_persp_viz)
        btn_viz.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon); btn_viz.setIconSize(QtCore.QSize(28, 28)); btn_viz.setAutoRaise(True)
        
        btn_comp = QtWidgets.QToolButton(); btn_comp.setDefaultAction(self.act_persp_comp)
        btn_comp.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon); btn_comp.setIconSize(QtCore.QSize(28, 28)); btn_comp.setAutoRaise(True)
        
        g_persp_row.addWidget(btn_viz)
        g_persp_row.addWidget(btn_comp)

        # Ferramentas
        g_tools, g_tools_row = make_group("Ferramentas")
        btn_snap = make_tool_btn("Snapshot", self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        btn_snap.clicked.connect(self.take_snapshot)
        g_tools_row.addWidget(btn_snap)

        home_lay.addWidget(g_dados); home_lay.addWidget(g_persp); home_lay.addWidget(g_tools); home_lay.addStretch(1)
        self.ribbon_tabs.addTab(tab_home, "Home")

        # ==================== ABA VIEW ====================
        tab_view, view_lay = make_tab()

        # Vista 
        g_vista, g_vista_row = make_group("Vista")
        
        ico3d = self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon)
        ico2d = self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogContentsView)
        icomet = self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogInfoView)
        icorank = self.style().standardIcon(QtWidgets.QStyle.SP_ArrowUp) # Ícone para Ranking

        self.act_view_3d = QtWidgets.QAction(ico3d, "3D", self); self.act_view_3d.setCheckable(True)
        self.act_view_2d = QtWidgets.QAction(ico2d, "Mapas 2D", self); self.act_view_2d.setCheckable(True)
        self.act_view_metrics = QtWidgets.QAction(icomet, "Métricas", self); self.act_view_metrics.setCheckable(True)
        self.act_view_ranking = QtWidgets.QAction(icorank, "Ranking", self); self.act_view_ranking.setCheckable(True)

        grp = QtWidgets.QActionGroup(self); grp.setExclusive(True)
        grp.addAction(self.act_view_3d); grp.addAction(self.act_view_2d); grp.addAction(self.act_view_metrics); grp.addAction(self.act_view_ranking)
        self.act_view_3d.setChecked(True)

        # Conecta aos métodos unificados
        self.act_view_3d.triggered.connect(self.show_main_3d_view)
        self.act_view_2d.triggered.connect(self.show_map2d_view)
        self.act_view_metrics.triggered.connect(self.show_metrics_view)
        self.act_view_ranking.triggered.connect(self.show_ranking_view) # Novo slot

        b3d = QtWidgets.QToolButton(); b3d.setDefaultAction(self.act_view_3d)
        b3d.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon); b3d.setIconSize(QtCore.QSize(28, 28)); b3d.setAutoRaise(True)
        
        b2d = QtWidgets.QToolButton(); b2d.setDefaultAction(self.act_view_2d)
        b2d.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon); b2d.setIconSize(QtCore.QSize(28, 28)); b2d.setAutoRaise(True)
        
        bmet = QtWidgets.QToolButton(); bmet.setDefaultAction(self.act_view_metrics)
        bmet.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon); bmet.setIconSize(QtCore.QSize(28, 28)); bmet.setAutoRaise(True)

        brank = QtWidgets.QToolButton(); brank.setDefaultAction(self.act_view_ranking)
        brank.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon); brank.setIconSize(QtCore.QSize(28, 28)); brank.setAutoRaise(True)

        g_vista_row.addWidget(b3d); g_vista_row.addWidget(b2d); g_vista_row.addWidget(bmet); g_vista_row.addWidget(brank)

        # Modo
        g_modo, g_modo_row = make_group("Modo")
        self.btn_mode = QtWidgets.QToolButton(self)
        self.btn_mode.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.btn_mode.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.btn_mode.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogListView))
        self.btn_mode.setIconSize(QtCore.QSize(28, 28))
        self.btn_mode.setAutoRaise(True)
        
        menu_mode = QtWidgets.QMenu(self.btn_mode)
        
        # --- CORREÇÃO: Conecta o sinal no MENU, não no botão ---
        menu_mode.aboutToShow.connect(self.populate_mode_menu)

        # Lista inicial (idêntica ao seu código original para garantir que inicie correto)
        modes = [
            ("Fácies", "facies"), 
            ("Reservatório", "reservoir"), 
            ("Clusters", "clusters"), 
            ("Maior Cluster", "largest"), 
            ("Espessura Local", "thickness_local"),
            ("Entropia (Incerteza)", "entropy")
        ]
        
        for text, data in modes:
            action = menu_mode.addAction(text)
            action.triggered.connect(lambda ch, t=text, d=data: self._update_mode_btn(t, d))
        self.btn_mode.setMenu(menu_mode)
        self._update_mode_btn("Fácies", "facies")
        g_modo_row.addWidget(self.btn_mode)

        # Espessura
        g_esp, g_esp_row = make_group("Espessura")
        self.btn_thick = QtWidgets.QToolButton(self)
        self.btn_thick.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.btn_thick.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.btn_thick.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView))
        self.btn_thick.setIconSize(QtCore.QSize(28, 28))
        self.btn_thick.setAutoRaise(True)
        
        menu_thick = QtWidgets.QMenu(self.btn_thick)
        thickness_opts = ["Espessura", "NTG coluna", "NTG envelope", "Maior pacote", "Nº pacotes", "ICV", "Qv", "Qv absoluto"]
        for label in thickness_opts:
            action = menu_thick.addAction(label)
            action.triggered.connect(lambda ch, l=label: self._update_thick_btn(l))
        self.btn_thick.setMenu(menu_thick)
        self._update_thick_btn("Espessura")
        g_esp_row.addWidget(self.btn_thick)

        # Inspeção de Poços
        g_wells, g_wells_row = make_group("Inspeção")
        
        # Combo Janela
        self.cmb_debug_win = QtWidgets.QComboBox()
        self.cmb_debug_win.addItems(["1x1", "3x3", "5x5", "7x7", "9x9"])
        self.cmb_debug_win.setCurrentIndex(1) # Default 3x3
        self.cmb_debug_win.setToolTip("Tamanho da Janela de Busca")
        self.cmb_debug_win.setFixedWidth(60)
        
        # Botão Toggle
        self.btn_debug_all = make_tool_btn("Destacar\nTodos", self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogContentsView), checkable=True)
        self.btn_debug_all.clicked.connect(self.toggle_global_well_debug)
        
        # Conecta mudança do combo para atualizar em tempo real:
        # 1. Se "Destacar Todos" estiver ligado (3D).
        # 2. Se a visualização atual for "Ranking" (Recalcula tabela).
        self.cmb_debug_win.currentIndexChanged.connect(self._on_global_window_size_changed)

        # Layout vertical para o combo
        v_box_combo = QtWidgets.QVBoxLayout()
        v_box_combo.setSpacing(0)
        v_box_combo.setContentsMargins(0, 0, 0, 0)
        v_box_combo.addWidget(QtWidgets.QLabel("Janela:"))
        v_box_combo.addWidget(self.cmb_debug_win)
        
        g_wells_row.addLayout(v_box_combo)
        g_wells_row.addWidget(self.btn_debug_all)

        # --- GRUPO RELATÓRIOS (Movido para View) ---
        g_rep, g_rep_row = make_group("Relatórios")
        
        btn_rep_open = make_tool_btn("Abrir\nRelatório", self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        btn_rep_open.clicked.connect(self.open_reports_dialog)
        
        btn_rep_selected = make_tool_btn("Poços\nSelecionados", self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogContentsView))
        btn_rep_selected.clicked.connect(self.open_selected_well_reports)
        
        g_rep_row.addWidget(btn_rep_open)
        g_rep_row.addWidget(btn_rep_selected)
        
        # Janelas
        g_windows, g_windows_row = make_group("Janelas")
        self.btn_toggle_explorer = make_tool_btn("Explorer", self.style().standardIcon(QtWidgets.QStyle.SP_DirHomeIcon), checkable=True)
        self.btn_toggle_props = make_tool_btn("Inspector", self.style().standardIcon(QtWidgets.QStyle.SP_DesktopIcon), checkable=True)
        self.btn_toggle_explorer.setEnabled(False); self.btn_toggle_props.setEnabled(False)
        g_windows_row.addWidget(self.btn_toggle_explorer); g_windows_row.addWidget(self.btn_toggle_props)

        view_lay.addWidget(g_vista); view_lay.addWidget(g_modo); view_lay.addWidget(g_esp)
        view_lay.addWidget(g_wells); view_lay.addWidget(g_rep); view_lay.addWidget(g_windows) # Inserido g_rep
        view_lay.addStretch(1)
        
        self.ribbon_tabs.addTab(tab_view, "View")

    def populate_mode_menu(self):
        """Reconstrói o menu de Modos/Propriedades baseado no grid atual."""
        menu = self.btn_mode.menu()
        menu.clear()
        
        # 1. Modos Padrão (Manuais)
        modes_std = [
            ("Fácies (Discreto)", "facies"), 
            ("Reservatório (Binário)", "reservoir"), 
            ("Clusters (Conectividade)", "clusters"), 
            ("Maior Cluster", "largest"),
            ("Espessura Local", "thickness_local"),
            ("Entropia (Incerteza)", "entropy")
        ]
        
        menu.addSection("Análise Estrutural")
        for text, data in modes_std:
            action = menu.addAction(text)
            action.triggered.connect(lambda ch, t=text, d=data: self._update_mode_btn(t, d))

        # 2. Propriedades do Grid (Dinâmico)
        # Pega o grid ativo (Base ou de um modelo comparado)
        grid = self.state.get("current_grid_source")
        if grid is None and "base" in self.models:
            grid = self.models["base"].get("grid")
            
        if grid:
            # Lista de nomes EXATOS para ignorar (já tratados acima ou internos)
            exact_ignore = {
                "vtkOriginalCellIds", "vtkOriginalPointIds", 
                "Facies", "facies", "Entropy", "Texture Coordinates", 
                "StratigraphicThickness", "cell_thickness",
                "Reservoir", "reservoir", "Clusters", "clusters", 
                "LargestCluster", "Volume", "NTG_local"
            }
            
            found_any = False
            
            # Ordena para ficar bonito no menu
            for name in sorted(grid.cell_data.keys()):
                # FILTRO 1: Ignora nomes exatos da lista negra
                if name in exact_ignore: continue
                
                # FILTRO 2: Ignora variáveis internas ou de controle
                # (Indices i,j,k, métricas verticais caculadas, ou vetores fantasma)
                if name.endswith("_index"): continue  # Remove i_index, k_index
                if name.startswith("vert_"): continue # Remove métricas verticais internas
                if "Ghost" in name: continue
                
                # Se passou pelos filtros, é uma propriedade válida (PORO, PERM, etc)
                if not found_any:
                    menu.addSection("Propriedades do Grid")
                    found_any = True

                # Cria ação para visualizar essa propriedade
                action = menu.addAction(f"{name}")
                action.triggered.connect(lambda ch, n=name: self.change_scalar_view(n))
    
    def change_scalar_view(self, scalar_name):
        """Visualiza uma propriedade escalar arbitrária (PORO, PERM, etc)."""
        import numpy as np
        
        self.btn_mode.setText(f"Prop:\n{scalar_name}")
        
        grid = self.state.get("current_grid_source")
        if grid is None:
            if "base" in self.models:
                grid = self.models["base"].get("grid")
        
        if grid is None or scalar_name not in grid.cell_data:
            return

        arr = grid.cell_data[scalar_name]
        
        # Configura como um Preset de Espessura (Hack visual para usar o renderizador escalar)
        presets = self.state.get("thickness_presets", {})
        presets[scalar_name] = (scalar_name, f"Propriedade: {scalar_name}")
        self.state["thickness_presets"] = presets
        self.state["thickness_mode"] = scalar_name
        
        # Escala automática
        valid_arr = arr[np.isfinite(arr)]
        if valid_arr.size > 0:
            vmin, vmax = float(np.min(valid_arr)), float(np.max(valid_arr))
        else:
            vmin, vmax = 0.0, 1.0
            
        self.state["thickness_clim"] = (vmin, vmax)
        self.state["thickness_cmap"] = "viridis" 
        
        # Ativa modo
        self.state["mode"] = "thickness_local"
        
        # Refresh
        refresh = self.state.get("refresh")
        if callable(refresh): refresh()
        
        if hasattr(self, "update_2d_map"): self.update_2d_map()
    
    def show_ranking_view(self):
        """Alterna para a visão de Ranking (Seja no modo Visualização ou Comparação)."""
        is_comparison = (self.central_stack.currentIndex() == 1)
        
        if is_comparison:
            # Ranking não disponível no modo comparação lado a lado por enquanto
            QtWidgets.QMessageBox.information(self, "Modo Comparação", "Volte para o modo Visualização para ver o Ranking detalhado.")
            return

        # Na visualização simples, o Ranking é o índice 3
        if hasattr(self, "viz_container"):
            self.viz_container.setCurrentIndex(3)
            # Atualiza o conteúdo do ranking
            self.update_ranking_view_content()

    def _on_global_window_size_changed(self):
        """Callback quando o tamanho da janela global (Ribbon) é alterado."""
        # 1. Se estiver visualizando Ranking, recalcula a tabela
        if hasattr(self, "viz_container") and self.viz_container.currentIndex() == 3:
            self.update_ranking_view_content()
        
        # 2. Se a visualização de debug 3D estiver ativa, atualiza os atores
        if self.btn_debug_all.isChecked():
            self.toggle_global_well_debug()

    def update_ranking_view_content(self):
        """
        Calcula o ranking considerando MODELOS e POÇOS selecionados.
        Preenche a tabela incluindo a coluna 'Study'.
        """
        # 1. Pega tamanho da janela
        try:
            txt = self.cmb_debug_win.currentText()
            ws = int(txt.split("x")[0])
        except: ws = 1
        self.well_rank_window_size = ws
        
        # 2. Pega MODELOS selecionados
        checked_data = self.get_checked_models()
        selected_keys = [k for k, name in checked_data]

        # 3. Pega POÇOS selecionados
        selected_wells = self.get_checked_wells()

        # Se faltar modelo ou poço, limpa a tabela e sai
        if not selected_keys or not selected_wells:
            self.tbl_models.setRowCount(0)
            self.tbl_wells.setRowCount(0)
            return

        # 4. Calcula Ranking com os filtros aplicados
        ranking = self.evaluate_models_against_wells(
            model_keys=selected_keys,
            well_names=selected_wells,
            window_size=ws,
            n_bins=200,
            w_strat=0.7,
            w_thick=0.3,
            ignore_real_zeros=True,
            use_kappa=True,
        )

        # 5. Atualiza Tabela
        self._current_ranking_data = ranking
        self.tbl_models.setRowCount(0)
        
        if not ranking: return

        for i, r in enumerate(ranking, start=1):
            row = self.tbl_models.rowCount()
            self.tbl_models.insertRow(row)
            
            # Chave do modelo
            m_key = r.get("model_key")

            # --- Col 0: Rank ---
            it_rank = QtWidgets.QTableWidgetItem(f"{i:02d}")
            it_rank.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            it_rank.setData(QtCore.Qt.UserRole, m_key)
            self.tbl_models.setItem(row, 0, it_rank)

            # --- Col 1: Study (NOVO) ---
            study_name = "Geral"
            if m_key in self.models:
                study_name = self.models[m_key].get("study", "Geral")
            
            it_study = QtWidgets.QTableWidgetItem(str(study_name))
            if m_key == "base": 
                it_study.setBackground(QtGui.QBrush(QtGui.QColor(230, 240, 255)))
            self.tbl_models.setItem(row, 1, it_study)

            # --- Col 2: Modelo ---
            self.tbl_models.setItem(row, 2, QtWidgets.QTableWidgetItem(r.get("model_name", "")))

            # --- Col 3: Score ---
            it_score = QtWidgets.QTableWidgetItem(f"{r.get('score', 0.0):.3f}")
            it_score.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            if r.get("score", 0.0) > 0.5:
                it_score.setFont(QtGui.QFont("Arial", weight=QtGui.QFont.Bold))
            self.tbl_models.setItem(row, 3, it_score)

            # Detalhes médios
            details = r.get("details", {}) or {}
            accs = [float(s.get("strat_acc", 0.0)) for s in details.values()]
            kappas = [float(s.get("strat_kappa_norm", s.get("strat_kappa", 0.0))) for s in details.values()]
            
            mean_acc = sum(accs) / len(accs) if accs else 0.0
            mean_kap = sum(kappas) / len(kappas) if kappas else 0.0

            it_acc = QtWidgets.QTableWidgetItem(f"{mean_acc:.3f}")
            it_acc.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            self.tbl_models.setItem(row, 4, it_acc)

            it_kap = QtWidgets.QTableWidgetItem(f"{mean_kap:.3f}")
            it_kap.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            self.tbl_models.setItem(row, 5, it_kap)

            self.tbl_models.setItem(row, 6, QtWidgets.QTableWidgetItem(str(r.get("n_wells_used", 0))))
        
        self.tbl_models.resizeColumnsToContents()
        
        if self.tbl_models.rowCount() > 0:
            self.tbl_models.selectRow(0)

    def _create_well_debug_actors(self, grid, well_name, best_i, best_j, window_size, z_exag, scale_z):
        """
        Gera a lista de atores (janela sólida com cores manuais e destaque) para um poço.
        """
        import numpy as np
        from config import load_facies_colors

        actors = []
        well = self.wells.get(well_name)
        if well is None: return actors

        # Localiza centro
        wx = float(well.data["X"].mean())
        wy = float(well.data["Y"].mean())
        center_i, center_j = self._get_ij_from_xy(grid, wx, wy)
        if center_i is None: return actors

        i_idx = grid.cell_data.get("i_index")
        
        if i_idx is not None:
            half = window_size // 2
            i_min, i_max = center_i - half, center_i + half
            j_min, j_max = center_j - half, center_j + half

            # --- A. JANELA SÓLIDA ---
            mask_win = (i_idx >= i_min) & (i_idx <= i_max) & \
                       (grid.cell_data["j_index"] >= j_min) & (grid.cell_data["j_index"] <= j_max)
            window_grid = grid.extract_cells(mask_win)

            if window_grid.n_cells > 0:
                pts = window_grid.points.copy()
                if scale_z > 1.0: pts[:, 2] *= z_exag
                window_grid.points = pts

                # PINTURA MANUAL (CORREÇÃO DE CORES)
                if "Facies" in window_grid.cell_data:
                    f_colors = load_facies_colors()
                    facies_vals = window_grid.cell_data["Facies"]
                    n_cells = len(facies_vals)
                    rgba_colors = np.zeros((n_cells, 4), dtype=np.uint8)
                    
                    for i in range(n_cells):
                        f_val = int(facies_vals[i])
                        rgb_norm = f_colors.get(f_val, (0.7, 0.7, 0.7))
                        rgba_colors[i, 0] = int(rgb_norm[0] * 255)
                        rgba_colors[i, 1] = int(rgb_norm[1] * 255)
                        rgba_colors[i, 2] = int(rgb_norm[2] * 255)
                        rgba_colors[i, 3] = 255 # Opacidade total

                    window_grid.cell_data["ManualColors"] = rgba_colors
                    window_grid.set_active_scalars("ManualColors")

                    act_win = self.plotter.add_mesh(
                        window_grid,
                        rgb=True, # Usa a cor direta do array
                        show_edges=True, edge_color="black", line_width=1.0,
                        reset_camera=False, show_scalar_bar=False,
                        name=f"debug_solid_win_{well_name}"
                    )
                    actors.append(act_win)

            # --- B. DESTAQUE (WIRE AMARELO) ---
            if best_i is not None:
                mask_best = (i_idx == best_i) & (grid.cell_data["j_index"] == best_j)
                best_grid = grid.extract_cells(mask_best)

                if best_grid.n_cells > 0:
                    pts_b = best_grid.points.copy()
                    if scale_z > 1.0: pts_b[:, 2] *= z_exag
                    best_grid.points = pts_b

                    outline = best_grid.outline()
                    act_high = self.plotter.add_mesh(
                        outline,
                        color="yellow", style="wireframe", line_width=4,
                        name=f"debug_high_{well_name}", reset_camera=False
                    )
                    actors.append(act_high)

                    # Label
                    top_z = np.max(best_grid.bounds[4:6])
                    top_pt = [best_grid.center[0], best_grid.center[1], top_z]
                    # lbl = self.plotter.add_point_labels(
                    #     [top_pt], [f"{well_name}\nMelhor"],
                    #     font_size=16, text_color="yellow", always_visible=True,
                    #     shape_opacity=0.4, name=f"debug_lbl_{well_name}"
                    # )
                    # actors.append(lbl)
        
        return actors
    
    def toggle_global_well_debug(self):
        """
        Liga/Desliga a visualização de janelas para TODOS os poços do modelo ativo.
        """
        # 1. Limpa tudo primeiro
        if hasattr(self, "_debug_actors"):
            for a in self._debug_actors:
                try: self.plotter.remove_actor(a)
                except: pass
        self._debug_actors = []

        # Se o botão estiver desligado (ou não existir), restaura e sai
        if not getattr(self, "btn_debug_all", None) or not self.btn_debug_all.isChecked():
            main_actor = self.state.get("main_actor")
            if main_actor: main_actor.GetProperty().SetOpacity(1.0)
            self.plotter.render()
            return

        # 2. Prepara Modelo
        model_key = self.state.get("active_model_key")
        if not model_key or model_key not in self.models: return
        
        self.switch_main_view_to_model(model_key)
        if hasattr(self, "viz_container"): self.viz_container.setCurrentIndex(0)

        grid = self.state.get("current_grid_source")
        if grid is None: return

        # Parâmetros de Visualização
        z_exag = float(self.state.get("z_exag", 15.0))
        main_actor = self.state.get("main_actor")
        scale_z = main_actor.GetScale()[2] if main_actor else 1.0

        # Tamanho da Janela do Combo
        try:
            txt = self.cmb_debug_win.currentText()
            w_size = int(txt.split("x")[0])
        except: w_size = 3

        # Efeito Fantasma no Grid
        if main_actor: main_actor.GetProperty().SetOpacity(0.001)

        # 3. Calcula e Desenha para CADA poço
        # Usa evaluate para obter os Best Matches (best_i, best_j)
        results = self.evaluate_models_against_wells(
            model_keys=[model_key], 
            window_size=w_size, 
            ignore_real_zeros=True
        )

        if not results: return

        rec = results[0]
        details = rec.get("details", {})

        for well_name, s in details.items():
            best_i = s.get("best_i")
            best_j = s.get("best_j")
            
            # Chama a auxiliar e acumula os atores
            new_actors = self._create_well_debug_actors(
                grid, well_name, best_i, best_j, w_size, z_exag, scale_z
            )
            self._debug_actors.extend(new_actors)

        self.plotter.render()


    def toggle_comparison_view_type(self):
        """Alterna entre ver o Grid 3D e as Tabelas de Métricas na Comparação."""
        if self.central_stack.currentIndex() == 1:
            if self.btn_view_type.isChecked():
                self.btn_view_type.setText("Voltar ao 3D")
                self.btn_view_type.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowBack))
                self.compare_stack.setCurrentIndex(1) # Página tabelas
            else:
                self.btn_view_type.setText("Métricas")
                self.btn_view_type.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon))
                self.compare_stack.setCurrentIndex(0) # Página 3D

    def _update_mode_btn(self, text, data):
        # Texto do botão (ribbon)
        if hasattr(self, "btn_mode") and self.btn_mode is not None:
            self.btn_mode.setText(f"Modo\n{text}")
            self.btn_mode.setToolTip(f"Modo: {text}")

        # Sempre salva no state
        self.state["mode"] = data

        # Só aplica (render) se o visualize.run já tiver registrado refresh
        if "refresh" in self.state and callable(self.state["refresh"]):
            self.change_mode(data)


    def _update_thick_btn(self, label):
        # Texto do botão (ribbon)
        if hasattr(self, "btn_thick") and self.btn_thick is not None:
            self.btn_thick.setText(f"Espessura\n{label}")
            self.btn_thick.setToolTip(f"Espessura: {label}")

        # Sempre salva no state
        self.state["thickness_mode"] = label

        # Só aplica (render) se o visualize.run já tiver registrado refresh
        if "refresh" in self.state and callable(self.state["refresh"]):
            self.change_thickness_mode(label)


    def show_main_3d_view(self):
        """Alterna para a visualização 3D (Seja no modo Visualização ou Comparação)."""
        is_comparison = (self.central_stack.currentIndex() == 1)
        
        if is_comparison:
            # Na comparação, o 3D é o índice 0
            if hasattr(self, "compare_stack"):
                self.compare_stack.setCurrentIndex(0)
                self.refresh_comparison_active_view()
        else:
            # Na visualização simples, o 3D é o índice 0 do viz_container
            if hasattr(self, "viz_container"):
                self.viz_container.setCurrentIndex(0)
                model_key = self.state.get("active_model_key", "base")
                try: self.switch_main_view_to_model(model_key)
                except: pass


    def show_map2d_view(self):
        """Alterna para Mapas 2D (Seja no modo Visualização ou Comparação)."""
        is_comparison = (self.central_stack.currentIndex() == 1)
        
        if is_comparison:
            # Na comparação, os Mapas 2D são o índice 2 (conforme o update_dynamic_comparison_view)
            if hasattr(self, "compare_stack"):
                self.compare_stack.setCurrentIndex(2)
                self.refresh_comparison_active_view()
        else:
            # Na visualização simples, os Mapas 2D são o índice 1
            if hasattr(self, "viz_container"):
                self.viz_container.setCurrentIndex(1)
                model_key = self.state.get("active_model_key", "base")
                try: 
                    self.switch_main_view_to_model(model_key)
                    self.update_2d_map()
                except: pass


    def show_metrics_view(self):
        """Alterna para Métricas (Seja no modo Visualização ou Comparação)."""
        is_comparison = (self.central_stack.currentIndex() == 1)
        
        if is_comparison:
            # Na comparação, as Métricas são o índice 1 (Tabela Comparativa)
            if hasattr(self, "compare_stack"):
                self.compare_stack.setCurrentIndex(1)
                self.refresh_comparison_active_view()
        else:
            # Na visualização simples, as Métricas são o índice 2
            if hasattr(self, "viz_container"):
                self.viz_container.setCurrentIndex(2)
                model_key = self.state.get("active_model_key", "base")
                try: self.update_metrics_view_content(model_key)
                except: pass

    def _wrap_expanding(self, widget):
            """Helper: força o widget a ocupar toda a área do dock."""
            container = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(container)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)
            lay.addWidget(widget, 1)  # stretch = 1
            container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            return container


    def setup_docks(self, nx, ny, nz):
        # --- DOCK EXPLORER - ESQUERDA ---
        self.dock_explorer = QtWidgets.QDockWidget("Project Explorer", self)
        self.dock_explorer.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.dock_explorer.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFloatable |
            QtWidgets.QDockWidget.DockWidgetClosable
        )

        self.project_tree = QtWidgets.QTreeWidget()
        self.project_tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.project_tree.setHeaderLabel("Hierarquia")
        self.project_tree.itemDoubleClicked.connect(self.on_tree_double_clicked)
        self.project_tree.itemSelectionChanged.connect(self.on_tree_selection_changed)
        self.project_tree.itemChanged.connect(self.on_tree_item_changed)

        self.dock_explorer.setWidget(self.project_tree)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dock_explorer)

        # Modelos (top-level) + Poços (top-level)
        self.add_model_to_tree("base", "Modelo Base")

        self.wells_root_item = QtWidgets.QTreeWidgetItem(self.project_tree, ["Poços"])
        self.wells_root_item.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_DirIcon))
        self.wells_root_item.setData(0, QtCore.Qt.UserRole, "wells_root")
        self.wells_root_item.setExpanded(True)

        # --- DOCK INSPECTOR - DIREITA ---
        self.dock_props = QtWidgets.QDockWidget("Inspector", self)
        self.dock_props.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.dock_props.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFloatable |
            QtWidgets.QDockWidget.DockWidgetClosable
        )

        # Abas do inspector (Geometria / Propriedades / Comparação)
        self.inspector_tabs = QtWidgets.QTabWidget()

        # ----- Geometria -----
        self.page_grid = QtWidgets.QWidget()
        pg_layout = QtWidgets.QVBoxLayout(self.page_grid)
        pg_layout.setContentsMargins(4, 4, 4, 4)
        self.slicer_widget = GridSlicerWidget(nx, ny, nz, self.on_ui_slice_changed)
        pg_layout.addWidget(self.slicer_widget)
        pg_layout.addStretch(1)
        self.inspector_tabs.addTab(self.page_grid, "Geometria")

        # ----- Propriedades -----
        self.page_props = QtWidgets.QWidget()
        pp_layout = QtWidgets.QVBoxLayout(self.page_props)
        pp_layout.setContentsMargins(4, 4, 4, 4)

        self.legend_group = QtWidgets.QGroupBox("Legenda & Filtro")
        lgl = QtWidgets.QVBoxLayout(self.legend_group)
        lgl.setContentsMargins(2, 6, 2, 2)

        # --- NOVO: Botões de Seleção ---
        h_btn_leg = QtWidgets.QHBoxLayout()
        btn_sel_all = QtWidgets.QPushButton("Todos")
        btn_sel_all.setToolTip("Selecionar todas as fácies")
        btn_sel_all.clicked.connect(lambda: self.toggle_all_facies_legend(True))
        
        btn_sel_none = QtWidgets.QPushButton("Nenhum")
        btn_sel_none.setToolTip("Desmarcar todas as fácies")
        btn_sel_none.clicked.connect(lambda: self.toggle_all_facies_legend(False))
        
        h_btn_leg.addWidget(btn_sel_all)
        h_btn_leg.addWidget(btn_sel_none)
        lgl.addLayout(h_btn_leg)
        # -------------------------------

        self.facies_legend_table = QtWidgets.QTableWidget()
        self.facies_legend_table.setColumnCount(4)
        self.facies_legend_table.setHorizontalHeaderLabels(["Cor", "ID", "N", "Res"])
        self.facies_legend_table.verticalHeader().setVisible(False)
        self.facies_legend_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.facies_legend_table.itemChanged.connect(self.on_legend_item_changed)
        lgl.addWidget(self.facies_legend_table)

        self.clusters_legend_table = QtWidgets.QTableWidget()
        self.clusters_legend_table.setColumnCount(3)
        self.clusters_legend_table.setHorizontalHeaderLabels(["Cor", "ID", "Células"])
        self.clusters_legend_table.verticalHeader().setVisible(False)
        self.clusters_legend_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.clusters_legend_table.setVisible(False)
        lgl.addWidget(self.clusters_legend_table)

        self.facies_legend_table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.legend_group.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        pp_layout.addWidget(self.legend_group, 1)

        self.inspector_tabs.addTab(self.page_props, "Propriedades")

        # ----- Comparação -----
        self.page_compare = self.setup_comparison_dock_content()
        self.inspector_tabs.addTab(self.page_compare, "Comparação")
        
        self.inspector_tabs.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # IMPORTANTE: também força as tabelas internas a expandirem
        for tbl in self.inspector_tabs.findChildren(QtWidgets.QTableWidget):
            tbl.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.dock_props.setWidget(self._wrap_expanding(self.inspector_tabs))
        self.dock_props.setMinimumWidth(420)

        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dock_props)

        # Ajuste inicial de larguras
        self.resizeDocks([self.dock_explorer, self.dock_props], [320, 420], QtCore.Qt.Horizontal)

        # Liga botões "Janelas" do ribbon aos docks (reabrir Explorer/Inspector)
        if hasattr(self, "btn_toggle_explorer") and isinstance(self.btn_toggle_explorer, QtWidgets.QToolButton):
            act = self.dock_explorer.toggleViewAction()
            act.setText("Explorer")
            act.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DirHomeIcon))
            self.btn_toggle_explorer.setDefaultAction(act)
            self.btn_toggle_explorer.setEnabled(True)

        if hasattr(self, "btn_toggle_props") and isinstance(self.btn_toggle_props, QtWidgets.QToolButton):
            act = self.dock_props.toggleViewAction()
            act.setText("Inspector")
            act.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DesktopIcon))
            self.btn_toggle_props.setDefaultAction(act)
            self.btn_toggle_props.setEnabled(True)

    def _apply_reservoir_filter_and_refresh(self):
        """Aplica filtro (Reservatório) e atualiza 3D/2D/Métricas sem precisar trocar de vista."""
        model_key = self.state.get("active_model_key", "base")
        if model_key not in self.models:
            model_key = "base"

        rf = self.models[model_key].get("reservoir_facies", set())
        if not isinstance(rf, set):
            rf = set(rf)

        # Atualiza campo Reservoir/Clusters no state do visualize.run
        if "update_reservoir_fields" in self.state:
            try:
                self.state["update_reservoir_fields"](rf)
            except Exception as e:
                print(f"[apply_reservoir_filter] update_reservoir_fields erro: {e}")

        # Refresh do 3D
        refresh = self.state.get("refresh")
        if callable(refresh):
            try:
                refresh()
            except Exception as e:
                print(f"[apply_reservoir_filter] refresh erro: {e}")

        # Atualiza 2D se estiver na aba 2D
        if hasattr(self, "viz_container") and self.viz_container.currentIndex() == 1:
            try:
                self.update_2d_map()
            except Exception as e:
                print(f"[apply_reservoir_filter] update_2d_map erro: {e}")

        # Atualiza texto/tabela se estiver em métricas
        if hasattr(self, "viz_container") and self.viz_container.currentIndex() == 2:
            try:
                self.update_sidebar_metrics_text(model_key)
            except Exception:
                pass
            try:
                # Se você tiver uma função específica pra montar as métricas do centro
                if hasattr(self, "update_metrics_view_content"):
                    self.update_metrics_view_content(model_key)
            except Exception:
                pass

        # Atualiza legenda (N e Res)
        try:
            self.populate_facies_legend()
        except Exception:
            pass


    def add_model_to_tree(self, model_key, model_name, study_name=None):
        """Adiciona o item do modelo na árvore, dentro de um Study se especificado."""
        
        # Cria o item do modelo
        item = QtWidgets.QTreeWidgetItem([model_name])
        item.setData(0, QtCore.Qt.UserRole, "model_root")
        item.setData(0, QtCore.Qt.UserRole + 1, model_key)
        item.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon))

        # Checkbox logic
        is_comparison = hasattr(self, "central_stack") and (self.central_stack.currentIndex() == 1)
        initial_check = QtCore.Qt.Checked if is_comparison else QtCore.Qt.Unchecked
        
        # Habilita o checkbox
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(0, initial_check)
        self._set_item_checkbox_visible(item, True)

        if model_key == "base":
            self.project_tree.insertTopLevelItem(0, item)
            item.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_DriveHDIcon))
        else:
            target_study = study_name if study_name else "Geral"
            parent = self._get_or_create_study_item(target_study)
            parent.addChild(item)
            # Expande a pasta ao adicionar
            parent.setExpanded(True)

        return item

    # --- LÓGICA DE INTERAÇÃO TREE ---

    def on_tree_double_clicked(self, item, col):
        """Duplo clique: abre relatório do poço no modelo ativo, ou abre a view do grid."""
        role = item.data(0, QtCore.Qt.UserRole)
        data = item.data(0, QtCore.Qt.UserRole + 1)

        if role == "well_item":
            well_name = data

            model_key = self.state.get("active_model_key", "base")
            # ✅ sanitize
            if model_key not in self.models:
                model_key = "base"

            self.show_well_comparison_report(well_name, model_key)
            return

        if role == "grid_settings" and data:
            self.switch_main_view_to_model(data)
            self.tabs.setCurrentIndex(0)
            return


    def switch_main_view_to_model(self, model_key):
        """
        Troca o modelo ativo na vista principal SEM reconstruir o plotter (sem run()).
        Mantém câmera/zoom e NÃO redesenha poços automaticamente.
        """
        import numpy as np
        from load_data import grid as base_grid, facies as base_facies

        # --- pega grid/facies do modelo escolhido ---
        if model_key in ("base", "Base", "Modelo Base"):
            source_grid = base_grid
            target_facies = base_facies
            model_key_norm = "base"
        else:
            model = self.models.get(model_key)
            if not model:
                print(f"[WARN] Modelo '{model_key}' não encontrado.")
                return
            source_grid = model.get("grid", None)
            target_facies = model.get("facies", None)
            model_key_norm = model_key

            if source_grid is None:
                # fallback (evita crash)
                source_grid = base_grid
            if target_facies is None:
                target_facies = base_facies

        # --- normaliza facies ---
        try:
            target_facies = np.asarray(target_facies).ravel().astype(np.int32)
        except Exception:
            target_facies = np.asarray(base_facies).ravel().astype(np.int32)

        # --- preserva modo global ---
        desired_mode = self.state.get("mode", "facies")

        # --- atualiza state ---
        self.active_model_key = model_key_norm
        self.state["active_model_key"] = model_key_norm
        self.state["current_grid_source"] = source_grid
        self.state["current_facies"] = target_facies
        self.state["mode"] = desired_mode

        # garante Facies no grid atual (se possível)
        try:
            source_grid.cell_data["Facies"] = target_facies
        except Exception:
            pass

        # --- atualiza campos derivados (reservatório/clusters/thickness) no grid atual ---
        rf_raw = self.state.get("reservoir_facies", set()) or set()
        rf_set = set()
        for x in rf_raw:
            try:
                rf_set.add(int(x))
            except Exception:
                # caso raro de set aninhado
                if isinstance(x, (set, list, tuple, np.ndarray)):
                    for y in x:
                        try:
                            rf_set.add(int(y))
                        except Exception:
                            pass

        upd = self.state.get("update_reservoir_fields")
        if callable(upd):
            try:
                upd(rf_set)
            except Exception as e:
                print("[switch_main_view_to_model] update_reservoir_fields falhou:", e)

        # --- refresh sem resetar câmera (o refresh do visualize não deveria resetar) ---
        refresh = self.state.get("refresh")
        if callable(refresh):
            try:
                refresh()
            except Exception as e:
                print("[switch_main_view_to_model] refresh falhou:", e)

        # --- CORREÇÃO: Atualiza a legenda de Propriedades ---
        # Isso garante que a tabela mostre as contagens do novo modelo imediatamente
        if hasattr(self, "populate_facies_legend"):
            self.populate_facies_legend()

        # --- atualiza UI lateral (sem recriar plotter) ---
        try:
            self.update_sidebar_metrics_text(model_key_norm)
        except Exception:
            pass

        # se estiver na aba métricas/2D, atualiza conteúdo sem trocar de vista
        try:
            if hasattr(self, "viz_container"):
                idx = self.viz_container.currentIndex()
                if idx == 1 and hasattr(self, "update_2d_map"):
                    self.update_2d_map()
                elif idx == 2 and hasattr(self, "update_metrics_view_content"):
                    self.update_metrics_view_content(model_key_norm)
        except Exception:
            pass



    def on_tree_selection_changed(self):
        items = self.project_tree.selectedItems()
        if not items:
            return

        item = items[0]
        role = item.data(0, QtCore.Qt.UserRole)

        # Seleção de modelo
        if role == "model_root":
            model_key = item.data(0, QtCore.Qt.UserRole + 1) or "base"
            self.state["active_model_key"] = model_key

            try:
                self.update_sidebar_metrics_text(model_key)
            except Exception:
                pass

            # Em modo visualização: mostra Propriedades por padrão
            if hasattr(self, "inspector_tabs") and self.central_stack.currentIndex() == 0:
                self.inspector_tabs.setCurrentWidget(self.page_props)

            # Atualiza a vista ativa (3D / 2D / métricas)
            if self.central_stack.currentIndex() == 0:
                idx = self.viz_container.currentIndex() if hasattr(self, "viz_container") else 0

                if idx == 2:
                    self.update_metrics_view_content(model_key)
                else:
                    self.switch_main_view_to_model(model_key)
                    if idx == 1:
                        self.update_2d_map()

            elif self.central_stack.currentIndex() == 1:
                if hasattr(self, "inspector_tabs"):
                    self.inspector_tabs.setCurrentWidget(self.page_compare)

            return

        # Seleção de poço: não muda a vista por padrão
        if role in ("well_item", "wells_root"):
            return


    def _set_item_checkbox_visible(self, item, visible):
        """Define se um item da árvore tem checkbox visível ou não."""
        if visible:
            # Adiciona a permissão de ter checkbox
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            
            # CRÍTICO: Se o item não tiver um estado definido (None), o Qt não desenha o quadrado.
            # Forçamos um estado inicial (Desmarcado) se ele estiver 'vazio'.
            if item.data(0, QtCore.Qt.CheckStateRole) is None:
                item.setCheckState(0, QtCore.Qt.Unchecked)
        else:
            # Remove a permissão
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsUserCheckable)
            # Remove o dado visual para o quadrado sumir completamente
            item.setData(0, QtCore.Qt.CheckStateRole, None)
    
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
        
        target_central = getattr(self, "central_metrics_text", None)
        metrics = data["metrics"]
        perc = data["perc"]
        
        # Recupera as fácies de reservatório do modelo
        res_facies = sorted(list(self.models[model_key]["reservoir_facies"]))
        res_str = ", ".join(map(str, res_facies)) if res_facies else "Nenhuma"
        
        if target_central:
            if metrics:
                lines = [
                    f"=== {self.models[model_key]['name']} ===",
                    f"Fácies Selecionadas (Reservatório): {res_str}", # <--- ADICIONADO
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

        # Atualiza Tabela (Agora seguro sem sobrescrever cache)
        df = data.get("df")
        self.set_facies_metrics(df)

    # --- LÓGICA DE CÁLCULO E DADOS ---

    def set_metrics(self, metrics, perc):
        """Salva métricas globais no cache do modelo ativo (não só base)."""
        model_key = self.state.get("active_model_key", "base")
        if model_key not in self.cached_metrics:
            self.cached_metrics[model_key] = {"metrics": None, "perc": None, "df": None}

        self.cached_metrics[model_key]["metrics"] = metrics
        self.cached_metrics[model_key]["perc"] = perc

        # Se estiver na aba métricas, atualiza o painel central
        if hasattr(self, "tabs") and self.tabs.currentIndex() == 2:
            self.update_metrics_view_content(model_key)


    def set_facies_metrics(self, df):
        """Salva DataFrame detalhado no cache do modelo ativo (não só base) e preenche tabela."""
        model_key = self.state.get("active_model_key", "base")
        if model_key not in self.cached_metrics:
            self.cached_metrics[model_key] = {"metrics": None, "perc": None, "df": None}

        self.cached_metrics[model_key]["df"] = df

        if df is None or df.empty:
            self.facies_table.setRowCount(0)
            return

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
        self.facies_table.setHorizontalHeaderLabels([pretty.get(c, c) for c in df.columns])

        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                val = df.iloc[i][col]
                if isinstance(val, (float, np.floating)):
                    if col in ["fraction", "connected_fraction", "Perc_X", "Perc_Y", "Perc_Z"]:
                        txt = f"{val:.3f}"
                    elif "volume" in col:
                        txt = f"{val:.2e}"
                    else:
                        txt = f"{val:.2f}"
                else:
                    txt = str(val)
                self.facies_table.setItem(i, j, QtWidgets.QTableWidgetItem(txt))

        self.facies_table.resizeColumnsToContents()


    def change_reservoir_facies(self, reservoir_set):
        import numpy as np
        from load_data import facies as base_facies
        from analysis import compute_global_metrics_for_array, generate_detailed_metrics_df

        # Normaliza entrada (GLOBAL)
        rf_global = set(int(x) for x in (reservoir_set or []))
        self.state["reservoir_facies"] = set(rf_global)

        # Atualiza "reservoir_facies" de cada modelo como interseção
        for mk, m in (self.models or {}).items():
            f = m.get("facies")
            if f is None and mk == "base":
                f = base_facies
            if f is None:
                m["reservoir_facies"] = set()
                continue

            present = set(int(v) for v in np.unique(np.asarray(f).astype(int)))
            # Interseção: Só mantém fácies que existem no modelo E foram selecionadas
            rf_local = set(rf_global & present)
            m["reservoir_facies"] = rf_local
            
            # --- CORREÇÃO: Recalcula Cache de Métricas para o modelo, passando o GRID correto ---
            # Isso corrige o "Resumo Global: 0" e "Volumes Negativos"
            grid_local = m.get("grid") # Pega o grid específico do modelo (pode ser Base ou Compare)
            if grid_local is None and mk == "base": 
                from load_data import grid as grid_local
            
            if f is not None:
                # Agora passamos grid_local para calcular volumes corretos
                met, perc = compute_global_metrics_for_array(f, rf_local, target_grid=grid_local)
                df_det = generate_detailed_metrics_df(f, target_grid=grid_local)
                
                if mk not in self.cached_metrics:
                    self.cached_metrics[mk] = {}
                self.cached_metrics[mk]["metrics"] = met
                self.cached_metrics[mk]["perc"] = perc
                self.cached_metrics[mk]["df"] = df_det

        # --- Atualiza UI para o modelo ATIVO ---
        active_key = self.state.get("active_model_key", "base")
        
        # Aplica no state do visualize (Visualização 3D)
        upd = self.state.get("update_reservoir_fields")
        rf_active = self.models.get(active_key, {}).get("reservoir_facies", set())
        
        if callable(upd):
            try:
                upd(set(rf_active))
            except Exception as e:
                print("[change_reservoir_facies] update_reservoir_fields falhou:", e)

        # Refresh do Plotter
        refresh = self.state.get("refresh")
        if callable(refresh):
            try: refresh()
            except: pass

        # Atualiza a vista atual (Tabelas e Textos)
        try:
            if hasattr(self, "viz_container"):
                idx = self.viz_container.currentIndex()
                if idx == 1 and hasattr(self, "update_2d_map"):
                    self.update_2d_map()
                elif idx == 2 and hasattr(self, "update_metrics_view_content"):
                    self.update_metrics_view_content(active_key)
        except Exception: pass

        # Sidebar
        try: self.update_sidebar_metrics_text(active_key)
        except: pass

        # Atualiza legenda sem recursão
        try:
            self._block_facies_legend_signal = True
            if hasattr(self, "populate_facies_legend"):
                self.populate_facies_legend()
        finally:
            self._block_facies_legend_signal = False

    def build_reports_ribbon_panel(self):
        """Painel do tab Reports (para colocar no ribbon)."""
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(12)

        def make_btn(text, icon, slot):
            b = QtWidgets.QToolButton()
            b.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
            b.setText(text)
            b.setIcon(icon)
            b.setIconSize(QtCore.QSize(28, 28))
            b.setAutoRaise(True)
            b.clicked.connect(slot)
            return b

        ico = self.style()
        lay.addWidget(make_btn("Relatório\nPoços", ico.standardIcon(QtWidgets.QStyle.SP_FileDialogInfoView), self.open_reports_dialog))
        lay.addWidget(make_btn("Relatórios\nSelecionados", ico.standardIcon(QtWidgets.QStyle.SP_DirOpenIcon), self.open_selected_well_reports))
        lay.addWidget(make_btn("Ranking\nModelos", ico.standardIcon(QtWidgets.QStyle.SP_ArrowUp), self.show_models_well_fit_ranking))
        lay.addStretch(1)

        return w
    
    def _schedule_wells_update(self):
        """Evita travar/piscar: agrupa várias chamadas num único redraw."""
        if getattr(self, "_wells_update_pending", False):
            return
        self._wells_update_pending = True
        QtCore.QTimer.singleShot(40, self._run_wells_update)

    def _run_wells_update(self):
        self._wells_update_pending = False
        try:
            self.update_wells_3d()
        except Exception as e:
            print("[wells] update_wells_3d falhou:", e)



    def load_compare_model(self, grdecl_path, study_name="Geral"):
        import os, time
        import numpy as np
        from PyQt5 import QtWidgets

        # Carrega geometria + facies do modelo
        try:
            from load_data import load_grid_from_grdecl, nx, ny, nz
            grid_compare, fac_compare = load_grid_from_grdecl(grdecl_path)
        except Exception as e:
            print(f"Erro ao carregar {grdecl_path}: {e}")
            return

        # Compatibilidade
        if fac_compare.size != nx * ny * nz:
            print(f"Grid incompatível: {grdecl_path}")
            return

        model_id = f"compare_{int(time.time() * 1000)}_{os.path.basename(grdecl_path)}"
        model_name = os.path.basename(grdecl_path)

        # ---------- Reservoir GLOBAL (flatten + interseção) ----------
        rf_raw = self.state.get("reservoir_facies", set()) or set()
        rf_global = set()
        for x in rf_raw:
            if isinstance(x, (set, list, tuple, np.ndarray)):
                for y in x: rf_global.add(int(y))
            else: rf_global.add(int(x))

        present = set(int(v) for v in np.unique(np.asarray(fac_compare).astype(int)))
        rf = rf_global & present

        # Guarda modelo
        self.models[model_id] = {
            "name": model_name,
            "facies": fac_compare,
            "grid": grid_compare,
            "reservoir_facies": set(rf),
            "view_mode": self.state.get("mode", "facies"),
            "study": study_name # Guarda metadado do study
        }

        # Estatísticas e métricas
        try:
            from analysis import facies_distribution_array, compute_global_metrics_for_array, reservoir_facies_distribution_array
            stats, _ = facies_distribution_array(fac_compare, target_grid=grid_compare)
            cm, cp = compute_global_metrics_for_array(fac_compare, rf, target_grid=grid_compare)
            df_detail = self.generate_detailed_metrics_df(fac_compare, target_grid=grid_compare)

            self.cached_metrics[model_id] = {"metrics": cm, "perc": cp, "df": df_detail}

            self.compare_facies = fac_compare
            self.compare_facies_stats = stats
            self.comp_res_stats, _ = reservoir_facies_distribution_array(fac_compare, rf, target_grid=grid_compare)
        except Exception as e:
            print(f"Erro métricas {model_name}: {e}")

        # Adiciona na árvore dentro do Study
        self.add_model_to_tree(model_id, model_name, study_name=study_name)

        # Atualiza UI
        if hasattr(self, "update_comparison_tables"): self.update_comparison_tables()
        if hasattr(self, "refresh_comparison_active_view"): self.refresh_comparison_active_view()

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
        """Atualiza os mapas 2D de todos os modelos ativos na comparação."""
        if not hasattr(self, "state"): return
        
        # 1. Descobre qual o modo de espessura atual
        presets = self.state.get("thickness_presets", {})
        mode = self.state.get("thickness_mode", "Espessura")
        
        if mode not in presets: 
            if presets: mode = list(presets.keys())[0]
            else: return

        scalar, title_suffix = presets[mode]
        
        # 2. Se houver plotters 2D de comparação ativos (se você reativar essa feature)
        # Como o layout dinâmico atual foca no 3D, esta função serve para garantir
        # que o loop de atualização não quebre e esteja pronto para uso futuro.
        
        if hasattr(self, "active_comp_2d_plotters") and hasattr(self, "active_comp_states"):
            for plotter, state in zip(self.active_comp_2d_plotters, self.active_comp_states):
                grid = state.get("current_grid_source")
                model_key = state.get("model_key", "?")
                
                # Busca nome amigável
                model_name = str(model_key)
                if model_key in self.models:
                    model_name = self.models[model_key].get("name", str(model_key))
                
                full_title = f"{model_name}\n{title_suffix}"
                
                if grid:
                    self._draw_2d_map_local(plotter, grid, scalar, full_title)

    def _draw_2d_map_local(self, plotter, grid_source, scalar_name_3d, title):
        from load_data import nx, ny, nz
        import pyvista as pv
        import numpy as np
        
        # Limpa o plotter antes de desenhar
        plotter.clear()

        if scalar_name_3d not in grid_source.cell_data:
            plotter.render() # Força limpeza visual
            return
            
        # Prepara os dados 2D a partir do 3D
        arr3d = grid_source.cell_data[scalar_name_3d].reshape((nx, ny, nz), order="F")
        thickness_2d = np.full((nx, ny), np.nan, dtype=float)
        
        # Mapa de máximo/soma na coluna (conforme sua lógica original)
        for ix in range(nx):
            for iy in range(ny):
                col = arr3d[ix, iy, :]
                col = col[col > 0]
                if col.size > 0: thickness_2d[ix, iy] = col.max()
                
        # Cria o grid estruturado 2D
        x_min, x_max, y_min, y_max, _, z_max = grid_source.bounds
        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)
        xs, ys = np.meshgrid(xs, ys, indexing="ij")
        zs = np.full_like(xs, z_max)
        
        surf = pv.StructuredGrid(xs, ys, zs)
        name2d = scalar_name_3d + "_2d"
        
        # Ajusta array para o pyvista (Cell Data)
        surf.cell_data[name2d] = thickness_2d[:nx-1, :ny-1].ravel(order="F")
        
        # Trata NaNs
        arr = surf.cell_data[name2d]
        arr = np.where(arr < 0, np.nan, arr)
        surf.cell_data[name2d] = arr
        
        # Pega limites de cor
        clim = get_2d_clim(scalar_name_3d, arr)
        
        # Adiciona a malha
        plotter.add_mesh(
            surf, 
            scalars=name2d, 
            cmap="plasma", 
            show_edges=True, 
            edge_color="black", 
            line_width=0.5, 
            nan_color="white", 
            show_scalar_bar=False, 
            clim=clim
        )
        
        # Configura câmera 2D
        plotter.view_xy()
        plotter.enable_parallel_projection()
        plotter.set_background("white")
        plotter.add_scalar_bar(title=title, n_labels=5, fmt="%.1f")
        plotter.reset_camera()
        
        # --- CORREÇÃO: Força o redesenho imediato na tela ---
        plotter.render()

    # --- ABA COMPARAÇÃO ---

    def setup_comparison_tab(self):
        self.compare_tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.compare_tab)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        w_2d = QtWidgets.QWidget()
        l_2d = QtWidgets.QHBoxLayout(w_2d)
        self.comp_plotter_base_2d, base_2d_widget = self._make_embedded_plotter(parent=w_2d)
        self.comp_plotter_comp_2d, comp_2d_widget = self._make_embedded_plotter(parent=w_2d)
        l_2d.addWidget(base_2d_widget); l_2d.addWidget(comp_2d_widget)
        splitter.addWidget(w_2d)

        w_table = QtWidgets.QWidget()
        table_layout = QtWidgets.QHBoxLayout(w_table)

        self.res_table_base_cmp = QtWidgets.QTableWidget()
        self.res_table_base_cmp.setColumnCount(2)
        self.res_table_base_cmp.setHorizontalHeaderLabels(["Fácies", "Reservatório?"])
        self.res_table_base_cmp.verticalHeader().setVisible(False)

        self.res_table_comp_cmp = QtWidgets.QTableWidget()
        self.res_table_comp_cmp.setColumnCount(2)
        self.res_table_comp_cmp.setHorizontalHeaderLabels(["Fácies", "Reservatório?"])
        self.res_table_comp_cmp.verticalHeader().setVisible(False)

        table_layout.addWidget(self.res_table_base_cmp)
        table_layout.addWidget(self.res_table_comp_cmp)
        splitter.addWidget(w_table)

        wt = QtWidgets.QWidget()
        lt = QtWidgets.QHBoxLayout(wt)
        self.comp_plotter_base, base_3d_widget = self._make_embedded_plotter(parent=wt)
        self.comp_plotter_comp, comp_3d_widget = self._make_embedded_plotter(parent=wt)
        lt.addWidget(base_3d_widget); lt.addWidget(comp_3d_widget); split.addWidget(wt)

        layout.addWidget(splitter)
        self.tabs.addTab(self.compare_tab, "Comparação")

        self.res_table_base_cmp.itemChanged.connect(self.update_base_reservoir_compare)
        self.res_table_comp_cmp.itemChanged.connect(self.update_comp_reservoir_compare)


    def update_comparison_tables(self):
        # --- garante que a tabela GLOBAL existe e está na UI ---
        if not hasattr(self, "global_compare_table"):
            self.global_compare_table = QtWidgets.QTableWidget()
            self.global_compare_table.setColumnCount(4)
            self.global_compare_table.setHorizontalHeaderLabels(["Métrica", "Base", "Comp", "Dif"])
            self.global_compare_table.verticalHeader().setVisible(False)
            self.global_compare_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            self.global_compare_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

            # Se existir o tab widget de métricas comparadas, coloca como primeira aba
            if hasattr(self, "tabs_compare_metrics") and isinstance(self.tabs_compare_metrics, QtWidgets.QTabWidget):
                # evita duplicar se já existir
                existing = [self.tabs_compare_metrics.tabText(i) for i in range(self.tabs_compare_metrics.count())]
                if "Global" not in existing:
                    tab_global = QtWidgets.QWidget()
                    l = QtWidgets.QVBoxLayout(tab_global)
                    l.setContentsMargins(0, 0, 0, 0)
                    l.addWidget(self.global_compare_table)
                    self.tabs_compare_metrics.insertTab(0, tab_global, "Global")

        # --- validações mínimas (não derruba o app) ---
        if not hasattr(self, "cached_metrics") or "base" not in self.cached_metrics or "compare" not in self.cached_metrics:
            return

        # 1. Recupera as métricas do cache
        m0 = self.cached_metrics["base"].get("metrics", {})
        m1 = self.cached_metrics["compare"].get("metrics", {})

        # --- TABELA 1: GLOBAL ---
        rows = []

        def get(m, k):
            return m.get(k) if m else None

        rows.append(("NTG", get(m0, "ntg"), get(m1, "ntg")))
        rows.append(("Total Cel", get(m0, "total_cells"), get(m1, "total_cells")))
        rows.append(("Res Cel", get(m0, "res_cells"), get(m1, "res_cells")))
        rows.append(("Conectividade", get(m0, "connected_fraction"), get(m1, "connected_fraction")))
        rows.append(("Clusters", get(m0, "n_clusters"), get(m1, "n_clusters")))
        rows.append(("Maior Cluster", get(m0, "largest_size"), get(m1, "largest_size")))

        self.global_compare_table.setRowCount(len(rows))
        for i, (label, a, b) in enumerate(rows):
            self.global_compare_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(label)))

            val_a = f"{a:.3f}" if isinstance(a, float) else str(a) if a is not None else "-"
            val_b = f"{b:.3f}" if isinstance(b, float) else str(b) if b is not None else "-"

            self.global_compare_table.setItem(i, 1, QtWidgets.QTableWidgetItem(val_a))
            self.global_compare_table.setItem(i, 2, QtWidgets.QTableWidgetItem(val_b))

            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                diff = b - a
                item_diff = QtWidgets.QTableWidgetItem(f"{diff:.3f}")
                if diff > 0:
                    item_diff.setForeground(QColor("green"))
                elif diff < 0:
                    item_diff.setForeground(QColor("red"))
                self.global_compare_table.setItem(i, 3, item_diff)
            else:
                self.global_compare_table.setItem(i, 3, QtWidgets.QTableWidgetItem("-"))

        self.global_compare_table.resizeColumnsToContents()

        # --- TABELA 2: DISTRIBUIÇÃO DE FÁCIES (GRID INTEIRO) ---
        stats0 = getattr(self, "base_facies_stats", {}) or {}
        stats1 = getattr(self, "compare_facies_stats", {}) or {}

        if hasattr(self, "facies_compare_table") and stats0:
            all_facies = sorted(set(stats0.keys()) | set(stats1.keys()))
            self.facies_compare_table.setRowCount(len(all_facies))

            for row, fac in enumerate(all_facies):
                s0 = stats0.get(fac, {"cells": 0, "fraction": 0.0})
                s1 = stats1.get(fac, {"cells": 0, "fraction": 0.0})

                self.facies_compare_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(fac)))
                self.facies_compare_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(s0.get("cells", 0))))
                self.facies_compare_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{100*s0.get('fraction',0.0):.1f}%"))
                self.facies_compare_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(s1.get("cells", 0))))
                self.facies_compare_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{100*s1.get('fraction',0.0):.1f}%"))
                self.facies_compare_table.setItem(row, 5, QtWidgets.QTableWidgetItem(f"{s0.get('volume',0):.2e}"))
                self.facies_compare_table.setItem(row, 6, QtWidgets.QTableWidgetItem(f"{s1.get('volume',0):.2e}"))
                self.facies_compare_table.setItem(row, 7, QtWidgets.QTableWidgetItem(f"{s0.get('thickness_gross',0):.1f}"))
                self.facies_compare_table.setItem(row, 8, QtWidgets.QTableWidgetItem(f"{s1.get('thickness_gross',0):.1f}"))

            self.facies_compare_table.resizeColumnsToContents()

        # --- TABELA 3: RESERVATÓRIO ---
        # Nota: aqui sua versão original usa o "facies" importado do load_data
        stats0r, _ = reservoir_facies_distribution_array(facies, self.models["base"]["reservoir_facies"])
        stats1r = getattr(self, "comp_res_stats", {}) or {}

        if hasattr(self, "reservoir_facies_compare_table") and stats0r:
            res_union = sorted(set(stats0r.keys()) | set(stats1r.keys()))
            self.reservoir_facies_compare_table.setRowCount(len(res_union))

            for row, fac in enumerate(res_union):
                s0 = stats0r.get(fac, {"cells": 0, "fraction": 0.0})
                s1 = stats1r.get(fac, {"cells": 0, "fraction": 0.0})

                self.reservoir_facies_compare_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(fac)))
                self.reservoir_facies_compare_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(s0.get("cells", 0))))
                self.reservoir_facies_compare_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{100*s0.get('fraction',0.0):.1f}%"))
                self.reservoir_facies_compare_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(s1.get("cells", 0))))
                self.reservoir_facies_compare_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{100*s1.get('fraction',0.0):.1f}%"))

            self.reservoir_facies_compare_table.resizeColumnsToContents()
    
    def update_multi_model_filter_table(self, model_data_list):
        """
        Atualiza a tabela de filtros com nomes encurtados e colunas estreitas.
        """
        import numpy as np
        from PyQt5 import QtGui, QtWidgets

        if not hasattr(self, "multi_model_table"): return

        # 1. Normaliza entrada
        normalized_models = []
        for item in model_data_list:
            if isinstance(item, (tuple, list)):
                key, name = str(item[0]), str(item[1])
            else:
                key = str(item)
                name = self.models.get(key, {}).get("name", key) if hasattr(self, "models") else key
            normalized_models.append((key, name))

        t = self.multi_model_table
        t.blockSignals(True)
        try:
            # 2. Coleta União de Fácies
            facies_union = set()
            facies_by_model = {}

            for m_key, _ in normalized_models:
                _, f = self._get_model_payload(m_key)
                if f is None: 
                    uniq = []
                else: 
                    try: uniq = np.unique(np.asarray(f).astype(int))
                    except: uniq = []
                
                s = set(int(x) for x in uniq)
                facies_by_model[m_key] = s
                facies_union |= s

            facies_list = sorted(list(facies_union))

            # 3. Configura Tabela e Cabeçalhos Personalizados
            t.clear()
            t.setRowCount(len(facies_list))
            t.setColumnCount(1 + len(normalized_models))

            # Cabeçalho Coluna 0
            t.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem("Fácies"))

            # Cabeçalhos dos Modelos (Encurtados)
            for c, (m_key, m_name) in enumerate(normalized_models, start=1):
                # Lógica de encurtamento: Pega os últimos 20 caracteres
                if len(m_name) > 20:
                    display_name = "..." + m_name[-20:]
                else:
                    display_name = m_name
                
                item_header = QtWidgets.QTableWidgetItem(display_name)
                item_header.setToolTip(m_name) # Mostra nome completo ao passar o mouse
                t.setHorizontalHeaderItem(c, item_header)

            # --- AJUSTE DE LARGURA ---
            header = t.horizontalHeader()
            
            # Coluna Fácies: Ajusta ao conteúdo (pequena)
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
            
            # Colunas Modelos: Interativa (ajustável) e largura fixa inicial
            for c in range(1, t.columnCount()):
                header.setSectionResizeMode(c, QtWidgets.QHeaderView.Interactive)
                t.setColumnWidth(c, 90) # Largura fina (90 pixels)

            # Helper ícone
            def make_icon(fac_id):
                if not hasattr(self, "facies_colors") or not self.facies_colors: return None, None
                rgba = self.facies_colors.get(int(fac_id))
                if rgba is None: return None, None
                c = QtGui.QColor(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
                pm = QtGui.QPixmap(14, 14); pm.fill(c)
                return QtGui.QIcon(pm), c

            # 4. Preenche Linhas
            for r, fac in enumerate(facies_list):
                # Coluna 0
                it_fac = QtWidgets.QTableWidgetItem(str(fac))
                it_fac.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                icon, color = make_icon(fac)
                if icon: it_fac.setIcon(icon)
                if color: 
                    bg = QtGui.QColor(color); bg.setAlpha(40)
                    it_fac.setBackground(QtGui.QBrush(bg))
                t.setItem(r, 0, it_fac)

                # Colunas Modelos
                for c, (m_key, _) in enumerate(normalized_models, start=1):
                    present = fac in facies_by_model.get(m_key, set())
                    it_chk = QtWidgets.QTableWidgetItem("")
                    it_chk.setData(QtCore.Qt.UserRole, (m_key, int(fac)))
                    
                    if not present:
                        it_chk.setFlags(QtCore.Qt.ItemIsSelectable)
                        it_chk.setBackground(QtGui.QBrush(QtGui.QColor(245, 245, 245)))
                    else:
                        it_chk.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsSelectable)
                        rf = set()
                        if hasattr(self, "models") and m_key in self.models:
                            rf = self.models[m_key].get("reservoir_facies", set()) or set()
                        it_chk.setCheckState(QtCore.Qt.Checked if int(fac) in rf else QtCore.Qt.Unchecked)
                    
                    t.setItem(r, c, it_chk)
            
        except Exception as e:
            print(f"Erro tabela filtro: {e}")
        finally:
            t.blockSignals(False)


    def contextMenuEvent(self, event):
        menu = self.createPopupMenu()
        if menu is not None:
            menu.exec_(event.globalPos())


    def _get_model_payload(self, model_key):
        """Retorna (grid, facies) para Base ou modelo adicional."""
        def pick(d, *keys):
            if not isinstance(d, dict):
                return None
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return None

        key_str = str(model_key)
        is_base = key_str.lower() == "base"

        if is_base:
            # >>> No seu projeto o BASE grid vem de load_data.grid (global)
            from load_data import grid as grid_base
            fac = None
            if hasattr(self, "models") and isinstance(self.models, dict) and "base" in self.models:
                fac = pick(self.models["base"], "facies", "facies_data")
            return grid_base, fac

        # Modelos adicionais (você guarda grid e facies no self.models[model_id])
        if hasattr(self, "models") and isinstance(self.models, dict) and key_str in self.models:
            m = self.models[key_str]
            grid = pick(m, "grid", "ugrid", "pv_grid")
            facies = pick(m, "facies", "facies_data")
            return grid, facies

        return None, None




    def _get_reservoir_facies_for_base(self):
        """Reservoir facies do modelo base, se existir."""
        if hasattr(self, "models") and isinstance(self.models, dict) and "base" in self.models:
            return self.models["base"].get("reservoir_facies")
        return None



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
        """Carrega múltiplos modelos e os agrupa em um Study."""
        # 1. Seleciona arquivos
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Selecionar Modelos", "grids", "GRDECL (*.grdecl)")
        if not paths: return

        # 2. Pergunta o nome do Estudo (Grupo)
        study_name, ok = QtWidgets.QInputDialog.getText(
            self, "Novo Estudo", "Nome do Estudo / Grupo de Calibração:", 
            text=f"Calibração {len(self.models)}"
        )
        
        if not ok or not study_name.strip():
            study_name = "Importação Recente"

        # 3. Carrega
        # Mostra um cursor de espera ou barra de progresso simples seria ideal, mas vamos direto
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for path in paths: 
                self.load_compare_model(path, study_name=study_name)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _create_legend_table(self, h):
        t = QtWidgets.QTableWidget(); t.setColumnCount(len(h)); t.setHorizontalHeaderLabels(h)
        return t

    def init_compare_3d(self):
        if self.models["base"]["facies"] is None:
            return

        from visualize import run
        from load_data import grid as grid_base

        # BASE
        self.comp_plotter_base.clear()
        self.compare_states["base"] = {}

        base_grid = self.models["base"].get("grid", grid_base)
        g0 = base_grid.copy(deep=True)
        g0.cell_data["Facies"] = self.models["base"]["facies"]

        run(
            mode="facies",
            external_plotter=self.comp_plotter_base,
            external_state=self.compare_states["base"],
            target_grid=g0,
            target_facies=self.models["base"]["facies"],
        )

        # COMPARE
        self.comp_plotter_comp.clear()
        self.compare_states["compare"] = {}

        if self.models["compare"]["facies"] is not None:
            compare_grid = self.models["compare"].get("grid", None)
            if compare_grid is None:
                # fallback (não deveria acontecer se você usar load_compare_model corrigido)
                compare_grid = grid_base

            g1 = compare_grid.copy(deep=True)
            g1.cell_data["Facies"] = self.models["compare"]["facies"]

            run(
                mode="facies",
                external_plotter=self.comp_plotter_comp,
                external_state=self.compare_states["compare"],
                target_grid=g1,
                target_facies=self.models["compare"]["facies"],
            )

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
        # Pode ser chamado antes do visualize.run(...) preencher callbacks.
        refresh_main = self.state.get("refresh")

        if axis == "z" and mode == "scale":
            self.state["z_exag"] = float(value)
            if callable(refresh_main):
                refresh_main()
        else:
            if "set_slice" in self.state:
                self.state["set_slice"](axis, mode, value)
                if callable(refresh_main):
                    refresh_main()

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
        from PyQt5 import QtCore

        if getattr(self, "_block_facies_legend_signal", False):
            return
        if not item:
            return
        if item.column() != 3:
            return

        table = self.facies_legend_table
        if table is None:
            return

        new_set = set()
        for r in range(table.rowCount()):
            it = table.item(r, 3)
            if it and it.checkState() == QtCore.Qt.Checked:
                fid = it.data(QtCore.Qt.UserRole)
                if fid is None:
                    try:
                        fid = int(table.item(r, 1).text())
                    except Exception:
                        continue
                new_set.add(int(fid))

        self.change_reservoir_facies(new_set)


    def refresh_wells_in_view(self):
        self._schedule_wells_update()




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
        """Preenche a tabela de Clusters separada."""
        self.clusters_legend_table.blockSignals(True)
        
        sizes = self.state.get("clusters_sizes")
        lut = self.state.get("clusters_lut")
        
        if not sizes or not lut:
            self.clusters_legend_table.setRowCount(0)
            self.clusters_legend_table.blockSignals(False)
            return
            
        labels = sorted(sizes.keys(), key=lambda k: sizes[k], reverse=True)
        self.clusters_legend_table.setRowCount(len(labels))
        
        for row, lab in enumerate(labels):
            # Cor
            r, g, b, a = lut.GetTableValue(int(lab))
            c = QColor(int(r*255), int(g*255), int(b*255))
            item_c = QtWidgets.QTableWidgetItem()
            item_c.setBackground(QBrush(c))
            item_c.setFlags(QtCore.Qt.ItemIsEnabled)
            self.clusters_legend_table.setItem(row, 0, item_c)
            
            # ID
            item_id = QtWidgets.QTableWidgetItem(str(lab))
            item_id.setFlags(QtCore.Qt.ItemIsEnabled)
            self.clusters_legend_table.setItem(row, 1, item_id)
            
            # Count
            item_n = QtWidgets.QTableWidgetItem(str(sizes[lab]))
            item_n.setFlags(QtCore.Qt.ItemIsEnabled)
            self.clusters_legend_table.setItem(row, 2, item_n)
            
        self.clusters_legend_table.resizeColumnsToContents()
        self.clusters_legend_table.blockSignals(False)

    def change_mode(self, new_mode):
        import numpy as np
        from load_data import facies as base_facies

        # --- Tratamento Especial: Entropia ---
        if new_mode == "entropy":
            # Calcula e configura o visualizador para modo escalar
            self.recalc_entropy_view()
            # Atualiza botão do ribbon para refletir
            if hasattr(self, "btn_mode") and self.btn_mode is not None:
                self.btn_mode.setText("Modo\nEntropia")
            # Não continuamos para a lógica padrão de Fácies/Reservatório
            return

        # --- Lógica Padrão (Fácies, Reservatório, Clusters...) ---
        
        # A. Atualiza Estado Global
        self.state["mode"] = new_mode
        for k in self.models.keys():
            self.models[k]["view_mode"] = new_mode

        # B. Atualiza Visualização PRINCIPAL (Aba 0)
        current_f = self.state.get("current_facies")
        if current_f is None: current_f = base_facies
        
        rf_global = set(self.state.get("reservoir_facies", set()) or [])
        present = set(int(v) for v in np.unique(np.asarray(current_f).astype(int)))
        rf_active = rf_global & present

        if new_mode in ("reservoir", "clusters", "largest"):
            upd = self.state.get("update_reservoir_fields")
            if callable(upd): 
                try: upd(set(rf_active))
                except: pass

        refresh = self.state.get("refresh")
        if callable(refresh): refresh()

        # Legendas
        try:
            if new_mode in ("clusters", "largest"):
                if hasattr(self, "facies_legend_table"): self.facies_legend_table.setVisible(False)
                if hasattr(self, "clusters_legend_table"):
                    self.clusters_legend_table.setVisible(True)
                    self.populate_clusters_legend()
            else:
                if hasattr(self, "clusters_legend_table"): self.clusters_legend_table.setVisible(False)
                if hasattr(self, "facies_legend_table"):
                    self.facies_legend_table.setVisible(True)
                    self.populate_facies_legend()
        except: pass
        
        if hasattr(self, "viz_container"):
            idx = self.viz_container.currentIndex()
            if idx == 1 and hasattr(self, "update_2d_map"): self.update_2d_map()

        # C. ATUALIZAÇÃO DA COMPARAÇÃO (Aba 1)
        if hasattr(self, "central_stack") and self.central_stack.currentIndex() == 1:
            if hasattr(self, "active_comp_states"):
                for st in self.active_comp_states:
                    st["mode"] = new_mode
                    m_key = st.get("model_key")
                    rf_local = set()
                    if m_key and m_key in self.models:
                        rf_local = self.models[m_key].get("reservoir_facies", set())
                    
                    if new_mode in ("reservoir", "clusters", "largest"):
                        if "update_reservoir_fields" in st:
                            st["update_reservoir_fields"](rf_local)
                    
                    if "refresh" in st: st["refresh"]()

        self._schedule_wells_update()

    def change_thickness_mode(self, label):
        self.state["thickness_mode"] = label
        
        # Se saiu do modo Entropia, restaura o colormap padrão (plasma)
        if label != "Entropy":
            self.state["thickness_cmap"] = "plasma"

        # 1. Atualiza Visualização PRINCIPAL
        if "update_thickness" in self.state and callable(self.state["update_thickness"]):
            self.state["update_thickness"]()

        refresh = self.state.get("refresh")
        if callable(refresh): refresh()

        # Atualiza 2D Main
        if hasattr(self, "update_2d_map") and callable(self.update_2d_map):
            self.update_2d_map()

        # 2. ATUALIZA COMPARAÇÃO
        if hasattr(self, "active_comp_states"):
            for st in self.active_comp_states:
                st["thickness_mode"] = label
                if label != "Entropy": st["thickness_cmap"] = "plasma"
                
                if "update_thickness" in st: st["update_thickness"]()
                if "refresh" in st: st["refresh"]()
                if "plotter_ref" in st: st["plotter_ref"].render()
        
        self.update_compare_2d_maps()


    def toggle_tree_checkboxes(self, show):
        """Habilita ou desabilita checkboxes em todos os itens (Recursivo)."""
        root = self.project_tree.invisibleRootItem()
        
        def set_visible_recursive(parent):
            for i in range(parent.childCount()):
                item = parent.child(i)
                role = item.data(0, QtCore.Qt.UserRole)
                
                # Aplica para Modelos (Base/Filhos) E Pastas (Studies)
                if role in ("model_root", "study_folder"):
                    self._set_item_checkbox_visible(item, show)
                
                # Continua descendo na hierarquia (para pegar modelos dentro de pastas)
                set_visible_recursive(item)

        set_visible_recursive(root)
    
    def on_comp_slice_changed(self, axis, mode, value):
        """Recebe evento do Slicer da Comparação e aplica em TODOS os grids ativos."""
        # Atualiza a UI do próprio slicer (spinner vs slider)
        self.comp_slicer.external_update(axis, mode, value)
        
        if not hasattr(self, 'active_comp_states'): return
        
        for state in self.active_comp_states:
            # Aplica Exagero Z
            if axis == "z" and mode == "scale":
                state["z_exag"] = float(value)
            
            # Aplica Cortes (I, J, K)
            elif "set_slice" in state:
                state["set_slice"](axis, mode, value)
            
            # Redesenha
            if "refresh" in state: state["refresh"]()
            
        # Força render dos plotters (caso o refresh seja lazy)
        if hasattr(self, 'active_comp_plotters'):
            for p in self.active_comp_plotters:
                p.render()

    def setup_comparison_dock_content(self):
        """
        Painel Lateral de Comparação:
        Topo: Slicer (Cortes X/Y/Z)
        Fundo: Tabela Matriz de Filtros (Fácies x Modelos)
        """
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        
        # 1. SLICER
        from load_data import nx, ny, nz
        self.comp_slicer = GridSlicerWidget(nx, ny, nz, self.on_comp_slice_changed)
        gb_slice = QtWidgets.QGroupBox("Cortes & Escala (Sincronizado)")
        l_sl = QtWidgets.QVBoxLayout(gb_slice)
        l_sl.addWidget(self.comp_slicer)
        layout.addWidget(gb_slice)
        
        # 2. FILTRO MATRIZ (Multi-Modelo)
        self.comp_filter_group = QtWidgets.QGroupBox("Filtro de Reservatório por Modelo")
        l_filt = QtWidgets.QVBoxLayout(self.comp_filter_group)
        
        # --- NOVO: Botões de Seleção ---
        h_btn_multi = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("Todos")
        btn_all.clicked.connect(lambda: self.toggle_all_multi_model(True))
        
        btn_none = QtWidgets.QPushButton("Nenhum")
        btn_none.clicked.connect(lambda: self.toggle_all_multi_model(False))
        
        h_btn_multi.addWidget(btn_all)
        h_btn_multi.addWidget(btn_none)
        l_filt.addLayout(h_btn_multi)
        # -------------------------------
        
        self.multi_model_table = QtWidgets.QTableWidget()
        self.multi_model_table.verticalHeader().setVisible(False)
        self.multi_model_table.itemChanged.connect(self.on_multi_model_filter_changed)
        
        l_filt.addWidget(self.multi_model_table)
        layout.addWidget(self.comp_filter_group)
        
        return container
    
    def toggle_all_facies_legend(self, check):
        """Marca ou desmarca todas as fácies na legenda de Propriedades."""
        self._block_facies_legend_signal = True
        try:
            state = QtCore.Qt.Checked if check else QtCore.Qt.Unchecked
            rows = self.facies_legend_table.rowCount()
            for r in range(rows):
                item = self.facies_legend_table.item(r, 3) # Coluna do Checkbox
                if item:
                    item.setCheckState(state)
        finally:
            self._block_facies_legend_signal = False
        
        # Dispara atualização final (simula um clique na primeira linha válida)
        if self.facies_legend_table.rowCount() > 0:
            self.on_legend_item_changed(self.facies_legend_table.item(0, 3))

    def toggle_all_multi_model(self, check):
        """Marca ou desmarca tudo na tabela de filtro por modelo."""
        self._block_multi_model_filter = True
        try:
            state = QtCore.Qt.Checked if check else QtCore.Qt.Unchecked
            rows = self.multi_model_table.rowCount()
            cols = self.multi_model_table.columnCount()
            
            # Atualiza visual da tabela e o set de dados interno
            for c in range(2, cols): # Colunas de modelos começam no índice 2
                # Tenta descobrir o model_key olhando para o UserRole da primeira célula válida da coluna
                model_key = None
                for r_chk in range(rows):
                    it_chk = self.multi_model_table.item(r_chk, c)
                    if it_chk and it_chk.data(QtCore.Qt.UserRole):
                        # UserRole guarda (model_key, fac_id)
                        model_key = it_chk.data(QtCore.Qt.UserRole)[0]
                        break
                
                if not model_key or model_key not in self.models: continue
                
                target_set = self.models[model_key]["reservoir_facies"]
                
                for r in range(rows):
                    it = self.multi_model_table.item(r, c)
                    if it and (it.flags() & QtCore.Qt.ItemIsUserCheckable):
                        it.setCheckState(state)
                        # Atualiza o set de dados manualmente pois bloqueamos o sinal
                        data_tuple = it.data(QtCore.Qt.UserRole)
                        if data_tuple:
                            fac_id = data_tuple[1]
                            if check: target_set.add(fac_id)
                            else: target_set.discard(fac_id)
                        
        finally:
            self._block_multi_model_filter = False
            
        # Força refresh visual de comparação
        if hasattr(self, "active_comp_states"):
            for st in self.active_comp_states:
                # Atualiza campo no state visual
                mk = st.get("model_key")
                if mk and mk in self.models:
                    rf = self.models[mk]["reservoir_facies"]
                    st["reservoir_facies"] = rf
                    if "update_reservoir_fields" in st:
                        st["update_reservoir_fields"](rf)
                    if "refresh" in st: st["refresh"]()
            
            # Atualiza 2D maps
            self.update_compare_2d_maps()
    
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
        """Alterna entre visualização (0) e comparação (1)."""
        if mode == "visualization":
            self.central_stack.setCurrentIndex(0)
            self.act_persp_viz.setChecked(True)
            self.act_persp_comp.setChecked(False)

            if hasattr(self, "inspector_tabs"):
                self.inspector_tabs.setCurrentWidget(self.page_props)

            # --- CORREÇÃO: Checkboxes sempre VISÍVEIS (necessário para o Ranking) ---
            self.toggle_tree_checkboxes(True)
            
            # Opcional: Expandir tudo para facilitar visualização
            # self.project_tree.expandAll()

            if hasattr(self, "act_view_3d"):
                self.act_view_3d.setChecked(True)
            self.show_main_3d_view()

        elif mode == "comparison":
            self.central_stack.setCurrentIndex(1)
            self.act_persp_viz.setChecked(False)
            self.act_persp_comp.setChecked(True)

            if hasattr(self, "inspector_tabs"):
                self.inspector_tabs.setCurrentWidget(self.page_compare)

            # Garante checkboxes visíveis
            self.toggle_tree_checkboxes(True)
            
            # Atualiza a vista de comparação
            checked_models = self.get_checked_models()
            self.update_dynamic_comparison_view(checked_models)

        self.current_perspective = mode

    
    def update_comparison_tables_multi(self, checked_models):
        """
        Atualiza as tabelas de comparação (Global, Fácies, Reservatório)
        para N modelos selecionados.
        checked_models: lista de tuplas [(model_key, model_name), ...]
        """
        if not checked_models:
            self.global_compare_table.setRowCount(0)
            self.facies_compare_table.setRowCount(0)
            self.reservoir_facies_compare_table.setRowCount(0)
            return

        # --- 1) TABELA GLOBAL ---
        t_glob = self.global_compare_table
        t_glob.clear()

        headers_glob = ["Métrica"] + [name for _, name in checked_models]
        t_glob.setColumnCount(len(headers_glob))
        t_glob.setHorizontalHeaderLabels(headers_glob)

        metrics_list = [
            ("NTG", "ntg", "{:.3f}"),
            ("Células Totais", "total_cells", "{:d}"),
            ("Células Res.", "res_cells", "{:d}"),
            ("Conectividade", "connected_fraction", "{:.3f}"),
            ("Clusters", "n_clusters", "{:d}"),
            ("Maior Cluster", "largest_size", "{:d}"),
            ("Vol. Res (m3)", "reservoir_volume", "{:.2e}"),
        ]

        t_glob.setRowCount(len(metrics_list))

        for r, (label, key, fmt) in enumerate(metrics_list):
            t_glob.setItem(r, 0, QtWidgets.QTableWidgetItem(label))

            for c, (m_key, _) in enumerate(checked_models):
                data = self.cached_metrics.get(m_key) or {}
                # ✅ suporta os dois formatos (novo e antigo)
                m = data.get("metrics") or data.get("global") or {}

                if not m:
                    val_str = "-"
                else:
                    val = m.get(key, 0)
                    try:
                        val_str = fmt.format(val)
                    except Exception:
                        val_str = str(val)

                t_glob.setItem(r, c + 1, QtWidgets.QTableWidgetItem(val_str))

        t_glob.resizeColumnsToContents()

        # --- 2. TABELA POR FÁCIES ---
        t_fac = self.facies_compare_table
        t_fac.clear()
        
        # Colunas: Fácies | Mod1(Cel) | Mod1(%) | Mod2(Cel) | Mod2(%) ...
        headers_fac = ["Fácies"]
        for _, name in checked_models:
            headers_fac.append(f"{name}\n(Cél)")
            headers_fac.append(f"{name}\n(%)")
            
        t_fac.setColumnCount(len(headers_fac))
        t_fac.setHorizontalHeaderLabels(headers_fac)
        t_fac.setRowCount(len(sorted_facies))
        
        for r, fac in enumerate(sorted_facies):
            t_fac.setItem(r, 0, QtWidgets.QTableWidgetItem(str(fac)))
            
            for i, (m_key, _) in enumerate(checked_models):
                col_base = 1 + (i * 2)
                stats = model_stats.get(m_key, {})
                s = stats.get(fac, {"cells": 0, "fraction": 0.0})
                
                # Células
                t_fac.setItem(r, col_base, QtWidgets.QTableWidgetItem(str(s["cells"])))
                # Porcentagem
                t_fac.setItem(r, col_base + 1, QtWidgets.QTableWidgetItem(f"{s['fraction']*100:.1f}%"))
                
        t_fac.resizeColumnsToContents()

        # --- 3. TABELA RESERVATÓRIO ---
        t_res = self.reservoir_facies_compare_table
        t_res.clear()
        
        # Mesmo layout de colunas, mas filtrando apenas o que é reservatório em ALGUM modelo
        # Ou mostramos a união das fácies que são reservatório nos modelos selecionados
        res_facies_union = set()
        for m_key, _ in checked_models:
            res_facies_union.update(self.models[m_key]["reservoir_facies"])
            
        sorted_res = sorted(list(res_facies_union))
        
        t_res.setColumnCount(len(headers_fac)) # Mesmos headers (Cel e %)
        t_res.setHorizontalHeaderLabels(headers_fac)
        t_res.setRowCount(len(sorted_res))
        
        for r, fac in enumerate(sorted_res):
            t_res.setItem(r, 0, QtWidgets.QTableWidgetItem(str(fac)))
            
            for i, (m_key, _) in enumerate(checked_models):
                col_base = 1 + (i * 2)
                
                # Verifica se essa fácies é reservatório NESTE modelo específico
                is_res_here = fac in self.models[m_key]["reservoir_facies"]
                
                if is_res_here:
                    stats = model_stats.get(m_key, {})
                    s = stats.get(fac, {"cells": 0, "fraction": 0.0})
                    # Recalcula fração relativa ao reservatório total deste modelo?
                    # Ou mantém fração global? Vamos usar GLOBAL por enquanto para consistência com a tabela anterior.
                    # Se quiser fração do reservatório, precisaria dividir s['cells'] pelo total de res_cells do modelo.
                    
                    val_cel = str(s["cells"])
                    val_perc = f"{s['fraction']*100:.1f}%"
                else:
                    val_cel = "-"
                    val_perc = "-"
                
                t_res.setItem(r, col_base, QtWidgets.QTableWidgetItem(val_cel))
                t_res.setItem(r, col_base + 1, QtWidgets.QTableWidgetItem(val_perc))
                
        t_res.resizeColumnsToContents()
  

    def update_dynamic_comparison_view(self, checked_models=None):
        from visualize import run
        from PyQt5 import QtWidgets, QtCore

        # --- 1. PREPARAÇÃO DA LISTA ---
        if checked_models is None:
            checked_models = self.get_checked_models()

        final_list = []
        
        # CORREÇÃO 1: Removida a adição forçada do "base". 
        # Agora dependemos apenas do que está em checked_models.
        
        for item in checked_models:
            if isinstance(item, (tuple, list)):
                raw_key, raw_name = item[0], item[1] if len(item) > 1 else str(item[0])
            else:
                raw_key, raw_name = item, str(item)
            
            m_key = str(raw_key)
            m_name = str(raw_name)

            if m_name.startswith("('"): 
                 if m_key in self.models: m_name = self.models[m_key].get("name", m_name)

            # CORREÇÃO 1: Removido o filtro "if m_key == 'base': continue"
            final_list.append((m_key, m_name))

        self.update_multi_model_filter_table(final_list)

        # --- 2. LIMPEZA ---
        if hasattr(self, "active_comp_plotters"):
            for p in self.active_comp_plotters:
                try: p.close()
                except: pass
        self.active_comp_plotters = []
        self.active_comp_states = []
        self.compare_states_multi = {}

        while self.comp_layout_3d.count():
            item = self.comp_layout_3d.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        # Parâmetros Globais
        mode = self.state.get("mode", "facies")
        # CORREÇÃO 2: Pega o modo de espessura atual para passar aos novos gráficos
        thickness_mode = self.state.get("thickness_mode", "Espessura") 
        z_exag = float(self.state.get("z_exag", 15.0))
        show_scalar_bar = bool(self.state.get("show_scalar_bar", True))

        # --- 3. LAYOUT GRID ---
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        
        container_widget = QtWidgets.QWidget()
        grid_layout = QtWidgets.QGridLayout(container_widget)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(4)
        
        n_models = len(final_list)
        cols = 2 if n_models > 1 else 1
        
        if n_models == 0:
            lbl = QtWidgets.QLabel("Nenhum modelo selecionado.")
            self.comp_layout_3d.addWidget(lbl)
            return

        for idx, (model_key, model_name) in enumerate(final_list):
            row = idx // cols
            col = idx % cols
            
            w_container = QtWidgets.QWidget()
            w_container.setStyleSheet("border: 1px solid #ccc;") 
            v_lay = QtWidgets.QVBoxLayout(w_container)
            v_lay.setContentsMargins(0, 0, 0, 0)
            v_lay.setSpacing(0)

            lbl = QtWidgets.QLabel(f"{model_name}")
            lbl.setStyleSheet("font-weight: bold; background-color: #ddd; padding: 4px; border: none;")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setFixedHeight(24)
            v_lay.addWidget(lbl)

            plotter, plotter_widget = self._make_embedded_plotter(parent=w_container)
            plotter_widget.setStyleSheet("border: none;")
            plotter_widget.setMinimumSize(300, 300)
            
            try: plotter.set_background("white")
            except: pass
            
            v_lay.addWidget(plotter_widget)
            grid_layout.addWidget(w_container, row, col)

            # --- 4. CARREGA DADOS ---
            grid_obj, facies_obj = self._get_model_payload(model_key)
            
            if grid_obj is None:
                plotter.add_text("GRID OFF", font_size=12)
            else:
                # CORREÇÃO 2: Injeta o thickness_mode no estado inicial
                local_state = {"thickness_mode": thickness_mode}
                
                _, local_state = run(
                    mode=mode,
                    z_exag=z_exag,
                    show_scalar_bar=show_scalar_bar,
                    external_plotter=plotter,
                    external_state=local_state,
                    target_grid=grid_obj,
                    target_facies=facies_obj,
                )
                local_state["model_key"] = model_key
                local_state["plotter_ref"] = plotter
                
                rf = set()
                if hasattr(self, "models") and model_key in self.models:
                    rf = self.models[model_key].get("reservoir_facies", set()) or set()
                
                local_state["reservoir_facies"] = rf
                
                if "update_reservoir_fields" in local_state:
                    local_state["update_reservoir_fields"](rf)

                self.active_comp_states.append(local_state)
                self.compare_states_multi[str(model_key)] = local_state

            self.active_comp_plotters.append(plotter)

        scroll.setWidget(container_widget)
        self.comp_layout_3d.addWidget(scroll)

        if len(self.active_comp_plotters) > 1:
            self.sync_multi_cameras(self.active_comp_plotters)

    def _build_multi_model_filter_table(self, checked_models):
        """Constrói a tabela matriz: Linhas = Fácies, Colunas = Modelos."""
        table = self.multi_model_table
        table.blockSignals(True)
        table.clear()
        
        # 1. Coleta União de Fácies de todos os modelos selecionados
        all_facies = set()
        for key, _ in checked_models:
            f_arr = self.models[key]["facies"]
            if f_arr is not None:
                all_facies.update(np.unique(f_arr))
        sorted_facies = sorted(list(all_facies))
        
        # 2. Configura Colunas: [Cor, ID] + [Modelo 1, Modelo 2, ...]
        headers = ["Cor", "ID"] + [name for key, name in checked_models]
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(sorted_facies))
        
        colors = load_facies_colors()
        
        for r, fac in enumerate(sorted_facies):
            # Cor
            rgba = colors.get(fac, (0.8, 0.8, 0.8, 1.0))
            c = QColor(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
            item_c = QtWidgets.QTableWidgetItem(); item_c.setBackground(QBrush(c)); item_c.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(r, 0, item_c)
            
            # ID
            item_id = QtWidgets.QTableWidgetItem(str(fac)); item_id.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(r, 1, item_id)
            
            # Checkboxes por Modelo
            for c, (key, _) in enumerate(checked_models):
                col_idx = 2 + c
                
                # Verifica se a fácies existe neste modelo
                model_facies_arr = self.models[key]["facies"]
                exists = False
                if model_facies_arr is not None:
                    # Otimização: verificação rápida se existe
                    # (Para grandes arrays, melhor usar sets pré-calculados na carga do modelo)
                    if fac in model_facies_arr: exists = True 
                    # NOTA: O ideal é ter self.models[key]["unique_facies_set"] calculado no load
                
                item_chk = QtWidgets.QTableWidgetItem()
                if exists:
                    item_chk.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                    # Verifica se está no reservoir_facies desse modelo
                    is_sel = fac in self.models[key]["reservoir_facies"]
                    item_chk.setCheckState(QtCore.Qt.Checked if is_sel else QtCore.Qt.Unchecked)
                    item_chk.setData(QtCore.Qt.UserRole, (key, fac)) # Guarda chave e fácies
                else:
                    item_chk.setFlags(QtCore.Qt.NoItemFlags) # Desabilita se não existir no grid
                    item_chk.setBackground(QBrush(QColor(240, 240, 240))) # Cinza claro
                    
                table.setItem(r, col_idx, item_chk)
                
        table.resizeColumnsToContents()
        table.blockSignals(False)

    def on_multi_model_filter_changed(self, item):
        """Atualiza o filtro de reservatório sincronizando o estado visual."""
        from analysis import compute_global_metrics_for_array

        if getattr(self, "_block_multi_model_filter", False): return

        data = item.data(QtCore.Qt.UserRole)
        if not data: return 

        model_key, fac = data
        model_key = str(model_key)
        fac = int(fac)

        if model_key not in self.models: return
        model_data = self.models[model_key]
        target_set = model_data.setdefault("reservoir_facies", set())

        # Atualiza o Set de dados do Modelo
        if item.checkState() == QtCore.Qt.Checked: target_set.add(fac)
        else: target_set.discard(fac)

        # 1. Recalcula Métricas (Silent)
        if model_data.get("facies") is not None:
            m, p = compute_global_metrics_for_array(model_data["facies"], target_set)
            cache = self.cached_metrics.setdefault(model_key, {"metrics": None, "perc": None, "df": None})
            cache["metrics"] = m; cache["perc"] = p

        # 2. ATUALIZAÇÃO VISUAL
        if hasattr(self, "compare_states_multi") and model_key in self.compare_states_multi:
            st = self.compare_states_multi[model_key]
            
            # --- CORREÇÃO CRÍTICA: Atualiza a memória do state visual ---
            # Se não fizermos isso, o st["refresh"]() vai usar o valor antigo e desfazer tudo.
            st["reservoir_facies"] = target_set
            
            # Agora pode chamar o refresh seguro
            if "refresh" in st:
                st["refresh"]()
            
            # Força renderização imediata
            if "plotter_ref" in st:
                st["plotter_ref"].render()

        # 3. Atualiza tabela de métricas se visível
        if hasattr(self, "compare_stack") and self.compare_stack.currentIndex() == 1:
             if hasattr(self, "update_dynamic_comparison_metrics"):
                 self.update_dynamic_comparison_metrics(self.get_checked_models())

        # 4. Atualiza sidebar
        if self.state.get("active_model_key") == model_key:
            self.update_sidebar_metrics_text(model_key)

    
    def on_comp_view_changed(self, index):
        """Callback do combo do ribbon: troca aba da comparação."""
        if not hasattr(self, "compare_stack"):
            return

        # 0=3D, 1=Métricas, 2=2D (mesmo order do stack)
        if index < 0:
            index = 0
        if index > 2:
            index = 2

        self.compare_stack.setCurrentIndex(index)
        self.refresh_comparison_active_view()
    
    def get_checked_wells(self):
        """Retorna a lista de nomes dos poços marcados (Checked) na árvore."""
        checked = []
        
        # Se a pasta de poços não existir, retorna a lista completa (segurança)
        if not hasattr(self, "wells_root_item") or self.wells_root_item is None:
            return list(self.wells.keys())

        # Varre os filhos da pasta Poços
        for i in range(self.wells_root_item.childCount()):
            item = self.wells_root_item.child(i)
            if item.checkState(0) == QtCore.Qt.Checked:
                # O nome do poço está salvo no UserRole+1 (conforme load_well_dialog)
                well_name = item.data(0, QtCore.Qt.UserRole + 1)
                if well_name:
                    checked.append(well_name)
        
        return checked
    
    def get_checked_models(self):
        """Retorna EXATAMENTE o que está marcado com checkbox na árvore (Base + Filhos de Pastas)."""
        checked = []
        root = self.project_tree.invisibleRootItem()

        def traverse(parent_item):
            for i in range(parent_item.childCount()):
                item = parent_item.child(i)
                role = item.data(0, QtCore.Qt.UserRole)
                
                # Se for modelo (pode ser o Base no topo ou filho de pasta)
                if role == "model_root":
                    if item.checkState(0) == QtCore.Qt.Checked:
                        mk = item.data(0, QtCore.Qt.UserRole + 1)
                        if mk: checked.append((mk, item.text(0)))
                
                # Se for pasta, entra nela
                elif role == "study_folder":
                    traverse(item)

        traverse(root)
        return checked

    def refresh_comparison_active_view(self):
        """Atualiza 3D / Métricas / 2D conforme a aba ativa da comparação."""
        if not hasattr(self, "central_stack") or self.central_stack.currentIndex() != 1:
            return
        if not hasattr(self, "compare_stack"):
            return

        # Coleta modelos marcados no tree
        checked_models = []
        it = QtWidgets.QTreeWidgetItemIterator(self.project_tree)
        while it.value():
            item = it.value()
            if item.data(0, QtCore.Qt.UserRole) == "model_root":
                if item.checkState(0) == QtCore.Qt.Checked:
                    checked_models.append((item.data(0, QtCore.Qt.UserRole + 1), item.text(0)))
            it += 1

        page = self.compare_stack.currentIndex()

        if page == 0:
            self.update_dynamic_comparison_view(checked_models)
        elif page == 1:
            # Se você já tiver update_dynamic_comparison_metrics, use:
            if hasattr(self, "update_dynamic_comparison_metrics"):
                self.update_dynamic_comparison_metrics(checked_models)
            else:
                # fallback: tenta manter o que existia
                self.update_comparison_tables()
        elif page == 2:
            self.update_dynamic_comparison_2d(checked_models)


    
    def update_dynamic_comparison_metrics(self, checked_models):
        """Monta a tabela de métricas para N modelos marcados."""
        # normaliza
        normalized = []
        for m in (checked_models or []):
            if isinstance(m, (tuple, list)):
                key = m[0]
                name = m[1] if len(m) > 1 else self.models.get(key, {}).get("name", str(key))
            else:
                key = m
                name = self.models.get(key, {}).get("name", str(key))
            if key in self.models:
                normalized.append((key, name))

        if not normalized:
            self.global_compare_table.setRowCount(0)
            self.facies_compare_table.setRowCount(0)
            self.reservoir_facies_compare_table.setRowCount(0)
            return

        # GLOBAL
        headers = ["Métrica"] + [name for _, name in normalized]
        self.global_compare_table.clear()
        self.global_compare_table.setColumnCount(len(headers))
        self.global_compare_table.setHorizontalHeaderLabels(headers)

        metrics_rows = [
            ("NTG", "ntg", "{:.3f}"),
            ("Total Células", "total_cells", "{:d}"),
            ("Células Res", "res_cells", "{:d}"),
            ("Conectividade", "connected_fraction", "{:.3f}"),
            ("Nº Clusters", "n_clusters", "{:d}"),
            ("Maior Cluster", "largest_size", "{:d}"),
        ]

        self.global_compare_table.setRowCount(len(metrics_rows))

        for r, (lbl, key, fmt) in enumerate(metrics_rows):
            self.global_compare_table.setItem(r, 0, QtWidgets.QTableWidgetItem(lbl))
            for c, (mk, _) in enumerate(normalized):
                data = self.cached_metrics.get(mk, {})
                m = data.get("metrics") or {}
                val = m.get(key, None)
                if val is None:
                    txt = "-"
                else:
                    try: txt = fmt.format(val)
                    except: txt = str(val)
                self.global_compare_table.setItem(r, c+1, QtWidgets.QTableWidgetItem(txt))

        self.global_compare_table.resizeColumnsToContents()


    
    def update_dynamic_comparison_2d(self, checked_models):
        """Reconstrói a visualização de Mapas 2D."""

        # --- LIMPEZA ---
        if hasattr(self, 'active_comp_2d_plotters'):
            for p in self.active_comp_2d_plotters:
                try: p.close()
                except: pass
        self.active_comp_2d_plotters = []

        while self.comp_2d_layout.count():
            item = self.comp_2d_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # --- NORMALIZA checked_models (aceita tuplas longas, tuplas curtas e strings) ---
        normalized = []
        for m in (checked_models or []):
            if isinstance(m, (tuple, list)):
                model_key = m[0]
                model_name = m[1] if len(m) > 1 else self.models.get(model_key, {}).get("name", str(model_key))
            else:
                model_key = m
                model_name = self.models.get(model_key, {}).get("name", str(model_key))

            if model_key in self.models:
                normalized.append((model_key, model_name))

        if not normalized:
            self.comp_2d_layout.addWidget(QtWidgets.QLabel("Selecione modelos."))
            return

        # --- GRID LAYOUT ---
        n_models = len(normalized)
        cols = 2 if n_models > 1 else 1

        grid_container = QtWidgets.QWidget()
        grid_layout = QtWidgets.QGridLayout(grid_container)
        grid_layout.setContentsMargins(0,0,0,0)
        grid_layout.setSpacing(2)
        self.comp_2d_layout.addWidget(grid_container)

        # Recupera configuração de espessura
        presets = self.state.get("thickness_presets") or {}
        thick_mode = self.state.get("thickness_mode", "Espessura")
        if thick_mode not in presets and presets:
            thick_mode = list(presets.keys())[0]
        if thick_mode not in presets:
            thick_mode = "Espessura"

        scalar, title = presets.get(thick_mode, ("vert_Ttot_reservoir", "Espessura"))

        # ✅ grid base seguro: usa o grid já carregado no estado, se existir
        from load_data import grid as global_grid
        base_grid = self.models.get("base", {}).get("grid") or self.state.get("current_grid_source") or global_grid

        for idx, (model_key, model_name) in enumerate(normalized):
            row, col = idx // cols, idx % cols
            model_data = self.models[model_key]

            p2d = BackgroundPlotter(show=False)
            self.active_comp_2d_plotters.append(p2d)

            src_grid = model_data.get("grid") or base_grid
            if src_grid is None:
                continue

            temp_grid = src_grid.copy(deep=True)
            temp_grid.cell_data["Facies"] = model_data["facies"]

            # Recalcula métricas verticais para este modelo
            self.recalc_vertical_metrics(temp_grid, model_data["facies"], model_data["reservoir_facies"])

            # Desenha
            self._draw_2d_map_local(p2d, temp_grid, scalar, f"{model_name} - {title}")

            w = QtWidgets.QWidget()
            vl = QtWidgets.QVBoxLayout(w)
            vl.setContentsMargins(0,0,0,0)
            vl.setSpacing(0)

            lbl = QtWidgets.QLabel(f"  {model_name} ({thick_mode})")
            lbl.setStyleSheet("background: #ddd; font-weight: bold;")
            vl.addWidget(lbl)
            vl.addWidget(p2d.interactor)

            grid_layout.addWidget(w, row, col)

        # Atualiza filtro matriz (multi-model)
        self._build_multi_model_filter_table(normalized)


    def on_tree_item_changed(self, item, column):
        """Lida com alterações na árvore (Modelos, Studies e Poços) com lógica Pai/Filho manual."""
        if not item: return
        if getattr(self, "_block_tree_signals", False): return

        role = item.data(0, QtCore.Qt.UserRole)

        # ---------------------------------------------------------
        # LÓGICA DE PASTAS (Studies ou Raiz de Poços)
        # ---------------------------------------------------------
        if role in ("study_folder", "wells_root"):
            self._block_tree_signals = True
            try:
                new_state = item.checkState(0)
                # Corrige estado parcial forçado pelo Qt
                if new_state == QtCore.Qt.PartiallyChecked:
                    new_state = QtCore.Qt.Checked
                    item.setCheckState(0, QtCore.Qt.Checked)

                # Aplica a todos os filhos
                for i in range(item.childCount()):
                    child = item.child(i)
                    child.setCheckState(0, new_state)
            finally:
                self._block_tree_signals = False
            
            # Se for poço, atualiza 3D (tubos) E Ranking
            if role == "wells_root":
                self._schedule_wells_update()
            
            self._schedule_heavy_update()
            return

        # ---------------------------------------------------------
        # LÓGICA DE FILHOS (Modelos ou Poços Individuais)
        # ---------------------------------------------------------
        if role in ("model_root", "well_item"):
            parent = item.parent()
            
            # Atualiza o Pai (se houver)
            if parent:
                parent_role = parent.data(0, QtCore.Qt.UserRole)
                if parent_role in ("study_folder", "wells_root"):
                    self._block_tree_signals = True
                    try:
                        checked_count = 0
                        total_count = parent.childCount()
                        for i in range(total_count):
                            if parent.child(i).checkState(0) == QtCore.Qt.Checked:
                                checked_count += 1
                        
                        if checked_count == 0:
                            parent.setCheckState(0, QtCore.Qt.Unchecked)
                        elif checked_count == total_count:
                            parent.setCheckState(0, QtCore.Qt.Checked)
                        else:
                            parent.setCheckState(0, QtCore.Qt.PartiallyChecked)
                    finally:
                        self._block_tree_signals = False

            # Dispara as atualizações necessárias
            if role == "well_item":
                self._schedule_wells_update() # Atualiza visualização 3D dos tubos
            
            self._schedule_heavy_update() # Atualiza Ranking e Comparação
            return

    def _schedule_heavy_update(self):
        """Agrupa chamadas de atualização para evitar congelamento da UI."""
        # Cancela timer anterior se existir
        if hasattr(self, "_update_timer") and self._update_timer.isActive():
            self._update_timer.stop()
        
        # Cria novo timer para rodar daqui a 200ms (tempo suficiente para clicar em vários)
        self._update_timer = QtCore.QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._perform_heavy_update)
        self._update_timer.start(200)

    def _perform_heavy_update(self):
        """Executa a atualização pesada (Ranking, 3D Entropia, Comparação) uma única vez."""
        
        # 1. Se estiver no modo Entropia, recalcula o mapa (pois a seleção de modelos mudou)
        # Verificamos o texto do botão ou uma flag interna
        is_entropy = False
        if hasattr(self, "btn_mode") and "Entropia" in self.btn_mode.text():
            is_entropy = True
        
        if is_entropy:
            self.recalc_entropy_view()

        # 2. Atualiza Comparação 3D se estiver visível (Aba Comparação)
        if hasattr(self, "refresh_comparison_active_view"):
            try: self.refresh_comparison_active_view()
            except: pass
            
        # 3. Atualiza Ranking se estiver visível (Aba Ranking)
        if hasattr(self, "viz_container") and self.viz_container.currentIndex() == 3:
            if hasattr(self, "update_ranking_view_content"): 
                self.update_ranking_view_content()

    def sync_multi_cameras(self, plotters):
        """Sincroniza N plotters."""
        self._is_syncing = False
        
        def sync(src, others):
            if self._is_syncing: return
            self._is_syncing = True
            try:
                for dst in others:
                    dst.camera.position = src.camera.position
                    dst.camera.focal_point = src.camera.focal_point
                    dst.camera.view_angle = src.camera.view_angle
                    dst.camera.up = src.camera.up
                    dst.camera.clipping_range = src.camera.clipping_range
                    dst.render()
            finally:
                self._is_syncing = False
        
        for i, p in enumerate(plotters):
            others = plotters[:i] + plotters[i+1:]
            # Lambda com default value para capturar o p correto no loop
            p.camera.AddObserver("ModifiedEvent", lambda *args, src=p, dsts=others: sync(src, dsts))

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
        """Garante que todos os processos do VTK sejam encerrados antes de matar a janela."""
        
        # 1. Fecha os plotters principais (Aba Visualização)
        if hasattr(self, 'plotter') and self.plotter is not None: 
            self.plotter.close()
        if hasattr(self, 'plotter_2d') and self.plotter_2d is not None: 
            self.plotter_2d.close()
        
        # 2. Fecha plotters antigos (se existirem por legado)
        if hasattr(self, 'comp_plotter_base'): self.comp_plotter_base.close()
        if hasattr(self, 'comp_plotter_comp'): self.comp_plotter_comp.close()
        if hasattr(self, 'comp_plotter_base_2d'): self.comp_plotter_base_2d.close()
        if hasattr(self, 'comp_plotter_comp_2d'): self.comp_plotter_comp_2d.close()
        
        # 3. --- CORREÇÃO PRINCIPAL: Fecha as listas dinâmicas de Comparação ---
        if hasattr(self, 'active_comp_plotters'):
            for p in self.active_comp_plotters:
                try: p.close()
                except: pass
        
        if hasattr(self, 'active_comp_2d_plotters'):
            for p in self.active_comp_2d_plotters:
                try: p.close()
                except: pass
        
        # Aceita o evento de fechamento do Qt
        event.accept()

    def recalc_vertical_metrics(self, target_grid, facies_array, reservoir_set):
        """Recalcula métricas verticais usando a geometria do grid do próprio modelo."""
        if target_grid is None or target_grid.n_cells != nx * ny * nz:
            return

        fac_3d = facies_array.reshape((nx, ny, nz), order="F")

        centers = target_grid.cell_centers().points
        z_vals = centers[:, 2].reshape((nx, ny, nz), order="F")

        keys = [
            "vert_Ttot_reservoir", "vert_NTG_col_reservoir", "vert_NTG_env_reservoir",
            "vert_n_packages_reservoir", "vert_Tpack_max_reservoir",
            "vert_ICV_reservoir", "vert_Qv_reservoir", "vert_Qv_abs_reservoir"
        ]
        data_map = {k: np.zeros((nx, ny, nz), dtype=float) for k in keys}
        res_list = list(reservoir_set)

        for ix in range(nx):
            for iy in range(ny):
                col_fac = fac_3d[ix, iy, :]
                mask = np.isin(col_fac, res_list)
                if not np.any(mask):
                    continue

                col_z = z_vals[ix, iy, :]
                z_min, z_max = np.nanmin(col_z), np.nanmax(col_z)
                T_col = abs(z_max - z_min)
                if T_col == 0:
                    continue

                dz = T_col / nz
                idx = np.where(mask)[0]
                n_res = len(idx)
                T_tot = n_res * dz
                T_env = (idx[-1] - idx[0] + 1) * dz if n_res > 0 else 0.0

                NTG_col = T_tot / T_col
                NTG_env = T_tot / T_env if T_env > 0 else 0.0

                packages = []
                start = idx[0]
                prev = idx[0]
                for k in idx[1:]:
                    if k == prev + 1:
                        prev = k
                    else:
                        packages.append(prev - start + 1)
                        start = prev = k
                packages.append(prev - start + 1)

                T_pack_max = max(packages) * dz if packages else 0.0
                n_packages = len(packages)

                ICV = T_pack_max / T_env if T_env > 0 else 0.0
                Qv = NTG_col * ICV
                Qv_abs = ICV * (T_pack_max / T_col)

                data_map["vert_Ttot_reservoir"][ix, iy, mask] = T_tot
                data_map["vert_NTG_col_reservoir"][ix, iy, mask] = NTG_col
                data_map["vert_NTG_env_reservoir"][ix, iy, mask] = NTG_env
                data_map["vert_n_packages_reservoir"][ix, iy, mask] = float(n_packages)
                data_map["vert_Tpack_max_reservoir"][ix, iy, mask] = T_pack_max
                data_map["vert_ICV_reservoir"][ix, iy, mask] = ICV
                data_map["vert_Qv_reservoir"][ix, iy, mask] = Qv
                data_map["vert_Qv_abs_reservoir"][ix, iy, mask] = Qv_abs

        for k, v in data_map.items():
            target_grid.cell_data[k] = v.reshape(-1, order="F")




    def _open_matplotlib_report(self, well_name, sim_model_name, real_depth, real_fac, base_depth, base_fac, sim_depth, sim_fac, best_depth=None, best_fac=None, window_size_str="1x1"):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection
        from config import load_facies_colors
        import numpy as np
        from analysis import resample_to_normalized_depth

        # --- Cores e Cast ---
        f_colors = load_facies_colors()
        def get_color(fac_code):
            return f_colors.get(int(fac_code), (0.5, 0.5, 0.5, 1.0))

        real_fac = real_fac.astype(int)
        base_fac = base_fac.astype(int)
        sim_fac = sim_fac.astype(int)
        all_facies = sorted(list(set(real_fac) | set(base_fac) | set(sim_fac)))

        # --- Janela ---
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Relatório Poço: {well_name}")
        dialog.resize(1600, 850) 
        dialog.setWindowFlags(dialog.windowFlags() | QtCore.Qt.WindowMinMaxButtonsHint)
        
        main_layout = QtWidgets.QVBoxLayout(dialog)
        tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(tabs)

        # =================================================================
        # ABA 1: LOGS + VOLUME
        # =================================================================
        tab1 = QtWidgets.QWidget()
        l1 = QtWidgets.QVBoxLayout(tab1)
        
        fig1, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(14, 7), 
                                                gridspec_kw={'width_ratios': [0.2, 0.2, 0.2, 3]})
        
        # Dados de espessura (zero-based)
        r_thick_arr = real_depth - real_depth[0]
        r_total = r_thick_arr[-1] if len(r_thick_arr) > 0 else 0
        b_thick_arr = base_depth - base_depth[0] if len(base_depth) > 0 else np.array([])
        b_total = b_thick_arr[-1] if len(b_thick_arr) > 0 else 0
        s_thick_arr = sim_depth - sim_depth[0] if len(sim_depth) > 0 else np.array([])
        s_total = s_thick_arr[-1] if len(s_thick_arr) > 0 else 0
        g_max = max(r_total, b_total, s_total)

        # Helper para desenhar logs simples
        def draw_log(ax, d_arr, f_arr, title):
            patches = []
            colors = []
            if len(d_arr) < 2: return
            
            curr = f_arr[0]
            top = d_arr[0]
            
            def add_text(h_blk, t_pos, code):
                if h_blk > (g_max * 0.02):
                    ax.text(0.5, t_pos + h_blk/2, str(code), 
                            ha='center', va='center', fontsize=7, fontweight='bold',
                            color='white' if sum(get_color(code)[:3]) < 1.5 else 'black')

            for i in range(1, len(f_arr)):
                if f_arr[i] != curr:
                    base = d_arr[i]
                    h = base - top
                    patches.append(Rectangle((0, top), 1, h))
                    colors.append(get_color(curr))
                    add_text(h, top, curr)
                    curr = f_arr[i]
                    top = base
            
            base = d_arr[-1]
            h = base - top
            if h > 0:
                patches.append(Rectangle((0, top), 1, h))
                colors.append(get_color(curr))
                add_text(h, top, curr)

            col = PatchCollection(patches, match_original=True)
            col.set_facecolors(colors)
            ax.add_collection(col)
            ax.set_xlim(0, 1)
            ax.set_ylim(g_max, 0)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

        draw_log(ax1, b_thick_arr, base_fac, f"Base\n{b_total:.1f}m")
        ax1.set_ylabel("Espessura (m)")
        ax1.set_yticks(np.linspace(0, g_max, 10))
        draw_log(ax2, s_thick_arr, sim_fac, f"Simul\n{s_total:.1f}m")
        draw_log(ax3, r_thick_arr, real_fac, f"Real\n{r_total:.1f}m")

        # Gráfico Volume
        def calc_net(d, f):
            if len(d) < 2: return {}
            dz = np.diff(d, prepend=d[0]); dz[0]=0
            c = {}
            for code in all_facies:
                mask = (f == code)
                c[code] = np.sum(dz[mask])
            return c

        net_base = calc_net(base_depth, base_fac)
        net_sim = calc_net(sim_depth, sim_fac)
        net_real = calc_net(real_depth, real_fac)
        
        y_pos = np.arange(len(all_facies))
        h = 0.25
        vals_b = [net_base.get(f,0) for f in all_facies]
        vals_s = [net_sim.get(f,0) for f in all_facies]
        vals_r = [net_real.get(f,0) for f in all_facies]
        
        ax4.barh(y_pos + h, vals_b, h, label='Base', color='#999999')
        ax4.barh(y_pos,     vals_s, h, label='Simulado', color='#007acc')
        ax4.barh(y_pos - h, vals_r, h, label='Real', color='#444444')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([str(f) for f in all_facies])
        ax4.set_title("Balanço Volumétrico")
        ax4.legend()
        ax4.grid(axis='x', linestyle='--', alpha=0.5)
        
        for i, (vr, vs) in enumerate(zip(vals_r, vals_s)):
            if vr > 0:
                diff_perc = ((vs - vr) / vr) * 100
                txt = f"{diff_perc:+.1f}%"
                color = 'green' if abs(diff_perc) < 20 else 'red'
            else:
                txt = "Novo" if vs > 0 else ""
                color = 'blue'
            max_val = max(vr, vals_b[i], vs)
            if max_val > 0:
                ax4.text(max_val, y_pos[i], f" {txt}", va='center', color=color, fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        canvas1 = FigureCanvas(fig1)
        l1.addWidget(canvas1)
        tabs.addTab(tab1, "Logs & Volume")

        # =================================================================
        # ABA 2: MATRIZ & FAMÍLIAS
        # =================================================================
        tab2 = QtWidgets.QWidget()
        l2 = QtWidgets.QVBoxLayout(tab2)
        fig2, (ax2a, ax2b) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        n_bins = 200
        r_norm = resample_to_normalized_depth(real_depth, real_fac, n_bins)
        s_norm = resample_to_normalized_depth(sim_depth, sim_fac, n_bins)
        
        n_classes = len(all_facies)
        conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
        f_to_i = {f: i for i, f in enumerate(all_facies)}
        
        for rv, sv in zip(r_norm, s_norm):
            i = f_to_i.get(rv)
            j = f_to_i.get(sv)
            if i is not None and j is not None:
                conf_matrix[i, j] += 1

        ax2a.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
        ax2a.set_xticks(np.arange(n_classes)); ax2a.set_yticks(np.arange(n_classes))
        ax2a.set_xticklabels([str(f) for f in all_facies], rotation=45)
        ax2a.set_yticklabels([str(f) for f in all_facies])
        ax2a.set_xlabel("Simulado"); ax2a.set_ylabel("Real")
        ax2a.set_title("Matriz de Trocas")

        for i in range(n_classes):
            for j in range(n_classes):
                val = conf_matrix[i, j]
                color = "white" if val > conf_matrix.max()/2 else "black"
                if val > 0:
                    ax2a.text(j, i, str(val), ha="center", va="center", color=color)
                if i == j:
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='gold', linewidth=3)
                    ax2a.add_patch(rect)

        # Famílias
        def get_family(f_code):
            s = str(f_code)
            if s.startswith('1'): return "Siliciclásticos"
            if s.startswith('2'): return "Carbonatos"
            return "Outros"

        fam_stats = {"Real": {}, "Sim": {}, "Base": {}}
        tot_r = sum(net_real.values()) if net_real else 1
        tot_s = sum(net_sim.values()) if net_sim else 1
        tot_b = sum(net_base.values()) if net_base else 1

        for f in all_facies:
            fam = get_family(f)
            fam_stats["Real"][fam] = fam_stats["Real"].get(fam, 0) + net_real.get(f, 0)
            fam_stats["Sim"][fam] = fam_stats["Sim"].get(fam, 0) + net_sim.get(f, 0)
            fam_stats["Base"][fam] = fam_stats["Base"].get(fam, 0) + net_base.get(f, 0)

        families = sorted(list(fam_stats["Real"].keys()))
        x_fam = np.arange(len(families))
        bars_b = [(fam_stats["Base"][fam] / tot_b)*100 for fam in families]
        bars_s = [(fam_stats["Sim"][fam] / tot_s)*100 for fam in families]
        bars_r = [(fam_stats["Real"][fam] / tot_r)*100 for fam in families]

        ax2b.bar(x_fam - 0.2, bars_b, 0.2, label='Base', color='#999999')
        ax2b.bar(x_fam,       bars_s, 0.2, label='Simulado', color='#007acc')
        ax2b.bar(x_fam + 0.2, bars_r, 0.2, label='Real', color='#444444')
        ax2b.set_xticks(x_fam); ax2b.set_xticklabels(families)
        ax2b.set_ylabel("Proporção (%)")
        ax2b.set_title("Balanço por Família (%)")
        ax2b.legend()
        ax2b.set_ylim(0, 100)

        plt.tight_layout()
        canvas2 = FigureCanvas(fig2)
        l2.addWidget(canvas2)
        tabs.addTab(tab2, "Matriz & Famílias")

        # =================================================================
        # ABA 3: TABELA DETALHADA
        # =================================================================
        tab3 = QtWidgets.QWidget()
        l3 = QtWidgets.QVBoxLayout(tab3)
        
        table = QtWidgets.QTableWidget()
        cols = ["Fácies", "Real (m)", "Base (m)", "Sim (m)", "Erro Sim/Real (%)"]
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels(cols)
        table.setRowCount(len(all_facies))
        
        for row, fac in enumerate(all_facies):
            vr = net_real.get(fac, 0); vb = net_base.get(fac, 0); vs = net_sim.get(fac, 0)
            if vr > 0: err_perc = ((vs - vr) / vr) * 100
            else: err_perc = 100.0 if vs > 0 else 0.0
            
            item_fac = QtWidgets.QTableWidgetItem(str(fac)); item_fac.setTextAlignment(QtCore.Qt.AlignCenter)
            rgba = get_color(fac); bg = QColor(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
            item_fac.setBackground(QBrush(bg)); 
            if sum(rgba[:3]) < 1.5: item_fac.setForeground(QColor("white"))
            
            table.setItem(row, 0, item_fac)
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{vr:.2f}"))
            table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{vb:.2f}"))
            table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{vs:.2f}"))
            item_err = QtWidgets.QTableWidgetItem(f"{err_perc:+.1f}%")
            if abs(err_perc) > 20: item_err.setForeground(QColor("red"))
            elif abs(err_perc) < 5: item_err.setForeground(QColor("green"))
            table.setItem(row, 4, item_err)

        table.resizeColumnsToContents()
        l3.addWidget(table)
        tabs.addTab(tab3, "Tabela de Métricas")

        # =================================================================
        # ABA 4: CORRELAÇÃO & RANKING (Lado a Lado)
        # =================================================================
        tab4 = QtWidgets.QWidget()
        layout4 = QtWidgets.QHBoxLayout(tab4)
        
        layout4.addStretch(1)

        # --- ESQUERDA: Correlação ---
        fig4a, ax4a = plt.subplots(figsize=(5, 8))
        fig4a.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.05)

        b_norm = resample_to_normalized_depth(base_depth, base_fac, n_bins)
        s_norm = resample_to_normalized_depth(sim_depth, sim_fac, n_bins)
        r_norm = resample_to_normalized_depth(real_depth, real_fac, n_bins)

        self._plot_strat_correlation_real_depth(
            ax4a,
            n_bins=n_bins,
            base_fac_bins=b_norm,
            sim_fac_bins=s_norm,
            real_fac_bins=r_norm,
            b_total=b_total,
            s_total=s_total,
            r_total=r_total,
            get_color=get_color,
            min_bins=1,
            link_alpha=0.18
        )
        ax4a.set_title("Correlação Estratigráfica", fontsize=10, pad=10)
        
        canvas4a = FigureCanvas(fig4a)
        
        # --- [1] AJUSTE A LARGURA DA CORRELAÇÃO AQUI (px) ---
        canvas4a.setMinimumWidth(450)
        canvas4a.setMaximumWidth(550) 
        
        layout4.addWidget(canvas4a)

        # --- DIREITA: Ranking Detail Tracks ---
        if best_depth is not None and len(best_depth) > 0:
            best_thick_arr = best_depth - best_depth[0]
            best_total = best_thick_arr[-1]
        else:
            best_depth, best_fac = sim_depth, sim_fac
            best_thick_arr, best_total = s_thick_arr, s_total

        fig4b, axs4b = plt.subplots(1, 4, figsize=(5, 8), sharey=True)
        fig4b.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.05, wspace=0.3)

        # [NOVA] Função para agrupar camadas adjacentes iguais (remove linhas)
        def group_layers(depth, facies, is_grid_format=True):
            if len(depth) == 0: return []
            
            # 1. Converte formato Grid (Pares Top/Base) para Blocos Brutos
            raw_blocks = []
            if is_grid_format:
                for k in range(0, len(depth)-1, 2):
                    raw_blocks.append((depth[k], depth[k+1], int(facies[k])))
            else:
                # Formato Log contínuo
                curr = int(facies[0])
                top = depth[0]
                for k in range(1, len(facies)):
                    if int(facies[k]) != curr:
                        raw_blocks.append((top, depth[k], curr))
                        top = depth[k]
                        curr = int(facies[k])
                raw_blocks.append((top, depth[-1], curr))

            # 2. Funde blocos adjacentes iguais
            if not raw_blocks: return []
            
            merged = []
            curr_top, curr_base, curr_fac = raw_blocks[0]
            
            for i in range(1, len(raw_blocks)):
                next_top, next_base, next_fac = raw_blocks[i]
                
                # Se for a mesma fácies e estiver "colado" (tolera gap mínimo de arredondamento)
                if next_fac == curr_fac and abs(next_top - curr_base) < 0.05:
                    curr_base = next_base # Estende a base
                else:
                    merged.append((curr_top, curr_base, curr_fac))
                    curr_top, curr_base, curr_fac = next_top, next_base, next_fac
            
            merged.append((curr_top, curr_base, curr_fac))
            return merged

        def plot_track(ax, d, f, title, is_grid=True):
            ax.set_title(title, fontsize=8, pad=8)
            ax.set_xticks([])
            ax.set_facecolor('white')
            
            d_rel = d - d[0] if len(d)>0 else []
            
            # [USANDO] A função que agrupa
            layers = group_layers(d_rel, f, is_grid)
            
            max_y = max(b_total, s_total, best_total, r_total)
            ax.set_ylim(max_y, 0)
            
            for top, base, fac in layers:
                h = base - top
                if h <= 0: continue
                # [MODIFICADO] edgecolor='none' remove a linha preta entre blocos
                rect = Rectangle((0, top), 1, h, facecolor=get_color(fac), edgecolor='none')
                ax.add_patch(rect)
                
                if h > max_y * 0.03:
                    lum = sum(get_color(fac)[:3])
                    txt_c = 'white' if lum < 1.5 else 'black'
                    ax.text(0.5, top + h/2, str(fac), ha='center', va='center', fontsize=6, color=txt_c, fontweight='bold')

        plot_track(axs4b[0], base_depth, base_fac, f"BASE\n{b_total:.1f}m", True)
        plot_track(axs4b[1], sim_depth, sim_fac, f"SIM (Orig)\n{s_total:.1f}m", True)
        plot_track(axs4b[2], best_depth, best_fac, f"SIM ({window_size_str})\n{best_total:.1f}m", True)
        plot_track(axs4b[3], real_depth, real_fac, f"REAL\n{r_total:.1f}m", False)

        for ax in axs4b[1:]: ax.set_yticks([])
        axs4b[0].set_ylabel("Espessura Relativa (m)", fontsize=9)

        canvas4b = FigureCanvas(fig4b)
        
        # --- [2] AJUSTE A LARGURA DO BEST MATCH AQUI (px) ---
        canvas4b.setMinimumWidth(450)
        canvas4b.setMaximumWidth(550)
        
        layout4.addWidget(canvas4b)
        
        layout4.addStretch(1)

        tabs.addTab(tab4, "Correlação & Best Match")

        return dialog
    
    def _compute_strat_links(self, fac_a, fac_b):
        """
        Cria links como SEGMENTOS contínuos em profundidade normalizada.
        Retorna:
        blocks_a: lista (s,e,fac)
        blocks_b: lista (s,e,fac)
        links: lista (a_idx, b_idx, s, e) onde [s,e) é o intervalo (em bins)
                em que fac_a e fac_b ficam constantes.
        """
        import numpy as np

        def blocks_from_series(f):
            blocks = []
            if len(f) == 0:
                return blocks
            start = 0
            curr = int(f[0])
            for i in range(1, len(f)):
                if int(f[i]) != curr:
                    blocks.append((start, i, curr))
                    start = i
                    curr = int(f[i])
            blocks.append((start, len(f), curr))
            return blocks

        fac_a = np.asarray(fac_a).astype(int)
        fac_b = np.asarray(fac_b).astype(int)
        n = len(fac_a)

        blocks_a = blocks_from_series(fac_a)
        blocks_b = blocks_from_series(fac_b)

        # map bin -> bloco
        bin_to_a = np.empty(n, dtype=int)
        for idx, (s, e, _) in enumerate(blocks_a):
            bin_to_a[s:e] = idx

        bin_to_b = np.empty(n, dtype=int)
        for idx, (s, e, _) in enumerate(blocks_b):
            bin_to_b[s:e] = idx

        # links como "runs" contínuos de (a_idx, b_idx)
        links = []
        if n == 0:
            return blocks_a, blocks_b, links

        cur_a = int(bin_to_a[0])
        cur_b = int(bin_to_b[0])
        run_s = 0

        for k in range(1, n):
            a = int(bin_to_a[k])
            b = int(bin_to_b[k])
            if a != cur_a or b != cur_b:
                links.append((cur_a, cur_b, run_s, k))  # [run_s, k)
                cur_a, cur_b, run_s = a, b, k

        links.append((cur_a, cur_b, run_s, n))
        return blocks_a, blocks_b, links
    
    def _compute_bin_runs(self, fac_from, fac_to, n_bins):
        """
        Cria 'runs' contínuos ao longo dos bins onde (fac_from, fac_to) não muda.
        Retorna lista de tuplas: (f_from, f_to, k0, k1) com intervalo [k0, k1).
        """
        import numpy as np

        fac_from = np.asarray(fac_from).astype(int)
        fac_to = np.asarray(fac_to).astype(int)

        runs = []
        if len(fac_from) == 0:
            return runs

        f0 = int(fac_from[0])
        t0 = int(fac_to[0])
        k_start = 0

        for k in range(1, n_bins):
            ff = int(fac_from[k])
            tt = int(fac_to[k])
            if ff != f0 or tt != t0:
                runs.append((f0, t0, k_start, k))
                f0, t0, k_start = ff, tt, k

        runs.append((f0, t0, k_start, n_bins))
        return runs


    
    def _plot_strat_correlation(
        self, ax, n_bins,
        blocks_left, blocks_mid, blocks_right,
        links_lm, links_mr,
        get_color,
        min_bins=1,
        link_alpha=0.25,
        color_links_by="left"  # "left" ou "mid"
    ):
        import numpy as np
        from matplotlib.patches import Rectangle, Polygon
        from matplotlib.collections import PatchCollection

        # ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_xticks([])
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_ylabel("Profundidade normalizada")

        xL0, xL1 = 0.05, 0.20
        xM0, xM1 = 0.40, 0.55
        xR0, xR1 = 0.75, 0.90

        def draw_column(blocks, x0, x1, label):
            patches = []
            colors = []
            for (s, e, fac) in blocks:
                y0 = s / n_bins
                y1 = e / n_bins
                h = y1 - y0
                patches.append(Rectangle((x0, y0), x1 - x0, h))
                colors.append(get_color(fac))
                if h > 0.05:
                    ax.text((x0 + x1) / 2, y0 + h / 2, str(fac),
                            ha='center', va='center', fontsize=9,
                            color='white' if sum(get_color(fac)[:3]) < 1.5 else 'black',
                            fontweight='bold')
            col = PatchCollection(patches, match_original=True)
            col.set_facecolors(colors)
            col.set_edgecolor((0, 0, 0, 0.15))
            ax.add_collection(col)
            ax.text((x0 + x1) / 2, -0.04, label, ha="center", va="top", fontsize=10)

        draw_column(blocks_left, xL0, xL1, "Base")
        draw_column(blocks_mid,  xM0, xM1, "Simulado")
        draw_column(blocks_right,xR0, xR1, "Real")

        # Helpers para pegar facies de um bloco
        def fac_of(blocks, idx):
            return int(blocks[idx][2])

        def draw_links(x_from, x_to, blocks_from, blocks_to, links, color_mode):
            for (iA, iB, s, e) in links:
                w = e - s
                if w < min_bins:
                    continue

                y0 = s / n_bins
                y1 = e / n_bins

                # faixa com mesma altura exata do intervalo
                # (conecta a borda direita da coluna origem à borda esquerda da coluna destino)
                poly = Polygon([
                    (x_from, y0),
                    (x_to,   y0),
                    (x_to,   y1),
                    (x_from, y1),
                ], closed=True)

                if color_mode == "left":
                    rgba = get_color(fac_of(blocks_from, iA))
                else:
                    rgba = get_color(fac_of(blocks_to, iB))

                # deixa mais transparente
                poly.set_facecolor((rgba[0], rgba[1], rgba[2], link_alpha))
                poly.set_edgecolor(None)
                ax.add_patch(poly)

        # Base -> Sim (cor pelo Base)
        draw_links(xL1, xM0, blocks_left, blocks_mid, links_lm, "left")

        # Sim -> Real (cor pelo Sim)
        draw_links(xM1, xR0, blocks_mid, blocks_right, links_mr, "left")

    def _plot_strat_correlation_real_depth(
        self,
        ax,
        n_bins,
        base_fac_bins, sim_fac_bins, real_fac_bins,
        b_total, s_total, r_total,
        get_color,
        min_bins=2,
        link_alpha=0.22,
        well_width_px=85,   # <<< largura "máxima" em pixels (ajuste aqui)
        gap_px=110          # <<< afastamento entre poços em pixels (ajuste aqui)
    ):
        import numpy as np
        from matplotlib.patches import Rectangle, Polygon
        from matplotlib.collections import PatchCollection

        def compute_bin_runs(f_from, f_to, n_bins_):
            f_from = np.asarray(f_from).astype(int)
            f_to = np.asarray(f_to).astype(int)
            runs_ = []
            if len(f_from) == 0:
                return runs_
            cur_from = int(f_from[0])
            cur_to = int(f_to[0])
            k0 = 0
            for k in range(1, n_bins_):
                ff = int(f_from[k])
                tt = int(f_to[k])
                if ff != cur_from or tt != cur_to:
                    runs_.append((cur_from, cur_to, k0, k))
                    cur_from, cur_to, k0 = ff, tt, k
            runs_.append((cur_from, cur_to, k0, n_bins_))
            return runs_

        base_fac_bins = np.asarray(base_fac_bins).astype(int)
        sim_fac_bins  = np.asarray(sim_fac_bins).astype(int)
        real_fac_bins = np.asarray(real_fac_bins).astype(int)

        g_max = max(b_total, s_total, r_total)

        # ---------------- Layout geral ----------------
        # ax.set_title(title, pad=16, fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(g_max, 0)
        ax.set_xticks([])
        ax.set_yticks(np.linspace(0, g_max, 10))
        ax.set_ylabel("Espessura (m)")

        # remove bordas pretas
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)

        # ---------------- Largura em px -> fração do eixo ----------------
        fig = ax.figure
        dpi = fig.get_dpi()
        fig_w_px = fig.get_size_inches()[0] * dpi

        left_margin = 0.10
        right_margin = 0.90
        avail = right_margin - left_margin

        w_frac = well_width_px / fig_w_px
        gap_frac = gap_px / fig_w_px

        total_needed = 3 * w_frac + 2 * gap_frac
        if total_needed > avail:
            scale = avail / total_needed
            w_frac *= scale
            gap_frac *= scale

        # posições finais (3 poços)
        xL0 = left_margin
        xL1 = xL0 + w_frac
        xM0 = xL1 + gap_frac
        xM1 = xM0 + w_frac
        xR0 = xM1 + gap_frac
        xR1 = xR0 + w_frac

        def blocks_from_bins(f_bins):
            blocks = []
            if len(f_bins) == 0:
                return blocks
            start = 0
            curr = int(f_bins[0])
            for k in range(1, len(f_bins)):
                if int(f_bins[k]) != curr:
                    blocks.append((start, k, curr))
                    start = k
                    curr = int(f_bins[k])
            blocks.append((start, len(f_bins), curr))
            return blocks

        def draw_column_from_bins(f_bins, total, x0, x1, label):
            blocks = blocks_from_bins(f_bins)
            patches, colors = [], []

            for (k0, k1, fac) in blocks:
                y0 = (k0 / n_bins) * total
                y1 = (k1 / n_bins) * total
                h = y1 - y0
                if h <= 0:
                    continue
                patches.append(Rectangle((x0, y0), x1 - x0, h))
                colors.append(get_color(fac))

                if h > 0.05 * g_max:
                    ax.text((x0 + x1) / 2, y0 + h / 2, str(fac),
                            ha='center', va='center', fontsize=9,
                            color='white' if sum(get_color(fac)[:3]) < 1.5 else 'black',
                            fontweight='bold')

            col = PatchCollection(patches, match_original=True)
            col.set_facecolors(colors)
            col.set_edgecolor("none")
            ax.add_collection(col)

            # textos no topo, sem sobrepor
            ax.text((x0 + x1) / 2, -0.055 * g_max, label,
                    ha="center", va="top", fontsize=10)
            ax.text((x0 + x1) / 2, -0.025 * g_max, f"{total:.1f}m",
                    ha="center", va="top", fontsize=10, fontweight="bold")

        # colunas
        draw_column_from_bins(base_fac_bins, b_total, xL0, xL1, "Base")
        draw_column_from_bins(sim_fac_bins,  s_total, xM0, xM1, "Simul")
        draw_column_from_bins(real_fac_bins, r_total, xR0, xR1, "Real")

        # links
        runs_bs = compute_bin_runs(base_fac_bins, sim_fac_bins, n_bins)
        runs_sr = compute_bin_runs(sim_fac_bins, real_fac_bins, n_bins)

        def draw_links(runs, x_from, x_to, total_from, total_to):
            for (f_from, f_to, k0, k1) in runs:
                w = k1 - k0
                if w < min_bins:
                    continue

                y0_from = (k0 / n_bins) * total_from
                y1_from = (k1 / n_bins) * total_from
                y0_to   = (k0 / n_bins) * total_to
                y1_to   = (k1 / n_bins) * total_to

                rgba = get_color(f_from)
                face = (rgba[0], rgba[1], rgba[2], link_alpha)

                poly = Polygon([
                    (x_from, y0_from),
                    (x_to,   y0_to),
                    (x_to,   y1_to),
                    (x_from, y1_from),
                ], closed=True, facecolor=face, edgecolor=None)
                ax.add_patch(poly)

        draw_links(runs_bs, xL1, xM0, b_total, s_total)  # Base -> Sim
        draw_links(runs_sr, xM1, xR0, s_total, r_total)  # Sim -> Real

    def _drop_last_block(self, depth, facies):
        """
        Remove o último bloco contínuo de fácies (última camada do barcode).
        """
        import numpy as np
        depth = np.asarray(depth, dtype=float)
        facies = np.asarray(facies).astype(int)

        if len(depth) < 2 or len(facies) < 2:
            return depth, facies

        last = int(facies[-1])
        i0 = len(facies) - 1
        while i0 > 0 and int(facies[i0 - 1]) == last:
            i0 -= 1

        # se tudo é um bloco só, não corta
        if i0 <= 0:
            return depth, facies

        return depth[:i0], facies[:i0]
    
    def _compute_auto_well_shift_xy(self, well):
        """Calcula (dx,dy) para trazer o poço para o centro do grid BASE."""
        import numpy as np
        from load_data import grid as base_grid

        if base_grid is None or well is None or well.data is None or well.data.empty:
            return 0.0, 0.0

        b = base_grid.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        grid_cx = 0.5 * (b[0] + b[1])
        grid_cy = 0.5 * (b[2] + b[3])

        wx = float(np.nanmean(well.data["X"].astype(float).values))
        wy = float(np.nanmean(well.data["Y"].astype(float).values))

        dx = grid_cx - wx
        dy = grid_cy - wy
        return dx, dy


    def _apply_well_shift_if_needed(self, well):
        """Aplica shift (dx,dy,dz) no poço. Guarda o shift e reutiliza pros próximos."""
        import numpy as np
        from config import (
            AUTO_WELL_SHIFT, AUTO_WELL_SHIFT_THRESHOLD,
            WELL_OFFSET_X, WELL_OFFSET_Y, WELL_OFFSET_Z
        )

        if well is None or well.data is None or well.data.empty:
            return

        # shift base (manual)
        mdx, mdy, mdz = float(WELL_OFFSET_X), float(WELL_OFFSET_Y), float(WELL_OFFSET_Z)

        # shift automático (uma vez) e reutiliza para todos os poços do mesmo projeto
        if AUTO_WELL_SHIFT:
            if not hasattr(self, "_well_xy_shift"):
                dx, dy = self._compute_auto_well_shift_xy(well)
                dist = float(np.hypot(dx, dy))
                if dist >= float(AUTO_WELL_SHIFT_THRESHOLD):
                    self._well_xy_shift = (dx, dy)
                    print(f"[AUTO_WELL_SHIFT] dx={dx:.3f}, dy={dy:.3f} (dist={dist:.3f})")
                else:
                    self._well_xy_shift = (0.0, 0.0)
                    print(f"[AUTO_WELL_SHIFT] shift ignorado (dist={dist:.3f} < threshold)")

            dx, dy = self._well_xy_shift
        else:
            dx, dy = 0.0, 0.0

        # aplica (auto + manual)
        well.apply_xyz_shift(dx + mdx, dy + mdy, mdz)

    def open_reports_dialog(self):
        from PyQt5 import QtWidgets, QtCore

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Relatórios de Poços")
        dlg.resize(420, 520)

        layout = QtWidgets.QVBoxLayout(dlg)

        # Modelo
        layout.addWidget(QtWidgets.QLabel("Modelo (SIM):"))
        cmb = QtWidgets.QComboBox()
        model_keys = [k for k in self.models.keys() if self.models[k].get("facies") is not None]
        if "base" in self.models and "base" not in model_keys:
            model_keys = ["base"] + model_keys

        for k in model_keys:
            name = self.models.get(k, {}).get("name", k)
            cmb.addItem(name, userData=k)

        layout.addWidget(cmb)

        # Poços
        layout.addWidget(QtWidgets.QLabel("Poços:"))
        lst = QtWidgets.QListWidget()
        lst.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        for wname in sorted(self.wells.keys()):
            it = QtWidgets.QListWidgetItem(wname)
            it.setData(QtCore.Qt.UserRole, wname)
            lst.addItem(it)

        layout.addWidget(lst)

        # Botões
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Open | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(btns)

        def _open():
            model_key = cmb.currentData()
            selected = [i.data(QtCore.Qt.UserRole) for i in lst.selectedItems()]
            if not selected:
                QtWidgets.QMessageBox.information(dlg, "Info", "Selecione ao menos 1 poço.")
                return
            for w in selected:
                self.show_well_comparison_report(w, model_key)
            dlg.accept()

        btns.accepted.connect(_open)
        btns.rejected.connect(dlg.reject)

        dlg.exec_()

    def init_tree_context_menu(self):
        self.project_tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.project_tree.customContextMenuRequested.connect(self.on_tree_context_menu)

    def on_tree_context_menu(self, pos):
        item = self.project_tree.itemAt(pos)
        if not item:
            return

        role = item.data(0, QtCore.Qt.UserRole)
        menu = QtWidgets.QMenu(self.project_tree)

        def _selected_wells():
            out = []
            for it in self.project_tree.selectedItems():
                if it.data(0, QtCore.Qt.UserRole) == "well_item":
                    out.append(it.data(0, QtCore.Qt.UserRole + 1))
            # se não tiver multiselect, usa o item clicado
            if not out and role == "well_item":
                out = [item.data(0, QtCore.Qt.UserRole + 1)]
            return out

        def _open_reports(model_key: str):
            for w in _selected_wells():
                self.show_well_comparison_report(w, model_key)

        if role == "well_item":
            # Abrir no modelo ativo
            act_open_current = menu.addAction("Abrir relatório (modelo atual)")
            act_open_current.triggered.connect(lambda: _open_reports(self.state.get("active_model_key", "base")))

            # Submenu: escolher modelo
            sub = menu.addMenu("Abrir relatório para…")
            act_base = sub.addAction("Modelo Base")
            act_base.triggered.connect(lambda: _open_reports("base"))

            # modelos comparados
            for mk in self.models.keys():
                if mk == "base":
                    continue
                sub_act = sub.addAction(f"{self.models[mk].get('name', mk)}")
                sub_act.triggered.connect(lambda _=False, mk=mk: _open_reports(mk))

            menu.addSeparator()

            # Toggle visibilidade no 3D
            act_toggle = menu.addAction("Mostrar/Ocultar no 3D (checkbox)")
            def _toggle():
                # alterna apenas o item clicado
                st = item.checkState(0)
                item.setCheckState(0, QtCore.Qt.Unchecked if st == QtCore.Qt.Checked else QtCore.Qt.Checked)
                self.update_wells_3d()
            act_toggle.triggered.connect(_toggle)

        menu.exec_(self.project_tree.viewport().mapToGlobal(pos))

    def evaluate_models_against_wells(
        self,
        *,
        well_names=None,
        model_keys=None,
        window_size=1,
        n_bins=200,
        w_strat=0.7,
        w_thick=0.3,
        ignore_real_zeros=True,
        use_kappa=True,
    ):
        import numpy as np
        from analysis import compute_well_match_score
        from load_data import grid as global_grid, facies as global_facies # Import para fallback

        if not self.wells: return []

        # 1. Prepara Poços
        if well_names is None: well_names = list(self.wells.keys())
        else: well_names = [w for w in well_names if w in self.wells]
        if not well_names: return []

        # 2. Prepara Modelos
        if model_keys is None:
            model_keys = list(self.models.keys())
            # Garante base se nenhum filtro for passado
            if "base" in self.models and "base" not in model_keys:
                model_keys.append("base")
        
        # Filtra apenas chaves que existem
        model_keys = [k for k in model_keys if k in self.models]
        if not model_keys: return []

        # Garante window_size ímpar
        try: window_size = int(window_size)
        except: window_size = 1
        if window_size < 1: window_size = 1
        if window_size % 2 == 0: window_size += 1

        results = []

        for mk in model_keys:
            m = self.models.get(mk, {})
            
            # --- CORREÇÃO: Resolve Grid e Facies do Base ---
            g = m.get("grid", None)
            fac = m.get("facies", None)

            # Se for base e não tiver grid no dict, usa o global
            if mk == "base":
                if g is None: g = global_grid
                if fac is None: fac = global_facies

            # Se mesmo assim não tiver grid, pula
            if g is None:
                continue

            # Injeta Facies no Grid se necessário (para garantir consistência)
            if fac is not None:
                try:
                    # Verifica se precisa atualizar (evita overhead desnecessário)
                    current_f = g.cell_data.get("Facies")
                    if current_f is None or current_f is not fac:
                        g.cell_data["Facies"] = np.asarray(fac).astype(int)
                except Exception: pass

            per_well = {}
            score_list = []
            w_list = []

            for wn in well_names:
                well = self.wells.get(wn)
                if well is None or well.data is None or well.data.empty:
                    continue

                # REAL: Preparação
                if "DEPT" not in well.data.columns: continue

                col_real = None
                if "fac" in well.data.columns: col_real = "fac"
                elif "lito_upscaled" in well.data.columns: col_real = "lito_upscaled"
                else: continue

                full_depth = well.data["DEPT"].to_numpy(dtype=float)
                full_real  = well.data[col_real].to_numpy(dtype=float)

                # Filtro por Marcadores (Real)
                key = str(wn).strip()
                markers = self.markers_db.get(key, [])
                real_depth = full_depth
                real_fac   = np.where(np.isfinite(full_real), full_real, 0.0).astype(int)

                if markers:
                    mds = sorted([mm.get("md") for mm in markers if mm.get("md") is not None])
                    if len(mds) >= 2:
                        top_md, base_md = float(mds[0]), float(mds[-1])
                        dmin, dmax = float(full_depth.min()), float(full_depth.max())
                        if (top_md <= dmax + 1e-6) and (base_md >= dmin - 1e-6) and (base_md > top_md):
                            mask = (full_depth >= top_md) & (full_depth <= base_md)
                            if np.any(mask):
                                real_depth = full_depth[mask]
                                real_fac   = real_fac[mask]

                # (X,Y) de referência
                xy = self._pick_reference_xy_for_well_report(well, markers)
                if xy is None: continue
                xref, yref = xy

                # --------- CÁLCULO (1x1 ou NxN) ---------
                if window_size == 1:
                    # 1x1: coluna mais próxima
                    sim_depth, sim_fac, _ = self._column_profile_from_grid(g, xref, yref)
                    if sim_depth is None or len(sim_depth) < 2:
                        continue

                    s = compute_well_match_score(
                        real_depth, real_fac,
                        sim_depth,  sim_fac,
                        n_bins=n_bins,
                        w_strat=w_strat,
                        w_thick=w_thick,
                        ignore_real_zeros=ignore_real_zeros,
                        use_kappa=use_kappa,
                    )
                    # Adiciona dados espaciais dummy para compatibilidade
                    s["best_i"], s["best_j"] = self._get_ij_from_xy(g, xref, yref)

                else:
                    # NxN: varredura
                    sim_depth, sim_fac, sim_total, i_best, j_best, s = self._best_profile_score_in_window(
                        g,
                        xref, yref,
                        real_depth, real_fac,
                        window_size=window_size,
                        n_bins=n_bins,
                        w_strat=w_strat,
                        w_thick=w_thick,
                        ignore_real_zeros=ignore_real_zeros,
                        use_kappa=use_kappa,
                    )
                    if sim_depth is None or len(sim_depth) < 2:
                        continue
                    
                    # Salva coordenadas do melhor match
                    s = dict(s)
                    s["best_i"] = i_best
                    s["best_j"] = j_best
                # ----------------------------------------

                weight = max(int(s.get("n_valid_bins", 0)), 0)
                per_well[str(wn)] = s
                score_list.append(float(s.get("score", 0.0)))
                w_list.append(weight)

            if not w_list or sum(w_list) <= 0:
                continue

            score_model = float(np.average(np.asarray(score_list, dtype=float),
                                           weights=np.asarray(w_list, dtype=float)))

            results.append({
                "model_key": mk,
                "model_name": str(m.get("name", mk)),
                "score": score_model,
                "n_wells_used": int(len(w_list)),
                "details": per_well,
            })

        results.sort(key=lambda d: d["score"], reverse=True)
        return results

    
    def show_models_well_fit_ranking(self):
        from PyQt5 import QtWidgets

        ws = int(getattr(self, "well_rank_window_size", 1) or 1)

        ranking = self.evaluate_models_against_wells(
            window_size=ws,
            n_bins=200,
            w_strat=0.7,
            w_thick=0.3,
            ignore_real_zeros=True,
            use_kappa=True,
        )

        if not ranking:
            QtWidgets.QMessageBox.warning(
                self,
                "Ranking modelos x poços",
                "Não consegui calcular ranking.\n"
                "Verifique se há modelos com grid, poços com DEPT e fac/fácies, e dados válidos."
            )
            return

        self.open_models_ranking_dialog(ranking)


    def open_models_ranking_dialog(self, ranking):
        """
        Abre uma janela limpa com:
        - Tabela de modelos (ranking)
        - Tabela de detalhe por poço do modelo selecionado
        - ComboBox para janela espacial (1x1, 3x3, 5x5...)
        """
        from PyQt5 import QtWidgets, QtCore

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Avaliação dos modelos vs poços")
        dlg.setMinimumSize(980, 640)

        # guarda ranking no dialog
        dlg._ranking = ranking

        layout = QtWidgets.QVBoxLayout(dlg)

        # --- Top controls (compactos) ---
        top_bar = QtWidgets.QHBoxLayout()

        lbl = QtWidgets.QLabel("Clique em um modelo para ver o detalhe por poço.")
        top_bar.addWidget(lbl)

        # ComboBox: janela espacial do ranking
        top_bar.addSpacing(12)
        top_bar.addWidget(QtWidgets.QLabel("Janela:"))

        cmb_window = QtWidgets.QComboBox()
        cmb_window.addItems(["1x1", "3x3", "5x5", "7x7", "9x9"])
        top_bar.addWidget(cmb_window)

        top_bar.addStretch(1)

        btn_copy = QtWidgets.QPushButton("Copiar tabela (modelos)")
        btn_copy.clicked.connect(lambda: self._copy_models_table_to_clipboard(dlg))
        top_bar.addWidget(btn_copy)

        layout.addLayout(top_bar)

        # --- Splitter (modelos em cima, poços embaixo) ---
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        layout.addWidget(splitter)

        # Tabela 1: Modelos
        tbl_models = QtWidgets.QTableWidget()
        tbl_models.setColumnCount(6)
        tbl_models.setHorizontalHeaderLabels([
            "Rank", "Modelo", "Score", "Fácies (acc)", "Fácies (kappa)", "Poços"
        ])
        tbl_models.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        tbl_models.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        tbl_models.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        tbl_models.setSortingEnabled(True)
        tbl_models.horizontalHeader().setStretchLastSection(True)
        tbl_models.verticalHeader().setVisible(False)

        splitter.addWidget(tbl_models)

        # Tabela 2: Detalhe por poço
        tbl_wells = QtWidgets.QTableWidget()
        tbl_wells.setColumnCount(7)
        tbl_wells.setHorizontalHeaderLabels([
            "Poço", "Score", "Fácies (acc)", "Fácies (kappa)", "Espessura", "T_real", "T_sim"
        ])
        tbl_wells.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        tbl_wells.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        tbl_wells.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        tbl_wells.setSortingEnabled(True)
        tbl_wells.horizontalHeader().setStretchLastSection(True)
        tbl_wells.verticalHeader().setVisible(False)

        splitter.addWidget(tbl_wells)

        # guarda refs no dialog
        dlg._tbl_models = tbl_models
        dlg._tbl_wells = tbl_wells
        dlg._cmb_rank_window = cmb_window

        # seta combo para o window_size atual
        ws = int(getattr(self, "well_rank_window_size", 1) or 1)
        if ws not in (1, 3, 5, 7, 9):
            ws = 1
        cmb_window.setCurrentText(f"{ws}x{ws}")

        # função interna: recalcula ranking com o window_size selecionado
        def _recompute():
            # lê window_size do combo
            try:
                txt = cmb_window.currentText()
                ws2 = int(txt.split("x")[0])
            except Exception:
                ws2 = 1

            self.well_rank_window_size = ws2

            new_ranking = self.evaluate_models_against_wells(
                window_size=ws2,
                n_bins=200,
                w_strat=0.7,
                w_thick=0.3,
                ignore_real_zeros=True,
                use_kappa=True,
            )
            dlg._ranking = new_ranking

            # repopula tabelas
            self._populate_models_ranking_table(dlg)
            if tbl_models.rowCount() > 0:
                tbl_models.selectRow(0)

        # preenche inicialmente
        self._populate_models_ranking_table(dlg)

        # seleção -> detalhe
        tbl_models.itemSelectionChanged.connect(lambda: self._on_models_table_selection_changed(dlg))

        # troca de janela -> recalcula ranking
        cmb_window.currentIndexChanged.connect(lambda *_: _recompute())

        # seleciona o 1º automaticamente
        if tbl_models.rowCount() > 0:
            tbl_models.selectRow(0)

        dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        dlg.show()

        # guarda referência para evitar GC
        self.open_reports.append(dlg)


    def _populate_models_ranking_table(self, dlg):
        from PyQt5 import QtWidgets, QtCore

        ranking = getattr(dlg, "_ranking", [])
        tbl = dlg._tbl_models
        tbl.setRowCount(0)

        for i, r in enumerate(ranking, start=1):
            row = tbl.rowCount()
            tbl.insertRow(row)

            # helpers de item
            def item(text, align_right=False):
                it = QtWidgets.QTableWidgetItem(str(text))
                if align_right:
                    it.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
                return it

            # Rank
            it_rank = QtWidgets.QTableWidgetItem(f"{i:02d}")
            it_rank.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            it_rank.setData(QtCore.Qt.UserRole, r.get("model_key"))  # guardo model_key aqui
            tbl.setItem(row, 0, it_rank)

            # Modelo
            tbl.setItem(row, 1, QtWidgets.QTableWidgetItem(r.get("model_name", "")))

            # Score
            it_score = QtWidgets.QTableWidgetItem(f"{r.get('score', 0.0):.3f}")
            it_score.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            tbl.setItem(row, 2, it_score)

            # Para mostrar médias de facies/espessura do modelo (a partir dos poços)
            details = r.get("details", {}) or {}
            accs = []
            kappas = []
            thks = []
            for _, s in details.items():
                accs.append(float(s.get("strat_acc", 0.0)))
                kappas.append(float(s.get("strat_kappa_norm", s.get("strat_kappa", 0.0))))
                thks.append(float(s.get("thick_score", 0.0)))

            mean_acc = sum(accs) / len(accs) if accs else 0.0
            mean_kap = sum(kappas) / len(kappas) if kappas else 0.0
            mean_thk = sum(thks) / len(thks) if thks else 0.0

            it_acc = QtWidgets.QTableWidgetItem(f"{mean_acc:.3f}")
            it_acc.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            tbl.setItem(row, 3, it_acc)

            it_kap = QtWidgets.QTableWidgetItem(f"{mean_kap:.3f}")
            it_kap.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            tbl.setItem(row, 4, it_kap)

            tbl.setItem(row, 5, QtWidgets.QTableWidgetItem(str(r.get("n_wells_used", 0))))

        tbl.resizeColumnsToContents()

    def _on_models_table_selection_changed(self):
        """Callback quando o usuário clica num modelo na tabela de ranking."""
        sel = self.tbl_models.selectedItems()
        if not sel: return

        row = self.tbl_models.currentRow()
        if row < 0: return

        # model_key está no item Rank (col 0)
        it_rank = self.tbl_models.item(row, 0)
        model_key = it_rank.data(QtCore.Qt.UserRole)

        # Encontra dados no cache
        ranking = getattr(self, "_current_ranking_data", [])
        rec = None
        for r in ranking:
            if str(r.get("model_key")) == str(model_key):
                rec = r
                break

        if rec:
            self._populate_wells_detail_table(rec)

    def _populate_wells_detail_table(self, model_record):
        """Popula a tabela inferior de poços para o modelo selecionado."""
        self.tbl_wells.setRowCount(0)
        
        model_key = model_record.get("model_key")
        details = model_record.get("details", {}) or {}
        
        items = list(details.items())
        items.sort(key=lambda kv: float(kv[1].get("score", 0.0)), reverse=True)

        for well_name, s in items:
            row = self.tbl_wells.rowCount()
            self.tbl_wells.insertRow(row)

            # Dados Numéricos
            self.tbl_wells.setItem(row, 0, QtWidgets.QTableWidgetItem(str(well_name)))
            self.tbl_wells.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{float(s.get('score',0)):.3f}"))
            self.tbl_wells.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{float(s.get('strat_acc',0)):.3f}"))
            self.tbl_wells.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{float(s.get('strat_kappa_norm',0)):.3f}"))
            self.tbl_wells.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{float(s.get('thick_score',0)):.3f}"))
            self.tbl_wells.setItem(row, 5, QtWidgets.QTableWidgetItem(f"{float(s.get('t_real',0)):.2f}"))
            self.tbl_wells.setItem(row, 6, QtWidgets.QTableWidgetItem(f"{float(s.get('t_sim',0)):.2f}"))

            # --- COLUNA DE AÇÕES ---
            widget = QtWidgets.QWidget()
            h_lay = QtWidgets.QHBoxLayout(widget)
            h_lay.setContentsMargins(4, 2, 4, 2)
            h_lay.setSpacing(6)
            
            best_i = s.get("best_i")
            best_j = s.get("best_j")
            
            # Botão Gráfico (Lupa/Lista) - Abre relatório detalhado
            btn_graph = QtWidgets.QPushButton()
            btn_graph.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogContentsView))
            btn_graph.setToolTip("Relatório Comparativo (Base vs Origem vs Melhor da Janela)")
            btn_graph.setFixedSize(30, 24)
            btn_graph.clicked.connect(lambda _, mk=model_key, wn=well_name, bi=best_i, bj=best_j: 
                self.open_advanced_rank_report(mk, wn, bi, bj))

            h_lay.addWidget(btn_graph)
            h_lay.addStretch(1) 
            
            self.tbl_wells.setCellWidget(row, 7, widget)
        
        self.tbl_wells.resizeColumnsToContents()

    def _best_profile_score_in_window(
        self,
        grid,
        xref, yref,
        real_depth, real_fac,
        *,
        window_size=1,     # 1,3,5,7...
        n_bins=200,
        w_strat=0.7,
        w_thick=0.3,
        ignore_real_zeros=True,
        use_kappa=True,
    ):
        """
        Retorna o melhor match REAL vs pseudo-poço do grid em uma janela NxN.

        Saída (sempre 6 itens):
        sim_depth, sim_fac, sim_total, i_best, j_best, fit_best

        - Se window_size=1: compara somente a coluna central (1x1).
        - Se window_size=3: varre 3x3 e pega o melhor score.
        - Se window_size=5: varre 5x5, etc.
        """
        import numpy as np
        from analysis import compute_well_match_score  # ou compute_well_fit_score, se for o seu nome

        if grid is None:
            return np.array([]), np.array([]), 0.0, None, None, {"score": 0.0}

        ij = self._get_ij_from_xy(grid, xref, yref)
        if ij is None:
            return np.array([]), np.array([]), 0.0, None, None, {"score": 0.0}

        i0, j0 = ij

        # garante ímpar >=1
        window_size = int(window_size)
        if window_size < 1:
            window_size = 1
        if window_size % 2 == 0:
            window_size += 1

        half = window_size // 2

        best_fit = None
        best_depth = None
        best_fac = None
        best_total = 0.0
        best_i = None
        best_j = None

        for di in range(-half, half + 1):
            for dj in range(-half, half + 1):
                ii = i0 + di
                jj = j0 + dj

                sim_depth, sim_fac, sim_total = self._column_profile_from_grid_ij(grid, ii, jj)
                if sim_depth is None or len(sim_depth) < 2:
                    continue

                fit = compute_well_match_score(
                    real_depth, real_fac,
                    sim_depth, sim_fac,
                    n_bins=n_bins,
                    w_strat=w_strat,
                    w_thick=w_thick,
                    ignore_real_zeros=ignore_real_zeros,
                    use_kappa=use_kappa,
                )

                if best_fit is None or float(fit.get("score", 0.0)) > float(best_fit.get("score", 0.0)):
                    best_fit = fit
                    best_depth = sim_depth
                    best_fac = sim_fac
                    best_total = float(sim_total) if np.isfinite(sim_total) else 0.0
                    best_i, best_j = int(ii), int(jj)

        if best_fit is None:
            return np.array([]), np.array([]), 0.0, int(i0), int(j0), {"score": 0.0}

        return best_depth, best_fac, best_total, best_i, best_j, best_fit


    def _copy_models_table_to_clipboard(self, dlg):
        from PyQt5 import QtWidgets

        tbl = dlg._tbl_models
        rows = tbl.rowCount()
        cols = tbl.columnCount()

        headers = [tbl.horizontalHeaderItem(c).text() for c in range(cols)]
        lines = ["\t".join(headers)]

        for r in range(rows):
            vals = []
            for c in range(cols):
                it = tbl.item(r, c)
                vals.append(it.text() if it else "")
            lines.append("\t".join(vals))

        QtWidgets.QApplication.clipboard().setText("\n".join(lines))

    def _best_profile_score_in_window_3x3(self, grid, xref, yref, real_depth, real_fac, **kwargs):
        return self._best_profile_score_in_window(
            grid, xref, yref, real_depth, real_fac,
            window_size=3,
            **kwargs
        )

    
    def _best_profile_in_window_3x3(self, grid, x, y, real_depth, real_fac, *, n_bins=200):
        """
        Varre uma janela 3x3 em torno da coluna mais próxima de (x,y) e retorna
        o perfil do grid (pseudo-poço) que MAIS se parece com o poço real.

        Critério: compute_well_fit_score (score final que você já usa no ranking),
        comparando (real_depth/real_fac) vs (sim_depth/sim_fac).
        """
        import numpy as np
        from analysis import compute_well_fit_score  # função do seu analysis.py

        # pega a coluna central (a mais próxima)
        _, _, _, ic, jc = self._column_profile_from_grid(grid, x, y, return_ij=True)

        if ic is None or jc is None:
            return np.array([]), np.array([]), 0.0, None, None, {"score": 0.0}

        best = None
        best_score = -1.0

        # offsets da janela 3x3
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                i0 = ic + di
                j0 = jc + dj

                sim_depth, sim_fac, sim_total = self._column_profile_from_grid(
                    grid, x, y, i0=i0, j0=j0, return_ij=False
                )

                if sim_depth is None or len(sim_depth) < 2:
                    continue

                fit = compute_well_fit_score(
                    real_depth=real_depth,
                    real_facies=real_fac,
                    sim_depth=sim_depth,
                    sim_facies=sim_fac,
                    n_bins=n_bins,
                    use_kappa=True,
                    ignore_real_zeros=True,
                    w_strat=0.9,
                    w_thick=0.1,
                )

                score = float(fit.get("score", 0.0))
                if score > best_score:
                    best_score = score
                    best = (sim_depth, sim_fac, sim_total, i0, j0, fit)

        if best is None:
            return np.array([]), np.array([]), 0.0, ic, jc, {"score": 0.0}

        return best
    
    def draw_search_window_3d(self, model_key, well_name, _unused_i, _unused_j, best_i, best_j, window_size):
        """
        Visualização 3D Individual (Ranking Report).
        """
        # Desmarca o botão global para evitar conflito visual
        if hasattr(self, "btn_debug_all"):
            self.btn_debug_all.setChecked(False)

        # Toggle Lógica
        current_debug_key = (model_key, well_name)
        last_debug_key = getattr(self, "_last_debug_key", None)
        
        if hasattr(self, "_debug_actors"):
            for a in self._debug_actors:
                try: self.plotter.remove_actor(a)
                except: pass
        self._debug_actors = []
        
        main_actor = self.state.get("main_actor")
        if main_actor: main_actor.GetProperty().SetOpacity(1.0)
        self.plotter.render()

        if last_debug_key == current_debug_key:
            self._last_debug_key = None
            return

        self._last_debug_key = current_debug_key

        # Setup
        self.switch_main_view_to_model(model_key)
        if hasattr(self, "compare_stack") and self.central_stack.currentIndex() == 1:
            self.compare_stack.setCurrentIndex(0)
        elif hasattr(self, "viz_container"):
            self.viz_container.setCurrentIndex(0)

        grid = self.state.get("current_grid_source")
        if grid is None: return
        z_exag = float(self.state.get("z_exag", 15.0))
        
        main_actor = self.state.get("main_actor")
        scale_z = main_actor.GetScale()[2] if main_actor else 1.0
        if main_actor: main_actor.GetProperty().SetOpacity(0.001)

        # Chama a auxiliar
        new_actors = self._create_well_debug_actors(
            grid, well_name, best_i, best_j, window_size, z_exag, scale_z
        )
        self._debug_actors.extend(new_actors)
        
        self.plotter.render()

    def open_advanced_rank_report(self, model_key, well_name, best_i, best_j):
        """
        Relatório de Ranking Detalhado - Design 'Slim'.
        """
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.patches import Rectangle
        from config import load_facies_colors
        import numpy as np

        # --- COLETA DE DADOS (Mantida igual) ---
        well = self.wells.get(well_name)
        if not well: return
        if "DEPT" not in well.data.columns: return
        
        col_real = "fac" if "fac" in well.data.columns else "lito_upscaled"
        raw_depth = well.data["DEPT"].to_numpy()
        raw_fac = well.data[col_real].fillna(0).to_numpy()

        markers = self.markers_db.get(well_name, [])
        mds = sorted([m['md'] for m in markers if 'md' in m])
        if len(mds) >= 2:
            top_md, base_md = mds[0], mds[-1]
            mask = (raw_depth >= top_md) & (raw_depth <= base_md)
            if mask.any():
                real_depth = raw_depth[mask]; real_fac = raw_fac[mask]
            else: real_depth, real_fac = raw_depth, raw_fac
        else:
            valid_mask = (raw_fac > 0)
            if valid_mask.any():
                start = np.argmax(valid_mask)
                end = len(valid_mask) - np.argmax(valid_mask[::-1])
                real_depth = raw_depth[start:end]; real_fac = raw_fac[start:end]
            else: real_depth, real_fac = raw_depth, raw_fac

        if len(real_depth) == 0: return

        ref_top = real_depth[0]
        ref_base = real_depth[-1]
        total_thick = ref_base - ref_top

        grid_base = self.models.get("base", {}).get("grid")
        if not grid_base: from load_data import grid as grid_base
        grid_sim = self.models[model_key].get("grid")
        if not grid_sim: return

        wx = float(well.data["X"].mean())
        wy = float(well.data["Y"].mean())

        def extract_and_align(g, i=None, j=None):
            d, f, _ = self._column_profile_from_grid(g, wx, wy, i0=i, j0=j)
            if len(d) == 0: return [], []
            return d + ref_top, f

        db_d, db_f = extract_and_align(grid_base)
        dso_d, dso_f = extract_and_align(grid_sim)
        dsb_d, dsb_f = extract_and_align(grid_sim, i=best_i, j=best_j)

        max_d = ref_base
        if len(db_d) > 0: max_d = max(max_d, db_d[-1])
        if len(dso_d) > 0: max_d = max(max_d, dso_d[-1])
        if len(dsb_d) > 0: max_d = max(max_d, dsb_d[-1])

        # --- PLOTAGEM (AJUSTES DE TAMANHO AQUI) ---
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Ranking Detalhado: {well_name}")
        
        # 1. TAMANHO DA JANELA (LARGURA, ALTURA) em Pixels
        # Mude o 550 para menos se quiser a janela mais fina, ou mais se quiser larga
        dialog.resize(550, 850) 
        
        layout = QtWidgets.QVBoxLayout(dialog)

        # 2. PROPORÇÃO DO GRÁFICO (LARGURA, ALTURA) em Polegadas
        # figsize=(4, 9) cria colunas bem estreitas. 
        # Se quiser poços mais gordos, aumente o 4 para 6 ou 8.
        fig, axs = plt.subplots(1, 4, figsize=(4, 9), sharey=True)
        
        # Ajuste de margens internas (wspace=0.6 separa os poços)
        fig.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.03, wspace=0.6)

        colors = load_facies_colors()
        def get_c(c): return colors.get(int(c), (0.8,0.8,0.8,1.0))

        def group_layers(depth, facies, is_grid_format=True):
            if len(depth) == 0: return []
            layers = []
            if is_grid_format:
                for k in range(0, len(depth)-1, 2):
                    layers.append((depth[k], depth[k+1], int(facies[k])))
            else:
                current_top = depth[0]
                current_fac = int(facies[0])
                for k in range(1, len(facies)):
                    if int(facies[k]) != current_fac:
                        layers.append((current_top, depth[k], current_fac))
                        current_top = depth[k]
                        current_fac = int(facies[k])
                layers.append((current_top, depth[-1], current_fac))
            
            merged = []
            if not layers: return []
            curr_top, curr_base, curr_fac = layers[0]
            for i in range(1, len(layers)):
                next_top, next_base, next_fac = layers[i]
                if next_fac == curr_fac and abs(next_top - curr_base) < 0.1:
                    curr_base = next_base
                else:
                    merged.append((curr_top, curr_base, curr_fac))
                    curr_top, curr_base, curr_fac = next_top, next_base, next_fac
            merged.append((curr_top, curr_base, curr_fac))
            return merged

        def plot_track(ax, d, f, title, is_grid=True):
            ax.set_title(title, fontsize=8, pad=6)
            ax.set_xticks([])
            ax.set_xlim(0, 1)
            ax.set_facecolor('white')
            
            layers = group_layers(d, f, is_grid)
            
            for top, base, fac in layers:
                h = base - top
                if h <= 0: continue
                rect = Rectangle((0, top), 1, h, facecolor=get_c(fac), edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
                
                if h > (max_d - ref_top) * 0.025:
                    lum = sum(get_c(fac)[:3])
                    txt_c = 'white' if lum < 1.5 else 'black'
                    ax.text(0.5, top + h/2, str(fac), ha='center', va='center', fontsize=7, color=txt_c, fontweight='bold')

        th_b = db_d[-1]-db_d[0] if len(db_d) else 0
        th_so = dso_d[-1]-dso_d[0] if len(dso_d) else 0
        th_sb = dsb_d[-1]-dsb_d[0] if len(dsb_d) else 0

        plot_track(axs[0], db_d, db_f, f"BASE\n{th_b:.1f}m", True)
        plot_track(axs[1], dso_d, dso_f, f"SIM (Orig)\n{th_so:.1f}m", True)
        plot_track(axs[2], dsb_d, dsb_f, f"SIM (Melhor)\n{th_sb:.1f}m", True)
        plot_track(axs[3], real_depth, real_fac, f"REAL\n{total_thick:.1f}m", False)

        axs[0].set_ylabel("Profundidade (MD)", fontsize=9)
        axs[0].set_ylim(max_d, ref_top)
        
        layout.addWidget(FigureCanvas(fig))
        dialog.exec_()

    def _get_or_create_study_item(self, study_name):
        """Encontra ou cria o item pai (Pasta/Study) na árvore."""
        root = self.project_tree.invisibleRootItem()
        
        # 1. Procura se já existe
        for i in range(root.childCount()):
            item = root.child(i)
            if item.data(0, QtCore.Qt.UserRole) == "study_folder" and item.text(0) == study_name:
                return item
        
        # 2. Se não existe, cria
        study_item = QtWidgets.QTreeWidgetItem([study_name])
        study_item.setData(0, QtCore.Qt.UserRole, "study_folder")
        study_item.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_DirIcon))
        
        # --- CORREÇÃO: Removemos ItemIsTristate para evitar comportamento automático imprevisível ---
        # Usamos apenas ItemIsUserCheckable. Nós gerenciaremos o estado visual manualmente.
        study_item.setFlags(study_item.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        study_item.setCheckState(0, QtCore.Qt.Checked)
        
        if hasattr(self, "wells_root_item") and self.wells_root_item is not None:
            idx = self.project_tree.indexOfTopLevelItem(self.wells_root_item)
            if idx >= 0:
                self.project_tree.insertTopLevelItem(idx, study_item)
            else:
                self.project_tree.addTopLevelItem(study_item)
        else:
            self.project_tree.addTopLevelItem(study_item)
            
        study_item.setExpanded(True)
        return study_item
    
    def recalc_entropy_view(self):
        """Coleta modelos marcados, calcula entropia e configura o visualizador."""
        from analysis import compute_facies_entropy_map
        import numpy as np

        # 1. Identifica modelos marcados
        checked_data = self.get_checked_models()
        
        arrays = []
        for m_key, _ in checked_data:
            if m_key in self.models:
                arr = self.models[m_key].get("facies")
                if arr is not None:
                    arrays.append(arr)
        
        # 2. Define Grid Alvo (Base)
        grid_target = self.models["base"].get("grid")
        if grid_target is None:
            grid_target = self.state.get("current_grid_source")
        
        if grid_target is None: return

        # 3. Calcula Entropia
        if not arrays:
            ent_map = np.zeros(grid_target.n_cells)
        else:
            ent_map = compute_facies_entropy_map(arrays, target_grid=grid_target)

        # 4. Injeta no Grid
        scalar_name = "Entropy"
        grid_target.cell_data[scalar_name] = ent_map
        
        # 5. --- CORREÇÃO: Registra Entropia como um Preset Válido ---
        # Recupera os presets existentes (Espessura, NTG, etc)
        presets = self.state.get("thickness_presets", {})
        
        # Adiciona/Atualiza o preset 'Entropy'
        # Formato: "NomeModo": ("NomeArrayNoGrid", "Título do Gráfico")
        presets["Entropy"] = (scalar_name, f"Entropia (Incerteza) - N={len(arrays)}")
        self.state["thickness_presets"] = presets

        # 6. Configura o Estado para usar esse Preset
        self.state["current_grid_source"] = grid_target
        self.state["thickness_mode"] = "Entropy"  # <--- Isso corrige o título e o array
        
        # Configurações visuais específicas para Entropia
        vmax = np.max(ent_map) if len(ent_map) > 0 else 1.0
        if vmax == 0: vmax = 0.1
        
        self.state["thickness_clim"] = (0.0, vmax)
        self.state["thickness_cmap"] = "jet" # Mapa de cores mais intuitivo para calor/incerteza
        
        # 7. Força o modo de visualização escalar
        self.state["mode"] = "thickness_local" 
        
        # 8. Atualiza UI
        refresh = self.state.get("refresh")
        if callable(refresh): refresh()
        
        if hasattr(self, "update_2d_map"): self.update_2d_map()