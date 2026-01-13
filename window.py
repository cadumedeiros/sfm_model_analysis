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

        # --- Exibir (para reabrir docks/toolbars) ---
        self.view_menu = menubar.addMenu("Exibir")

        # Perspectivas (interno)
        self.act_persp_viz = QtWidgets.QAction("Visualização", self)
        self.act_persp_viz.setCheckable(True)
        self.act_persp_viz.setChecked(True)
        self.act_persp_viz.triggered.connect(lambda: self.switch_perspective("visualization"))

        self.act_persp_comp = QtWidgets.QAction("Comparação", self)
        self.act_persp_comp.setCheckable(True)
        self.act_persp_comp.triggered.connect(lambda: self.switch_perspective("comparison"))

        # Ribbon (cria o widget)
        self.setup_toolbar_controls()

        # >>> Ribbon em ToolBar (isso força docks abaixo do painel superior)
        self.ribbon_toolbar = QtWidgets.QToolBar("Ribbon")
        self.ribbon_toolbar.setMovable(False)
        self.ribbon_toolbar.setFloatable(False)
        self.ribbon_toolbar.setAllowedAreas(QtCore.Qt.TopToolBarArea)
        self.ribbon_toolbar.setStyleSheet("QToolBar { border: 0px; }")
        self.ribbon_toolbar.addWidget(self.ribbon)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.ribbon_toolbar)

        # Central: somente o stack (sem ribbon dentro)
        self.central_stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.central_stack)

        # Docks (Explorer/Props)
        self.setup_docks(nx, ny, nz)

        # Ações do menu Exibir (toggle docks/toolbars)
        self.view_menu.addAction(self.dock_explorer.toggleViewAction())
        self.view_menu.addAction(self.dock_props.toggleViewAction())
        self.view_menu.addSeparator()
        self.view_menu.addAction(self.ribbon_toolbar.toggleViewAction())

        # --- PERSPECTIVA 1: VISUALIZAÇÃO ---
        self.viz_container = QtWidgets.QStackedWidget()
        self.tabs = self.viz_container

        self.viz_tab = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(self.viz_tab)
        vl.setContentsMargins(0, 0, 0, 0)
        self.plotter, plotter_widget = self._make_embedded_plotter(parent=self.viz_tab)
        vl.addWidget(plotter_widget)
        self.viz_container.addWidget(self.viz_tab)

        self.map2d_tab = QtWidgets.QWidget()
        ml = QtWidgets.QVBoxLayout(self.map2d_tab)
        ml.setContentsMargins(0, 0, 0, 0)
        self.plotter_2d, plotter_2d_widget = self._make_embedded_plotter(parent=self.map2d_tab)
        ml.addWidget(plotter_2d_widget)
        self.viz_container.addWidget(self.map2d_tab)

        self.details_tab = QtWidgets.QWidget()
        l_det = QtWidgets.QVBoxLayout(self.details_tab)
        self.central_metrics_text = QtWidgets.QTextEdit()
        self.central_metrics_text.setReadOnly(True)
        self.central_metrics_text.setMaximumHeight(150)
        l_det.addWidget(QtWidgets.QLabel("Resumo Global"))
        l_det.addWidget(self.central_metrics_text)
        self.facies_table = QtWidgets.QTableWidget()
        l_det.addWidget(QtWidgets.QLabel("Detalhamento por Fácies"))
        l_det.addWidget(self.facies_table)
        self.viz_container.addWidget(self.details_tab)

        self.central_stack.addWidget(self.viz_container)

        # --- PERSPECTIVA 2: COMPARAÇÃO ---
        self.compare_stack = QtWidgets.QStackedWidget()

        self.comp_page_3d = QtWidgets.QWidget()
        self.comp_layout_3d = QtWidgets.QVBoxLayout(self.comp_page_3d)
        self.comp_layout_3d.setContentsMargins(0, 0, 0, 0)
        self.compare_stack.addWidget(self.comp_page_3d)

        self.comp_page_metrics = QtWidgets.QWidget()
        self.comp_metrics_layout = QtWidgets.QVBoxLayout(self.comp_page_metrics)
        self.comp_metrics_layout.setContentsMargins(6, 6, 6, 6)

        self.tabs_compare_metrics = QtWidgets.QTabWidget()

        t_fa = QtWidgets.QWidget()
        l_fa = QtWidgets.QVBoxLayout(t_fa)
        self.facies_compare_table = QtWidgets.QTableWidget()
        l_fa.addWidget(self.facies_compare_table)
        self.tabs_compare_metrics.addTab(t_fa, "Fácies")

        t_res = QtWidgets.QWidget()
        l_res = QtWidgets.QVBoxLayout(t_res)
        self.reservoir_facies_compare_table = QtWidgets.QTableWidget()
        l_res.addWidget(self.reservoir_facies_compare_table)
        self.tabs_compare_metrics.addTab(t_res, "Reservatório")

        self.comp_metrics_layout.addWidget(self.tabs_compare_metrics)
        self.compare_stack.addWidget(self.comp_page_metrics)

        self.comp_page_2d = QtWidgets.QWidget()
        self.comp_2d_layout = QtWidgets.QVBoxLayout(self.comp_page_2d)
        self.comp_2d_layout.setContentsMargins(0, 0, 0, 0)
        self.compare_stack.addWidget(self.comp_page_2d)

        self.central_stack.addWidget(self.compare_stack)



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
        
        old_z = self.state.get("z_exag", 1.0)
        self.state["z_exag"] = new_z
        
        # 1. Atualiza Grid (escala)
        if self.state["current_grid_source"]:
            self.state["current_grid_source"].points[:, 2] /= old_z
            self.state["current_grid_source"].points[:, 2] *= new_z
            
        # 2. Atualiza Poços (Redesenha)
        self.update_wells_3d()

    def update_wells_3d(self):
        if not hasattr(self, 'plotter'):
            return

        # Decide quais poços estão visíveis via árvore (checkbox)
        visible = set(self.wells.keys())
        if hasattr(self, "wells_root_item") and self.wells_root_item is not None:
            visible = set()
            for i in range(self.wells_root_item.childCount()):
                it = self.wells_root_item.child(i)
                if it.data(0, QtCore.Qt.UserRole) == "well_item":
                    name = it.data(0, QtCore.Qt.UserRole + 1)
                    if it.checkState(0) == QtCore.Qt.Checked:
                        visible.add(name)

        # Limpa poços antigos
        for name in list(self.wells.keys()):
            self.plotter.remove_actor(f"well_{name}")
            self.plotter.remove_actor(f"marker_{name}")
        self.plotter.remove_actor("well_labels")

        z_exag = self.state.get("z_exag", 1.0)
        lbl_pos, lbl_txt = [], []

        for name, well in self.wells.items():
            if name not in visible:
                continue

            tube = well.get_vtk_polydata(z_exag=z_exag)
            if tube:
                self.plotter.add_mesh(
                    tube,
                    scalars="Facies_Real",
                    cmap=self.pv_cmap,
                    clim=self.clim,
                    name=f"well_{name}",
                    smooth_shading=False,
                    show_scalar_bar=False,
                    interpolate_before_map=False
                )

                min_md_idx = np.argmin(well.data["DEPT"].values)
                top = well.data.iloc[min_md_idx][["X", "Y", "Z"]].values.copy()
                top[2] *= z_exag
                top[2] -= (50 * z_exag)
                lbl_pos.append(top)
                lbl_txt.append(name)

            if name in self.markers_db:
                glyphs, _ = well.get_markers_mesh(self.markers_db[name], z_exag=z_exag)
                if glyphs:
                    self.plotter.add_mesh(glyphs, color="red", name=f"marker_{name}")

        if lbl_pos:
            self.plotter.add_point_labels(
                lbl_pos, lbl_txt,
                font_size=16, text_color="black",
                point_size=0, always_visible=True,
                name="well_labels"
            )


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

        - REAL: limitado pelos marcadores (quando compatíveis).
        - BASE e SIM: pegam a coluna (i,j) mais próxima do (X,Y) de referência do poço
        e constroem o perfil topo->base usando StratigraphicThickness.
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
        base_depth, base_facies, _ = self._column_profile_from_grid(base_grid, xref, yref)
        window_size = getattr(self, "well_compare_window_size", 1)

        sim_depth, sim_facies, _, i_best, j_best, fit_best = self._best_profile_score_in_window(
            grid_sim_source,
            xref, yref,
            real_depth=real_depth0,
            real_fac=np.where(np.isfinite(real_facies0), real_facies0, 0.0).astype(int),
            window_size=window_size,
            n_bins=200,
            w_strat=0.7,
            w_thick=0.3,
            ignore_real_zeros=True,
            use_kappa=True,
        )

        # REAL: não deixar NaN virar lixo
        real_depth = real_depth0
        real_facies = np.where(np.isfinite(real_facies0), real_facies0, 0.0).astype(int)

        # --- cria dialog não-modal (permite vários abertos) ---
        report_dialog = self._open_matplotlib_report(
            well_name=well_name,
            sim_model_name=sim_model_name,
            real_depth=real_depth, real_fac=real_facies,
            base_depth=base_depth, base_fac=base_facies,
            sim_depth=sim_depth, sim_fac=sim_facies
        )

        # importante: não modal + auto-destruir ao fechar
        report_dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        report_dialog.show()

        # guarda referência (evita GC + você pode gerenciar janelas abertas)
        self.open_reports.append(report_dialog)

        def _cleanup():
            try:
                self.open_reports = [d for d in self.open_reports if d is not report_dialog]
            except Exception:
                pass

        report_dialog.destroyed.connect(_cleanup)


    def setup_comparison_3d_view(self, container):
        """Prepara o container para receber o grid dinâmico."""
        # Apenas define um layout base. O conteúdo será injetado por update_dynamic_comparison_view
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Placeholder inicial (opcional)
        label = QtWidgets.QLabel("Selecione os modelos na árvore para comparar.")
        label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(label)

    def setup_toolbar_controls(self):
        """Ribbon (estilo Office) com grupos discretos."""
        self.ribbon = QtWidgets.QTabWidget()
        self.ribbon.setDocumentMode(True)
        self.ribbon.setMovable(False)
        self.ribbon.setUsesScrollButtons(True)
        self.ribbon.setElideMode(QtCore.Qt.ElideRight)
        self.ribbon.setMaximumHeight(130)

        # >>> Títulos dos grupos discretos (sem negrito, menor)
        self.ribbon.setStyleSheet("""
            QTabWidget::pane { border: 0px; }
            QTabBar::tab { padding: 6px 12px; }
            QGroupBox {
                font-weight: normal;
                font-size: 11px;
                margin-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0px 2px;
                color: #444;
            }
        """)

        def make_tab():
            w = QtWidgets.QWidget()
            l = QtWidgets.QHBoxLayout(w)
            l.setContentsMargins(10, 8, 10, 8)
            l.setSpacing(12)
            return w, l

        def make_group(title):
            gb = QtWidgets.QGroupBox(title)
            gb.setFlat(True)
            gl = QtWidgets.QHBoxLayout(gb)
            gl.setContentsMargins(10, 6, 10, 6)
            gl.setSpacing(8)
            return gb, gl

        def make_btn(text, std_icon, slot=None, checkable=False):
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.setIcon(self.style().standardIcon(std_icon))
            b.setIconSize(QtCore.QSize(26, 26))
            b.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
            b.setAutoRaise(True)
            b.setCheckable(checkable)
            if slot is not None:
                b.clicked.connect(slot)
            return b

        # HOME
        self.ribbon_tab_home, l_home = make_tab()

        gb_data, gl_data = make_group("Dados")
        gl_data.addWidget(make_btn("Modelo", QtWidgets.QStyle.SP_DialogOpenButton, self.open_compare_dialog))
        gl_data.addWidget(make_btn("Poços", QtWidgets.QStyle.SP_FileDialogStart, self.load_well_dialog))
        l_home.addWidget(gb_data)

        gb_persp, gl_persp = make_group("Perspectiva")
        self.btn_persp_viz = make_btn("Visualizar", QtWidgets.QStyle.SP_DesktopIcon,
                                    lambda: self.switch_perspective("visualization"), checkable=True)
        self.btn_persp_comp = make_btn("Comparar", QtWidgets.QStyle.SP_DirLinkIcon,
                                    lambda: self.switch_perspective("comparison"), checkable=True)
        self._persp_group = QtWidgets.QButtonGroup(self)
        self._persp_group.setExclusive(True)
        self._persp_group.addButton(self.btn_persp_viz)
        self._persp_group.addButton(self.btn_persp_comp)
        self.btn_persp_viz.setChecked(True)
        gl_persp.addWidget(self.btn_persp_viz)
        gl_persp.addWidget(self.btn_persp_comp)
        l_home.addWidget(gb_persp)

        gb_tools, gl_tools = make_group("Ferramentas")
        gl_tools.addWidget(make_btn("Snapshot", QtWidgets.QStyle.SP_DialogSaveButton, self.take_snapshot))
        l_home.addWidget(gb_tools)

        l_home.addStretch(1)
        self.ribbon.addTab(self.ribbon_tab_home, "Home")

        # VIEW
        self.ribbon_tab_view, l_view = make_tab()
        gb_mode, gl_mode = make_group("Modo")
        self.btn_mode = QtWidgets.QToolButton(self)
        self.btn_mode.setText("Modo: Fácies")
        self.btn_mode.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon))
        self.btn_mode.setIconSize(QtCore.QSize(26, 26))
        self.btn_mode.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.btn_mode.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.btn_mode.setAutoRaise(True)
        menu_mode = QtWidgets.QMenu(self.btn_mode)
        for text, data in [("Fácies", "facies"), ("Reservatório", "reservoir"), ("Clusters", "clusters"),
                        ("Maior Cluster", "largest"), ("Espessura Local", "thickness_local")]:
            a = menu_mode.addAction(text)
            a.triggered.connect(lambda ch, t=text, d=data: self._update_mode_btn(t, d))
        self.btn_mode.setMenu(menu_mode)
        gl_mode.addWidget(self.btn_mode)
        l_view.addWidget(gb_mode)

        gb_thick, gl_thick = make_group("Espessura")
        self.btn_thick = QtWidgets.QToolButton(self)
        self.btn_thick.setText("Espessura: Espessura")
        self.btn_thick.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView))
        self.btn_thick.setIconSize(QtCore.QSize(26, 26))
        self.btn_thick.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.btn_thick.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.btn_thick.setAutoRaise(True)
        menu_thick = QtWidgets.QMenu(self.btn_thick)
        for label in ["Espessura", "NTG coluna", "NTG envelope", "Maior pacote", "Nº pacotes", "ICV", "Qv", "Qv absoluto"]:
            a = menu_thick.addAction(label)
            a.triggered.connect(lambda ch, l=label: self._update_thick_btn(l))
        self.btn_thick.setMenu(menu_thick)
        gl_thick.addWidget(self.btn_thick)
        l_view.addWidget(gb_thick)

        l_view.addStretch(1)
        self.ribbon.addTab(self.ribbon_tab_view, "View")

        # COMPARE + REPORTS (mantém seu conteúdo atual)
        self.ribbon_tab_compare, l_cmp = make_tab()
        gb_cmp_view, gl_cmp_view = make_group("Vista")
        self.lbl_comp_view = QtWidgets.QLabel("Vista:")
        self.combo_comp_view = QtWidgets.QComboBox()
        self.combo_comp_view.addItems(["Visualização 3D", "Métricas Comparadas", "Mapas 2D"])
        self.combo_comp_view.currentIndexChanged.connect(self.on_comp_view_changed)
        gl_cmp_view.addWidget(self.lbl_comp_view)
        gl_cmp_view.addWidget(self.combo_comp_view)
        l_cmp.addWidget(gb_cmp_view)
        gb_cmp_actions, gl_cmp_actions = make_group("Ações")
        gl_cmp_actions.addWidget(make_btn("Atualizar", QtWidgets.QStyle.SP_BrowserReload, self.refresh_comparison_active_view))
        l_cmp.addWidget(gb_cmp_actions)
        l_cmp.addStretch(1)
        self.ribbon.addTab(self.ribbon_tab_compare, "Compare")

        self.ribbon_tab_reports, l_rep = make_tab()
        gb_reports, gl_reports = make_group("Relatórios")
        gl_reports.addWidget(make_btn("Abrir...", QtWidgets.QStyle.SP_DialogOpenButton, self.open_reports_dialog))
        gl_reports.addWidget(make_btn("Selecionados", QtWidgets.QStyle.SP_FileDialogContentsView, self.open_selected_well_reports))
        l_rep.addWidget(gb_reports)
        gb_rank, gl_rank = make_group("Score")
        gl_rank.addWidget(make_btn("Ranking", QtWidgets.QStyle.SP_ArrowUp, self.show_models_well_fit_ranking))
        l_rep.addWidget(gb_rank)
        l_rep.addStretch(1)
        self.ribbon.addTab(self.ribbon_tab_reports, "Reports")




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

    # Helpers para atualizar o texto do botão quando clica no menu
    def _update_mode_btn(self, text, data):
        self.btn_mode.setText(f"Modo: {text}")
        self.change_mode(data)

    def _update_thick_btn(self, label):
        self.btn_thick.setText(f"Espessura: {label}")
        self.change_thickness_mode(label)

    def setup_docks(self, nx, ny, nz):
        """Cria o Project Explorer (esquerda) e o Inspector (direita)."""
        # ---------------------------
        # Dock ESQUERDO: Project Explorer
        # ---------------------------
        self.dock_explorer = QtWidgets.QDockWidget("Project Explorer", self)
        self.dock_explorer.setObjectName("dock_explorer")
        self.dock_explorer.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.dock_explorer.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFloatable |
            QtWidgets.QDockWidget.DockWidgetClosable
        )

        explorer_widget = QtWidgets.QWidget()
        explorer_layout = QtWidgets.QVBoxLayout(explorer_widget)
        explorer_layout.setContentsMargins(6, 6, 6, 6)

        self.project_tree = QtWidgets.QTreeWidget()
        self.project_tree.setHeaderLabel("Hierarquia")
        self.project_tree.itemSelectionChanged.connect(self.on_tree_selection_changed)
        self.project_tree.itemChanged.connect(self.on_tree_item_changed)
        self.project_tree.itemDoubleClicked.connect(self.on_tree_double_clicked)

        explorer_layout.addWidget(self.project_tree)
        self.dock_explorer.setWidget(explorer_widget)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dock_explorer)

        # ---------------------------
        # Dock DIREITO: Inspector (Geometria + Propriedades + Compare)
        # ---------------------------
        self.dock_props = QtWidgets.QDockWidget("Propriedades", self)
        self.dock_props.setObjectName("dock_props")
        self.dock_props.setAllowedAreas(QtCore.Qt.RightDockWidgetArea | QtCore.Qt.LeftDockWidgetArea)
        self.dock_props.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFloatable |
            QtWidgets.QDockWidget.DockWidgetClosable
        )

        self.inspector_tabs = QtWidgets.QTabWidget()
        self.inspector_tabs.setDocumentMode(True)

        # Aba 1: Geometria (reusa seu widget atual de cortes)
        # Se você já tem um widget pronto, mantenha o nome/instância existente.
        # Vou assumir que você cria algo tipo self.grid_slicer_widget na versão atual.
        # Se não existir, você pode adaptar para o seu construtor real.
        if hasattr(self, "grid_slicer_widget") and self.grid_slicer_widget is not None:
            self.inspector_tab_geom = self.grid_slicer_widget
        else:
            # fallback: chama seu construtor atual (ajuste o nome se for diferente no seu código)
            self.inspector_tab_geom = self._build_grid_slicer_widget(nx, ny, nz) if hasattr(self, "_build_grid_slicer_widget") else QtWidgets.QWidget()

        self.inspector_tabs.addTab(self.inspector_tab_geom, "Geometria")

        # Aba 2: Propriedades (reusa a página que já existia)
        if hasattr(self, "page_grid_props") and self.page_grid_props is not None:
            self.inspector_tab_props = self.page_grid_props
        else:
            # fallback: tenta criar a página de props se existir builder
            self.inspector_tab_props = self._build_grid_props_page() if hasattr(self, "_build_grid_props_page") else QtWidgets.QWidget()

        self.inspector_tabs.addTab(self.inspector_tab_props, "Propriedades")

        # Aba 3: Compare (só existe se você já tinha page_compare)
        if hasattr(self, "page_compare") and self.page_compare is not None:
            self.inspector_tabs.addTab(self.page_compare, "Comparação")

        self.dock_props.setWidget(self.inspector_tabs)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dock_props)

        # (Opcional) deixa a direita um pouco mais estreita por padrão
        try:
            self.resizeDocks([self.dock_explorer, self.dock_props], [320, 320], QtCore.Qt.Horizontal)
        except Exception:
            pass



    def add_model_to_tree(self, model_key, name, is_base=False):
        """Adiciona o modelo no Project Explorer com sub-itens mínimos."""
        root = self.project_tree.invisibleRootItem()

        item = QtWidgets.QTreeWidgetItem(root)
        item.setText(0, name)
        item.setData(0, QtCore.Qt.UserRole, "model_root")
        item.setData(0, QtCore.Qt.UserRole + 1, "base" if is_base else model_key)

        # Ícone (pasta/modelo)
        item.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_DirIcon))

        # Sub-itens enxutos
        sub_geom = QtWidgets.QTreeWidgetItem(item)
        sub_geom.setText(0, "Geometria (Grid)")
        sub_geom.setData(0, QtCore.Qt.UserRole, "model_geom")
        sub_geom.setData(0, QtCore.Qt.UserRole + 1, "base" if is_base else model_key)
        sub_geom.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView))

        sub_props = QtWidgets.QTreeWidgetItem(item)
        sub_props.setText(0, "Propriedades & Filtros")
        sub_props.setData(0, QtCore.Qt.UserRole, "model_props")
        sub_props.setData(0, QtCore.Qt.UserRole + 1, "base" if is_base else model_key)
        sub_props.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon))

        item.setExpanded(True)
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
        """Carrega grid, restaura filtros e modo de visualização específicos do modelo."""
        if model_key not in self.models:
            return

        target_facies = self.models[model_key]["facies"]
        if target_facies is None:
            return

        from load_data import grid as global_grid
        from scipy.ndimage import label, generate_binary_structure

        saved_mode = self.models[model_key].get("view_mode", "facies")
        current_res_set = self.models[model_key]["reservoir_facies"]

        self.state["current_facies"] = target_facies
        self.state["reservoir_facies"] = current_res_set
        self.state["mode"] = saved_mode

        if hasattr(self, "btn_mode"):
            labels = {
                "facies": "Fácies",
                "reservoir": "Reservatório",
                "clusters": "Clusters",
                "largest": "Maior Cluster",
                "thickness_local": "Espessura Local",
            }
            self.btn_mode.setText(f"Modo: {labels.get(saved_mode, saved_mode)}")

        # >>> CORREÇÃO: usa a GEOMETRIA do modelo, não o global_grid do Base
        source_grid = self.models[model_key].get("grid", global_grid)
        active_grid = source_grid.copy(deep=True)
        active_grid.cell_data["Facies"] = target_facies

        # Recalcula propriedades
        is_res = np.isin(target_facies, list(current_res_set)).astype(np.uint8)
        active_grid.cell_data["Reservoir"] = is_res

        arr_3d = is_res.reshape((nx, ny, nz), order="F")
        structure = generate_binary_structure(3, 1)
        labeled, _ = label(arr_3d.transpose(2, 1, 0), structure=structure)
        clusters_1d = labeled.transpose(2, 1, 0).reshape(-1, order="F").astype(np.int32)
        active_grid.cell_data["Clusters"] = clusters_1d

        counts = np.bincount(clusters_1d.ravel())
        if counts.size > 0:
            counts[0] = 0
        largest_lbl = counts.argmax() if counts.size > 0 else 0
        active_grid.cell_data["LargestCluster"] = (clusters_1d == largest_lbl).astype(np.uint8)

        # Recalcula espessuras/verticais no grid correto
        self.recalc_vertical_metrics(active_grid, target_facies, current_res_set)

        # Atualiza estado/legendas
        lut, rng = make_clusters_lut(clusters_1d)
        self.state["clusters_lut"] = lut
        self.state["clusters_rng"] = rng
        self.state["clusters_sizes"] = compute_cluster_sizes(clusters_1d)

        self.state["current_grid_source"] = active_grid
        self.state["refresh"]()

        if saved_mode == "clusters":
            self.populate_clusters_legend()
        else:
            self.populate_facies_legend()

        self.update_sidebar_metrics_text(model_key)
        self.update_2d_map()
        

    def on_tree_selection_changed(self):
        items = self.project_tree.selectedItems()
        if not items:
            return

        it = items[0]
        it_type = it.data(0, QtCore.Qt.UserRole)
        model_key = it.data(0, QtCore.Qt.UserRole + 1)

        # Se clicou no root do modelo, só atualiza modelo ativo
        if it_type == "model_root":
            if model_key:
                self.active_model_key = model_key
            return

        # Se clicou em sub-itens do modelo: troca aba do Inspector
        if hasattr(self, "inspector_tabs") and isinstance(self.inspector_tabs, QtWidgets.QTabWidget):
            if it_type == "model_geom":
                # Aba "Geometria"
                idx = self.inspector_tabs.indexOf(self.inspector_tab_geom) if hasattr(self, "inspector_tab_geom") else 0
                if idx >= 0:
                    self.inspector_tabs.setCurrentIndex(idx)
            elif it_type == "model_props":
                idx = self.inspector_tabs.indexOf(self.inspector_tab_props) if hasattr(self, "inspector_tab_props") else 1
                if idx >= 0:
                    self.inspector_tabs.setCurrentIndex(idx)

        # Mantém seu comportamento anterior de atualizar a seleção/modelo
        if model_key:
            self.active_model_key = model_key



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
        # --- CORREÇÃO: Salva no cache para persistir ao navegar na árvore ---
        self.cached_metrics["base"]["df"] = df
        
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
        self.facies_table.setHorizontalHeaderLabels([pretty.get(c,c) for c in df.columns])
        
        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                val = df.iloc[i][col]
                if isinstance(val, (float, np.floating)):
                    if col in ["fraction", "connected_fraction", "Perc_X", "Perc_Y", "Perc_Z"]: txt = f"{val:.3f}"
                    elif "volume" in col: txt = f"{val:.2e}"
                    else: txt = f"{val:.2f}"
                else: txt = str(val)
                self.facies_table.setItem(i, j, QtWidgets.QTableWidgetItem(txt))
        self.facies_table.resizeColumnsToContents()

    def change_reservoir_facies(self, reservoir_set):
        if not isinstance(reservoir_set, set): return
        
        # Identifica modelo ativo
        current_model_key = "base"
        sel = self.project_tree.selectedItems()
        if sel:
            key = sel[0].data(0, QtCore.Qt.UserRole + 1)
            if key in self.models:
                current_model_key = key

        # Salva
        self.models[current_model_key]["reservoir_facies"] = reservoir_set
        self.state["reservoir_facies"] = reservoir_set
        
        # Recalcula Visualização
        self.switch_main_view_to_model(current_model_key)
        
        # Recalcula Métricas
        target_facies = self.models[current_model_key]["facies"]
        m, p = compute_global_metrics_for_array(target_facies, reservoir_set)
        self.cached_metrics[current_model_key]["metrics"] = m
        self.cached_metrics[current_model_key]["perc"] = p
        
        # Atualiza Texto Lateral
        self.update_sidebar_metrics_text(current_model_key)
        
        # Sincroniza Comparação se for o caso
        if self.compare_facies is not None:
            self.update_comparison_tables()
            if hasattr(self, 'update_compare_3d_mode_single'):
                self.update_compare_3d_mode_single("base")
                self.update_compare_3d_mode_single("compare")

    def load_compare_model(self, grdecl_path):
        # >>> CORREÇÃO PRINCIPAL: carregar GEOMETRIA (grid) + facies do modelo
        try:
            from load_data import load_grid_from_grdecl
            grid_compare, fac_compare = load_grid_from_grdecl(grdecl_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erro", str(e))
            return

        # Confere compatibilidade básica
        if fac_compare.size != nx * ny * nz:
            QtWidgets.QMessageBox.warning(self, "Erro", "Grid incompatível")
            return

        import time
        model_id = f"compare_{int(time.time() * 1000)}"
        model_name = os.path.basename(grdecl_path)

        # Independente do base
        rf = set()

        # >>> Agora guardamos também o GRID (geometria própria)
        self.models[model_id] = {
            "name": model_name,
            "facies": fac_compare,
            "grid": grid_compare,              # <<< ESSENCIAL
            "reservoir_facies": rf,
            "view_mode": "facies"
        }

        # Estatísticas e Métricas
        stats, _ = facies_distribution_array(fac_compare)
        cm, cp = compute_global_metrics_for_array(fac_compare, rf)
        df_detail = self.generate_detailed_metrics_df(fac_compare)

        self.cached_metrics[model_id] = {"metrics": cm, "perc": cp, "df": df_detail}

        self.compare_facies = fac_compare
        self.compare_facies_stats = stats
        self.comp_res_stats, _ = reservoir_facies_distribution_array(fac_compare, rf)

        self.active_compare_id = model_id

        self.add_model_to_tree(model_id, f"Comparado: {model_name}")

        self.update_comparison_tables()

        # Força atualização se estiver na aba de comparação
        if self.central_stack.currentIndex() == 1:
            self.update_dynamic_comparison_view()

        

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
    
    def update_multi_model_filter_table(self, model_keys):
        """Matriz Fácies x Modelos (checkbox) para definir reservoir_facies por modelo.
        - Inclui união de fácies de todos os modelos
        - Trata NaN/float com segurança (NaN -> 0)
        - Desabilita checkbox quando a fácies não existe naquele modelo
        - Mostra cor da fácies (ícone + fundo suave) usando self.facies_colors
        - Nunca deixa signals bloqueados (try/finally)
        """
        if not hasattr(self, "multi_model_table"):
            return

        def safe_unique_facies(f):
            if f is None:
                return np.array([], dtype=int)
            a = np.asarray(f)
            if a.size == 0:
                return np.array([], dtype=int)

            if np.issubdtype(a.dtype, np.floating):
                a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(int, copy=False)
            elif a.dtype == object:
                out = []
                for x in a.ravel():
                    try:
                        if x is None:
                            out.append(0)
                        else:
                            fx = float(x)
                            if np.isnan(fx) or np.isinf(fx):
                                out.append(0)
                            else:
                                out.append(int(fx))
                    except Exception:
                        out.append(0)
                a = np.asarray(out, dtype=int)
            else:
                a = a.astype(int, copy=False)

            return np.unique(a)

        def facies_color_icon(fac_id: int):
            """Retorna (QIcon, QColor_bg) ou (None, None) se não existir."""
            if not hasattr(self, "facies_colors") or not self.facies_colors:
                return None, None
            rgba = self.facies_colors.get(int(fac_id))
            if rgba is None:
                return None, None
            try:
                r, g, b = rgba[0], rgba[1], rgba[2]
                q = QtGui.QColor(int(r * 255), int(g * 255), int(b * 255))
                pm = QtGui.QPixmap(14, 14)
                pm.fill(q)
                icon = QtGui.QIcon(pm)
                bg = QtGui.QColor(q)
                bg.setAlpha(45)
                return icon, bg
            except Exception:
                return None, None

        t = self.multi_model_table
        t.blockSignals(True)
        try:
            facies_union = set()
            facies_by_model = {}

            for mk in model_keys:
                _, f = self._get_model_payload(mk)
                uniq = safe_unique_facies(f)
                s = set(int(x) for x in uniq.tolist())
                facies_by_model[mk] = s
                facies_union |= s

            facies_list = sorted(int(x) for x in facies_union)

            headers = ["Fácies"]
            for mk in model_keys:
                if str(mk).lower() == "base":
                    headers.append("Base")
                elif hasattr(self, "models") and mk in self.models and self.models[mk].get("name"):
                    headers.append(self.models[mk]["name"])
                else:
                    headers.append(str(mk))

            t.clear()
            t.setRowCount(len(facies_list))
            t.setColumnCount(1 + len(model_keys))
            t.setHorizontalHeaderLabels(headers)

            for r, fac in enumerate(facies_list):
                it_fac = QtWidgets.QTableWidgetItem(str(fac))
                it_fac.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)

                icon, bg = facies_color_icon(fac)
                if icon is not None:
                    it_fac.setIcon(icon)
                if bg is not None:
                    it_fac.setBackground(QtGui.QBrush(bg))

                t.setItem(r, 0, it_fac)

                for c, mk in enumerate(model_keys, start=1):
                    present = fac in facies_by_model.get(mk, set())

                    it = QtWidgets.QTableWidgetItem("")
                    it.setData(QtCore.Qt.UserRole, (mk, int(fac)))

                    if not present:
                        it.setFlags(QtCore.Qt.ItemIsSelectable)  # sem checkbox
                        it.setCheckState(QtCore.Qt.Unchecked)
                        t.setItem(r, c, it)
                        continue

                    it.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsSelectable)

                    rf = set()
                    if hasattr(self, "models") and mk in self.models:
                        rf = self.models[mk].get("reservoir_facies") or set()

                    it.setCheckState(QtCore.Qt.Checked if int(fac) in rf else QtCore.Qt.Unchecked)
                    t.setItem(r, c, it)

            t.resizeColumnsToContents()

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
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Selecionar Modelos", "grids", "GRDECL (*.grdecl)")
        for path in paths: self.load_compare_model(path)

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
        # 1. Atualiza preferência em TODOS os modelos (Sincronização Global)
        for key in self.models:
            self.models[key]["view_mode"] = new_mode

        # 2. Atualiza visualização principal (Active Grid)
        self.state["mode"] = new_mode
        self.current_mode = new_mode
        self.state["refresh"]()
        
        # 3. Atualiza estados de Comparação (se existirem)
        if self.compare_states.get("base"):
            self.compare_states["base"]["mode"] = new_mode
            if new_mode == "clusters":
                rf = self.models["base"]["reservoir_facies"]
                if "update_reservoir_fields" in self.compare_states["base"]:
                    self.compare_states["base"]["update_reservoir_fields"](rf)
            self.compare_states["base"]["refresh"]()

        if self.compare_states.get("compare"):
            self.compare_states["compare"]["mode"] = new_mode
            if new_mode == "clusters":
                # Pega o ID correto do modelo comparado ativo
                target_id = getattr(self, "active_compare_id", "compare")
                # Se active_compare_id não estiver em models, fallback seguro
                if target_id not in self.models: target_id = "compare"
                
                rf = self.models[target_id]["reservoir_facies"]
                if "update_reservoir_fields" in self.compare_states["compare"]:
                    self.compare_states["compare"]["update_reservoir_fields"](rf)
            self.compare_states["compare"]["refresh"]()

        # 4. Se estiver na visualização dinâmica (Aba Comparação), atualiza tudo
        if self.central_stack.currentIndex() == 1:
            self.update_dynamic_comparison_view()

        # 5. Interface Lateral (Toggle de Tabelas)
        if new_mode == "clusters":
            self.legend_group.setTitle("Filtro (Fácies) & Legenda (Clusters)")
            # Mostra AMBAS: Fácies (para filtrar) e Clusters (para ver cores)
            self.facies_legend_table.setVisible(True)
            self.clusters_legend_table.setVisible(True)
            
            self.populate_facies_legend() # Garante que filtro esteja atualizado
            self.populate_clusters_legend() # Preenche cores dos clusters
        else:
            self.legend_group.setTitle("Legenda & Filtro")
            self.facies_legend_table.setVisible(True)
            self.clusters_legend_table.setVisible(False)
            self.populate_facies_legend()
            
        if hasattr(self, 'comp_plotter_base'): self.comp_plotter_base.render()
        if hasattr(self, 'comp_plotter_comp'): self.comp_plotter_comp.render()

    def change_thickness_mode(self, label):
        self.state["thickness_mode"] = label
        
        # Atualiza o modelo único (Visualização)
        if "update_thickness" in self.state: self.state["update_thickness"]()
        self.state["refresh"]()
        self.update_2d_map()
        
        # --- CORREÇÃO: Atualiza a Comparação (se estiver ativa) ---
        if self.central_stack.currentIndex() == 1:
            self.refresh_comparison_active_view()

    def toggle_tree_checkboxes(self, show):
        """Habilita ou desabilita checkboxes em todos os modelos raiz."""
        root = self.project_tree.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            # Verifica se é um item de modelo (segurança extra)
            if item.data(0, QtCore.Qt.UserRole) == "model_root":
                self._set_item_checkbox_visible(item, show)
    
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
        
        # 1. SLICER (Reutiliza a lógica, mas precisa comunicar com todos os modelos)
        # Criamos um novo SlicerWidget específico para comparação ou reutilizamos
        # Para simplificar, vamos criar um novo que aponta para o callback de sync
        from load_data import nx, ny, nz
        self.comp_slicer = GridSlicerWidget(nx, ny, nz, self.on_comp_slice_changed)
        gb_slice = QtWidgets.QGroupBox("Cortes & Escala (Sincronizado)")
        l_sl = QtWidgets.QVBoxLayout(gb_slice)
        l_sl.addWidget(self.comp_slicer)
        layout.addWidget(gb_slice)
        
        # 2. FILTRO MATRIZ (Multi-Modelo)
        self.comp_filter_group = QtWidgets.QGroupBox("Filtro de Reservatório por Modelo")
        l_filt = QtWidgets.QVBoxLayout(self.comp_filter_group)
        
        self.multi_model_table = QtWidgets.QTableWidget()
        self.multi_model_table.verticalHeader().setVisible(False)
        self.multi_model_table.itemChanged.connect(self.on_multi_model_filter_changed)
        
        l_filt.addWidget(self.multi_model_table)
        layout.addWidget(self.comp_filter_group)
        
        return container
    
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
        if mode == "visualization":
            if hasattr(self, 'act_persp_viz'):
                self.act_persp_viz.setChecked(True)
            if hasattr(self, 'act_persp_comp'):
                self.act_persp_comp.setChecked(False)

            if hasattr(self, 'btn_persp_viz'):
                self.btn_persp_viz.setChecked(True)
            if hasattr(self, 'btn_persp_comp'):
                self.btn_persp_comp.setChecked(False)

            # 1. Central: Visualização Padrão
            self.central_stack.setCurrentIndex(0)

            # 2. Árvore: Modo Normal (Expansível, sem Checkboxes)
            self.project_tree.setItemsExpandable(True)
            self.project_tree.expandAll()
            self.toggle_tree_checkboxes(False)
            self.dock_explorer.setWindowTitle("Project Explorer")

            # 3. Painel Lateral: Propriedades do item selecionado
            if hasattr(self, 'page_grid_props'):
                self.props_stack.setCurrentWidget(self.page_grid_props)
            self.dock_props.setWindowTitle("Propriedades")
            self.on_tree_selection_changed()

            # Ribbon: aponta pro tab de View
            if hasattr(self, 'ribbon') and hasattr(self, 'ribbon_tab_view'):
                self.ribbon.setCurrentWidget(self.ribbon_tab_view)

        elif mode == "comparison":
            if hasattr(self, 'act_persp_viz'):
                self.act_persp_viz.setChecked(False)
            if hasattr(self, 'act_persp_comp'):
                self.act_persp_comp.setChecked(True)

            if hasattr(self, 'btn_persp_viz'):
                self.btn_persp_viz.setChecked(False)
            if hasattr(self, 'btn_persp_comp'):
                self.btn_persp_comp.setChecked(True)

            # 1. Central: Modo Comparação
            self.central_stack.setCurrentIndex(1)

            # Força o update da visualização atual baseada no combo
            if hasattr(self, 'combo_comp_view'):
                self.on_comp_view_changed(self.combo_comp_view.currentIndex())

            # 2. Árvore: Modo Seletor (Travado, Colapsado, com Checkboxes)
            self.project_tree.collapseAll()
            self.project_tree.setItemsExpandable(False)
            self.toggle_tree_checkboxes(True)
            self.dock_explorer.setWindowTitle("Seletor de Modelos")

            # 3. Painel Lateral: Filtros de Comparação
            if hasattr(self, 'page_compare'):
                self.props_stack.setCurrentWidget(self.page_compare)
            self.dock_props.setWindowTitle("Painel de Comparação")

            # Ribbon: aponta pro tab de Compare
            if hasattr(self, 'ribbon') and hasattr(self, 'ribbon_tab_compare'):
                self.ribbon.setCurrentWidget(self.ribbon_tab_compare)

    
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

        # --- 1. TABELA GLOBAL ---
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
            ("Vol. Res (m3)", "reservoir_volume", "{:.2e}")
        ]
        
        t_glob.setRowCount(len(metrics_list))
        for r, (label, key, fmt) in enumerate(metrics_list):
            t_glob.setItem(r, 0, QtWidgets.QTableWidgetItem(label))
            for c, (m_key, _) in enumerate(checked_models):
                data = self.cached_metrics.get(m_key)
                val_str = "-"
                if data and "metrics" in data and data["metrics"]:
                    val = data["metrics"].get(key, 0)
                    try: val_str = fmt.format(val)
                    except: val_str = str(val)
                t_glob.setItem(r, c + 1, QtWidgets.QTableWidgetItem(val_str))
        t_glob.resizeColumnsToContents()

        # --- PREPARAÇÃO DADOS POR FÁCIES ---
        # Coleta união de todas as fácies presentes nos modelos selecionados
        all_facies = set()
        model_stats = {} # Cache local de estatísticas {key: stats_dict}
        
        for m_key, _ in checked_models:
            # Recupera estatísticas já calculadas ou calcula agora se necessário
            # (Geralmente calculamos no load, mas vamos garantir)
            if m_key == "base":
                stats = getattr(self, "base_facies_stats", {})
            elif hasattr(self, "compare_facies_stats") and m_key == self.active_compare_id:
                 stats = self.compare_facies_stats
            else:
                # Fallback: Recalcula se não achar pronto (para N modelos)
                from analysis import facies_distribution_array
                f_arr = self.models[m_key]["facies"]
                stats, _ = facies_distribution_array(f_arr)
            
            model_stats[m_key] = stats
            if stats:
                all_facies.update(stats.keys())
        
        sorted_facies = sorted(list(all_facies))

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

        if checked_models is None:
            checked_models = self.get_checked_models()

        checked_non_base = [m for m in checked_models if str(m).lower() != "base"]
        models_to_show = ["base"] + checked_non_base

        # >>> Atualiza a matriz de filtro sempre que muda seleção
        self.update_multi_model_filter_table(models_to_show)

        # parâmetros globais
        mode = self.state.get("mode", "facies")
        z_exag = float(self.state.get("z_exag", 15.0))
        show_scalar_bar = bool(self.state.get("show_scalar_bar", True))

        # limpa plotters antigos
        if hasattr(self, "active_comp_plotters"):
            for p in self.active_comp_plotters:
                try: p.close()
                except Exception: pass

        while self.comp_layout_3d.count():
            child = self.comp_layout_3d.takeAt(0)
            if child.widget(): child.widget().deleteLater()

        self.active_comp_plotters = []
        self.active_comp_states = []   # <<< ESSENCIAL p/ sliders + filtro
        self.compare_states_multi = {}

        grid_layout = QtWidgets.QGridLayout()
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(6)

        n = len(models_to_show)
        cols = 3 if n >= 3 else n

        for idx, model_key in enumerate(models_to_show):
            row = idx // cols
            col = idx % cols

            w = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(w)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(2)

            display_name = "Base" if str(model_key).lower() == "base" else str(model_key)
            if hasattr(self, "models") and model_key in self.models and self.models[model_key].get("name"):
                display_name = self.models[model_key]["name"]

            lbl = QtWidgets.QLabel(display_name)
            lbl.setAlignment(QtCore.Qt.AlignCenter)

            plotter, plotter_widget = self._make_embedded_plotter(parent=w)
            try: plotter.set_background("white")
            except Exception: pass

            v.addWidget(lbl)
            v.addWidget(plotter_widget)
            grid_layout.addWidget(w, row, col)

            grid, facies = self._get_model_payload(model_key)
            if grid is None or facies is None:
                try: plotter.add_text("Grid não carregado", font_size=10)
                except Exception: pass
                self.active_comp_plotters.append(plotter)
                continue

            local_state = {}
            _, local_state = run(
                mode=mode,
                z_exag=z_exag,
                show_scalar_bar=show_scalar_bar,
                external_plotter=plotter,
                external_state=local_state,
                target_grid=grid,
                target_facies=facies,
            )

            # >>> marca qual modelo esse state pertence (p/ filtro)
            local_state["model_key"] = model_key

            # >>> garante campos de reservatório (p/ modos reservoir/clusters)
            rf = set()
            if hasattr(self, "models") and model_key in self.models:
                rf = self.models[model_key].get("reservoir_facies", set()) or set()
            if "update_reservoir_fields" in local_state:
                local_state["update_reservoir_fields"](rf)

            self.active_comp_plotters.append(plotter)
            self.active_comp_states.append(local_state)
            self.compare_states_multi[str(model_key)] = local_state

        container = QtWidgets.QWidget()
        container.setLayout(grid_layout)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)

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
        """Callback ao clicar num checkbox da tabela matriz."""
        if item is None:
            return

        # Se não é checkable (fácies não existe naquele modelo), ignora
        if not (item.flags() & QtCore.Qt.ItemIsUserCheckable):
            return

        data = item.data(QtCore.Qt.UserRole)
        if not data or not isinstance(data, tuple) or len(data) != 2:
            return

        model_key, fac = data
        model_key = "base" if str(model_key).lower() == "base" else model_key
        try:
            fac = int(fac)
        except Exception:
            fac = 0

        if not hasattr(self, "models") or model_key not in self.models:
            return

        # Garante set de reservoir_facies
        if self.models[model_key].get("reservoir_facies") is None:
            self.models[model_key]["reservoir_facies"] = set()

        target_set = self.models[model_key]["reservoir_facies"]

        is_checked = (item.checkState() == QtCore.Qt.Checked)
        if is_checked:
            target_set.add(fac)
        else:
            target_set.discard(fac)

        # Atualiza o(s) estados ativos daquele modelo (3D ou 2D)
        if hasattr(self, "active_comp_states"):
            for state in self.active_comp_states:
                if state.get("model_key") == model_key:
                    if "update_reservoir_fields" in state:
                        # passa uma cópia (evita efeitos colaterais)
                        state["update_reservoir_fields"](set(int(x) for x in target_set))
                    if "refresh" in state:
                        state["refresh"]()

    
    def on_comp_view_changed(self, index):
        """Muda a página do Stack de Comparação baseado no ComboBox."""
        if not hasattr(self, 'compare_stack'):
            return
        # O índice 0 é 3D, 1 é Métricas, 2 é Mapas 2D
        if index < self.compare_stack.count():
            self.compare_stack.setCurrentIndex(index)
            self.refresh_comparison_active_view()
    
    def get_checked_models(self):
        """Retorna a lista de model_keys marcados (checkbox) na árvore."""
        checked = []
        root = self.project_tree.invisibleRootItem()

        for i in range(root.childCount()):
            item = root.child(i)
            if item.data(0, QtCore.Qt.UserRole) != "model_root":
                continue

            # Só considera se o item é checkable e está marcado
            if (item.flags() & QtCore.Qt.ItemIsUserCheckable) and item.checkState(0) == QtCore.Qt.Checked:
                model_key = item.data(0, QtCore.Qt.UserRole + 1)
                if model_key is None:
                    model_key = item.text(0)
                checked.append(model_key)

        return checked



    def refresh_comparison_active_view(self):
        if not hasattr(self, "combo_comp_view"):
            return

        checked_models = self.get_checked_models() if hasattr(self, "get_checked_models") else []

        # Não tentar renderizar se ainda não tem pelo menos Base + 1 modelo
        checked_non_base = [m for m in checked_models if str(m).lower() != "base"]
        if self.combo_comp_view.currentIndex() in (0, 2) and len(checked_non_base) == 0:
            # Página 3D
            if self.combo_comp_view.currentIndex() == 0:
                while self.comp_layout_3d.count():
                    child = self.comp_layout_3d.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
                label = QtWidgets.QLabel("Marque 1+ modelos no Project Explorer para comparar.")
                label.setAlignment(QtCore.Qt.AlignCenter)
                self.comp_layout_3d.addWidget(label)
            # Página 2D
            if self.combo_comp_view.currentIndex() == 2:
                while self.comp_2d_layout.count():
                    child = self.comp_2d_layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
                label = QtWidgets.QLabel("Marque 1+ modelos no Project Explorer para comparar.")
                label.setAlignment(QtCore.Qt.AlignCenter)
                self.comp_2d_layout.addWidget(label)
            return

        # Render normal
        if self.combo_comp_view.currentIndex() == 0:
            self.update_dynamic_comparison_view(checked_models)
        elif self.combo_comp_view.currentIndex() == 1:
            self.update_comparison_tables()
        elif self.combo_comp_view.currentIndex() == 2:
            self.update_dynamic_comparison_2d(checked_models)

    
    def update_dynamic_comparison_2d(self, checked_models):
        from visualize import run

        checked_non_base = [m for m in checked_models if str(m).lower() != "base"]
        models_to_show = ["base"] + checked_non_base

        self.update_multi_model_filter_table(models_to_show)

        mode = self.state.get("mode", "facies")
        z_exag = float(self.state.get("z_exag", 15.0))
        show_scalar_bar = bool(self.state.get("show_scalar_bar", True))

        if hasattr(self, "active_comp_plotters"):
            for p in self.active_comp_plotters:
                try: p.close()
                except Exception: pass

        while self.comp_2d_layout.count():
            child = self.comp_2d_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

        self.active_comp_plotters = []
        self.active_comp_states = []

        grid_layout = QtWidgets.QGridLayout()
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(6)

        n = len(models_to_show)
        cols = 3 if n >= 3 else n

        for idx, model_key in enumerate(models_to_show):
            row = idx // cols
            col = idx % cols

            w = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(w)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(2)

            display_name = "Base" if str(model_key).lower() == "base" else str(model_key)
            if hasattr(self, "models") and model_key in self.models and self.models[model_key].get("name"):
                display_name = self.models[model_key]["name"]

            lbl = QtWidgets.QLabel(display_name)
            lbl.setAlignment(QtCore.Qt.AlignCenter)

            plotter, plotter_widget = self._make_embedded_plotter(parent=w)
            try: plotter.set_background("white")
            except Exception: pass

            v.addWidget(lbl)
            v.addWidget(plotter_widget)
            grid_layout.addWidget(w, row, col)

            grid, facies = self._get_model_payload(model_key)
            if grid is None or facies is None:
                try: plotter.add_text("Grid não carregado", font_size=10)
                except Exception: pass
                self.active_comp_plotters.append(plotter)
                continue

            local_state = {}
            _, local_state = run(
                mode=mode,
                z_exag=z_exag,
                show_scalar_bar=show_scalar_bar,
                external_plotter=plotter,
                external_state=local_state,
                target_grid=grid,
                target_facies=facies,
            )

            local_state["model_key"] = model_key

            rf = set()
            if hasattr(self, "models") and model_key in self.models:
                rf = self.models[model_key].get("reservoir_facies", set()) or set()
            if "update_reservoir_fields" in local_state:
                local_state["update_reservoir_fields"](rf)

            self.active_comp_plotters.append(plotter)
            self.active_comp_states.append(local_state)

        container = QtWidgets.QWidget()
        container.setLayout(grid_layout)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)

        self.comp_2d_layout.addWidget(scroll)


    def on_tree_item_changed(self, item, column):
        if not hasattr(self, 'central_stack'): return
        
        if item.data(0, QtCore.Qt.UserRole) == "model_root":
            if self.central_stack.currentIndex() == 1:
                # Chama a atualização da view ATIVA (3D, 2D ou Métricas)
                self.refresh_comparison_active_view()

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
        """Recalcula métricas verticais COMPLETAS para o grid ativo."""
        # Prepara dados
        fac_3d = facies_array.reshape((nx, ny, nz), order="F")
        
        # Recupera Z (Assume geometria constante)
        from load_data import grid as global_grid
        centers = global_grid.cell_centers().points
        z_vals = centers[:, 2].reshape((nx, ny, nz), order="F")
        
        # Inicializa TODOS os arrays necessários
        keys = [
            "vert_Ttot_reservoir", "vert_NTG_col_reservoir", "vert_NTG_env_reservoir",
            "vert_n_packages_reservoir", "vert_Tpack_max_reservoir", 
            "vert_ICV_reservoir", "vert_Qv_reservoir", "vert_Qv_abs_reservoir"
        ]
        data_map = {k: np.zeros((nx, ny, nz), dtype=float) for k in keys}
        
        res_list = list(reservoir_set)
        res_set_fast = set(res_list) # Lookup O(1)
        
        for ix in range(nx):
            for iy in range(ny):
                col_fac = fac_3d[ix, iy, :]
                mask = np.isin(col_fac, res_list)
                
                if not np.any(mask): continue
                
                col_z = z_vals[ix, iy, :]
                z_min, z_max = np.nanmin(col_z), np.nanmax(col_z)
                T_col = abs(z_max - z_min)
                if T_col == 0: continue
                
                dz = T_col / nz
                
                idx = np.where(mask)[0]
                n_res = len(idx)
                T_tot = n_res * dz
                
                # Envelope
                if n_res > 0:
                    T_env = (idx[-1] - idx[0] + 1) * dz
                else: T_env = 0.0
                    
                NTG_col = T_tot / T_col
                NTG_env = T_tot / T_env if T_env > 0 else 0.0
                
                # Pacotes
                packages = []
                if n_res > 0:
                    start = idx[0]
                    prev = idx[0]
                    for k in idx[1:]:
                        if k == prev + 1: prev = k
                        else:
                            packages.append(prev - start + 1)
                            start = prev = k
                    packages.append(prev - start + 1)
                
                T_pack_max = max(packages) * dz if packages else 0.0
                n_packages = len(packages)
                
                # ICV e Qv
                ICV = T_pack_max / T_env if T_env > 0 else 0.0
                Qv = NTG_col * ICV
                Qv_abs = ICV * (T_pack_max / T_col)

                # Preenchimento (apenas nas células de reservatório para visualização 3D correta)
                data_map["vert_Ttot_reservoir"][ix, iy, mask] = T_tot
                data_map["vert_NTG_col_reservoir"][ix, iy, mask] = NTG_col
                data_map["vert_NTG_env_reservoir"][ix, iy, mask] = NTG_env
                data_map["vert_n_packages_reservoir"][ix, iy, mask] = float(n_packages)
                data_map["vert_Tpack_max_reservoir"][ix, iy, mask] = T_pack_max
                data_map["vert_ICV_reservoir"][ix, iy, mask] = ICV
                data_map["vert_Qv_reservoir"][ix, iy, mask] = Qv
                data_map["vert_Qv_abs_reservoir"][ix, iy, mask] = Qv_abs

        # Salva no grid
        for k, v in data_map.items():
            target_grid.cell_data[k] = v.reshape(-1, order="F")


    def _open_matplotlib_report(self, well_name, sim_model_name, real_depth, real_fac, base_depth, base_fac, sim_depth, sim_fac):
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
        dialog.resize(1500, 850)
        dialog.setWindowFlags(dialog.windowFlags() | QtCore.Qt.WindowMinMaxButtonsHint)
        
        main_layout = QtWidgets.QVBoxLayout(dialog)
        tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(tabs)

        # =================================================================
        # ABA 1: LOGS + VOLUME (Com Porcentagens Restauradas)
        # =================================================================
        tab1 = QtWidgets.QWidget()
        l1 = QtWidgets.QVBoxLayout(tab1)
        
        fig1, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(14, 7), 
                                                gridspec_kw={'width_ratios': [0.2, 0.2, 0.2, 3]})
        
        # --- Cálculo de Geometria Independente ---
        # Real
        r_thick = real_depth - real_depth[0]
        r_total = r_thick[-1] if len(r_thick) > 0 else 0
        
        # Base (Pode ser diferente do Simulado agora)
        b_thick = base_depth - base_depth[0] if len(base_depth) > 0 else np.array([])
        b_total = b_thick[-1] if len(b_thick) > 0 else 0
        
        # Simulado
        s_thick = sim_depth - sim_depth[0] if len(sim_depth) > 0 else np.array([])
        s_total = s_thick[-1] if len(s_thick) > 0 else 0
        
        g_max = max(r_total, b_total, s_total)

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
            
            # Último
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

        # 1. BASE
        draw_log(ax1, b_thick, base_fac, f"Base\n{b_total:.1f}m")
        ax1.set_ylabel("Espessura (m)")
        ax1.set_yticks(np.linspace(0, g_max, 10))
        
        # 2. SIMULADO
        draw_log(ax2, s_thick, sim_fac, f"Simul\n{s_total:.1f}m")
        
        # 3. REAL
        draw_log(ax3, r_thick, real_fac, f"Real\n{r_total:.1f}m")

        # --- GRÁFICO DE VOLUME (Com Porcentagens Restauradas) ---
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
        ax4.set_title("Balanço Volumétrico por Fácies")
        ax4.legend()
        ax4.grid(axis='x', linestyle='--', alpha=0.5)
        
        # --- Loop de Texto de Porcentagem (RESTAURADO) ---
        # Compara Simulado vs Real (que é o objetivo da calibração)
        for i, (vr, vs) in enumerate(zip(vals_r, vals_s)):
            if vr > 0:
                diff_perc = ((vs - vr) / vr) * 100
                txt = f"{diff_perc:+.1f}%"
                color = 'green' if abs(diff_perc) < 20 else 'red'
            else:
                txt = "Novo" if vs > 0 else ""
                color = 'blue'
            
            # Posiciona o texto à direita da maior barra
            max_val = max(vr, vals_b[i], vs)
            if max_val > 0:
                ax4.text(max_val, y_pos[i], f" {txt}", va='center', color=color, fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        canvas1 = FigureCanvas(fig1)
        l1.addWidget(canvas1)
        tabs.addTab(tab1, "Logs & Volume")

        # =================================================================
        # ABA 2: MATRIZ DE TROCAS + FAMÍLIAS (%)
        # =================================================================
        tab2 = QtWidgets.QWidget()
        l2 = QtWidgets.QVBoxLayout(tab2)
        fig2, (ax2a, ax2b) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        # Matriz
        n_bins = 200
        r_norm = resample_to_normalized_depth(real_depth, real_fac, n_bins)
        s_norm = resample_to_normalized_depth(sim_depth, sim_fac, n_bins)
        b_norm = resample_to_normalized_depth(base_depth, base_fac, n_bins)
        
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
        # Título Limpo (Sem nome do modelo)
        ax2a.set_title("Matriz de Trocas (Real vs Simulado)")

        for i in range(n_classes):
            for j in range(n_classes):
                val = conf_matrix[i, j]
                color = "white" if val > conf_matrix.max()/2 else "black"
                if val > 0:
                    fw = 'bold' if i == j else 'normal'
                    ax2a.text(j, i, str(val), ha="center", va="center", color=color, fontweight=fw)
                if i == j:
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='gold', linewidth=3)
                    ax2a.add_patch(rect)

        # --- Famílias (AGORA EM PORCENTAGEM) ---
        def get_family(f_code):
            s = str(f_code)
            if s.startswith('1'): return "Siliciclásticos"
            if s.startswith('2'): return "Carbonatos"
            return "Outros"

        # Soma espessura total por família
        fam_stats = {"Real": {}, "Sim": {}, "Base": {}}
        
        # Totais para normalizar
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
        
        # Converte para %
        bars_b = [(fam_stats["Base"][fam] / tot_b)*100 for fam in families]
        bars_s = [(fam_stats["Sim"][fam] / tot_s)*100 for fam in families]
        bars_r = [(fam_stats["Real"][fam] / tot_r)*100 for fam in families]

        ax2b.bar(x_fam - 0.2, bars_b, 0.2, label='Base', color='#999999')
        ax2b.bar(x_fam,       bars_s, 0.2, label='Simulado', color='#007acc')
        ax2b.bar(x_fam + 0.2, bars_r, 0.2, label='Real', color='#444444')
        
        ax2b.set_xticks(x_fam); ax2b.set_xticklabels(families)
        ax2b.set_ylabel("Proporção (%)")
        ax2b.set_title("Balanço por Família (Porcentagem)")
        ax2b.legend()
        ax2b.set_ylim(0, 100) # Fixa escala 0-100%

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
        # ABA 4: CORRELAÇÃO EM ESPESSURA REAL (links inclinados)
        # =================================================================
        tab4 = QtWidgets.QWidget()
        l4 = QtWidgets.QVBoxLayout(tab4)

        fig4, ax4 = plt.subplots(figsize=(14, 7))
        fig4.tight_layout(rect=[0, 0.06, 1, 0.95])

        # bins normalizados (define "quem compara com quem")
        n_bins = 200
        b_norm = resample_to_normalized_depth(base_depth, base_fac, n_bins)
        s_norm = resample_to_normalized_depth(sim_depth, sim_fac, n_bins)
        r_norm = resample_to_normalized_depth(real_depth, real_fac, n_bins)

        self._plot_strat_correlation_real_depth(
            ax4,
            n_bins=n_bins,
            base_fac_bins=b_norm,
            sim_fac_bins=s_norm,
            real_fac_bins=r_norm,
            b_total=b_total,
            s_total=s_total,
            r_total=r_total,
            get_color=get_color,
            # title="Correlação Base → Simulado → Real (espessura real)",
            min_bins=1,
            link_alpha=0.18
        )

        canvas4 = FigureCanvas(fig4)
        l4.addWidget(canvas4)
        tabs.addTab(tab4, "Correlação (m)")

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
        window_size=1,   # <-- NOVO: 1,3,5,7...
        n_bins=200,
        w_strat=0.7,
        w_thick=0.3,
        ignore_real_zeros=True,
        use_kappa=True,
    ):
        """
        Retorna lista ranqueada:
        [{"model_key","model_name","score","details","n_wells_used"}, ...]
        """
        import numpy as np
        from analysis import compute_well_match_score

        if not self.wells:
            return []

        # poços a avaliar
        if well_names is None:
            well_names = list(self.wells.keys())
        else:
            well_names = [w for w in well_names if w in self.wells]

        if not well_names:
            return []

        # modelos a avaliar
        if model_keys is None:
            model_keys = list(self.models.keys())
        else:
            model_keys = [k for k in model_keys if k in self.models]

        if not model_keys:
            return []

        # garante ímpar >= 1
        try:
            window_size = int(window_size)
        except Exception:
            window_size = 1
        if window_size < 1:
            window_size = 1
        if window_size % 2 == 0:
            window_size += 1

        results = []

        for mk in model_keys:
            m = self.models.get(mk, {})
            g = m.get("grid", None)
            if g is None:
                continue

            # garante Facies no grid, se seu modelo guarda facies fora
            if "Facies" not in getattr(g, "cell_data", {}):
                fac = m.get("facies", None)
                if fac is not None:
                    try:
                        g.cell_data["Facies"] = np.asarray(fac).astype(int)
                    except Exception:
                        pass

            per_well = {}
            score_list = []
            w_list = []

            for wn in well_names:
                well = self.wells.get(wn)
                if well is None or well.data is None or well.data.empty:
                    continue

                # REAL: igual seu relatório
                if "DEPT" not in well.data.columns:
                    continue

                if "fac" in well.data.columns:
                    col_real = "fac"
                elif "lito_upscaled" in well.data.columns:
                    col_real = "lito_upscaled"
                else:
                    continue

                full_depth = well.data["DEPT"].to_numpy(dtype=float)
                full_real  = well.data[col_real].to_numpy(dtype=float)

                key = str(wn).strip()
                markers = self.markers_db.get(key, [])

                real_depth = full_depth
                real_fac   = np.where(np.isfinite(full_real), full_real, 0.0).astype(int)

                # recorte por marcadores (mesmo critério do show_well_comparison_report)
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

                # pega (X,Y) de referência exatamente como no relatório
                xy = self._pick_reference_xy_for_well_report(well, markers)
                if xy is None:
                    continue

                xref, yref = xy

                # --------- AQUI É O ÚNICO PONTO QUE MUDA (1x1 vs NxN) ---------
                if window_size == 1:
                    # 1x1: coluna mais próxima (comportamento antigo)
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

                else:
                    # NxN: pega o melhor match dentro da janela (3x3, 5x5, ...)
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

                    # opcional: guardar diagnóstico do deslocamento
                    s = dict(s)
                    s["best_i"] = i_best
                    s["best_j"] = j_best
                # ----------------------------------------------------------------

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

    def _on_models_table_selection_changed(self, dlg):
        from PyQt5 import QtCore

        tbl = dlg._tbl_models
        sel = tbl.selectedItems()
        if not sel:
            return

        # pega a linha selecionada
        row = tbl.currentRow()
        if row < 0:
            return

        # model_key está no item Rank (col 0)
        it_rank = tbl.item(row, 0)
        model_key = it_rank.data(QtCore.Qt.UserRole)

        # encontra o registro do ranking
        ranking = getattr(dlg, "_ranking", [])
        rec = None
        for r in ranking:
            if str(r.get("model_key")) == str(model_key):
                rec = r
                break

        if rec is None:
            return

        self._populate_wells_detail_table(dlg, rec)

    def _populate_wells_detail_table(self, dlg, model_record):
        from PyQt5 import QtWidgets, QtCore

        tbl = dlg._tbl_wells
        tbl.setRowCount(0)

        details = model_record.get("details", {}) or {}

        # ordena por score do poço (desc)
        items = list(details.items())
        items.sort(key=lambda kv: float(kv[1].get("score", 0.0)), reverse=True)

        for well_name, s in items:
            row = tbl.rowCount()
            tbl.insertRow(row)

            score = float(s.get("score", 0.0))
            acc = float(s.get("strat_acc", 0.0))
            kap = float(s.get("strat_kappa_norm", s.get("strat_kappa", 0.0)))
            thk = float(s.get("thick_score", 0.0))
            t_real = float(s.get("t_real", 0.0))
            t_sim = float(s.get("t_sim", 0.0))

            def ritem(v):
                it = QtWidgets.QTableWidgetItem(f"{v:.3f}")
                it.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
                return it

            tbl.setItem(row, 0, QtWidgets.QTableWidgetItem(str(well_name)))
            tbl.setItem(row, 1, ritem(score))
            tbl.setItem(row, 2, ritem(acc))
            tbl.setItem(row, 3, ritem(kap))
            tbl.setItem(row, 4, ritem(thk))

            it_tr = QtWidgets.QTableWidgetItem(f"{t_real:.2f}")
            it_tr.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            tbl.setItem(row, 5, it_tr)

            it_ts = QtWidgets.QTableWidgetItem(f"{t_sim:.2f}")
            it_ts.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            tbl.setItem(row, 6, it_ts)

        tbl.resizeColumnsToContents()

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

    














