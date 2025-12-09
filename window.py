# window.py
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QColor, QBrush
from pyvistaqt import BackgroundPlotter
import numpy as np
import os
import pandas as pd
from scipy.ndimage import label, generate_binary_structure
from matplotlib.colors import ListedColormap

from visualize import run, get_2d_clim, make_clusters_lut, compute_cluster_sizes
from load_data import facies, nx, ny, nz, load_facies_from_grdecl
from config import load_facies_colors, load_markers
from analysis import (
    facies_distribution_array,
    reservoir_facies_distribution_array,
    compute_global_metrics_for_array,
    _get_cell_volumes,
    _get_cell_z_coords,
    sample_well_from_grid, 
    calculate_well_accuracy
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
        self.setWindowTitle("Grid View Analysis")
        
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
        # self.fill_unified_facies_table()
        
        # Calcula métricas iniciais para o modelo base
        self.change_reservoir_facies(initial_reservoir)

        # Seleciona o primeiro item da árvore (Base) para inicializar a UI lateral
        top_item = self.project_tree.topLevelItem(0)
        if top_item: 
            top_item.setExpanded(True)
            self.project_tree.setCurrentItem(top_item)

        

    def setup_ui(self, nx, ny, nz):
        self.resize(1600, 900)
        
        # 1. Menu Bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Arquivo")
        action_load = QtWidgets.QAction("Carregar Modelo Adicional...", self)
        action_load.triggered.connect(self.open_compare_dialog)
        action_load_well = QtWidgets.QAction("Carregar Poço (.las + .dev)...", self)
        action_load_well.triggered.connect(self.load_well_dialog)
        file_menu.addAction(action_load_well)
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

        # 2. Toolbar
        self.setup_toolbar_controls()

        # 3. Docks
        self.setup_docks(nx, ny, nz)
        
        # 4. Central Stack
        self.central_stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.central_stack)
        
        # --- PERSPECTIVA 1: VISUALIZAÇÃO (Individual) ---
        self.viz_container = QtWidgets.QStackedWidget() 
        self.tabs = self.viz_container 
        
        self.plotter = BackgroundPlotter(show=False)
        self.viz_tab = QtWidgets.QWidget(); vl = QtWidgets.QVBoxLayout(self.viz_tab); vl.setContentsMargins(0,0,0,0)
        vl.addWidget(self.plotter.interactor); self.viz_container.addWidget(self.viz_tab)
        
        self.plotter_2d = BackgroundPlotter(show=False)
        self.map2d_tab = QtWidgets.QWidget(); ml = QtWidgets.QVBoxLayout(self.map2d_tab); ml.setContentsMargins(0,0,0,0)
        ml.addWidget(self.plotter_2d.interactor); self.viz_container.addWidget(self.map2d_tab)
        
        self.details_tab = QtWidgets.QWidget(); l_det = QtWidgets.QVBoxLayout(self.details_tab)
        self.central_metrics_text = QtWidgets.QTextEdit(); self.central_metrics_text.setReadOnly(True); self.central_metrics_text.setMaximumHeight(150)
        l_det.addWidget(QtWidgets.QLabel("Resumo Global")); l_det.addWidget(self.central_metrics_text)
        self.facies_table = QtWidgets.QTableWidget(); l_det.addWidget(QtWidgets.QLabel("Detalhamento por Fácies")); l_det.addWidget(self.facies_table)
        self.viz_container.addWidget(self.details_tab)
        
        self.central_stack.addWidget(self.viz_container)
        
        # --- PERSPECTIVA 2: COMPARAÇÃO (Stack: 3D vs Relatório) ---
        self.compare_stack = QtWidgets.QStackedWidget()
        
        # PÁGINA 0: Visualização 3D/2D (Grid Layout Dinâmico)
        self.comp_page_viz = QtWidgets.QWidget()
        self.comp_viz_layout = QtWidgets.QVBoxLayout(self.comp_page_viz)
        self.comp_viz_layout.setContentsMargins(0,0,0,0)
        # O conteúdo será injetado por update_dynamic_comparison_view
        self.compare_stack.addWidget(self.comp_page_viz)
        
        # PÁGINA 1: Relatório de Métricas (Tabelas Grandes)
        self.comp_page_metrics = QtWidgets.QWidget()
        self.comp_metrics_layout = QtWidgets.QVBoxLayout(self.comp_page_metrics)
        
        # Abas para organizar as tabelas
        self.tabs_compare_metrics = QtWidgets.QTabWidget()
        
        # Aba Global
        t_glob = QtWidgets.QWidget(); l_glob = QtWidgets.QVBoxLayout(t_glob)
        self.global_compare_table = QtWidgets.QTableWidget(); l_glob.addWidget(self.global_compare_table)
        self.tabs_compare_metrics.addTab(t_glob, "Métricas Globais")
        
        # Aba Fácies
        t_fac = QtWidgets.QWidget(); l_fac = QtWidgets.QVBoxLayout(t_fac)
        self.facies_compare_table = QtWidgets.QTableWidget(); l_fac.addWidget(self.facies_compare_table)
        self.tabs_compare_metrics.addTab(t_fac, "Por Fácies")
        
        # Aba Reservatório
        t_res = QtWidgets.QWidget(); l_res = QtWidgets.QVBoxLayout(t_res)
        self.reservoir_facies_compare_table = QtWidgets.QTableWidget(); l_res.addWidget(self.reservoir_facies_compare_table)
        self.tabs_compare_metrics.addTab(t_res, "Reservatório")
        
        self.comp_metrics_layout.addWidget(self.tabs_compare_metrics)
        self.compare_stack.addWidget(self.comp_page_metrics)
        
        self.central_stack.addWidget(self.compare_stack)

        # PÁGINA 2: Mapas 2D (Verifique se isso existe no seu setup_ui)
        self.comp_page_2d = QtWidgets.QWidget()
        self.comp_2d_layout = QtWidgets.QVBoxLayout(self.comp_page_2d)
        self.comp_2d_layout.setContentsMargins(0,0,0,0)
        self.compare_stack.addWidget(self.comp_page_2d)

    def load_well_dialog(self):
        """Dialogo para selecionar par de arquivos"""
        # 1. Seleciona LAS
        las_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Selecione o arquivo .LAS", "", "LAS Files (*.las)")
        if not las_path: return
        
        # 2. Tenta adivinhar o .dev (mesmo nome, extensão diferente?) ou pede
        base_name = os.path.splitext(las_path)[0]
        suggested_dev = base_name + "_dev" # ou .dev
        
        if os.path.exists(suggested_dev):
            dev_path = suggested_dev
        else:
            dev_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Selecione o arquivo de Trajetória (_dev)", os.path.dirname(las_path), "All Files (*)")
            if not dev_path: return

        well_name = os.path.basename(base_name)
        
        try:
            new_well = Well(well_name, dev_path, las_path)
            if new_well.data is None or new_well.data.empty:
                raise ValueError("Falha ao sincronizar LAS e DEV.")
                
            self.wells[well_name] = new_well
            print(f"Poço {well_name} carregado com {len(new_well.data)} pontos.")
            
            # Atualiza árvore
            self.add_well_to_tree(well_name)
            
            # Plota no 3D imediatamente se estiver na aba visualização
            self.update_wells_3d()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erro", f"Erro ao carregar poço:\n{str(e)}")

    def add_well_to_tree(self, well_name):
        # Cria nó "Poços" se não existir
        root = self.project_tree.invisibleRootItem()
        well_root = None
        for i in range(root.childCount()):
            if root.child(i).text(0) == "Poços":
                well_root = root.child(i)
                break
        
        if not well_root:
            well_root = QtWidgets.QTreeWidgetItem(root, ["Poços"])
            well_root.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_DirIcon))
        
        # Adiciona o poço
        item = QtWidgets.QTreeWidgetItem(well_root, [well_name])
        item.setData(0, QtCore.Qt.UserRole, "well_item")
        item.setData(0, QtCore.Qt.UserRole + 1, well_name)
        item.setCheckState(0, QtCore.Qt.Checked) # Visível por padrão

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
        if not hasattr(self, 'plotter'): return
        
        # Limpa poços antigos
        for name in self.wells.keys():
            self.plotter.remove_actor(f"well_{name}")
            self.plotter.remove_actor(f"marker_{name}")
        self.plotter.remove_actor("well_labels")
        
        z_exag = self.state.get("z_exag", 1.0)
        lbl_pos = []
        lbl_txt = []
        
        for name, well in self.wells.items():
            # Tubo com Z corrigido
            tube = well.get_vtk_polydata(z_exag=z_exag)
            
            if tube:
                # Plota tubo
                self.plotter.add_mesh(
                    tube,
                    scalars="Facies_Real",
                    cmap=self.pv_cmap,  # Suas cores exatas
                    clim=self.clim,     # Seus limites exatos
                    name=f"well_{name}",
                    smooth_shading=False, # False para ver os pixels/cores reais
                    show_scalar_bar=False,
                    interpolate_before_map=False # IMPORTANTE: Não deixa misturar cores
                )
                
                # Prepara etiqueta
                min_md_idx = np.argmin(well.data["DEPT"].values)
                top = well.data.iloc[min_md_idx][["X", "Y", "Z"]].values.copy()
                top[2] *= z_exag
                top[2] -= (50 * z_exag)
                lbl_pos.append(top)
                lbl_txt.append(name)
            
            # Plota Marcadores
            if name in self.markers_db:
                glyphs, _ = well.get_markers_mesh(self.markers_db[name], z_exag=z_exag)
                if glyphs:
                    self.plotter.add_mesh(glyphs, color="red", name=f"marker_{name}")

        # Plota todas as etiquetas de uma vez
        if lbl_pos:
            self.plotter.add_point_labels(
                lbl_pos, lbl_txt, 
                font_size=16, text_color="black", 
                point_size=0, always_visible=True,
                name="well_labels"
            )

    def show_well_comparison_report(self, well_name):
        """
        Gera a janela de 'Barcode' comparando Real vs Simulado.
        Isso deve ser chamado ao clicar num poço na árvore.
        """
        well = self.wells.get(well_name)
        if not well: return
        
        # 1. Obter Grid Ativo (Simulado)
        grid_source = self.state["current_grid_source"]
        
        # 2. Extrair dados simulados
        sim_facies = sample_well_from_grid(well.data, grid_source)
        real_facies = well.data["lito_upscaled"].values # Usa lito_upscaled como verdade
        
        # 3. Calcular Métricas
        acc, total = calculate_well_accuracy(real_facies, sim_facies)
        
        # 4. Criar Janela de Relatório (Matplotlib embutido ou Widgets Qt Puros)
        # Vamos usar Widgets Qt Puros (QGraphicsView ou Pintura) para ficar leve
        # Ou simplesmente abrir um Dialog com Matplotlib
        
        self._open_matplotlib_report(well_name, well.data["DEPT"], real_facies, sim_facies, acc)

    def _open_matplotlib_report(self, well_name, depth, real, sim, accuracy):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Relatório Poço: {well_name} (Acc: {accuracy*100:.1f}%)")
        dialog.resize(600, 800)
        layout = QtWidgets.QVBoxLayout(dialog)
        
        fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(8, 10))
        
        # Cores (Dicionário simplificado, ideal usar o seu config.py)
        # cmap = plt.get_cmap("tab20", 20)
        
        # Track 1: Real
        # Usamos pcolormesh ou imshow expandido
        # Truque: criar uma matriz (N, 1) para plotar como imagem
        real_img = real.reshape(-1, 1)
        sim_img = sim.reshape(-1, 1)
        # Diferença (0 = Igual, 1 = Diferente)
        diff_img = (real != sim).astype(int).reshape(-1, 1)
        
        # Extensão Vertical
        min_d, max_d = depth.min(), depth.max()
        extent = [0, 1, max_d, min_d] # Invertido para profundidade crescer para baixo
        
        ax[0].imshow(real_img, aspect='auto', extent=extent, cmap='tab20', interpolation='nearest')
        ax[0].set_title("Real (Log)")
        ax[0].set_ylabel("Profundidade (m)")
        
        ax[1].imshow(sim_img, aspect='auto', extent=extent, cmap='tab20', interpolation='nearest')
        ax[1].set_title("Simulado (Grid)")
        
        # Track 3: Erro (Branco=Acerto, Vermelho=Erro)
        from matplotlib.colors import ListedColormap
        cmap_err = ListedColormap(['white', 'red'])
        ax[2].imshow(diff_img, aspect='auto', extent=extent, cmap=cmap_err, interpolation='nearest')
        ax[2].set_title("Erro")
        
        plt.tight_layout()
        
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        dialog.exec_()

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
        """Prepara o container para receber o grid dinâmico."""
        # Apenas define um layout base. O conteúdo será injetado por update_dynamic_comparison_view
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Placeholder inicial (opcional)
        label = QtWidgets.QLabel("Selecione os modelos na árvore para comparar.")
        label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(label)

    def setup_toolbar_controls(self):
        # Tenta encontrar a toolbar existente ou cria uma nova
        toolbar = self.findChild(QtWidgets.QToolBar)
        if not toolbar:
            toolbar = self.addToolBar("Controles")
        
        toolbar.setMovable(False)
        toolbar.clear()
        
        toolbar.setIconSize(QtCore.QSize(16, 16))
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        
        # --- Botão Modo (Fácies, Reservatório, etc) ---
        self.btn_mode = QtWidgets.QToolButton(self)
        self.btn_mode.setText("Modo: Fácies") 
        self.btn_mode.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogListView))
        self.btn_mode.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.btn_mode.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.btn_mode.setAutoRaise(True)
        
        menu_mode = QtWidgets.QMenu(self.btn_mode)
        modes = [("Fácies", "facies"), ("Reservatório", "reservoir"), ("Clusters", "clusters"), ("Maior Cluster", "largest"), ("Espessura Local", "thickness_local")]
        for text, data in modes:
            action = menu_mode.addAction(text)
            action.triggered.connect(lambda ch, t=text, d=data: self._update_mode_btn(t, d))
        self.btn_mode.setMenu(menu_mode)
        toolbar.addWidget(self.btn_mode)
        
        toolbar.addSeparator()
        
        # --- Botão Espessura (Métricas Verticais) ---
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
        
        # --- SELETOR DE VISÃO PARA COMPARAÇÃO (ComboBox) ---
        
        self.lbl_comp_view = QtWidgets.QLabel("  Vista Comparação: ")
        self.combo_comp_view = QtWidgets.QComboBox()
        self.combo_comp_view.addItems(["Visualização 3D", "Métricas Comparadas", "Mapas 2D"])
        self.combo_comp_view.currentIndexChanged.connect(self.on_comp_view_changed)
        
        # IMPORTANTE: addWidget retorna uma QAction. Precisamos guardar a referência dela!
        self.act_lbl_comp = toolbar.addWidget(self.lbl_comp_view)
        self.act_combo_comp = toolbar.addWidget(self.combo_comp_view)
        
        # Começam ESCONDIDOS (Visible=False) na Ação
        self.act_lbl_comp.setVisible(False)
        self.act_combo_comp.setVisible(False)
        
        toolbar.addSeparator()

        # --- Botão Snapshot ---
        btn_ss = QtWidgets.QAction("Snapshot", self)
        btn_ss.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        btn_ss.triggered.connect(self.take_snapshot)
        toolbar.addAction(btn_ss)

    def update_dynamic_comparison_2d(self, checked_models):
        """Reconstrói a visualização de Mapas 2D para os modelos selecionados."""
        
        # --- LIMPEZA DE PLOTTERS ANTIGOS ---
        if hasattr(self, 'active_comp_2d_plotters'):
            for p in self.active_comp_2d_plotters: 
                p.close()
        self.active_comp_2d_plotters = []
        
        # Limpa widgets do layout
        if self.comp_2d_layout.count() > 0:
            while self.comp_2d_layout.count():
                item = self.comp_2d_layout.takeAt(0)
                if item.widget(): item.widget().deleteLater()
                elif item.layout():
                    import sip
                    sip.delete(item.layout())

        if not checked_models:
            self.comp_2d_layout.addWidget(QtWidgets.QLabel("Selecione modelos na árvore à esquerda."))
            return

        # --- GRID LAYOUT ---
        n_models = len(checked_models)
        cols = 2 if n_models > 1 else 1
        
        # Container para isolar o grid visual
        grid_container = QtWidgets.QWidget()
        grid_layout = QtWidgets.QGridLayout(grid_container)
        grid_layout.setContentsMargins(0,0,0,0)
        grid_layout.setSpacing(2)
        self.comp_2d_layout.addWidget(grid_container)
        
        # Recupera configuração de espessura (Tipo de Mapa)
        presets = self.state.get("thickness_presets") or {}
        thick_mode = self.state.get("thickness_mode", "Espessura")
        
        # Fallback se o preset não existir
        if thick_mode not in presets: thick_mode = "Espessura"
        if thick_mode in presets:
            scalar, title = presets[thick_mode]
        else:
            scalar, title = ("vert_Ttot_reservoir", "Espessura")
        
        from load_data import grid as global_grid
        
        for idx, (model_key, model_name) in enumerate(checked_models):
            row, col = idx // cols, idx % cols
            model_data = self.models[model_key]
            
            # Cria Plotter 2D
            p2d = BackgroundPlotter(show=False)
            self.active_comp_2d_plotters.append(p2d)
            
            # Precisamos calcular as métricas no grid temporário
            # pois cada modelo tem sua distribuição de fácies
            temp_grid = global_grid.copy(deep=True)
            temp_grid.cell_data["Facies"] = model_data["facies"]
            
            # Recalcula métricas verticais para este modelo
            self.recalc_vertical_metrics(temp_grid, model_data["facies"], model_data["reservoir_facies"])
            
            # Desenha o Mapa usando a função auxiliar existente
            try:
                self._draw_2d_map_local(p2d, temp_grid, scalar, title)
            except Exception as e:
                print(f"Erro ao desenhar mapa 2D para {model_name}: {e}")
            
            # Monta o Widget da célula
            w = QtWidgets.QWidget()
            vl = QtWidgets.QVBoxLayout(w)
            vl.setContentsMargins(0,0,0,0)
            vl.setSpacing(0)
            
            lbl = QtWidgets.QLabel(f"  {model_name} ({thick_mode})")
            lbl.setStyleSheet("background: #ddd; font-weight: bold; padding: 2px;")
            
            vl.addWidget(lbl)
            vl.addWidget(p2d.interactor)
            
            grid_layout.addWidget(w, row, col)

        # Atualiza a tabela lateral também (Filtros afetam o cálculo de espessura)
        self._build_multi_model_filter_table(checked_models)

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
        # --- DOCK EXPLORER (Hierarquia) - ESQUERDA ---
        self.dock_explorer = QtWidgets.QDockWidget("Project Explorer", self)
        self.dock_explorer.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.dock_explorer.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | 
                                       QtWidgets.QDockWidget.DockWidgetFloatable | 
                                       QtWidgets.QDockWidget.DockWidgetClosable)
        
        self.project_tree = QtWidgets.QTreeWidget()
        self.project_tree.setHeaderLabel("Hierarquia")
        self.project_tree.itemDoubleClicked.connect(self.on_tree_double_clicked)
        self.project_tree.itemSelectionChanged.connect(self.on_tree_selection_changed)
        self.project_tree.itemChanged.connect(self.on_tree_item_changed)
        
        self.dock_explorer.setWidget(self.project_tree)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dock_explorer)
        
        self.add_model_to_tree("base", "Modelo Base")

        # --- DOCK PROPRIEDADES (Contextual) - AGORA NA ESQUERDA ---
        self.dock_props = QtWidgets.QDockWidget("Propriedades", self)
        self.dock_props.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.dock_props.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | 
                                    QtWidgets.QDockWidget.DockWidgetFloatable | 
                                    QtWidgets.QDockWidget.DockWidgetClosable)
        
        self.props_stack = QtWidgets.QStackedWidget()
        
        # Pág 0: Vazio
        self.props_stack.addWidget(QtWidgets.QLabel("Selecione um item na árvore.")) 
        
        # Pág 1: Geometria
        self.page_grid = QtWidgets.QWidget(); pg_layout = QtWidgets.QVBoxLayout(self.page_grid)
        self.slicer_widget = GridSlicerWidget(nx, ny, nz, self.on_ui_slice_changed)
        pg_layout.addWidget(self.slicer_widget); pg_layout.addStretch()
        self.props_stack.addWidget(self.page_grid)
        
        # Pág 2: Propriedades Visualização
        self.page_props = QtWidgets.QWidget()
        pp_layout = QtWidgets.QVBoxLayout(self.page_props); pp_layout.setContentsMargins(2,2,2,2)
        
        self.legend_group = QtWidgets.QGroupBox("Legenda & Filtro")
        lgl = QtWidgets.QVBoxLayout(self.legend_group); lgl.setContentsMargins(2,5,2,2)
        
        self.facies_legend_table = QtWidgets.QTableWidget(); self.facies_legend_table.setColumnCount(4)
        self.facies_legend_table.setHorizontalHeaderLabels(["Cor", "ID", "N", "Res"])
        self.facies_legend_table.verticalHeader().setVisible(False)
        self.facies_legend_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.facies_legend_table.itemChanged.connect(self.on_legend_item_changed)
        lgl.addWidget(self.facies_legend_table)
        
        self.clusters_legend_table = QtWidgets.QTableWidget(); self.clusters_legend_table.setColumnCount(3)
        self.clusters_legend_table.setHorizontalHeaderLabels(["Cor", "ID", "Células"])
        self.clusters_legend_table.verticalHeader().setVisible(False)
        self.clusters_legend_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.clusters_legend_table.setVisible(False)
        lgl.addWidget(self.clusters_legend_table)
        
        pp_layout.addWidget(self.legend_group)
        self.props_stack.addWidget(self.page_props)
        
        # Pág 3: Comparação
        self.page_compare = self.setup_comparison_dock_content()
        self.props_stack.addWidget(self.page_compare)
        
        self.dock_props.setWidget(self.props_stack)
        
        # ADICIONA NA ESQUERDA
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dock_props)
        
        # FORÇA LADO A LADO (Explorer | Propriedades)
        self.splitDockWidget(self.dock_explorer, self.dock_props, QtCore.Qt.Horizontal)
        
        # Define larguras iniciais
        self.resizeDocks([self.dock_explorer, self.dock_props], [250, 350], QtCore.Qt.Horizontal)

    def add_model_to_tree(self, model_key, model_name):
        root_item = QtWidgets.QTreeWidgetItem(self.project_tree, [model_name])
        root_item.setData(0, QtCore.Qt.UserRole, "model_root")
        root_item.setData(0, QtCore.Qt.UserRole + 1, model_key)
        root_item.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_DirHomeIcon))
        
        # Verifica se estamos no modo Comparação
        is_comparison = False
        if hasattr(self, 'central_stack'):
            if self.central_stack.currentIndex() == 1:
                is_comparison = True
        
        # Usa a função auxiliar para configurar o checkbox corretamente
        self._set_item_checkbox_visible(root_item, is_comparison)
        
        # Se estamos criando no modo comparação, já marcamos ele como Checked por conveniência
        if is_comparison:
            root_item.setCheckState(0, QtCore.Qt.Checked)

        root_item.setExpanded(True)
        
        # --- Sub-itens (Mantidos iguais) ---
        item_grid = QtWidgets.QTreeWidgetItem(root_item, ["Geometria (Grid)"])
        item_grid.setData(0, QtCore.Qt.UserRole, "grid_settings")
        item_grid.setData(0, QtCore.Qt.UserRole + 1, model_key)
        item_grid.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView))
        
        item_props = QtWidgets.QTreeWidgetItem(root_item, ["Propriedades & Filtros"])
        item_props.setData(0, QtCore.Qt.UserRole, "prop_settings")
        item_props.setData(0, QtCore.Qt.UserRole + 1, model_key)
        item_props.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogListView))

        item_metrics = QtWidgets.QTreeWidgetItem(root_item, ["Métricas & Estatísticas"])
        item_metrics.setData(0, QtCore.Qt.UserRole, "metrics_view")
        item_metrics.setData(0, QtCore.Qt.UserRole + 1, model_key)
        item_metrics.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogInfoView))

        item_2d = QtWidgets.QTreeWidgetItem(root_item, ["Mapas 2D"])
        item_2d.setData(0, QtCore.Qt.UserRole, "map2d_view")
        item_2d.setData(0, QtCore.Qt.UserRole + 1, model_key)
        item_2d.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogContentsView))

    # --- LÓGICA DE INTERAÇÃO TREE ---

    def on_tree_double_clicked(self, item, col):
        """Duplo clique em Geometria força a troca para aba 3D."""
        role = item.data(0, QtCore.Qt.UserRole)
        model_key = item.data(0, QtCore.Qt.UserRole + 1)
        data = item.data(0, QtCore.Qt.UserRole + 1)

        if role == "well_item":
            self.show_well_comparison_report(data)
        
        if role == "grid_settings" and model_key:
            self.switch_main_view_to_model(model_key)
            self.tabs.setCurrentIndex(0) # Força ir para 3D

    def switch_main_view_to_model(self, model_key):
        """Carrega grid, restaura filtros e modo de visualização específicos do modelo."""
        # Proteção se o modelo não existir
        if model_key not in self.models: return
        
        target_facies = self.models[model_key]["facies"]
        if target_facies is None: return

        from load_data import grid as global_grid
        from scipy.ndimage import label, generate_binary_structure
        
        # 1. Configurações
        saved_mode = self.models[model_key].get("view_mode", "facies")
        current_res_set = self.models[model_key]["reservoir_facies"]
        
        self.state["current_facies"] = target_facies
        self.state["reservoir_facies"] = current_res_set
        self.state["mode"] = saved_mode 

        if hasattr(self, "btn_mode"):
            labels = {"facies": "Fácies", "reservoir": "Reservatório", "clusters": "Clusters", 
                      "largest": "Maior Cluster", "thickness_local": "Espessura Local"}
            self.btn_mode.setText(f"Modo: {labels.get(saved_mode, saved_mode)}")

        # 2. Prepara Grid
        # SE NÃO FOR BASE, É COMPARAÇÃO (Cópia)
        if model_key != "base":
            active_grid = global_grid.copy(deep=True)
            active_grid.cell_data["Facies"] = target_facies
        else:
            active_grid = global_grid
            active_grid.cell_data["Facies"] = facies # Original
            
        # 3. Recalcula Propriedades
        is_res = np.isin(target_facies, list(current_res_set)).astype(np.uint8)
        active_grid.cell_data["Reservoir"] = is_res
        
        arr_3d = is_res.reshape((nx, ny, nz), order="F")
        structure = generate_binary_structure(3, 1)
        labeled, _ = label(arr_3d.transpose(2, 1, 0), structure=structure)
        clusters_1d = labeled.transpose(2, 1, 0).reshape(-1, order="F").astype(np.int32)
        active_grid.cell_data["Clusters"] = clusters_1d
        
        counts = np.bincount(clusters_1d.ravel())
        if counts.size > 0: counts[0] = 0
        largest_lbl = counts.argmax() if counts.size > 0 else 0
        active_grid.cell_data["LargestCluster"] = (clusters_1d == largest_lbl).astype(np.uint8)

        # 4. Recalcula Espessura
        self.recalc_vertical_metrics(active_grid, target_facies, current_res_set)

        # 5. Atualiza Cores e Estado
        lut, rng = make_clusters_lut(clusters_1d)
        self.state["clusters_lut"] = lut
        self.state["clusters_rng"] = rng
        self.state["clusters_sizes"] = compute_cluster_sizes(clusters_1d)

        self.state["current_grid_source"] = active_grid
        self.state["refresh"]()
        
        if saved_mode == "clusters": self.populate_clusters_legend()
        else: self.populate_facies_legend()
            
        self.update_sidebar_metrics_text(model_key)
        self.update_2d_map()

    def on_tree_selection_changed(self):
        items = self.project_tree.selectedItems()
        if not items:
            # Se não tiver nada selecionado e não for comparação, mostra vazio
            if self.central_stack.currentIndex() == 0:
                self.props_stack.setCurrentIndex(0)
            return
            
        item = items[0]
        role = item.data(0, QtCore.Qt.UserRole)
        model_key = item.data(0, QtCore.Qt.UserRole + 1)
        
        # Verifica se estamos no modo Comparação
        is_comparison_mode = (self.central_stack.currentIndex() == 1)
        
        # --- LÓGICA DO PAINEL LATERAL (COLUNA 2) ---
        if is_comparison_mode:
            # No modo comparação, a coluna 2 é SEMPRE o Painel de Comparação
            # independente do que você clica (a menos que queira ver geometria, mas Dionisos geralmente fixa)
            self.props_stack.setCurrentWidget(self.page_compare)
            self.dock_props.setWindowTitle("Painel de Comparação")
        else:
            # Modo Visualização: A coluna 2 reage ao clique
            if role == "grid_settings":
                self.props_stack.setCurrentWidget(self.page_grid)
                self.dock_props.setWindowTitle("Geometria")
            else:
                self.props_stack.setCurrentWidget(self.page_props)
                self.dock_props.setWindowTitle("Propriedades")
            
        # --- LÓGICA DA ÁREA CENTRAL ---
        if model_key:
            self.update_sidebar_metrics_text(model_key)

            if not is_comparison_mode:
                if role == "metrics_view":
                    self.viz_container.setCurrentIndex(2)
                    self.update_metrics_view_content(model_key)
                elif role == "map2d_view":
                    self.switch_main_view_to_model(model_key)
                    self.viz_container.setCurrentIndex(1)
                    self.update_2d_map()
                elif role in ["grid_settings", "prop_settings", "model_root"]:
                    self.switch_main_view_to_model(model_key)
                    self.viz_container.setCurrentIndex(0)

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
        try: fac_compare = load_facies_from_grdecl(grdecl_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erro", str(e)); return
        
        if fac_compare.size != nx * ny * nz:
             QtWidgets.QMessageBox.warning(self, "Erro", "Grid incompatível"); return
        
        import time
        model_id = f"compare_{int(time.time() * 1000)}"
        model_name = os.path.basename(grdecl_path)
        
        # --- CORREÇÃO: Inicia com conjunto VAZIO para total independência ---
        # (Antes copiava do base, o que causava confusão de 'sincronia')
        rf = set() 
        
        self.models[model_id] = {
            "name": model_name,
            "facies": fac_compare,
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
        
        # self.fill_unified_facies_table()
        self.update_comparison_tables()
        
        if hasattr(self, "comp_plotter_comp"):
            # self.init_compare_3d()
            pass
    
        # Força atualização da visualização dinâmica se estiver na aba de comparação
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
            self.act_persp_viz.setChecked(True)
            self.act_persp_comp.setChecked(False)
            
            # --- Esconde as AÇÕES da toolbar (Label e Combo) ---
            if hasattr(self, 'act_combo_comp'): 
                self.act_combo_comp.setVisible(False)
            if hasattr(self, 'act_lbl_comp'): 
                self.act_lbl_comp.setVisible(False)

            # 1. Central: Visualização Padrão
            self.central_stack.setCurrentIndex(0) 
            
            # 2. Árvore: Modo Normal (Expansível, sem Checkboxes)
            self.project_tree.setItemsExpandable(True)
            self.project_tree.expandAll()
            self.toggle_tree_checkboxes(False)
            self.dock_explorer.setWindowTitle("Project Explorer")
            
            # 3. Propriedades: Modo Normal
            self.dock_props.setWindowTitle("Propriedades")
            # Força atualização para restaurar o widget correto (Grid ou Props)
            self.on_tree_selection_changed()
            
        elif mode == "comparison":
            self.act_persp_viz.setChecked(False)
            self.act_persp_comp.setChecked(True)
            
            # --- Mostra as AÇÕES da toolbar ---
            if hasattr(self, 'act_combo_comp'): 
                self.act_combo_comp.setVisible(True)
            if hasattr(self, 'act_lbl_comp'): 
                self.act_lbl_comp.setVisible(True)

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
        """Reconstrói a visualização 3D mantendo a posição da câmera e o MODO DE ESPESSURA."""
        
        if checked_models is None:
            checked_models = []
            iterator = QtWidgets.QTreeWidgetItemIterator(self.project_tree)
            while iterator.value():
                item = iterator.value()
                if item.data(0, QtCore.Qt.UserRole) == "model_root":
                    if item.checkState(0) == QtCore.Qt.Checked:
                        checked_models.append((item.data(0, QtCore.Qt.UserRole + 1), item.text(0)))
                iterator += 1

        # --- SALVA CÂMERA ---
        saved_camera = None
        if hasattr(self, 'active_comp_plotters') and len(self.active_comp_plotters) > 0:
            try:
                cam = self.active_comp_plotters[0].camera
                saved_camera = {
                    "position": cam.position,
                    "focal_point": cam.focal_point,
                    "view_up": cam.up,
                    "view_angle": cam.view_angle,
                    "clipping_range": cam.clipping_range
                }
            except Exception: pass

        # --- LIMPEZA ---
        if hasattr(self, 'active_comp_plotters'):
            for p in self.active_comp_plotters: p.close()
        self.active_comp_plotters = []
        self.active_comp_states = []

        if self.comp_viz_layout.count() > 0:
            while self.comp_viz_layout.count():
                item = self.comp_viz_layout.takeAt(0)
                if item.widget(): item.widget().deleteLater()
                elif item.layout():
                    import sip
                    sip.delete(item.layout())

        if not checked_models:
            self.comp_viz_layout.addWidget(QtWidgets.QLabel("Selecione modelos."))
            return

        # --- CRIAÇÃO 3D ---
        n_models = len(checked_models)
        cols = 2 if n_models > 1 else 1
        
        grid_container = QtWidgets.QWidget()
        grid_layout = QtWidgets.QGridLayout(grid_container)
        grid_layout.setContentsMargins(0,0,0,0); grid_layout.setSpacing(2)
        self.comp_viz_layout.addWidget(grid_container)
        
        from visualize import run
        from load_data import grid as global_grid
        
        for idx, (model_key, model_name) in enumerate(checked_models):
            row, col = idx // cols, idx % cols
            model_data = self.models[model_key]
            
            plotter = BackgroundPlotter(show=False)
            self.active_comp_plotters.append(plotter)
            
            temp_grid = global_grid.copy(deep=True)
            temp_grid.cell_data["Facies"] = model_data["facies"]
            
            # Estado Local
            local_state = {"model_key": model_key}
            
            # --- CORREÇÃO AQUI: Passa o modo de espessura atual para o estado local ---
            local_state["thickness_mode"] = self.state.get("thickness_mode", "Espessura")
            
            if hasattr(self, 'comp_slicer'):
                local_state["z_exag"] = self.comp_slicer.spin_z.value()
                local_state["k_min"] = self.comp_slicer.k_widgets['slider_min'].value()
                local_state["k_max"] = self.comp_slicer.k_widgets['slider_max'].value()
                local_state["i_min"] = self.comp_slicer.i_widgets['slider_min'].value()
                local_state["i_max"] = self.comp_slicer.i_widgets['slider_max'].value()
                local_state["j_min"] = self.comp_slicer.j_widgets['slider_min'].value()
                local_state["j_max"] = self.comp_slicer.j_widgets['slider_max'].value()
            
            run(mode=self.state.get("mode", "facies"), 
                external_plotter=plotter, external_state=local_state, 
                target_grid=temp_grid, target_facies=model_data["facies"])
            
            if "update_reservoir_fields" in local_state:
                local_state["update_reservoir_fields"](model_data["reservoir_facies"])
                if "refresh" in local_state: 
                    local_state["refresh"]()
            
            self.active_comp_states.append(local_state)
            
            w = QtWidgets.QWidget(); vl = QtWidgets.QVBoxLayout(w); vl.setContentsMargins(0,0,0,0); vl.setSpacing(0)
            lbl = QtWidgets.QLabel(f"  {model_name}"); lbl.setStyleSheet("background: #ddd; font-weight: bold;")
            vl.addWidget(lbl); vl.addWidget(plotter.interactor)
            grid_layout.addWidget(w, row, col)

        # --- RESTAURA CÂMERA ---
        if saved_camera:
            for p in self.active_comp_plotters:
                p.camera.position = saved_camera["position"]
                p.camera.focal_point = saved_camera["focal_point"]
                p.camera.up = saved_camera["view_up"]
                p.camera.view_angle = saved_camera["view_angle"]
                p.camera.clipping_range = saved_camera["clipping_range"]
                p.render()

        if len(self.active_comp_plotters) > 1:
            self.sync_multi_cameras(self.active_comp_plotters)

        self._build_multi_model_filter_table(checked_models)

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
        data = item.data(QtCore.Qt.UserRole)
        if not data: return
        
        model_key, fac = data
        is_checked = (item.checkState() == QtCore.Qt.Checked)
        
        target_set = self.models[model_key]["reservoir_facies"]
        if is_checked: target_set.add(fac)
        else: target_set.discard(fac)
        
        # 1. Atualiza visualização 3D (se houver estados ativos)
        if hasattr(self, 'active_comp_states'):
            for state in self.active_comp_states:
                if state.get("model_key") == model_key:
                    if "update_reservoir_fields" in state:
                        state["update_reservoir_fields"](target_set)
                    if "refresh" in state: state["refresh"]()
                    break
        
        # 2. --- CORREÇÃO: Atualiza Mapas 2D se estiverem visíveis ---
        # Verifica se estamos na Perspectiva Comparação (1) e na Aba 2D (2)
        if self.central_stack.currentIndex() == 1 and self.compare_stack.currentIndex() == 2:
            self.refresh_comparison_active_view()
    
    def on_comp_view_changed(self, index):
        """Muda a página do Stack de Comparação baseado no ComboBox."""
        # O índice 0 é 3D, 1 é Métricas, 2 é Mapas 2D
        # Se a página 2 não existir no stack, nada acontece.
        if index < self.compare_stack.count():
            self.compare_stack.setCurrentIndex(index)
            # Força a atualização da view que acabou de ser selecionada
            self.refresh_comparison_active_view()

    def refresh_comparison_active_view(self):
        """Identifica qual view está ativa e chama a função de update correspondente."""
        # Se não estiver no modo comparação global, ignora
        if self.central_stack.currentIndex() != 1: return

        idx = self.compare_stack.currentIndex()
        
        # Coleta modelos marcados na árvore
        checked_models = []
        iterator = QtWidgets.QTreeWidgetItemIterator(self.project_tree)
        while iterator.value():
            item = iterator.value()
            if item.data(0, QtCore.Qt.UserRole) == "model_root":
                if item.checkState(0) == QtCore.Qt.Checked:
                    checked_models.append((item.data(0, QtCore.Qt.UserRole + 1), item.text(0)))
            iterator += 1

        if idx == 0: # 3D Grid
            self.update_dynamic_comparison_view(checked_models)
        elif idx == 1: # Relatório de Métricas
            self.update_comparison_tables_multi(checked_models)
            # Atualiza também o filtro lateral pois ele é útil para ver dados
            self._build_multi_model_filter_table(checked_models)
        elif idx == 2: # Mapas 2D
            self.update_dynamic_comparison_2d(checked_models)
    
    def update_dynamic_comparison_2d(self, checked_models):
        """Reconstrói a visualização de Mapas 2D."""
        
        # --- LIMPEZA ---
        # Fecha plotters 2D antigos se houver
        if hasattr(self, 'active_comp_2d_plotters'):
            for p in self.active_comp_2d_plotters: p.close()
        self.active_comp_2d_plotters = []
        
        while self.comp_2d_layout.count():
            item = self.comp_2d_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        if not checked_models:
            self.comp_2d_layout.addWidget(QtWidgets.QLabel("Selecione modelos."))
            return

        # --- GRID LAYOUT ---
        n_models = len(checked_models)
        cols = 2 if n_models > 1 else 1
        
        grid_container = QtWidgets.QWidget()
        grid_layout = QtWidgets.QGridLayout(grid_container)
        grid_layout.setContentsMargins(0,0,0,0); grid_layout.setSpacing(2)
        self.comp_2d_layout.addWidget(grid_container)
        
        # Recupera configuração de espessura
        presets = self.state.get("thickness_presets") or {}
        thick_mode = self.state.get("thickness_mode", "Espessura")
        if thick_mode not in presets: thick_mode = "Espessura"
        scalar, title = presets[thick_mode]
        
        from load_data import grid as global_grid
        
        for idx, (model_key, model_name) in enumerate(checked_models):
            row, col = idx // cols, idx % cols
            model_data = self.models[model_key]
            
            p2d = BackgroundPlotter(show=False)
            self.active_comp_2d_plotters.append(p2d)
            
            # Precisamos calcular as métricas no grid temporário
            temp_grid = global_grid.copy(deep=True)
            temp_grid.cell_data["Facies"] = model_data["facies"]
            
            # Recalcula métricas verticais para este modelo
            self.recalc_vertical_metrics(temp_grid, model_data["facies"], model_data["reservoir_facies"])
            
            # Desenha
            self._draw_2d_map_local(p2d, temp_grid, scalar, f"{model_name} - {title}")
            
            w = QtWidgets.QWidget(); vl = QtWidgets.QVBoxLayout(w); vl.setContentsMargins(0,0,0,0); vl.setSpacing(0)
            lbl = QtWidgets.QLabel(f"  {model_name} ({thick_mode})"); lbl.setStyleSheet("background: #ddd; font-weight: bold;")
            vl.addWidget(lbl); vl.addWidget(p2d.interactor)
            grid_layout.addWidget(w, row, col)

        # Atualiza a tabela lateral também para permitir filtro
        self._build_multi_model_filter_table(checked_models)

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