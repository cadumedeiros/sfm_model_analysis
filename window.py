# window.py
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QColor, QBrush
from pyvistaqt import BackgroundPlotter
import numpy as np
import os
import pandas as pd
import json
import re
from scipy.ndimage import label, generate_binary_structure
from matplotlib.colors import ListedColormap

from visualize import run, get_2d_clim
from load_data import facies, nx, ny, nz
from config import load_facies_colors, load_facies_reference, load_markers

from analysis import (
    facies_distribution_array,
    reservoir_facies_distribution_array,
    _get_cell_volumes,
    _get_cell_z_coords,
    compute_vertical_metrics_for_grid,
    get_vertical_metric_presets,
    is_vertical_metric_normalized_name,
)
from wells import Well

# --- WIDGET CUSTOMIZADO PARA OS SLIDERS (Grid Explorer) ---
class GridSlicerWidget(QtWidgets.QGroupBox):
    def __init__(self, nx, ny, nz, callback, initial_z=1.0):
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

class FaciesGroupingDialog(QtWidgets.QDialog):
    """Diálogo para configurar agrupamentos (grupos) de fácies a partir do color_reference_facies.txt.

    A ideia é: cada fácies original recebe um ID de "grupo" (que também é um ID de fácies existente na referência).
    O resto do software pode então operar sobre a fácies agrupada como se fosse uma fácies normal.

    Formato de persistência (JSON):
        {
        "version": 1,
        "mapping": {"231": 23, "232": 23, ...}
        }
    """

    def __init__(self, facies_reference, colors_dict, current_mapping=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuração de Grupos de Fácies")
        self.resize(720, 520)

        # lista ordenada [(facies, rgba), ...]
        self.facies_reference = list(facies_reference or [])
        self.colors_dict = dict(colors_dict or {})

        self.facies_ids = [int(f) for f, _ in self.facies_reference]
        self.allowed_groups = set(self.facies_ids)

        # mapping {orig:int -> group:int}
        if current_mapping is None:
            self.mapping = {int(fid): int(fid) for fid in self.facies_ids}
        else:
            self.mapping = {int(k): int(v) for k, v in dict(current_mapping).items()}
            # completa faltantes com identidade
            for fid in self.facies_ids:
                self.mapping.setdefault(int(fid), int(fid))

        self._building = False

        layout = QtWidgets.QVBoxLayout(self)

        # Filtro + ações em massa
        top = QtWidgets.QHBoxLayout()
        self.txt_filter = QtWidgets.QLineEdit()
        self.txt_filter.setPlaceholderText("Filtrar fácies... (ex.: 23, 231, 11)")
        self.txt_filter.textChanged.connect(self._apply_filter)
        top.addWidget(self.txt_filter, 1)

        self.cmb_mass_group = QtWidgets.QComboBox()
        self.cmb_mass_group.setToolTip("Grupo a aplicar nas linhas selecionadas")
        for fid in self.facies_ids:
            self.cmb_mass_group.addItem(str(fid), int(fid))
        top.addWidget(self.cmb_mass_group)

        btn_apply = QtWidgets.QPushButton("Aplicar ao selecionado")
        btn_apply.clicked.connect(self._apply_group_to_selection)
        top.addWidget(btn_apply)

        layout.addLayout(top)

        # Tabela
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Cor", "Fácies", "Grupo", "Cor Grupo"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table, 1)

        # Botões
        btns = QtWidgets.QHBoxLayout()
        btn_reset_sel = QtWidgets.QPushButton("Reset selecionados")
        btn_reset_sel.clicked.connect(self._reset_selected)
        btns.addWidget(btn_reset_sel)

        btn_reset_all = QtWidgets.QPushButton("Reset tudo")
        btn_reset_all.clicked.connect(self._reset_all)
        btns.addWidget(btn_reset_all)

        btns.addStretch(1)

        btn_load = QtWidgets.QPushButton("Carregar...")
        btn_load.clicked.connect(self._load_json)
        btns.addWidget(btn_load)

        btn_save = QtWidgets.QPushButton("Salvar...")
        btn_save.clicked.connect(self._save_json)
        btns.addWidget(btn_save)

        btn_ok = QtWidgets.QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        btns.addWidget(btn_ok)

        btn_cancel = QtWidgets.QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        btns.addWidget(btn_cancel)

        layout.addLayout(btns)

        self._populate_table()

    def get_mapping(self):
        # garante ints
        return {int(k): int(v) for k, v in self.mapping.items()}

    # ---------- UI helpers ----------
    def _populate_table(self):
        self._building = True
        try:
            self.table.setRowCount(len(self.facies_ids))
            for row, fid in enumerate(self.facies_ids):
                # Cor original
                rgba = self.colors_dict.get(int(fid), (0.8, 0.8, 0.8, 1.0))
                c = QColor(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
                item_c = QtWidgets.QTableWidgetItem()
                item_c.setBackground(QBrush(c))
                item_c.setFlags(QtCore.Qt.ItemIsEnabled)
                self.table.setItem(row, 0, item_c)

                # ID facies
                item_id = QtWidgets.QTableWidgetItem(str(int(fid)))
                item_id.setData(QtCore.Qt.UserRole, int(fid))
                item_id.setFlags(QtCore.Qt.ItemIsEnabled)
                self.table.setItem(row, 1, item_id)

                # Grupo (editável)
                g = int(self.mapping.get(int(fid), int(fid)))
                item_g = QtWidgets.QTableWidgetItem(str(g))
                item_g.setData(QtCore.Qt.UserRole, int(fid))
                item_g.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsSelectable)
                self.table.setItem(row, 2, item_g)

                # Cor do grupo
                self._set_group_color_cell(row, g)

        finally:
            self._building = False

        self.table.resizeColumnsToContents()

    def _set_group_color_cell(self, row, group_id):
        rgba_g = self.colors_dict.get(int(group_id), (0.6, 0.6, 0.6, 1.0))
        cg = QColor(int(rgba_g[0]*255), int(rgba_g[1]*255), int(rgba_g[2]*255))
        item_cg = QtWidgets.QTableWidgetItem()
        item_cg.setBackground(QBrush(cg))
        item_cg.setFlags(QtCore.Qt.ItemIsEnabled)
        self.table.setItem(row, 3, item_cg)

    def _apply_filter(self, text):
        t = (text or "").strip()
        for row in range(self.table.rowCount()):
            fid = self.table.item(row, 1).text()
            show = (t == "") or (t in fid) or (t in self.table.item(row, 2).text())
            self.table.setRowHidden(row, not show)

    def _apply_group_to_selection(self):
        group_id = int(self.cmb_mass_group.currentData())
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return
        self._building = True
        try:
            for idx in sel:
                row = idx.row()
                fid = int(self.table.item(row, 1).text())
                self.mapping[fid] = group_id
                self.table.item(row, 2).setText(str(group_id))
                self._set_group_color_cell(row, group_id)
        finally:
            self._building = False

    def _reset_selected(self):
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return
        self._building = True
        try:
            for idx in sel:
                row = idx.row()
                fid = int(self.table.item(row, 1).text())
                self.mapping[fid] = fid
                self.table.item(row, 2).setText(str(fid))
                self._set_group_color_cell(row, fid)
        finally:
            self._building = False

    def _reset_all(self):
        self.mapping = {int(fid): int(fid) for fid in self.facies_ids}
        self._populate_table()

    def _on_item_changed(self, item):
        if self._building:
            return
        # Coluna 2 é grupo editável
        if item.column() != 2:
            return
        try:
            fid = int(item.data(QtCore.Qt.UserRole))
        except Exception:
            # fallback: usa coluna 1
            try:
                fid = int(self.table.item(item.row(), 1).text())
            except Exception:
                return

        raw = (item.text() or "").strip()
        try:
            gid = int(raw)
        except Exception:
            gid = fid

        self.mapping[int(fid)] = int(gid)
        self._set_group_color_cell(item.row(), int(gid))

    # ---------- Persistência ----------
    def _load_json(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Carregar configuração", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            mapping = data.get("mapping", {}) if isinstance(data, dict) else {}
            # converte chaves
            new_map = {}
            for k, v in mapping.items():
                try:
                    new_map[int(k)] = int(v)
                except Exception:
                    pass
            # completa faltantes
            for fid in self.facies_ids:
                new_map.setdefault(int(fid), int(fid))
            self.mapping = new_map
            self._populate_table()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Erro", f"Falha ao carregar configuração:\n{e}")

    def _save_json(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Salvar configuração", "", "JSON (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            data = {"version": 1, "mapping": {str(k): int(v) for k, v in self.get_mapping().items()}}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Erro", f"Falha ao salvar configuração:\n{e}")

class ProportionPropsDialog(QtWidgets.QDialog):
    """Diálogo para o usuário marcar quais propriedades são proporções (0–1)."""

    def __init__(self, prop_names, current_set=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Propriedades proporcionais por célula")
        self.resize(520, 520)

        self._prop_names = list(prop_names or [])
        self._prop_names.sort(key=lambda s: str(s).lower())
        self._current = set(current_set or set())

        layout = QtWidgets.QVBoxLayout(self)

        info = QtWidgets.QLabel(
            "Marque as propriedades que representam proporções (0 a 1).\n"
            "Na aba Comparação, estas propriedades terão legenda fixada em 0–1."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.txt_filter = QtWidgets.QLineEdit()
        self.txt_filter.setPlaceholderText("Filtrar... (ex.: Sand, NTG, Qv, ICV)")
        layout.addWidget(self.txt_filter)

        self.listw = QtWidgets.QListWidget()
        self.listw.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        layout.addWidget(self.listw, 1)

        btns_top = QtWidgets.QHBoxLayout()
        self.btn_all = QtWidgets.QPushButton("Marcar tudo")
        self.btn_none = QtWidgets.QPushButton("Desmarcar tudo")
        btns_top.addWidget(self.btn_all)
        btns_top.addWidget(self.btn_none)
        btns_top.addStretch(1)
        layout.addLayout(btns_top)

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        self.btn_ok = QtWidgets.QPushButton("OK")
        self.btn_cancel = QtWidgets.QPushButton("Cancelar")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        btns.addWidget(self.btn_ok)
        btns.addWidget(self.btn_cancel)
        layout.addLayout(btns)

        self._rebuild_list()

        self.txt_filter.textChanged.connect(self._rebuild_list)
        self.btn_all.clicked.connect(self._mark_all)
        self.btn_none.clicked.connect(self._mark_none)

    def _rebuild_list(self):
        self.listw.blockSignals(True)
        try:
            self.listw.clear()
            flt = (self.txt_filter.text() or "").strip().lower()

            for name in self._prop_names:
                s = str(name)
                if flt and flt not in s.lower():
                    continue

                it = QtWidgets.QListWidgetItem(s)
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                it.setCheckState(QtCore.Qt.Checked if s in self._current else QtCore.Qt.Unchecked)
                self.listw.addItem(it)
        finally:
            self.listw.blockSignals(False)

    def _mark_all(self):
        for i in range(self.listw.count()):
            self.listw.item(i).setCheckState(QtCore.Qt.Checked)

    def _mark_none(self):
        for i in range(self.listw.count()):
            self.listw.item(i).setCheckState(QtCore.Qt.Unchecked)

    def get_selected(self):
        out = set()
        for i in range(self.listw.count()):
            it = self.listw.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                out.add(str(it.text()))
        return out


class MainWindow(QtWidgets.QMainWindow):

    # ------------------------------------------------------------------
    # Compat: open_compare_dialog (some builds call this from the menu)
    # ------------------------------------------------------------------
    def open_compare_dialog(self):
        """Abrir diálogo para carregar modelos adicionais (comparação)."""
        # Prefer an existing dedicated dialog if present
        fn = getattr(self, "open_compare_models_dialog", None)
        if callable(fn):
            return fn()

        # Fallback: file picker + load_compare_model if available
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Selecionar Modelos", "grids", "GRDECL (*.grdecl)"
        )
        if not paths:
            return

        study_name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Novo Estudo",
            "Nome do Estudo / Grupo de Calibração:",
            text="Importação Recente",
        )
        if not ok or not study_name.strip():
            study_name = "Importação Recente"

        loader = getattr(self, "load_compare_model", None)
        if not callable(loader):
            QtWidgets.QMessageBox.warning(
                self,
                "Função indisponível",
                "load_compare_model não está disponível nesta versão do window.py.",
            )
            return

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for path in paths:
                loader(path, study_name=study_name)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def __init__(self, mode="facies", z_exag=1.0, show_scalar_bar=True, reservoir_facies=None):
        super().__init__()
        self.setWindowTitle("Grid View Analysis")

        if reservoir_facies is None:
            reservoir_facies = {0}

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

        # --- Agrupamento de fácies (configuração) ---
        # Baseado no color_reference_facies.txt (lista global de fácies e cores),
        # e não no conteúdo de um modelo específico.
        try:
            self.facies_reference = load_facies_reference()
        except Exception:
            self.facies_reference = []
        self.facies_colors_dict = load_facies_colors()
        self.facies_grouping_map = {int(f): int(f) for f, _ in (self.facies_reference or [])}
        self.use_facies_grouping = False
        self._fg_src = np.array(sorted(self.facies_grouping_map.keys()), dtype=np.int32) if self.facies_grouping_map else np.array([], dtype=np.int32)
        self._fg_dst = np.array([self.facies_grouping_map[k] for k in self._fg_src], dtype=np.int32) if self.facies_grouping_map else np.array([], dtype=np.int32)

        # Mantém seleções de reservatório em raw e agrupado, para alternar sem perder intenção.
        self.state_reservoir_raw = set(initial_reservoir)
        self.state_reservoir_grouped = {int(self.facies_grouping_map.get(int(x), int(x))) for x in self.state_reservoir_raw}
        
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

        self.state = {"reservoir_facies": set(initial_reservoir), "mode": mode, "reservoir_facies_raw": set(initial_reservoir), "reservoir_facies_grouped": set(initial_reservoir)}
        self.compare_states = {"base": {}, "compare": {}}
        # Inicializa conjuntos raw/agrupado (por enquanto identidade)
        self.state["reservoir_facies_raw"] = set(self.state_reservoir_raw)
        self.state_reservoir_grouped = {int(self.facies_grouping_map.get(int(x), int(x))) for x in self.state_reservoir_raw}
        self.state["reservoir_facies_grouped"] = set(self.state_reservoir_grouped)
        self.state["reservoir_facies"] = set(self.state_reservoir_raw)
        self.state["current_facies_raw"] = np.asarray(facies).ravel().astype(np.int32)

        self.state["lock_axes_bounds"] = True

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

        # --- 3D Selection (Cell/Column) ---
        # visualize.run instala um picker VTK que dispara state["on_cell_picked"]
        # quando pick_mode está ativo.
        self.state["on_cell_picked"] = self._on_3d_pick
        # callback simples para feedback na status bar (opcional)
        try:
            self.state['status_callback'] = lambda msg, ms=4000: self.statusBar().showMessage(str(msg), int(ms))
        except Exception:
            pass
        self.state.setdefault("pick_mode", None)

        # instala filtro Qt para capturar clique (robusto em QtInteractor embedado)
        try:
            self._install_3d_pick_filter()
        except Exception:
            pass
        
        self._map2d_hover_targets = {}
        self._last_2d_hover_msg = ""
        
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

        app = QtWidgets.QApplication.instance()
        if app is not None:
            try:
                app.aboutToQuit.connect(self._cleanup_vtk)
            except Exception:
                pass

        self.showMaximized()

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
        # A janela é maximizada ao final da montagem da UI.
        # Chamar showMaximized() aqui, antes dos docks/ribbon/plotters,
        # pode deixar a primeira abertura “cortada” em alguns ambientes Qt/Windows.

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
        self.view_menu.addAction(self.dock_map2d_summary.toggleViewAction())
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
        ml.setSpacing(6)
        self.plotter_2d, plotter_2d_widget = self._make_embedded_plotter(parent=self.map2d_tab)
        ml.addWidget(plotter_2d_widget, 1)

        self.viz_container.addWidget(self.map2d_tab)

        self.plotter_2d._hover2d_model_name = None
        self.plotter_2d._map2d_summary_target = getattr(self, "map2d_summary_text", None)
        self._install_2d_hover_filter(self.plotter_2d, model_name=None)
        self._sync_context_docks_visibility()

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
        # Pag 3: Ranking (SCORE POR PROPORÇÃO + VISÃO GERAL)
        self.ranking_tab = QtWidgets.QWidget()
        l_rank = QtWidgets.QVBoxLayout(self.ranking_tab)
        l_rank.setContentsMargins(8, 8, 8, 8)
        l_rank.setSpacing(6)

        # --- Barra superior: parâmetros do score ---
        h_ctrl = QtWidgets.QHBoxLayout()
        lbl_rank = QtWidgets.QLabel("Ranking Global (score por proporção de fácies)")
        lbl_rank.setStyleSheet("font-weight: 600;")
        h_ctrl.addWidget(lbl_rank)
        h_ctrl.addStretch(1)

        h_ctrl.addWidget(QtWidgets.QLabel("t_min (m):"))
        self.spin_rank_tmin = QtWidgets.QDoubleSpinBox()
        self.spin_rank_tmin.setRange(0.0, 100.0)
        self.spin_rank_tmin.setDecimals(2)
        self.spin_rank_tmin.setSingleStep(0.05)
        self.spin_rank_tmin.setValue(float(getattr(self, "well_rank_t_min", 0.30) or 0.30))
        self.spin_rank_tmin.setToolTip(
            "Espessura mínima para suavizar o log: segmentos < t_min são mesclados ao vizinho antes de calcular proporções."
        )
        self.spin_rank_tmin.valueChanged.connect(self._on_rank_params_changed)
        h_ctrl.addWidget(self.spin_rank_tmin)

        btn_recalc = QtWidgets.QPushButton("Recalcular")
        btn_recalc.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload))
        btn_recalc.clicked.connect(self.update_ranking_view_content)
        h_ctrl.addWidget(btn_recalc)

        l_rank.addLayout(h_ctrl)

        # --- Split: esquerda (tabelas) | direita (visão geral) ---
        self.rank_main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # ===== Esquerda: tabelas =====
        w_left = QtWidgets.QWidget()
        l_left = QtWidgets.QVBoxLayout(w_left)
        l_left.setContentsMargins(0, 0, 0, 0)

        self.ranking_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        # --- Tabela de Modelos ---
        w_top = QtWidgets.QWidget()
        l_top = QtWidgets.QVBoxLayout(w_top)
        l_top.setContentsMargins(0, 0, 0, 0)

        h_top_bar = QtWidgets.QHBoxLayout()
        h_top_bar.addWidget(QtWidgets.QLabel("Ranking de Modelos"))
        h_top_bar.addStretch(1)
        btn_copy_models = QtWidgets.QPushButton("Copiar")
        btn_copy_models.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        btn_copy_models.clicked.connect(lambda: self._copy_table_to_clipboard(self.tbl_models))
        h_top_bar.addWidget(btn_copy_models)
        l_top.addLayout(h_top_bar)

        self.tbl_models = QtWidgets.QTableWidget()
        # Colunas: Rank, Study, Modelo, Score, ΣT_real (m), Poços
        self.tbl_models.setColumnCount(6)
        self.tbl_models.setHorizontalHeaderLabels(["Rank", "Study", "Modelo", "Score", "ΣT_real (m)", "Poços"])
        self.tbl_models.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_models.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tbl_models.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_models.setSortingEnabled(True)
        self.tbl_models.itemSelectionChanged.connect(self._on_models_table_selection_changed)
        l_top.addWidget(self.tbl_models)

        # --- Tabela de Poços (modelo selecionado) ---
        w_bot = QtWidgets.QWidget()
        l_bot = QtWidgets.QVBoxLayout(w_bot)
        l_bot.setContentsMargins(0, 0, 0, 0)

        h_bot_bar = QtWidgets.QHBoxLayout()
        h_bot_bar.addWidget(QtWidgets.QLabel("Detalhamento por Poço (modelo selecionado)"))
        h_bot_bar.addStretch(1)
        btn_copy_wells = QtWidgets.QPushButton("Copiar")
        btn_copy_wells.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        btn_copy_wells.clicked.connect(lambda: self._copy_table_to_clipboard(self.tbl_wells))
        h_bot_bar.addWidget(btn_copy_wells)
        l_bot.addLayout(h_bot_bar)

        self.tbl_wells = QtWidgets.QTableWidget()
        # Colunas: Poço, Score, D_prop, T_real, T_sim, ΔT, Ações
        self.tbl_wells.setColumnCount(7)
        self.tbl_wells.setHorizontalHeaderLabels(["Poço", "Score", "D_prop", "T_real (m)", "T_sim (m)", "ΔT (m)", "Ações"])
        self.tbl_wells.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_wells.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tbl_wells.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_wells.setSortingEnabled(True)
        l_bot.addWidget(self.tbl_wells)

        self.ranking_splitter.addWidget(w_top)
        self.ranking_splitter.addWidget(w_bot)
        self.ranking_splitter.setStretchFactor(0, 1)
        self.ranking_splitter.setStretchFactor(1, 2)

        l_left.addWidget(self.ranking_splitter)
        self.rank_main_splitter.addWidget(w_left)

        # ===== Direita: visão geral =====
        w_right = QtWidgets.QWidget()
        l_right = QtWidgets.QVBoxLayout(w_right)
        l_right.setContentsMargins(0, 0, 0, 0)
        gb = QtWidgets.QGroupBox("Visão geral dos poços (REAL vs SIM)")
        gb_l = QtWidgets.QVBoxLayout(gb)

        info = QtWidgets.QLabel("Cada linha mostra dois logs simplificados (REAL e SIM) com a suavização t_min aplicada.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #555;")
        gb_l.addWidget(info)

        self.rank_overview_canvas = None
        self.rank_overview_ax = None
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            fig = Figure(figsize=(6.0, 4.0), tight_layout=True)
            self.rank_overview_canvas = FigureCanvas(fig)
            self.rank_overview_ax = fig.add_subplot(111)
            gb_l.addWidget(self.rank_overview_canvas, 1)
        except Exception:
            gb_l.addWidget(QtWidgets.QLabel("Matplotlib não disponível para a visão geral."))

        l_right.addWidget(gb, 1)
        self.rank_main_splitter.addWidget(w_right)
        self.rank_main_splitter.setStretchFactor(0, 3)
        self.rank_main_splitter.setStretchFactor(1, 2)

        l_rank.addWidget(self.rank_main_splitter, 1)

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

        # --- PERSPECTIVA 3: INCERTEZA (NOVA) ---
        self.uncertainty_page = QtWidgets.QWidget()
        self.setup_uncertainty_tab(self.uncertainty_page)
        self.central_stack.addWidget(self.uncertainty_page)
    
    def setup_uncertainty_tab(self, parent_widget):
        """
        Página geral de Cálculo de Mapas.

        A categoria principal é controlada pelo Ribbon. O painel lateral é contextual:
        - modelos no topo;
        - parâmetros específicos da tarefa no meio;
        - resultado/diagnóstico e botão de cálculo embaixo.
        """
        from load_data import nx, ny, nz

        # Limpa layout anterior
        if parent_widget.layout():
            old = parent_widget.layout()
            while old.count():
                it = old.takeAt(0)
                if it.widget():
                    it.widget().deleteLater()
            QtWidgets.QWidget().setLayout(old)

        layout = QtWidgets.QHBoxLayout(parent_widget)
        # Alinha a margem esquerda da página de Mapas com a margem dos grupos do Ribbon.
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        # ============================================================
        # PAINEL ESQUERDO
        # ============================================================
        left_panel = QtWidgets.QFrame()
        left_panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        left_panel.setFixedWidth(360)

        vl_left = QtWidgets.QVBoxLayout(left_panel)
        vl_left.setContentsMargins(8, 8, 8, 8)
        vl_left.setSpacing(8)
        vl_left.setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        vl_left.setAlignment(QtCore.Qt.AlignTop)

        self.lbl_mapcalc_title = QtWidgets.QLabel("<b>Cálculo de Mapas</b>")
        self.lbl_mapcalc_title.setStyleSheet("font-size: 12px;")
        vl_left.addWidget(self.lbl_mapcalc_title)

        self.lbl_mapcalc_help = QtWidgets.QLabel(
            "Escolha o tipo de mapa no Ribbon. Use a lista abaixo para escolher o(s) modelo(s)."
        )
        self.lbl_mapcalc_help.setWordWrap(True)
        self.lbl_mapcalc_help.setMaximumHeight(58)
        self.lbl_mapcalc_help.setStyleSheet("color: #555;")
        vl_left.addWidget(self.lbl_mapcalc_help)

        # ------------------------------------------------------------
        # Modelos — substitui o combo Usar Base/Usar Explorer
        # ------------------------------------------------------------
        gb_models = QtWidgets.QGroupBox("Modelos")
        l_models = QtWidgets.QVBoxLayout(gb_models)
        l_models.setContentsMargins(6, 6, 6, 6)
        l_models.setSpacing(4)

        self.lbl_mapcalc_models_hint = QtWidgets.QLabel(
            "Mapa individual: usa o modelo selecionado. Ensemble/Incerteza: usa os modelos marcados."
        )
        self.lbl_mapcalc_models_hint.setWordWrap(True)
        self.lbl_mapcalc_models_hint.setStyleSheet("color: #555;")
        self.lbl_mapcalc_models_hint.setMaximumHeight(44)
        l_models.addWidget(self.lbl_mapcalc_models_hint)

        self.lst_mapcalc_models = QtWidgets.QListWidget()
        self.lst_mapcalc_models.setMaximumHeight(150)
        self.lst_mapcalc_models.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.lst_mapcalc_models.itemChanged.connect(self._on_mapcalc_model_item_changed)
        self.lst_mapcalc_models.itemSelectionChanged.connect(self._on_mapcalc_model_selection_changed)
        l_models.addWidget(self.lst_mapcalc_models)

        h_model_btns = QtWidgets.QHBoxLayout()
        self.btn_mapcalc_all_models = QtWidgets.QPushButton("Todos")
        self.btn_mapcalc_all_models.clicked.connect(lambda: self._set_all_mapcalc_models_checked(True))
        self.btn_mapcalc_no_models = QtWidgets.QPushButton("Nenhum")
        self.btn_mapcalc_no_models.clicked.connect(lambda: self._set_all_mapcalc_models_checked(False))
        h_model_btns.addWidget(self.btn_mapcalc_all_models)
        h_model_btns.addWidget(self.btn_mapcalc_no_models)
        l_models.addLayout(h_model_btns)

        vl_left.addWidget(gb_models)
        self._compact_groupbox(gb_models)

        # ------------------------------------------------------------
        # Stack de configuração contextual
        # ------------------------------------------------------------
        self.mapcalc_config_stack = QtWidgets.QStackedWidget()
        self.mapcalc_config_stack.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Maximum
        )
        vl_left.addWidget(self.mapcalc_config_stack)

        self._build_mapcalc_vertical_page()
        self._build_mapcalc_property_page()
        self._build_mapcalc_ensemble_page()
        self._build_mapcalc_uncertainty_page()
        self._build_mapcalc_difference_page()

        # ------------------------------------------------------------
        # Diagnóstico / Resultado
        # ------------------------------------------------------------
        gb_diag = QtWidgets.QGroupBox("Resultado")
        l_diag = QtWidgets.QVBoxLayout(gb_diag)

        self.lbl_uncert_n = QtWidgets.QLabel("Entrada: -")
        self.lbl_uncert_max_theo = QtWidgets.QLabel("Info: -")
        self.lbl_uncert_max_real = QtWidgets.QLabel("Resultado: -")
        self.lbl_uncert_max_real.setStyleSheet("font-weight: bold; font-size: 13px;")

        l_diag.addWidget(self.lbl_uncert_n)
        l_diag.addWidget(self.lbl_uncert_max_theo)
        l_diag.addWidget(self.lbl_uncert_max_real)

        vl_left.addWidget(gb_diag)
        self._compact_groupbox(gb_diag)

        # Cálculo automático: o botão permanece como fallback interno, mas fica oculto.
        self.btn_run_mapcalc = QtWidgets.QPushButton("Calcular")
        self.btn_run_mapcalc.setFixedHeight(40)
        self.btn_run_mapcalc.setStyleSheet("background-color: #d0f0c0; font-weight: bold;")
        self.btn_run_mapcalc.clicked.connect(self.run_map_calculation)
        self.btn_run_mapcalc.hide()
        vl_left.addWidget(self.btn_run_mapcalc)

        # ============================================================
        # CENTRO: RESULTADOS
        # ============================================================
        self.uncert_result_stack = QtWidgets.QStackedWidget()

        # Página 0: 3D
        self.uncert_page_3d = QtWidgets.QWidget()
        l_3d = QtWidgets.QVBoxLayout(self.uncert_page_3d)
        l_3d.setContentsMargins(0, 0, 0, 0)
        self.uncert_plotter, uncert_widget = self._make_embedded_plotter(parent=self.uncert_page_3d)
        l_3d.addWidget(uncert_widget, 1)
        self.uncert_result_stack.addWidget(self.uncert_page_3d)

        # Página 1: 2D
        self.uncert_page_2d = QtWidgets.QWidget()
        l_2d = QtWidgets.QVBoxLayout(self.uncert_page_2d)
        l_2d.setContentsMargins(0, 0, 0, 0)
        self.uncert_plotter_2d, uncert_2d_widget = self._make_embedded_plotter(parent=self.uncert_page_2d)
        l_2d.addWidget(uncert_2d_widget, 1)
        self.uncert_result_stack.addWidget(self.uncert_page_2d)

        # Página 2: tabela
        self.uncert_page_table = QtWidgets.QWidget()
        l_table = QtWidgets.QVBoxLayout(self.uncert_page_table)
        l_table.setContentsMargins(8, 8, 8, 8)

        self.lbl_uncert_table_title = QtWidgets.QLabel("<b>Resumo</b>")
        l_table.addWidget(self.lbl_uncert_table_title)

        self.txt_uncert_summary = QtWidgets.QTextEdit()
        self.txt_uncert_summary.setReadOnly(True)
        self.txt_uncert_summary.setMaximumHeight(110)
        self.txt_uncert_summary.setPlainText("Configure o cálculo e clique em Calcular.")
        l_table.addWidget(self.txt_uncert_summary)

        self.tbl_uncert_summary = QtWidgets.QTableWidget()
        self.tbl_uncert_summary.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_uncert_summary.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_uncert_summary.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tbl_uncert_summary.setSortingEnabled(True)
        self.tbl_uncert_summary.setAlternatingRowColors(True)
        l_table.addWidget(self.tbl_uncert_summary, 1)

        self.uncert_result_stack.addWidget(self.uncert_page_table)

        # ============================================================
        # DIREITA: FÁCIES-ALVO / GEOMETRIA
        # ============================================================
        self.uncert_right_panel = QtWidgets.QFrame()
        self.uncert_right_panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        # Painel direito mais estreito, mas com altura expansível para a tabela de fácies.
        self.uncert_right_panel.setMinimumWidth(300)

        vl_right = QtWidgets.QVBoxLayout(self.uncert_right_panel)
        vl_right.setContentsMargins(8, 8, 8, 8)

        self.mapcalc_right_tabs = QtWidgets.QTabWidget()
        self.mapcalc_right_tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )

        # Aba dedicada de fácies-alvo para Cálculo de Mapas
        self.mapcalc_facies_page = QtWidgets.QWidget()
        l_facies = QtWidgets.QVBoxLayout(self.mapcalc_facies_page)
        l_facies.setContentsMargins(6, 6, 6, 6)
        l_facies.setSpacing(6)

        info_facies = QtWidgets.QLabel(
            "Selecione aqui as fácies-alvo usadas nos mapas verticais. "
            "A tabela reúne as fácies presentes nos modelos carregados."
        )
        info_facies.setWordWrap(True)
        info_facies.setStyleSheet("color: #555;")
        l_facies.addWidget(info_facies)

        h_facies_btns = QtWidgets.QHBoxLayout()
        self.btn_mapcalc_facies_all = QtWidgets.QPushButton("Todas")
        self.btn_mapcalc_facies_none = QtWidgets.QPushButton("Nenhuma")
        self.btn_mapcalc_facies_all.clicked.connect(lambda: self._set_all_mapcalc_target_facies(True))
        self.btn_mapcalc_facies_none.clicked.connect(lambda: self._set_all_mapcalc_target_facies(False))
        h_facies_btns.addWidget(self.btn_mapcalc_facies_all)
        h_facies_btns.addWidget(self.btn_mapcalc_facies_none)
        l_facies.addLayout(h_facies_btns)

        self.tbl_mapcalc_target_facies = QtWidgets.QTableWidget()
        self.tbl_mapcalc_target_facies.setColumnCount(3)
        self.tbl_mapcalc_target_facies.setHorizontalHeaderLabels(["Cor", "ID", "Alvo"])
        self.tbl_mapcalc_target_facies.verticalHeader().setVisible(False)
        self.tbl_mapcalc_target_facies.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_mapcalc_target_facies.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_mapcalc_target_facies.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tbl_mapcalc_target_facies.itemChanged.connect(self._on_mapcalc_target_facies_item_changed)
        self.tbl_mapcalc_target_facies.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.tbl_mapcalc_target_facies.horizontalHeader().setStretchLastSection(True)
        self.tbl_mapcalc_target_facies.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        l_facies.addWidget(self.tbl_mapcalc_target_facies, 1)

        self.mapcalc_right_tabs.addTab(self.mapcalc_facies_page, "Fácies-alvo")

        # Aba de geometria/cortes para resultados 3D
        self.mapcalc_geometry_page = QtWidgets.QWidget()
        l_geo_page = QtWidgets.QVBoxLayout(self.mapcalc_geometry_page)
        l_geo_page.setContentsMargins(6, 6, 6, 6)
        self.uncert_slicer = GridSlicerWidget(nx, ny, nz, self.on_uncert_slice_changed)
        l_geo_page.addWidget(self.uncert_slicer)
        l_geo_page.addStretch(1)
        self.mapcalc_right_tabs.addTab(self.mapcalc_geometry_page, "Geometria")

        vl_right.addWidget(self.mapcalc_right_tabs, 1)

        # Scroll geral do painel esquerdo:
        # - conteúdo compacto no topo;
        # - sem esticar verticalmente os grupos;
        # - largura interna próxima da largura externa, sem faixa cinza grande.
        self.mapcalc_left_scroll = QtWidgets.QScrollArea()
        self.mapcalc_left_scroll.setWidgetResizable(False)
        self.mapcalc_left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.mapcalc_left_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.mapcalc_left_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.mapcalc_left_scroll.setFixedWidth(378)
        self.mapcalc_left_scroll.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        left_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Maximum
        )
        left_panel.setFixedWidth(360)

        self.mapcalc_left_scroll.setWidget(left_panel)

        # Painel direito:
        # - volta a uma largura menor;
        # - altura acompanha a página;
        # - sem mínimo/máximo de altura, para a tabela de fácies ocupar o espaço disponível.
        self.mapcalc_right_scroll = QtWidgets.QScrollArea()
        self.mapcalc_right_scroll.setWidgetResizable(True)
        self.mapcalc_right_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.mapcalc_right_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.mapcalc_right_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.mapcalc_right_scroll.setFixedWidth(320)
        self.mapcalc_right_scroll.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        self.uncert_right_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        self.uncert_right_panel.setMinimumWidth(300)

        self.mapcalc_right_scroll.setWidget(self.uncert_right_panel)

        layout.addWidget(self.mapcalc_left_scroll)
        layout.addWidget(self.uncert_result_stack, 1)
        layout.addWidget(self.mapcalc_right_scroll)

        # Estado
        self.uncert_view_state = None
        self._uncert_has_result = False
        self.mapcalc_mode = "vertical"
        self.mapcalc_model_key = "base"
        self.mapcalc_selected_property = None
        self.mapcalc_selected_operation = "weighted_mean"
        self.mapcalc_selected_metric = {
            "scalar": "__total_column_thickness__",
            "title": "Espessura total da coluna (m)",
            "label": "Espessura total do modelo",
            "formula": "Ttot(i,j) = Σ_k h_ijk",
            "description": "Soma da espessura estratigráfica de todas as células da coluna.",
        }
        self.mapcalc_selected_stat = "std"
        self.mapcalc_selected_scope = "column"
        self._updating_mapcalc_models = False
        self._updating_mapcalc_facies = False
        self._mapcalc_ready = False
        self._ensure_mapcalc_auto_timer()

        self._refresh_mapcalc_models_panel()
        self._refresh_mapcalc_property_list()
        self._refresh_mapcalc_target_facies_table()
        self._update_mapcalc_target_facies_label()
        self.set_mapcalc_mode("vertical")
        self._mapcalc_ready = True
        # Não calcula automaticamente durante a construção da página.
        # O primeiro cálculo automático acontece quando o usuário muda modelo, métrica ou fácies,
        # ou quando a aba Mapas é aberta explicitamente.

    def _make_mapcalc_card_button(self, text, icon=None, checkable=True):
        b = QtWidgets.QToolButton()
        b.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        if icon is not None:
            b.setIcon(icon)
        b.setIconSize(QtCore.QSize(26, 26))
        b.setText(text)
        b.setCheckable(checkable)
        b.setAutoRaise(True)
        b.setMinimumWidth(74)
        b.setMinimumHeight(58)
        return b


    def _make_mapcalc_info_box(self, min_height=72, max_height=105):
        """Caixa simples de explicação.

        As fórmulas foram removidas da UI porque o Qt não renderiza LaTeX/equation
        com a mesma qualidade do PDF. A caixa mantém rolagem própria para textos
        maiores, sem criar scroll geral na lateral.
        """
        box = QtWidgets.QTextEdit()
        box.setReadOnly(True)
        box.setAcceptRichText(False)
        box.setMinimumHeight(int(min_height))
        box.setMaximumHeight(int(max_height))
        box.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        box.setStyleSheet("QTextEdit { background: #fafafa; color: #333; border: 1px solid #bbb; }")
        return box
    
    def _compact_groupbox(self, gb):
        """
        Mantém o QGroupBox compacto no topo da lateral.

        Não usamos altura fixa aqui, porque alguns grupos mudam dinamicamente
        quando o usuário troca a categoria ou o grupo de descritores.
        """
        gb.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Maximum
        )
        gb.setMaximumHeight(16777215)
        return gb

    def _prepare_mapcalc_page_widget(self, page):
        """
        Faz a página contextual ficar compacta no topo do QStackedWidget.
        """
        if page is None:
            return
        page.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Maximum
        )
        lay = page.layout()
        if lay is not None:
            lay.setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
            lay.setAlignment(QtCore.Qt.AlignTop)

    def _sync_mapcalc_stack_height(self):
        """
        Ajusta a altura do stack lateral para a página atual.

        Isso evita o efeito em que o QStackedWidget usa a altura da maior
        página e estica os grupos das páginas menores. Se o conteúdo total
        passar da altura da tela, o QScrollArea externo mostra a rolagem geral.
        """
        try:
            if not hasattr(self, "mapcalc_config_stack"):
                return
            page = self.mapcalc_config_stack.currentWidget()
            if page is None:
                return

            page.adjustSize()
            h = max(1, int(page.sizeHint().height()))
            self.mapcalc_config_stack.setMinimumHeight(h)
            self.mapcalc_config_stack.setMaximumHeight(h)
            self.mapcalc_config_stack.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred,
                QtWidgets.QSizePolicy.Fixed
            )

            # Mantém larguras fixas e só ajusta a altura natural do conteúdo esquerdo.
            # Não usamos adjustSize() no painel direito, pois ele deve acompanhar a altura da página.
            if hasattr(self, "mapcalc_left_scroll"):
                w = self.mapcalc_left_scroll.widget()
                if w is not None:
                    w.setFixedWidth(360)
                    w.setFixedHeight(max(1, int(w.sizeHint().height())))
                    self.mapcalc_left_scroll.setWidgetResizable(False)

            if hasattr(self, "mapcalc_right_scroll"):
                self.mapcalc_right_scroll.setWidgetResizable(True)
        except Exception:
            pass

    def _mapcalc_vertical_metric_groups(self):
        """
        Metadados dos descritores verticais por coluna, organizados pelos títulos do documento.

        Observação:
        - scalar é o nome do campo calculado em cell_data.
        - A espessura total da coluna usa o campo especial __total_column_thickness__.
        - As demais métricas usam o prefixo vert_ criado por compute_vertical_metrics_for_grid().
        """
        return {
            "espessura": {
                "title": "Espessura",
                "description": "Mapas de espessura por coluna, com ou sem condicionamento pela fácies-alvo.",
                "items": [
                    {
                        "label": "Espessura total do modelo",
                        "short": "T total",
                        "scalar": "__total_column_thickness__",
                        "title": "Espessura total da coluna (m)",
                        "formula": "Ttot(i,j) = Σ_k h_ijk",
                        "description": "Soma da espessura estratigráfica de todas as células da coluna. Não depende da fácies-alvo.",
                    },
                    {
                        "label": "Espessura da fácies-alvo",
                        "short": "T alvo",
                        "scalar": "vert_Ttot_reservoir",
                        "title": "Espessura total da fácies-alvo (m)",
                        "formula": "Ttarget(i,j) = Σ_k h_ijk · I(f_ijk ∈ S)",
                        "description": "Soma da espessura das células pertencentes ao conjunto de fácies-alvo S.",
                    },
                    {
                        "label": "Espessura do envelope",
                        "short": "Envelope",
                        "scalar": "vert_Tenv_reservoir",
                        "title": "Espessura do envelope vertical da fácies-alvo (m)",
                        "formula": "Tenv(i,j) = Σ_k h_ijk entre a primeira e a última ocorrência da fácies-alvo",
                        "description": "Espessura do intervalo vertical entre a primeira e a última ocorrência da fácies-alvo na coluna.",
                    },
                ],
            },
            "proporcao": {
                "title": "Proporção",
                "description": "Mapas de proporção vertical da fácies-alvo na coluna e no envelope.",
                "items": [
                    {
                        "label": "Proporção na coluna",
                        "short": "P coluna",
                        "scalar": "vert_NTG_col_reservoir",
                        "title": "Proporção de fácies-alvo na coluna",
                        "formula": "Pcol(i,j) = Ttarget(i,j) / Ttot(i,j)",
                        "description": "Fração da espessura total da coluna ocupada pela fácies-alvo.",
                    },
                    {
                        "label": "Proporção no envelope",
                        "short": "P envelope",
                        "scalar": "vert_NTG_env_reservoir",
                        "title": "Proporção de fácies-alvo no envelope",
                        "formula": "Penv(i,j) = Ttarget(i,j) / Tenv(i,j)",
                        "description": "Fração do envelope vertical ocupada pela fácies-alvo.",
                    },
                ],
            },
            "sequencias": {
                "title": "Sequências Contínuas",
                "description": "Descritores dos pacotes verticais contínuos da fácies-alvo.",
                "items": [
                    {
                        "label": "Número de sequências contínuas",
                        "short": "Nº pacotes",
                        "scalar": "vert_n_packages_reservoir",
                        "title": "Número de sequências contínuas da fácies-alvo",
                        "formula": "Npack = número de sequências contínuas da fácies-alvo",
                        "description": "Conta quantos pacotes contínuos da fácies-alvo aparecem na coluna. Valores maiores indicam maior fragmentação vertical.",
                    },
                    {
                        "label": "Maior sequência contínua",
                        "short": "Maior pacote",
                        "scalar": "vert_Tpack_max_reservoir",
                        "title": "Espessura da maior sequência contínua (m)",
                        "formula": "Tpack,max = max{Tpack}",
                        "description": "Espessura do pacote contínuo mais espesso da fácies-alvo.",
                    },
                    {
                        "label": "Espessura média das sequências",
                        "short": "Tpack média",
                        "scalar": "vert_Tpack_mean_reservoir",
                        "title": "Espessura média das sequências contínuas (m)",
                        "formula": "Tpack,mean = Ttarget / Npack",
                        "description": "Espessura média dos pacotes contínuos da fácies-alvo.",
                    },
                    {
                        "label": "Dominância do maior pacote",
                        "short": "Dominância",
                        "scalar": "vert_Cdom_reservoir",
                        "title": "Dominância da maior sequência contínua",
                        "formula": "Cdom = Tpack,max / Ttarget",
                        "description": "Mede quanto da espessura total da fácies-alvo está concentrada no maior pacote.",
                    },
                ],
            },
            "conectividade": {
                "title": "Conectividade",
                "description": "Índices simplificados de conectividade vertical e concentração da fácies-alvo.",
                "items": [
                    {
                        "label": "ICV no envelope",
                        "short": "ICV",
                        "scalar": "vert_ICV_reservoir",
                        "title": "Índice de conectividade vertical no envelope",
                        "formula": "ICV = Tpack,max / Tenv",
                        "description": "Mede quanto do envelope vertical é ocupado pelo maior pacote contínuo.",
                    },
                    {
                        "label": "ICV na coluna",
                        "short": "ICV coluna",
                        "scalar": "vert_ICV_col_reservoir",
                        "title": "Índice de conectividade vertical na coluna",
                        "formula": "ICVcol = Tpack,max / Ttot",
                        "description": "Mede quanto da coluna inteira é ocupado pelo maior pacote contínuo.",
                    },
                    {
                        "label": "Qualidade vertical",
                        "short": "Qv",
                        "scalar": "vert_Qv_reservoir",
                        "title": "Qualidade vertical (Qv)",
                        "formula": "Qv = Pcol · ICV",
                        "description": "Combina presença da fácies-alvo na coluna com concentração da continuidade principal no envelope.",
                    },
                ],
            },
            "gaps": {
                "title": "Gaps",
                "description": "Interrupções internas entre pacotes contínuos dentro do envelope da fácies-alvo.",
                "items": [
                    {
                        "label": "Soma dos gaps",
                        "short": "Soma gaps",
                        "scalar": "vert_Tgap_sum_reservoir",
                        "title": "Soma dos gaps verticais (m)",
                        "formula": "Tgap,sum = Σ_k Tgap,k",
                        "description": "Soma das espessuras dos intervalos sem fácies-alvo entre a primeira e a última ocorrência.",
                    },
                    {
                        "label": "Maior gap",
                        "short": "Maior gap",
                        "scalar": "vert_Tgap_max_reservoir",
                        "title": "Maior gap vertical (m)",
                        "formula": "Tgap,max = max{Tgap}",
                        "description": "Representa a maior descontinuidade interna dentro do envelope da fácies-alvo.",
                    },
                    {
                        "label": "Fração de gaps no envelope",
                        "short": "G envelope",
                        "scalar": "vert_Gap_env_reservoir",
                        "title": "Fração de gaps no envelope",
                        "formula": "Genv = Tgap,sum / Tenv",
                        "description": "Fração do envelope ocupada por interrupções sem a fácies-alvo.",
                    },
                ],
            },
            "trocas": {
                "title": "Trocas",
                "description": "Alternâncias verticais entre fácies-alvo e não-alvo ao longo da coluna.",
                "items": [
                    {
                        "label": "Número de trocas",
                        "short": "Nº trocas",
                        "scalar": "vert_Nswitch_reservoir",
                        "title": "Número de trocas verticais",
                        "formula": "Nswitch = Σ_k δswitch,k",
                        "description": "Conta transições entre alvo e não-alvo na sequência vertical da coluna.",
                    },
                    {
                        "label": "Densidade de trocas na coluna",
                        "short": "D troca col.",
                        "scalar": "vert_Dswitch_col_reservoir",
                        "title": "Densidade de trocas na coluna",
                        "formula": "Dswitch,col = Nswitch / Ttot",
                        "description": "Número de trocas normalizado pela espessura total da coluna.",
                    },
                    {
                        "label": "Densidade de trocas no envelope",
                        "short": "D troca env.",
                        "scalar": "vert_Dswitch_env_reservoir",
                        "title": "Densidade de trocas no envelope",
                        "formula": "Dswitch,env = Nswitch / Tenv",
                        "description": "Número de trocas normalizado pela espessura do envelope da fácies-alvo.",
                    },
                    {
                        "label": "Troca ponderada por espessura",
                        "short": "W troca",
                        "scalar": "vert_Wswitch_reservoir",
                        "title": "Troca ponderada por espessura",
                        "formula": "Wswitch = Σ_k δswitch,k · (h_k + h_{k+1}) / 2",
                        "description": "Cada troca recebe o peso da espessura média das duas células adjacentes.",
                    },
                    {
                        "label": "Troca ponderada normalizada",
                        "short": "W troca col.",
                        "scalar": "vert_WswitchN_col_reservoir",
                        "title": "Troca ponderada normalizada na coluna",
                        "formula": "Wswitch,col = Wswitch / Ttot",
                        "description": "Troca ponderada normalizada pela espessura total da coluna.",
                    },
                ],
            },
            "permanencias": {
                "title": "Permanências",
                "description": "Continuidade vertical entre células adjacentes pertencentes à fácies-alvo.",
                "items": [
                    {
                        "label": "Número de permanências",
                        "short": "Nº perman.",
                        "scalar": "vert_Npersist_reservoir",
                        "title": "Número de permanências verticais",
                        "formula": "Npersist = Σ_k δpersist,k",
                        "description": "Conta adjacências verticais em que a fácies-alvo permanece entre células consecutivas.",
                    },
                    {
                        "label": "Permanência relativa",
                        "short": "R perman.",
                        "scalar": "vert_Rpersist_reservoir",
                        "title": "Permanência relativa",
                        "formula": "Rpersist = Npersist / (ntarget - 1), se ntarget > 1; 0 caso contrário",
                        "description": "Proporção de permanências em relação ao máximo possível entre células-alvo.",
                    },
                    {
                        "label": "Permanência ponderada",
                        "short": "W perman.",
                        "scalar": "vert_Wpersist_reservoir",
                        "title": "Permanência ponderada por espessura",
                        "formula": "Wpersist = Σ_k δpersist,k · (h_k + h_{k+1}) / 2",
                        "description": "Cada permanência recebe o peso da espessura média das duas células adjacentes.",
                    },
                    {
                        "label": "Permanência ponderada normalizada",
                        "short": "W perman. norm.",
                        "scalar": "vert_WpersistN_reservoir",
                        "title": "Permanência ponderada normalizada",
                        "formula": "Wpersist,col = Wpersist / Ttot",
                        "description": "Permanência ponderada normalizada pela espessura total da coluna.",
                    },
                ],
            },
        }

    def _get_mapcalc_metric_meta_by_scalar(self, scalar_name):
        scalar_name = str(scalar_name or "")
        for group in self._mapcalc_vertical_metric_groups().values():
            for meta in group.get("items", []):
                if str(meta.get("scalar")) == scalar_name:
                    return dict(meta)
        return None

    def _clear_layout_widgets(self, layout):
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
            elif item.layout() is not None:
                self._clear_layout_widgets(item.layout())

    def _mapcalc_escape_html(self, value):
        """Mantido por compatibilidade; as caixas de explicação agora usam texto puro."""
        import html
        return html.escape(str(value if value is not None else "-"))

    def _format_metric_info_text(self, meta, group_title=None):
        """Texto simples para as caixas de explicação dos descritores."""
        if not isinstance(meta, dict):
            return "Selecione uma métrica."

        lines = []
        if group_title:
            lines.append(f"Grupo: {group_title}")
        lines.append(f"Métrica: {meta.get('label', '-')}")
        lines.append("")
        lines.append(f"Descrição: {meta.get('description', '-')}")
        scalar = meta.get("scalar", None)
        if scalar:
            lines.append("")
            lines.append(f"Campo calculado: {scalar}")
        return "\n".join(lines)

    def _update_mapcalc_vertical_info_box(self, meta=None, group_title=None):
        if meta is None:
            meta = getattr(self, "mapcalc_selected_metric", None)
        if not isinstance(meta, dict):
            return
        if group_title is None:
            for g in self._mapcalc_vertical_metric_groups().values():
                if any(it.get("scalar") == meta.get("scalar") for it in g.get("items", [])):
                    group_title = g.get("title")
                    break

        text = self._format_metric_info_text(meta, group_title)
        for attr in ("txt_mapcalc_vertical_info", "txt_mapcalc_ensemble_metric_info"):
            w = getattr(self, attr, None)
            if w is not None:
                w.setPlainText(text)

    def _set_mapcalc_metric_group(self, group_key, panel="vertical"):
        groups = self._mapcalc_vertical_metric_groups()
        if group_key not in groups:
            group_key = next(iter(groups.keys()))
        if panel == "ensemble":
            self.mapcalc_ensemble_group_key = group_key
            target_layout = getattr(self, "mapcalc_ensemble_metric_items_layout", None)
            button_group = getattr(self, "mapcalc_ensemble_metric_buttons", None)
        else:
            self.mapcalc_vertical_group_key = group_key
            target_layout = getattr(self, "mapcalc_vertical_metric_items_layout", None)
            button_group = getattr(self, "mapcalc_metric_buttons", None)
        self._populate_mapcalc_metric_items(group_key, target_layout, button_group)

    def _populate_mapcalc_metric_items(self, group_key, layout, button_group):
        groups = self._mapcalc_vertical_metric_groups()
        group = groups.get(group_key) or next(iter(groups.values()))
        items = list(group.get("items", []))
        if layout is None:
            return
        self._clear_layout_widgets(layout)
        if button_group is None:
            button_group = QtWidgets.QButtonGroup(self)
            button_group.setExclusive(True)
        else:
            try:
                for b in list(button_group.buttons()):
                    button_group.removeButton(b)
            except Exception:
                pass
        for idx, meta in enumerate(items):
            b = QtWidgets.QPushButton(meta.get("short") or meta.get("label") or "Métrica")
            b.setCheckable(True)
            b.setMinimumHeight(28)
            b.setToolTip(meta.get("description", meta.get("label", "")))
            b.clicked.connect(lambda checked=False, m=dict(meta), gt=group.get("title"): self.set_mapcalc_vertical_metric_from_meta(m, gt))
            button_group.addButton(b)
            layout.addWidget(b, idx // 2, idx % 2)
            current = getattr(self, "mapcalc_selected_metric", {}) or {}
            if current.get("scalar") == meta.get("scalar"):
                b.setChecked(True)
        # Se a métrica atual não estiver no grupo, seleciona a primeira do grupo.
        current_scalar = (getattr(self, "mapcalc_selected_metric", {}) or {}).get("scalar")
        if items and not any(m.get("scalar") == current_scalar for m in items):
            first_button = button_group.buttons()[0] if button_group.buttons() else None
            if first_button is not None:
                first_button.setChecked(True)
            self.set_mapcalc_vertical_metric_from_meta(dict(items[0]), group.get("title"), schedule=False)
        else:
            self._update_mapcalc_vertical_info_box(group_title=group.get("title"))

        try:
            QtCore.QTimer.singleShot(0, self._sync_mapcalc_stack_height)
        except Exception:
            pass

    def set_mapcalc_vertical_metric_from_meta(self, meta, group_title=None, schedule=True):
        if not isinstance(meta, dict):
            return
        self.mapcalc_selected_metric = {
            "scalar": meta.get("scalar"),
            "title": meta.get("title", meta.get("label", "Métrica vertical")),
            "label": meta.get("label", meta.get("title", "Métrica vertical")),
            "formula": meta.get("formula", ""),
            "description": meta.get("description", ""),
        }
        self._update_mapcalc_vertical_info_box(meta, group_title)
        if schedule:
            self._schedule_mapcalc_auto_update()

    def _mapcalc_property_operation_meta(self, operation=None):
        operation = str(operation or getattr(self, "mapcalc_selected_operation", "weighted_mean"))
        metas = {
            "weighted_mean": {
                "title": "Média ponderada por espessura",
                "formula": "p̄_h(i,j) = Σ_k h_ijk · p_ijk / Σ_k h_ijk",
                "description": "Calcula o valor médio da propriedade na coluna, ponderando cada célula pela espessura.",
            },
            "equivalent": {
                "title": "Espessura equivalente",
                "formula": "Tp(i,j) = Σ_k h_ijk · p_ijk",
                "description": "Transforma uma propriedade fracionária entre 0 e 1 em espessura equivalente por coluna.",
            },
            "mean": {
                "title": "Média aritmética vertical",
                "formula": "p̄(i,j) = mean_k(p_ijk)",
                "description": "Média simples da propriedade ao longo das células da coluna.",
            },
            "sum": {
                "title": "Soma vertical",
                "formula": "S_p(i,j) = Σ_k p_ijk",
                "description": "Soma dos valores da propriedade ao longo da coluna.",
            },
            "max": {
                "title": "Máximo vertical",
                "formula": "p_max(i,j) = max_k(p_ijk)",
                "description": "Maior valor da propriedade encontrado na coluna.",
            },
        }
        return metas.get(operation, metas["weighted_mean"])

    def _update_mapcalc_property_info_box(self):
        box = getattr(self, "txt_mapcalc_property_info", None)
        if box is None:
            return
        meta = self._mapcalc_property_operation_meta()
        prop = getattr(self, "mapcalc_selected_property", None) or "-"
        text = (
            f"Operação: {meta['title']}\n\n"
            f"Descrição: {meta['description']}\n\n"
            f"Propriedade selecionada: {prop}"
        )
        box.setPlainText(text)

    def _mapcalc_stat_meta(self, stat=None):
        stat = str(stat or getattr(self, "mapcalc_selected_stat", "mean"))
        metas = {
            "mean": {
                "title": "Média do ensemble",
                "formula": "μ_D(i,j) = (1/M) · Σ_m D^(m)(i,j)",
                "description": "Mapa médio do ensemble. Representa tendência central, não incerteza por si só.",
            },
            "std": {
                "title": "Desvio padrão do ensemble",
                "formula": "σ_D(i,j) = sqrt((1/M) · Σ_m (D^(m)(i,j) - μ_D(i,j))²)",
                "description": "Mede a dispersão dos modelos em torno da média, na mesma unidade da métrica.",
            },
            "var": {
                "title": "Variância do ensemble",
                "formula": "σ²_D(i,j) = (1/M) · Σ_m (D^(m)(i,j) - μ_D(i,j))²",
                "description": "Mede a dispersão quadrática entre modelos.",
            },
            "range": {
                "title": "Amplitude do ensemble",
                "formula": "R_D(i,j) = max_m D^(m)(i,j) - min_m D^(m)(i,j)",
                "description": "Diferença entre o maior e o menor valor entre os modelos selecionados.",
            },
        }
        return metas.get(stat, metas["mean"])

    def _update_mapcalc_ensemble_info_box(self):
        box = getattr(self, "txt_mapcalc_ensemble_info", None)
        if box is None:
            return
        stat = self._mapcalc_stat_meta()
        metric = getattr(self, "mapcalc_selected_metric", None) or {}
        text = (
            f"Estatística: {stat['title']}\n\n"
            f"Descrição: {stat['description']}\n\n"
            f"Métrica vertical: {metric.get('label', '-')}"
        )
        box.setPlainText(text)

    def _update_mapcalc_uncertainty_info_box(self):
        box = getattr(self, "txt_mapcalc_uncertainty_info", None)
        if box is None:
            return
        text = (
            "Método: Entropia de fácies\n\n"
            "Descrição: mede a discordância categórica entre modelos em cada célula. "
            "Valores baixos indicam concordância entre cenários; valores altos indicam maior diversidade de fácies na mesma célula.\n\n"
            "Interpretação: quanto maior a entropia, maior a variação categórica de fácies entre os modelos selecionados."
        )
        box.setPlainText(text)




    def _build_mapcalc_vertical_page(self):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(8)

        # Grupo principal: Espessura, Proporção, Sequências, etc.
        gb_group = QtWidgets.QGroupBox("Grupo de descritores")
        grid_group = QtWidgets.QGridLayout(gb_group)
        grid_group.setSpacing(6)

        self.mapcalc_metric_group_buttons = QtWidgets.QButtonGroup(self)
        self.mapcalc_metric_group_buttons.setExclusive(True)

        groups = self._mapcalc_vertical_metric_groups()
        for idx, (key, group) in enumerate(groups.items()):
            b = QtWidgets.QPushButton(group.get("title", key))
            b.setCheckable(True)
            b.setMinimumHeight(30)
            b.setToolTip(group.get("description", ""))
            b.clicked.connect(lambda checked=False, k=key: self._set_mapcalc_metric_group(k, panel="vertical"))
            self.mapcalc_metric_group_buttons.addButton(b)
            grid_group.addWidget(b, idx // 2, idx % 2)
            if idx == 0:
                b.setChecked(True)
                self.mapcalc_vertical_group_key = key

        lay.addWidget(gb_group)
        self._compact_groupbox(gb_group)

        # Itens do grupo selecionado
        gb_metric = QtWidgets.QGroupBox("Métrica vertical")
        self.mapcalc_vertical_metric_items_layout = QtWidgets.QGridLayout(gb_metric)
        self.mapcalc_vertical_metric_items_layout.setSpacing(6)
        self.mapcalc_metric_buttons = QtWidgets.QButtonGroup(self)
        self.mapcalc_metric_buttons.setExclusive(True)
        lay.addWidget(gb_metric)
        self._compact_groupbox(gb_metric)

        # Explicação
        gb_info = QtWidgets.QGroupBox("Explicação")
        l_info = QtWidgets.QVBoxLayout(gb_info)
        self.txt_mapcalc_vertical_info = self._make_mapcalc_info_box(min_height=90, max_height=120)
        l_info.addWidget(self.txt_mapcalc_vertical_info)
        lay.addWidget(gb_info)
        self._compact_groupbox(gb_info)

        gb_target = QtWidgets.QGroupBox("Fácies-alvo")
        l_target = QtWidgets.QVBoxLayout(gb_target)
        self.lbl_mapcalc_target_facies = QtWidgets.QLabel("Fácies selecionadas: -")
        self.lbl_mapcalc_target_facies.setWordWrap(True)
        l_target.addWidget(self.lbl_mapcalc_target_facies)
        hint = QtWidgets.QLabel("Use a tabela à direita para marcar/desmarcar as fácies-alvo.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555;")
        l_target.addWidget(hint)
        lay.addWidget(gb_target)
        self._compact_groupbox(gb_target)

        self._prepare_mapcalc_page_widget(page)
        self.mapcalc_config_stack.addWidget(page)

        # Popula o primeiro grupo depois que os widgets existem.
        first_key = next(iter(groups.keys())) if groups else "espessura"
        self._set_mapcalc_metric_group(first_key, panel="vertical")


    def _build_mapcalc_property_page(self):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(8)

        gb_op = QtWidgets.QGroupBox("Operação")
        grid_op = QtWidgets.QGridLayout(gb_op)
        grid_op.setSpacing(6)

        self.mapcalc_operation_buttons = QtWidgets.QButtonGroup(self)
        self.mapcalc_operation_buttons.setExclusive(True)

        ops = [
            ("Média ponderada", "weighted_mean"),
            ("Esp. equivalente", "equivalent"),
            ("Média vertical", "mean"),
            ("Soma vertical", "sum"),
            ("Máximo vertical", "max"),
        ]

        for idx, (label, key) in enumerate(ops):
            meta = self._mapcalc_property_operation_meta(key)
            b = QtWidgets.QPushButton(label)
            b.setCheckable(True)
            b.setMinimumHeight(28)
            b.setToolTip(meta["title"])
            b.clicked.connect(lambda checked=False, k=key: self.set_mapcalc_property_operation(k))
            self.mapcalc_operation_buttons.addButton(b)
            grid_op.addWidget(b, idx // 2, idx % 2)

            if idx == 0:
                b.setChecked(True)

        lay.addWidget(gb_op)

        gb_info = QtWidgets.QGroupBox("Explicação")
        l_info = QtWidgets.QVBoxLayout(gb_info)
        self.txt_mapcalc_property_info = self._make_mapcalc_info_box(min_height=90, max_height=120)
        l_info.addWidget(self.txt_mapcalc_property_info)
        lay.addWidget(gb_info)

        gb_prop = QtWidgets.QGroupBox("Propriedade")
        l_prop = QtWidgets.QVBoxLayout(gb_prop)

        self.txt_mapcalc_prop_filter = QtWidgets.QLineEdit()
        self.txt_mapcalc_prop_filter.setPlaceholderText("Pesquisar propriedade...")
        self.txt_mapcalc_prop_filter.textChanged.connect(self._refresh_mapcalc_property_list)
        l_prop.addWidget(self.txt_mapcalc_prop_filter)

        self.lst_mapcalc_properties = QtWidgets.QListWidget()
        self.lst_mapcalc_properties.setMinimumHeight(130)
        self.lst_mapcalc_properties.setMaximumHeight(210)
        self.lst_mapcalc_properties.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )
        self.lst_mapcalc_properties.itemSelectionChanged.connect(self._on_mapcalc_property_selected)
        l_prop.addWidget(self.lst_mapcalc_properties)

        lay.addWidget(gb_prop)
        self._compact_groupbox(gb_op)
        self._compact_groupbox(gb_info)
        self._compact_groupbox(gb_prop)
        self._prepare_mapcalc_page_widget(page)
        self.mapcalc_config_stack.addWidget(page)
        self._update_mapcalc_property_info_box()


    def _build_mapcalc_ensemble_page(self):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(8)

        gb_group = QtWidgets.QGroupBox("Grupo de descritores")
        grid_group = QtWidgets.QGridLayout(gb_group)
        grid_group.setSpacing(6)

        self.mapcalc_ensemble_metric_group_buttons = QtWidgets.QButtonGroup(self)
        self.mapcalc_ensemble_metric_group_buttons.setExclusive(True)

        groups = self._mapcalc_vertical_metric_groups()
        for idx, (key, group) in enumerate(groups.items()):
            b = QtWidgets.QPushButton(group.get("title", key))
            b.setCheckable(True)
            b.setMinimumHeight(30)
            b.setToolTip(group.get("description", ""))
            b.clicked.connect(lambda checked=False, k=key: self._set_mapcalc_metric_group(k, panel="ensemble"))
            self.mapcalc_ensemble_metric_group_buttons.addButton(b)
            grid_group.addWidget(b, idx // 2, idx % 2)
            if idx == 0:
                b.setChecked(True)
                self.mapcalc_ensemble_group_key = key

        lay.addWidget(gb_group)

        gb_metric = QtWidgets.QGroupBox("Métrica vertical alvo")
        self.mapcalc_ensemble_metric_items_layout = QtWidgets.QGridLayout(gb_metric)
        self.mapcalc_ensemble_metric_items_layout.setSpacing(6)
        self.mapcalc_ensemble_metric_buttons = QtWidgets.QButtonGroup(self)
        self.mapcalc_ensemble_metric_buttons.setExclusive(True)
        lay.addWidget(gb_metric)

        gb_stat = QtWidgets.QGroupBox("Estatística")
        grid_stat = QtWidgets.QGridLayout(gb_stat)
        grid_stat.setSpacing(6)

        self.mapcalc_stat_buttons = QtWidgets.QButtonGroup(self)
        self.mapcalc_stat_buttons.setExclusive(True)

        stats = [
            ("Média", "mean"),
            ("Desvio padrão", "std"),
            ("Variância", "var"),
            ("Amplitude", "range"),
        ]

        for idx, (label, key) in enumerate(stats):
            meta = self._mapcalc_stat_meta(key)
            b = QtWidgets.QPushButton(label)
            b.setCheckable(True)
            b.setMinimumHeight(28)
            b.setToolTip(meta["title"])
            b.clicked.connect(lambda checked=False, k=key: self.set_mapcalc_stat(k))
            self.mapcalc_stat_buttons.addButton(b)
            grid_stat.addWidget(b, idx // 2, idx % 2)
            if idx == 0:
                b.setChecked(True)

        lay.addWidget(gb_stat)

        gb_scope = QtWidgets.QGroupBox("Saída")
        grid_scope = QtWidgets.QGridLayout(gb_scope)
        grid_scope.setSpacing(6)

        self.mapcalc_scope_buttons = QtWidgets.QButtonGroup(self)
        self.mapcalc_scope_buttons.setExclusive(True)

        scopes = [
            ("Coluna 2D", "column"),
            ("Resumo / Tabela", "model"),
        ]

        for idx, (label, key) in enumerate(scopes):
            b = QtWidgets.QPushButton(label)
            b.setCheckable(True)
            b.setMinimumHeight(28)
            b.clicked.connect(lambda checked=False, k=key: self.set_mapcalc_scope(k))
            self.mapcalc_scope_buttons.addButton(b)
            grid_scope.addWidget(b, 0, idx)
            if key == "column":
                b.setChecked(True)

        lay.addWidget(gb_scope)

        gb_info = QtWidgets.QGroupBox("Explicação")
        l_info = QtWidgets.QVBoxLayout(gb_info)
        self.txt_mapcalc_ensemble_metric_info = self._make_mapcalc_info_box(min_height=80, max_height=105)
        self.txt_mapcalc_ensemble_info = self._make_mapcalc_info_box(min_height=80, max_height=105)
        l_info.addWidget(self.txt_mapcalc_ensemble_metric_info)
        l_info.addWidget(self.txt_mapcalc_ensemble_info)
        lay.addWidget(gb_info)

        self._compact_groupbox(gb_group)
        self._compact_groupbox(gb_metric)
        self._compact_groupbox(gb_stat)
        self._compact_groupbox(gb_scope)
        self._compact_groupbox(gb_info)
        self._prepare_mapcalc_page_widget(page)
        self.mapcalc_config_stack.addWidget(page)

        first_key = next(iter(groups.keys())) if groups else "espessura"
        self._set_mapcalc_metric_group(first_key, panel="ensemble")
        self._update_mapcalc_ensemble_info_box()


    def _build_mapcalc_uncertainty_page(self):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(8)

        gb_unc = QtWidgets.QGroupBox("Incerteza / discordância")
        l_unc = QtWidgets.QVBoxLayout(gb_unc)

        info = QtWidgets.QLabel(
            "Nesta etapa, a incerteza está focada na Entropia de Fácies, que mede a discordância categórica entre modelos célula a célula."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #555;")
        l_unc.addWidget(info)

        self.txt_mapcalc_uncertainty_info = self._make_mapcalc_info_box(min_height=95, max_height=115)
        l_unc.addWidget(self.txt_mapcalc_uncertainty_info)

        self.mapcalc_uncertainty_target = "facies_entropy"
        lay.addWidget(gb_unc)
        self._compact_groupbox(gb_unc)

        gb_scope = QtWidgets.QGroupBox("Saída")
        l_scope = QtWidgets.QVBoxLayout(gb_scope)
        l_scope.setContentsMargins(8, 8, 8, 8)

        self.lbl_mapcalc_uncert_scope = QtWidgets.QLabel(
            "Resultado: campo 3D de entropia célula a célula."
        )
        self.lbl_mapcalc_uncert_scope.setWordWrap(True)
        self.lbl_mapcalc_uncert_scope.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Maximum
        )

        l_scope.addWidget(self.lbl_mapcalc_uncert_scope)

        gb_scope.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Maximum
        )
        gb_scope.setMaximumHeight(85)

        lay.addWidget(gb_scope)

        self._compact_groupbox(gb_scope)
        self._prepare_mapcalc_page_widget(page)
        self.mapcalc_config_stack.addWidget(page)
        self._update_mapcalc_uncertainty_info_box()

    def _build_mapcalc_difference_page(self):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(8)

        info = QtWidgets.QLabel(
            "Diferença entre modelos será implementada depois.\n\n"
        )
        info.setWordWrap(True)
        lay.addWidget(info)
        page.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Maximum
        )

        self._prepare_mapcalc_page_widget(page)
        self.mapcalc_config_stack.addWidget(page)

    def show_mapcalc_page(self, mode_key="vertical"):
        """Abre a página de Cálculo de Mapas, fecha docks externos e ativa um modo."""
        # A aba Mapas tem seus próprios painéis laterais; os docks do modo Visualizar confundem aqui.
        try:
            if hasattr(self, "dock_explorer"):
                self.dock_explorer.hide()
            if hasattr(self, "dock_props"):
                self.dock_props.hide()
            if hasattr(self, "dock_map2d_summary"):
                self.dock_map2d_summary.hide()
        except Exception:
            pass

        # Usa a perspectiva interna de mapas/cálculo sem passar pela UI antiga de incerteza.
        try:
            self.switch_perspective("uncertainty")
        except Exception:
            if hasattr(self, "central_stack") and hasattr(self, "uncertainty_page"):
                self.central_stack.setCurrentWidget(self.uncertainty_page)

        try:
            if hasattr(self, "central_stack") and hasattr(self, "uncertainty_page"):
                self.central_stack.setCurrentWidget(self.uncertainty_page)
        except Exception:
            pass

        try:
            if hasattr(self, "ribbon_tabs"):
                for i in range(self.ribbon_tabs.count()):
                    if self.ribbon_tabs.tabText(i) == "Mapas":
                        if self.ribbon_tabs.currentIndex() != i:
                            self.ribbon_tabs.blockSignals(True)
                            self.ribbon_tabs.setCurrentIndex(i)
                            self.ribbon_tabs.blockSignals(False)
                        break
        except Exception:
            pass

        try:
            self._refresh_mapcalc_models_panel()
            self._refresh_mapcalc_target_facies_table()
            self._update_mapcalc_target_facies_label()
            self._update_mapcalc_model_status_label()
        except Exception:
            pass

        self.set_mapcalc_mode(mode_key)
        try:
            QtCore.QTimer.singleShot(0, self._normalize_mapcalc_window_size)
            QtCore.QTimer.singleShot(120, self._normalize_mapcalc_window_size)
        except Exception:
            pass
        # Calcula uma vez ao abrir a aba Mapas, mas sem modal de erro se algo ainda não estiver pronto.
        self._schedule_mapcalc_auto_update(350)


    def _normalize_mapcalc_window_size(self):
        """Evita que a aba Mapas force a janela para fora da tela.

        Os painéis laterais agora são roláveis; esta rotina apenas remove
        mínimos herdados e reaplica a maximização quando necessário.
        """
        try:
            self.setMinimumSize(0, 0)
            if hasattr(self, "uncertainty_page"):
                self.uncertainty_page.setMinimumSize(0, 0)
            if hasattr(self, "mapcalc_left_scroll"):
                self.mapcalc_left_scroll.setMinimumHeight(0)
            if hasattr(self, "mapcalc_right_scroll"):
                self.mapcalc_right_scroll.setMinimumHeight(0)
            if self.isMaximized():
                self.showMaximized()
        except Exception:
            pass


    def set_mapcalc_mode(self, mode_key):
        """Troca o painel contextual do Cálculo de Mapas."""
        self.mapcalc_mode = str(mode_key or "vertical")

        idx_map = {"vertical": 0, "property": 1, "ensemble": 2, "uncertainty": 3, "difference": 4}
        idx = idx_map.get(self.mapcalc_mode, 0)
        if hasattr(self, "mapcalc_config_stack"):
            self.mapcalc_config_stack.setCurrentIndex(idx)

        ribbon_btn_map = {
            "vertical": getattr(self, "btn_mapcalc_vertical", None),
            "property": getattr(self, "btn_mapcalc_property", None),
            "ensemble": getattr(self, "btn_mapcalc_ensemble", None),
            "uncertainty": getattr(self, "btn_mapcalc_uncert", None),
            "difference": getattr(self, "btn_mapcalc_diff", None),
        }
        rb = ribbon_btn_map.get(self.mapcalc_mode)
        if rb is not None:
            rb.setChecked(True)

        is_multi = self.mapcalc_mode in ("ensemble", "uncertainty")
        for attr in ("btn_mapcalc_all_models", "btn_mapcalc_no_models"):
            w = getattr(self, attr, None)
            if w is not None:
                w.setVisible(is_multi)

        if hasattr(self, "lbl_mapcalc_models_hint"):
            self.lbl_mapcalc_models_hint.setText("Marque os modelos que entram no cálculo." if is_multi else "Selecione o modelo usado neste mapa.")

        if hasattr(self, "mapcalc_right_tabs"):
            if self.mapcalc_mode in ("vertical", "ensemble"):
                self.mapcalc_right_tabs.setCurrentWidget(self.mapcalc_facies_page)
            elif self.mapcalc_mode == "uncertainty":
                self.mapcalc_right_tabs.setCurrentWidget(self.mapcalc_geometry_page)
            else:
                self.mapcalc_right_tabs.setCurrentWidget(self.mapcalc_facies_page)

        if self.mapcalc_mode == "vertical":
            if not isinstance(getattr(self, "mapcalc_selected_metric", None), dict):
                self.mapcalc_selected_metric = {
                    "scalar": "__total_column_thickness__",
                    "title": "Espessura total da coluna (m)",
                    "label": "Espessura",
                }
            self.lbl_mapcalc_title.setText("<b>Mapa Vertical por Coluna</b>")
            self.lbl_mapcalc_help.setText("Calcule métricas por coluna do modelo.")
            self.btn_run_mapcalc.setText("Calcular Mapa Vertical")
            self._set_uncert_result_mode("column")
            self._update_mapcalc_vertical_info_box()
            try:
                if hasattr(self, "mapcalc_right_tabs") and hasattr(self, "mapcalc_facies_page"):
                    self.mapcalc_right_tabs.setCurrentWidget(self.mapcalc_facies_page)
            except Exception:
                pass
        elif self.mapcalc_mode == "property":
            self.lbl_mapcalc_title.setText("<b>Propriedade por Coluna</b>")
            self.lbl_mapcalc_help.setText("Transforme uma propriedade por célula em mapa por coluna.")
            self.btn_run_mapcalc.setText("Calcular Propriedade")
            self._set_uncert_result_mode("column")
            self._refresh_mapcalc_property_list()
            self._update_mapcalc_property_info_box()
        elif self.mapcalc_mode == "ensemble":
            self.lbl_mapcalc_title.setText("<b>Estatística do Ensemble</b>")
            self.lbl_mapcalc_help.setText("Escolha a métrica vertical, os modelos e a estatística do ensemble.")
            self.btn_run_mapcalc.setText("Calcular Estatística")
            self._set_uncert_result_mode(getattr(self, "mapcalc_selected_scope", "column"))
            self._update_mapcalc_vertical_info_box()
            self._update_mapcalc_ensemble_info_box()
        elif self.mapcalc_mode == "uncertainty":
            self.lbl_mapcalc_title.setText("<b>Incerteza / Discordância</b>")
            self.lbl_mapcalc_help.setText("Entropia de fácies entre modelos: discordância categórica célula a célula.")
            self.btn_run_mapcalc.setText("Calcular Incerteza")
            self._set_uncert_result_mode("cell")
            self._update_mapcalc_uncertainty_info_box()
        elif self.mapcalc_mode == "difference":
            self.lbl_mapcalc_title.setText("<b>Diferença entre Modelos</b>")
            self.lbl_mapcalc_help.setText("Mapas de diferença serão implementados na próxima etapa.")
            self.btn_run_mapcalc.setText("Calcular Diferença")
            self._set_uncert_result_mode("column")

        try:
            QtCore.QTimer.singleShot(0, self._sync_mapcalc_stack_height)
        except Exception:
            pass

        self._update_mapcalc_model_status_label()
        self._schedule_mapcalc_auto_update()


    def set_mapcalc_vertical_metric(self, scalar, title, label=None):
        meta = self._get_mapcalc_metric_meta_by_scalar(scalar)
        if meta is None:
            meta = {
                "scalar": scalar,
                "title": title,
                "label": label or title,
                "formula": "-",
                "description": title or label or "Métrica vertical",
            }
        self.set_mapcalc_vertical_metric_from_meta(meta, schedule=True)


    def set_mapcalc_property_operation(self, operation):
        self.mapcalc_selected_operation = str(operation or "weighted_mean")
        self._update_mapcalc_property_info_box()
        self._schedule_mapcalc_auto_update()


    def set_mapcalc_stat(self, stat):
        self.mapcalc_selected_stat = str(stat or "std")
        self._update_mapcalc_ensemble_info_box()
        self._schedule_mapcalc_auto_update()


    def set_mapcalc_scope(self, scope):
        self.mapcalc_selected_scope = str(scope or "column")
        self._set_uncert_result_mode(self.mapcalc_selected_scope)
        self._update_mapcalc_ensemble_info_box()
        self._schedule_mapcalc_auto_update()


    def set_mapcalc_uncertainty_target(self, target):
        self.mapcalc_uncertainty_target = str(target or "facies_entropy")
        self._update_mapcalc_uncertainty_info_box()
        self._schedule_mapcalc_auto_update()

    def _ensure_mapcalc_auto_timer(self):
        if hasattr(self, "_mapcalc_auto_timer"):
            return
        self._mapcalc_auto_timer = QtCore.QTimer(self)
        self._mapcalc_auto_timer.setSingleShot(True)
        self._mapcalc_auto_timer.timeout.connect(self._run_mapcalc_auto_update)

    def _schedule_mapcalc_auto_update(self, delay_ms=250):
        if not getattr(self, "_mapcalc_ready", False):
            return
        if not hasattr(self, "central_stack") or not hasattr(self, "uncertainty_page"):
            return
        try:
            if self.central_stack.currentWidget() is not self.uncertainty_page:
                return
        except Exception:
            return
        self._ensure_mapcalc_auto_timer()
        self._mapcalc_auto_timer.start(int(delay_ms))

    def _run_mapcalc_auto_update(self):
        if not getattr(self, "_mapcalc_ready", False):
            return
        self._mapcalc_auto_running = True
        try:
            self.run_map_calculation()
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                self.statusBar().showMessage(f"Erro no cálculo automático de mapas: {e}", 6000)
            except Exception:
                pass
        finally:
            self._mapcalc_auto_running = False

    def _get_all_loaded_facies_ids_for_mapcalc(self):
        ids = set()
        for _k, m in getattr(self, "models", {}).items():
            arr = m.get("facies") if isinstance(m, dict) else None
            if arr is None:
                continue
            try:
                vals = np.unique(np.asarray(arr).ravel())
                for v in vals:
                    if np.isfinite(v):
                        ids.add(int(v))
            except Exception:
                pass
        if not ids:
            try:
                ids.update(int(f) for f, _ in (self.facies_reference or []))
            except Exception:
                pass
        return sorted(ids)

    def _refresh_mapcalc_target_facies_table(self):
        if not hasattr(self, "tbl_mapcalc_target_facies"):
            return
        current = set(int(x) for x in (self.state.get("reservoir_facies", set()) or set()))
        ids = self._get_all_loaded_facies_ids_for_mapcalc()
        colors = getattr(self, "facies_colors_dict", {}) or getattr(self, "facies_colors", {}) or {}
        self._updating_mapcalc_facies = True
        try:
            self.tbl_mapcalc_target_facies.clearContents()
            self.tbl_mapcalc_target_facies.setRowCount(len(ids))
            for row, fid in enumerate(ids):
                rgba = colors.get(int(fid), (0.8, 0.8, 0.8, 1.0))
                try:
                    r, g, b = [int(float(c) * 255) for c in rgba[:3]]
                except Exception:
                    r, g, b = 200, 200, 200
                item_color = QtWidgets.QTableWidgetItem("")
                item_color.setBackground(QBrush(QColor(r, g, b)))
                item_color.setFlags(QtCore.Qt.ItemIsEnabled)
                self.tbl_mapcalc_target_facies.setItem(row, 0, item_color)

                item_id = QtWidgets.QTableWidgetItem(str(int(fid)))
                item_id.setData(QtCore.Qt.UserRole, int(fid))
                item_id.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                self.tbl_mapcalc_target_facies.setItem(row, 1, item_id)

                item_chk = QtWidgets.QTableWidgetItem("")
                item_chk.setData(QtCore.Qt.UserRole, int(fid))
                item_chk.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsSelectable)
                item_chk.setCheckState(QtCore.Qt.Checked if int(fid) in current else QtCore.Qt.Unchecked)
                self.tbl_mapcalc_target_facies.setItem(row, 2, item_chk)
            self.tbl_mapcalc_target_facies.resizeColumnsToContents()
        finally:
            self._updating_mapcalc_facies = False

    def _on_mapcalc_target_facies_item_changed(self, item):
        if getattr(self, "_updating_mapcalc_facies", False):
            return
        if item is None or item.column() != 2:
            return
        selected = set()
        for row in range(self.tbl_mapcalc_target_facies.rowCount()):
            chk = self.tbl_mapcalc_target_facies.item(row, 2)
            id_item = self.tbl_mapcalc_target_facies.item(row, 1)
            if chk is not None and id_item is not None and chk.checkState() == QtCore.Qt.Checked:
                try:
                    selected.add(int(id_item.data(QtCore.Qt.UserRole)))
                except Exception:
                    try:
                        selected.add(int(id_item.text()))
                    except Exception:
                        pass
        self.state["reservoir_facies"] = set(selected)
        self.state["reservoir_facies_raw"] = set(selected)
        try:
            self.state_reservoir_raw = set(selected)
        except Exception:
            pass
        self._update_mapcalc_target_facies_label()
        self._schedule_mapcalc_auto_update()

    def _set_all_mapcalc_target_facies(self, checked=True):
        if not hasattr(self, "tbl_mapcalc_target_facies"):
            return
        self._updating_mapcalc_facies = True
        try:
            for row in range(self.tbl_mapcalc_target_facies.rowCount()):
                item = self.tbl_mapcalc_target_facies.item(row, 2)
                if item is not None:
                    item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        finally:
            self._updating_mapcalc_facies = False
        selected = set()
        if checked:
            for row in range(self.tbl_mapcalc_target_facies.rowCount()):
                id_item = self.tbl_mapcalc_target_facies.item(row, 1)
                if id_item is not None:
                    try:
                        selected.add(int(id_item.data(QtCore.Qt.UserRole)))
                    except Exception:
                        pass
        self.state["reservoir_facies"] = set(selected)
        self.state["reservoir_facies_raw"] = set(selected)
        self._update_mapcalc_target_facies_label()
        self._schedule_mapcalc_auto_update()

    def _refresh_mapcalc_models_panel(self):
        """Atualiza a lista lateral de modelos da página de Cálculo de Mapas."""
        if not hasattr(self, "lst_mapcalc_models"):
            return

        current_key = getattr(self, "mapcalc_model_key", "base")
        checked_keys = set(self._get_checked_mapcalc_model_keys()) if hasattr(self, "lst_mapcalc_models") else set()
        if not checked_keys:
            checked_keys = {"base"}

        self._updating_mapcalc_models = True
        try:
            self.lst_mapcalc_models.clear()

            for k, v in self.models.items():
                if k == "compare":
                    continue

                if k == "base":
                    name = "Modelo Base"
                else:
                    name = v.get("name", str(k))

                if k != "base" and v.get("grid") is None and v.get("facies") is None:
                    continue

                it = QtWidgets.QListWidgetItem(str(name))
                it.setData(QtCore.Qt.UserRole, str(k))
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                it.setCheckState(QtCore.Qt.Checked if str(k) in checked_keys else QtCore.Qt.Unchecked)
                self.lst_mapcalc_models.addItem(it)

                if str(k) == str(current_key):
                    self.lst_mapcalc_models.setCurrentItem(it)

            if self.lst_mapcalc_models.currentItem() is None and self.lst_mapcalc_models.count() > 0:
                self.lst_mapcalc_models.setCurrentRow(0)
                current_item = self.lst_mapcalc_models.currentItem()
                if current_item is not None:
                    self.mapcalc_model_key = current_item.data(QtCore.Qt.UserRole)

        finally:
            self._updating_mapcalc_models = False

        self._update_mapcalc_model_status_label()


    def _on_mapcalc_model_selection_changed(self):
        if getattr(self, "_updating_mapcalc_models", False):
            return

        item = self.lst_mapcalc_models.currentItem() if hasattr(self, "lst_mapcalc_models") else None
        if item is None:
            return

        key = item.data(QtCore.Qt.UserRole)
        if key is not None:
            self.mapcalc_model_key = str(key)
            if self.mapcalc_mode not in ("ensemble", "uncertainty"):
                # Em mapas de um modelo, a seleção também marca o modelo ativo.
                self._updating_mapcalc_models = True
                try:
                    for i in range(self.lst_mapcalc_models.count()):
                        it = self.lst_mapcalc_models.item(i)
                        it.setCheckState(QtCore.Qt.Checked if it is item else QtCore.Qt.Unchecked)
                finally:
                    self._updating_mapcalc_models = False

        self._update_mapcalc_model_status_label()
        self._schedule_mapcalc_auto_update()


    def _on_mapcalc_model_item_changed(self, item):
        if getattr(self, "_updating_mapcalc_models", False):
            return

        if self.mapcalc_mode not in ("ensemble", "uncertainty") and item.checkState() == QtCore.Qt.Checked:
            self._updating_mapcalc_models = True
            try:
                for i in range(self.lst_mapcalc_models.count()):
                    it = self.lst_mapcalc_models.item(i)
                    if it is not item:
                        it.setCheckState(QtCore.Qt.Unchecked)
                self.lst_mapcalc_models.setCurrentItem(item)
                key = item.data(QtCore.Qt.UserRole)
                if key is not None:
                    self.mapcalc_model_key = str(key)
            finally:
                self._updating_mapcalc_models = False

        self._update_mapcalc_model_status_label()
        self._schedule_mapcalc_auto_update()


    def _set_all_mapcalc_models_checked(self, checked=True):
        if not hasattr(self, "lst_mapcalc_models"):
            return

        self._updating_mapcalc_models = True
        try:
            for i in range(self.lst_mapcalc_models.count()):
                self.lst_mapcalc_models.item(i).setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        finally:
            self._updating_mapcalc_models = False

        if not checked and self.lst_mapcalc_models.count() > 0 and self.mapcalc_mode not in ("ensemble", "uncertainty"):
            it = self.lst_mapcalc_models.item(0)
            it.setCheckState(QtCore.Qt.Checked)
            self.lst_mapcalc_models.setCurrentItem(it)
            self.mapcalc_model_key = str(it.data(QtCore.Qt.UserRole))

        self._update_mapcalc_model_status_label()
        self._schedule_mapcalc_auto_update()


    def _get_checked_mapcalc_model_keys(self):
        keys = []
        if not hasattr(self, "lst_mapcalc_models"):
            return keys

        for i in range(self.lst_mapcalc_models.count()):
            it = self.lst_mapcalc_models.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                key = it.data(QtCore.Qt.UserRole)
                if key is not None:
                    keys.append(str(key))
        return keys


    def _sync_mapcalc_model_selection_from_panel(self, single=False):
        """Atualiza mapcalc_model_key ou lista de ensemble a partir do painel lateral."""
        if single:
            item = self.lst_mapcalc_models.currentItem() if hasattr(self, "lst_mapcalc_models") else None
            if item is not None:
                key = item.data(QtCore.Qt.UserRole)
                if key is not None:
                    self.mapcalc_model_key = str(key)
            return [getattr(self, "mapcalc_model_key", "base")]

        keys = self._get_checked_mapcalc_model_keys()
        if not keys:
            # fallback seguro: usa o modelo atual
            keys = [getattr(self, "mapcalc_model_key", "base")]
        self._set_uncert_models_from_keys(keys)
        return keys


    def _update_mapcalc_model_status_label(self):
        if not hasattr(self, "lbl_uncert_n"):
            return

        if getattr(self, "mapcalc_mode", "vertical") in ("ensemble", "uncertainty"):
            n = len(self._get_checked_mapcalc_model_keys())
            self.lbl_uncert_n.setText(f"Entrada: {n} modelo(s) marcado(s)")
        else:
            key = getattr(self, "mapcalc_model_key", "base")
            name = self.models.get(key, {}).get("name", "Modelo Base" if key == "base" else key)
            self.lbl_uncert_n.setText(f"Entrada: {name}")


    def _get_selected_model_keys_from_explorer(self):
        """Compatibilidade: tenta ler os modelos selecionados no Project Explorer principal."""
        keys = []
        if not hasattr(self, "project_tree"):
            return keys

        for it in self.project_tree.selectedItems():
            key = it.data(0, QtCore.Qt.UserRole + 1)
            if key is None and it.parent() is not None:
                key = it.parent().data(0, QtCore.Qt.UserRole + 1)
            if key is None:
                continue
            key = str(key)
            if key in self.models and key not in keys:
                keys.append(key)
        return keys


    def set_mapcalc_single_model(self, model_key):
        model_key = str(model_key or "base")
        if model_key not in self.models:
            model_key = "base"
        self.mapcalc_model_key = model_key

        if hasattr(self, "lst_mapcalc_models"):
            self._updating_mapcalc_models = True
            try:
                for i in range(self.lst_mapcalc_models.count()):
                    it = self.lst_mapcalc_models.item(i)
                    k = str(it.data(QtCore.Qt.UserRole))
                    is_current = (k == model_key)
                    it.setCheckState(QtCore.Qt.Checked if is_current else QtCore.Qt.Unchecked)
                    if is_current:
                        self.lst_mapcalc_models.setCurrentItem(it)
            finally:
                self._updating_mapcalc_models = False

        self._update_mapcalc_model_status_label()


    def set_mapcalc_single_model_from_explorer(self):
        keys = self._get_selected_model_keys_from_explorer()
        if keys:
            self.set_mapcalc_single_model(keys[0])
        else:
            self._sync_mapcalc_model_selection_from_panel(single=True)


    def set_mapcalc_ensemble_from_explorer(self):
        keys = self._get_selected_model_keys_from_explorer()
        if not keys:
            keys = self._get_checked_mapcalc_model_keys()
        self._set_uncert_models_from_keys(keys)
        self._update_mapcalc_model_status_label()


    def set_mapcalc_ensemble_all(self):
        keys = []
        for k, v in self.models.items():
            if k == "compare":
                continue
            if k == "base" or v.get("grid") is not None or v.get("facies") is not None:
                keys.append(str(k))

        if hasattr(self, "lst_mapcalc_models"):
            self._updating_mapcalc_models = True
            try:
                for i in range(self.lst_mapcalc_models.count()):
                    it = self.lst_mapcalc_models.item(i)
                    it.setCheckState(QtCore.Qt.Checked if str(it.data(QtCore.Qt.UserRole)) in keys else QtCore.Qt.Unchecked)
            finally:
                self._updating_mapcalc_models = False

        self._set_uncert_models_from_keys(keys)
        self._update_mapcalc_model_status_label()


    def _set_uncert_models_from_keys(self, keys):
        """Mantém compatibilidade com calculate_uncertainty(), que usa lst_uncert_models."""
        if not hasattr(self, "lst_uncert_models"):
            self.lst_uncert_models = QtWidgets.QListWidget()
            self.lst_uncert_models.hide()

        self.lst_uncert_models.clear()
        key_set = {str(k) for k in (keys or [])}

        for k, v in self.models.items():
            if k == "compare":
                continue
            name = "Modelo Base" if k == "base" else v.get("name", str(k))
            it = QtWidgets.QListWidgetItem(str(name))
            it.setData(QtCore.Qt.UserRole, str(k))
            it.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if str(k) in key_set else QtCore.Qt.Unchecked)
            self.lst_uncert_models.addItem(it)

    # --- Callbacks necessários para a interface acima não dar erro ---
    
    def on_uncert_settings_changed(self, state=None):
        """
        Atualiza a interface de Cálculo de Mapas.

        Nesta tela, não recalculamos automaticamente ao trocar combos.
        O usuário configura e depois clica em 'Calcular Mapa'.
        """
        self._apply_mapcalc_category_ui()

        # Marca que a configuração mudou, mas não recalcula automaticamente.
        self._uncert_has_result = False
    
    def _on_mapcalc_category_changed(self):
        """
        Atualiza a tela quando o usuário troca a categoria do cálculo.
        """
        self._uncert_has_result = False

        self._refresh_mapcalc_model_combo()
        self._refresh_mapcalc_property_combo()
        self._apply_mapcalc_category_ui()
        self._clear_uncert_summary_table()

        if hasattr(self, "lbl_uncert_n"):
            self.lbl_uncert_n.setText("Modelos (N): -")
        if hasattr(self, "lbl_uncert_max_theo"):
            self.lbl_uncert_max_theo.setText("Info: -")
        if hasattr(self, "lbl_uncert_max_real"):
            self.lbl_uncert_max_real.setText("Resultado: -")


    def _apply_mapcalc_category_ui(self):
        """
        Mostra/esconde campos conforme a categoria selecionada.

        Categorias:
            vertical    -> um modelo + métrica vertical
            property    -> um modelo + propriedade + operação vertical
            ensemble    -> vários modelos + estatística
            uncertainty -> vários modelos + discordância/incerteza
            difference  -> reservado
        """
        if not hasattr(self, "cmb_mapcalc_category"):
            return

        cat = self.cmb_mapcalc_category.currentData()

        is_vertical = cat == "vertical"
        is_property = cat == "property"
        is_ensemble = cat == "ensemble"
        is_uncertainty = cat == "uncertainty"
        is_difference = cat == "difference"

        is_single_model = is_vertical or is_property
        is_ensemble_like = is_ensemble or is_uncertainty

        # ------------------------------------------------------------
        # Campo: modelo único
        # ------------------------------------------------------------
        self.lbl_mapcalc_model.setVisible(is_single_model)
        self.cmb_mapcalc_model.setVisible(is_single_model)

        # ------------------------------------------------------------
        # Campo: propriedade/descritor
        # ------------------------------------------------------------
        self.lbl_uncert_prop.setVisible(not is_difference)
        self.cmb_uncert_prop.setVisible(not is_difference)

        # ------------------------------------------------------------
        # Campo: operação vertical
        # ------------------------------------------------------------
        self.lbl_mapcalc_operation.setVisible(is_property)
        self.cmb_mapcalc_operation.setVisible(is_property)

        # ------------------------------------------------------------
        # Campo: escala
        # ------------------------------------------------------------
        self.lbl_uncert_scope.setVisible(is_ensemble_like)
        self.cmb_uncert_scope.setVisible(is_ensemble_like)

        # ------------------------------------------------------------
        # Campo: estatística
        # ------------------------------------------------------------
        self.lbl_uncert_metric.setVisible(is_ensemble_like)
        self.cmb_uncert_metric.setVisible(is_ensemble_like)

        # ------------------------------------------------------------
        # Lista de modelos
        # ------------------------------------------------------------
        self.lbl_uncert_models.setVisible(is_ensemble_like)
        self.lst_uncert_models.setVisible(is_ensemble_like)

        # Botões Todos/Nenhum
        try:
            for i in range(self._mapcalc_model_buttons_layout.count()):
                item = self._mapcalc_model_buttons_layout.itemAt(i)
                if item and item.widget():
                    item.widget().setVisible(is_ensemble_like)
        except Exception:
            pass

        # ------------------------------------------------------------
        # Escala absoluta: somente entropia de fácies
        # ------------------------------------------------------------
        data = self.cmb_uncert_prop.currentData() if hasattr(self, "cmb_uncert_prop") else None
        is_entropy = isinstance(data, dict) and data.get("kind") == "facies"
        self.chk_abs_scale.setVisible(bool(is_entropy and is_uncertainty))

        # ------------------------------------------------------------
        # Textos, botão e página central
        # ------------------------------------------------------------
        if is_vertical:
            self.lbl_mapcalc_help.setText(
                "Gere mapas verticais por coluna para um único modelo. "
                "Use para espessura total, proporção, ICV, Qv, pacotes, gaps, trocas e permanências."
            )
            self.lbl_uncert_prop.setText("Métrica vertical:")
            self.btn_run_mapcalc.setText("Calcular Mapa Vertical")
            self._set_uncert_result_mode("column")

        elif is_property:
            self.lbl_mapcalc_help.setText(
                "Gere mapas por coluna a partir de propriedades por célula. "
                "Use para média ponderada por espessura, espessura equivalente, soma vertical ou média vertical."
            )
            self.lbl_uncert_prop.setText("Propriedade:")
            self.btn_run_mapcalc.setText("Calcular Propriedade")
            self._set_uncert_result_mode("column")

        elif is_ensemble:
            self.lbl_mapcalc_help.setText(
                "Calcule estatísticas entre modelos. "
                "A média representa o mapa médio do ensemble; desvio, variância e amplitude representam variabilidade."
            )
            self.lbl_uncert_prop.setText("Mapa / propriedade / descritor:")
            self.btn_run_mapcalc.setText("Calcular Estatística")

            scope = self.cmb_uncert_scope.currentData() if hasattr(self, "cmb_uncert_scope") else "cell"
            self._set_uncert_result_mode(scope)

        elif is_uncertainty:
            self.lbl_mapcalc_help.setText(
                "Calcule mapas de discordância ou incerteza. "
                "Para fácies, use entropia. Para propriedades contínuas, use desvio padrão, variância ou amplitude."
            )
            self.lbl_uncert_prop.setText("Alvo da discordância:")
            self.btn_run_mapcalc.setText("Calcular Discordância")

            # Evita chamar média de incerteza.
            if hasattr(self, "cmb_uncert_metric") and self.cmb_uncert_metric.currentData() == "mean":
                idx_std = self.cmb_uncert_metric.findData("std")
                if idx_std >= 0:
                    self.cmb_uncert_metric.blockSignals(True)
                    self.cmb_uncert_metric.setCurrentIndex(idx_std)
                    self.cmb_uncert_metric.blockSignals(False)

            scope = self.cmb_uncert_scope.currentData() if hasattr(self, "cmb_uncert_scope") else "cell"
            self._set_uncert_result_mode(scope)

        elif is_difference:
            self.lbl_mapcalc_help.setText(
                "A categoria Diferença entre modelos será implementada depois. "
                "Ela será usada para mapas como modelo - base, diferença absoluta e diferença percentual."
            )
            self.btn_run_mapcalc.setText("Calcular Diferença")
            self._set_uncert_result_mode("column")


    def _set_uncert_result_mode(self, scope):
        """Escolhe qual página central deve aparecer, mantendo o painel direito visível na aba Mapas."""
        if not hasattr(self, "uncert_result_stack"):
            return
        scope = str(scope or "cell").lower()
        if scope == "model":
            self.uncert_result_stack.setCurrentIndex(2)
        elif scope == "column":
            self.uncert_result_stack.setCurrentIndex(1)
        else:
            self.uncert_result_stack.setCurrentIndex(0)
            try:
                if hasattr(self, "mapcalc_right_tabs") and hasattr(self, "mapcalc_geometry_page"):
                    self.mapcalc_right_tabs.setCurrentWidget(self.mapcalc_geometry_page)
            except Exception:
                pass
        if hasattr(self, "uncert_right_panel"):
            self.uncert_right_panel.setVisible(True)
        if hasattr(self, "mapcalc_right_scroll"):
            self.mapcalc_right_scroll.setVisible(True)

    def _clear_uncert_summary_table(self):
        if hasattr(self, "tbl_uncert_summary"):
            self.tbl_uncert_summary.clear()
            self.tbl_uncert_summary.setRowCount(0)
            self.tbl_uncert_summary.setColumnCount(0)

        if hasattr(self, "txt_uncert_summary"):
            self.txt_uncert_summary.clear()


    def _fill_uncert_summary_table(self, df):
        if not hasattr(self, "tbl_uncert_summary"):
            return

        self.tbl_uncert_summary.setSortingEnabled(False)
        self.tbl_uncert_summary.clear()

        if df is None or df.empty:
            self.tbl_uncert_summary.setRowCount(0)
            self.tbl_uncert_summary.setColumnCount(0)
            self.tbl_uncert_summary.setSortingEnabled(True)
            return

        self.tbl_uncert_summary.setRowCount(len(df))
        self.tbl_uncert_summary.setColumnCount(len(df.columns))
        self.tbl_uncert_summary.setHorizontalHeaderLabels([str(c) for c in df.columns])

        for r in range(len(df)):
            for c, col in enumerate(df.columns):
                val = df.iloc[r, c]

                if isinstance(val, (float, np.floating)):
                    txt = f"{float(val):.6g}"
                else:
                    txt = str(val)

                item = QtWidgets.QTableWidgetItem(txt)

                if isinstance(val, (int, float, np.integer, np.floating)):
                    item.setData(QtCore.Qt.UserRole, float(val))

                self.tbl_uncert_summary.setItem(r, c, item)

        self.tbl_uncert_summary.resizeColumnsToContents()
        self.tbl_uncert_summary.horizontalHeader().setStretchLastSection(True)
        self.tbl_uncert_summary.setSortingEnabled(True)


    def _draw_uncertainty_2d_map(self, grid_template, array_2d, title, clim=None, cmap="jet"):
        """
        Desenha um mapa 2D de resultado por coluna.
        """
        import numpy as np
        import pyvista as pv

        if not hasattr(self, "uncert_plotter_2d"):
            return

        plotter = self.uncert_plotter_2d
        plotter.clear()

        try:
            plotter.remove_scalar_bar()
        except Exception:
            pass

        if grid_template is None or array_2d is None:
            plotter.render()
            return

        dims = self._infer_grid_cell_dims(grid_template)
        if not dims:
            plotter.render()
            return

        nx_, ny_, nz_ = dims

        arr2d = np.asarray(array_2d, dtype=float)

        if arr2d.shape != (nx_, ny_):
            try:
                arr2d = arr2d.reshape((nx_, ny_), order="F")
            except Exception:
                plotter.render()
                return

        x_min, x_max, y_min, y_max, _, z_max = grid_template.bounds

        xs = np.linspace(x_min, x_max, nx_ + 1)
        ys = np.linspace(y_max, y_min, ny_ + 1)
        xs, ys = np.meshgrid(xs, ys, indexing="ij")
        zs = np.full_like(xs, z_max, dtype=float)

        surf = pv.StructuredGrid(xs, ys, zs)

        scalar_name = "mapcalc_2d"
        surf.cell_data[scalar_name] = arr2d.ravel(order="F")

        finite = arr2d[np.isfinite(arr2d)]

        if clim is None:
            if finite.size:
                vmin = float(np.nanmin(finite))
                vmax = float(np.nanmax(finite))
                if vmin >= 0.0:
                    vmin = 0.0
                if vmax <= vmin:
                    vmax = vmin + 1e-6
                clim = (vmin, vmax)
            else:
                clim = (0.0, 1.0)

        plotter.add_mesh(
            surf,
            scalars=scalar_name,
            cmap=cmap or "jet",
            show_edges=True,
            edge_color="black",
            line_width=0.5,
            nan_color="white",
            show_scalar_bar=False,
            clim=clim,
        )

        plotter.view_xy()
        plotter.enable_parallel_projection()
        plotter.enable_image_style()
        plotter.set_background("white")
        plotter.add_axes()

        plotter.show_bounds(
            grid="front",
            location="outer",
            ticks="outside",
            color="gray",
            minor_ticks=True,
            n_xlabels=4,
            n_ylabels=4,
            font_size=8,
            fmt="%.0f",
            xtitle="X",
            ytitle="Y",
        )

        plotter.add_scalar_bar(
            title=title,
            n_labels=5,
            fmt="%.3g",
            title_font_size=14,
            label_font_size=12,
        )

        try:
            plotter.add_text(title, position="upper_left", font_size=10, color="black")
        except Exception:
            pass

        plotter.render()


    def _render_uncertainty_3d(self, vis_grid, scalar_name, result_map, title, clim):
        """
        Renderiza resultado célula a célula em 3D.
        """
        from visualize import run
        import numpy as np

        if vis_grid is None or result_map is None:
            return

        vis_grid.cell_data[scalar_name] = np.asarray(result_map, dtype=float)

        uncert_state = {
            "mode": "scalar",
            "current_scalar_name": scalar_name,
            "current_scalar_title": title,
            "current_scalar_clim": clim,
            "current_scalar_cmap": "jet",
            "z_exag": float(self.state.get("z_exag", 1.0)),
            "show_scalar_bar": True,
        }

        self.uncert_plotter.clear()

        _, final_state = run(
            mode="scalar",
            z_exag=uncert_state["z_exag"],
            show_scalar_bar=True,
            external_plotter=self.uncert_plotter,
            external_state=uncert_state,
            target_grid=vis_grid,
            target_facies=None,
        )

        self.uncert_view_state = final_state

        try:
            if hasattr(self.uncert_plotter, "scalar_bars"):
                for k in list(self.uncert_plotter.scalar_bars.keys()):
                    self.uncert_plotter.remove_scalar_bar(k)
        except Exception:
            pass

        mapper = final_state.get("main_actor").mapper if final_state.get("main_actor") else None

        if mapper:
            mapper.SetScalarRange(clim)
            self.uncert_plotter.add_scalar_bar(
                title=title,
                mapper=mapper,
                fmt="%.3g",
                title_font_size=14,
                label_font_size=12,
            )

        self.uncert_plotter.reset_camera()

    def _get_mapcalc_model_grid_facies(self, model_key):
        """
        Retorna grid, facies e nome para um modelo.

        Importante:
        Não usar 'or' com arrays NumPy, porque isso gera:
        ValueError: truth value of an array is ambiguous.
        """
        from load_data import grid as global_grid, facies as global_facies

        if model_key is None:
            model_key = "base"

        model_key = str(model_key)

        if model_key == "base":
            m = self.models.get("base", {})

            g = m.get("grid", None)
            if g is None:
                g = self.state.get("current_grid_source", None)
            if g is None:
                g = global_grid

            f = m.get("facies", None)
            if f is None:
                f = self.state.get("current_facies", None)
            if f is None:
                f = global_facies

            name = m.get("name", "Modelo Base")
            if not name:
                name = "Modelo Base"

            return g, f, name

        m = self.models.get(model_key, {})

        g = m.get("grid", None)
        f = m.get("facies", None)

        name = m.get("name", model_key)
        if not name:
            name = model_key

        return g, f, name


    def _refresh_mapcalc_model_combo(self):
        """
        Atualiza o combo de modelo único usado nos cálculos de um modelo.
        """
        if not hasattr(self, "cmb_mapcalc_model"):
            return

        current = self.cmb_mapcalc_model.currentData()

        self.cmb_mapcalc_model.blockSignals(True)
        try:
            self.cmb_mapcalc_model.clear()

            # Base
            if "base" in self.models:
                self.cmb_mapcalc_model.addItem("Modelo Base", "base")

            # Demais modelos carregados
            for k, v in self.models.items():
                if k == "base":
                    continue

                name = v.get("name", None)

                # ignora placeholders vazios
                if not name and k == "compare":
                    continue

                if not name:
                    name = str(k)

                # precisa ter grid ou facies para ser útil
                if v.get("grid", None) is None and v.get("facies", None) is None:
                    continue

                self.cmb_mapcalc_model.addItem(str(name), str(k))

            if current is not None:
                idx = self.cmb_mapcalc_model.findData(current)
                if idx >= 0:
                    self.cmb_mapcalc_model.setCurrentIndex(idx)

        finally:
            self.cmb_mapcalc_model.blockSignals(False)


    def _refresh_mapcalc_property_combo(self):
        """
        Atualiza o combo principal de mapa/propriedade/descritor conforme a categoria.
        """
        if not hasattr(self, "cmb_uncert_prop"):
            return

        cat = self.cmb_mapcalc_category.currentData() if hasattr(self, "cmb_mapcalc_category") else "ensemble"
        current_text = self.cmb_uncert_prop.currentText()

        self.cmb_uncert_prop.blockSignals(True)
        try:
            self.cmb_uncert_prop.clear()

            # ------------------------------------------------------------
            # 1. Mapa vertical por coluna
            # ------------------------------------------------------------
            if cat == "vertical":
                self.cmb_uncert_prop.addItem(
                    "Espessura total da coluna",
                    {
                        "kind": "vertical_metric",
                        "label": "Espessura total da coluna",
                        "scalar": "__total_column_thickness__",
                        "title": "Espessura total da coluna (m)",
                        "reduction": "max",
                    }
                )

                try:
                    presets = get_vertical_metric_presets(prefix="vert_", include_filtered=False)
                    for label, pair in presets.items():
                        scalar, title = pair
                        self.cmb_uncert_prop.addItem(
                            label,
                            {
                                "kind": "vertical_metric",
                                "label": label,
                                "scalar": scalar,
                                "title": title,
                                "reduction": "max",
                            }
                        )
                except Exception:
                    pass

            # ------------------------------------------------------------
            # 2. Propriedade ponderada / equivalente
            # ------------------------------------------------------------
            elif cat == "property":
                try:
                    props = self._get_union_grid_property_names()
                except Exception:
                    props = []

                for p in props:
                    s = str(p)
                    if s in ("Facies", "Reservoir", "Clusters", "LargestCluster"):
                        continue
                    if s.endswith("_index"):
                        continue
                    if s.startswith("vert_"):
                        continue
                    if "Ghost" in s:
                        continue

                    self.cmb_uncert_prop.addItem(
                        s,
                        {
                            "kind": "scalar",
                            "scalar": s,
                            "reduction": "weighted_mean",
                        }
                    )

            # ------------------------------------------------------------
            # 3. Estatística do ensemble / incerteza-discordância
            # ------------------------------------------------------------
            else:
                self.cmb_uncert_prop.addItem(
                    "Fácies (Entropia)",
                    {"kind": "facies", "scalar": None, "reduction": "entropy"}
                )

                try:
                    props = self._get_union_grid_property_names()
                except Exception:
                    props = []

                for p in props:
                    s = str(p)
                    if s in ("Facies", "Reservoir", "Clusters", "LargestCluster"):
                        continue
                    if s.endswith("_index"):
                        continue
                    if "Ghost" in s:
                        continue

                    self.cmb_uncert_prop.addItem(
                        f"Propriedade: {s}",
                        {"kind": "scalar", "scalar": s, "reduction": "weighted_mean"}
                    )

                try:
                    presets = get_vertical_metric_presets(prefix="vert_", include_filtered=False)
                    for label, pair in presets.items():
                        scalar, title = pair
                        self.cmb_uncert_prop.addItem(
                            f"Métrica vertical: {label}",
                            {"kind": "scalar", "scalar": scalar, "reduction": "max"}
                        )
                except Exception:
                    pass

            # tenta restaurar seleção anterior
            for i in range(self.cmb_uncert_prop.count()):
                if self.cmb_uncert_prop.itemText(i) == current_text:
                    self.cmb_uncert_prop.setCurrentIndex(i)
                    break

        finally:
            self.cmb_uncert_prop.blockSignals(False)

        self._apply_mapcalc_category_ui()


    def _refresh_uncert_property_combo(self):
        """
        Wrapper para compatibilidade com show_uncertainty_view().
        Agora a página é Cálculo de Mapas.
        """
        self._refresh_mapcalc_property_combo()

    def run_map_calculation(self):
        """Executa o cálculo conforme o modo selecionado no Ribbon de Mapas."""
        mode = getattr(self, "mapcalc_mode", "vertical")
        if mode == "vertical":
            self._run_mapcalc_vertical_single_model()
            return
        if mode == "property":
            self._run_mapcalc_property_single_model()
            return
        if mode == "ensemble":
            self._run_mapcalc_ensemble_vertical_metric()
            return
        if mode == "uncertainty":
            self._sync_legacy_uncert_controls_from_mapcalc(mode="uncertainty")
            self.calculate_uncertainty()
            return
        if mode == "difference":
            self.statusBar().showMessage("Diferença entre modelos será implementada depois.", 5000)
            return

    def _run_mapcalc_ensemble_vertical_metric(self):
        """Calcula estatística do ensemble para uma métrica vertical por coluna."""
        from analysis import compute_vertical_metrics_for_grid, reduce_grid_scalar_to_column_map
        import numpy as np
        import pandas as pd

        keys = self._get_checked_mapcalc_model_keys()
        if not keys:
            keys = [getattr(self, "mapcalc_model_key", "base")]

        metric = getattr(self, "mapcalc_selected_metric", None)
        if not isinstance(metric, dict):
            self.statusBar().showMessage("Selecione uma métrica vertical para o ensemble.", 5000)
            return

        scalar_name = metric.get("scalar")
        title = metric.get("title", scalar_name)
        label = metric.get("label", scalar_name)
        stat = getattr(self, "mapcalc_selected_stat", "mean")
        scope = getattr(self, "mapcalc_selected_scope", "column")

        maps = []
        names = []
        grids = []
        rf = set(self.state.get("reservoir_facies", set()) or [])

        for key in keys:
            g, f, name = self._get_mapcalc_model_grid_facies(key)
            if g is None:
                continue
            if scalar_name != "__total_column_thickness__":
                if f is None:
                    continue
                try:
                    compute_vertical_metrics_for_grid(
                        g, f, rf, prefix="vert_",
                        thin_lamination_threshold=0.30,
                        include_filtered=True,
                    )
                except Exception as e:
                    print(f"Falha ao calcular métrica vertical em {name}: {e}")
                    continue

            m2d = reduce_grid_scalar_to_column_map(g, scalar_name, reduction="max", clip_to_01=False)
            if m2d is None:
                continue
            maps.append(np.asarray(m2d, dtype=float))
            names.append(name)
            grids.append(g)

        if not maps:
            self.statusBar().showMessage("Nenhum mapa válido foi gerado para o ensemble.", 5000)
            return

        stack = np.stack([m.reshape(-1, order="F") for m in maps], axis=0)
        if stat == "mean":
            out_flat = np.nanmean(stack, axis=0)
            stat_label = "Média"
        elif stat == "std":
            out_flat = np.nanstd(stack, axis=0)
            stat_label = "Desvio padrão"
        elif stat == "var":
            out_flat = np.nanvar(stack, axis=0)
            stat_label = "Variância"
        else:
            out_flat = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
            stat_label = "Amplitude"

        result_2d = out_flat.reshape(maps[0].shape, order="F")

        if scope == "model":
            rows = []
            for name, m in zip(names, maps):
                vals = np.asarray(m, dtype=float)
                finite = vals[np.isfinite(vals)]
                if finite.size == 0:
                    continue
                rows.append({
                    "modelo": name,
                    "valor_medio_espacial": float(np.nanmean(finite)),
                    "desvio_espacial": float(np.nanstd(finite)),
                    "min_espacial": float(np.nanmin(finite)),
                    "max_espacial": float(np.nanmax(finite)),
                })
            df = pd.DataFrame(rows)
            if not df.empty:
                vals = df["valor_medio_espacial"].to_numpy(dtype=float)
                ens_mean = float(np.nanmean(vals))
                ens_std = float(np.nanstd(vals))
                ens_var = float(np.nanvar(vals))
                ens_range = float(np.nanmax(vals) - np.nanmin(vals))
                df["desvio_abs_da_media_ensemble"] = np.abs(df["valor_medio_espacial"] - ens_mean)
            else:
                ens_mean = ens_std = ens_var = ens_range = 0.0
            self._set_uncert_result_mode("model")
            self._fill_uncert_summary_table(df)
            if hasattr(self, "txt_uncert_summary"):
                self.txt_uncert_summary.setPlainText(
                    f"{stat_label} do ensemble\n"
                    f"Métrica: {label}\n"
                    f"Modelos: {len(names)}\n\n"
                    f"Média ensemble: {ens_mean:.6g}\n"
                    f"Desvio padrão: {ens_std:.6g}\n"
                    f"Variância: {ens_var:.6g}\n"
                    f"Amplitude: {ens_range:.6g}"
                )
            self.lbl_uncert_max_real.setText(f"{stat_label}: tabela")
            return

        finite = result_2d[np.isfinite(result_2d)]
        if finite.size:
            vmax = float(np.nanmax(finite))
            vmin = float(np.nanmin(finite))
            if vmin >= 0:
                vmin = 0.0
            if vmax <= vmin:
                vmax = vmin + 1e-6
            clim = (vmin, vmax)
        else:
            clim = (0.0, 1.0)

        full_title = f"{stat_label} ensemble: {title}"
        self._set_uncert_result_mode("column")
        self._draw_uncertainty_2d_map(grids[0], result_2d, full_title, clim=clim, cmap=self.state.get("thickness_cmap", "jet"))
        self.lbl_uncert_n.setText(f"Entrada: {len(names)} modelo(s)")
        self.lbl_uncert_max_theo.setText(f"Métrica: {label}")
        self.lbl_uncert_max_real.setText(f"Máx.: {float(np.nanmax(finite)):.6g}" if finite.size else "Resultado: -")
        self._uncert_has_result = True

    def _sync_legacy_uncert_controls_from_mapcalc(self, mode="ensemble"):
        """
        Prepara os widgets antigos usados por calculate_uncertainty().
        """
        # Garante lista de modelos a partir do painel lateral
        keys = self._get_checked_mapcalc_model_keys() if hasattr(self, "lst_mapcalc_models") else []
        if not keys:
            keys = [getattr(self, "mapcalc_model_key", "base")]
        self._set_uncert_models_from_keys(keys)

        # Cria combos antigos se não existirem
        if not hasattr(self, "cmb_uncert_scope"):
            self.cmb_uncert_scope = QtWidgets.QComboBox()
            self.cmb_uncert_scope.addItem("Célula a célula", "cell")
            self.cmb_uncert_scope.addItem("Por coluna", "column")
            self.cmb_uncert_scope.addItem("Resumo por modelo / ensemble", "model")
            self.cmb_uncert_scope.hide()

        if not hasattr(self, "cmb_uncert_metric"):
            self.cmb_uncert_metric = QtWidgets.QComboBox()
            self.cmb_uncert_metric.addItem("Média", "mean")
            self.cmb_uncert_metric.addItem("Desvio padrão", "std")
            self.cmb_uncert_metric.addItem("Variância", "var")
            self.cmb_uncert_metric.addItem("Amplitude", "range")
            self.cmb_uncert_metric.hide()

        if not hasattr(self, "cmb_uncert_prop"):
            self.cmb_uncert_prop = QtWidgets.QComboBox()
            self.cmb_uncert_prop.hide()

        # Escala
        scope = getattr(self, "mapcalc_selected_scope", "column")
        idx = self.cmb_uncert_scope.findData(scope)
        if idx >= 0:
            self.cmb_uncert_scope.setCurrentIndex(idx)

        # Estatística
        stat = getattr(self, "mapcalc_selected_stat", "std")
        if mode == "uncertainty" and stat == "mean":
            stat = "std"

        idx = self.cmb_uncert_metric.findData(stat)
        if idx >= 0:
            self.cmb_uncert_metric.setCurrentIndex(idx)

        # Propriedade/alvo
        self.cmb_uncert_prop.clear()

        if mode == "uncertainty":
            target = getattr(self, "mapcalc_uncertainty_target", "facies_entropy")

            if target == "facies_entropy":
                self.cmb_uncert_prop.addItem(
                    "Fácies (Entropia)",
                    {"kind": "facies", "scalar": None, "reduction": "entropy"}
                )
            else:
                prop = getattr(self, "mapcalc_selected_property", None)
                if not prop:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Cálculo de Mapas",
                        "Selecione uma propriedade no modo Propriedade antes de calcular incerteza de propriedade."
                    )
                    return

                self.cmb_uncert_prop.addItem(
                    f"Propriedade: {prop}",
                    {"kind": "scalar", "scalar": prop, "reduction": "weighted_mean"}
                )

        else:
            metric = getattr(self, "mapcalc_selected_metric", None)

            if metric:
                self.cmb_uncert_prop.addItem(
                    f"Métrica vertical: {metric.get('label', metric.get('scalar'))}",
                    {"kind": "scalar", "scalar": metric.get("scalar"), "reduction": "max"}
                )
            else:
                self.cmb_uncert_prop.addItem(
                    "Espessura total da coluna",
                    {"kind": "scalar", "scalar": "__total_column_thickness__", "reduction": "max"}
                )

        self.cmb_uncert_prop.setCurrentIndex(0)

    def _run_mapcalc_vertical_single_model(self):
        """
        Calcula/visualiza um mapa vertical por coluna para um único modelo.
        """
        from analysis import (
            compute_vertical_metrics_for_grid,
            reduce_grid_scalar_to_column_map,
            expand_column_map_to_cell_data,
        )
        import numpy as np

        model_key = getattr(self, "mapcalc_model_key", "base")
        grid, facies_arr, model_name = self._get_mapcalc_model_grid_facies(model_key)

        if grid is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Cálculo de Mapas",
                "Nenhum grid foi encontrado para o modelo selecionado."
            )
            return

        metric = getattr(self, "mapcalc_selected_metric", None)
        if not isinstance(metric, dict):
            metric = {
                "scalar": "__total_column_thickness__",
                "title": "Espessura total da coluna (m)",
                "label": "Espessura",
            }
            self.mapcalc_selected_metric = metric

        scalar_name = metric.get("scalar")
        label = metric.get("label", scalar_name)
        title = metric.get("title", label)

        if scalar_name != "__total_column_thickness__":
            rf = set(self.state.get("reservoir_facies", set()) or [])
            try:
                compute_vertical_metrics_for_grid(
                    grid,
                    facies_arr,
                    rf,
                    prefix="vert_",
                    thin_lamination_threshold=0.30,
                    include_filtered=True,
                )
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Cálculo de Mapas",
                    f"Falha ao recalcular métricas verticais:\n{e}"
                )
                return

        result_2d = reduce_grid_scalar_to_column_map(
            grid,
            scalar_name,
            reduction="max",
            clip_to_01=False,
        )

        if result_2d is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Cálculo de Mapas",
                f"Não foi possível gerar o mapa '{label}'."
            )
            return

        # Salva também como cell_data replicado, útil para visualização 3D futura
        try:
            expand_column_map_to_cell_data(grid, result_2d, scalar_name)
            extra = set(self.state.get("extra_sync_cell_data", set()) or set())
            extra.add(str(scalar_name))
            self.state["extra_sync_cell_data"] = extra
        except Exception:
            pass

        finite = np.asarray(result_2d, dtype=float)
        finite = finite[np.isfinite(finite)]

        vmax = float(np.nanmax(finite)) if finite.size else 1.0
        clim = (0.0, vmax if vmax > 0 else 1.0)

        full_title = f"{title} | {model_name}"

        self._set_uncert_result_mode("column")
        self._draw_uncertainty_2d_map(
            grid,
            result_2d,
            full_title,
            clim=clim,
            cmap=self.state.get("thickness_cmap", "jet"),
        )

        if hasattr(self, "lbl_uncert_n"):
            self.lbl_uncert_n.setText(f"Entrada: {model_name}")
            self.lbl_uncert_max_theo.setText(f"Mapa: {label}")
            self.lbl_uncert_max_real.setText(f"Máx.: {vmax:.6g}")

        self._uncert_has_result = True


    def _run_mapcalc_property_single_model(self):
        """
        Calcula/visualiza mapa por coluna a partir de uma propriedade por célula.
        """
        from analysis import reduce_grid_scalar_to_column_map, expand_column_map_to_cell_data
        import numpy as np
        import re

        model_key = getattr(self, "mapcalc_model_key", "base")
        grid, facies_arr, model_name = self._get_mapcalc_model_grid_facies(model_key)

        if grid is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Cálculo de Mapas",
                "Nenhum grid foi encontrado para o modelo selecionado."
            )
            return

        scalar_name = getattr(self, "mapcalc_selected_property", None)
        if not scalar_name:
            QtWidgets.QMessageBox.warning(
                self,
                "Cálculo de Mapas",
                "Selecione uma propriedade na lista."
            )
            return

        if scalar_name not in grid.cell_data:
            QtWidgets.QMessageBox.warning(
                self,
                "Cálculo de Mapas",
                f"A propriedade selecionada não existe no grid do modelo '{model_name}'.\n\n"
                f"Propriedade: {scalar_name}"
            )
            return

        operation = getattr(self, "mapcalc_selected_operation", "weighted_mean")

        op_title = {
            "weighted_mean": "Média ponderada por espessura",
            "equivalent": "Espessura equivalente",
            "mean": "Média aritmética vertical",
            "sum": "Soma vertical",
            "max": "Máximo vertical",
        }.get(operation, "Mapa derivado")

        prefix = {
            "weighted_mean": "wmean_th",
            "equivalent": "eq_th",
            "mean": "vmean",
            "sum": "vsum",
            "max": "vmax",
        }.get(operation, "derived")

        safe = re.sub(r"[^0-9a-zA-Z_]+", "_", str(scalar_name)).strip("_")
        out_name = f"{prefix}_{safe}"

        try:
            clip_to_01 = self._is_normalized_property(scalar_name)
        except Exception:
            clip_to_01 = False

        result_2d = reduce_grid_scalar_to_column_map(
            grid,
            scalar_name,
            reduction=operation,
            clip_to_01=clip_to_01,
        )

        if result_2d is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Cálculo de Mapas",
                f"Não foi possível calcular '{op_title}' para '{scalar_name}'."
            )
            return

        try:
            expand_column_map_to_cell_data(grid, result_2d, out_name)
            extra = set(self.state.get("extra_sync_cell_data", set()) or set())
            extra.add(out_name)
            self.state["extra_sync_cell_data"] = extra
        except Exception:
            pass

        finite = np.asarray(result_2d, dtype=float)
        finite = finite[np.isfinite(finite)]

        if operation == "weighted_mean" and clip_to_01:
            clim = (0.0, 1.0)
            vmax = 1.0
        elif finite.size:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))

            if vmin >= 0.0:
                vmin = 0.0

            if vmax <= vmin:
                vmax = vmin + 1e-6

            clim = (vmin, vmax)
        else:
            vmax = 1.0
            clim = (0.0, 1.0)

        full_title = f"{op_title}: {scalar_name} | {model_name}"

        self._set_uncert_result_mode("column")
        self._draw_uncertainty_2d_map(
            grid,
            result_2d,
            full_title,
            clim=clim,
            cmap=self.state.get("thickness_cmap", "jet"),
        )

        if hasattr(self, "lbl_uncert_n"):
            self.lbl_uncert_n.setText(f"Entrada: {model_name}")
            self.lbl_uncert_max_theo.setText(f"Mapa: {label}")
            self.lbl_uncert_max_real.setText(f"Máx.: {vmax:.6g}")

        self._uncert_has_result = True

    def _set_uncert_result_mode(self, scope):
        """Escolhe a página central e mantém o painel direito da aba Mapas visível."""
        scope = str(scope or "cell").lower()

        if not hasattr(self, "uncert_result_stack"):
            return

        if scope == "model":
            self.uncert_result_stack.setCurrentIndex(2)
        elif scope == "column":
            self.uncert_result_stack.setCurrentIndex(1)
        else:
            self.uncert_result_stack.setCurrentIndex(0)
            try:
                if hasattr(self, "mapcalc_right_tabs") and hasattr(self, "mapcalc_geometry_page"):
                    self.mapcalc_right_tabs.setCurrentWidget(self.mapcalc_geometry_page)
            except Exception:
                pass

        # Diferente da antiga aba Incerteza, a aba Mapas precisa do painel direito
        # para fácies-alvo mesmo quando o resultado central é 2D.
        if hasattr(self, "uncert_right_panel"):
            self.uncert_right_panel.setVisible(True)
        if hasattr(self, "mapcalc_right_scroll"):
            self.mapcalc_right_scroll.setVisible(True)


    def _clear_uncert_summary_table(self):
        """Limpa a tabela central de resumo da aba Incerteza."""
        if hasattr(self, "tbl_uncert_summary"):
            self.tbl_uncert_summary.clear()
            self.tbl_uncert_summary.setRowCount(0)
            self.tbl_uncert_summary.setColumnCount(0)

        if hasattr(self, "txt_uncert_summary"):
            self.txt_uncert_summary.clear()


    def _fill_uncert_summary_table(self, df):
        """
        Preenche a tabela central do resumo por modelo / ensemble.
        """
        if not hasattr(self, "tbl_uncert_summary"):
            return

        self.tbl_uncert_summary.setSortingEnabled(False)
        self.tbl_uncert_summary.clear()

        if df is None or df.empty:
            self.tbl_uncert_summary.setRowCount(0)
            self.tbl_uncert_summary.setColumnCount(0)
            self.tbl_uncert_summary.setSortingEnabled(True)
            return

        self.tbl_uncert_summary.setRowCount(len(df))
        self.tbl_uncert_summary.setColumnCount(len(df.columns))
        self.tbl_uncert_summary.setHorizontalHeaderLabels([str(c) for c in df.columns])

        for r in range(len(df)):
            for c, col in enumerate(df.columns):
                val = df.iloc[r, c]

                if isinstance(val, (float, np.floating)):
                    txt = f"{float(val):.6g}"
                else:
                    txt = str(val)

                item = QtWidgets.QTableWidgetItem(txt)

                # Permite ordenação numérica em colunas numéricas
                if isinstance(val, (int, float, np.integer, np.floating)):
                    item.setData(QtCore.Qt.UserRole, float(val))

                self.tbl_uncert_summary.setItem(r, c, item)

        self.tbl_uncert_summary.resizeColumnsToContents()
        self.tbl_uncert_summary.horizontalHeader().setStretchLastSection(True)
        self.tbl_uncert_summary.setSortingEnabled(True)


    def _draw_uncertainty_2d_map(self, grid_template, array_2d, title, clim=None, cmap="jet"):
        """
        Desenha um mapa 2D de incerteza por coluna diretamente a partir de um array (nx, ny).
        """
        import numpy as np
        import pyvista as pv

        if not hasattr(self, "uncert_plotter_2d"):
            return

        plotter = self.uncert_plotter_2d
        plotter.clear()

        try:
            plotter.remove_scalar_bar()
        except Exception:
            pass

        if grid_template is None or array_2d is None:
            plotter.render()
            return

        dims = self._infer_grid_cell_dims(grid_template)
        if not dims:
            plotter.render()
            return

        nx_, ny_, nz_ = dims

        arr2d = np.asarray(array_2d, dtype=float)
        if arr2d.shape != (nx_, ny_):
            try:
                arr2d = arr2d.reshape((nx_, ny_), order="F")
            except Exception:
                plotter.render()
                return

        x_min, x_max, y_min, y_max, _, z_max = grid_template.bounds

        xs = np.linspace(x_min, x_max, nx_ + 1)
        ys = np.linspace(y_max, y_min, ny_ + 1)
        xs, ys = np.meshgrid(xs, ys, indexing="ij")
        zs = np.full_like(xs, z_max, dtype=float)

        surf = pv.StructuredGrid(xs, ys, zs)

        scalar_name = "uncertainty_2d"
        surf.cell_data[scalar_name] = arr2d.ravel(order="F")

        flat = arr2d[np.isfinite(arr2d)]

        if clim is None:
            if flat.size:
                vmin = float(np.nanmin(flat))
                vmax = float(np.nanmax(flat))
                if vmin >= 0.0:
                    vmin = 0.0
                if vmax <= vmin:
                    vmax = vmin + 1e-6
                clim = (vmin, vmax)
            else:
                clim = (0.0, 1.0)

        plotter.add_mesh(
            surf,
            scalars=scalar_name,
            cmap=cmap or "jet",
            show_edges=True,
            edge_color="black",
            line_width=0.5,
            nan_color="white",
            show_scalar_bar=False,
            clim=clim,
        )

        plotter.view_xy()
        plotter.enable_parallel_projection()
        plotter.enable_image_style()
        plotter.set_background("white")
        plotter.add_axes()

        plotter.show_bounds(
            grid="front",
            location="outer",
            ticks="outside",
            color="gray",
            minor_ticks=True,
            n_xlabels=4,
            n_ylabels=4,
            font_size=8,
            fmt="%.0f",
            xtitle="X",
            ytitle="Y",
        )

        plotter.add_scalar_bar(
            title=title,
            n_labels=5,
            fmt="%.3g",
            title_font_size=14,
            label_font_size=12,
        )

        try:
            plotter.add_text(title, position="upper_left", font_size=10, color="black")
        except Exception:
            pass

        plotter.render()


    def _render_uncertainty_3d(self, vis_grid, scalar_name, result_map, title, clim):
        """
        Renderiza a incerteza célula a célula em 3D.
        """
        from visualize import run
        import numpy as np

        if vis_grid is None or result_map is None:
            return

        vis_grid.cell_data[scalar_name] = np.asarray(result_map, dtype=float)

        uncert_state = {
            "mode": "scalar",
            "current_scalar_name": scalar_name,
            "current_scalar_title": title,
            "current_scalar_clim": clim,
            "current_scalar_cmap": "jet",
            "z_exag": float(self.state.get("z_exag", 1.0)),
            "show_scalar_bar": True,
        }

        self.uncert_plotter.clear()

        _, final_state = run(
            mode="scalar",
            z_exag=uncert_state["z_exag"],
            show_scalar_bar=True,
            external_plotter=self.uncert_plotter,
            external_state=uncert_state,
            target_grid=vis_grid,
            target_facies=None,
        )

        self.uncert_view_state = final_state

        try:
            if hasattr(self.uncert_plotter, "scalar_bars"):
                for k in list(self.uncert_plotter.scalar_bars.keys()):
                    self.uncert_plotter.remove_scalar_bar(k)
        except Exception:
            pass

        mapper = final_state.get("main_actor").mapper if final_state.get("main_actor") else None

        if mapper:
            mapper.SetScalarRange(clim)
            self.uncert_plotter.add_scalar_bar(
                title=title,
                mapper=mapper,
                fmt="%.3g",
                title_font_size=14,
                label_font_size=12,
            )

        self.uncert_plotter.reset_camera()


    def on_uncert_slice_changed(self, axis, mode, value):
        """Aplica cortes da aba de Incerteza."""
        # Atualiza a UI do slicer
        self.uncert_slicer.external_update(axis, mode, value)
        
        if not hasattr(self, "uncert_view_state") or not self.uncert_view_state:
            return

        state = self.uncert_view_state
        
        if axis == "z" and mode == "scale":
            state["z_exag"] = float(value)
        elif "set_slice" in state:
            state["set_slice"](axis, mode, value)
            
        if "refresh" in state: state["refresh"]()
        if hasattr(self, "uncert_plotter"): self.uncert_plotter.render()

    def _uncert_sel_all(self):
        for i in range(self.lst_uncert_models.count()):
            self.lst_uncert_models.item(i).setCheckState(QtCore.Qt.Checked)

    def _uncert_sel_none(self):
        for i in range(self.lst_uncert_models.count()):
            self.lst_uncert_models.item(i).setCheckState(QtCore.Qt.Unchecked)

    def show_uncertainty_view(self):
        """Ativa a página de Cálculo de Mapas."""
        
        # 1. FORÇA BRUTA: Fecha os painéis laterais imediatamente
        if hasattr(self, "dock_explorer"): 
            self.dock_explorer.close()
        if hasattr(self, "dock_props"): 
            self.dock_props.close()

        # 2. Troca o estado lógico
        self.switch_perspective("uncertainty")
        
        # 3. Garante que o stack central mostre a aba de Incerteza (Índice 2)
        if hasattr(self, "central_stack"):
            self.central_stack.setCurrentIndex(2) 
        
        # 4. Atualiza visualmente o botão da Ribbon
        if hasattr(self, "act_view_uncert"):
            self.act_view_uncert.setChecked(True)

        # 5. Garante que a interface interna esteja montada
        if hasattr(self, "uncertainty_page") and not getattr(self, "_uncert_tab_ready", False):
            self.setup_uncertainty_tab(self.uncertainty_page)
            self._uncert_tab_ready = True

        # 6. Atualiza listas da página Cálculo de Mapas
        self._refresh_uncert_model_list()
        self._refresh_mapcalc_model_combo()
        self._refresh_uncert_property_combo()
        self._apply_mapcalc_category_ui()

        # Atualiza a página de Cálculo de Mapas
        try:
            self._refresh_mapcalc_models_panel()
        except Exception:
            pass

        try:
            self._refresh_mapcalc_property_list()
        except Exception:
            pass
        try:
            self._refresh_mapcalc_target_facies_table()
        except Exception:
            pass

        try:
            self._update_mapcalc_target_facies_label()
        except Exception:
            pass

        try:
            self._update_mapcalc_model_status_label()
        except Exception:
            pass
    
    def _refresh_uncert_property_combo(self):
        """
        Atualiza lista de propriedades disponíveis para cálculo de incerteza.
        """
        if not hasattr(self, "cmb_uncert_prop"):
            return

        current_text = self.cmb_uncert_prop.currentText()

        self.cmb_uncert_prop.blockSignals(True)
        try:
            self.cmb_uncert_prop.clear()

            self.cmb_uncert_prop.addItem(
                "Fácies (Entropia)",
                {"kind": "facies", "scalar": None, "reduction": "entropy"}
            )

            # Propriedades escalares carregadas do grid
            try:
                props = self._get_union_grid_property_names()
            except Exception:
                props = []

            for p in props:
                self.cmb_uncert_prop.addItem(
                    f"Propriedade: {p}",
                    {"kind": "scalar", "scalar": p, "reduction": "weighted_mean"}
                )

            # Métricas verticais já calculadas
            try:
                presets = get_vertical_metric_presets(prefix="vert_", include_filtered=False)
                for label, (scalar, title) in presets.items():
                    self.cmb_uncert_prop.addItem(
                        f"Métrica vertical: {label}",
                        {"kind": "scalar", "scalar": scalar, "reduction": "max"}
                    )
            except Exception:
                pass

            # tenta restaurar seleção anterior
            for i in range(self.cmb_uncert_prop.count()):
                if self.cmb_uncert_prop.itemText(i) == current_text:
                    self.cmb_uncert_prop.setCurrentIndex(i)
                    break

        finally:
            self.cmb_uncert_prop.blockSignals(False)
        
    def _refresh_uncert_model_list(self):
        """
        Atualiza a lista interna de modelos usada pelos cálculos antigos.

        Na UI nova de Cálculo de Mapas, a seleção visível de modelos fica em
        lst_mapcalc_models. Mesmo assim, calculate_uncertainty() ainda espera
        lst_uncert_models. Portanto esta lista é criada/atualizada como widget
        oculto de compatibilidade.
        """
        if not hasattr(self, "lst_uncert_models"):
            self.lst_uncert_models = QtWidgets.QListWidget()
            self.lst_uncert_models.hide()

        self.lst_uncert_models.clear()

        # Se a nova lista lateral de modelos existir, usa a seleção dela como fonte.
        checked_from_panel = set()
        if hasattr(self, "lst_mapcalc_models"):
            try:
                for i in range(self.lst_mapcalc_models.count()):
                    src = self.lst_mapcalc_models.item(i)
                    key = src.data(QtCore.Qt.UserRole)
                    if key is not None and src.checkState() == QtCore.Qt.Checked:
                        checked_from_panel.add(str(key))
            except Exception:
                checked_from_panel = set()

        for k, v in self.models.items():
            if k == "compare":
                continue

            # inclui base; para os demais, exige grid ou fácies para evitar placeholders vazios
            if k != "base" and v.get("grid", None) is None and v.get("facies", None) is None:
                continue

            name = "Modelo Base" if k == "base" else v.get("name", str(k))
            if not name:
                continue

            it = QtWidgets.QListWidgetItem(str(name))
            it.setData(QtCore.Qt.UserRole, str(k))
            it.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)

            if checked_from_panel:
                checked = str(k) in checked_from_panel
            else:
                checked = True

            it.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
            self.lst_uncert_models.addItem(it)

    def calculate_uncertainty(self):
        """
        Calcula incerteza do ensemble em três escalas:

        - cell:
            estatística célula a célula entre modelos.

        - column:
            reduz cada modelo para mapa 2D por coluna e calcula estatística entre modelos.

        - model:
            resume cada modelo em um valor médio espacial e calcula estatísticas do ensemble.
        """
        from analysis import (
            compute_facies_entropy_map,
            compute_continuous_uncertainty_map,
            compute_column_ensemble_stat_map,
            expand_column_map_to_cell_data,
            compute_model_level_ensemble_summary,
            compute_vertical_metrics_for_grid,
        )
        from visualize import run
        from load_data import grid as global_grid
        import numpy as np
        import pandas as pd

        # ------------------------------------------------------------
        # 1. Modelos selecionados
        # ------------------------------------------------------------
        selected_keys = []
        if hasattr(self, "lst_uncert_models"):
            for i in range(self.lst_uncert_models.count()):
                it = self.lst_uncert_models.item(i)
                if it.checkState() == QtCore.Qt.Checked:
                    selected_keys.append(it.data(QtCore.Qt.UserRole))

        if not selected_keys:
            QtWidgets.QMessageBox.warning(self, "Aviso", "Selecione modelos na lista.")
            return

        grids = []
        facies_arrays = []
        model_names = []

        for k in selected_keys:
            if k not in self.models:
                continue

            m = self.models[k]
            g = m.get("grid")
            if g is None and k == "base":
                g = self.state.get("current_grid_source") or global_grid

            f = m.get("facies")

            if g is None:
                continue

            grids.append(g)
            facies_arrays.append(f)
            model_names.append(m.get("name", "Modelo Base" if k == "base" else str(k)))

        if not grids:
            return

        # ------------------------------------------------------------
        # 2. Configuração escolhida na interface
        # ------------------------------------------------------------
        data = self.cmb_uncert_prop.currentData() if hasattr(self, "cmb_uncert_prop") else None
        if not isinstance(data, dict):
            data = {"kind": "facies", "scalar": None, "reduction": "entropy"}

        kind = data.get("kind", "facies")
        scalar_name = data.get("scalar", None)
        reduction = data.get("reduction", "weighted_mean")

        scope = self.cmb_uncert_scope.currentData() if hasattr(self, "cmb_uncert_scope") else "cell"
        metric = self.cmb_uncert_metric.currentData() if hasattr(self, "cmb_uncert_metric") else "std"

        category = self.cmb_mapcalc_category.currentData() if hasattr(self, "cmb_mapcalc_category") else "ensemble"

        # Em "Incerteza / discordância", a média não deve ser usada como métrica de incerteza.
        if category == "uncertainty" and metric == "mean":
            metric = "std"
            if hasattr(self, "cmb_uncert_metric"):
                idx_std = self.cmb_uncert_metric.findData("std")
                if idx_std >= 0:
                    self.cmb_uncert_metric.blockSignals(True)
                    self.cmb_uncert_metric.setCurrentIndex(idx_std)
                    self.cmb_uncert_metric.blockSignals(False)

        # Fácies só faz sentido como entropia neste fluxo
        if kind == "facies":
            scope = "cell"
            metric = "entropy"
        
        self._clear_uncert_summary_table()
        self._set_uncert_result_mode(scope)

        # Métricas verticais são descritores por coluna.
        # Não devem ser analisadas como campo célula a célula.
        if scalar_name is not None and str(scalar_name).startswith("vert_") and scope == "cell":
            scope = "column"

            if hasattr(self, "cmb_uncert_scope"):
                idx_col = self.cmb_uncert_scope.findData("column")
                if idx_col >= 0:
                    self.cmb_uncert_scope.blockSignals(True)
                    self.cmb_uncert_scope.setCurrentIndex(idx_col)
                    self.cmb_uncert_scope.blockSignals(False)

        # ------------------------------------------------------------
        # 3. Grid de visualização
        # ------------------------------------------------------------
        grid_template = self.models.get("base", {}).get("grid") or self.state.get("current_grid_source") or global_grid
        vis_grid = grid_template.copy(deep=True)

        scalar_out = "Uncertainty"
        title = "Incerteza"
        result_map = None
        result_2d = None
        clim = None

        # ------------------------------------------------------------
        # 4A. Incerteza categórica: entropia de fácies célula a célula
        # ------------------------------------------------------------
        if kind == "facies":
            arrays = [a for a in facies_arrays if a is not None]
            if not arrays:
                return

            n_models = len(arrays)
            result_map = compute_facies_entropy_map(arrays, target_grid=vis_grid)

            max_real = float(np.nanmax(result_map)) if result_map.size else 0.0
            max_theo = float(np.log(n_models)) if n_models > 1 else 0.0

            title = f"Entropia de fácies (N={n_models})"

            use_abs = self.chk_abs_scale.isChecked() if hasattr(self, "chk_abs_scale") else False
            if use_abs:
                clim = (0.0, max_theo if max_theo > 0 else 0.1)
            else:
                clim = (0.0, max_real if max_real > 0 else 0.1)

            if hasattr(self, "lbl_uncert_n"):
                self.lbl_uncert_n.setText(f"Modelos (N): {n_models}")
                self.lbl_uncert_max_theo.setText(f"Máx. Teórico: {max_theo:.3f}")
                self.lbl_uncert_max_real.setText(f"Máx. Encontrado: {max_real:.3f}")

        # ------------------------------------------------------------
        # 4B. Propriedade contínua / descritor
        # ------------------------------------------------------------
        else:
            if not scalar_name:
                return

            # Se for métrica vertical, garante que foi recalculada em todos os modelos
            if str(scalar_name).startswith("vert_"):
                rf_global = set(self.state.get("reservoir_facies", set()) or [])
                for g, f in zip(grids, facies_arrays):
                    if g is None or f is None:
                        continue
                    try:
                        compute_vertical_metrics_for_grid(
                            g,
                            f,
                            rf_global,
                            prefix="vert_",
                            thin_lamination_threshold=0.30,
                            include_filtered=True,
                        )
                    except Exception:
                        pass

            clip_to_01 = self._is_normalized_property(scalar_name)

            # ------------------------
            # Célula a célula
            # ------------------------
            if scope == "cell":
                arrays = []
                valid_names = []

                for g, name in zip(grids, model_names):
                    if scalar_name not in getattr(g, "cell_data", {}):
                        continue

                    arr = np.asarray(g.cell_data[scalar_name], dtype=float)
                    if arr.size != g.n_cells:
                        continue

                    if clip_to_01:
                        arr = np.clip(arr, 0.0, 1.0)

                    arrays.append(arr)
                    valid_names.append(name)

                if not arrays:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Incerteza",
                        f"A propriedade '{scalar_name}' não foi encontrada nos modelos selecionados."
                    )
                    return

                result_map = compute_continuous_uncertainty_map(
                    arrays,
                    target_grid=vis_grid,
                    metric=metric,
                )

                stat_label = {
                    "mean": "Média",
                    "std": "Desvio padrão",
                    "var": "Variância",
                    "range": "Amplitude",
                }.get(metric, metric)

                title = f"{stat_label} célula a célula: {scalar_name}"

                finite = result_map[np.isfinite(result_map)]
                vmax = float(np.nanmax(finite)) if finite.size else 1.0
                if metric == "mean" and clip_to_01:
                    clim = (0.0, 1.0)
                else:
                    clim = (0.0, vmax if vmax > 0 else 1.0)

                if hasattr(self, "lbl_uncert_n"):
                    self.lbl_uncert_n.setText(f"Modelos (N): {len(arrays)}")
                    self.lbl_uncert_max_theo.setText("Máx. Teórico: -")
                    self.lbl_uncert_max_real.setText(f"Máx. Encontrado: {vmax:.4g}")

            # ------------------------
            # Por coluna
            # ------------------------
            elif scope == "column":
                # Para propriedades escalares, usa média ponderada por espessura.
                # Para métricas verticais, usa max porque elas já são constantes por coluna.
                red = "max" if str(scalar_name).startswith("vert_") else reduction

                result_2d = compute_column_ensemble_stat_map(
                    grids,
                    scalar_name,
                    metric=metric,
                    reduction=red,
                    clip_to_01=clip_to_01,
                )

                if result_2d is None:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Incerteza",
                        f"Não foi possível calcular incerteza por coluna para '{scalar_name}'."
                    )
                    return

                result_2d = np.asarray(result_2d, dtype=float)

                stat_label = {
                    "mean": "Média",
                    "std": "Desvio padrão",
                    "var": "Variância",
                    "range": "Amplitude",
                }.get(metric, metric)

                title = f"{stat_label} por coluna: {scalar_name}"

                finite = result_2d[np.isfinite(result_2d)]
                vmax = float(np.nanmax(finite)) if finite.size else 1.0

                if metric == "mean" and clip_to_01:
                    clim = (0.0, 1.0)
                else:
                    clim = (0.0, vmax if vmax > 0 else 1.0)

                if hasattr(self, "lbl_uncert_n"):
                    self.lbl_uncert_n.setText(f"Modelos (N): {len(grids)}")
                    self.lbl_uncert_max_theo.setText("Máx. Teórico: -")
                    self.lbl_uncert_max_real.setText(f"Máx. Encontrado: {vmax:.4g}")

            # ------------------------
            # Resumo por modelo / ensemble
            # ------------------------
            else:
                red = "max" if str(scalar_name).startswith("vert_") else reduction

                summary = compute_model_level_ensemble_summary(
                    grids,
                    model_names,
                    scalar_name,
                    reduction=red,
                    clip_to_01=clip_to_01,
                )

                df = summary["per_model"]
                ens = summary["ensemble"]

                self._set_uncert_result_mode("model")
                self._fill_uncert_summary_table(df)

                n_df = len(df) if df is not None else 0

                if hasattr(self, "lbl_uncert_n"):
                    self.lbl_uncert_n.setText(f"Modelos (N): {n_df}")
                    self.lbl_uncert_max_theo.setText(f"Média ensemble: {ens['mean']:.6g}")
                    self.lbl_uncert_max_real.setText(
                        f"std={ens['std']:.6g} | var={ens['var']:.6g} | amp={ens['range']:.6g}"
                    )

                if hasattr(self, "txt_uncert_summary"):
                    stat_label = {
                        "mean": "Média",
                        "std": "Desvio padrão",
                        "var": "Variância",
                        "range": "Amplitude",
                    }.get(metric, str(metric))

                    self.txt_uncert_summary.setPlainText(
                        f"Resumo por modelo / ensemble\n"
                        f"Propriedade/descritor: {scalar_name}\n"
                        f"Redução espacial por modelo: média espacial do mapa reduzido por coluna\n"
                        f"Estatística selecionada: {stat_label}\n\n"
                        f"Média ensemble: {ens['mean']:.6g}\n"
                        f"Desvio padrão: {ens['std']:.6g}\n"
                        f"Variância: {ens['var']:.6g}\n"
                        f"Amplitude: {ens['range']:.6g}"
                    )

                self._uncert_has_result = True
                return

        # ------------------------------------------------------------
        # 5. Renderização
        # ------------------------------------------------------------

        # Incerteza por coluna: resultado é mapa 2D, não volume 3D.
        if scope == "column":
            if result_2d is None:
                return

            self._set_uncert_result_mode("column")
            self._draw_uncertainty_2d_map(
                grid_template,
                result_2d,
                title,
                clim=clim,
                cmap="jet",
            )

            self._uncert_has_result = True
            return

        # Incerteza célula a célula: resultado é volume 3D.
        if result_map is None:
            return

        self._set_uncert_result_mode("cell")
        self._render_uncertainty_3d(
            vis_grid,
            scalar_out,
            result_map,
            title,
            clim,
        )

        self._uncert_has_result = True

    def _refresh_mapcalc_property_list(self):
        if not hasattr(self, "lst_mapcalc_properties"):
            return

        text = ""
        if hasattr(self, "txt_mapcalc_prop_filter"):
            text = (self.txt_mapcalc_prop_filter.text() or "").strip().lower()

        try:
            props = self._get_union_grid_property_names()
        except Exception:
            props = []

        self.lst_mapcalc_properties.blockSignals(True)
        try:
            self.lst_mapcalc_properties.clear()

            for p in sorted([str(x) for x in props], key=str.lower):
                if text and text not in p.lower():
                    continue

                if p in ("Facies", "facies", "Reservoir", "reservoir", "Clusters", "LargestCluster"):
                    continue
                if p in ("vtkOriginalCellIds", "vtkOriginalPointIds", "Texture Coordinates"):
                    continue
                if p.endswith("_index"):
                    continue
                if p.startswith("vert_"):
                    continue
                if "Ghost" in p:
                    continue

                self.lst_mapcalc_properties.addItem(p)

            if self.lst_mapcalc_properties.count() > 0:
                self.lst_mapcalc_properties.setCurrentRow(0)
                self.mapcalc_selected_property = self.lst_mapcalc_properties.item(0).text()

        finally:
            self.lst_mapcalc_properties.blockSignals(False)


    def _on_mapcalc_property_selected(self):
        if not hasattr(self, "lst_mapcalc_properties"):
            return

        item = self.lst_mapcalc_properties.currentItem()
        if item is None:
            self.mapcalc_selected_property = None
            self._update_mapcalc_property_info_box()
            return

        self.mapcalc_selected_property = str(item.text())
        self._update_mapcalc_property_info_box()
        self._schedule_mapcalc_auto_update()

    def open_mapcalc_target_facies_dialog(self):
        """Mostra a aba dedicada de fácies-alvo da página de Cálculo de Mapas."""
        try:
            self.show_mapcalc_page("vertical")
            if hasattr(self, "mapcalc_right_tabs") and hasattr(self, "mapcalc_facies_page"):
                self.mapcalc_right_tabs.setCurrentWidget(self.mapcalc_facies_page)
            if hasattr(self, "tbl_mapcalc_target_facies"):
                self.tbl_mapcalc_target_facies.setFocus()
        except Exception:
            pass
        self._update_mapcalc_target_facies_label()

    def _update_mapcalc_target_facies_label(self):
        selected = sorted([int(x) for x in self.state.get("reservoir_facies", set()) or set()])
        if not selected:
            txt = "Fácies selecionadas: nenhuma"
        elif len(selected) <= 8:
            txt = "Fácies selecionadas: " + ", ".join(map(str, selected))
        else:
            txt = f"Fácies selecionadas: {len(selected)} fácies"
        if hasattr(self, "lbl_mapcalc_target_facies"):
            self.lbl_mapcalc_target_facies.setText(txt)

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

        import numpy as np
        from PyQt5 import QtWidgets, QtCore

        z_exag = float(self.state.get("z_exag", 1.0))

        # ---------- quais poços estão marcados ----------
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

        # ---------- cache ----------
        state_key = (tuple(sorted(checked)), z_exag)
        if getattr(self, "_wells_draw_state", None) == state_key:
            # garante que a caixa esteja travada corretamente (pode ter sido recriada em outro refresh)
            self._lock_axes_bounds_to_grid(self.state)
            return
        self._wells_draw_state = state_key

        if not hasattr(self, "_well_actors"):
            self._well_actors = {}

        def _exclude_from_bounds(actor):
            """Impede que este ator influencie o cálculo de bounds do renderer/cube-axes."""
            try:
                if actor is None:
                    return
                if hasattr(actor, "SetUseBounds"):
                    actor.SetUseBounds(False)
            except Exception:
                pass

        
        try:
            self.plotter.disable_render()
        except Exception:
            pass

        try:
            # ---------- remove atores de poços desmarcados ----------
            for wn in list(self._well_actors.keys()):
                if wn not in checked:
                    actors = self._well_actors.pop(wn)
                    if not isinstance(actors, list):
                        actors = [actors]
                    for a in actors:
                        try:
                            self.plotter.remove_actor(a)
                        except Exception:
                            pass

            markers_db = getattr(self, "markers_db", {})

            # ---------- desenha poços marcados ----------
            for well_name in sorted(checked):
                if well_name in self._well_actors:
                    continue

                well = self.wells.get(well_name)
                if well is None:
                    continue

                well_actors = []

                try:
                    mesh = well.get_vtk_polydata(z_exag)
                    if mesh is None or getattr(mesh, "n_points", 0) == 0:
                        continue

                    top_point = np.array(mesh.points[0], dtype=float)

                    actor_line = self.plotter.add_mesh(
                        mesh,
                        color="saddlebrown",
                        line_width=3,
                        name=f"well_{well_name}",
                        reset_camera=False,
                        show_scalar_bar=False
                    )
                    _exclude_from_bounds(actor_line)
                    well_actors.append(actor_line)

                    actor_name = self.plotter.add_point_labels(
                        [top_point],
                        [well_name],
                        font_size=10,
                        text_color="black",
                        shape="rounded_rect",
                        shape_color="white",
                        shape_opacity=0.25,
                        show_points=False,
                        reset_camera=False,
                        always_visible=True,
                        name=f"name_{well_name}"
                    )
                    _exclude_from_bounds(actor_name)
                    well_actors.append(actor_name)

                    # Marcadores (1 mesh + 1 labels por poço, leve)
                    m_list = markers_db.get(well_name, [])
                    if m_list:
                        m_mesh, m_labels = well.get_markers_mesh(m_list, z_exag)
                        if m_mesh is not None and getattr(m_mesh, "n_points", 0) > 0:
                            actor_markers = self.plotter.add_mesh(
                                m_mesh,
                                color="red",
                                render_points_as_spheres=True,
                                point_size=8,
                                reset_camera=False,
                                name=f"markers_{well_name}"
                            )
                            _exclude_from_bounds(actor_markers)
                            well_actors.append(actor_markers)

                            enriched = []
                            for i, base_txt in enumerate(m_labels):
                                depth = None
                                try:
                                    item = m_list[i] if i < len(m_list) else None
                                    if isinstance(item, dict):
                                        for k in ("md", "MD", "tvd", "TVD", "depth", "Depth"):
                                            if k in item:
                                                depth = float(item[k])
                                                break
                                    elif isinstance(item, (tuple, list)) and len(item) >= 2:
                                        depth = float(item[1])
                                except Exception:
                                    depth = None

                                if depth is not None and np.isfinite(depth):
                                    enriched.append(f"{base_txt} ({depth:.0f} m)")
                                else:
                                    enriched.append(str(base_txt))

                            actor_labels = self.plotter.add_point_labels(
                                m_mesh.points,
                                enriched,
                                font_size=10,
                                point_color="red",
                                text_color="black",
                                show_points=False,
                                reset_camera=False,
                                shape_opacity=0.15,
                                always_visible=True,
                                name=f"mlabels_{well_name}"
                            )
                            _exclude_from_bounds(actor_labels)
                            well_actors.append(actor_labels)

                    self._well_actors[well_name] = well_actors

                except Exception as e:
                    print(f"[WARN] Falha ao desenhar poço {well_name}: {e}")

            
            self._lock_axes_bounds_to_grid(self.state)

        finally:
            try:
                self.plotter.enable_render()
            except Exception:
                pass

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


    def _column_profile_from_grid(self, grid, x, y, *, i0=None, j0=None, return_ij=False, facies_override=None):
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

        # facies (pode ser sobrescrita via facies_override)
        fac = None
        if facies_override is not None:
            try:
                fac = np.asarray(facies_override).astype(int)
                if fac.size != g.n_cells:
                    fac = None
            except Exception:
                fac = None
        if fac is None:
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

        from load_data import grid as base_grid, facies as base_facies_global
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
        for cand in ("fac", "FACIES", "FAC", "lito_upscaled", "LITO_UPSCALED", "fac_dion", "FAC_DION", "lito", "LITO"):
            if cand in well.data.columns:
                col_real = cand
                break

        full_real = (
            well.data[col_real].to_numpy(dtype=float)
            if col_real is not None
            else np.zeros_like(full_depth, dtype=float)
        )

        real_depth0 = full_depth
        real_facies0 = full_real
        well_logs_report = well.data.copy()

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
                        well_logs_report = well.data.loc[mask_r].copy()

        # --- BASE e SIM por coluna (i,j) ---
        xy = self._pick_reference_xy_for_well_report(well, markers)
        if xy is None:
            QtWidgets.QMessageBox.warning(self, "Aviso", "Não consegui obter (X,Y) do poço para comparação.")
            return

        xref, yref = xy        # Base (sempre 1x1 no local) - usa facies RAW do BASE para não ser contaminado pela facies do modelo ativo
        base_facies_raw = None
        try:
            if base_grid is not None and hasattr(base_grid, "cell_data"):
                fb = base_grid.cell_data.get("FaciesRaw", None)
                if fb is not None:
                    fb = np.asarray(fb).astype(int)
                    if fb.size == getattr(base_grid, "n_cells", fb.size):
                        base_facies_raw = fb
        except Exception:
            base_facies_raw = None
        if base_facies_raw is None:
            try:
                base_facies_raw = np.asarray(base_facies_global).astype(int)
            except Exception:
                base_facies_raw = None

        base_depth, base_facies, _ = self._column_profile_from_grid(
            base_grid, xref, yref, facies_override=base_facies_raw
        )
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
            best_depth=best_depth, best_fac=best_facies,
            window_size_str=f"{w_size}x{w_size}",
            well_logs_df=well_logs_report,
            real_log_col=col_real,
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
        Ribbon reorganizado em:
        - Home
        - Visualizar
        - Mapas
        - Relatórios

        Mantém o estilo atual: ícone + texto abaixo, grupos com moldura
        e uso de standard icons do Qt.
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
            f = lbl.font()
            f.setBold(False)
            lbl.setFont(f)
            lbl.setStyleSheet("color: rgba(0,0,0,160);")
            v.addWidget(lbl)

            return frame, h

        def make_tab():
            w = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(w)
            lay.setContentsMargins(8, 6, 8, 6)
            lay.setSpacing(10)
            return w, lay

        def mk_tbtn(act):
            b = QtWidgets.QToolButton()
            b.setDefaultAction(act)
            b.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
            b.setIconSize(QtCore.QSize(28, 28))
            b.setAutoRaise(True)
            return b

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

        # =========================================================
        # AÇÕES DE NAVEGAÇÃO DE PÁGINAS
        # =========================================================
        ico3d = self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon)
        ico2d = self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogContentsView)
        icomet = self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogInfoView)
        icorank = self.style().standardIcon(QtWidgets.QStyle.SP_ArrowUp)
        ico_calc = self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView)

        self.act_view_3d = QtWidgets.QAction(ico3d, "3D", self)
        self.act_view_3d.setCheckable(True)

        self.act_view_2d = QtWidgets.QAction(ico2d, "Mapas 2D", self)
        self.act_view_2d.setCheckable(True)

        self.act_view_metrics = QtWidgets.QAction(icomet, "Métricas", self)
        self.act_view_metrics.setCheckable(True)

        self.act_view_ranking = QtWidgets.QAction(icorank, "Ranking", self)
        self.act_view_ranking.setCheckable(True)

        # Mantive o nome interno act_view_uncert para não quebrar o resto do código,
        # mas visualmente agora ele representa "Cálculo de Mapas".
        self.act_view_uncert = QtWidgets.QAction(ico_calc, "Cálculo\nde Mapas", self)
        self.act_view_uncert.setCheckable(True)

        grp_views = QtWidgets.QActionGroup(self)
        grp_views.setExclusive(True)
        grp_views.addAction(self.act_view_3d)
        grp_views.addAction(self.act_view_2d)
        grp_views.addAction(self.act_view_metrics)
        grp_views.addAction(self.act_view_ranking)
        grp_views.addAction(self.act_view_uncert)
        self.act_view_3d.setChecked(True)

        self.act_view_3d.triggered.connect(self.show_main_3d_view)
        self.act_view_2d.triggered.connect(self.show_map2d_view)
        self.act_view_metrics.triggered.connect(self.show_metrics_view)
        self.act_view_ranking.triggered.connect(self.show_ranking_view)
        self.act_view_uncert.triggered.connect(self.show_uncertainty_view)

        # =========================================================
        # ABA HOME
        # =========================================================
        tab_home, home_lay = make_tab()

        # Dados
        g_dados, g_dados_row = make_group("Dados")
        btn_modelo = make_tool_btn(
            "Modelo",
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton)
        )
        btn_modelo.clicked.connect(self.open_compare_dialog)

        btn_pocos = make_tool_btn(
            "Poços",
            self.style().standardIcon(QtWidgets.QStyle.SP_DirIcon)
        )
        btn_pocos.clicked.connect(self.load_well_dialog)

        g_dados_row.addWidget(btn_modelo)
        g_dados_row.addWidget(btn_pocos)

        # Perspectiva
        g_persp, g_persp_row = make_group("Perspectiva")
        self.act_persp_viz.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon))
        self.act_persp_comp.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView))

        btn_viz = QtWidgets.QToolButton()
        btn_viz.setDefaultAction(self.act_persp_viz)
        btn_viz.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        btn_viz.setIconSize(QtCore.QSize(28, 28))
        btn_viz.setAutoRaise(True)

        btn_comp = QtWidgets.QToolButton()
        btn_comp.setDefaultAction(self.act_persp_comp)
        btn_comp.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        btn_comp.setIconSize(QtCore.QSize(28, 28))
        btn_comp.setAutoRaise(True)

        g_persp_row.addWidget(btn_viz)
        g_persp_row.addWidget(btn_comp)

        # Saída
        g_saida, g_saida_row = make_group("Saída")
        btn_snap = make_tool_btn(
            "Snapshot",
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton)
        )
        btn_snap.clicked.connect(self.take_snapshot)
        g_saida_row.addWidget(btn_snap)

        home_lay.addWidget(g_dados)
        home_lay.addWidget(g_persp)
        home_lay.addWidget(g_saida)
        home_lay.addStretch(1)

        self.ribbon_tabs.addTab(tab_home, "Home")

        # =========================================================
        # ABA VISUALIZAR
        # =========================================================
        tab_view, view_lay = make_tab()

        # Navegação
        g_nav, g_nav_row = make_group("Navegação")
        g_nav_row.addWidget(mk_tbtn(self.act_view_3d))
        g_nav_row.addWidget(mk_tbtn(self.act_view_2d))
        g_nav_row.addWidget(mk_tbtn(self.act_view_metrics))
        # Se quiser mostrar ranking depois, descomente:
        # g_nav_row.addWidget(mk_tbtn(self.act_view_ranking))

        # Modo 3D
        g_modo, g_modo_row = make_group("Modo 3D")
        self.btn_mode = QtWidgets.QToolButton(self)
        self.btn_mode.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.btn_mode.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.btn_mode.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogListView))
        self.btn_mode.setIconSize(QtCore.QSize(28, 28))
        self.btn_mode.setAutoRaise(True)

        menu_mode = QtWidgets.QMenu(self.btn_mode)
        menu_mode.aboutToShow.connect(self.populate_mode_menu)

        # Lista inicial simplificada — sem métricas por coluna e sem incerteza
        modes = [
            ("Fácies (Global)", "facies"),
            ("Fácies-alvo", "reservoir"),
            ("Clusters (conectividade)", "clusters"),
            ("Maior Cluster", "largest"),
        ]
        for text, data in modes:
            action = menu_mode.addAction(text)
            action.triggered.connect(lambda ch, t=text, d=data: self._update_mode_btn(t, d))

        self.btn_mode.setMenu(menu_mode)
        self._update_mode_btn("Fácies", "facies")
        g_modo_row.addWidget(self.btn_mode)

        # Estilo
        g_style, g_style_row = make_group("Estilo")

        lbl_cmap = QtWidgets.QLabel("Paleta:")
        self.cmb_colormap = QtWidgets.QComboBox()
        self.cmb_colormap.setToolTip("Altera a escala de cores para propriedades contínuas.")
        self.cmb_colormap.setIconSize(QtCore.QSize(80, 14))
        self.cmb_colormap.setFixedWidth(92)

        self._init_colormap_combo(
            ["jet", "viridis", "magma", "cividis", "turbo", "plasma", "seismic", "coolwarm"],
            default_name="jet"
        )
        self.cmb_colormap.currentIndexChanged.connect(self._on_colormap_combo_changed)

        v_box_style = QtWidgets.QVBoxLayout()
        v_box_style.setSpacing(0)
        v_box_style.setContentsMargins(0, 0, 0, 0)
        v_box_style.addWidget(lbl_cmap)
        v_box_style.addWidget(self.cmb_colormap)
        g_style_row.addLayout(v_box_style)

        # Inspeção
        g_insp, g_insp_row = make_group("Inspeção")

        self.cmb_debug_win = QtWidgets.QComboBox()
        self.cmb_debug_win.addItems(["1x1", "3x3", "5x5", "7x7", "9x9"])
        self.cmb_debug_win.setCurrentIndex(1)  # 3x3
        self.cmb_debug_win.setToolTip("Tamanho da janela de busca")
        self.cmb_debug_win.setFixedWidth(60)

        self.btn_debug_all = make_tool_btn(
            "Destacar\nTodos",
            self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogContentsView),
            checkable=True
        )
        self.btn_debug_all.clicked.connect(self.toggle_global_well_debug)
        self.cmb_debug_win.currentIndexChanged.connect(self._on_global_window_size_changed)

        v_box_combo = QtWidgets.QVBoxLayout()
        v_box_combo.setSpacing(0)
        v_box_combo.setContentsMargins(0, 0, 0, 0)
        v_box_combo.addWidget(QtWidgets.QLabel("Janela:"))
        v_box_combo.addWidget(self.cmb_debug_win)

        g_insp_row.addLayout(v_box_combo)
        g_insp_row.addWidget(self.btn_debug_all)

        try:
            sep_sel = QtWidgets.QFrame()
            sep_sel.setFrameShape(QtWidgets.QFrame.VLine)
            sep_sel.setFrameShadow(QtWidgets.QFrame.Sunken)
            sep_sel.setFixedWidth(1)
            g_insp_row.addWidget(sep_sel)
        except Exception:
            pass

        self.btn_pick_cell = make_tool_btn(
            "Selecionar\nCélula",
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogYesButton),
            checkable=True,
        )
        self.btn_pick_cell.setToolTip("Ativa modo de seleção de célula no 3D.")

        self.btn_pick_column = make_tool_btn(
            "Selecionar\nColuna",
            self.style().standardIcon(QtWidgets.QStyle.SP_ArrowUp),
            checkable=True,
        )
        self.btn_pick_column.setToolTip("Ativa modo de seleção de coluna no 3D.")

        self.btn_pick_clear = make_tool_btn(
            "Limpar\nSeleção",
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton),
            checkable=False,
        )
        self.btn_pick_clear.setToolTip("Remove o destaque e limpa o inspector.")

        self.btn_pick_cell.clicked.connect(lambda checked: self.set_pick_mode("cell" if checked else None))
        self.btn_pick_column.clicked.connect(lambda checked: self.set_pick_mode("column" if checked else None))
        self.btn_pick_clear.clicked.connect(self.clear_pick_selection)

        g_insp_row.addWidget(self.btn_pick_cell)
        g_insp_row.addWidget(self.btn_pick_column)
        g_insp_row.addWidget(self.btn_pick_clear)

        # Painéis
        g_panels, g_panels_row = make_group("Painéis")
        self.btn_toggle_explorer = make_tool_btn(
            "Explorer",
            self.style().standardIcon(QtWidgets.QStyle.SP_DirHomeIcon),
            checkable=True
        )
        self.btn_toggle_props = make_tool_btn(
            "Inspector",
            self.style().standardIcon(QtWidgets.QStyle.SP_DesktopIcon),
            checkable=True
        )
        self.btn_toggle_explorer.setEnabled(False)
        self.btn_toggle_props.setEnabled(False)

        g_panels_row.addWidget(self.btn_toggle_explorer)
        g_panels_row.addWidget(self.btn_toggle_props)

        view_lay.addWidget(g_nav)
        view_lay.addWidget(g_modo)
        view_lay.addWidget(g_style)
        view_lay.addWidget(g_insp)
        view_lay.addWidget(g_panels)
        view_lay.addStretch(1)

        self.ribbon_tabs.addTab(tab_view, "Visualizar")

        # =========================================================
        # ABA MAPAS
        # =========================================================
        tab_maps, maps_lay = make_tab()

        # ---------------------------------------------------------
        # Grupo: Tipo de Mapa
        # ---------------------------------------------------------
        g_map_type, g_map_type_row = make_group("Tipo de Mapa")

        def make_map_type_btn(text, icon, mode_key, tooltip=""):
            btn = make_tool_btn(text, icon, checkable=True)
            btn.setToolTip(tooltip)
            btn.clicked.connect(lambda checked=False, m=mode_key: self.show_mapcalc_page(m))
            return btn

        self.btn_mapcalc_vertical = make_map_type_btn(
            "Mapa\nVertical",
            self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView),
            "vertical",
            "Espessura, proporção, ICV, Qv, pacotes, gaps, trocas e permanências."
        )

        self.btn_mapcalc_property = make_map_type_btn(
            "Propriedade",
            self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogContentsView),
            "property",
            "Média ponderada, espessura equivalente, soma e média vertical."
        )

        self.btn_mapcalc_ensemble = make_map_type_btn(
            "Ensemble",
            self.style().standardIcon(QtWidgets.QStyle.SP_DirIcon),
            "ensemble",
            "Média, desvio padrão, variância e amplitude entre modelos."
        )

        self.btn_mapcalc_uncert = make_map_type_btn(
            "Incerteza",
            self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxWarning),
            "uncertainty",
            "Entropia, discordância e dispersão entre cenários."
        )

        self.btn_mapcalc_diff = make_map_type_btn(
            "Diferença",
            self.style().standardIcon(QtWidgets.QStyle.SP_ArrowRight),
            "difference",
            "Diferença entre modelo-base e simulações."
        )

        self.mapcalc_type_buttons = QtWidgets.QButtonGroup(self)
        self.mapcalc_type_buttons.setExclusive(True)
        for b in [
            self.btn_mapcalc_vertical,
            self.btn_mapcalc_property,
            self.btn_mapcalc_ensemble,
            self.btn_mapcalc_uncert,
            self.btn_mapcalc_diff,
        ]:
            self.mapcalc_type_buttons.addButton(b)
            g_map_type_row.addWidget(b)

        # ---------------------------------------------------------
        # Grupo: Saída
        # ---------------------------------------------------------
        g_map_out, g_map_out_row = make_group("Saída")

        btn_map_snapshot = make_tool_btn(
            "Snapshot",
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton)
        )
        btn_map_snapshot.clicked.connect(self.take_snapshot)
        g_map_out_row.addWidget(btn_map_snapshot)

        # ---------------------------------------------------------
        # Grupo: Filtros
        # ---------------------------------------------------------
        g_map_filters, g_map_filters_row = make_group("Filtros")

        btn_target_facies = make_tool_btn(
            "Fácies\nAlvo",
            self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogListView)
        )
        btn_target_facies.clicked.connect(self.open_mapcalc_target_facies_dialog)

        self.btn_thickness_filter = make_tool_btn(
            "Filtrar\nFinas",
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton),
            checkable=True,
        )
        self.btn_thickness_filter.setToolTip("Ignora laminações finas nas métricas verticais por coluna.")
        self.btn_thickness_filter.toggled.connect(self._on_toggle_thickness_filtered)
        self.state.setdefault("thickness_use_filtered", False)

        g_map_filters_row.addWidget(btn_target_facies)
        g_map_filters_row.addWidget(self.btn_thickness_filter)

        maps_lay.addWidget(g_map_type)
        maps_lay.addWidget(g_map_out)
        maps_lay.addWidget(g_map_filters)
        maps_lay.addStretch(1)

        self.ribbon_tabs.addTab(tab_maps, "Mapas")

        # =========================================================
        # ABA RELATÓRIOS
        # =========================================================
        tab_reports, rep_lay = make_tab()

        g_rep, g_rep_row = make_group("Relatórios")

        btn_rep_open = make_tool_btn(
            "Abrir\nRelatório",
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton)
        )
        btn_rep_open.clicked.connect(self.open_reports_dialog)

        btn_rep_selected = make_tool_btn(
            "Poços\nSelecionados",
            self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogContentsView)
        )
        btn_rep_selected.clicked.connect(self.open_selected_well_reports)

        g_rep_row.addWidget(btn_rep_open)
        g_rep_row.addWidget(btn_rep_selected)

        g_rep_out, g_rep_out_row = make_group("Saída")
        btn_snap_rep = make_tool_btn(
            "Snapshot",
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton)
        )
        btn_snap_rep.clicked.connect(self.take_snapshot)
        g_rep_out_row.addWidget(btn_snap_rep)

        rep_lay.addWidget(g_rep)
        rep_lay.addWidget(g_rep_out)
        rep_lay.addStretch(1)

        self.ribbon_tabs.addTab(tab_reports, "Relatórios")

        try:
            self.ribbon_tabs.currentChanged.connect(self._on_ribbon_tab_changed)
        except Exception:
            pass


    def _make_cmap_icon(self, cmap_name, w=160, h=28):
        """Cria um ícone (amostra) da paleta para usar no ComboBox."""
        from PyQt5 import QtGui, QtCore
        import numpy as np
        import matplotlib.cm as cm

        try:
            cmap = cm.get_cmap(cmap_name)
        except Exception:
            cmap = cm.get_cmap("jet")

        grad = np.linspace(0, 1, w, dtype=float)
        rgba = (cmap(grad) * 255).astype(np.uint8)  # (w,4)
        img = np.repeat(rgba[np.newaxis, :, :], h, axis=0)  # (h,w,4)

        qimg = QtGui.QImage(img.data, w, h, QtGui.QImage.Format_RGBA8888)
        qimg = qimg.copy()  # garante ownership
        pix = QtGui.QPixmap.fromImage(qimg)
        return QtGui.QIcon(pix)

    def _init_colormap_combo(self, cmap_names, default_name="jet"):
        """Preenche o combo de paletas com amostras (ícones) em vez de texto."""
        from PyQt5 import QtCore

        if not hasattr(self, "cmb_colormap"):
            return

        self.cmb_colormap.blockSignals(True)
        try:
            self.cmb_colormap.clear()
            for name in cmap_names:
                icon = self._make_cmap_icon(name)
                self.cmb_colormap.addItem(icon, "", userData=name)
                idx = self.cmb_colormap.count() - 1
                # tooltip com o nome da paleta
                self.cmb_colormap.setItemData(idx, name, QtCore.Qt.ToolTipRole)

            # seleciona default
            for i in range(self.cmb_colormap.count()):
                if self.cmb_colormap.itemData(i) == default_name:
                    self.cmb_colormap.setCurrentIndex(i)
                    break
        finally:
            self.cmb_colormap.blockSignals(False)

    def _on_colormap_combo_changed(self, idx):
        """Handler do combo de paletas (usa itemData)."""
        if not hasattr(self, "cmb_colormap"):
            return
        cmap_name = self.cmb_colormap.itemData(idx)
        if isinstance(cmap_name, str) and cmap_name.strip():
            self.change_colormap_ui(cmap_name)

    def change_colormap_ui(self, cmap_name):
        """Troca a paleta para propriedades contínuas (Espessura, Porosidade, etc.) em 3D/2D e Comparação."""
        if not cmap_name:
            return

        # Estado global (usado pelo 3D principal e pelos estados de comparação)
        self.state["thickness_cmap"] = cmap_name
        self.state["current_scalar_cmap"] = cmap_name

        # Atualiza 3D principal
        refresh = self.state.get("refresh")
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass

        # Atualiza mapa 2D da visualização
        try:
            if hasattr(self, "update_2d_map"):
                self.update_2d_map()
        except Exception:
            pass

        # Atualiza estados 3D de comparação (se houver)
        if hasattr(self, "active_comp_states"):
            for st in self.active_comp_states:
                try:
                    st["thickness_cmap"] = cmap_name
                    st["current_scalar_cmap"] = cmap_name
                    if callable(st.get("refresh")):
                        st["refresh"]()
                except Exception:
                    pass

        # Se estiver vendo 2D na comparação, reconstrói para aplicar a paleta
        try:
            if hasattr(self, "compare_stack") and self.central_stack.currentIndex() == 1 and self.compare_stack.currentIndex() == 2:
                self.update_dynamic_comparison_2d(self.get_checked_models())
        except Exception:
            pass


    def _on_ribbon_tab_changed(self, index):
        """Mantém a aba do ribbon coerente com a página central."""
        try:
            name = self.ribbon_tabs.tabText(index)
        except Exception:
            return
        if name == "Mapas":
            self.show_mapcalc_page("vertical")
        elif name == "Visualizar":
            self.show_main_3d_view()

    def populate_mode_menu(self):
        """Reconstrói o menu de Modos/Propriedades baseado no grid atual."""
        menu = self.btn_mode.menu()
        menu.clear()

        modes_std = [
            ("Fácies (Global)", "facies"),
            ("Fácies-alvo", "reservoir"),
            ("Clusters (Conectividade)", "clusters"),
            ("Maior Cluster", "largest"),
        ]

        menu.addSection("Análise Estrutural")
        for text, data in modes_std:
            action = menu.addAction(text)
            action.triggered.connect(lambda ch, t=text, d=data: self._update_mode_btn(t, d))

        # Propriedades do Grid (união em comparação / grid ativo em visualização)
        grids = []
        if hasattr(self, "central_stack") and self.central_stack.currentIndex() == 1 and hasattr(self, "active_comp_states") and self.active_comp_states:
            for st in self.active_comp_states:
                g = st.get("current_grid_source")
                if g is not None:
                    grids.append(g)
        else:
            g = self.state.get("current_grid_source")
            if g is None and "base" in self.models:
                g = self.models["base"].get("grid")
            if g is not None:
                grids.append(g)

        all_cell_keys = set()
        for g in grids:
            try:
                all_cell_keys |= set(g.cell_data.keys())
            except Exception:
                pass

        if all_cell_keys:
            exact_ignore = {
                "vtkOriginalCellIds", "vtkOriginalPointIds",
                "Facies", "facies", "Entropy", "Texture Coordinates",
                "StratigraphicThickness", "cell_thickness",
                "Reservoir", "reservoir", "Clusters", "clusters",
                "LargestCluster", "Volume", "NTG_local"
            }

            found_any = False
            for name in sorted(all_cell_keys):
                if name in exact_ignore:
                    continue
                if str(name).endswith("_index"):
                    continue
                if str(name).startswith("vert_"):
                    continue
                if "Ghost" in str(name):
                    continue

                if not found_any:
                    menu.addSection("Propriedades do Grid")
                    found_any = True

                action = menu.addAction(f"{name}")
                action.triggered.connect(lambda ch, n=name: self.change_scalar_view(n))

        # Config
        menu.addSeparator()
        menu.addSection("Configurações")
        act_cfg = menu.addAction("Selecionar propriedades que são proporção (0–1)...")
        act_cfg.triggered.connect(self.open_proportion_props_dialog)
    
    def change_scalar_view(self, scalar_name):
        """Visualiza uma propriedade escalar arbitrária (PORO, PERM, Sand, Basement, etc)."""
        import numpy as np

        self.btn_mode.setText(f"Prop:\n{scalar_name}")

        grid = self.state.get("current_grid_source")
        if grid is None and "base" in self.models:
            grid = self.models["base"].get("grid")

        if grid is None or scalar_name not in getattr(grid, "cell_data", {}):
            return

        arr = np.asarray(grid.cell_data[scalar_name], dtype=float)
        finite = arr[np.isfinite(arr)]

        if self._is_normalized_property(scalar_name):
            clim = (0.0, 1.0)
        elif finite.size > 0:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
            if vmin >= 0.0:
                vmin = 0.0
            if vmax <= vmin:
                vmax = vmin + 1e-6
            clim = (vmin, vmax)
        else:
            clim = (0.0, 1.0)

        title = f"Propriedade: {scalar_name}"
        cmap_use = self.state.get(
            "current_scalar_cmap",
            self.state.get("thickness_cmap", "jet")
        )

        # Compatibilidade com o fluxo antigo do 2D/comparação
        presets = self.state.get("thickness_presets", {})
        presets[scalar_name] = (scalar_name, title)
        self.state["thickness_presets"] = presets
        self.state["thickness_mode"] = scalar_name
        self.state["thickness_clim"] = clim
        self.state["thickness_clim_manual"] = True
        self.state["thickness_global_clim"] = None

        # Fluxo correto para propriedade escalar genérica
        self.state["mode"] = "scalar"
        self.state["current_scalar_name"] = scalar_name
        self.state["current_scalar_title"] = title
        self.state["current_scalar_clim"] = clim
        self.state["current_scalar_cmap"] = cmap_use

        # Sincroniza estados ativos da comparação 3D
        for st in getattr(self, "active_comp_states", []) or []:
            try:
                st_presets = st.get("thickness_presets", {})
                st_presets[scalar_name] = (scalar_name, title)
                st["thickness_presets"] = st_presets
                st["thickness_mode"] = scalar_name
                st["thickness_clim"] = clim
                st["thickness_clim_manual"] = True
                st["thickness_global_clim"] = None

                st["mode"] = "scalar"
                st["current_scalar_name"] = scalar_name
                st["current_scalar_title"] = title
                st["current_scalar_clim"] = clim
                st["current_scalar_cmap"] = cmap_use
            except Exception:
                pass

        # Se estiver em comparação, padroniza a escala global entre os modelos
        try:
            if hasattr(self, "central_stack") and self.central_stack.currentIndex() == 1:
                self._apply_global_clim_to_active_comparison()
        except Exception:
            pass

        # Refresh 3D principal
        refresh = self.state.get("refresh")
        if callable(refresh):
            refresh()

        # Refresh 2D principal
        if hasattr(self, "update_2d_map"):
            self.update_2d_map()

        # Refresh 2D comparação
        try:
            if (
                hasattr(self, "compare_stack")
                and self.central_stack.currentIndex() == 1
                and self.compare_stack.currentIndex() == 2
            ):
                self.update_dynamic_comparison_2d(self.get_checked_models())
        except Exception:
            pass

    def _on_global_window_size_changed(self):
        """Callback quando o tamanho da janela global (Ribbon) é alterado."""
        # 1. Se estiver visualizando Ranking, recalcula a tabela
        if hasattr(self, "viz_container") and self.viz_container.currentIndex() == 3:
            self.update_ranking_view_content()
        
        # 2. Se a visualização de debug 3D estiver ativa, atualiza os atores
        if self.btn_debug_all.isChecked():
            self.toggle_global_well_debug()

    def _on_rank_params_changed(self):
        """Atualiza parâmetros do ranking (ex.: t_min) e agenda recálculo."""
        try:
            if hasattr(self, "spin_rank_tmin"):
                self.well_rank_t_min = float(self.spin_rank_tmin.value())
        except Exception:
            pass

        # Agenda atualização pesada (ranking/entropia/comparação)
        if hasattr(self, "_schedule_heavy_update"):
            self._schedule_heavy_update()
        else:
            # fallback
            try:
                self.update_ranking_view_content()
            except Exception:
                pass

    def update_ranking_view_content(self):
        """Recalcula o ranking considerando MODELOS e POÇOS marcados no Project Explorer.

        **Score por poço (proporção por espessura):**
        - Constrói runs (fácies, espessura) a partir do log (REAL e SIM).
        - Suaviza segmentos finos: segmentos com espessura < t_min são mesclados ao vizinho.
        - Calcula proporções p(f)=T_f/T_total e distância L1:
                D_prop = 0.5 * Σ_f |p_real(f) - p_sim(f)|
                Score = clip(1 - D_prop, 0, 1)

        **Score do modelo:**
        - Média ponderada dos scores por poço usando `t_real_valid` como peso.
        """
        # 1) janela global (Ribbon)
        try:
            txt = self.cmb_debug_win.currentText()
            ws = int(txt.split("x")[0])
        except Exception:
            ws = 1
        self.well_rank_window_size = ws

        # 2) modelos / poços marcados
        checked_data = self.get_checked_models()
        selected_keys = [k for k, _name in checked_data]
        selected_wells = self.get_checked_wells()

        if not selected_keys or not selected_wells:
            self.tbl_models.setRowCount(0)
            self.tbl_wells.setRowCount(0)
            self._current_ranking_data = []
            self._update_ranking_overview_plot(None)
            return

        # 3) t_min do score
        try:
            if hasattr(self, "spin_rank_tmin"):
                self.well_rank_t_min = float(self.spin_rank_tmin.value())
        except Exception:
            pass
        tmin = float(getattr(self, "well_rank_t_min", 0.30) or 0.30)

        ranking = self.evaluate_models_against_wells(
            model_keys=selected_keys,
            well_names=selected_wells,
            window_size=ws,
            t_min=tmin,
            ignore_real_zeros=True,
        )

        self._current_ranking_data = ranking or []

        # 4) preenche tabela de modelos
        self.tbl_models.setSortingEnabled(False)
        self.tbl_models.setRowCount(0)

        if not ranking:
            self.tbl_models.setSortingEnabled(True)
            self.tbl_wells.setRowCount(0)
            self._update_ranking_overview_plot(None)
            return

        for i, r in enumerate(ranking, start=1):
            row = self.tbl_models.rowCount()
            self.tbl_models.insertRow(row)

            m_key = r.get("model_key")
            model_name = str(r.get("model_name", ""))

            # Col 0: Rank (guarda model_key)
            it_rank = QtWidgets.QTableWidgetItem(f"{i:02d}")
            it_rank.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            it_rank.setData(QtCore.Qt.UserRole, m_key)
            self.tbl_models.setItem(row, 0, it_rank)

            # Col 1: Study
            study_name = "Geral"
            if m_key in self.models:
                study_name = self.models[m_key].get("study", "Geral")
            it_study = QtWidgets.QTableWidgetItem(str(study_name))
            if m_key == "base":
                it_study.setBackground(QtGui.QBrush(QtGui.QColor(230, 240, 255)))
            self.tbl_models.setItem(row, 1, it_study)

            # Col 2: Modelo
            it_model = QtWidgets.QTableWidgetItem(model_name)
            if m_key == "base":
                it_model.setBackground(QtGui.QBrush(QtGui.QColor(230, 240, 255)))
            self.tbl_models.setItem(row, 2, it_model)

            # Col 3: Score
            sc = float(r.get("score", 0.0) or 0.0)
            it_score = QtWidgets.QTableWidgetItem(f"{sc:.3f}")
            it_score.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            if sc >= 0.70:
                it_score.setFont(QtGui.QFont("Arial", weight=QtGui.QFont.Bold))
            self.tbl_models.setItem(row, 3, it_score)

            # Col 4: ΣT_real (m)
            details = r.get("details", {}) or {}
            sum_t = 0.0
            for s in details.values():
                try:
                    sum_t += float(s.get("t_real_valid", 0.0) or 0.0)
                except Exception:
                    pass
            it_t = QtWidgets.QTableWidgetItem(f"{sum_t:.1f}")
            it_t.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            self.tbl_models.setItem(row, 4, it_t)

            # Col 5: Poços
            self.tbl_models.setItem(row, 5, QtWidgets.QTableWidgetItem(str(r.get("n_wells_used", 0))))

        self.tbl_models.setSortingEnabled(True)
        self.tbl_models.resizeColumnsToContents()

        # seleciona o primeiro para popular a tabela de poços e o gráfico
        if self.tbl_models.rowCount() > 0:
            self.tbl_models.selectRow(0)

    def _update_ranking_overview_plot(self, model_record):
        """Atualiza o gráfico de visão geral (REAL vs SIM) para o modelo selecionado."""
        if getattr(self, "rank_overview_ax", None) is None or getattr(self, "rank_overview_canvas", None) is None:
            return

        ax = self.rank_overview_ax
        ax.clear()

        if not model_record:
            ax.set_title("Selecione um modelo para ver a visão geral")
            self.rank_overview_canvas.draw_idle()
            return

        details = model_record.get("details", {}) or {}
        if not details:
            ax.set_title("Sem poços válidos para o modelo selecionado")
            self.rank_overview_canvas.draw_idle()
            return

        # limita para não ficar pesado/ilegível
        well_names = list(details.keys())
        max_show = 12
        if len(well_names) > max_show:
            well_names = well_names[:max_show]

        # max espessura para alinhar o eixo X
        max_t = 0.0
        for wn in well_names:
            s = details.get(wn, {}) or {}
            max_t = max(max_t, float(s.get("t_real_valid", 0.0) or 0.0), float(s.get("t_sim_valid", 0.0) or 0.0))
        if max_t <= 0:
            max_t = 1.0

        # função cor
        def _col(f):
            try:
                c = self.facies_colors_dict.get(int(f), (0.7, 0.7, 0.7))
                if isinstance(c, QtGui.QColor):
                    return (c.redF(), c.greenF(), c.blueF())
                if isinstance(c, (list, tuple)) and len(c) >= 3:
                    return (float(c[0]), float(c[1]), float(c[2]))
            except Exception:
                pass
            return (0.7, 0.7, 0.7)

        
        # Parâmetros visuais: linhas finas e separação clara entre REAL (baixo) e SIM (cima)
        row_h = 0.35
        gap = 0.18
        pad = 0.28
        block_h = row_h * 2 + gap + pad
        y = 0.0

        # Cabeçalho dentro do gráfico
        ax.text(0.0, 1.02, "Cada poço: REAL (baixo)  |  SIM (cima)", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9, color="#444")

        # Desenha de cima pra baixo (poço no topo do gráfico)
        for wn in reversed(well_names):
            s = details.get(wn, {}) or {}
            runs_r = s.get("runs_real", []) or []
            runs_s = s.get("runs_sim", []) or []
            sc = float(s.get("score", s.get("score_prop", 0.0)) or 0.0)

            # Linha guia de separação do poço
            ax.hlines(y - 0.06, 0.0, max_t, colors="#dddddd", linewidth=0.8)

            # REAL (linha inferior)
            x0 = 0.0
            for fac, t in runs_r:
                t = float(t)
                if t <= 0:
                    continue
                ax.broken_barh([(x0, t)], (y, row_h), facecolors=_col(fac))
                x0 += t

            # SIM (linha superior)
            x0 = 0.0
            for fac, t in runs_s:
                t = float(t)
                if t <= 0:
                    continue
                ax.broken_barh([(x0, t)], (y + row_h + gap, row_h), facecolors=_col(fac))
                x0 += t

            # label do poço + score no lado direito
            ax.text(-0.015 * max_t, y + row_h + gap / 2.0, str(wn),
                    ha="right", va="center", fontsize=9)
            ax.text(max_t * 1.01, y + row_h + gap / 2.0, f"S={sc:.2f}",
                    ha="left", va="center", fontsize=8, color="#444")

            y += block_h

        ax.hlines(y - 0.06, 0.0, max_t, colors="#dddddd", linewidth=0.8)

        # espaço extra à direita para anotações
        ax.set_xlim(0.0, max_t * 1.12)
        ax.set_ylim(-0.2, y - 0.2)
        ax.set_yticks([])
        ax.set_xlabel("Espessura acumulada (m)")
        ax.set_title(
            f"{model_record.get('model_name','')}  |  Score={float(model_record.get('score',0.0) or 0.0):.3f}  |  t_min={float(model_record.get('t_min', getattr(self,'well_rank_t_min',0.3))):.2f} m"
        )
        self.rank_overview_canvas.draw_idle()

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
            self._sync_context_docks_visibility()

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

    def _all_thickness_presets(self):
        presets = self.state.get("thickness_presets")
        if isinstance(presets, dict) and presets:
            return presets
        return {
            "Espessura total da coluna": ("__total_column_thickness__", "Espessura total da coluna (m)"),
            **get_vertical_metric_presets(prefix="vert_", include_filtered=True),
        }

    def _thickness_mode_base_label(self, label):
        suffix = " (filtrado)"
        label = str(label or "Espessura total da coluna")
        return label[:-len(suffix)] if label.endswith(suffix) else label

    def _compose_thickness_mode_label(self, base_label, use_filtered=None):
        base_label = self._thickness_mode_base_label(base_label)
        if use_filtered is None:
            use_filtered = self._is_thickness_filter_enabled()
        if base_label == "Espessura total da coluna":
            return base_label
        filtered_label = f"{base_label} (filtrado)"
        return filtered_label if use_filtered and filtered_label in self._all_thickness_presets() else base_label

    def _is_thickness_filter_enabled(self):
        if hasattr(self, "btn_thickness_filter") and self.btn_thickness_filter is not None:
            try:
                return bool(self.btn_thickness_filter.isChecked())
            except Exception:
                pass
        return bool(self.state.get("thickness_use_filtered", False))

    def _current_thickness_base_label(self):
        return self._thickness_mode_base_label(self.state.get("thickness_mode", "Espessura total da coluna"))

    def _on_toggle_thickness_filtered(self, checked):
        self.state["thickness_use_filtered"] = bool(checked)
        self._update_thick_btn(self._current_thickness_base_label())

    def _update_thick_btn(self, label):
        base_label = self._thickness_mode_base_label(label)
        use_filtered = self._is_thickness_filter_enabled()
        effective_label = self._compose_thickness_mode_label(base_label, use_filtered=use_filtered)

        # Texto do botão (ribbon)
        if hasattr(self, "btn_thick") and self.btn_thick is not None:
            suffix = " (filtrado)" if effective_label != base_label else ""
            self.btn_thick.setText(f"Espessura\n{base_label}{suffix}")
            self.btn_thick.setToolTip(f"Espessura: {base_label}{suffix}")

        # Sempre salva no state
        self.state["thickness_use_filtered"] = use_filtered
        self.state["thickness_mode"] = effective_label

        # Só aplica (render) se o visualize.run já tiver registrado refresh
        if "refresh" in self.state and callable(self.state["refresh"]):
            self.change_thickness_mode(base_label)

    def show_main_3d_view(self):
        """Alterna para a visualização 3D respeitando a perspectiva atual."""
        # Se estiver em COMPARAÇÃO, volta para o 3D da comparação (não troca perspectiva)
        if hasattr(self, "compare_stack") and self.central_stack.currentIndex() == 1:
            self.compare_stack.setCurrentIndex(0)  # comp_page_3d
            try:
                self.refresh_comparison_active_view()
            except Exception as e:
                print(f"[show_main_3d_view] erro comp 3D: {e}")
            self._sync_context_docks_visibility()
            return

        # Caso contrário, força perspectiva de Visualização
        if self.central_stack.currentIndex() != 0:
            self.switch_perspective("visualization")

        if hasattr(self, "viz_container"):
            self.viz_container.setCurrentIndex(0)
            model_key = self.state.get("active_model_key", "base")
            try:
                self.switch_main_view_to_model(model_key)
            except Exception:
                pass
        self._sync_context_docks_visibility()

    def show_map2d_view(self):
        """Alterna para Mapas 2D respeitando a perspectiva atual."""
        try:
            if self.state.get("mode", "facies") != "scalar":
                self._update_thick_btn("Espessura total da coluna")
        except Exception:
            pass

        # COMPARAÇÃO: abre a página 2D da comparação
        if hasattr(self, "compare_stack") and self.central_stack.currentIndex() == 1:
            self.compare_stack.setCurrentIndex(2)  # comp_page_2d
            try:
                self.update_dynamic_comparison_2d(self.get_checked_models())
            except Exception as e:
                print(f"[show_map2d_view] erro comp 2D: {e}")
            self._sync_context_docks_visibility()
            return

        # VISUALIZAÇÃO: abre a página 2D da visualização
        if self.central_stack.currentIndex() != 0:
            self.switch_perspective("visualization")

        if hasattr(self, "viz_container"):
            self.viz_container.setCurrentIndex(1)
            model_key = self.state.get("active_model_key", "base")
            try:
                self.switch_main_view_to_model(model_key)
                self.update_2d_map()
            except Exception:
                pass
        self._sync_context_docks_visibility()

    def show_metrics_view(self):
        """Alterna para Métricas (Força a perspectiva de Visualização)."""
        if self.central_stack.currentIndex() != 0:
            self.switch_perspective("visualization")

        if hasattr(self, "viz_container"):
            self.viz_container.setCurrentIndex(2)
            model_key = self.state.get("active_model_key", "base")
            try: self.update_metrics_view_content(model_key)
            except: pass
        self._sync_context_docks_visibility()

    def show_ranking_view(self):
        """Alterna para a visão de Ranking (Força a perspectiva de Visualização)."""
        if self.central_stack.currentIndex() != 0:
            self.switch_perspective("visualization")

        # Na visualização simples, o Ranking é o índice 3
        if hasattr(self, "viz_container"):
            self.viz_container.setCurrentIndex(3)
            # Atualiza o conteúdo do ranking
            self.update_ranking_view_content()
        self._sync_context_docks_visibility()

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

    def _sync_context_docks_visibility(self):
        """Mostra/esconde docks contextuais conforme a vista ativa."""
        show_map2d_summary = False

        try:
            if self.current_perspective != "uncertainty":
                if self.central_stack.currentIndex() == 0 and hasattr(self, "viz_container"):
                    show_map2d_summary = (self.viz_container.currentIndex() == 1)
                elif self.central_stack.currentIndex() == 1 and hasattr(self, "compare_stack"):
                    show_map2d_summary = (self.compare_stack.currentIndex() == 2)
        except Exception:
            show_map2d_summary = False

        dock = getattr(self, "dock_map2d_summary", None)
        if dock is not None:
            try:
                dock.setVisible(bool(show_map2d_summary))
                if show_map2d_summary:
                    dock.raise_()
            except Exception:
                pass


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

        explorer_container = QtWidgets.QWidget()
        explorer_layout = QtWidgets.QVBoxLayout(explorer_container)
        explorer_layout.setContentsMargins(0, 0, 0, 0)
        explorer_layout.setSpacing(0)
        explorer_layout.addWidget(self.project_tree, 1)

        self.dock_explorer.setWidget(explorer_container)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dock_explorer)

        self.dock_map2d_summary = QtWidgets.QDockWidget("Resumo da coluna 2D", self)
        self.dock_map2d_summary.setObjectName("dock_map2d_summary")
        self.dock_map2d_summary.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.dock_map2d_summary.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFloatable |
            QtWidgets.QDockWidget.DockWidgetClosable
        )

        self.map2d_summary_text = QtWidgets.QTextBrowser()
        self.map2d_summary_text.setReadOnly(True)
        self.map2d_summary_text.setOpenExternalLinks(False)
        self.map2d_summary_text.setMinimumHeight(150)
        self.map2d_summary_text.setHtml(
            "<span style='color:#666'>Clique em uma célula do mapa 2D para ver o resumo da coluna.</span>"
        )
        self.dock_map2d_summary.setWidget(self._wrap_expanding(self.map2d_summary_text))
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dock_map2d_summary)
        try:
            self.splitDockWidget(self.dock_explorer, self.dock_map2d_summary, QtCore.Qt.Vertical)
        except Exception:
            pass
        self.dock_map2d_summary.hide()

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

        # --- Agrupamento de fácies ---
        h_group_cfg = QtWidgets.QHBoxLayout()
        self.chk_use_facies_grouping = QtWidgets.QCheckBox("Usar grupos")
        self.chk_use_facies_grouping.setToolTip("Aplica a configuração de grupos de fácies na visualização e nos filtros.")
        self.chk_use_facies_grouping.setChecked(bool(getattr(self, "use_facies_grouping", False)))
        self.chk_use_facies_grouping.toggled.connect(self.on_toggle_use_facies_grouping)
        btn_cfg_groups = QtWidgets.QPushButton("Configurar…")
        btn_cfg_groups.setToolTip("Definir agrupamentos (grupos) de fácies")
        btn_cfg_groups.clicked.connect(self.open_facies_grouping_dialog)
        h_group_cfg.addWidget(self.chk_use_facies_grouping)
        h_group_cfg.addStretch(1)
        h_group_cfg.addWidget(btn_cfg_groups)
        lgl.addLayout(h_group_cfg)

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
        self.dock_props.setMinimumWidth(60)

        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dock_props)

        # Ajuste inicial de larguras
        self.resizeDocks([self.dock_explorer, self.dock_props], [180, 250], QtCore.Qt.Horizontal)

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

        # Guarda facies raw e aplica agrupamento (se ativo)
        target_facies_raw = target_facies
        if bool(getattr(self, 'use_facies_grouping', False)):
            try:
                target_facies = self.apply_facies_grouping(target_facies_raw)
            except Exception:
                target_facies = target_facies_raw
        else:
            target_facies = target_facies_raw

        # --- preserva modo global ---
        desired_mode = self.state.get("mode", "facies")

        # --- atualiza state ---
        self.active_model_key = model_key_norm
        self.state["active_model_key"] = model_key_norm
        self.state["current_grid_source"] = source_grid
        self.state["current_facies_raw"] = target_facies_raw
        self.state["current_facies"] = target_facies
        self.state["mode"] = desired_mode

        # garante Facies no grid atual (se possível)
        try:
            try:
                source_grid.cell_data["FaciesRaw"] = target_facies_raw
            except Exception:
                pass
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

        # Normaliza entrada (GLOBAL) - respeita modo raw vs agrupado
        rf_active = set(int(x) for x in (reservoir_set or []))

        if bool(getattr(self, "use_facies_grouping", False)):
            # Seleção recebida está no espaço de "grupos"
            self.state_reservoir_grouped = set(rf_active)
            inv = self._build_inverse_grouping()
            rf_raw_global = set()
            for g in self.state_reservoir_grouped:
                rf_raw_global |= inv.get(int(g), {int(g)})
            self.state_reservoir_raw = set(rf_raw_global)
        else:
            # Seleção recebida está no espaço raw
            self.state_reservoir_raw = set(rf_active)
            self.state_reservoir_grouped = {int(self.facies_grouping_map.get(int(x), int(x))) for x in self.state_reservoir_raw}
            rf_raw_global = set(self.state_reservoir_raw)

        # Atualiza state (active = o que combina com a fácies exibida)
        self.state["reservoir_facies_raw"] = set(self.state_reservoir_raw)
        self.state["reservoir_facies_grouped"] = set(self.state_reservoir_grouped)
        self.state["reservoir_facies"] = set(self.state_reservoir_grouped if bool(getattr(self, "use_facies_grouping", False)) else self.state_reservoir_raw)

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
            rf_local = set(rf_raw_global & present)
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
        """Atualiza o plotter 2D principal usando o Grid Ativo."""
        if not hasattr(self, "plotter_2d"):
            return

        active_grid = self.state.get("current_grid_source")
        if active_grid is None:
            from load_data import grid as active_grid

        mode_3d = self.state.get("mode", "facies")

        if mode_3d == "scalar" and self.state.get("current_scalar_name"):
            scalar_name = self.state.get("current_scalar_name")
            title = self.state.get("current_scalar_title", scalar_name)
            cmap_use = self.state.get("current_scalar_cmap") or self.state.get("thickness_cmap", "jet")
            clim_override = self.state.get("current_scalar_clim")
        else:
            presets = self.state.get("thickness_presets") or {}
            mode = self.state.get("thickness_mode", "Espessura total da coluna")
            if mode not in presets:
                if "Espessura" in presets:
                    mode = "Espessura"
                else:
                    return

            scalar_name, title = presets[mode]
            cmap_use = self.state.get("thickness_cmap", "jet")
            clim_override = self.state.get("thickness_clim")

        if scalar_name == "__total_column_thickness__":
            clim_override = None
            title = "Espessura total da coluna (m)"
        elif self._is_equivalent_2d_property(scalar_name):
            clim_override = None
            title = f"{scalar_name} equivalente (m)"

        try:
            self._draw_2d_map_local(
                self.plotter_2d,
                active_grid,
                scalar_name,
                title,
                cmap=cmap_use,
                clim_override=clim_override,
                show_scalar_bar=True,
                scalar_bar_title=title,
            )
        except Exception as e:
            print(f"Erro ao atualizar mapa 2D: {e}")


    def update_compare_2d_maps(self):
        """Atualiza os mapas 2D de todos os modelos ativos na comparação."""
        if not hasattr(self, "state"):
            return

        mode_3d = self.state.get("mode", "facies")

        if mode_3d == "scalar" and self.state.get("current_scalar_name"):
            scalar = self.state.get("current_scalar_name")
            title_suffix = self.state.get("current_scalar_title", scalar)
            cmap_use = self.state.get("current_scalar_cmap") or self.state.get("thickness_cmap", "jet")
        else:
            presets = self.state.get("thickness_presets", {})
            mode = self.state.get("thickness_mode", "Espessura total da coluna")

            if mode not in presets:
                if presets:
                    mode = list(presets.keys())[0]
                else:
                    return

            scalar, title_suffix = presets[mode]
            cmap_use = self.state.get("thickness_cmap", "jet")

        if scalar == "__total_column_thickness__":
            title_suffix = "Espessura total da coluna (m)"
        elif self._is_equivalent_2d_property(scalar):
            title_suffix = f"{scalar} equivalente (m)"

        grids = []
        if hasattr(self, "active_comp_states"):
            for st in self.active_comp_states:
                g = st.get("current_grid_source")
                if g is not None:
                    grids.append(g)

        clim_override = self._compute_global_2d_clim(grids, scalar)

        if hasattr(self, "active_comp_2d_plotters") and hasattr(self, "active_comp_states"):
            for plotter, state in zip(self.active_comp_2d_plotters, self.active_comp_states):
                grid = state.get("current_grid_source")
                if grid is None:
                    continue

                self._draw_2d_map_local(
                    plotter,
                    grid,
                    scalar,
                    title_suffix,
                    cmap=cmap_use,
                    show_scalar_bar=True,
                    scalar_bar_title=title_suffix,
                    clim_override=clim_override,
                )

    def _draw_2d_map_local(
        self,
        plotter,
        grid_source,
        scalar_name_3d,
        title,
        *,
        cmap=None,
        show_scalar_bar=True,
        scalar_bar_title=None,
        clim_override=None,
    ):
        """Desenha um mapa 2D (redução por coluna) no plotter."""
        import pyvista as pv
        import numpy as np
        from visualize import get_2d_clim

        plotter.clear()
        try:
            plotter.remove_scalar_bar()
        except Exception:
            pass

        if grid_source is None:
            plotter.render()
            return

        dims = self._infer_grid_cell_dims(grid_source)
        if not dims:
            plotter.render()
            return

        nx, ny, nz = dims  # <- número de CÉLULAS

        map_2d = self._reduce_grid_scalar_to_2d(grid_source, scalar_name_3d)
        if map_2d is None:
            plotter.render()
            return

        total_thickness_2d = self._reduce_total_column_thickness_to_2d(grid_source)

        x_min, x_max, y_min, y_max, _, z_max = grid_source.bounds

        # Para representar nx x ny CÉLULAS, a superfície precisa ter (nx+1) x (ny+1) PONTOS
        xs = np.linspace(x_min, x_max, nx + 1)
        ys = np.linspace(y_max, y_min, ny + 1)
        xs, ys = np.meshgrid(xs, ys, indexing="ij")
        zs = np.full_like(xs, z_max, dtype=float)

        surf = pv.StructuredGrid(xs, ys, zs)

        name2d = scalar_name_3d + "_2d"

        arr2d = np.asarray(map_2d, dtype=float)
        if arr2d.shape != (nx, ny):
            try:
                arr2d = arr2d.reshape((nx, ny), order="F")
            except Exception:
                plotter.render()
                return

        # Agora usa TODAS as células do mapa, sem cortar [:nx-1, :ny-1]
        surf.cell_data[name2d] = arr2d.ravel(order="F")

        arr = np.asarray(surf.cell_data[name2d], dtype=float)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        surf.cell_data[name2d] = arr

        if scalar_name_3d == "__total_column_thickness__" or self._is_equivalent_2d_property(scalar_name_3d):
            finite = arr[np.isfinite(arr)]
            if finite.size > 0:
                vmin = float(np.nanmin(finite))
                vmax = float(np.nanmax(finite))
                if vmin >= 0.0:
                    vmin = 0.0
                if vmax <= vmin:
                    vmax = vmin + 1e-6
                clim = (vmin, vmax)
            else:
                clim = (0.0, 1.0)
        else:
            clim = clim_override if clim_override is not None else get_2d_clim(
                base_scalar_name=scalar_name_3d,
                arr=arr,
            )

        if cmap is None:
            cmap = self.state.get("current_scalar_cmap") or self.state.get("thickness_cmap") or "jet"

        plotter.add_mesh(
            surf,
            scalars=name2d,
            cmap=cmap,
            show_edges=True,
            edge_color="black",
            line_width=0.5,
            nan_color="white",
            show_scalar_bar=False,
            clim=clim,
        )

        plotter.view_xy()
        plotter.enable_parallel_projection()
        plotter.enable_image_style()
        plotter.set_background("white")
        plotter.add_axes()
        plotter.show_bounds(
            grid="front",
            location="outer",
            ticks="outside",
            color="gray",
            minor_ticks=True,
            n_xlabels=4,
            n_ylabels=4,
            font_size=8,
            fmt="%.0f",
            xtitle="X",
            ytitle="Y",
        )

        bar_title = scalar_bar_title or title
        if scalar_name_3d == "__total_column_thickness__":
            bar_title = "Espessura total da coluna (m)"
        elif self._is_equivalent_2d_property(scalar_name_3d):
            bar_title = f"{scalar_name_3d} equivalente (m)"

        if show_scalar_bar:
            plotter.add_scalar_bar(
                title=bar_title,
                n_labels=5,
                fmt="%.3g",
                title_font_size=14,
                label_font_size=12,
            )

        try:
            plotter._map2d_hover_meta = {
                "surf": surf,
                "name2d": name2d,
                "label": bar_title,
                "model_name": getattr(plotter, "_hover2d_model_name", None),
                "grid_source": grid_source,
                "scalar_name_3d": scalar_name_3d,
                # usa as dimensões da SUPERFÍCIE (em pontos), não as do grid em células
                "dims": tuple(int(v) for v in surf.dimensions),
                "total_thickness_2d": total_thickness_2d,
            }
        except Exception:
            pass

        try:
            self._install_2d_hover_filter(
                plotter,
                model_name=getattr(plotter, "_hover2d_model_name", None),
            )
        except Exception:
            pass

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




    
    # ------------------------------------------------------------------
    # Agrupamento de fácies (UI/configuração)
    # ------------------------------------------------------------------
    def _rebuild_facies_grouping_cache(self):
        """Reconstrói caches (src/dst) para aplicar mapeamento facies->grupo de forma rápida."""
        try:
            keys = sorted(int(k) for k in self.facies_grouping_map.keys())
        except Exception:
            keys = []
        self._fg_src = np.array(keys, dtype=np.int32) if keys else np.array([], dtype=np.int32)
        if keys:
            self._fg_dst = np.array([int(self.facies_grouping_map.get(int(k), int(k))) for k in keys], dtype=np.int32)
        else:
            self._fg_dst = np.array([], dtype=np.int32)

    def apply_facies_grouping(self, facies_array_1d):
        """Aplica o mapeamento facies->grupo em um array 1D (int32) de fácies.

        - Mantém valores que não existirem no mapping (ex.: 0, -999...) como estão.
        - Implementação vetorizada via searchsorted para performance.
        """
        arr = np.asarray(facies_array_1d).ravel().astype(np.int32)
        if not getattr(self, "facies_grouping_map", None):
            return arr
        if self._fg_src.size == 0:
            return arr

        flat = arr
        idx = np.searchsorted(self._fg_src, flat)
        mask = (idx < self._fg_src.size) & (self._fg_src[idx] == flat)
        if not np.any(mask):
            return flat
        out = flat.copy()
        out[mask] = self._fg_dst[idx[mask]]
        return out

    def _build_inverse_grouping(self):
        inv = {}
        for orig, grp in (self.facies_grouping_map or {}).items():
            try:
                o = int(orig); g = int(grp)
            except Exception:
                continue
            inv.setdefault(g, set()).add(o)
        return inv

    def on_toggle_use_facies_grouping(self, checked):
        """Alterna entre usar fácies originais (raw) e fácies agrupadas (grupo)."""
        self.use_facies_grouping = bool(checked)

        # Converte seleção de reservatório para manter intenção
        if self.use_facies_grouping:
            # raw -> grouped
            self.state_reservoir_raw = set(self.state.get("reservoir_facies_raw", set()) or set())
            self.state_reservoir_grouped = {int(self.facies_grouping_map.get(int(x), int(x))) for x in self.state_reservoir_raw}
        else:
            # grouped -> raw (expande membros)
            self.state_reservoir_grouped = set(self.state.get("reservoir_facies_grouped", set()) or set())
            inv = self._build_inverse_grouping()
            raw = set()
            for g in self.state_reservoir_grouped:
                raw |= inv.get(int(g), {int(g)})
            self.state_reservoir_raw = raw

        # Atualiza state
        self.state["reservoir_facies_raw"] = set(self.state_reservoir_raw)
        self.state["reservoir_facies_grouped"] = set(self.state_reservoir_grouped)
        self.state["reservoir_facies"] = set(self.state_reservoir_grouped if self.use_facies_grouping else self.state_reservoir_raw)

        # Reaplica no modelo ativo para atualizar campo Facies, legenda e filtros
        try:
            self.switch_main_view_to_model(getattr(self, "active_model_key", "base"))
        except Exception as e:
            print("[on_toggle_use_facies_grouping] falhou:", e)
            try:
                self.populate_facies_legend()
            except Exception:
                pass

        # Se estiver na aba Ranking, recalcula (para refletir o toggle)
        if hasattr(self, "viz_container") and self.viz_container.currentIndex() == 3:
            try:
                self._schedule_heavy_update()
            except Exception:
                try:
                    self.update_ranking_view_content()
                except Exception:
                    pass

    def open_facies_grouping_dialog(self):
        """Abre o diálogo de configuração de grupos de fácies."""
        dlg = FaciesGroupingDialog(self.facies_reference, self.facies_colors_dict, self.facies_grouping_map, parent=self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        new_map = dlg.get_mapping()
        self.facies_grouping_map = dict(new_map)
        self._rebuild_facies_grouping_cache()

        # Recalcula sets raw/agrupado conforme toggle atual
        self.state_reservoir_raw = set(self.state.get("reservoir_facies_raw", set()) or set())
        self.state_reservoir_grouped = {int(self.facies_grouping_map.get(int(x), int(x))) for x in self.state_reservoir_raw}

        self.state["reservoir_facies_raw"] = set(self.state_reservoir_raw)
        self.state["reservoir_facies_grouped"] = set(self.state_reservoir_grouped)
        self.state["reservoir_facies"] = set(self.state_reservoir_grouped if self.use_facies_grouping else self.state_reservoir_raw)

        # Atualiza visualização/legenda
        try:
            self.switch_main_view_to_model(getattr(self, "active_model_key", "base"))
        except Exception as e:
            print("[open_facies_grouping_dialog] refresh falhou:", e)
            try:
                self.populate_facies_legend()
            except Exception:
                pass


    def populate_facies_legend(self):
        """Preenche a legenda lateral com as estatísticas do GRID ATIVO na visualização."""
        self.facies_legend_table.blockSignals(True)
        colors_dict = getattr(self, 'facies_colors_dict', load_facies_colors())
        
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
            self.recalc_entropy_view()
            if hasattr(self, "btn_mode") and self.btn_mode is not None:
                self.btn_mode.setText("Modo\nEntropia")
            return

        # A) Atualiza Estado Global
        self.state["mode"] = new_mode
        for k in list(self.models.keys()):
            try:
                self.models[k]["view_mode"] = new_mode
            except Exception:
                pass

        # B) Atualiza Visualização PRINCIPAL
        current_f = self.state.get("current_facies")
        if current_f is None:
            current_f = base_facies

        rf_global = set(self.state.get("reservoir_facies", set()) or [])
        present = set(int(v) for v in np.unique(np.asarray(current_f).astype(int)))
        rf_active = rf_global & present

        if new_mode in ("reservoir", "clusters", "largest"):
            upd = self.state.get("update_reservoir_fields")
            if callable(upd):
                try:
                    upd(set(rf_active))
                except Exception:
                    pass

        refresh = self.state.get("refresh")
        if callable(refresh):
            refresh()

        # Legendas
        try:
            if new_mode in ("clusters", "largest"):
                if hasattr(self, "facies_legend_table"):
                    self.facies_legend_table.setVisible(False)
                if hasattr(self, "clusters_legend_table"):
                    self.clusters_legend_table.setVisible(True)
                    self.populate_clusters_legend()
            else:
                if hasattr(self, "clusters_legend_table"):
                    self.clusters_legend_table.setVisible(False)
                if hasattr(self, "facies_legend_table"):
                    self.facies_legend_table.setVisible(True)
                    self.populate_facies_legend()
        except Exception:
            pass

        if hasattr(self, "viz_container"):
            idx = self.viz_container.currentIndex()
            if idx == 1 and hasattr(self, "update_2d_map"):
                self.update_2d_map()

        # C) COMPARAÇÃO
        if hasattr(self, "central_stack") and self.central_stack.currentIndex() == 1:
            # 1) Atualiza estados dos plotters 3D (se existirem)
            if hasattr(self, "active_comp_states"):
                for st in self.active_comp_states:
                    try:
                        st["mode"] = new_mode
                        m_key = st.get("model_key")
                        rf_local = set()
                        if m_key and m_key in self.models:
                            rf_local = self.models[m_key].get("reservoir_facies", set()) or set()

                        if new_mode in ("reservoir", "clusters", "largest"):
                            if "update_reservoir_fields" in st and callable(st["update_reservoir_fields"]):
                                st["update_reservoir_fields"](rf_local)

                        if "refresh" in st and callable(st["refresh"]):
                            st["refresh"]()
                    except Exception:
                        pass

            # ✅ Correção: força atualizar a aba ativa (3D/2D/Métricas) + painel lateral
            try:
                if hasattr(self, "refresh_comparison_active_view"):
                    self.refresh_comparison_active_view()
            except Exception:
                pass

            try:
                if hasattr(self, "populate_mode_menu"):
                    self.populate_mode_menu()
            except Exception:
                pass

        self._schedule_wells_update()

    def change_thickness_mode(self, label):
        # Voltou para preset "normal" => deixa o visualize recalcular clim automático
        self.state["thickness_clim_manual"] = False
        self.state["thickness_global_clim"] = None

        base_label = self._thickness_mode_base_label(label)
        use_filtered = self._is_thickness_filter_enabled()
        effective_label = self._compose_thickness_mode_label(base_label, use_filtered=use_filtered)

        self.state["thickness_use_filtered"] = use_filtered
        self.state["thickness_mode"] = effective_label
        is_total_column_mode = (base_label == "Espessura total da coluna")

        # 1. Atualiza Visualização PRINCIPAL
        if "update_thickness" in self.state and callable(self.state["update_thickness"]):
            self.state["update_thickness"]()

        refresh = self.state.get("refresh")
        if callable(refresh):
            refresh()

        # Atualiza 2D Main
        if hasattr(self, "update_2d_map") and callable(self.update_2d_map):
            self.update_2d_map()

        if is_total_column_mode:
            try:
                if hasattr(self, "central_stack") and self.central_stack.currentIndex() == 0 and hasattr(self, "viz_container"):
                    self.viz_container.setCurrentIndex(1)
                elif hasattr(self, "central_stack") and self.central_stack.currentIndex() == 1 and hasattr(self, "compare_stack"):
                    self.compare_stack.setCurrentIndex(2)
            except Exception:
                pass

        # 2. ATUALIZA COMPARAÇÃO
        if hasattr(self, "active_comp_states"):
            for st in self.active_comp_states:
                st["thickness_clim_manual"] = False
                st["thickness_global_clim"] = None
                st["thickness_use_filtered"] = use_filtered
                st["thickness_mode"] = effective_label

                if "update_thickness" in st:
                    st["update_thickness"]()
                if "refresh" in st:
                    st["refresh"]()
                if "plotter_ref" in st:
                    st["plotter_ref"].render()

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
        
        # --- Agrupamento de fácies ---
        h_group_cfg = QtWidgets.QHBoxLayout()
        self.chk_use_facies_grouping = QtWidgets.QCheckBox("Usar grupos")
        self.chk_use_facies_grouping.setToolTip("Aplica a configuração de grupos de fácies na visualização e nos filtros.")
        self.chk_use_facies_grouping.setChecked(bool(getattr(self, "use_facies_grouping", False)))
        self.chk_use_facies_grouping.toggled.connect(self.on_toggle_use_facies_grouping)
        btn_cfg_groups = QtWidgets.QPushButton("Configurar…")
        btn_cfg_groups.setToolTip("Definir agrupamentos (grupos) de fácies")
        btn_cfg_groups.clicked.connect(self.open_facies_grouping_dialog)
        h_group_cfg.addWidget(self.chk_use_facies_grouping)
        h_group_cfg.addStretch(1)
        h_group_cfg.addWidget(btn_cfg_groups)
        l_filt.addLayout(h_group_cfg)

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
            for c in range(1, cols):  # Colunas de modelos começam no índice 1
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

            # Se estiver na aba 2D da comparação, reconstrói os mapas
            if hasattr(self, "compare_stack") and self.compare_stack.currentIndex() == 2:
                try:
                    self.update_dynamic_comparison_2d(self.get_checked_models())
                except Exception as e:
                    print(f"[toggle_all_multi_model] update_dynamic_comparison_2d falhou: {e}")
    
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
        """Alterna entre visualização (0), comparação (1) e incerteza (2)."""
        self.current_perspective = mode

        # --- CONTROLE DE VISIBILIDADE DOS DOCKS ---
        if mode == "uncertainty":
            # Esconde Explorer, Inspector e resumo 2D
            if hasattr(self, "dock_explorer"): self.dock_explorer.hide()
            if hasattr(self, "dock_props"): self.dock_props.hide()
            if hasattr(self, "dock_map2d_summary"): self.dock_map2d_summary.hide()
        else:
            # Mostra Explorer e Inspector nos outros modos
            if hasattr(self, "dock_explorer"): self.dock_explorer.show()
            if hasattr(self, "dock_props"): self.dock_props.show()

        # --- LÓGICA DE STACKS ---
        if mode == "visualization":
            self.central_stack.setCurrentIndex(0)
            self.act_persp_viz.setChecked(True)
            self.act_persp_comp.setChecked(False)
            if hasattr(self, "act_view_uncert"): self.act_view_uncert.setChecked(False)
            if hasattr(self, "act_view_3d"): self.act_view_3d.setChecked(True)
            self.show_main_3d_view()

        elif mode == "comparison":
            self.central_stack.setCurrentIndex(1)
            self.act_persp_viz.setChecked(False)
            self.act_persp_comp.setChecked(True)
            if hasattr(self, "act_view_uncert"): self.act_view_uncert.setChecked(False)
            checked_models = self.get_checked_models()
            self.update_dynamic_comparison_view(checked_models)

        elif mode == "uncertainty":
            self.central_stack.setCurrentIndex(2)
            self.act_persp_viz.setChecked(False)
            self.act_persp_comp.setChecked(False)
            if hasattr(self, "act_view_uncert"): self.act_view_uncert.setChecked(True)

        self._sync_context_docks_visibility()

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

        if checked_models is None:
            checked_models = self.get_checked_models()

        final_list = []
        for item in checked_models:
            if isinstance(item, (tuple, list)):
                raw_key, raw_name = item[0], item[1] if len(item) > 1 else str(item[0])
            else:
                raw_key, raw_name = item, str(item)

            m_key = str(raw_key)
            m_name = str(raw_name)

            if m_name.startswith("('"):
                if m_key in self.models:
                    m_name = self.models[m_key].get("name", m_name)

            final_list.append((m_key, m_name))

        self.update_multi_model_filter_table(final_list)

        # --- LIMPEZA ---
        if hasattr(self, "active_comp_plotters"):
            for p in self.active_comp_plotters:
                try:
                    p.close()
                except:
                    pass
        self.active_comp_plotters = []
        self.active_comp_states = []
        self.compare_states_multi = {}

        while self.comp_layout_3d.count():
            item = self.comp_layout_3d.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Globais
        mode = self.state.get("mode", "facies")
        thickness_mode = self.state.get("thickness_mode", "Espessura total da coluna")
        z_exag = float(self.state.get("z_exag", 15.0))
        show_scalar_bar = bool(self.state.get("show_scalar_bar", True))

        # Layout
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
            self.comp_layout_3d.addWidget(QtWidgets.QLabel("Nenhum modelo selecionado."))
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

            try:
                plotter.set_background("white")
            except:
                pass

            v_lay.addWidget(plotter_widget)
            grid_layout.addWidget(w_container, row, col)

            # Dados
            grid_obj, facies_obj = self._get_model_payload(model_key)

            if grid_obj is None:
                plotter.add_text("GRID OFF", font_size=12)
            else:
                local_state = {
                    "thickness_mode": thickness_mode,
                    "thickness_cmap": self.state.get("thickness_cmap", "jet"),
                    "current_scalar_cmap": self.state.get("current_scalar_cmap", self.state.get("thickness_cmap", "jet")),
                }

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

        try:
            self._apply_global_clim_to_active_comparison()
        except Exception:
            pass

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
            
            

            # Recalcula campos derivados (Reservoir/Clusters/Verticais) no state visual
            if "update_reservoir_fields" in st:
                try:
                    st["update_reservoir_fields"](target_set)
                except Exception as e:
                    print(f"[on_multi_model_filter_changed] update_reservoir_fields falhou: {e}")

            # Se estiver em modo de espessura local, atualiza o scalar (para refletir novo reservatório)
            if st.get("mode") == "thickness_local" and "update_thickness" in st:
                try:
                    st["update_thickness"]()
                except Exception:
                    pass

# Agora pode chamar o refresh seguro
            if "refresh" in st:
                st["refresh"]()
            
            # Força renderização imediata
            if "plotter_ref" in st:
                st["plotter_ref"].render()

        # Se estiver na aba 2D da comparação, reconstrói os mapas para refletir o novo reservatório
        if hasattr(self, "compare_stack") and self.compare_stack.currentIndex() == 2:
            try:
                self.update_dynamic_comparison_2d(self.get_checked_models())
            except Exception as e:
                print(f"[on_multi_model_filter_changed] update_dynamic_comparison_2d falhou: {e}")



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
        self._sync_context_docks_visibility()
    
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

        try:
            self.update_multi_model_filter_table(normalized)
        except Exception:
            pass

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
            ("Proporção", "ntg", "{:.3f}"),
            ("Total Células", "total_cells", "{:d}"),
            ("Células Selecionadas", "res_cells", "{:d}"),
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
        import numpy as np
        from PyQt5 import QtWidgets

        if hasattr(self, "active_comp_2d_plotters"):
            for p in self.active_comp_2d_plotters:
                try:
                    p.close()
                except Exception:
                    pass
        self.active_comp_2d_plotters = []

        while self.comp_2d_layout.count():
            item = self.comp_2d_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

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

        n_models = len(normalized)
        cols = 2 if n_models > 1 else 1

        grid_container = QtWidgets.QWidget()
        grid_layout = QtWidgets.QGridLayout(grid_container)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(2)
        self.comp_2d_layout.addWidget(grid_container)

        mode_3d = self.state.get("mode", "facies")
        is_scalar_mode = (mode_3d == "scalar" and bool(self.state.get("current_scalar_name")))

        if is_scalar_mode:
            scalar = self.state.get("current_scalar_name")
            title = self.state.get("current_scalar_title", scalar)
            cmap_use = self.state.get("current_scalar_cmap") or self.state.get("thickness_cmap", "jet")
        else:
            presets = self.state.get("thickness_presets") or {}
            thick_mode = self.state.get("thickness_mode", "Espessura total da coluna")
            if thick_mode not in presets and presets:
                thick_mode = list(presets.keys())[0]
            if thick_mode not in presets:
                thick_mode = "Espessura"

            scalar, title = presets.get(thick_mode, ("vert_Ttot_reservoir", "Espessura"))
            cmap_use = self.state.get("thickness_cmap", "jet")

        from load_data import grid as global_grid
        base_grid = self.models.get("base", {}).get("grid") or self.state.get("current_grid_source") or global_grid

        def _infer_dims(g):
            try:
                dims_pts = getattr(g, "dimensions", None)
                if dims_pts and len(dims_pts) == 3:
                    cx, cy, cz = int(dims_pts[0] - 1), int(dims_pts[1] - 1), int(dims_pts[2] - 1)
                    if cx > 0 and cy > 0 and cz > 0 and (cx * cy * cz == g.n_cells):
                        return cx, cy, cz
            except Exception:
                pass
            try:
                from load_data import nx as lnx, ny as lny, nz as lnz
                return int(lnx), int(lny), int(lnz)
            except Exception:
                return None

        def _reduce_grid_to_2d(g, scalar_name):
            if g is None or scalar_name not in getattr(g, "cell_data", {}):
                return None

            dims = _infer_dims(g)
            if not dims:
                return None

            nx_, ny_, nz_ = dims

            try:
                arr3d = np.asarray(g.cell_data[scalar_name], dtype=float).reshape((nx_, ny_, nz_), order="F")
            except Exception:
                return None

            out2d = np.full((nx_, ny_), np.nan, dtype=float)
            for ix in range(nx_):
                for iy in range(ny_):
                    colv = arr3d[ix, iy, :]
                    finite = colv[np.isfinite(colv)]
                    if finite.size > 0:
                        out2d[ix, iy] = float(np.nanmax(finite))
            return out2d

        prepared = []
        for (model_key, model_name) in normalized:
            model_data = self.models.get(model_key, {})
            src_grid = model_data.get("grid") or base_grid
            if src_grid is None:
                continue

            temp_grid = src_grid.copy(deep=True)

            fac = model_data.get("facies")
            if fac is not None:
                temp_grid.cell_data["Facies"] = fac

            # Só recalcula métricas verticais quando o scalar depende disso
            if str(scalar).startswith("vert_") or scalar in ("Reservoir", "Clusters", "LargestCluster", "NTG_local"):
                try:
                    self.recalc_vertical_metrics(temp_grid, fac, model_data.get("reservoir_facies"))
                except Exception as e:
                    print(f"[compare_2d] recalc_vertical_metrics falhou para {model_name}: {e}")

            prepared.append((model_key, model_name, temp_grid))

        clim_override = self._compute_global_2d_clim(
            [g for (_k, _n, g) in prepared],
            scalar
        )

        if scalar == "__total_column_thickness__":
            title = "Espessura total da coluna (m)"
        elif self._is_equivalent_2d_property(scalar):
            title = f"{scalar} equivalente (m)"

        for idx, (model_key, model_name, temp_grid) in enumerate(prepared):
            row, col = idx // cols, idx % cols

            p2d = BackgroundPlotter(show=False)
            self.active_comp_2d_plotters.append(p2d)
            p2d._hover2d_model_name = model_name
            p2d._map2d_summary_target = getattr(self, "map2d_summary_text", None)
            self._install_2d_hover_filter(p2d, model_name=model_name)

            self._draw_2d_map_local(
                p2d,
                temp_grid,
                scalar,
                title,
                cmap=cmap_use,
                show_scalar_bar=True,
                scalar_bar_title=title,
                clim_override=clim_override,
            )

            w = QtWidgets.QWidget()
            vl = QtWidgets.QVBoxLayout(w)
            vl.setContentsMargins(0, 0, 0, 0)
            vl.setSpacing(0)

            mode_label = self.state.get("thickness_mode", scalar) if not is_scalar_mode else scalar
            if scalar == "__total_column_thickness__":
                mode_label = "Espessura total da coluna (m)"
            elif self._is_equivalent_2d_property(scalar):
                mode_label = f"{scalar} equivalente (m)"
            lbl = QtWidgets.QLabel(f"  {model_name} ({mode_label})")
            lbl.setStyleSheet("background: #ddd; font-weight: bold;")
            vl.addWidget(lbl)
            vl.addWidget(p2d.interactor)

            grid_layout.addWidget(w, row, col)

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

    def recalc_vertical_metrics(self, target_grid, facies_array, reservoir_set):
        """Wrapper centralizado: métricas verticais passam a ser calculadas em analysis.py."""
        try:
            compute_vertical_metrics_for_grid(
                target_grid,
                facies_array,
                reservoir_set,
                prefix="vert_",
                thin_lamination_threshold=0.30,
                include_filtered=True,
            )
        except Exception as e:
            print(f"[recalc_vertical_metrics] falhou: {e}")


    def _open_matplotlib_report(
            self, well_name, sim_model_name,
            real_depth, real_fac, base_depth,
            base_fac, sim_depth, sim_fac,
            best_depth=None, best_fac=None,
            window_size_str="1x1",
            well_logs_df=None,
            real_log_col=None):
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
        # Se agrupamento estiver ativo, aplica também aos logs do poço (REAL/BASE/SIM)
        if getattr(self, "use_facies_grouping", False) and getattr(self, "facies_grouping_map", None):
            real_fac = self.apply_facies_grouping(real_fac)
            base_fac = self.apply_facies_grouping(base_fac)
            sim_fac = self.apply_facies_grouping(sim_fac)
            if best_fac is not None:
                best_fac = self.apply_facies_grouping(np.asarray(best_fac).astype(int))
        all_facies = sorted(list(set(real_fac) | set(base_fac) | set(sim_fac)))

        # --- Janela ---
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Relatório Poço: {well_name}")
        dialog.resize(1600, 850) 
        dialog.setWindowFlags(dialog.windowFlags() | QtCore.Qt.WindowMinMaxButtonsHint)
        
        main_layout = QtWidgets.QVBoxLayout(dialog)
        tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(tabs)

        # --- Métricas comuns para as abas ---
        r_thick_arr = real_depth - real_depth[0]
        r_total = r_thick_arr[-1] if len(r_thick_arr) > 0 else 0
        b_thick_arr = base_depth - base_depth[0] if len(base_depth) > 0 else np.array([])
        b_total = b_thick_arr[-1] if len(b_thick_arr) > 0 else 0
        s_thick_arr = sim_depth - sim_depth[0] if len(sim_depth) > 0 else np.array([])
        s_total = s_thick_arr[-1] if len(s_thick_arr) > 0 else 0
        g_max = max(r_total, b_total, s_total)

        def draw_log(ax, d_arr, f_arr, title):
            patches = []
            colors = []
            if len(d_arr) < 2:
                return

            curr = f_arr[0]
            top = d_arr[0]

            def add_text(h_blk, t_pos, code):
                if h_blk > (g_max * 0.02):
                    ax.text(
                        0.5, t_pos + h_blk / 2, str(code),
                        ha='center', va='center', fontsize=7, fontweight='bold',
                        color='white' if sum(get_color(code)[:3]) < 1.5 else 'black'
                    )

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
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        def calc_net(d, f):
            if len(d) < 2:
                return {}
            dz = np.diff(d, prepend=d[0])
            dz[0] = 0
            c = {}
            for code in all_facies:
                mask = (f == code)
                c[code] = np.sum(dz[mask])
            return c

        net_base = calc_net(base_depth, base_fac)
        net_sim = calc_net(sim_depth, sim_fac)
        net_real = calc_net(real_depth, real_fac)

        def get_family(f_code):
            s = str(f_code)
            if s.startswith("1"):
                return "Siliciclásticos"
            if s.startswith("2"):
                return "Carbonatos"
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

        families = sorted(list(set(
            list(fam_stats["Real"].keys()) +
            list(fam_stats["Sim"].keys()) +
            list(fam_stats["Base"].keys())
        )))

        bars_b = [(fam_stats["Base"].get(fam, 0) / tot_b) * 100 for fam in families]
        bars_s = [(fam_stats["Sim"].get(fam, 0) / tot_s) * 100 for fam in families]
        bars_r = [(fam_stats["Real"].get(fam, 0) / tot_r) * 100 for fam in families]

        # =================================================================
        # ABA 1: LOGS
        # =================================================================
        tab1 = QtWidgets.QWidget()
        tab1_layout = QtWidgets.QHBoxLayout(tab1)
        tab1_layout.setContentsMargins(6, 6, 6, 6)
        tab1_layout.setSpacing(10)

        # -------------------------------------------------
        # LADO ESQUERDO: LOGS
        # -------------------------------------------------
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # figura mais estreita: assim não sobra aquele branco gigante
        fig1 = plt.figure(figsize=(4.2, 8.6))

        # [left, bottom, width, height]
        # height maior para os poços irem mais para baixo
        ax1 = fig1.add_axes([0.20, 0.05, 0.17, 0.90])  # Base
        ax2 = fig1.add_axes([0.42, 0.05, 0.17, 0.90], sharey=ax1)  # Simul
        ax3 = fig1.add_axes([0.64, 0.05, 0.17, 0.90], sharey=ax1)  # Real

        # Dados de espessura (zero-based)
        r_thick_arr = real_depth - real_depth[0]
        r_total = r_thick_arr[-1] if len(r_thick_arr) > 0 else 0
        b_thick_arr = base_depth - base_depth[0] if len(base_depth) > 0 else np.array([])
        b_total = b_thick_arr[-1] if len(b_thick_arr) > 0 else 0
        s_thick_arr = sim_depth - sim_depth[0] if len(sim_depth) > 0 else np.array([])
        s_total = s_thick_arr[-1] if len(s_thick_arr) > 0 else 0
        g_max = max(r_total, b_total, s_total)

        def draw_log(ax, d_arr, f_arr, title):
            patches = []
            colors = []
            if len(d_arr) < 2:
                return

            curr = f_arr[0]
            top = d_arr[0]

            def add_text(h_blk, t_pos, code):
                if h_blk > (g_max * 0.02):
                    ax.text(
                        0.5, t_pos + h_blk / 2, str(code),
                        ha='center', va='center', fontsize=7, fontweight='bold',
                        color='white' if sum(get_color(code)[:3]) < 1.5 else 'black'
                    )

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
            ax.set_xticks([])

        draw_log(ax1, b_thick_arr, base_fac, f"Base\n{b_total:.1f}m")
        ax1.set_ylabel("Espessura (m)")
        ax1.set_yticks(np.linspace(0, g_max, 10))

        draw_log(ax2, s_thick_arr, sim_fac, f"Simul\n{s_total:.1f}m")
        draw_log(ax3, r_thick_arr, real_fac, f"Real\n{r_total:.1f}m")

        ax2.tick_params(axis='y', left=False, labelleft=False)
        ax3.tick_params(axis='y', left=False, labelleft=False)

        canvas1 = FigureCanvas(fig1)
        left_layout.addWidget(canvas1)

        # -------------------------------------------------
        # LADO DIREITO: TABELAS QT COPIÁVEIS
        # -------------------------------------------------
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QHBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 24, 0, 0)  # ajusta alinhamento vertical
        right_layout.setSpacing(8)
        right_layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        def extract_intervals(d_arr, f_arr, total_thickness):
            """
            Agrupa intervalos contínuos de mesma fácies.
            Retorna: (facies, thickness, pct_col)
            """
            intervals = []
            if len(d_arr) < 2 or len(f_arr) == 0:
                return intervals

            curr = int(f_arr[0])
            top = float(d_arr[0])

            for i in range(1, len(f_arr)):
                fac = int(f_arr[i])
                if fac != curr:
                    base = float(d_arr[i])
                    thick = max(base - top, 0.0)
                    if thick > 1e-9:
                        intervals.append((curr, thick))
                    curr = fac
                    top = base

            base = float(d_arr[-1])
            thick = max(base - top, 0.0)
            if thick > 1e-9:
                intervals.append((curr, thick))

            # merge final caso tenham sobrado fácies iguais adjacentes
            merged = []
            for fac, thick in intervals:
                if merged and merged[-1][0] == fac:
                    merged[-1] = (fac, merged[-1][1] + thick)
                else:
                    merged.append((fac, thick))

            # adiciona %
            out = []
            for fac, thick in merged:
                pct = (thick / total_thickness * 100.0) if total_thickness > 0 else 0.0
                out.append((fac, thick, pct))

            return out
        
        def _prepare_valid_intervals(intervals, min_interval_pct=0.1):
            """
            Mantém só intervalos interpretáveis:
            - facies != 0
            - %Col >= min_interval_pct
            """
            valid = []
            for i, (fac, thick, pct) in enumerate(intervals):
                fac = int(fac)
                pct = float(pct)
                if fac == 0:
                    continue
                if pct < min_interval_pct:
                    continue
                valid.append({
                    "orig_idx": i,
                    "fac": fac,
                    "thick": float(thick),
                    "pct": pct,
                })
            return valid


        def _build_pct_prefix(valid_list):
            pref = [0.0]
            for item in valid_list:
                pref.append(pref[-1] + item["pct"])
            return pref


        def _range_pct(pref, start, length):
            return pref[start + length] - pref[start]


        def _pattern_to_str(pattern):
            return "-".join(str(x) for x in pattern)


        def _calc_simrel(pct_sim, pct_real):
            mx = max(pct_sim, pct_real)
            if mx <= 0:
                return 0.0
            return min(pct_sim, pct_real) / mx


        def _calc_force(pct_sim, pct_real, length):
            """
            Força visual da sequência:
            - tamanho compartilhado
            - similaridade relativa
            - pequeno bônus por comprimento
            """
            pct_comp = min(pct_sim, pct_real)
            simrel = _calc_simrel(pct_sim, pct_real)
            return pct_comp * simrel * (1.0 + 0.15 * max(0, length - 1))


        def _mean_row(orig_rows):
            return sum(orig_rows) / len(orig_rows) if orig_rows else -1.0


        def _generate_candidates(
            sim_valid,
            real_valid,
            level_name,
            pct_min,
            simrel_min,
        ):
            """
            Gera todas as sequências candidatas contíguas e idênticas.
            """
            sim_codes = [x["fac"] for x in sim_valid]
            real_codes = [x["fac"] for x in real_valid]

            sim_pref = _build_pct_prefix(sim_valid)
            real_pref = _build_pct_prefix(real_valid)

            n_sim = len(sim_codes)
            n_real = len(real_codes)

            candidates = []

            for i in range(n_sim):
                for j in range(n_real):
                    L = 0
                    while (
                        i + L < n_sim
                        and j + L < n_real
                        and sim_codes[i + L] == real_codes[j + L]
                    ):
                        L += 1

                        pct_sim = _range_pct(sim_pref, i, L)
                        pct_real = _range_pct(real_pref, j, L)
                        pct_comp = min(pct_sim, pct_real)
                        simrel = _calc_simrel(pct_sim, pct_real)

                        if pct_comp < pct_min:
                            continue
                        if simrel < simrel_min:
                            continue

                        pattern = tuple(sim_codes[i:i + L])
                        sim_orig_rows = [sim_valid[k]["orig_idx"] for k in range(i, i + L)]
                        real_orig_rows = [real_valid[k]["orig_idx"] for k in range(j, j + L)]

                        candidates.append({
                            "level": level_name,
                            "pattern": pattern,
                            "pattern_str": _pattern_to_str(pattern),
                            "length": L,

                            "sim_start": i,
                            "sim_end": i + L - 1,
                            "real_start": j,
                            "real_end": j + L - 1,

                            "sim_orig_rows": sim_orig_rows,
                            "real_orig_rows": real_orig_rows,

                            "pct_sim": pct_sim,
                            "pct_real": pct_real,
                            "pct_comp": pct_comp,
                            "sim_rel": simrel,
                            "force": _calc_force(pct_sim, pct_real, L),

                            # bottom-up na interpretação:
                            # maior índice médio = mais basal
                            "mean_basal": (_mean_row(sim_orig_rows) + _mean_row(real_orig_rows)) / 2.0,
                        })

            return candidates


        def _select_best_ordered_candidates(candidates):
            """
            Seleção global monotônica:
            escolhe o melhor conjunto NÃO sobreposto e NÃO cruzado.

            Critério principal = maior soma de score global.
            """
            if not candidates:
                return []

            # score principal
            for c in candidates:
                c["score"] = c["force"]

            # ordenação para DP
            ordered = sorted(
                candidates,
                key=lambda c: (
                    c["sim_end"],
                    c["real_end"],
                    c["sim_start"],
                    c["real_start"],
                )
            )

            n = len(ordered)
            dp = [0.0] * n
            parent = [-1] * n

            for j in range(n):
                best_prev_val = 0.0
                best_prev_idx = -1

                for i in range(j):
                    compatible = (
                        ordered[i]["sim_end"] < ordered[j]["sim_start"]
                        and ordered[i]["real_end"] < ordered[j]["real_start"]
                    )
                    if compatible and dp[i] > best_prev_val:
                        best_prev_val = dp[i]
                        best_prev_idx = i

                dp[j] = ordered[j]["score"] + best_prev_val
                parent[j] = best_prev_idx

            # melhor final
            best_idx = max(range(n), key=lambda k: dp[k])

            selected = []
            cur = best_idx
            while cur != -1:
                selected.append(ordered[cur])
                cur = parent[cur]

            selected.reverse()
            return selected


        def _split_windows_from_primary(primary_selected, n_sim, n_real):
            """
            Cria janelas entre sequências principais.
            Cada janela é um retângulo monotônico:
            [sim_lo..sim_hi] x [real_lo..real_hi]
            """
            if not primary_selected:
                return [(0, n_sim - 1, 0, n_real - 1)]

            prim = sorted(primary_selected, key=lambda c: c["sim_start"])

            windows = []
            prev_sim_end = -1
            prev_real_end = -1

            for p in prim:
                sim_lo = prev_sim_end + 1
                sim_hi = p["sim_start"] - 1

                real_lo = prev_real_end + 1
                real_hi = p["real_start"] - 1

                if sim_lo <= sim_hi and real_lo <= real_hi:
                    windows.append((sim_lo, sim_hi, real_lo, real_hi))

                prev_sim_end = p["sim_end"]
                prev_real_end = p["real_end"]

            sim_lo = prev_sim_end + 1
            sim_hi = n_sim - 1

            real_lo = prev_real_end + 1
            real_hi = n_real - 1

            if sim_lo <= sim_hi and real_lo <= real_hi:
                windows.append((sim_lo, sim_hi, real_lo, real_hi))

            return windows
        
        def find_shared_sequences(
            sim_intervals,
            real_intervals,
            primary_pct_min=8.0,
            primary_simrel_min=0.50,
            secondary_pct_min=5.0,
            secondary_simrel_min=0.35,
            min_interval_pct=0.1,
        ):
            """
            Sequências compartilhadas:
            1) principais: matching global ordenado
            2) secundárias: matching ordenado nas janelas entre as principais
            """

            sim_valid = _prepare_valid_intervals(sim_intervals, min_interval_pct=min_interval_pct)
            real_valid = _prepare_valid_intervals(real_intervals, min_interval_pct=min_interval_pct)

            sim_labels = [""] * len(sim_intervals)
            real_labels = [""] * len(real_intervals)

            seq_colors = {}
            shared_seq_meta = []

            if not sim_valid or not real_valid:
                return sim_labels, real_labels, seq_colors, shared_seq_meta

            # -------------------------------------------------
            # 1) Sequências principais: matching global
            # -------------------------------------------------
            primary_candidates = _generate_candidates(
                sim_valid,
                real_valid,
                level_name="primary",
                pct_min=primary_pct_min,
                simrel_min=primary_simrel_min,
            )

            primary_selected = _select_best_ordered_candidates(primary_candidates)

            # -------------------------------------------------
            # 2) Sequências secundárias: matching por janelas
            # -------------------------------------------------
            secondary_candidates = _generate_candidates(
                sim_valid,
                real_valid,
                level_name="secondary",
                pct_min=secondary_pct_min,
                simrel_min=secondary_simrel_min,
            )

            windows = _split_windows_from_primary(
                primary_selected,
                n_sim=len(sim_valid),
                n_real=len(real_valid),
            )

            secondary_selected = []
            for sim_lo, sim_hi, real_lo, real_hi in windows:
                cands_in_window = [
                    c for c in secondary_candidates
                    if (
                        sim_lo <= c["sim_start"] <= c["sim_end"] <= sim_hi
                        and real_lo <= c["real_start"] <= c["real_end"] <= real_hi
                    )
                ]
                selected_here = _select_best_ordered_candidates(cands_in_window)
                secondary_selected.extend(selected_here)

            # -------------------------------------------------
            # 3) Junta tudo
            # - Seq. principais primeiro
            # - depois secundárias
            # - dentro de cada grupo, ordena bottom-up para nomear
            # -------------------------------------------------
            primary_selected = sorted(primary_selected, key=lambda c: c["mean_basal"], reverse=True)
            secondary_selected = sorted(secondary_selected, key=lambda c: c["mean_basal"], reverse=True)

            accepted = primary_selected + secondary_selected

            for seq_idx, cand in enumerate(accepted, start=1):
                seq_name = f"S{seq_idx}"
                hue = (seq_idx * 47) % 360
                seq_colors[seq_name] = QColor.fromHsv(hue, 70, 255)

                for row in cand["sim_orig_rows"]:
                    sim_labels[row] = seq_name

                for row in cand["real_orig_rows"]:
                    real_labels[row] = seq_name

                shared_seq_meta.append({
                    "seq": seq_name,
                    "level": cand["level"],

                    "pattern": cand["pattern"],
                    "pattern_str": cand["pattern_str"],
                    "n_intervals": cand["length"],

                    "sim_orig_rows": cand["sim_orig_rows"],
                    "real_orig_rows": cand["real_orig_rows"],

                    "sim_rows": ",".join(str(x + 1) for x in cand["sim_orig_rows"]),
                    "real_rows": ",".join(str(x + 1) for x in cand["real_orig_rows"]),

                    "pct_sim": cand["pct_sim"],
                    "pct_real": cand["pct_real"],
                    "pct_comp": cand["pct_comp"],
                    "sim_rel": cand["sim_rel"],
                    "force": cand["force"],

                    "mean_basal": cand["mean_basal"],
                })

            return sim_labels, real_labels, seq_colors, shared_seq_meta
        
        def build_shared_blocks(
            sim_intervals,
            real_intervals,
            sim_seq_labels,
            real_seq_labels,
            shared_seq_meta,
            gap_pct_max=3.0,
            min_interval_pct=0.1,
        ):
            """
            Agrupa Seq. em Blocos quando:
            - aparecem na mesma ordem bottom-up em Simul e Real
            - o gap entre elas é pequeno nos dois lados
            - facies 0 no gap é ignorada
            """

            block_colors = {}
            block_meta = []

            sim_block_labels = [""] * len(sim_intervals)
            real_block_labels = [""] * len(real_intervals)

            seq_info = {}
            for meta in shared_seq_meta:
                seq = meta["seq"]
                seq_info[seq] = {
                    "sim_rows": sorted(meta["sim_orig_rows"]),
                    "real_rows": sorted(meta["real_orig_rows"]),
                    "pattern_str": meta["pattern_str"],
                    "n_intervals": meta["n_intervals"],
                    "pct_sim": meta["pct_sim"],
                    "pct_real": meta["pct_real"],
                    "level": meta["level"],
                }

            if not seq_info:
                return sim_block_labels, real_block_labels, block_colors, block_meta

            # ordem bottom-up: linhas mais baixas primeiro
            sim_order = sorted(seq_info.keys(), key=lambda s: max(seq_info[s]["sim_rows"]), reverse=True)
            real_order = sorted(seq_info.keys(), key=lambda s: max(seq_info[s]["real_rows"]), reverse=True)

            real_rank = {seq: i for i, seq in enumerate(real_order)}

            def gap_pct(intervals, upper_rows, lower_rows):
                """
                upper_rows = sequência acima
                lower_rows = sequência abaixo
                gap = tudo que está entre elas, ignorando facies 0
                """
                if not upper_rows or not lower_rows:
                    return 999.0

                start = max(upper_rows) + 1
                end = min(lower_rows) - 1

                if end < start:
                    return 0.0

                acc = 0.0
                for k in range(start, end + 1):
                    fac = int(intervals[k][0])
                    pct = float(intervals[k][2])
                    if fac == 0:
                        continue
                    if pct < min_interval_pct:
                        continue
                    acc += pct
                return acc

            groups = []
            current_group = [sim_order[0]]

            for idx in range(1, len(sim_order)):
                prev_seq = current_group[-1]   # mais basal
                next_seq = sim_order[idx]      # mais acima

                # precisa manter a mesma ordem em Real
                if real_rank.get(next_seq, -999) != real_rank.get(prev_seq, -999) + 1:
                    groups.append(current_group)
                    current_group = [next_seq]
                    continue

                prev_sim_rows = seq_info[prev_seq]["sim_rows"]
                next_sim_rows = seq_info[next_seq]["sim_rows"]

                prev_real_rows = seq_info[prev_seq]["real_rows"]
                next_real_rows = seq_info[next_seq]["real_rows"]

                gap_sim = gap_pct(sim_intervals, upper_rows=next_sim_rows, lower_rows=prev_sim_rows)
                gap_real = gap_pct(real_intervals, upper_rows=next_real_rows, lower_rows=prev_real_rows)

                if gap_sim <= gap_pct_max and gap_real <= gap_pct_max:
                    current_group.append(next_seq)
                else:
                    groups.append(current_group)
                    current_group = [next_seq]

            groups.append(current_group)

            block_idx = 1
            for group in groups:
                if len(group) < 2:
                    continue

                block_name = f"B{block_idx}"
                hue = (block_idx * 61) % 360
                block_colors[block_name] = QColor.fromHsv(hue, 45, 255)

                sim_total_pct = 0.0
                real_total_pct = 0.0

                for seq in group:
                    sim_total_pct += seq_info[seq]["pct_sim"]
                    real_total_pct += seq_info[seq]["pct_real"]

                    for row in seq_info[seq]["sim_rows"]:
                        sim_block_labels[row] = block_name

                    for row in seq_info[seq]["real_rows"]:
                        real_block_labels[row] = block_name

                block_meta.append({
                    "block": block_name,
                    "seqs": group[:],
                    "pct_sim": sim_total_pct,
                    "pct_real": real_total_pct,
                })

                block_idx += 1

            return sim_block_labels, real_block_labels, block_colors, block_meta

        def make_table_copyable(table):
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence.Copy, table)

            def copy_selection():
                ranges = table.selectedRanges()
                if not ranges:
                    return

                r = ranges[0]
                lines = []
                for row in range(r.topRow(), r.bottomRow() + 1):
                    vals = []
                    for col in range(r.leftColumn(), r.rightColumn() + 1):
                        item = table.item(row, col)
                        vals.append("" if item is None else item.text())
                    lines.append("\t".join(vals))

                QtWidgets.QApplication.clipboard().setText("\n".join(lines))

            shortcut.activated.connect(copy_selection)

        def build_interval_table(
            title,
            intervals,
            seq_labels=None,
            seq_colors=None,
            block_labels=None,
            block_colors=None,
        ):
            panel = QtWidgets.QWidget()
            panel_layout = QtWidgets.QVBoxLayout(panel)
            panel_layout.setContentsMargins(0, 0, 0, 0)
            panel_layout.setSpacing(4)

            lbl = QtWidgets.QLabel(title)
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet("font-weight: 600;")
            panel_layout.addWidget(lbl)

            has_seq = seq_labels is not None
            has_block = block_labels is not None

            headers = []
            if has_block:
                headers.append("Bl.")
            if has_seq:
                headers.append("Seq.")
            headers += ["Fácies", "Esp. (m)", "% Col."]

            table = QtWidgets.QTableWidget()
            table.setColumnCount(len(headers))
            table.setHorizontalHeaderLabels(headers)
            table.setRowCount(len(intervals))
            table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
            table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
            table.verticalHeader().setVisible(False)
            table.setAlternatingRowColors(False)
            table.setSortingEnabled(False)
            table.setWordWrap(False)
            table.setShowGrid(True)
            table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)

            for row, (fac, thick, pct) in enumerate(intervals):
                col = 0

                if has_block:
                    block_txt = block_labels[row] if row < len(block_labels) else ""
                    block_item = QtWidgets.QTableWidgetItem(block_txt)
                    block_item.setTextAlignment(QtCore.Qt.AlignCenter)

                    if block_txt and block_colors is not None and block_txt in block_colors:
                        block_item.setBackground(QBrush(block_colors[block_txt]))
                        block_item.setForeground(QColor("black"))

                    table.setItem(row, col, block_item)
                    col += 1

                if has_seq:
                    seq_txt = seq_labels[row] if row < len(seq_labels) else ""
                    seq_item = QtWidgets.QTableWidgetItem(seq_txt)
                    seq_item.setTextAlignment(QtCore.Qt.AlignCenter)

                    if seq_txt and seq_colors is not None and seq_txt in seq_colors:
                        seq_item.setBackground(QBrush(seq_colors[seq_txt]))
                        seq_item.setForeground(QColor("black"))

                    table.setItem(row, col, seq_item)
                    col += 1

                fac_item = QtWidgets.QTableWidgetItem(str(fac))
                fac_item.setTextAlignment(QtCore.Qt.AlignCenter)

                rgba = get_color(fac)
                bg = QColor(
                    int(rgba[0] * 255),
                    int(rgba[1] * 255),
                    int(rgba[2] * 255)
                )
                fac_item.setBackground(QBrush(bg))
                fac_item.setForeground(QColor("white" if sum(rgba[:3]) < 1.5 else "black"))

                thick_item = QtWidgets.QTableWidgetItem(f"{thick:.2f}")
                thick_item.setTextAlignment(QtCore.Qt.AlignCenter)

                pct_item = QtWidgets.QTableWidgetItem(f"{pct:.1f}%")
                pct_item.setTextAlignment(QtCore.Qt.AlignCenter)

                table.setItem(row, col + 0, fac_item)
                table.setItem(row, col + 1, thick_item)
                table.setItem(row, col + 2, pct_item)

            header = table.horizontalHeader()
            for c in range(table.columnCount()):
                header.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)

            table.resizeColumnsToContents()
            table.resizeRowsToContents()

            table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

            scroll_w = table.style().pixelMetric(QtWidgets.QStyle.PM_ScrollBarExtent)

            total_w = (
                table.frameWidth() * 2
                + table.verticalHeader().width()
                + sum(table.columnWidth(c) for c in range(table.columnCount()))
                + scroll_w
                + 6
            )

            MAX_TABLE_HEIGHT = 750

            header_h = table.horizontalHeader().height()
            rows_h = sum(table.rowHeight(r) for r in range(table.rowCount()))
            total_h = table.frameWidth() * 2 + header_h + rows_h + 4

            table.setFixedWidth(total_w)
            table.setMaximumHeight(MAX_TABLE_HEIGHT)
            table.setMinimumHeight(min(total_h, MAX_TABLE_HEIGHT))

            make_table_copyable(table)

            panel_layout.addWidget(table, alignment=QtCore.Qt.AlignTop)
            return panel

        base_intervals = extract_intervals(b_thick_arr, base_fac, b_total)
        sim_intervals  = extract_intervals(s_thick_arr, sim_fac, s_total)
        real_intervals = extract_intervals(r_thick_arr, real_fac, r_total)

        sim_seq_labels, real_seq_labels, seq_colors, shared_seq_meta = find_shared_sequences(
            sim_intervals,
            real_intervals,
            primary_pct_min=8.0,
            primary_simrel_min=0.50,
            secondary_pct_min=3.0,
            secondary_simrel_min=0.2,
            min_interval_pct=0.05,
        )

        sim_block_labels, real_block_labels, block_colors, block_meta = build_shared_blocks(
            sim_intervals,
            real_intervals,
            sim_seq_labels,
            real_seq_labels,
            shared_seq_meta,
            gap_pct_max=3.0,
            min_interval_pct=0.1,
        )

        right_layout.addWidget(
            build_interval_table("Base", base_intervals),
            0, QtCore.Qt.AlignTop
        )

        right_layout.addWidget(
            build_interval_table(
                "Simul",
                sim_intervals,
                seq_labels=sim_seq_labels,
                seq_colors=seq_colors,
                block_labels=sim_block_labels,
                block_colors=block_colors,
            ),
            0, QtCore.Qt.AlignTop
        )

        right_layout.addWidget(
            build_interval_table(
                "Real",
                real_intervals,
                seq_labels=real_seq_labels,
                seq_colors=seq_colors,
                block_labels=real_block_labels,
                block_colors=block_colors,
            ),
            0, QtCore.Qt.AlignTop
        )

        right_layout.addStretch(1)

        # -------------------------------------------------
        # Junta tudo
        # -------------------------------------------------
        tab1_layout.setSpacing(8)

        tab1_layout.addWidget(left_panel, 0, QtCore.Qt.AlignTop)
        tab1_layout.addWidget(right_panel, 0, QtCore.Qt.AlignTop)
        tab1_layout.addStretch(1)

        tabs.addTab(tab1, "Logs")

        # =================================================================
        # ABA 2: BALANÇO
        # =================================================================
        tab2 = QtWidgets.QWidget()
        l2 = QtWidgets.QVBoxLayout(tab2)

        fig2, (ax2a, ax2b) = plt.subplots(
            nrows=1, ncols=2, figsize=(13, 6),
            gridspec_kw={'width_ratios': [2.2, 1.2]}
        )

        # --- Balanço volumétrico ---
        y_pos = np.arange(len(all_facies))
        h = 0.25
        vals_b = [net_base.get(f, 0) for f in all_facies]
        vals_s = [net_sim.get(f, 0) for f in all_facies]
        vals_r = [net_real.get(f, 0) for f in all_facies]

        ax2a.barh(y_pos + h, vals_b, h, label='Base', color='#999999')
        ax2a.barh(y_pos,     vals_s, h, label='Simulado', color='#007acc')
        ax2a.barh(y_pos - h, vals_r, h, label='Real', color='#444444')
        ax2a.set_yticks(y_pos)
        ax2a.set_yticklabels([str(f) for f in all_facies])
        ax2a.set_title("Balanço Volumétrico")
        ax2a.legend()
        ax2a.grid(axis='x', linestyle='--', alpha=0.5)

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
                ax2a.text(
                    max_val, y_pos[i], f" {txt}",
                    va='center', color=color, fontsize=8, fontweight='bold'
                )

        # --- Balanço por família ---
        x_fam = np.arange(len(families))
        ax2b.bar(x_fam - 0.2, bars_b, 0.2, label='Base', color='#999999')
        ax2b.bar(x_fam,       bars_s, 0.2, label='Simulado', color='#007acc')
        ax2b.bar(x_fam + 0.2, bars_r, 0.2, label='Real', color='#444444')
        ax2b.set_xticks(x_fam)
        ax2b.set_xticklabels(families)
        ax2b.set_ylabel("Proporção (%)")
        ax2b.set_title("Balanço por Família (%)")
        ax2b.legend()
        ax2b.set_ylim(0, 100)

        plt.tight_layout()
        canvas2 = FigureCanvas(fig2)
        l2.addWidget(canvas2)
        tabs.addTab(tab2, "Balanço")

        # =================================================================
        # ABA 3: MATRIZ DE TROCAS
        # =================================================================
        tab3 = QtWidgets.QWidget()
        l3 = QtWidgets.QVBoxLayout(tab3)

        fig3, ax3a = plt.subplots(figsize=(8, 6))

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

        ax3a.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
        ax3a.set_xticks(np.arange(n_classes))
        ax3a.set_yticks(np.arange(n_classes))
        ax3a.set_xticklabels([str(f) for f in all_facies], rotation=45)
        ax3a.set_yticklabels([str(f) for f in all_facies])
        ax3a.set_xlabel("Simulado")
        ax3a.set_ylabel("Real")
        ax3a.set_title("Matriz de Trocas")

        for i in range(n_classes):
            for j in range(n_classes):
                val = conf_matrix[i, j]
                color = "white" if val > conf_matrix.max() / 2 else "black"
                if val > 0:
                    ax3a.text(j, i, str(val), ha="center", va="center", color=color)
                if i == j:
                    rect = Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor='gold', linewidth=3
                    )
                    ax3a.add_patch(rect)

        plt.tight_layout()
        canvas3 = FigureCanvas(fig3)
        l3.addWidget(canvas3)
        tabs.addTab(tab3, "Matriz de Trocas")

        # =================================================================
        # ABA 4: TABELA DE MÉTRICAS (mantida por enquanto)
        # =================================================================
        tab4 = QtWidgets.QWidget()
        l4 = QtWidgets.QVBoxLayout(tab4)

        table = QtWidgets.QTableWidget()
        cols = ["Fácies", "Real (m)", "Base (m)", "Sim (m)", "Erro Sim/Real (%)"]
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels(cols)
        table.setRowCount(len(all_facies))

        for row, fac in enumerate(all_facies):
            vr = net_real.get(fac, 0)
            vb = net_base.get(fac, 0)
            vs = net_sim.get(fac, 0)

            if vr > 0:
                err_perc = ((vs - vr) / vr) * 100
            else:
                err_perc = 100.0 if vs > 0 else 0.0

            item_fac = QtWidgets.QTableWidgetItem(str(fac))
            item_fac.setTextAlignment(QtCore.Qt.AlignCenter)

            rgba = get_color(fac)
            bg = QColor(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
            item_fac.setBackground(QBrush(bg))
            if sum(rgba[:3]) < 1.5:
                item_fac.setForeground(QColor("white"))

            table.setItem(row, 0, item_fac)
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{vr:.2f}"))
            table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{vb:.2f}"))
            table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{vs:.2f}"))

            item_err = QtWidgets.QTableWidgetItem(f"{err_perc:+.1f}%")
            if abs(err_perc) > 20:
                item_err.setForeground(QColor("red"))
            elif abs(err_perc) < 5:
                item_err.setForeground(QColor("green"))

            table.setItem(row, 4, item_err)

        table.resizeColumnsToContents()
        l4.addWidget(table)
        tabs.addTab(tab4, "Tabela de Métricas")

        # # =================================================================
        # # ABA 5: BEST MATCH
        # # =================================================================
        # tab5 = QtWidgets.QWidget()
        # l5 = QtWidgets.QHBoxLayout(tab5)

        # if best_depth is not None and len(best_depth) > 0:
        #     best_thick_arr = best_depth - best_depth[0]
        #     best_total = best_thick_arr[-1]
        # else:
        #     best_depth, best_fac = sim_depth, sim_fac
        #     best_thick_arr, best_total = s_thick_arr, s_total

        # fig5, axs5 = plt.subplots(1, 4, figsize=(6, 8), sharey=True)
        # fig5.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.05, wspace=0.3)

        # def group_layers(depth, facies, is_grid_format=True):
        #     if len(depth) == 0:
        #         return []

        #     raw_blocks = []
        #     if is_grid_format:
        #         for k in range(0, len(depth) - 1, 2):
        #             raw_blocks.append((depth[k], depth[k + 1], int(facies[k])))
        #     else:
        #         curr = int(facies[0])
        #         top = depth[0]
        #         for k in range(1, len(facies)):
        #             if int(facies[k]) != curr:
        #                 raw_blocks.append((top, depth[k], curr))
        #                 top = depth[k]
        #                 curr = int(facies[k])
        #         raw_blocks.append((top, depth[-1], curr))

        #     if not raw_blocks:
        #         return []

        #     merged = []
        #     curr_top, curr_base, curr_fac = raw_blocks[0]

        #     for i in range(1, len(raw_blocks)):
        #         next_top, next_base, next_fac = raw_blocks[i]
        #         if next_fac == curr_fac and abs(next_top - curr_base) < 0.05:
        #             curr_base = next_base
        #         else:
        #             merged.append((curr_top, curr_base, curr_fac))
        #             curr_top, curr_base, curr_fac = next_top, next_base, next_fac

        #     merged.append((curr_top, curr_base, curr_fac))
        #     return merged

        # def plot_track(ax, d, f, title, is_grid=True):
        #     ax.set_title(title, fontsize=8, pad=8)
        #     ax.set_xticks([])
        #     ax.set_facecolor('white')

        #     d_rel = d - d[0] if len(d) > 0 else []
        #     layers = group_layers(d_rel, f, is_grid)

        #     max_y = max(b_total, s_total, best_total, r_total)
        #     ax.set_ylim(max_y, 0)

        #     for top, base, fac in layers:
        #         h = base - top
        #         if h <= 0:
        #             continue
        #         rect = Rectangle((0, top), 1, h, facecolor=get_color(fac), edgecolor='none')
        #         ax.add_patch(rect)

        #         if h > max_y * 0.03:
        #             lum = sum(get_color(fac)[:3])
        #             txt_c = 'white' if lum < 1.5 else 'black'
        #             ax.text(
        #                 0.5, top + h / 2, str(fac),
        #                 ha='center', va='center', fontsize=6,
        #                 color=txt_c, fontweight='bold'
        #             )

        # plot_track(axs5[0], base_depth, base_fac, f"BASE\n{b_total:.1f}m", True)
        # plot_track(axs5[1], sim_depth, sim_fac, f"SIM (Orig)\n{s_total:.1f}m", True)
        # plot_track(axs5[2], best_depth, best_fac, f"SIM ({window_size_str})\n{best_total:.1f}m", True)
        # plot_track(axs5[3], real_depth, real_fac, f"REAL\n{r_total:.1f}m", False)

        # for ax in axs5[1:]:
        #     ax.set_yticks([])
        # axs5[0].set_ylabel("Espessura Relativa (m)", fontsize=9)

        # canvas5 = FigureCanvas(fig5)
        # l5.addWidget(canvas5)
        # tabs.addTab(tab5, "Best Match")

        # =================================================================
        # ABA 5: CURVAS LAS (lado a lado + escala horizontal/vertical)
        # =================================================================
        if well_logs_df is not None and not well_logs_df.empty and "DEPT" in well_logs_df.columns:
            import pandas as pd
            import matplotlib.pyplot as plt

            tab5 = QtWidgets.QWidget()
            l5 = QtWidgets.QVBoxLayout(tab5)
            l5.setContentsMargins(6, 6, 6, 6)
            l5.setSpacing(6)

            # -------------------------------------------------
            # CONTROLES
            # -------------------------------------------------
            controls = QtWidgets.QHBoxLayout()
            controls.setContentsMargins(0, 0, 0, 0)

            lbl_x = QtWidgets.QLabel("Escala horizontal:")
            spin_x = QtWidgets.QDoubleSpinBox()
            spin_x.setRange(0.5, 5.0)
            spin_x.setSingleStep(0.25)
            spin_x.setDecimals(2)
            spin_x.setValue(1.0)

            lbl_y = QtWidgets.QLabel("Escala vertical:")
            spin_y = QtWidgets.QDoubleSpinBox()
            spin_y.setRange(0.5, 6.0)
            spin_y.setSingleStep(0.25)
            spin_y.setDecimals(2)
            spin_y.setValue(1.0)

            btn_reset = QtWidgets.QPushButton("Reset")
            btn_fit = QtWidgets.QPushButton("Ajustar")

            controls.addWidget(lbl_x)
            controls.addWidget(spin_x)
            controls.addSpacing(16)
            controls.addWidget(lbl_y)
            controls.addWidget(spin_y)
            controls.addSpacing(16)
            controls.addWidget(btn_reset)
            controls.addWidget(btn_fit)
            controls.addStretch()

            l5.addLayout(controls)

            # -------------------------------------------------
            # ÁREA COM SCROLL
            # -------------------------------------------------
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(False)
            scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

            container = QtWidgets.QWidget()
            container_layout = QtWidgets.QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(0)

            l5.addWidget(scroll)

            render_state = {
                "canvas": None,
                "fig": None,
            }

            def _iter_las_curve_columns(df):
                skip = {
                    "DEPT", "X", "Y", "Z",
                    "MD", "TVD", "DX", "DY", "AZIM", "INCL", "DLS"
                }

                cols = []
                for c in df.columns:
                    if c in skip:
                        continue

                    s = pd.to_numeric(df[c], errors="coerce")
                    if s.notna().sum() < 2:
                        continue

                    cols.append(c)

                return cols

            def _is_discrete_curve(arr):
                arr = np.asarray(arr, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    return False

                unique_vals = np.unique(arr)
                is_integer_like = np.all(np.abs(arr - np.round(arr)) < 1e-6)

                return is_integer_like and unique_vals.size <= 25

            def _curve_stats_text(arr):
                arr = np.asarray(arr, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    return "Sem dados válidos"
                return f"min={arr.min():.3g}   max={arr.max():.3g}"

            def render_las_tracks():
                # limpa canvas anterior
                if render_state["canvas"] is not None:
                    container_layout.removeWidget(render_state["canvas"])
                    render_state["canvas"].setParent(None)
                    render_state["canvas"].deleteLater()
                    render_state["canvas"] = None

                if render_state["fig"] is not None:
                    plt.close(render_state["fig"])
                    render_state["fig"] = None

                curve_cols = _iter_las_curve_columns(well_logs_df)

                if not curve_cols:
                    lbl = QtWidgets.QLabel("Nenhuma curva LAS numérica válida encontrada.")
                    lbl.setAlignment(QtCore.Qt.AlignCenter)
                    container_layout.addWidget(lbl)
                    container.resize(900, 300)
                    scroll.setWidget(container)
                    return

                n_tracks = len(curve_cols)

                # escalas controladas pelo usuário
                sx = float(spin_x.value())
                sy = float(spin_y.value())

                # tamanho base
                base_track_w = 2.2
                base_fig_h = 8.8

                track_w = base_track_w * sx
                fig_w = max(8.0, n_tracks * track_w)
                fig_h = base_fig_h * sy

                fig, axs = plt.subplots(
                    1, n_tracks,
                    figsize=(fig_w, fig_h),
                    sharey=True
                )

                if n_tracks == 1:
                    axs = [axs]

                depth_all = pd.to_numeric(
                    well_logs_df["DEPT"], errors="coerce"
                ).to_numpy(dtype=float)

                y_min = np.nanmin(depth_all) if np.isfinite(depth_all).any() else 0.0
                y_max = np.nanmax(depth_all) if np.isfinite(depth_all).any() else 1.0

                for i, col in enumerate(curve_cols):
                    ax = axs[i]

                    d = pd.to_numeric(
                        well_logs_df["DEPT"], errors="coerce"
                    ).to_numpy(dtype=float)

                    v = pd.to_numeric(
                        well_logs_df[col], errors="coerce"
                    ).to_numpy(dtype=float)

                    mask = np.isfinite(d) & np.isfinite(v)

                    if np.count_nonzero(mask) >= 2:
                        d = d[mask]
                        v = v[mask]

                        if _is_discrete_curve(v):
                            ax.step(v, d, where="post", linewidth=1.2)
                            ax.plot(v, d, "|", markersize=6)
                            uniq = np.unique(v)
                            if uniq.size <= 12:
                                ax.set_xticks(uniq)
                        else:
                            ax.plot(v, d, linewidth=1.2)

                        ax.grid(True, linestyle="--", alpha=0.35)
                        ax.margins(x=0.08)
                    else:
                        ax.text(
                            0.5, 0.5, "Sem dados válidos",
                            ha="center", va="center",
                            transform=ax.transAxes
                        )

                    ax.set_ylim(y_max, y_min)  # profundidade crescente para baixo
                    ax.set_xlabel(col, fontsize=9)

                    title = col
                    if col == real_log_col:
                        title += "\n(Real)"
                    else:
                        title += f"\n{_curve_stats_text(v)}"
                    ax.set_title(title, fontsize=9)

                    if i == 0:
                        ax.set_ylabel("MD (m)")
                    else:
                        ax.tick_params(axis="y", left=False, labelleft=False)

                fig.tight_layout(w_pad=1.2)

                canvas = FigureCanvas(fig)

                # tamanho visual do widget no Qt
                canvas_w = int(240 * n_tracks * sx)
                canvas_h = int(820 * sy)

                canvas.setMinimumSize(canvas_w, canvas_h)

                container_layout.addWidget(canvas)
                container.resize(canvas_w + 20, canvas_h + 20)
                scroll.setWidget(container)

                render_state["canvas"] = canvas
                render_state["fig"] = fig

            def reset_las_scales():
                spin_x.setValue(1.0)
                spin_y.setValue(1.0)

            def fit_las_scales():
                curve_cols = _iter_las_curve_columns(well_logs_df)
                n_tracks = max(1, len(curve_cols))

                # tenta deixar razoável dentro da janela
                sx = min(1.25, max(0.65, 6.0 / n_tracks))
                spin_x.setValue(sx)
                spin_y.setValue(1.0)

            spin_x.valueChanged.connect(lambda *_: render_las_tracks())
            spin_y.valueChanged.connect(lambda *_: render_las_tracks())
            btn_reset.clicked.connect(reset_las_scales)
            btn_fit.clicked.connect(fit_las_scales)

            render_las_tracks()

            tabs.addTab(tab5, "Curvas LAS")

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
        t_min=0.30,
        ignore_real_zeros=True,
    ):
        """Calcula ranking dos modelos vs poços usando SCORE POR PROPORÇÃO.

        - Para cada poço: calcula proporções por espessura das fácies (com suavização t_min)
          e compara REAL vs SIM por distância L1 (score=1-0.5*L1).
        - Para janela espacial NxN: escolhe a coluna (i,j) com melhor score.

        Peso do score do modelo:
          - Usa `t_real_valid` (espessura total válida após suavização) como peso.
        """
        import numpy as np
        from analysis import compute_well_match_score
        from load_data import grid as global_grid, facies as global_facies

        if not self.wells:
            return []

        # 1) Poços
        if well_names is None:
            well_names = list(self.wells.keys())
        else:
            well_names = [w for w in well_names if w in self.wells]
        if not well_names:
            return []

        # 2) Modelos
        if model_keys is None:
            model_keys = list(self.models.keys())
            if "base" in self.models and "base" not in model_keys:
                model_keys.append("base")
        model_keys = [k for k in model_keys if k in self.models]
        if not model_keys:
            return []

        # window_size ímpar
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

            # Resolve grid/facies do base
            g = m.get("grid", None)
            fac = m.get("facies", None)
            if mk == "base":
                if g is None:
                    g = global_grid
                if fac is None:
                    fac = global_facies
            if g is None:
                continue

            # injeta facies no grid (consistência)
            if fac is not None:
                try:
                    current_f = g.cell_data.get("Facies")
                    if current_f is None or current_f is not fac:
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

                if "DEPT" not in well.data.columns:
                    continue

                # coluna de fácies do poço
                if "fac" in well.data.columns:
                    col_real = "fac"
                elif "lito_upscaled" in well.data.columns:
                    col_real = "lito_upscaled"
                else:
                    continue

                full_depth = well.data["DEPT"].to_numpy(dtype=float)
                full_real = well.data[col_real].to_numpy(dtype=float)

                key = str(wn).strip()
                markers = self.markers_db.get(key, [])

                real_depth = full_depth
                real_fac = np.where(np.isfinite(full_real), full_real, 0.0).astype(int)

                # recorte por marcadores (Top->Base)
                if markers:
                    mds = sorted([mm.get("md") for mm in markers if mm.get("md") is not None])
                    if len(mds) >= 2:
                        top_md, base_md = float(mds[0]), float(mds[-1])
                        dmin, dmax = float(full_depth.min()), float(full_depth.max())
                        if (top_md <= dmax + 1e-6) and (base_md >= dmin - 1e-6) and (base_md > top_md):
                            mask = (full_depth >= top_md) & (full_depth <= base_md)
                            if np.any(mask):
                                real_depth = full_depth[mask]
                                real_fac = real_fac[mask]

                # aplica agrupamento no REAL quando o toggle estiver ligado
                if bool(getattr(self, "use_facies_grouping", False)):
                    try:
                        real_fac = self.apply_facies_grouping(real_fac)
                    except Exception:
                        pass

                # (X,Y) de referência do poço
                xy = self._pick_reference_xy_for_well_report(well, markers)
                if xy is None:
                    continue
                xref, yref = xy

                # --- cálculo (1x1 ou NxN) ---
                if window_size == 1:
                    sim_depth, sim_fac, _ = self._column_profile_from_grid(g, xref, yref)
                    # aplica agrupamento no SIM quando o toggle estiver ligado
                    if bool(getattr(self, "use_facies_grouping", False)):
                        try:
                            sim_fac = self.apply_facies_grouping(sim_fac)
                        except Exception:
                            pass
                    if sim_depth is None or len(sim_depth) < 2:
                        continue

                    s = compute_well_match_score(
                        real_depth, real_fac,
                        sim_depth, sim_fac,
                        t_min=t_min,
                        ignore_real_zeros=ignore_real_zeros,
                    )
                    s["best_i"], s["best_j"] = self._get_ij_from_xy(g, xref, yref)

                else:
                    sim_depth, sim_fac, sim_total, i_best, j_best, s = self._best_profile_score_in_window(
                        g,
                        xref, yref,
                        real_depth, real_fac,
                        window_size=window_size,
                        t_min=t_min,
                        ignore_real_zeros=ignore_real_zeros,
                    )
                    if sim_depth is None or len(sim_depth) < 2:
                        continue

                    s = dict(s)
                    s["best_i"] = i_best
                    s["best_j"] = j_best

                # peso = espessura válida do REAL
                weight = float(s.get("t_real_valid", 0.0))
                if not np.isfinite(weight) or weight <= 0.0:
                    continue

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
                "t_min": float(t_min),
            })

        results.sort(key=lambda d: d["score"], reverse=True)
        return results
    def show_models_well_fit_ranking(self):
        """Abre o ranking de modelos vs poços usando o NOVO score por proporção (sem bin-a-bin / sem kappa)."""
        from PyQt5 import QtWidgets

        ws = int(getattr(self, "well_rank_window_size", 1) or 1)
        tmin = float(getattr(self, "well_rank_t_min", 0.30) or 0.30)

        ranking = self.evaluate_models_against_wells(
            window_size=ws,
            t_min=tmin,
            ignore_real_zeros=True,
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
        """Janela de ranking com o NOVO score por proporções (com espessura mínima t_min)."""
        from PyQt5 import QtWidgets, QtCore

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Avaliação dos modelos vs poços (proporções)")
        dlg.setMinimumSize(1100, 680)

        dlg._ranking = ranking

        layout = QtWidgets.QVBoxLayout(dlg)

        # --- Top bar ---
        top_bar = QtWidgets.QHBoxLayout()

        lbl = QtWidgets.QLabel("Clique em um modelo para ver o detalhe por poço.")
        top_bar.addWidget(lbl)

        top_bar.addSpacing(14)
        top_bar.addWidget(QtWidgets.QLabel("Janela:"))
        cmb_window = QtWidgets.QComboBox()
        cmb_window.addItems(["1x1", "3x3", "5x5", "7x7", "9x9"])
        top_bar.addWidget(cmb_window)

        top_bar.addSpacing(14)
        top_bar.addWidget(QtWidgets.QLabel("t_min (m):"))
        sp_tmin = QtWidgets.QDoubleSpinBox()
        sp_tmin.setDecimals(2)
        sp_tmin.setSingleStep(0.05)
        sp_tmin.setRange(0.00, 50.0)
        sp_tmin.setValue(float(getattr(self, "well_rank_t_min", 0.30) or 0.30))
        top_bar.addWidget(sp_tmin)

        top_bar.addStretch(1)

        btn_copy = QtWidgets.QPushButton("Copiar tabela (modelos)")
        btn_copy.clicked.connect(lambda: self._copy_models_table_to_clipboard(dlg))
        top_bar.addWidget(btn_copy)

        layout.addLayout(top_bar)

        # --- Horizontal splitter: (tables) | (overview plot) ---
        sp_h = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(sp_h)

        # Left: vertical splitter (modelos / poços)
        left = QtWidgets.QWidget()
        l_left = QtWidgets.QVBoxLayout(left)
        l_left.setContentsMargins(0, 0, 0, 0)

        sp_v = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        l_left.addWidget(sp_v)

        # Tabela modelos
        tbl_models = QtWidgets.QTableWidget()
        tbl_models.setColumnCount(5)
        tbl_models.setHorizontalHeaderLabels(["Rank", "Modelo", "Score", "ΣT_real (m)", "Poços"])
        tbl_models.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        tbl_models.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        tbl_models.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        tbl_models.setSortingEnabled(True)
        tbl_models.horizontalHeader().setStretchLastSection(True)
        tbl_models.verticalHeader().setVisible(False)
        sp_v.addWidget(tbl_models)

        # Tabela poços
        tbl_wells = QtWidgets.QTableWidget()
        tbl_wells.setColumnCount(6)
        tbl_wells.setHorizontalHeaderLabels(["Poço", "Score", "T_real (m)", "T_sim (m)", "ΔT (m)", "D_prop"])
        tbl_wells.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        tbl_wells.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        tbl_wells.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        tbl_wells.setSortingEnabled(True)
        tbl_wells.horizontalHeader().setStretchLastSection(True)
        tbl_wells.verticalHeader().setVisible(False)
        sp_v.addWidget(tbl_wells)

        sp_h.addWidget(left)

        # Right: overview (matplotlib se disponível)
        right = QtWidgets.QWidget()
        l_right = QtWidgets.QVBoxLayout(right)
        l_right.setContentsMargins(6, 6, 6, 6)

        lbl_plot = QtWidgets.QLabel("Visão geral (REAL x SIM) — runs após suavização")
        lbl_plot.setAlignment(QtCore.Qt.AlignLeft)
        l_right.addWidget(lbl_plot)

        dlg._has_mpl = False
        dlg._fig = None
        dlg._ax = None
        dlg._canvas = None

        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            dlg._fig = Figure(figsize=(5, 4))
            dlg._ax = dlg._fig.add_subplot(111)
            dlg._canvas = FigureCanvas(dlg._fig)
            l_right.addWidget(dlg._canvas, 1)
            dlg._has_mpl = True
        except Exception:
            info = QtWidgets.QLabel("Matplotlib não disponível no ambiente.\n(sem gráfico de visão geral)")
            info.setAlignment(QtCore.Qt.AlignCenter)
            l_right.addWidget(info, 1)

        sp_h.addWidget(right)
        sp_h.setStretchFactor(0, 3)
        sp_h.setStretchFactor(1, 2)

        # guarda refs no dialog
        dlg._tbl_models = tbl_models
        dlg._tbl_wells = tbl_wells
        dlg._cmb_rank_window = cmb_window
        dlg._sp_tmin = sp_tmin

        # seta combo para o window_size atual
        ws = int(getattr(self, "well_rank_window_size", 1) or 1)
        if ws not in (1, 3, 5, 7, 9):
            ws = 1
        cmb_window.setCurrentText(f"{ws}x{ws}")

        # recompute
        def _recompute():
            try:
                ws2 = int(cmb_window.currentText().split("x")[0])
            except Exception:
                ws2 = 1

            self.well_rank_window_size = ws2
            self.well_rank_t_min = float(sp_tmin.value())

            new_ranking = self.evaluate_models_against_wells(
                window_size=ws2,
                t_min=float(sp_tmin.value()),
                ignore_real_zeros=True,
            )
            dlg._ranking = new_ranking
            self._populate_models_ranking_table(dlg)
            if tbl_models.rowCount() > 0:
                tbl_models.selectRow(0)

        # populate initial
        self._populate_models_ranking_table(dlg)

        # selection -> detail
        tbl_models.itemSelectionChanged.connect(lambda: self._on_models_table_selection_changed_dialog(dlg))

        cmb_window.currentIndexChanged.connect(lambda *_: _recompute())
        sp_tmin.valueChanged.connect(lambda *_: _recompute())

        if tbl_models.rowCount() > 0:
            tbl_models.selectRow(0)

        dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        dlg.show()
        self.open_reports.append(dlg)
    def _populate_models_ranking_table(self, dlg):
        from PyQt5 import QtWidgets, QtCore
        ranking = getattr(dlg, "_ranking", [])
        tbl = dlg._tbl_models
        tbl.setRowCount(0)

        for i, r in enumerate(ranking, start=1):
            row = tbl.rowCount()
            tbl.insertRow(row)

            # Rank (guarda model_key)
            it_rank = QtWidgets.QTableWidgetItem(f"{i:02d}")
            it_rank.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            it_rank.setData(QtCore.Qt.UserRole, r.get("model_key"))
            tbl.setItem(row, 0, it_rank)

            # Modelo
            tbl.setItem(row, 1, QtWidgets.QTableWidgetItem(str(r.get("model_name", ""))))

            # Score
            it_score = QtWidgets.QTableWidgetItem(f"{float(r.get('score', 0.0)):.3f}")
            it_score.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            tbl.setItem(row, 2, it_score)

            # ΣT_real (m) — soma das espessuras válidas dos poços
            details = r.get("details", {}) or {}
            tsum = 0.0
            for _, s in details.items():
                tsum += float(s.get("t_real_valid", s.get("t_real", 0.0)) or 0.0)

            it_t = QtWidgets.QTableWidgetItem(f"{tsum:.2f}")
            it_t.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            tbl.setItem(row, 3, it_t)

            # Poços
            tbl.setItem(row, 4, QtWidgets.QTableWidgetItem(str(r.get("n_wells_used", 0))))

        tbl.resizeColumnsToContents()
    def _on_models_table_selection_changed_dialog(self, dlg):
        """Callback do diálogo: usuário selecionou um modelo."""
        from PyQt5 import QtCore

        tbl = dlg._tbl_models
        row = tbl.currentRow()
        if row < 0:
            return

        it_rank = tbl.item(row, 0)
        if it_rank is None:
            return
        model_key = it_rank.data(QtCore.Qt.UserRole)

        ranking = getattr(dlg, "_ranking", [])
        rec = None
        for r in ranking:
            if str(r.get("model_key")) == str(model_key):
                rec = r
                break
        if rec is None:
            return

        self._populate_wells_detail_table_dialog(dlg, rec)
        self._update_wells_overview_plot_dialog(dlg, rec)

    def _populate_wells_detail_table_dialog(self, dlg, model_record):
        from PyQt5 import QtWidgets, QtCore

        tbl = dlg._tbl_wells
        tbl.setRowCount(0)

        details = model_record.get("details", {}) or {}
        items = list(details.items())
        items.sort(key=lambda kv: float(kv[1].get("score", 0.0)), reverse=True)

        for well_name, s in items:
            row = tbl.rowCount()
            tbl.insertRow(row)

            score = float(s.get("score", 0.0))
            t_real = float(s.get("t_real_valid", s.get("t_real", 0.0)) or 0.0)
            t_sim = float(s.get("t_sim_valid", s.get("t_sim", 0.0)) or 0.0)
            dt = t_sim - t_real
            dprop = float(s.get("prop_distance", 0.0))

            tbl.setItem(row, 0, QtWidgets.QTableWidgetItem(str(well_name)))

            it_s = QtWidgets.QTableWidgetItem(f"{score:.3f}")
            it_s.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            tbl.setItem(row, 1, it_s)

            it_tr = QtWidgets.QTableWidgetItem(f"{t_real:.2f}")
            it_tr.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            tbl.setItem(row, 2, it_tr)

            it_ts = QtWidgets.QTableWidgetItem(f"{t_sim:.2f}")
            it_ts.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            tbl.setItem(row, 3, it_ts)

            it_dt = QtWidgets.QTableWidgetItem(f"{dt:+.2f}")
            it_dt.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            tbl.setItem(row, 4, it_dt)

            it_dp = QtWidgets.QTableWidgetItem(f"{dprop:.3f}")
            it_dp.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            tbl.setItem(row, 5, it_dp)

        tbl.resizeColumnsToContents()

    def _update_wells_overview_plot_dialog(self, dlg, model_record):
        """Desenha uma visão geral simples de runs REAL e SIM (após suavização)."""
        if not getattr(dlg, "_has_mpl", False):
            return
        fig = dlg._fig
        ax = dlg._ax
        canvas = dlg._canvas
        if fig is None or ax is None or canvas is None:
            return

        ax.clear()

        details = model_record.get("details", {}) or {}
        well_items = list(details.items())
        well_items.sort(key=lambda kv: float(kv[1].get("score", 0.0)), reverse=True)

        colors = getattr(self, "facies_colors_dict", {}) or {}

        y = 0.0
        yticks = []
        ylabels = []

        max_t = 0.0
        for _, s in well_items:
            max_t = max(max_t,
                        float(s.get("t_real_valid", 0.0) or 0.0),
                        float(s.get("t_sim_valid", 0.0) or 0.0))
        max_t = max(max_t, 1.0)

        for wname, s in well_items[:30]:
            runs_sim = s.get("runs_sim", []) or []
            runs_real = s.get("runs_real", []) or []

            def draw_runs(runs, y0, side_label):
                x0 = 0.0
                for fac, t in runs:
                    fac = int(fac)
                    t = float(t)
                    if t <= 0:
                        continue
                    rgba = colors.get(fac, (0.7, 0.7, 0.7, 1.0))
                    ax.broken_barh([(x0, t)], (y0, 0.35), facecolors=[rgba], edgecolors='none')
                    x0 += t
                ax.text(max_t * 1.01, y0 + 0.17, side_label, va='center', fontsize=8)

            draw_runs(runs_sim, y, "SIM")
            draw_runs(runs_real, y + 0.45, "REAL")

            yticks.append(y + 0.225)
            ylabels.append(str(wname))
            y += 1.05

        ax.set_xlim(0, max_t * 1.15)
        ax.set_ylim(-0.1, y)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_xlabel("Espessura acumulada (m)")
        ax.set_title("Runs por poço (após t_min)")
        ax.grid(False)

        fig.tight_layout()
        canvas.draw_idle()

    def _on_models_table_selection_changed(self):
        # Quando seleciona um modelo, atualiza detalhamento por poço e a visão geral
        if not hasattr(self, "_current_ranking_data") or not self._current_ranking_data:
            return

        sel = self.tbl_models.selectedItems()
        if not sel:
            return

        row = sel[0].row()
        it_rank = self.tbl_models.item(row, 0)
        if it_rank is None:
            return

        model_key = it_rank.data(QtCore.Qt.UserRole)

        model_record = None
        for r in self._current_ranking_data:
            if r.get("model_key") == model_key:
                model_record = r
                break

        if model_record is None:
            return

        self._populate_wells_detail_table(model_record)
        self._update_ranking_overview_plot(model_record)

    def _populate_wells_detail_table(self, model_record):
        """Preenche tabela de poços para o modelo selecionado (score por proporção)."""
        details = model_record.get("details", {}) or {}
        model_key = model_record.get("model_key")

        self.tbl_wells.setSortingEnabled(False)
        self.tbl_wells.setRowCount(0)

        for wn, s in details.items():
            row = self.tbl_wells.rowCount()
            self.tbl_wells.insertRow(row)

            score = float(s.get("score", 0.0) or 0.0)
            dprop = float(s.get("prop_distance", 0.0) or 0.0)
            t_real = float(s.get("t_real_valid", s.get("t_real", 0.0)) or 0.0)
            t_sim = float(s.get("t_sim_valid", s.get("t_sim", 0.0)) or 0.0)
            dt = t_sim - t_real

            self.tbl_wells.setItem(row, 0, QtWidgets.QTableWidgetItem(str(wn)))

            it_sc = QtWidgets.QTableWidgetItem(f"{score:.3f}")
            it_sc.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            self.tbl_wells.setItem(row, 1, it_sc)

            it_d = QtWidgets.QTableWidgetItem(f"{dprop:.3f}")
            it_d.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            self.tbl_wells.setItem(row, 2, it_d)

            it_tr = QtWidgets.QTableWidgetItem(f"{t_real:.2f}")
            it_tr.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            self.tbl_wells.setItem(row, 3, it_tr)

            it_ts = QtWidgets.QTableWidgetItem(f"{t_sim:.2f}")
            it_ts.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            self.tbl_wells.setItem(row, 4, it_ts)

            it_dt = QtWidgets.QTableWidgetItem(f"{dt:+.2f}")
            it_dt.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            self.tbl_wells.setItem(row, 5, it_dt)

            # Ações: Relatório e (opcional) janela/best match
            widget = QtWidgets.QWidget()
            h_lay = QtWidgets.QHBoxLayout(widget)
            h_lay.setContentsMargins(0, 0, 0, 0)

            btn_rep = QtWidgets.QPushButton("Relatório")
            btn_rep.setToolTip("Abrir relatório do poço (Base/Sim/Real)")
            btn_rep.clicked.connect(lambda _=False, name=str(wn), mk=model_key: self.show_well_comparison_report(name, mk))
            h_lay.addWidget(btn_rep)

            btn_best = QtWidgets.QPushButton("Janela")
            btn_best.setToolTip("Abrir diagnóstico do best-match (janela NxN)")
            bi = s.get("best_i")
            bj = s.get("best_j")
            btn_best.setEnabled(bi is not None and bj is not None)
            btn_best.clicked.connect(lambda _=False, name=str(wn), ii=bi, jj=bj, mk=model_key: self.open_advanced_rank_report(model_key=mk, well_name=name, best_i=ii, best_j=jj))
            h_lay.addWidget(btn_best)

            h_lay.addStretch(1)
            self.tbl_wells.setCellWidget(row, 6, widget)

        self.tbl_wells.setSortingEnabled(True)
        self.tbl_wells.resizeColumnsToContents()

    def _best_profile_score_in_window(
        self,
        grid,
        xref, yref,
        real_depth, real_fac,
        *,
        window_size=1,     # 1,3,5,7...
        t_min=0.30,
        ignore_real_zeros=True,
        **_ignored,
    ):
        """Retorna o melhor match REAL vs pseudo-poço do grid em uma janela NxN (score por proporção)."""
        import numpy as np
        from analysis import compute_well_match_score

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
                # aplica agrupamento no SIM quando o toggle estiver ligado
                if bool(getattr(self, "use_facies_grouping", False)):
                    try:
                        sim_fac = self.apply_facies_grouping(sim_fac)
                    except Exception:
                        pass
                if sim_depth is None or len(sim_depth) < 2:
                    continue

                fit = compute_well_match_score(
                    real_depth, real_fac,
                    sim_depth, sim_fac,
                    t_min=t_min,
                    ignore_real_zeros=ignore_real_zeros,
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
    
    def _get_union_grid_property_names(self):
        """
        Retorna união de propriedades (cell_data keys) entre:
        - em visualização: grid ativo
        - em comparação: grids ativos da comparação (active_comp_states)
        """
        grids = []

        # Comparação ativa (central_stack index 1) -> união
        try:
            if hasattr(self, "central_stack") and self.central_stack.currentIndex() == 1 and getattr(self, "active_comp_states", None):
                for st in self.active_comp_states:
                    g = st.get("current_grid_source")
                    if g is not None:
                        grids.append(g)
            else:
                g = self.state.get("current_grid_source")
                if g is None and "base" in getattr(self, "models", {}):
                    g = self.models["base"].get("grid")
                if g is not None:
                    grids.append(g)
        except Exception:
            pass

        all_keys = set()
        for g in grids:
            try:
                all_keys |= set(g.cell_data.keys())
            except Exception:
                pass

        # Mesma limpeza que você já faz no menu
        exact_ignore = {
            "vtkOriginalCellIds", "vtkOriginalPointIds",
            "Facies", "facies", "Entropy", "Texture Coordinates",
            "StratigraphicThickness", "cell_thickness",
            "Reservoir", "reservoir", "Clusters", "clusters",
            "LargestCluster", "Volume", "NTG_local"
        }

        cleaned = []
        for name in all_keys:
            if name in exact_ignore:
                continue
            if str(name).endswith("_index"):
                continue
            if str(name).startswith("vert_"):
                continue
            if "Ghost" in str(name):
                continue
            cleaned.append(str(name))

        return sorted(cleaned, key=lambda s: s.lower())


    def open_proportion_props_dialog(self):
        """
        Abre diálogo para usuário escolher quais propriedades devem ser tratadas como proporção (0–1).
        Armazena em self.state['fraction_props'] (set de nomes).
        """
        props = self._get_union_grid_property_names()
        current = set(self.state.get("fraction_props", set()) or set())

        dlg = ProportionPropsDialog(props, current_set=current, parent=self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        self.state["fraction_props"] = set(dlg.get_selected())

        # Se estiver em comparação, reaplica escala global imediatamente
        try:
            if hasattr(self, "central_stack") and self.central_stack.currentIndex() == 1:
                # 3D comparação
                if hasattr(self, "active_comp_states") and self.active_comp_states:
                    self._apply_global_clim_to_active_comparison()
                # 2D comparação
                if hasattr(self, "compare_stack") and self.compare_stack.currentIndex() == 2:
                    self.update_dynamic_comparison_2d(self.get_checked_models())
        except Exception:
            pass


    def _is_normalized_property(self, scalar_name):
        """
        Propriedade que deve permanecer em 0–1 na legenda/clim.
        - Inclui as propriedades marcadas pelo usuário como proporção por célula.
        - Inclui também métricas verticais já normalizadas (NTG/ICV/Qv etc).
        """
        s = str(scalar_name)

        user_set = set(self.state.get("fraction_props", set()) or set())
        if s in user_set:
            return True

        if is_vertical_metric_normalized_name(s):
            return True

        return False


    def _is_equivalent_2d_property(self, scalar_name):
        """
        Propriedade que faz sentido converter para metros equivalentes no 2D.
        Aqui entram apenas as propriedades marcadas pelo usuário como proporção por célula.
        """
        s = str(scalar_name)
        user_set = set(self.state.get("fraction_props", set()) or set())
        return s in user_set


    def _is_fraction_property(self, scalar_name):
        """Compatibilidade com o código antigo."""
        return self._is_normalized_property(scalar_name)


    def _compute_global_clim_for_scalar(self, scalar_name, grids):
        """
        Retorna (vmin, vmax) global para comparação 3D.
        - Se for proporção: (0, 1)
        - Caso contrário: (min_global, max_global) e se todos >=0 => (0, max_global)
        """
        import numpy as np

        if self._is_normalized_property(scalar_name):
            return (0.0, 1.0)

        vals = []
        for g in (grids or []):
            try:
                if g is None or scalar_name not in g.cell_data:
                    continue
                arr = np.asarray(g.cell_data[scalar_name], dtype=float)
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    vals.append(finite)
            except Exception:
                continue

        if not vals:
            return (0.0, 1.0)

        allv = np.concatenate(vals)
        vmin = float(np.nanmin(allv))
        vmax = float(np.nanmax(allv))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return (0.0, 1.0)
        if vmax <= vmin:
            vmax = vmin + 1e-6

        if vmin >= 0.0:
            vmin = 0.0

        return (vmin, vmax)


    def _apply_global_clim_to_active_comparison(self):
        """
        Aplica escala GLOBAL nos plotters 3D da comparação.
        - Se estiver em modo scalar: usa current_scalar_name/current_scalar_clim
        - Se estiver em thickness_local: usa thickness_mode/thickness_clim
        """
        states = list(getattr(self, "active_comp_states", []) or [])
        if not states:
            return

        mode_3d = self.state.get("mode", "facies")

        grids = []
        for st in states:
            g = st.get("current_grid_source")
            if g is not None:
                grids.append(g)

        if mode_3d == "scalar" and self.state.get("current_scalar_name"):
            scalar = self.state.get("current_scalar_name")
            title = self.state.get("current_scalar_title", scalar)
            cmap_use = self.state.get(
                "current_scalar_cmap",
                self.state.get("thickness_cmap", "jet")
            )

            clim = self._compute_global_clim_for_scalar(scalar, grids)
            self.state["current_scalar_clim"] = clim

            for st in states:
                try:
                    st["mode"] = "scalar"
                    st["current_scalar_name"] = scalar
                    st["current_scalar_title"] = title
                    st["current_scalar_clim"] = clim
                    st["current_scalar_cmap"] = cmap_use

                    if "refresh" in st and callable(st["refresh"]):
                        st["refresh"]()
                    if "plotter_ref" in st:
                        st["plotter_ref"].render()
                except Exception:
                    pass
            return

        presets = self.state.get("thickness_presets") or {}
        thick_mode = self.state.get("thickness_mode", "Espessura total da coluna")
        if thick_mode not in presets and presets:
            thick_mode = list(presets.keys())[0]

        scalar, _title = presets.get(thick_mode, ("vert_Ttot_reservoir", "Espessura"))
        clim = self._compute_global_clim_for_scalar(scalar, grids)

        for st in states:
            try:
                st["thickness_clim"] = clim
                st["thickness_clim_manual"] = True
                st["thickness_global_clim"] = None

                if "update_thickness" in st and callable(st["update_thickness"]):
                    st["update_thickness"]()
                if "refresh" in st and callable(st["refresh"]):
                    st["refresh"]()
                if "plotter_ref" in st:
                    st["plotter_ref"].render()
            except Exception:
                pass

    def _lock_axes_bounds_to_grid(self, state=None):
        """
        Atualiza a caixa de bounds usando somente bounds brutos do estado
        + o z_exag atual.

        Nunca usa GetBounds() do ator, para evitar inconsistência entre:
        - grid inteiro
        - mesh cortado
        - ator já escalado
        """
        try:
            st = state or self.state
            ba = st.get("bounds_actor", None)
            if ba is None:
                return

            z_scale = float(st.get("z_exag", 1.0))

            if st.get("bounds_follow_slices", True):
                raw = st.get("slice_bounds_raw")
            else:
                raw = st.get("grid_bounds_raw")

            if raw is None:
                g = st.get("current_grid_source", None)
                if g is not None:
                    raw = g.bounds

            if raw is None:
                return

            xmin, xmax, ymin, ymax, zmin, zmax = raw
            display_bounds = (xmin, xmax, ymin, ymax, zmin * z_scale, zmax * z_scale)

            ba.SetBounds(display_bounds)
            st["_last_axes_bounds"] = tuple(display_bounds)

        except Exception:
            pass

    def _maybe_update_axes_bounds(self, state=None, rel_tol=0.01, abs_tol=0.5):
        """
        Atualiza bounds do CubeAxes só se mudou o suficiente,
        usando bounds brutos do estado + z_exag.
        """
        try:
            st = state or self.state
            ba = st.get("bounds_actor", None)
            if ba is None:
                return

            z_scale = float(st.get("z_exag", 1.0))

            if st.get("bounds_follow_slices", True):
                raw = st.get("slice_bounds_raw")
            else:
                raw = st.get("grid_bounds_raw")

            if raw is None:
                g = st.get("current_grid_source", None)
                if g is not None:
                    raw = g.bounds

            if raw is None:
                return

            xmin, xmax, ymin, ymax, zmin, zmax = raw
            newb = (xmin, xmax, ymin, ymax, zmin * z_scale, zmax * z_scale)

            oldb = st.get("_last_axes_bounds", None)
            if oldb is None:
                ba.SetBounds(newb)
                st["_last_axes_bounds"] = tuple(newb)
                return

            changed = False
            for i in range(6):
                a = float(oldb[i])
                b = float(newb[i])
                diff = abs(a - b)
                scale = max(abs(a), abs(b), 1.0)

                if diff > abs_tol and (diff / scale) > rel_tol:
                    changed = True
                    break

            if changed:
                ba.SetBounds(newb)
                st["_last_axes_bounds"] = tuple(newb)

        except Exception:
            pass

    def _weighted_mean_output_name(self, scalar_name):
        import re
        safe = re.sub(r"[^0-9a-zA-Z_]+", "_", str(scalar_name)).strip("_")
        return f"wmean_th_{safe}"


    def _register_extra_sync_cell_data(self, scalar_name):
        extra = set(self.state.get("extra_sync_cell_data", set()) or set())
        extra.add(str(scalar_name))
        self.state["extra_sync_cell_data"] = extra


    def _ensure_weighted_mean_on_grid(self, grid_source, scalar_name):
        """
        Calcula e salva no grid a média ponderada por espessura de scalar_name.
        """
        from analysis import compute_thickness_weighted_property_map
        import numpy as np

        if grid_source is None:
            return None, None

        if scalar_name not in getattr(grid_source, "cell_data", {}):
            return None, None

        output_name = self._weighted_mean_output_name(scalar_name)

        clip_to_01 = self._is_normalized_property(scalar_name)

        out_name, out_2d = compute_thickness_weighted_property_map(
            grid_source,
            scalar_name,
            output_name=output_name,
            clip_to_01=clip_to_01,
        )

        if out_name is None:
            return None, None

        self._register_extra_sync_cell_data(out_name)

        return out_name, out_2d


    def change_weighted_mean_view(self, scalar_name):
        """
        Cria e visualiza a média ponderada por espessura de uma propriedade.
        """
        import numpy as np

        grid = self.state.get("current_grid_source")
        if grid is None and "base" in self.models:
            grid = self.models["base"].get("grid")

        if grid is None:
            return

        out_name, _ = self._ensure_weighted_mean_on_grid(grid, scalar_name)
        if out_name is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Média ponderada",
                f"Não foi possível calcular média ponderada para '{scalar_name}'."
            )
            return

        title = f"Média ponderada por espessura: {scalar_name}"

        # Também calcula para os grids ativos da comparação, se existirem
        for st in getattr(self, "active_comp_states", []) or []:
            try:
                g = st.get("current_grid_source")
                if g is not None and scalar_name in g.cell_data:
                    self._ensure_weighted_mean_on_grid(g, scalar_name)
            except Exception:
                pass

        arr = np.asarray(grid.cell_data[out_name], dtype=float)
        finite = arr[np.isfinite(arr)]

        if self._is_normalized_property(scalar_name):
            clim = (0.0, 1.0)
        elif finite.size:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
            if vmin >= 0.0:
                vmin = 0.0
            if vmax <= vmin:
                vmax = vmin + 1e-6
            clim = (vmin, vmax)
        else:
            clim = (0.0, 1.0)

        # registra como preset de métricas por coluna
        presets = self.state.get("thickness_presets", {})
        label = f"Média ponderada: {scalar_name}"
        presets[label] = (out_name, title)
        self.state["thickness_presets"] = presets

        self.state["mode"] = "thickness_local"
        self.state["thickness_mode"] = label
        self.state["thickness_clim"] = clim
        self.state["thickness_clim_manual"] = True
        self.state["thickness_cmap"] = self.state.get("thickness_cmap", "jet")

        # limpa estado de scalar genérico para não conflitar
        self.state.pop("current_scalar_name", None)
        self.state.pop("current_scalar_title", None)
        self.state.pop("current_scalar_clim", None)

        # força sincronização com grid_base interno do visualize.py
        upd = self.state.get("update_reservoir_fields")
        if callable(upd):
            try:
                upd(set(self.state.get("reservoir_facies", set()) or []))
            except Exception:
                pass

        refresh = self.state.get("refresh")
        if callable(refresh):
            refresh()

        if hasattr(self, "update_2d_map"):
            self.update_2d_map()

        if hasattr(self, "btn_mode"):
            self.btn_mode.setText("Média\nPond.")

        if hasattr(self, "btn_thick"):
            self.btn_thick.setText(f"Média\n{scalar_name}")

    def _finalize_vtk_widget(self, obj):
        """Fecha plotters/QtInteractors de forma agressiva para evitar erros do VTK no shutdown."""
        if obj is None:
            return

        # 1) Tenta achar um vtkRenderWindow
        rw = None
        candidates = [
            obj,
            getattr(obj, "interactor", None),
            getattr(obj, "iren", None),
            getattr(obj, "plotter", None),
        ]

        for c in candidates:
            if c is None:
                continue
            # QtInteractor (pyvistaqt) costuma ter .ren_win
            if hasattr(c, "ren_win"):
                rw = getattr(c, "ren_win", None)
                if rw is not None:
                    break
            # alguns wrappers expõem .render_window
            if hasattr(c, "render_window"):
                rw = getattr(c, "render_window", None)
                if rw is not None:
                    break
            # vtkRenderWindowInteractor -> GetRenderWindow()
            if hasattr(c, "GetRenderWindow"):
                try:
                    rw = c.GetRenderWindow()
                    if rw is not None:
                        break
                except Exception:
                    pass

        # 2) Termina interactor (quando existir)
        iren = getattr(obj, "iren", None) or getattr(obj, "interactor", None)
        if iren is not None and hasattr(iren, "TerminateApp"):
            try:
                iren.TerminateApp()
            except Exception:
                pass

        # 3) Finaliza render window ANTES do Qt destruir o contexto OpenGL
        if rw is not None and hasattr(rw, "Finalize"):
            try:
                rw.Finalize()
            except Exception:
                pass

        # 4) Fecha widget/janela
        try:
            obj.close()
        except Exception:
            pass

        # 5) Se for QWidget, agenda destruição
        try:
            obj.setParent(None)
            obj.deleteLater()
        except Exception:
            pass

    def _install_2d_hover_filter(self, plotter, model_name=None):
        """Instala hover readout no plotter 2D sem bloquear pan/zoom."""
        if plotter is None:
            return

        try:
            plotter._hover2d_model_name = model_name
        except Exception:
            pass

        targets = []
        try:
            targets.append(plotter)
        except Exception:
            pass
        try:
            inter = getattr(plotter, "interactor", None)
            if inter is not None:
                targets.append(inter)
        except Exception:
            pass

        for t in targets:
            try:
                t.setMouseTracking(True)
            except Exception:
                pass
            try:
                t.installEventFilter(self)
            except Exception:
                pass
            try:
                self._map2d_hover_targets[id(t)] = plotter
            except Exception:
                pass


    def _qt_pos_to_vtk_xy(self, plotter, widget, pos):
        """Converte posição Qt -> coordenada de tela VTK."""
        inter = None
        try:
            inter = getattr(plotter, "interactor", None)
        except Exception:
            inter = None
        if inter is None:
            inter = widget

        p = pos
        try:
            if widget is not inter and hasattr(inter, "mapFrom"):
                p = inter.mapFrom(widget, pos)
        except Exception:
            p = pos

        try:
            rw, rh = plotter.ren_win.GetSize()
        except Exception:
            rw, rh = None, None

        try:
            qw = int(inter.width())
            qh = int(inter.height())
        except Exception:
            qw, qh = None, None

        if rw and rh and qw and qh and qw > 0 and qh > 0:
            sx = float(rw) / float(qw)
            sy = float(rh) / float(qh)
            x = float(p.x()) * sx
            y = float(qh - p.y()) * sy
        else:
            x = float(p.x())
            y = float(p.y())

        return int(x), int(y)


    def _update_2d_hover_status(self, plotter, widget, pos):
        """Mostra valor/X/Y da célula 2D sob o mouse na status bar."""
        import numpy as np
        try:
            from vtkmodules.vtkRenderingCore import vtkCellPicker
        except Exception:
            return

        meta = getattr(plotter, "_map2d_hover_meta", None)
        if not meta:
            return

        surf = meta.get("surf")
        name2d = meta.get("name2d")
        label = meta.get("label", name2d)
        model_name = meta.get("model_name", None)
        scalar_name_3d = meta.get("scalar_name_3d")
        dims = meta.get("dims")
        total_thickness_2d = meta.get("total_thickness_2d")

        if surf is None or not name2d or name2d not in getattr(surf, "cell_data", {}):
            return

        try:
            x, y = self._qt_pos_to_vtk_xy(plotter, widget, pos)
        except Exception:
            return

        try:
            picker = vtkCellPicker()
            picker.SetTolerance(0.0005)

            renderer = getattr(plotter, "renderer", None)
            if renderer is None and hasattr(plotter, "renderers"):
                try:
                    renderer = plotter.renderers[0]
                except Exception:
                    renderer = None
            if renderer is None:
                return

            ok = picker.Pick(x, y, 0, renderer)
            if not ok:
                return

            cell_id = int(picker.GetCellId())
            if cell_id < 0 or cell_id >= surf.n_cells:
                return
        except Exception:
            return

        try:
            val = float(np.asarray(surf.cell_data[name2d], dtype=float)[cell_id])
        except Exception:
            return

        if not np.isfinite(val):
            return

        try:
            cell = surf.get_cell(cell_id)
            pts = np.asarray(cell.points, dtype=float)
            center = pts.mean(axis=0) if pts.size else np.array([np.nan, np.nan, np.nan], dtype=float)
            xc = float(center[0])
            yc = float(center[1])
        except Exception:
            xc, yc = np.nan, np.nan

        total_col = None
        if self._is_equivalent_2d_property(scalar_name_3d) and total_thickness_2d is not None and dims:
            try:
                nx2 = int(dims[0] - 1)
                iy = int(cell_id // nx2)
                ix = int(cell_id % nx2)
                total_col = float(total_thickness_2d[ix, iy])
                if not np.isfinite(total_col):
                    total_col = None
            except Exception:
                total_col = None

        prefix = f"{model_name} | " if model_name else ""
        if total_col is not None:
            msg = f"{prefix}{label}: {val:.4g} m | Esp. coluna: {total_col:.4g} m | X={xc:.1f} | Y={yc:.1f}"
        else:
            unit = " m" if scalar_name_3d == "__total_column_thickness__" else ""
            msg = f"{prefix}{label}: {val:.4g}{unit} | X={xc:.1f} | Y={yc:.1f}"

        if msg != self._last_2d_hover_msg:
            self._last_2d_hover_msg = msg
            try:
                self.statusBar().showMessage(msg)
            except Exception:
                pass


    def _pick_2d_cell_id(self, plotter, widget, pos):
        """Retorna o cell_id da superfície 2D sob o mouse, ou None."""
        try:
            from vtkmodules.vtkRenderingCore import vtkCellPicker
        except Exception:
            return None

        meta = getattr(plotter, "_map2d_hover_meta", None)
        if not meta:
            return None

        surf = meta.get("surf")
        if surf is None:
            return None

        try:
            x, y = self._qt_pos_to_vtk_xy(plotter, widget, pos)
        except Exception:
            return None

        try:
            picker = vtkCellPicker()
            picker.SetTolerance(0.0005)
            renderer = getattr(plotter, "renderer", None)
            if renderer is None and hasattr(plotter, "renderers"):
                renderer = plotter.renderers[0]
            if renderer is None:
                return None
            ok = picker.Pick(x, y, 0, renderer)
            if not ok:
                return None
            cid = int(picker.GetCellId())
            if cid < 0 or cid >= surf.n_cells:
                return None
            return cid
        except Exception:
            return None


    def _surface_cell_id_to_ij(self, cell_id, dims):
        """Converte cell_id da superfície 2D para índices (i,j) do mapa reduzido."""
        if dims is None:
            return None, None

        try:
            nx_pts = int(dims[0])
            ny_pts = int(dims[1])
        except Exception:
            return None, None

        nx_cells = nx_pts - 1
        ny_cells = ny_pts - 1

        if nx_cells <= 0 or ny_cells <= 0:
            return None, None

        iy = int(cell_id // nx_cells)
        ix = int(cell_id % nx_cells)

        if ix < 0 or ix >= nx_cells or iy < 0 or iy >= ny_cells:
            return None, None

        return ix, iy


    def _build_2d_column_summary_html(self, plotter, cell_id):
        import numpy as np

        meta = getattr(plotter, "_map2d_hover_meta", None)
        if not meta:
            return None

        surf = meta.get("surf")
        name2d = meta.get("name2d")
        label = meta.get("label", name2d)
        model_name = meta.get("model_name")
        grid_source = meta.get("grid_source")
        scalar_name_3d = meta.get("scalar_name_3d")
        dims = meta.get("dims")
        total_thickness_2d = meta.get("total_thickness_2d")

        if surf is None or grid_source is None or dims is None:
            return None

        try:
            val = float(np.asarray(surf.cell_data[name2d], dtype=float)[cell_id])
        except Exception:
            val = np.nan

        try:
            cell = surf.get_cell(cell_id)
            pts = np.asarray(cell.points, dtype=float)
            center = pts.mean(axis=0) if pts.size else np.array([np.nan, np.nan, np.nan], dtype=float)
            xc = float(center[0])
            yc = float(center[1])
        except Exception:
            xc, yc = np.nan, np.nan

        ix, iy = self._surface_cell_id_to_ij(cell_id, dims)
        if ix is None or iy is None:
            return None

        if total_thickness_2d is None:
            total_thickness_2d = self._reduce_total_column_thickness_to_2d(grid_source)
        total_th = None
        try:
            total_th = float(total_thickness_2d[ix, iy])
            if not np.isfinite(total_th):
                total_th = None
        except Exception:
            total_th = None

        lines = []
        title_bits = ["<b>Resumo da coluna 2D</b>"]
        if model_name:
            title_bits.append(f"<span style='color:#444'>[{model_name}]</span>")
        lines.append(" ".join(title_bits))
        lines.append(f"<span style='color:#555'>X={xc:.1f} | Y={yc:.1f} | i={ix} | j={iy}</span>")
        lines.append(f"<b>{label}:</b> {val:.4g}{' m' if scalar_name_3d == '__total_column_thickness__' or self._is_equivalent_2d_property(scalar_name_3d) else ''}")
        if total_th is not None:
            lines.append(f"<b>Espessura total da coluna:</b> {total_th:.4g} m")

        prop_names = sorted(
            [p for p in set(self.state.get("fraction_props", set()) or set()) if p in getattr(grid_source, "cell_data", {})],
            key=lambda s: str(s).lower(),
        )

        if prop_names:
            th = self._get_grid_cell_thickness_array(grid_source)
            dims3 = self._infer_grid_cell_dims(grid_source)
            if th is not None and dims3:
                try:
                    nx, ny, nz = dims3
                    th3d = np.asarray(th, dtype=float).reshape((nx, ny, nz), order="F")
                    th_col = th3d[ix, iy, :]
                except Exception:
                    th_col = None
                if th_col is not None:
                    lines.append("<br><b>Equivalente por propriedade</b>")
                    for prop in prop_names:
                        try:
                            arr3d = np.asarray(grid_source.cell_data[prop], dtype=float).reshape((nx, ny, nz), order="F")
                            prop_col = arr3d[ix, iy, :]
                            mask = np.isfinite(prop_col) & np.isfinite(th_col)
                            if not np.any(mask):
                                continue
                            p = np.clip(prop_col[mask], 0.0, 1.0)
                            t = np.clip(th_col[mask], 0.0, None)
                            eq_m = float(np.sum(t * p))
                            mean_p = (eq_m / total_th) if (total_th is not None and total_th > 0.0) else np.nan
                            if np.isfinite(mean_p):
                                lines.append(f"• <b>{prop}</b>: {eq_m:.4g} m eq | média ponderada = {mean_p:.4g}")
                            else:
                                lines.append(f"• <b>{prop}</b>: {eq_m:.4g} m eq")
                        except Exception:
                            continue

        return "<br>".join(lines)


    def _on_2d_map_clicked(self, plotter, widget, pos):
        target = getattr(plotter, "_map2d_summary_target", None)
        if target is None:
            return

        cell_id = self._pick_2d_cell_id(plotter, widget, pos)
        if cell_id is None:
            return

        html = self._build_2d_column_summary_html(plotter, cell_id)
        if not html:
            return

        try:
            target.setHtml(html)
        except Exception:
            try:
                target.setPlainText(re.sub(r"<[^>]+>", "", html))
            except Exception:
                pass


    def _clear_2d_hover_status(self):
        try:
            self._last_2d_hover_msg = ""
            self.statusBar().clearMessage()
        except Exception:
            pass


    # ============================================================
    # 3D Selection Mode + Inspector (Cell / Column)
    # ============================================================

    def _adjust_main_z_exag(self, wheel_delta):
        """
        Ajusta o exagero vertical do 3D principal usando o mesmo controle
        do Inspector (self.slicer_widget.spin_z).
        """
        try:
            slicer = getattr(self, "slicer_widget", None)
            if slicer is None:
                return False

            spin = getattr(slicer, "spin_z", None)
            if spin is None:
                return False

            # 1 notch de wheel = 120
            if wheel_delta == 0:
                return False

            steps = int(wheel_delta / 120)
            if steps == 0:
                steps = 1 if wheel_delta > 0 else -1

            step_size = float(spin.singleStep()) if spin.singleStep() else 1.0
            current = float(spin.value())

            new_value = current + steps * step_size
            new_value = max(float(spin.minimum()), min(float(spin.maximum()), new_value))

            if abs(new_value - current) < 1e-9:
                return False

            # Isso já aciona todo o pipeline normal do seu app
            spin.setValue(new_value)
            return True

        except Exception:
            return False

    def _install_3d_pick_filter(self):
        if getattr(self, "_pick_filter_installed", False):
            return
        self._pick_filter_installed = True
        self._pick_press_pos = None
        self._pick_dragging = False
        self._mid_button_down_3d = False

        targets = []
        try:
            targets.append(self.plotter)
        except Exception:
            pass
        try:
            w = getattr(self.plotter, "interactor", None)
            if w is not None:
                targets.append(w)
        except Exception:
            pass

        for t in targets:
            try:
                t.installEventFilter(self)
            except Exception:
                pass
            
    def _qt_trigger_pick(self, widget, pos):
            """Converte coordenadas Qt (origem topo-esq) -> VTK/RenderWindow (origem base-esq) e dispara o pick.

            Observação: no Windows/HiDPI o tamanho do RenderWindow pode diferir do tamanho do widget Qt.
            Aqui a conversão usa a razão (RenderWindowSize / WidgetSize), que costuma ser o mais robusto.
            """
            try:
                fn = self.state.get("_pick_perform", None)
            except Exception:
                fn = None
            if not callable(fn):
                try:
                    self.statusBar().showMessage("Pick: handler não instalado (state['_pick_perform'] ausente)", 2500)
                except Exception:
                    pass
                return

            # Sempre tenta usar o widget do interactor (onde o VTK realmente desenha)
            inter = None
            try:
                inter = getattr(self.plotter, "interactor", None)
            except Exception:
                inter = None
            if inter is None:
                inter = widget

            # Normaliza posição para o referencial do interactor
            p = pos
            try:
                if widget is not inter and hasattr(inter, "mapFrom"):
                    p = inter.mapFrom(widget, pos)
            except Exception:
                p = pos

            # Escala para o tamanho real do RenderWindow (pixels do VTK)
            try:
                rw, rh = self.plotter.ren_win.GetSize()
            except Exception:
                rw, rh = None, None

            try:
                qw = int(inter.width())
                qh = int(inter.height())
            except Exception:
                qw, qh = None, None

            try:
                if rw and rh and qw and qh and qw > 0 and qh > 0:
                    sx = float(rw) / float(qw)
                    sy = float(rh) / float(qh)
                    x = float(p.x()) * sx
                    y = float(qh - p.y()) * sy  # VTK origin: bottom-left
                else:
                    x = float(p.x())
                    y = float(p.y())
            except Exception:
                return

            ok = False
            try:
                ok = bool(fn(int(x), int(y)))
            except Exception:
                ok = False

            if not ok:
                try:
                    self.statusBar().showMessage("Pick: nenhuma célula encontrada (tente clicar nas faces/arestas da malha)", 1500)
                except Exception:
                    pass

    def eventFilter(self, obj, event):
        """Captura clique sem bloquear rotação/zoom.

        - Se o usuário clicar (pressiona/solta) sem arrastar e pick_mode estiver ativo,
          dispara o pick no release.
        - Se arrastar (rotacionar), não seleciona.
        """
        # ---------------------------
        # Hover readout dos mapas 2D
        # ---------------------------
        try:
            plotter2d = self._map2d_hover_targets.get(id(obj), None)
        except Exception:
            plotter2d = None

        if plotter2d is not None:
            et = event.type()

            if et == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
                self._map2d_press_pos = event.pos()
                self._map2d_dragging = False

            elif et == QtCore.QEvent.MouseMove:
                try:
                    self._update_2d_hover_status(plotter2d, obj, event.pos())
                except Exception:
                    pass

                if getattr(self, "_map2d_press_pos", None) is not None:
                    try:
                        if (event.pos() - self._map2d_press_pos).manhattanLength() > 6:
                            self._map2d_dragging = True
                    except Exception:
                        pass

            elif et == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
                press = getattr(self, "_map2d_press_pos", None)
                dragging = bool(getattr(self, "_map2d_dragging", False))
                if press is not None and not dragging:
                    try:
                        self._on_2d_map_clicked(plotter2d, obj, event.pos())
                    except Exception:
                        pass
                self._map2d_press_pos = None
                self._map2d_dragging = False

            elif et in (QtCore.QEvent.Leave, QtCore.QEvent.Hide):
                try:
                    self._clear_2d_hover_status()
                except Exception:
                    pass

        try:
            is_target = (obj is getattr(self, "plotter", None)) or (obj is getattr(getattr(self, "plotter", None), "interactor", None))
        except Exception:
            is_target = False

        if is_target:
            et = event.type()

            # =========================================================
            # SHIFT + WHEEL = EXAGERO VERTICAL DO 3D PRINCIPAL
            # Wheel sozinho continua com o zoom normal do VTK
            # =========================================================
            if et == QtCore.QEvent.Wheel:
                try:
                    mods = event.modifiers()
                except Exception:
                    mods = QtCore.Qt.NoModifier

                if mods & QtCore.Qt.ShiftModifier:
                    try:
                        delta = event.angleDelta().y()
                    except Exception:
                        delta = 0

                    if self._adjust_main_z_exag(delta):
                        return True

            try:
                mode = self.state.get("pick_mode", None)
            except Exception:
                mode = None

            if mode in ("cell", "column"):
                et = event.type()
                if et == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
                    self._pick_press_pos = event.pos()
                    self._pick_dragging = False

                elif et == QtCore.QEvent.MouseMove and getattr(self, "_pick_press_pos", None) is not None:
                    try:
                        if (event.pos() - self._pick_press_pos).manhattanLength() > 6:
                            self._pick_dragging = True
                    except Exception:
                        pass

                elif et == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
                    try:
                        press = getattr(self, "_pick_press_pos", None)
                        dragging = bool(getattr(self, "_pick_dragging", False))
                    except Exception:
                        press = None
                        dragging = False

                    if press is not None and not dragging:
                        # dispara pick no release (clique)
                        self._qt_trigger_pick(obj, event.pos())

                    self._pick_press_pos = None
                    self._pick_dragging = False

        return super().eventFilter(obj, event)
    
    def set_pick_mode(self, mode):
        # mode: None | 'cell' | 'column'
        if mode not in (None, 'cell', 'column'):
            mode = None
        self.state['pick_mode'] = mode

        # feedback visual + garante foco no 3D
        try:
            if hasattr(self, 'plotter') and self.plotter is not None:
                self.plotter.setFocus()
                self.plotter.setCursor(QtCore.Qt.CrossCursor if mode else QtCore.Qt.ArrowCursor)
        except Exception:
            pass
        try:
            sb = self.statusBar() if hasattr(self, 'statusBar') else None
            if sb:
                msg = 'Seleção 3D: desligada' if mode is None else ('Seleção 3D: CÉLULA (clique no 3D)' if mode=='cell' else 'Seleção 3D: COLUNA (clique no 3D)')
                sb.showMessage(msg, 4000)
        except Exception:
            pass

        # exclusividade dos botões
        if hasattr(self, 'btn_pick_cell'):
            self.btn_pick_cell.blockSignals(True)
            self.btn_pick_cell.setChecked(mode == 'cell')
            self.btn_pick_cell.blockSignals(False)
        if hasattr(self, 'btn_pick_column'):
            self.btn_pick_column.blockSignals(True)
            self.btn_pick_column.setChecked(mode == 'column')
            self.btn_pick_column.blockSignals(False)

        if mode is None:
            self.clear_pick_selection()

    def clear_pick_selection(self):
        # remove highlight no 3D e limpa o inspector
        try:
            fn = self.state.get('clear_pick')
            if callable(fn):
                fn()
        except Exception:
            pass

        try:
            if getattr(self, '_inspector_dock', None) is not None:
                self._update_cell_table(None)
                self._update_column_tab(None)
        except Exception:
            pass

    def _ensure_inspector_dock(self):
        if getattr(self, '_inspector_dock', None) is not None:
            return

        dock = QtWidgets.QDockWidget('Inspector (Célula/Coluna)', self)
        dock.setObjectName('dock_cell_inspector')
        dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea
            | QtCore.Qt.RightDockWidgetArea
            | QtCore.Qt.BottomDockWidgetArea
        )

        root = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(root)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        self._insp_tabs = QtWidgets.QTabWidget()

        # --- TAB: Célula ---
        tab_cell = QtWidgets.QWidget()
        vc = QtWidgets.QVBoxLayout(tab_cell)
        vc.setContentsMargins(0, 0, 0, 0)
        vc.setSpacing(6)

        self._lbl_cell_title = QtWidgets.QLabel('Nenhuma célula selecionada')
        f = self._lbl_cell_title.font()
        f.setBold(True)
        self._lbl_cell_title.setFont(f)

        # Resumo (geom + localização)
        self._grp_cell_summary = QtWidgets.QGroupBox('Resumo')
        gs = QtWidgets.QGridLayout(self._grp_cell_summary)
        gs.setContentsMargins(8, 8, 8, 8)
        gs.setHorizontalSpacing(10)
        gs.setVerticalSpacing(4)

        def _mk_row_cell(r, label):
            lab = QtWidgets.QLabel(label)
            val = QtWidgets.QLabel('-')
            val.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            gs.addWidget(lab, r, 0)
            gs.addWidget(val, r, 1)
            return val

        self._val_cell_ijk = _mk_row_cell(0, '(i, j, k)')
        self._val_cell_length = _mk_row_cell(1, 'Comprimento (ΔX)')
        self._val_cell_width = _mk_row_cell(2, 'Largura (ΔY)')
        self._val_cell_thickness = _mk_row_cell(3, 'Thickness (ΔZ)')
        self._val_cell_volume = _mk_row_cell(4, 'Volume')
        self._val_cell_center = _mk_row_cell(5, 'Centro (X, Y, Z)')

        self._tbl_cell = QtWidgets.QTableWidget(0, 2)
        self._tbl_cell.setHorizontalHeaderLabels(['Propriedade', 'Valor'])
        self._tbl_cell.horizontalHeader().setStretchLastSection(True)
        self._tbl_cell.verticalHeader().setVisible(False)
        self._tbl_cell.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._tbl_cell.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

        vc.addWidget(self._lbl_cell_title)
        vc.addWidget(self._grp_cell_summary)
        vc.addWidget(self._tbl_cell)
        self._insp_tabs.addTab(tab_cell, 'Célula')

        # --- TAB: Coluna ---
        tab_col = QtWidgets.QWidget()
        vcol = QtWidgets.QVBoxLayout(tab_col)
        vcol.setContentsMargins(0, 0, 0, 0)
        vcol.setSpacing(6)

        self._lbl_col_title = QtWidgets.QLabel('Nenhuma coluna selecionada')
        f2 = self._lbl_col_title.font()
        f2.setBold(True)
        self._lbl_col_title.setFont(f2)
        # Resumo da coluna (I,J)
        self._grp_col_summary = QtWidgets.QGroupBox('Resumo')
        gc = QtWidgets.QGridLayout(self._grp_col_summary)
        gc.setContentsMargins(8, 8, 8, 8)
        gc.setHorizontalSpacing(10)
        gc.setVerticalSpacing(4)

        def _mk_row_col(r, label):
            lab = QtWidgets.QLabel(label)
            val = QtWidgets.QLabel('-')
            val.setWordWrap(True)
            val.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            gc.addWidget(lab, r, 0)
            gc.addWidget(val, r, 1)
            return val

        self._val_col_ij = _mk_row_col(0, '(i, j)')
        self._val_col_ncells = _mk_row_col(1, 'N células na coluna')
        self._val_col_top_base = _mk_row_col(2, 'Z topo / base')
        self._val_col_th_sum = _mk_row_col(3, 'Soma Thickness')
        self._val_col_facies = _mk_row_col(4, 'Facies (contagem)')

        # Tabela completa da coluna com primeira coluna "k" fixa
        self._col_model = QtGui.QStandardItemModel(0, 0, self)

        self._tbl_col_frozen = QtWidgets.QTableView()
        self._tbl_col_main = QtWidgets.QTableView()

        for tv in (self._tbl_col_frozen, self._tbl_col_main):
            tv.setModel(self._col_model)
            tv.verticalHeader().setVisible(False)
            tv.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            tv.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
            tv.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
            tv.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
            tv.setWordWrap(False)

        # frozen: sem scroll horizontal e sem barra vertical
        self._tbl_col_frozen.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._tbl_col_frozen.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._tbl_col_frozen.setFocusPolicy(QtCore.Qt.NoFocus)

        # main: scroll normal
        self._tbl_col_main.setFocusPolicy(QtCore.Qt.NoFocus)

        # sincroniza scroll vertical
        self._syncing_col_scroll = False

        def _sync_main_to_frozen(val):
            if self._syncing_col_scroll:
                return
            self._syncing_col_scroll = True
            try:
                self._tbl_col_frozen.verticalScrollBar().setValue(val)
            finally:
                self._syncing_col_scroll = False

        def _sync_frozen_to_main(val):
            if self._syncing_col_scroll:
                return
            self._syncing_col_scroll = True
            try:
                self._tbl_col_main.verticalScrollBar().setValue(val)
            finally:
                self._syncing_col_scroll = False

        self._tbl_col_main.verticalScrollBar().valueChanged.connect(_sync_main_to_frozen)
        self._tbl_col_frozen.verticalScrollBar().valueChanged.connect(_sync_frozen_to_main)

        col_container = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(col_container)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(0)
        hl.addWidget(self._tbl_col_frozen)
        hl.addWidget(self._tbl_col_main, 1)

        vcol.addWidget(self._lbl_col_title)
        vcol.addWidget(self._grp_col_summary)
        vcol.addWidget(col_container)


        self._insp_tabs.addTab(tab_col, 'Coluna (I,J)')

        v.addWidget(self._insp_tabs)

        dock.setWidget(root)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        try:
            # coloca abaixo do Project Explorer
            if hasattr(self, "dock_explorer") and self.dock_explorer is not None:
                self.splitDockWidget(self.dock_explorer, dock, QtCore.Qt.Vertical)
                self.resizeDocks([self.dock_explorer, dock], [320, 260], QtCore.Qt.Vertical)
        except Exception:
            pass
        dock.hide()

        self._inspector_dock = dock

    def _on_3d_pick(self, info):
        # callback do picker
        try:
            self._ensure_inspector_dock()
            self._inspector_dock.show()
            self._inspector_dock.raise_()
        except Exception:
            pass

        mode = None
        try:
            mode = None if info is None else info.get('mode')
        except Exception:
            mode = None

        if mode == 'cell':
            self._update_cell_table(info)
            try: self._insp_tabs.setCurrentIndex(0)
            except Exception: pass
        elif mode == 'column':
            self._update_cell_table(info)
            self._update_column_tab(info)
            try: self._insp_tabs.setCurrentIndex(1)
            except Exception: pass
        else:
            self._update_cell_table(None)
            self._update_column_tab(None)

    def _update_cell_table(self, info):
        try:
            if info is None:
                self._lbl_cell_title.setText('Nenhuma célula selecionada')
                self._tbl_cell.setRowCount(0)
                # limpa resumo
                for w in (getattr(self, '_val_cell_ijk', None),
                          getattr(self, '_val_cell_length', None),
                          getattr(self, '_val_cell_width', None),
                          getattr(self, '_val_cell_thickness', None),
                          getattr(self, '_val_cell_volume', None),
                          getattr(self, '_val_cell_center', None)):
                    try:
                        if w is not None:
                            w.setText('-')
                    except Exception:
                        pass
                return

            i = info.get('i'); j = info.get('j'); k = info.get('k'); cid = info.get('cell_id')
            self._lbl_cell_title.setText(f'Célula selecionada: cell_id={cid} | (i,j,k)=({i},{j},{k})')

            props = info.get('props', {}) or {}
            geom = info.get('geom', {}) or {}

            # resumo topo
            try:
                if getattr(self, '_val_cell_ijk', None) is not None:
                    self._val_cell_ijk.setText(f'({i}, {j}, {k})')
            except Exception:
                pass

            try:
                dx = geom.get('length', None)
                dy = geom.get('width', None)
                dz = geom.get('height', None)
                vol = geom.get('volume', None)
                cx = geom.get('center_x', None)
                cy = geom.get('center_y', None)
                cz = geom.get('center_z', None)

                # thickness preferencial (propriedade do grid)
                th = None
                for nm in ('StratigraphicThickness', 'cell_thickness', 'Thickness', 'thickness_local'):
                    if nm in props:
                        th = props.get(nm)
                        break
                if th is None:
                    th = dz

                if getattr(self, '_val_cell_length', None) is not None and dx is not None:
                    self._val_cell_length.setText(f'{float(dx):.3f}')
                if getattr(self, '_val_cell_width', None) is not None and dy is not None:
                    self._val_cell_width.setText(f'{float(dy):.3f}')
                if getattr(self, '_val_cell_thickness', None) is not None and th is not None:
                    try:
                        self._val_cell_thickness.setText(f'{float(th):.3f}')
                    except Exception:
                        self._val_cell_thickness.setText(str(th))
                if getattr(self, '_val_cell_volume', None) is not None and vol is not None:
                    self._val_cell_volume.setText(f'{float(vol):.3f}')
                if getattr(self, '_val_cell_center', None) is not None and (cx is not None and cy is not None and cz is not None):
                    self._val_cell_center.setText(f'({float(cx):.3f}, {float(cy):.3f}, {float(cz):.3f})')
            except Exception:
                pass

            # tabela de propriedades (depois do resumo)
            excluded = set([
                'i_index','j_index','k_index','vtkOriginalCellIds','vtkOriginalPointIds','vtkGhostType',
                'cell_thickness','thickness_local'
            ])

            ordered = []
            if 'Facies' in props:
                ordered.append(('Facies', props.get('Facies')))
            for key in sorted(props.keys()):
                if key in excluded or key == 'Facies':
                    continue
                ordered.append((key, props.get(key)))

            self._tbl_cell.setRowCount(len(ordered))
            for r,(kname,v) in enumerate(ordered):
                self._tbl_cell.setItem(r,0,QtWidgets.QTableWidgetItem(str(kname)))
                self._tbl_cell.setItem(r,1,QtWidgets.QTableWidgetItem(str(v)))
            try:
                self._tbl_cell.resizeColumnsToContents()
            except Exception:
                pass
        except Exception:
            pass
    def _update_column_tab(self, info):
        try:
            if info is None or info.get('mode') != 'column':
                self._lbl_col_title.setText('Nenhuma coluna selecionada')
                for w in (getattr(self, '_val_col_ij', None),
                        getattr(self, '_val_col_ncells', None),
                        getattr(self, '_val_col_top_base', None),
                        getattr(self, '_val_col_th_sum', None),
                        getattr(self, '_val_col_facies', None)):
                    try:
                        if w is not None:
                            w.setText('-')
                    except Exception:
                        pass
                try:
                    if hasattr(self, "_col_model"):
                        self._col_model.clear()
                    elif hasattr(self, "_tbl_col"):
                        self._tbl_col.setRowCount(0)
                        self._tbl_col.setColumnCount(0)
                except Exception:
                    pass
                return

            i = info.get('i')
            j = info.get('j')
            self._lbl_col_title.setText(f'Coluna selecionada: (i,j)=({i},{j})')

            try:
                if getattr(self, '_val_col_ij', None) is not None:
                    self._val_col_ij.setText(f'({i}, {j})')
            except Exception:
                pass

            nc = info.get('column_ncells', None)
            try:
                if getattr(self, '_val_col_ncells', None) is not None and nc is not None:
                    self._val_col_ncells.setText(str(nc))
            except Exception:
                pass

            try:
                topz = info.get('column_top_z', None)
                basez = info.get('column_base_z', None)
                if getattr(self, '_val_col_top_base', None) is not None and (topz is not None or basez is not None):
                    if topz is not None and basez is not None:
                        self._val_col_top_base.setText(f'{float(topz):.3f} / {float(basez):.3f}')
                    else:
                        self._val_col_top_base.setText(f'{topz} / {basez}')
            except Exception:
                pass

            try:
                if getattr(self, '_val_col_th_sum', None) is not None:
                    if 'column_thickness_sum' in info:
                        nm = info.get('column_thickness_name', 'Thickness')
                        try:
                            self._val_col_th_sum.setText(f'{nm}: {float(info.get("column_thickness_sum")):.3f}')
                        except Exception:
                            self._val_col_th_sum.setText(f'{nm}: {info.get("column_thickness_sum")}')
                    else:
                        self._val_col_th_sum.setText('-')
            except Exception:
                pass

            try:
                fc = info.get('column_facies_counts') or {}
                if getattr(self, '_val_col_facies', None) is not None:
                    if fc:
                        items = sorted(fc.items(), key=lambda kv: kv[0])
                        self._val_col_facies.setText(', '.join([f'{k}:{v}' for k, v in items]))
                    else:
                        self._val_col_facies.setText('-')
            except Exception:
                pass

            rows = info.get('column_rows') or []
            cols = info.get('column_columns') or []

            if not cols and rows:
                keys = set()
                for r in rows:
                    try:
                        keys.update(r.keys())
                    except Exception:
                        pass
                cols = ['k_index'] + sorted([k for k in keys if k != 'k_index'])

            def _fmt(v):
                try:
                    if v is None:
                        return ''
                    if isinstance(v, float):
                        return f'{v:.3f}'
                    return str(v)
                except Exception:
                    return str(v)

            def _is_zero_like(v):
                try:
                    if v is None:
                        return True
                    if isinstance(v, str):
                        s = v.strip()
                        if s == '':
                            return True
                        return float(s) == 0.0
                    return float(v) == 0.0
                except Exception:
                    return False

            # remove colunas inteiras de zero/vazio, exceto k_index e Facies
            filtered_cols = []
            for coln in cols:
                if coln in ('k_index', 'Facies'):
                    filtered_cols.append(coln)
                    continue

                vals = []
                for r in rows:
                    try:
                        vals.append(r.get(coln, None))
                    except Exception:
                        vals.append(None)

                if vals and all(_is_zero_like(v) for v in vals):
                    continue

                filtered_cols.append(coln)

            cols = filtered_cols

            # modo com k fixo
            if hasattr(self, "_col_model") and hasattr(self, "_tbl_col_main") and hasattr(self, "_tbl_col_frozen"):
                m = self._col_model
                m.clear()
                m.setRowCount(len(rows))
                m.setColumnCount(len(cols))
                m.setHorizontalHeaderLabels([('k' if str(c) == 'k_index' else str(c)) for c in cols])

                for r_i, row in enumerate(rows):
                    for c_i, coln in enumerate(cols):
                        try:
                            v = row.get(coln, '')
                        except Exception:
                            v = ''

                        item = QtGui.QStandardItem(_fmt(v))

                        # cor na coluna de facies
                        if str(coln) == 'Facies':
                            try:
                                fac = int(float(v))
                                if hasattr(self, 'facies_colors_dict'):
                                    color_rgb = self.facies_colors_dict.get(fac, None)
                                else:
                                    color_rgb = None

                                if color_rgb is not None:
                                    if len(color_rgb) >= 3:
                                        qcolor = QtGui.QColor(
                                            int(float(color_rgb[0]) * 255),
                                            int(float(color_rgb[1]) * 255),
                                            int(float(color_rgb[2]) * 255)
                                        )
                                        item.setBackground(QtGui.QBrush(qcolor))

                                        lum = 0.299 * qcolor.red() + 0.587 * qcolor.green() + 0.114 * qcolor.blue()
                                        txt = QtGui.QColor(0, 0, 0) if lum > 160 else QtGui.QColor(255, 255, 255)
                                        item.setForeground(QtGui.QBrush(txt))
                            except Exception:
                                pass

                        m.setItem(r_i, c_i, item)

                # frozen mostra só coluna 0; main esconde coluna 0
                for c in range(len(cols)):
                    if c == 0:
                        self._tbl_col_frozen.setColumnHidden(c, False)
                        self._tbl_col_main.setColumnHidden(c, True)
                    else:
                        self._tbl_col_frozen.setColumnHidden(c, True)
                        self._tbl_col_main.setColumnHidden(c, False)

                # largura fina do k
                try:
                    max_k = 0
                    try:
                        max_k = max(int(r.get("k_index", 0)) for r in rows) if rows else 0
                    except Exception:
                        max_k = 0

                    fm = self._tbl_col_frozen.fontMetrics()
                    w0 = fm.horizontalAdvance(str(max_k)) + 10
                    w0 = max(22, min(w0, 36))

                    self._tbl_col_frozen.setColumnWidth(0, w0)
                    self._tbl_col_frozen.setMinimumWidth(w0 + 2)
                    self._tbl_col_frozen.setMaximumWidth(w0 + 2)
                except Exception:
                    pass

                try:
                    for c in range(1, min(len(cols), 12)):
                        self._tbl_col_main.resizeColumnToContents(c)
                except Exception:
                    pass

                return

            # fallback antigo, se necessário
            if hasattr(self, "_tbl_col"):
                self._tbl_col.setRowCount(len(rows))
                self._tbl_col.setColumnCount(len(cols))
                self._tbl_col.setHorizontalHeaderLabels([('k' if str(c) == 'k_index' else str(c)) for c in cols])

                for r_i, row in enumerate(rows):
                    for c_i, coln in enumerate(cols):
                        try:
                            v = row.get(coln, '')
                        except Exception:
                            v = ''
                        self._tbl_col.setItem(r_i, c_i, QtWidgets.QTableWidgetItem(_fmt(v)))

                try:
                    self._tbl_col.resizeColumnsToContents()
                except Exception:
                    pass

        except Exception:
            pass

    def _infer_grid_cell_dims(self, grid_source):
        """Retorna (nx, ny, nz) em células."""
        try:
            dims_pts = getattr(grid_source, "dimensions", None)
            if dims_pts and len(dims_pts) == 3:
                cx, cy, cz = int(dims_pts[0] - 1), int(dims_pts[1] - 1), int(dims_pts[2] - 1)
                if cx > 0 and cy > 0 and cz > 0 and (cx * cy * cz == grid_source.n_cells):
                    return cx, cy, cz
        except Exception:
            pass

        try:
            from load_data import nx as lnx, ny as lny, nz as lnz
            return int(lnx), int(lny), int(lnz)
        except Exception:
            return None


    def _get_grid_cell_thickness_array(self, grid_source):
        """
        Retorna espessura por célula priorizando:
        StratigraphicThickness -> cell_thickness -> cálculo a partir dos vértices.
        """
        import numpy as np

        g = grid_source
        if g is None:
            return None

        for key in ("StratigraphicThickness", "Thickness", "stratigraphic_thickness", "thickness"):
            if key in g.cell_data:
                arr = np.asarray(g.cell_data[key], dtype=float)
                if arr.size == g.n_cells:
                    return arr

        for key in ("cell_thickness", "CellThickness"):
            if key in g.cell_data:
                arr = np.asarray(g.cell_data[key], dtype=float)
                if arr.size == g.n_cells:
                    return arr

        # fallback geométrico
        thick = np.zeros(g.n_cells, dtype=float)
        for cid in range(g.n_cells):
            try:
                cell = g.get_cell(cid)
                pts = np.asarray(cell.points)
            except Exception:
                continue

            if pts.size == 0:
                continue

            z_vals = pts[:, 2] if pts.ndim == 2 and pts.shape[1] >= 3 else np.asarray(pts)
            if z_vals.size >= 8:
                bottom = np.partition(z_vals, 4)[:4].mean()
                top = np.partition(z_vals, -4)[-4:].mean()
            else:
                bottom = float(np.min(z_vals))
                top = float(np.max(z_vals))

            thick[cid] = max(0.0, float(top - bottom))

        g.cell_data["cell_thickness"] = thick
        return thick


    def _reduce_total_column_thickness_to_2d(self, grid_source):
        """Retorna mapa 2D com a espessura total da coluna em cada (i,j)."""
        import numpy as np

        if grid_source is None:
            return None

        dims = self._infer_grid_cell_dims(grid_source)
        if not dims:
            return None

        nx, ny, nz = dims

        th = self._get_grid_cell_thickness_array(grid_source)
        if th is None:
            return None

        try:
            th3d = np.asarray(th, dtype=float).reshape((nx, ny, nz), order="F")
        except Exception:
            return None

        out2d = np.full((nx, ny), np.nan, dtype=float)

        for ix in range(nx):
            for iy in range(ny):
                col_th = th3d[ix, iy, :]
                mask = np.isfinite(col_th) & (col_th > 0.0)
                if np.any(mask):
                    out2d[ix, iy] = float(np.sum(col_th[mask]))

        return out2d


    def _reduce_grid_scalar_to_2d(self, grid_source, scalar_name_3d):
        """
        Reduz um campo 3D para 2D por coluna.

        Regras:
        - __total_column_thickness__: soma das espessuras da coluna
        - propriedade normal: usa máximo finito da coluna
        - propriedade marcada como proporção por célula: usa soma(thickness * proportion)
          => resultado em metros equivalentes
        """
        import numpy as np

        if grid_source is None:
            return None

        if scalar_name_3d == "__total_column_thickness__":
            return self._reduce_total_column_thickness_to_2d(grid_source)

        if scalar_name_3d not in getattr(grid_source, "cell_data", {}):
            return None

        dims = self._infer_grid_cell_dims(grid_source)
        if not dims:
            return None

        nx, ny, nz = dims

        try:
            arr3d = np.asarray(grid_source.cell_data[scalar_name_3d], dtype=float).reshape((nx, ny, nz), order="F")
        except Exception:
            return None

        out2d = np.full((nx, ny), np.nan, dtype=float)

        if self._is_equivalent_2d_property(scalar_name_3d):
            th = self._get_grid_cell_thickness_array(grid_source)
            if th is None:
                return None

            try:
                th3d = np.asarray(th, dtype=float).reshape((nx, ny, nz), order="F")
            except Exception:
                return None

            for ix in range(nx):
                for iy in range(ny):
                    prop_col = arr3d[ix, iy, :]
                    th_col = th3d[ix, iy, :]

                    mask = np.isfinite(prop_col) & np.isfinite(th_col)
                    if not np.any(mask):
                        continue

                    p = np.clip(prop_col[mask], 0.0, 1.0)
                    t = np.clip(th_col[mask], 0.0, None)
                    out2d[ix, iy] = float(np.sum(t * p))

            return out2d

        for ix in range(nx):
            for iy in range(ny):
                col = arr3d[ix, iy, :]
                finite = col[np.isfinite(col)]
                if finite.size > 0:
                    out2d[ix, iy] = float(np.nanmax(finite))

        return out2d


    def _compute_global_2d_clim(self, prepared_grids, scalar_name):
        """
        Calcula CLIM global para comparação 2D usando a MESMA redução do mapa.
        Para propriedades de proporção, isso será em metros equivalentes.
        """
        import numpy as np

        vals = []
        for g in (prepared_grids or []):
            out2d = self._reduce_grid_scalar_to_2d(g, scalar_name)
            if out2d is None:
                continue
            flat = out2d[np.isfinite(out2d)]
            if flat.size:
                vals.append(flat)

        if self._is_normalized_property(scalar_name):
            return (0.0, 1.0)

        if not vals:
            return None

        allv = np.concatenate(vals)
        vmin = float(np.nanmin(allv))
        vmax = float(np.nanmax(allv))

        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return None

        if vmax <= vmin:
            vmax = vmin + 1e-6

        if vmin >= 0.0:
            vmin = 0.0

        return (vmin, vmax)    

    def _cleanup_vtk(self):
        """Chamável múltiplas vezes sem problema."""
        if getattr(self, "_vtk_cleaned", False):
            return
        self._vtk_cleaned = True

        # principais plotters/widgets (ajusta conforme teus atributos)
        self._finalize_vtk_widget(getattr(self, "plotter", None))
        self._finalize_vtk_widget(getattr(self, "uncert_plotter", None))

        # plotters de comparação (se existirem no teu estado)
        for st in (getattr(self, "active_comp_states", None) or []):
            try:
                self._finalize_vtk_widget(st.get("plotter"))
            except Exception:
                pass

        # fecha qualquer janela PyVista aberta globalmente (extra)
        try:
            import pyvista as pv
            pv.close_all()
        except Exception:
            pass

    def closeEvent(self, event):
        # limpa VTK/PyVista antes do Qt derrubar o OpenGL context
        try:
            self._cleanup_vtk()
        finally:
            super().closeEvent(event)
    

