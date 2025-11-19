# visualize.py
import numpy as np
import pyvista as pv
pv.global_theme.allow_empty_mesh = True
import vtk
from scipy.ndimage import label as nd_label, generate_binary_structure

# Importamos globais para valores default
from load_data import grid as global_grid, nx, ny, nz
from config import load_facies_colors
from analysis import make_thickness_2d_from_grid

FACIES_COLORS = load_facies_colors()

# Variáveis Globais de Controle
THICKNESS_SCALAR_NAME = "thickness_local"
THICKNESS_SCALAR_TITLE = "Thickness local"
MODE = "facies"
Z_EXAG = 15.0
SHOW_SCALAR_BAR = False

def set_thickness_scalar(name, title=None):
    global THICKNESS_SCALAR_NAME, THICKNESS_SCALAR_TITLE
    THICKNESS_SCALAR_NAME = name
    THICKNESS_SCALAR_TITLE = title or name

def compute_cluster_sizes(clusters_array):
    arr = np.asarray(clusters_array, dtype=int)
    mask = arr > 0
    labels, counts = np.unique(arr[mask], return_counts=True)
    return {int(l): int(c) for l, c in zip(labels, counts)}

def make_facies_lut():
    all_facies_keys = list(FACIES_COLORS.keys())
    max_fac = max(all_facies_keys) if all_facies_keys else 255
    lut = pv.LookupTable(n_values=max_fac + 1)
    for v in range(max_fac + 1):
        lut.SetTableValue(v, 0.8, 0.8, 0.8, 1.0) # Default cinza
    for fac, rgba in FACIES_COLORS.items():
        if fac <= max_fac:
            lut.SetTableValue(fac, *rgba)
    return lut, (0, max_fac)

def make_clusters_lut(clusters_arr):
    labels = np.unique(clusters_arr)
    labels = labels[labels > 0]
    n = len(labels)
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(n + 1)
    lut.Build()
    lut.SetTableValue(0, 0.2, 0.2, 0.2, 1.0)
    series = vtk.vtkColorSeries()
    series.SetColorScheme(series.BREWER_QUALITATIVE_SET3)
    n_colors = series.GetNumberOfColors()
    for idx, lab in enumerate(labels, start=1):
        color = series.GetColor(idx % n_colors)
        lut.SetTableValue(idx, color.GetRed()/255.0, color.GetGreen()/255.0, color.GetBlue()/255.0, 1.0)
    return lut, (0, n + 1)

def prepare_grid_k_index(target_grid, nz_dim):
    if "k_index" not in target_grid.cell_data:
        k3d = np.zeros((nx, ny, nz_dim), dtype=int)
        for k in range(nz_dim):
            k3d[:, :, k] = nz_dim - 1 - k
        target_grid.cell_data["k_index"] = k3d.reshape(-1, order="F")
    return target_grid

# =============================================================================
# CÁLCULO VERTICAL LOCAL
# =============================================================================
def _calc_vertical_metrics(target_grid, facies_array, reservoir_set):
    keys = [
        "vert_Ttot_reservoir", "vert_NTG_col_reservoir", "vert_NTG_env_reservoir",
        "vert_n_packages_reservoir", "vert_Tpack_max_reservoir", "vert_ICV_reservoir",
        "vert_Qv_reservoir", "vert_Qv_abs_reservoir"
    ]
    data_map = {k: np.zeros((nx, ny, nz), dtype=float) for k in keys}

    if target_grid.n_cells != nx * ny * nz: return
        
    centers = target_grid.cell_centers().points
    z_vals = centers[:, 2]
    
    fac_3d = facies_array.reshape((nx, ny, nz), order="F")
    z_3d = z_vals.reshape((nx, ny, nz), order="F")
    res_set = set(reservoir_set)
    
    for ix in range(nx):
        for iy in range(ny):
            col_fac = fac_3d[ix, iy, :]
            col_z = z_3d[ix, iy, :]
            mask = np.isin(col_fac, list(res_set))
            if not np.any(mask): continue
            
            z_min, z_max = np.nanmin(col_z), np.nanmax(col_z)
            T_col = abs(z_max - z_min)
            if T_col == 0: continue
            dz = T_col / nz
            
            idx = np.where(mask)[0]
            n_res = len(idx)
            T_tot = n_res * dz
            
            if n_res > 0:
                T_env = (idx[-1] - idx[0] + 1) * dz
            else: T_env = 0
                
            NTG_col = T_tot / T_col
            NTG_env = T_tot / T_env if T_env > 0 else 0
            
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
            
            T_pack_max = max(packages) * dz if packages else 0
            ICV = T_pack_max / T_env if T_env > 0 else 0
            Qv = NTG_col * ICV
            Qv_abs = ICV * (T_pack_max / T_col)

            data_map["vert_Ttot_reservoir"][ix, iy, mask] = T_tot
            data_map["vert_NTG_col_reservoir"][ix, iy, mask] = NTG_col
            data_map["vert_NTG_env_reservoir"][ix, iy, mask] = NTG_env
            data_map["vert_n_packages_reservoir"][ix, iy, mask] = len(packages)
            data_map["vert_Tpack_max_reservoir"][ix, iy, mask] = T_pack_max
            data_map["vert_ICV_reservoir"][ix, iy, mask] = ICV
            data_map["vert_Qv_reservoir"][ix, iy, mask] = Qv
            data_map["vert_Qv_abs_reservoir"][ix, iy, mask] = Qv_abs

    for name, arr_3d in data_map.items():
        target_grid.cell_data[name] = arr_3d.reshape(-1, order="F")


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================
def run(
    mode="facies", 
    z_exag=15.0, 
    show_scalar_bar=False, 
    external_plotter=None, 
    external_state=None,
    target_grid=None,
    target_facies=None
):
    from load_data import facies as global_facies
    
    use_grid = target_grid if target_grid is not None else global_grid
    use_facies = target_facies if target_facies is not None else global_facies

    prepare_grid_k_index(use_grid, nz)

    if external_plotter is not None:
        plotter = external_plotter
    else:
        plotter = pv.Plotter()

    state = external_state if external_state is not None else {}
    state["mode"] = mode
    state["z_exag"] = z_exag
    state["show_scalar_bar"] = show_scalar_bar
    state["current_grid_source"] = use_grid
    state["current_facies"] = use_facies
    state.setdefault("last_mode", None) 

    grid_base = use_grid.copy()
    grid_base.cell_data["Facies"] = use_facies

    x_min, x_max, y_min, y_max, z_min, z_max = grid_base.bounds
    grid_base.points[:, 1] = y_max - (grid_base.points[:, 1] - y_min)
    grid_base.points[:, 2] *= z_exag

    state.setdefault("bg_actor", None)
    state.setdefault("main_actor", None)
    state.setdefault("k_min", 0)
    state.setdefault("box_bounds", grid_base.bounds)
    
    state["thickness_presets"] = {
        "Espessura": ("vert_Ttot_reservoir", "Espessura total reservatório (m)"),
        "NTG coluna": ("vert_NTG_col_reservoir", "NTG coluna (reservatório)"),
        "NTG envelope": ("vert_NTG_env_reservoir", "NTG envelope (reservatório)"),
        "Maior pacote": ("vert_Tpack_max_reservoir", "Maior pacote vertical (m)"),
        "Nº pacotes": ("vert_n_packages_reservoir", "Número de pacotes verticais"),
        "ICV": ("vert_ICV_reservoir", "Índice de continuidade vertical (ICV)"),
        "Qv": ("vert_Qv_reservoir", "Índice combinado Qv"),
        "Qv absoluto": ("vert_Qv_abs_reservoir", "Índice de qualidade vertical absoluta (Qv_abs)"),
    }
    state.setdefault("thickness_mode", "Espessura")

    def attach_cell_data_from_original(clipped, original):
        if "vtkOriginalCellIds" not in clipped.cell_data: return clipped
        orig_ids = clipped.cell_data["vtkOriginalCellIds"]
        for name, arr in original.cell_data.items():
            clipped.cell_data[name] = arr[orig_ids]
        return clipped

    # --- Update Fields ---
    def update_reservoir_fields(reservoir_facies):
        current_f = state["current_facies"]
        rf_list = list(reservoir_facies) if reservoir_facies else []
        is_res = np.isin(current_f, rf_list).astype(np.int8)
        
        arr_xyz = is_res.reshape((nx, ny, nz), order="F")
        from scipy.ndimage import label as nd_label, generate_binary_structure
        structure = generate_binary_structure(3, 1)
        is_res_3d = arr_xyz.transpose(2, 1, 0) 
        labeled_3d, _ = nd_label(is_res_3d, structure=structure)
        clusters_1d = labeled_3d.transpose(2, 1, 0).reshape(-1, order="F").astype(np.int32)

        counts = np.bincount(labeled_3d.ravel())
        if counts.size > 0: counts[0] = 0
        largest_label = counts.argmax() if counts.size > 0 else 0
        largest_mask_1d = (clusters_1d == largest_label).astype(np.uint8)

        use_grid.cell_data["Reservoir"] = is_res.astype(np.uint8)
        use_grid.cell_data["Clusters"] = clusters_1d
        use_grid.cell_data["LargestCluster"] = largest_mask_1d
        
        _calc_vertical_metrics(use_grid, current_f, rf_list)

        sync_names = ["Reservoir", "Clusters", "LargestCluster", "NTG_local", "Facies"]
        for key in use_grid.cell_data.keys():
            if key.startswith("vert_"): sync_names.append(key)
        
        for name in sync_names:
            if name in use_grid.cell_data:
                grid_base.cell_data[name] = use_grid.cell_data[name].copy()

        state["clusters_lut"], state["clusters_rng"] = make_clusters_lut(clusters_1d)
        state["clusters_sizes"] = compute_cluster_sizes(clusters_1d)
        _update_thickness_from_state()

    state["update_reservoir_fields"] = update_reservoir_fields

    def _update_thickness_from_state():
        presets = state.get("thickness_presets") or {}
        mode_label = state.get("thickness_mode", "Espessura")
        if mode_label in presets:
            s_name, s_title = presets[mode_label]
            set_thickness_scalar(s_name, s_title)
            if s_name in grid_base.cell_data:
                arr = grid_base.cell_data[s_name]
                vmin = 1e-6
                vmax = float(arr.max())
                state["thickness_clim"] = (vmin, vmax if vmax > vmin else vmin + 1)

    state["update_thickness"] = _update_thickness_from_state
    
    if "Reservoir" not in grid_base.cell_data:
        update_reservoir_fields([])
    else:
        if not any(k.startswith("vert_") for k in grid_base.cell_data.keys()):
             update_reservoir_fields([])

    # --- Filtros ---
    def apply_k_filter(mesh):
        kmin = state.get("k_min", 0)
        if kmin <= 0: return mesh
        if "k_index" in mesh.cell_data:
            k_arr = mesh.cell_data["k_index"]
        elif "vtkOriginalCellIds" in mesh.cell_data and "k_index" in grid_base.cell_data:
            orig_ids = mesh.cell_data["vtkOriginalCellIds"]
            k_arr = grid_base.cell_data["k_index"][orig_ids]
        else: return mesh
        idx = np.where(k_arr >= kmin)[0]
        return mesh.extract_cells(idx) if idx.size > 0 else mesh.extract_cells([])

    # --- UPDATER GENÉRICO ---
    def _update_mapper_generic(actor, mesh, scalar_name=None, cmap=None, clim=None, lut=None, show_scalar=True):
        """Atualiza apenas o input, mantendo propriedades se não mudarem."""
        mapper = actor.mapper
        mapper.SetInputData(mesh)
        
        # Se pedir atualização de cor, aplica
        if scalar_name:
            mapper.SetScalarModeToUseCellFieldData()
            mapper.SelectColorArray(scalar_name)
            mapper.SetScalarVisibility(show_scalar)
        
        if lut: mapper.lookup_table = lut
        if clim: mapper.scalar_range = clim
        
        mapper.Update()

    # --- CLEANER DE BARRAS ---
    def _clean_all_bars(plotter):
        if hasattr(plotter, "scalar_bars"):
            for title in list(plotter.scalar_bars.keys()):
                plotter.remove_scalar_bar(title=title)

    # --- Render Principal ---
    def show_mesh(mesh):
        mode = state["mode"]
        last_mode = state.get("last_mode")
        
        mesh = attach_cell_data_from_original(mesh, grid_base)
        mesh = apply_k_filter(mesh)
        
        # [DECISÃO NUCLEAR]: Se mudou de modo, DESTRÓI os atores para garantir cor limpa.
        # Se for o mesmo modo (apenas corte/scroll), atualiza o mapper.
        needs_full_reset = (mode != last_mode) or (state["main_actor"] is None)

        if needs_full_reset:
            if state["bg_actor"]: 
                try: plotter.remove_actor(state["bg_actor"])
                except: pass
                state["bg_actor"] = None
            if state["main_actor"]:
                try: plotter.remove_actor(state["main_actor"])
                except: pass
                state["main_actor"] = None
            
            # Reativa box widget se tiver sumido
            if state.get("box_widget"): state["box_widget"].On()
            state["last_mode"] = mode

        _clean_all_bars(plotter)
        show_sb = state.get("show_scalar_bar", False)

        # --- LÓGICA POR MODO ---

        if mode == "facies":
            lut, rng = make_facies_lut()
            if not needs_full_reset and state["main_actor"]:
                _update_mapper_generic(state["main_actor"], mesh)
            else:
                act = plotter.add_mesh(mesh, scalars="Facies", show_edges=True, reset_camera=False, show_scalar_bar=False)
                act.mapper.lookup_table = lut
                act.mapper.scalar_range = rng
                state["main_actor"] = act

        elif mode == "reservoir":
            try:
                bg = mesh.threshold(0.5, invert=True, scalars="Reservoir")
                main = mesh.threshold(0.5, scalars="Reservoir")
            except:
                bg = mesh
                main = mesh.extract_cells([])

            # BG
            if bg.n_cells > 0:
                if not needs_full_reset and state["bg_actor"]:
                    _update_mapper_generic(state["bg_actor"], bg)
                else:
                    state["bg_actor"] = plotter.add_mesh(bg, color=(0.8,0.8,0.8), opacity=0.02, show_edges=False)
            else:
                if state["bg_actor"]: 
                    plotter.remove_actor(state["bg_actor"])
                    state["bg_actor"] = None

            # Main
            if main.n_cells > 0:
                lut, rng = make_facies_lut()
                if not needs_full_reset and state["main_actor"]:
                    _update_mapper_generic(state["main_actor"], main)
                else:
                    act = plotter.add_mesh(main, scalars="Facies", opacity=1.0, show_edges=True, show_scalar_bar=False)
                    act.mapper.lookup_table = lut
                    act.mapper.scalar_range = rng
                    state["main_actor"] = act
            else:
                if state["main_actor"]: state["main_actor"].SetVisibility(False)

        elif mode == "clusters":
            try:
                bg = mesh.threshold(0.5, invert=True, scalars="Clusters")
                main = mesh.threshold(0.5, scalars="Clusters")
            except:
                bg = mesh
                main = mesh.extract_cells([])
            
            if bg.n_cells > 0:
                if not needs_full_reset and state["bg_actor"]:
                    _update_mapper_generic(state["bg_actor"], bg)
                else:
                    state["bg_actor"] = plotter.add_mesh(bg, color=(0.8,0.8,0.8), opacity=0.02, show_edges=False)
            
            if main.n_cells > 0:
                lut = state.get("clusters_lut")
                rng = state.get("clusters_rng")
                if not lut: lut, rng = make_clusters_lut(grid_base.cell_data["Clusters"])

                if not needs_full_reset and state["main_actor"]:
                    _update_mapper_generic(state["main_actor"], main)
                else:
                    act = plotter.add_mesh(main, scalars="Clusters", show_edges=True, show_scalar_bar=False)
                    act.mapper.lookup_table = lut
                    act.mapper.scalar_range = rng
                    state["main_actor"] = act
            else:
                if state["main_actor"]: state["main_actor"].SetVisibility(False)

        elif mode == "thickness_local":
            _update_thickness_from_state()
            s_name = THICKNESS_SCALAR_NAME
            if s_name in mesh.cell_data:
                try:
                    bg = mesh.threshold(1e-6, invert=True, scalars=s_name)
                    main = mesh.threshold(1e-6, scalars=s_name)
                    
                    if bg.n_cells > 0:
                         if not needs_full_reset and state["bg_actor"]:
                             _update_mapper_generic(state["bg_actor"], bg)
                         else:
                             state["bg_actor"] = plotter.add_mesh(bg, color=(0.8,0.8,0.8), opacity=0.01, show_edges=False)
                    
                    if main.n_cells > 0:
                        clim = state.get("thickness_clim")
                        if not needs_full_reset and state["main_actor"]:
                            _update_mapper_generic(state["main_actor"], main, scalar_name=s_name, cmap="plasma", clim=clim, show_scalar=True)
                        else:
                            # Cria ator novo
                            act = plotter.add_mesh(main, scalars=s_name, cmap="plasma", clim=clim, show_edges=True, show_scalar_bar=False)
                            state["main_actor"] = act
                        
                        # Adiciona legenda
                        plotter.add_scalar_bar(title=THICKNESS_SCALAR_TITLE)

                    else:
                        if state["main_actor"]: state["main_actor"].SetVisibility(False)
                except: pass

        elif mode == "ntg_local":
            if "NTG_local" in mesh.cell_data:
                if state["bg_actor"]: 
                    try: plotter.remove_actor(state["bg_actor"])
                    except: pass
                    state["bg_actor"] = None

                if not needs_full_reset and state["main_actor"]:
                     _update_mapper_generic(state["main_actor"], mesh, scalar_name="NTG_local", cmap="plasma", clim=[0,1], show_scalar=True)
                else:
                    act = plotter.add_mesh(mesh, scalars="NTG_local", cmap="plasma", clim=[0,1], show_edges=True, show_scalar_bar=False)
                    state["main_actor"] = act
                
                plotter.add_scalar_bar(title="NTG Local")
        
        elif mode == "largest":
             try:
                 bg = mesh.threshold(0.5, invert=True, scalars="LargestCluster")
                 main = mesh.threshold(0.5, scalars="LargestCluster")
             except:
                 bg = mesh
                 main = mesh.extract_cells([])
             
             if bg.n_cells > 0:
                 if not needs_full_reset and state["bg_actor"]:
                     _update_mapper_generic(state["bg_actor"], bg)
                 else:
                     state["bg_actor"] = plotter.add_mesh(bg, color=(0.8,0.8,0.8), opacity=0.02, show_edges=False)
             
             if main.n_cells > 0:
                 if not needs_full_reset and state["main_actor"]:
                     _update_mapper_generic(state["main_actor"], main)
                 else:
                     state["main_actor"] = plotter.add_mesh(main, color="lightcoral", opacity=1.0, show_edges=True)
             else:
                 if state["main_actor"]: state["main_actor"].SetVisibility(False)

    # --- Callbacks ---
    def change_k(delta):
        kmin = state.get("k_min", 0)
        new = int(np.clip(kmin + delta, 0, nz - 1))
        state["k_min"] = new
        _refresh()
        if state.get("on_k_changed"): state["on_k_changed"](new)

    def _refresh():
        box = state.get("box_bounds", grid_base.bounds)
        base = grid_base.clip_box(box, invert=False, crinkle=True)
        show_mesh(base)

    def box_callback(box):
        state["box_bounds"] = box
        _refresh()
        if state.get("on_box_changed"): state["on_box_changed"](box)

    # --- Setup Final ---
    
    # Limpeza inicial
    plotter.clear_actors()
    _clean_all_bars(plotter)
    state["bg_actor"] = None
    state["main_actor"] = None
    state["last_mode"] = None

    plotter.add_key_event("z", lambda: change_k(-1))
    plotter.add_key_event("x", lambda: change_k(+1))
    
    bw = plotter.add_box_widget(
        callback=box_callback, 
        bounds=grid_base.bounds, 
        rotation_enabled=False, 
        interaction_event="always"
    )
    prop = bw.GetHandleProperty()
    prop.SetOpacity(1.0)
    prop.SetColor(0.9, 0.9, 0.9)
    prop.SetAmbient(0.2)
    prop.SetDiffuse(0.8)
    prop.SetSpecular(0.5)
    
    bw.GetSelectedHandleProperty().SetOpacity(1.0)
    bw.GetSelectedHandleProperty().SetColor(1.0, 0.6, 0.6)
    bw.GetOutlineProperty().SetOpacity(0.2)
    state["box_widget"] = bw

    plotter.set_background("white", top="lightgray")
    plotter.enable_lightkit()
    plotter.add_axes()
    
    _refresh()
    plotter.reset_camera()
    
    state["refresh"] = _refresh
    return plotter, state


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def show_thickness_2d(surf, scalar_name, title=None):
    p = pv.Plotter(window_size=(1000, 800))
    p.add_mesh(
        surf,
        scalars=scalar_name,
        cmap="plasma",
        show_edges=True,
        edge_color="black",
        line_width=0.5,
        show_scalar_bar=False,
        nan_color="white",
        preference="cell",
    )
    p.view_xy()
    p.enable_parallel_projection()
    p.enable_image_style()
    p.set_background("white")
    p.add_scalar_bar(title=title if title else scalar_name)
    p.show()

def update_2d_plot(plotter, array_name_3d, title="Mapa 2D"):
    surf = make_thickness_2d_from_grid(array_name_3d, array_name_3d + "_2d")
    scalar_name_2d = array_name_3d + "_2d"
    arr = surf.cell_data[scalar_name_2d]
    arr = np.where(arr < 0, np.nan, arr)
    surf.cell_data[scalar_name_2d] = arr
    
    plotter.clear()
    plotter.add_mesh(surf, scalars=scalar_name_2d, cmap="plasma", show_edges=True, edge_color="black", line_width=0.5, nan_color="white", show_scalar_bar=False)
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.set_background("white")
    plotter.add_scalar_bar(title=title)
    plotter.reset_camera()