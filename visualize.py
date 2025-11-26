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
        lut.SetTableValue(v, 0.8, 0.8, 0.8, 1.0) 
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

def prepare_grid_indices(target_grid):
    """
    Adiciona índices I, J, K (estruturais) como escalares no grid
    para permitir filtros de threshold (cortes) rápidos.
    """
    # K index (Bottom -> Top)
    if "k_index" not in target_grid.cell_data:
        k3d = np.zeros((nx, ny, nz), dtype=int)
        for k in range(nz):
            k3d[:, :, k] = nz - 1 - k # K=0 é base, K=nz-1 é topo
        target_grid.cell_data["k_index"] = k3d.reshape(-1, order="F")

    # I index (X axis)
    if "i_index" not in target_grid.cell_data:
        i3d = np.zeros((nx, ny, nz), dtype=int)
        for i in range(nx):
            i3d[i, :, :] = i
        target_grid.cell_data["i_index"] = i3d.reshape(-1, order="F")

    # J index (Y axis)
    if "j_index" not in target_grid.cell_data:
        j3d = np.zeros((nx, ny, nz), dtype=int)
        for j in range(ny):
            j3d[:, j, :] = j
        target_grid.cell_data["j_index"] = j3d.reshape(-1, order="F")

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

    # Prepara índices I, J, K para cortes
    prepare_grid_indices(use_grid)

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

    # Grid Base para visualização (Geometry Only)
    grid_base = use_grid.copy()
    grid_base.cell_data["Facies"] = use_facies

    # Aplica exagero Z
    x_min, x_max, y_min, y_max, z_min, z_max = grid_base.bounds
    grid_base.points[:, 1] = y_max - (grid_base.points[:, 1] - y_min)
    grid_base.points[:, 2] *= z_exag

    state.setdefault("bg_actor", None)
    state.setdefault("main_actor", None)
    
    # --- NOVOS ESTADOS DE CORTE (Min e Max para cada eixo) ---
    state.setdefault("k_min", 0)
    state.setdefault("k_max", nz - 1)
    
    state.setdefault("i_min", 0)
    state.setdefault("i_max", nx - 1)
    
    state.setdefault("j_min", 0)
    state.setdefault("j_max", ny - 1)
    
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

        sync_names = ["Reservoir", "Clusters", "LargestCluster", "NTG_local", "Facies", "i_index", "j_index", "k_index"]
        for key in use_grid.cell_data.keys():
            if key.startswith("vert_") or key in sync_names:
                grid_base.cell_data[key] = use_grid.cell_data[key].copy()

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

    # --- FILTRO UNIFICADO (I, J, K - Min/Max) ---
    def apply_slices_filter(mesh):
        """Aplica cortes ortogonais bidirecionais (Range) com proteção."""
        kmin, kmax = state.get("k_min", 0), state.get("k_max", nz-1)
        imin, imax = state.get("i_min", 0), state.get("i_max", nx-1)
        jmin, jmax = state.get("j_min", 0), state.get("j_max", ny-1)
        
        # --- SEGURANÇA CONTRA CRASH (Impede min > max) ---
        if kmin > kmax: kmin = kmax
        if imin > imax: imin = imax
        if jmin > jmax: jmin = jmax

        # Otimização: se estiver tudo completo, retorna o original sem filtrar
        if (kmin == 0 and kmax == nz-1 and 
            imin == 0 and imax == nx-1 and 
            jmin == 0 and jmax == ny-1):
            return mesh
            
        out = mesh
        
        # Filtro K (Range)
        if kmin > 0 or kmax < nz-1:
            out = out.threshold([kmin, kmax], scalars="k_index")
            
        # Filtro I (Range)
        if imin > 0 or imax < nx-1:
            out = out.threshold([imin, imax], scalars="i_index")
            
        # Filtro J (Range)
        if jmin > 0 or jmax < ny-1:
            out = out.threshold([jmin, jmax], scalars="j_index")
            
        return out

    def _update_mapper_generic(actor, mesh, scalar_name=None, cmap=None, clim=None, lut=None, show_scalar=True):
        mapper = actor.mapper
        mapper.SetInputData(mesh)
        if scalar_name:
            mapper.SetScalarModeToUseCellFieldData()
            mapper.SelectColorArray(scalar_name)
            mapper.SetScalarVisibility(show_scalar)
        if lut: mapper.lookup_table = lut
        if clim: mapper.scalar_range = clim
        mapper.Update()

    def _clean_all_bars(plotter):
        if hasattr(plotter, "scalar_bars"):
            for title in list(plotter.scalar_bars.keys()):
                plotter.remove_scalar_bar(title=title)

    def show_mesh(mesh):
        mode = state["mode"]
        last_mode = state.get("last_mode")
        
        # Anexa dados originais e APLICA CORTES
        mesh = attach_cell_data_from_original(mesh, grid_base)
        mesh = apply_slices_filter(mesh)
        
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
            state["last_mode"] = mode

        _clean_all_bars(plotter)

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

            if bg.n_cells > 0:
                if not needs_full_reset and state["bg_actor"]:
                    _update_mapper_generic(state["bg_actor"], bg)
                else:
                    state["bg_actor"] = plotter.add_mesh(bg, color=(0.8,0.8,0.8), opacity=0.02, show_edges=False)
            
            if main.n_cells > 0:
                lut, rng = make_facies_lut()
                if not needs_full_reset and state["main_actor"]:
                    _update_mapper_generic(state["main_actor"], main)
                else:
                    act = plotter.add_mesh(main, scalars="Facies", opacity=1.0, show_edges=True, show_scalar_bar=False)
                    act.mapper.lookup_table = lut
                    act.mapper.scalar_range = rng
                    state["main_actor"] = act

        elif mode == "clusters":
            try:
                bg = mesh.threshold(0.5, invert=True, scalars="Clusters")
                main = mesh.threshold(0.5, scalars="Clusters")
            except:
                bg, main = mesh, mesh.extract_cells([])
            
            if bg.n_cells > 0:
                if not needs_full_reset and state["bg_actor"]: _update_mapper_generic(state["bg_actor"], bg)
                else: state["bg_actor"] = plotter.add_mesh(bg, color=(0.8,0.8,0.8), opacity=0.02, show_edges=False)
            
            if main.n_cells > 0:
                lut = state.get("clusters_lut")
                rng = state.get("clusters_rng")
                if not lut: lut, rng = make_clusters_lut(grid_base.cell_data["Clusters"])
                if not needs_full_reset and state["main_actor"]: _update_mapper_generic(state["main_actor"], main)
                else:
                    act = plotter.add_mesh(main, scalars="Clusters", show_edges=True, show_scalar_bar=False)
                    act.mapper.lookup_table = lut
                    act.mapper.scalar_range = rng
                    state["main_actor"] = act

        elif mode == "thickness_local":
            _update_thickness_from_state()
            s_name = THICKNESS_SCALAR_NAME
            if s_name in mesh.cell_data:
                try:
                    bg = mesh.threshold(1e-6, invert=True, scalars=s_name)
                    main = mesh.threshold(1e-6, scalars=s_name)
                    if bg.n_cells > 0:
                         if not needs_full_reset and state["bg_actor"]: _update_mapper_generic(state["bg_actor"], bg)
                         else: state["bg_actor"] = plotter.add_mesh(bg, color=(0.8,0.8,0.8), opacity=0.01, show_edges=False)
                    if main.n_cells > 0:
                        clim = state.get("thickness_clim")
                        if not needs_full_reset and state["main_actor"]:
                            _update_mapper_generic(state["main_actor"], main, scalar_name=s_name, cmap="plasma", clim=clim, show_scalar=True)
                        else:
                            act = plotter.add_mesh(main, scalars=s_name, cmap="plasma", clim=clim, show_edges=True, show_scalar_bar=False)
                            state["main_actor"] = act
                        plotter.add_scalar_bar(title=THICKNESS_SCALAR_TITLE)
                except: pass

        elif mode == "largest":
             try:
                 bg = mesh.threshold(0.5, invert=True, scalars="LargestCluster")
                 main = mesh.threshold(0.5, scalars="LargestCluster")
             except: bg, main = mesh, mesh.extract_cells([])
             if bg.n_cells > 0:
                 if not needs_full_reset and state["bg_actor"]: _update_mapper_generic(state["bg_actor"], bg)
                 else: state["bg_actor"] = plotter.add_mesh(bg, color=(0.8,0.8,0.8), opacity=0.02, show_edges=False)
             if main.n_cells > 0:
                 if not needs_full_reset and state["main_actor"]: _update_mapper_generic(state["main_actor"], main)
                 else: state["main_actor"] = plotter.add_mesh(main, color="lightcoral", opacity=1.0, show_edges=True)

    def _refresh():
        show_mesh(grid_base)

    # --- FUNC PARA SETAR SLICE DO EXTERNO (UI) ---
    def set_slice(axis, mode, value):
        key = f"{axis}_{mode}"
        limit = 0
        if axis == "k": limit = nz-1
        elif axis == "i": limit = nx-1
        elif axis == "j": limit = ny-1
        
        val = int(np.clip(value, 0, limit))
        state[key] = val
        _refresh()
        
        # Notifica UI se necessário (opcional, evita loop infinito se controlado)
        if state.get("on_slice_changed"):
            state["on_slice_changed"](axis, mode, val)

    state["set_slice"] = set_slice

    # --- CONTROLE DE TECLADO ---
    def key_change_slice(axis, mode, delta):
        key = f"{axis}_{mode}"
        curr = state.get(key, 0)
        
        limit = 0
        if axis == "k": limit = nz-1
        elif axis == "i": limit = nx-1
        elif axis == "j": limit = ny-1
        
        new_val = int(np.clip(curr + delta, 0, limit))
        
        # Validação cruzada: min não pode passar max
        if mode == "min":
            max_val = state.get(f"{axis}_max", limit)
            if new_val > max_val: new_val = max_val
        else:
            min_val = state.get(f"{axis}_min", 0)
            if new_val < min_val: new_val = min_val
            
        state[key] = new_val
        _refresh()
        
        # Notifica a UI para atualizar os sliders
        if state.get("on_slice_changed"):
            state["on_slice_changed"](axis, mode, new_val)

    # --- Setup Final ---
    plotter.clear_actors()
    _clean_all_bars(plotter)
    state["bg_actor"] = None
    state["main_actor"] = None
    state["last_mode"] = None

    # --- BINDINGS SEGUROS (Evitando Q e W padrões) ---
    
    # Z (Camadas - K) -> Mantido
    plotter.add_key_event("t", lambda: key_change_slice("k", "min", -1))
    plotter.add_key_event("y", lambda: key_change_slice("k", "min", +1))
    plotter.add_key_event("g", lambda: key_change_slice("k", "max", -1))
    plotter.add_key_event("h", lambda: key_change_slice("k", "max", +1))
    
    # X (Frente/Trás - I) -> Usando Números 1-4
    plotter.add_key_event("1", lambda: key_change_slice("i", "min", -1))
    plotter.add_key_event("2", lambda: key_change_slice("i", "min", +1))
    plotter.add_key_event("3", lambda: key_change_slice("i", "max", -1))
    plotter.add_key_event("4", lambda: key_change_slice("i", "max", +1))

    # Y (Esq/Dir - J) -> Usando Números 5-8
    plotter.add_key_event("5", lambda: key_change_slice("j", "min", -1))
    plotter.add_key_event("6", lambda: key_change_slice("j", "min", +1))
    plotter.add_key_event("7", lambda: key_change_slice("j", "max", -1))
    plotter.add_key_event("8", lambda: key_change_slice("j", "max", +1))

    if "box_widget" in state: del state["box_widget"]

    # plotter.set_background("white", top="lightgray")
    plotter.enable_lightkit()
    plotter.add_axes()
    
    _refresh()
    plotter.reset_camera()
    
    state["refresh"] = _refresh
    return plotter, state

def show_thickness_2d(surf, scalar_name, title=None):
    p = pv.Plotter(window_size=(1000, 800))
    p.add_mesh(surf, scalars=scalar_name, cmap="plasma", show_edges=True, edge_color="black", line_width=0.5, show_scalar_bar=False, nan_color="white", preference="cell")
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