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
# Faixas padrão para mapas 2D 
THICKNESS_2D_CLIM = {
    # métricas normalizadas (0–1)
    "vert_NTG_col_reservoir": (0.0, 1.0),
    "vert_NTG_env_reservoir": (0.0, 1.0),
    "vert_ICV_reservoir": (0.0, 1.0),
    "vert_Qv_reservoir": (0.0, 1.0),
    "vert_Qv_abs_reservoir": (0.0, 1.0),

    # espessuras em metros
    "vert_Ttot_reservoir": (0.0, 200.0),
    "vert_Tpack_max_reservoir": (0.0, 200.0),
}

MODE = "facies"
Z_EXAG = 15.0
SHOW_SCALAR_BAR = False

def get_2d_clim(base_scalar_name, arr=None):
    """
    Retorna (vmin, vmax) para mapas 2D.
    Se houver preset para o campo, usa o preset.
    Caso contrário, usa os próprios dados como fallback.
    """
    preset = THICKNESS_2D_CLIM.get(base_scalar_name)
    if preset is not None:
        return preset

    if arr is None:
        return None

    arr = np.asarray(arr)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None

    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return (vmin, vmax)


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
    
    # Inicializa com as fácies corretas
    grid_base.cell_data["Facies"] = use_facies

    # Aplica exagero Z
    grid_base.points[:, 2] *= z_exag

    state.setdefault("bg_actor", None)
    state.setdefault("main_actor", None)
    
    # --- ESTADOS DE CORTE ---
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

    def update_reservoir_fields(reservoir_facies):
        # Pega o grid e facies que estão ATIVOS no momento
        current_g = state.get("current_grid_source", use_grid)
        current_f = state.get("current_facies", use_facies)
        
        # --- CORREÇÃO CRÍTICA ---
        # Garante que o grid de cálculo tenha as Fácies corretas antes de calcular qualquer coisa.
        # Isso evita que ele use dados antigos/errados que estavam salvos dentro do objeto grid.
        if current_f is not None:
            current_g.cell_data["Facies"] = current_f

        rf_list = list(reservoir_facies) if reservoir_facies else []
        is_res = np.isin(current_f, rf_list).astype(np.int8)
        
        try:
            arr_xyz = is_res.reshape((nx, ny, nz), order="F")
        except ValueError:
            return 

        structure = generate_binary_structure(3, 1)
        is_res_3d = arr_xyz.transpose(2, 1, 0) 
        labeled_3d, _ = nd_label(is_res_3d, structure=structure)
        clusters_1d = labeled_3d.transpose(2, 1, 0).reshape(-1, order="F").astype(np.int32)

        counts = np.bincount(labeled_3d.ravel())
        if counts.size > 0: counts[0] = 0
        largest_label = counts.argmax() if counts.size > 0 else 0
        largest_mask_1d = (clusters_1d == largest_label).astype(np.uint8)

        # Grava os dados calculados no grid ATUAL
        current_g.cell_data["Reservoir"] = is_res.astype(np.uint8)
        current_g.cell_data["Clusters"] = clusters_1d
        current_g.cell_data["LargestCluster"] = largest_mask_1d
        
        # Recalcula métricas verticais
        _calc_vertical_metrics(current_g, current_f, rf_list) 

        # Sincroniza campos calculados com o grid base de visualização
        sync_names = ["Reservoir", "Clusters", "LargestCluster", "NTG_local", "Facies", "i_index", "j_index", "k_index"]
        
        for key in current_g.cell_data.keys():
            if key.startswith("vert_") or key in sync_names:
                grid_base.cell_data[key] = current_g.cell_data[key]

        state["clusters_lut"], state["clusters_rng"] = make_clusters_lut(clusters_1d)
        state["clusters_sizes"] = compute_cluster_sizes(clusters_1d)
        _update_thickness_from_state()

    state["update_reservoir_fields"] = update_reservoir_fields

    def _update_thickness_from_state():
        presets = state.get("thickness_presets") or {}
        mode_label = state.get("thickness_mode", "Espessura")
        
        if mode_label in presets:
            s_name, s_title = presets[mode_label]
            state["current_thickness_scalar"] = s_name
            state["current_thickness_title"] = s_title
            
            if s_name in grid_base.cell_data:
                arr = grid_base.cell_data[s_name]
                vmin = 1e-6
                vmax = float(np.nanmax(arr)) 
                if np.isnan(vmax): vmax = 1.0
                state["thickness_clim"] = (vmin, vmax if vmax > vmin else vmin + 1)

    state["update_thickness"] = _update_thickness_from_state
    
    if "Reservoir" not in grid_base.cell_data:
        update_reservoir_fields([])

    # --- FILTRO UNIFICADO (I, J, K - Min/Max) ---
    def apply_slices_filter(mesh):
        kmin, kmax = state.get("k_min", 0), state.get("k_max", nz-1)
        imin, imax = state.get("i_min", 0), state.get("i_max", nx-1)
        jmin, jmax = state.get("j_min", 0), state.get("j_max", ny-1)
        
        if kmin > kmax: kmin = kmax
        if imin > imax: imin = imax
        if jmin > jmax: jmin = jmax

        if (kmin == 0 and kmax == nz-1 and 
            imin == 0 and imax == nx-1 and 
            jmin == 0 and jmax == ny-1):
            return mesh

        try:
            if mesh is None or mesh.n_cells == 0: return mesh
        except Exception: return mesh

        if not ("i_index" in mesh.cell_data and "j_index" in mesh.cell_data and "k_index" in mesh.cell_data):
            try: prepare_grid_indices(mesh)
            except Exception: pass

        if "i_index" in mesh.cell_data and "j_index" in mesh.cell_data and "k_index" in mesh.cell_data:
            try:
                i = mesh.cell_data["i_index"]
                j = mesh.cell_data["j_index"]
                k = mesh.cell_data["k_index"]
                mask = (i >= imin) & (i <= imax) & \
                       (j >= jmin) & (j <= jmax) & \
                       (k >= kmin) & (k <= kmax)
                out = mesh.extract_cells(mask)
                return out
            except Exception: pass

        out = mesh
        if (len(getattr(out, "cell_data", {})) == 0) and (len(getattr(out, "point_data", {})) == 0):
            return out

        if (kmin > 0 or kmax < nz-1) and ("k_index" in out.cell_data or "k_index" in out.point_data):
            out = out.threshold([kmin, kmax], scalars="k_index")
        if (imin > 0 or imax < nx-1) and ("i_index" in out.cell_data or "i_index" in out.point_data):
            out = out.threshold([imin, imax], scalars="i_index")
        if (jmin > 0 or jmax < ny-1) and ("j_index" in out.cell_data or "j_index" in out.point_data):
            out = out.threshold([jmin, jmax], scalars="j_index")

        return out

    def _clean_all_bars(plotter):
        try:
            if hasattr(plotter, 'scalar_bars'):
                keys = list(plotter.scalar_bars.keys())
                for k in keys:
                    plotter.remove_scalar_bar(k)
        except Exception: pass

    def show_mesh(mesh):
        mode = state["mode"]

        mesh = attach_cell_data_from_original(mesh, grid_base)

        try:
            if mesh is not None and mesh.n_cells > 0:
                if not ("i_index" in mesh.cell_data and "j_index" in mesh.cell_data and "k_index" in mesh.cell_data):
                    prepare_grid_indices(mesh)
        except Exception: pass

        mesh = apply_slices_filter(mesh)

        _clean_all_bars(plotter)
        mesh_main = None
        mesh_bg = None
        scalar_name = None
        lut = None
        clim = None
        cmap = None
        show_scalar = True
        color_main = None
        opacity_main = 1.0
        bar_title = ""
        
        if mode == "facies":
            mesh_main = mesh
            scalar_name = "Facies"
            lut, clim = make_facies_lut()
            
        elif mode == "reservoir":
            try:
                mesh_bg = mesh.threshold(0.5, invert=True, scalars="Reservoir")
                mesh_main = mesh.threshold(0.5, scalars="Reservoir")
            except: mesh_bg, mesh_main = mesh, None
            scalar_name = "Facies"
            lut, clim = make_facies_lut()
            
        elif mode == "clusters":
            try:
                mesh_bg = mesh.threshold(0.5, invert=True, scalars="Clusters")
                mesh_main = mesh.threshold(0.5, scalars="Clusters")
            except: mesh_bg, mesh_main = mesh, None
            scalar_name = "Clusters"
            lut = state.get("clusters_lut")
            clim = state.get("clusters_rng")
            if not lut: lut, clim = make_clusters_lut(grid_base.cell_data["Clusters"])

        elif mode == "largest":
            try:
                mesh_bg = mesh.threshold(0.5, invert=True, scalars="LargestCluster")
                mesh_main = mesh.threshold(0.5, scalars="LargestCluster")
            except: mesh_bg, mesh_main = mesh, None
            show_scalar = False
            color_main = "lightcoral"

        elif mode == "thickness_local":
            _update_thickness_from_state()
            s_name = state.get("current_thickness_scalar", THICKNESS_SCALAR_NAME)
            bar_title = state.get("current_thickness_title", THICKNESS_SCALAR_TITLE)
            
            if s_name in mesh.cell_data:
                try:
                    mesh_bg = mesh.threshold(1e-6, invert=True, scalars=s_name)
                    mesh_main = mesh.threshold(1e-6, scalars=s_name)
                except: mesh_bg, mesh_main = mesh, None
                
                scalar_name = s_name
                cmap = state.get("thickness_cmap", "plasma") 
                
                if mesh_main and mesh_main.n_cells > 0:
                    arr = mesh_main.cell_data[s_name]
                    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
                    if vmax <= vmin: vmax = vmin + 1e-6
                    clim = (vmin, vmax)
                else:
                    clim = (0.0, 1.0)

            else:
                # Propriedade não existe neste grid -> deixa transparente
                show_scalar = False
                opacity_main = 0.0

        elif mode == "scalar":
            s_name = state.get("current_scalar_name")
            bar_title = state.get("current_scalar_title", s_name)
            
            if s_name and s_name in mesh.cell_data:
                mesh_main = mesh # Mostra geometria completa (incluindo zeros)
                scalar_name = s_name
                cmap = state.get("current_scalar_cmap", "viridis")
                clim = state.get("current_scalar_clim", None)
                
                if clim is None:
                    arr = mesh.cell_data[s_name]
                    v_valid = arr[np.isfinite(arr)]
                    if v_valid.size > 0:
                        clim = (float(v_valid.min()), float(v_valid.max()))
                    else:
                        clim = (0.0, 1.0)
            else:
                mesh_main = mesh
                show_scalar = False
                opacity_main = 0.0

        def sync_actor(actor_key, mesh_data, is_bg=False):
            actor = state.get(actor_key)
            if mesh_data is None or mesh_data.n_cells == 0:
                if actor: actor.SetVisibility(False)
                return actor

            if show_scalar and scalar_name:
                if scalar_name in mesh_data.cell_data:
                    mesh_data.set_active_scalars(scalar_name, preference="cell")
            
            if actor is None:
                if is_bg:
                    actor = plotter.add_mesh(mesh_data, color=(0.8,0.8,0.8), opacity=0.02, show_edges=False, reset_camera=False)
                else:
                    actor = plotter.add_mesh(mesh_data, show_edges=True, reset_camera=False, show_scalar_bar=False)
                state[actor_key] = actor
            
            actor.SetVisibility(True)
            actor.mapper.SetInputData(mesh_data)

            # Garante que não "vaze" scalar visibility entre modos
            if not show_scalar:
                try:
                    actor.mapper.SetScalarVisibility(False)
                except Exception:
                    pass

            if is_bg: return actor

            # Aplica opacidade (para deixar "vazio" quando faltar propriedade)
            try:
                actor.GetProperty().SetOpacity(float(opacity_main))
            except Exception:
                pass

            if show_scalar and scalar_name and scalar_name in mesh_data.cell_data:
                actor.mapper.SetScalarVisibility(True)
                actor.mapper.SetScalarModeToUseCellFieldData()
                actor.mapper.SelectColorArray(scalar_name)
                
                if lut:
                    actor.mapper.SetLookupTable(lut)
                    if clim: actor.mapper.SetScalarRange(clim)
                elif cmap:
                    new_lut = pv.LookupTable(cmap, n_values=256)
                    if clim: new_lut.SetRange(clim)
                    actor.mapper.SetLookupTable(new_lut)
                    if clim: actor.mapper.SetScalarRange(clim)
            else:
                actor.mapper.SetScalarVisibility(False)
                if color_main: actor.prop.color = color_main
            
            actor.prop.opacity = opacity_main
            return actor

        sync_actor("bg_actor", mesh_bg, is_bg=True)
        main_actor = sync_actor("main_actor", mesh_main, is_bg=False)

        z_scale = state.get("z_exag", 15.0)
        
        if state.get("bg_actor"): 
            state["bg_actor"].SetScale(1.0, 1.0, z_scale)
            
        if main_actor: 
            main_actor.SetScale(1.0, 1.0, z_scale)
            if (mode == "thickness_local" or mode == "scalar") and bar_title:
                plotter.add_scalar_bar(title=bar_title, mapper=main_actor.mapper, n_labels=5, fmt="%.1f")

    def _refresh():
        new_source = state.get("current_grid_source")
        nonlocal grid_base
        if new_source is not None:
            grid_base = new_source
            rf = state.get("reservoir_facies", set())
            update_reservoir_fields(rf)
        show_mesh(grid_base)

    def set_slice(axis, mode, value):
        key = f"{axis}_{mode}"
        limit = 0
        if axis == "k": limit = nz-1
        elif axis == "i": limit = nx-1
        elif axis == "j": limit = ny-1
        val = int(np.clip(value, 0, limit))
        state[key] = val
        _refresh()
        if state.get("on_slice_changed"):
            state["on_slice_changed"](axis, mode, val)

    state["set_slice"] = set_slice

    def key_change_slice(axis, mode, delta):
        key = f"{axis}_{mode}"
        curr = state.get(key, 0)
        limit = 0
        if axis == "k": limit = nz-1
        elif axis == "i": limit = nx-1
        elif axis == "j": limit = ny-1
        
        new_val = int(np.clip(curr + delta, 0, limit))
        if mode == "min":
            max_val = state.get(f"{axis}_max", limit)
            if new_val > max_val: new_val = max_val
        else:
            min_val = state.get(f"{axis}_min", 0)
            if new_val < min_val: new_val = min_val
            
        state[key] = new_val
        _refresh()
        if state.get("on_slice_changed"):
            state["on_slice_changed"](axis, mode, new_val)

    plotter.clear_actors()
    _clean_all_bars(plotter)
    state["bg_actor"] = None
    state["main_actor"] = None
    state["last_mode"] = None
    
    # Bindings
    plotter.add_key_event("z", lambda: key_change_slice("k", "min", -1))
    plotter.add_key_event("x", lambda: key_change_slice("k", "min", +1))
    plotter.add_key_event("1", lambda: key_change_slice("k", "max", -1))
    plotter.add_key_event("2", lambda: key_change_slice("k", "max", +1))
    
    plotter.add_key_event("c", lambda: key_change_slice("i", "min", -1))
    plotter.add_key_event("b", lambda: key_change_slice("i", "min", +1))
    plotter.add_key_event("4", lambda: key_change_slice("i", "max", -1))
    plotter.add_key_event("5", lambda: key_change_slice("i", "max", +1))

    plotter.add_key_event("n", lambda: key_change_slice("j", "min", -1))
    plotter.add_key_event("m", lambda: key_change_slice("j", "min", +1))
    plotter.add_key_event("7", lambda: key_change_slice("j", "max", -1))
    plotter.add_key_event("8", lambda: key_change_slice("j", "max", +1))

    if "box_widget" in state: del state["box_widget"]

    plotter.enable_lightkit()
    plotter.add_axes()
    
    _refresh()
    plotter.reset_camera()
    
    state["refresh"] = _refresh
    return plotter, state

def show_thickness_2d(surf, scalar_name, title=None, cmap="plasma"):
    # 1) limpa valores negativos (viram NaN)
    arr = surf.cell_data[scalar_name]
    arr = np.where(arr < 0, np.nan, arr)
    surf.cell_data[scalar_name] = arr

    # 2) descobre o nome base (sem _2d) para buscar a faixa padrão
    base_name = scalar_name[:-4] if scalar_name.endswith("_2d") else scalar_name
    clim = get_2d_clim(base_name, arr)

    # 3) plota usando clim fixo (se houver)
    p = pv.Plotter(window_size=(1000, 800))
    p.add_mesh(
        surf,
        scalars=scalar_name,
        cmap=cmap,
        show_edges=True,
        edge_color="black",
        line_width=0.5,
        show_scalar_bar=False,
        nan_color="white",
        preference="cell",
        clim=clim,  # << chave para padronizar cores
    )
    p.view_xy()
    p.enable_parallel_projection()
    p.enable_image_style()
    p.set_background("white")
    p.add_scalar_bar(title=title if title else scalar_name)
    p.show()


def update_2d_plot(plotter, array_name_3d, title="Mapa 2D", cmap="plasma"):
    surf = make_thickness_2d_from_grid(array_name_3d, array_name_3d + "_2d")
    scalar_name_2d = array_name_3d + "_2d"

    # limpa negativos
    arr = surf.cell_data[scalar_name_2d]
    arr = np.where(arr < 0, np.nan, arr)
    surf.cell_data[scalar_name_2d] = arr

    # usa a mesma lógica de faixa fixa
    clim = get_2d_clim(array_name_3d, arr)

    plotter.clear()
    plotter.add_mesh(
        surf,
        scalars=scalar_name_2d,
        cmap="plasma",
        show_edges=True,
        edge_color="black",
        line_width=0.5,
        nan_color="white",
        show_scalar_bar=False,
        preference="cell",
        clim=clim,
    )
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.set_background("white")
    plotter.add_scalar_bar(title=title)
    plotter.reset_camera()
