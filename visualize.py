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

    # Para alguns campos de "espessura" a escala NÃO deve ser fixa (ex.: 0–200),
    # porque isso atrapalha a comparação quando os modelos mudam. Para esses casos,
    # usamos os próprios dados (quando arr foi fornecido) e fixamos apenas o vmin em 0.
    if arr is not None and base_scalar_name in (
        "vert_Ttot_reservoir",
        "vert_Tpack_max_reservoir",
        "vert_n_packages_reservoir",
    ):
        arr = np.asarray(arr)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return (0.0, 1.0)
        vmax = float(np.nanmax(finite))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        return (0.0, vmax)

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
    # grid_base.points[:, 2] *= z_exag

    state.setdefault("base_bounds", grid_base.bounds)   # bounds no z_exag = 1
    state.setdefault("bounds_actor", None)
    state.setdefault("last_bounds_z", None)

    state.setdefault("bg_actor", None)
    state.setdefault("main_actor", None)
    state.setdefault("main_actor_data", None)
    
    # --- ESTADOS DE CORTE ---
    state.setdefault("k_min", 0)
    state.setdefault("k_max", nz - 1)
    state.setdefault("i_min", 0)
    state.setdefault("i_max", nx - 1)
    state.setdefault("j_min", 0)
    state.setdefault("j_max", ny - 1)
    
    default_presets = {
        "Espessura": ("vert_Ttot_reservoir", "Espessura total reservatório (m)"),
        "Proporção de fácies (coluna)": ("vert_NTG_col_reservoir", "Proporção de fácies (coluna)"),
        "Proporção de fácies (envelope)": ("vert_NTG_env_reservoir", "Proporção de fácies (envelope)"),
        "Maior pacote": ("vert_Tpack_max_reservoir", "Maior pacote (m)"),
        "Nº pacotes": ("vert_n_packages_reservoir", "Nº pacotes"),
        "ICV": ("vert_ICV_reservoir", "ICV"),
        "Qv": ("vert_Qv_reservoir", "Qv"),
        "Qv absoluto": ("vert_Qv_abs_reservoir", "Qv absoluto"),
    }


    existing = state.get("thickness_presets")
    if not isinstance(existing, dict):
        existing = {}

    merged = dict(existing)
    for k, v in default_presets.items():
        merged.setdefault(k, v)

    state["thickness_presets"] = merged

    # state.setdefault("thickness_mode", "Espessura")

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

        if mode_label not in presets:
            return

        s_name, s_title = presets[mode_label]
        state["current_thickness_scalar"] = s_name
        state["current_thickness_title"] = s_title

        if state.get("thickness_clim_manual", False):
            return

        if s_name in grid_base.cell_data:
            arr = grid_base.cell_data[s_name]
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                state["thickness_clim"] = (0.0, 1.0)
                return

            if ("Proporção de fácies" in s_name) or ("ICV" in s_name) or ("Qv" in s_name):
                state["thickness_clim"] = (0.0, 1.0)
                return

            if ("Ttot" in s_name) or ("thickness" in s_name.lower()) or ("Espessura" in mode_label):
                vmax = float(np.nanmax(finite))
                if not np.isfinite(vmax) or vmax <= 0:
                    vmax = 1.0
                state["thickness_clim"] = (0.0, vmax)
                return

            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
            if not np.isfinite(vmin): vmin = 0.0
            if not np.isfinite(vmax) or vmax <= vmin: vmax = vmin + 1e-6
            state["thickness_clim"] = (vmin, vmax)


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

        try:
            state["base_bounds"] = mesh.bounds
        except Exception:
            pass

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
            # Atualiza scalar/título conforme preset selecionado (sem pisar em clim manual)
            _update_thickness_from_state()

            s_name = state.get("current_thickness_scalar", THICKNESS_SCALAR_NAME)
            bar_title = state.get("current_thickness_title", THICKNESS_SCALAR_TITLE)

            if s_name in mesh.cell_data:
                try:
                    mesh_bg = mesh.threshold(1e-6, invert=True, scalars=s_name)
                    mesh_main = mesh.threshold(1e-6, scalars=s_name)
                except:
                    mesh_bg, mesh_main = mesh, None

                scalar_name = s_name
                cmap = state.get("thickness_cmap", "plasma")

                # 1) comparação pode setar global_clim
                if state.get("thickness_global_clim") is not None:
                    clim = state["thickness_global_clim"]
                else:
                    # 2) caso normal: usa sempre o clim do STATE (calculado no grid inteiro)
                    clim = state.get("thickness_clim")
                    if clim is None and (s_name in grid_base.cell_data):
                        arr = grid_base.cell_data[s_name]
                        finite = arr[np.isfinite(arr)]
                        if finite.size > 0:
                            clim = (float(np.nanmin(finite)), float(np.nanmax(finite)))
                        else:
                            clim = (0.0, 1.0)
            else:
                mesh_main = mesh
                show_scalar = False

        elif mode == "scalar":
            s_name = state.get("current_scalar_name")
            bar_title = state.get("current_scalar_title", s_name)

            if s_name and s_name in mesh.cell_data:
                mesh_main = mesh
                scalar_name = s_name
                cmap = state.get("current_scalar_cmap", "viridis")

                clim = state.get("current_scalar_clim")
                if clim is None:
                    # pega do grid inteiro (não do mesh cortado), para a legenda não variar com slices
                    if s_name in grid_base.cell_data:
                        arr_src = grid_base.cell_data[s_name]
                    else:
                        arr_src = mesh.cell_data[s_name]

                    finite = arr_src[np.isfinite(arr_src)]
                    if finite.size == 0:
                        clim = (0.0, 1.0)
                    else:
                        vmin = float(np.nanmin(finite))
                        vmax = float(np.nanmax(finite))
                        if vmax <= vmin:
                            vmax = vmin + 1e-6
                        clim = (vmin, vmax)

                    # salva para o 2D/compare usar igual
                    state["current_scalar_clim"] = clim
            else:
                mesh_main = mesh
                show_scalar = False

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
            # cache do dataset atual para inspeção/picking
            if actor_key == 'main_actor':
                state['main_actor_data'] = mesh_data
            # cache do dataset atual do ator (necessário para picking/inspector)
            try:
                state[f"{actor_key}_data"] = mesh_data
            except Exception:
                pass
            
            if is_bg: return actor

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
                plotter.add_scalar_bar(title=bar_title, mapper=main_actor.mapper, n_labels=5, fmt="%.1f", title_font_size=14, label_font_size=12)

        if state.get("bounds_actor") is None:
            state["bounds_actor"] = plotter.show_bounds(
                grid="back",
                location="outer",
                ticks="outside",
                color='gray',
                minor_ticks=True,
                n_xlabels=5,
                n_ylabels=5,
                n_zlabels=5,
                font_size=8,
                fmt="%.0f",
            )
        
        base_bounds = state.get("base_bounds")
        last_bb = state.get("last_base_bounds")
        last_z = state.get("last_bounds_z")

        need_update = False

        if base_bounds is not None:
            # compara base_bounds com o último aplicado (tolerância pequena)
            if last_bb is None:
                need_update = True
            else:
                try:
                    tol = 1e-6
                    for a, b in zip(base_bounds, last_bb):
                        if abs(float(a) - float(b)) > tol:
                            need_update = True
                            break
                except Exception:
                    need_update = True

        # se z_scale mudou, também precisa atualizar
        if last_z != z_scale:
            need_update = True

        if need_update and (state.get("bounds_actor") is not None) and base_bounds is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = base_bounds
            state["bounds_actor"].SetBounds(xmin, xmax, ymin, ymax, zmin * z_scale, zmax * z_scale)

            state["last_bounds_z"] = z_scale
            state["last_base_bounds"] = tuple(base_bounds)

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


    # ============================================================
    # Selection Mode (3D picking): None | "cell" | "column"
    # ============================================================
    state.setdefault("pick_mode", None)
    state.setdefault("_pick_observer_added", False)
    state.setdefault("_pick_highlight_cell", None)
    state.setdefault("_pick_highlight_column", None)

    def _clear_pick_highlights(render=True):
        for k in ("_pick_highlight_cell", "_pick_highlight_column"):
            a = state.get(k)
            if a is None:
                continue
            try:
                plotter.remove_actor(a)
            except Exception:
                try:
                    a.SetVisibility(False)
                except Exception:
                    pass
            state[k] = None
        if render:
            try:
                plotter.render()
            except Exception:
                pass

    state["clear_pick"] = _clear_pick_highlights

    def _ensure_pick_observer():
        if state.get("_pick_observer_added", False):
            return

        iren = getattr(plotter, "iren", None)
        if iren is None:
            iren = getattr(plotter, "interactor", None)
        if iren is None:
            try:
                iren = plotter.ren_win.GetInteractor()
            except Exception:
                iren = None
        if iren is None:
            return

        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.0005)

        def _do_pick(x, y):
            mode_pick = state.get("pick_mode", None)
            if mode_pick not in ("cell", "column"):
                return False

            main_actor = state.get("main_actor")
            mesh_data = state.get("main_actor_data")
            if main_actor is None or mesh_data is None or getattr(mesh_data, "n_cells", 0) == 0:
                return False

            try:
                picker.InitializePickList()
                picker.AddPickList(main_actor)
                picker.PickFromListOn()
            except Exception:
                pass

            ren = getattr(plotter, "renderer", None)
            if ren is None:
                try:
                    ren = plotter.renderers.active_renderer
                except Exception:
                    ren = None
            if ren is None:
                return False

            # Tenta coordenadas em diferentes convenções (top/bottom origin, HiDPI)
            cand = []
            try:
                xx = int(x)
                yy = int(y)
                cand.append((xx, yy))
            except Exception:
                cand.append((x, y))

            try:
                wsize = None
                try:
                    wsize = plotter.ren_win.GetSize()
                except Exception:
                    wsize = None
                if wsize and len(wsize) == 2 and int(wsize[1]) > 0:
                    H = int(wsize[1])
                    cand.append((cand[0][0], H - cand[0][1]))
            except Exception:
                pass

            picked = False
            cid = -1
            for xx, yy in cand:
                try:
                    ok = picker.Pick(float(xx), float(yy), 0.0, ren)
                except Exception:
                    ok = False
                if not ok:
                    continue
                try:
                    cid = int(picker.GetCellId())
                except Exception:
                    cid = -1
                if 0 <= cid < int(getattr(mesh_data, "n_cells", 0)):
                    picked = True
                    break

            # feedback opcional para UI (status bar)
            try:
                cb_status = state.get("status_callback", None)
                if callable(cb_status):
                    cb_status(
                        f"Pick: cell_id={cid}" if picked else "Pick: nenhuma célula (clique na malha)",
                        1500,
                    )
            except Exception:
                pass

            if not picked:
                return False

            def _safe_get(name):
                try:
                    arr = mesh_data.cell_data.get(name, None)
                    if arr is None:
                        return None
                    return arr[cid]
                except Exception:
                    return None

            i0 = _safe_get("i_index")
            j0 = _safe_get("j_index")
            k0 = _safe_get("k_index")

            props = {}
            try:
                for name in list(mesh_data.cell_data.keys()):
                    v = _safe_get(name)
                    if v is None:
                        continue
                    try:
                        if hasattr(v, "item"):
                            v = v.item()
                    except Exception:
                        pass
                    props[name] = v
            except Exception:
                pass

            info = {
                "mode": mode_pick,
                "cell_id": cid,
                "i": None if i0 is None else int(i0),
                "j": None if j0 is None else int(j0),
                "k": None if k0 is None else int(k0),
                "props": props,
            }

            # --- Resumo geométrico da célula selecionada (para o inspector) ---
            try:
                _sel_geom = mesh_data.extract_cells([cid])
                b = getattr(_sel_geom, "bounds", None)
                if b and len(b) == 6:
                    xmin, xmax, ymin, ymax, zmin, zmax = map(float, b)
                    geom = {
                        "length": float(xmax - xmin),
                        "width": float(ymax - ymin),
                        "height": float(zmax - zmin),
                        "center_x": float((xmin + xmax) * 0.5),
                        "center_y": float((ymin + ymax) * 0.5),
                        "center_z": float((zmin + zmax) * 0.5),
                        "bounds": (xmin, xmax, ymin, ymax, zmin, zmax),
                    }
                    # Volume (best-effort)
                    try:
                        _cs = _sel_geom.compute_cell_sizes(length=False, area=False, volume=True)
                        vv = _cs.cell_data.get("Volume", None)
                        if vv is not None and len(vv) > 0:
                            geom["volume"] = float(vv[0])
                    except Exception:
                        pass
                    info["geom"] = geom
            except Exception:
                pass

            # ------------------ HIGHLIGHT (sem flicker) ------------------
            def _add_mesh_no_render(*args, **kwargs):
                # evita 1 frame com scale=1 antes do SetScale()
                try:
                    kwargs.setdefault("render", False)
                    return plotter.add_mesh(*args, **kwargs)
                except TypeError:
                    # versões antigas podem não ter 'render'
                    kwargs.pop("render", None)
                    return plotter.add_mesh(*args, **kwargs)

            def _inherit_transform(actor):
                if actor is None:
                    return
                try:
                    actor.SetScale(*main_actor.GetScale())
                    actor.SetPosition(*main_actor.GetPosition())
                    actor.SetOrientation(*main_actor.GetOrientation())
                except Exception:
                    pass

            # desliga render enquanto remove/insere os highlights
            _disabled = False
            try:
                plotter.disable_render()
                _disabled = True
            except Exception:
                _disabled = False

            try:
                _clear_pick_highlights(render=False)

                col = None
                ids = None

                if mode_pick == "cell":
                    sel = mesh_data.extract_cells([cid])
                    state["_pick_highlight_cell"] = _add_mesh_no_render(
                        sel, style="wireframe", line_width=4, color="yellow", reset_camera=False
                    )
                    _inherit_transform(state.get("_pick_highlight_cell"))

                elif mode_pick == "column":
                    if i0 is None or j0 is None:
                        sel = mesh_data.extract_cells([cid])
                        state["_pick_highlight_cell"] = _add_mesh_no_render(
                            sel, style="wireframe", line_width=4, color="yellow", reset_camera=False
                        )
                        _inherit_transform(state.get("_pick_highlight_cell"))
                        info["mode"] = "cell"
                    else:
                        import numpy as _np
                        i0i, j0i = int(i0), int(j0)

                        ii = mesh_data.cell_data.get("i_index", None)
                        jj = mesh_data.cell_data.get("j_index", None)

                        if ii is None or jj is None:
                            sel = mesh_data.extract_cells([cid])
                            state["_pick_highlight_cell"] = _add_mesh_no_render(
                                sel, style="wireframe", line_width=4, color="yellow", reset_camera=False
                            )
                            _inherit_transform(state.get("_pick_highlight_cell"))
                            info["mode"] = "cell"
                        else:
                            ii = _np.asarray(ii).astype(int)
                            jj = _np.asarray(jj).astype(int)
                            mask = (ii == i0i) & (jj == j0i)
                            ids = _np.where(mask)[0]

                            if ids.size > 0:
                                col = mesh_data.extract_cells(ids)

                                state["_pick_highlight_column"] = _add_mesh_no_render(
                                    col, style="wireframe", line_width=3, color="cyan", reset_camera=False
                                )
                                _inherit_transform(state.get("_pick_highlight_column"))

                                sel = mesh_data.extract_cells([cid])
                                state["_pick_highlight_cell"] = _add_mesh_no_render(
                                    sel, style="wireframe", line_width=5, color="yellow", reset_camera=False
                                )
                                _inherit_transform(state.get("_pick_highlight_cell"))
                            else:
                                sel = mesh_data.extract_cells([cid])
                                state["_pick_highlight_cell"] = _add_mesh_no_render(
                                    sel, style="wireframe", line_width=4, color="yellow", reset_camera=False
                                )
                                _inherit_transform(state.get("_pick_highlight_cell"))
                                info["mode"] = "cell"

            finally:
                if _disabled:
                    try:
                        plotter.enable_render()
                    except Exception:
                        pass

            # ------------------ RESUMO + TABELA DA COLUNA ------------------
            if info.get("mode") == "column" and col is not None and ids is not None:
                import numpy as _np

                # facies counts
                try:
                    fac = col.cell_data.get("Facies", None)
                    if fac is not None:
                        fac = _np.asarray(fac).astype(int)
                        uniq, cnt = _np.unique(fac, return_counts=True)
                        info["column_facies_counts"] = {int(u): int(c) for u, c in zip(uniq, cnt)}
                except Exception:
                    pass

                # thickness sum + nome do campo thickness
                th_name = None
                try:
                    for nm in ("StratigraphicThickness", "cell_thickness", "Thickness", "thickness_local"):
                        if nm in col.cell_data:
                            th_name = nm
                            break
                except Exception:
                    th_name = None

                if th_name is not None:
                    try:
                        th = _np.asarray(col.cell_data[th_name], dtype=float)
                        info["column_thickness_name"] = th_name
                        info["column_thickness_sum"] = float(_np.nansum(th))
                    except Exception:
                        pass

                try:
                    # 1 linha por k, 1 coluna por propriedade
                    kcol = col.cell_data.get("k_index", None)
                    if kcol is not None:
                        kcol = _np.asarray(kcol).astype(int)

                    # ordem (k crescente)
                    try:
                        order = _np.argsort(kcol) if kcol is not None else _np.arange(int(getattr(col, "n_cells", 0)))
                    except Exception:
                        order = _np.arange(int(getattr(col, "n_cells", 0)))

                    # bounds topo/base
                    try:
                        bb = getattr(col, "bounds", None)
                        if bb and len(bb) == 6:
                            xmin, xmax, ymin, ymax, zmin, zmax = map(float, bb)
                            info["column_bounds"] = (xmin, xmax, ymin, ymax, zmin, zmax)
                            info["column_top_z"] = float(zmax)
                            info["column_base_z"] = float(zmin)
                    except Exception:
                        pass

                    # propriedades disponíveis
                    excluded = {
                        "vtkOriginalCellIds",
                        "vtkOriginalPointIds",
                        "vtkGhostType",
                        "i_index",
                        "j_index",
                    }

                    prop_names = []
                    try:
                        for nm in list(col.cell_data.keys()):
                            if nm in excluded:
                                continue
                            prop_names.append(nm)
                    except Exception:
                        prop_names = []

                    # ordem preferida
                    preferred = []
                    for nm in ("Facies", th_name, "StratigraphicThickness", "cell_thickness", "Thickness",
                            "Reservoir", "Clusters", "LargestCluster"):
                        if nm and (nm in prop_names) and (nm not in preferred):
                            preferred.append(nm)

                    ordered_cols = []
                    for nm in preferred:
                        if nm in prop_names and nm not in ordered_cols:
                            ordered_cols.append(nm)
                    for nm in prop_names:
                        if nm == "k_index":
                            continue
                        if nm in ordered_cols:
                            continue
                        ordered_cols.append(nm)

                    info["column_columns"] = ["k_index"] + ordered_cols

                    rows_dict = []
                    profile = []

                    for idx2 in list(order):
                        row = {}
                        try:
                            row_k = int(kcol[idx2]) if kcol is not None else int(idx2)
                        except Exception:
                            row_k = int(idx2)
                        row["k_index"] = row_k

                        for nm in ordered_cols:
                            try:
                                arr = col.cell_data.get(nm, None)
                                if arr is None:
                                    continue
                                v = arr[idx2]
                                if hasattr(v, "item"):
                                    v = v.item()
                                row[nm] = v
                            except Exception:
                                continue

                        # perfil compacto (k, facies, thickness)
                        f_ = row.get("Facies", None)
                        t_ = None
                        try:
                            if th_name is not None and th_name in row:
                                t_ = row.get(th_name)
                            elif "StratigraphicThickness" in row:
                                t_ = row.get("StratigraphicThickness")
                            elif "cell_thickness" in row:
                                t_ = row.get("cell_thickness")
                            elif "Thickness" in row:
                                t_ = row.get("Thickness")
                        except Exception:
                            t_ = None

                        profile.append((row_k, f_, t_))
                        rows_dict.append(row)

                    info["column_profile"] = profile
                    info["column_rows"] = rows_dict
                    info["column_ncells"] = int(len(rows_dict))
                except Exception:
                    pass

            cb = state.get("on_cell_picked", None)
            if callable(cb):
                try:
                    cb(info)
                except Exception:
                    pass

            try:
                plotter.render()
            except Exception:
                pass

            return True

        def _handler(*args):
            try:
                x, y = iren.GetEventPosition()
            except Exception:
                return
            _do_pick(x, y)

        # Permite que a UI (Qt) dispare picks de forma confiável (sem depender do VTK observer)
        state["_pick_perform"] = _do_pick


        try:
            if hasattr(iren, 'AddObserver'):
                iren.AddObserver('LeftButtonPressEvent', _handler)
            elif hasattr(iren, 'add_observer'):
                iren.add_observer('LeftButtonPressEvent', _handler)
            state['_pick_observer_added'] = True
            state['_vtk_cell_picker'] = picker
        except Exception:
            pass

    _ensure_pick_observer()


    plotter.enable_lightkit()
    plotter.add_axes()
    
    _refresh()
    plotter.reset_camera()
    
    state["refresh"] = _refresh
    return plotter, state