# analysis.py
import numpy as np
import pandas as pd
import os
from datetime import datetime
import pyvista as pv

from load_data import grid, facies, nx, ny, nz
from derived_fields import ensure_reservoir, ensure_clusters
from typing import Iterable
from scipy.ndimage import label as nd_label, generate_binary_structure

# =============================================================================
# HELPERS DE GEOMETRIA
# =============================================================================
def _get_cell_volumes(target_grid=None):
    """Retorna array de volumes das células (ABSOLUTO). Calcula se não existir."""
    g = target_grid if target_grid is not None else grid
    
    # 1) Tenta pegar cache
    for key in ("Volume", "Volume ", "volume", "Volume_"):
        if key in g.cell_data:
            return np.abs(g.cell_data[key]) # Garante positivo

    # 2) Calcula via PyVista
    tmp = g.compute_cell_sizes(length=False, area=False, volume=True)
    vol_arr = None
    for key in ("Volume", "Volume ", "volume", "Volume_"):
        if key in tmp.cell_data:
            vol_arr = tmp.cell_data[key]
            break

    if vol_arr is None:
        # Fallback de segurança
        return np.ones(g.n_cells) 

    # CORREÇÃO CRÍTICA: Volumes físicos devem ser positivos
    vol_arr = np.abs(vol_arr)
    
    g.cell_data["Volume"] = vol_arr
    return vol_arr

def _get_cell_z_coords(target_grid=None):
    g = target_grid if target_grid is not None else grid
    return g.cell_centers().points[:, 2]

def _get_cell_thickness(target_grid=None):
    """
    Retorna espessura por cÃ©lula priorizando o campo fornecido pelo grid
    (`StratigraphicThickness`). Se nÃ£o existir, calcula a partir dos vÃ©rtices
    e cacheia em `cell_thickness`.
    """
    g = target_grid if target_grid is not None else grid

    # PreferÃªncia: usar diretamente a propriedade vinda do grid
    for key in ("StratigraphicThickness", "stratigraphic_thickness"):
        if key in g.cell_data:
            arr = np.asarray(g.cell_data[key], dtype=float)
            if arr is not None and len(arr) == g.n_cells:
                return arr

    # Cache local
    for key in ("cell_thickness", "CellThickness", "thickness"):
        if key in g.cell_data:
            arr = g.cell_data[key]
            if arr is not None and len(arr) == g.n_cells:
                return arr

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

def _calc_stats_for_subset(subset_mask, volumes, z_coords):
    """Calcula estatísticas básicas (Cells, Vol, Thickness) para um subconjunto."""
    count = int(subset_mask.sum())
    if count == 0:
        return {"cells": 0, "volume": 0.0, "thickness_gross": 0.0}
    
    vol_total = float(volumes[subset_mask].sum())
    
    # Espessura bruta: Amplitude Z (Top - Base) onde a fácies ocorre
    z_vals = z_coords[subset_mask]
    thickness_gross = float(z_vals.max() - z_vals.min()) if count > 0 else 0.0
    
    return {
        "cells": count,
        "volume": vol_total,
        "thickness_gross": thickness_gross
    }

# =============================================================================
# ANÁLISE DE DISTRIBUIÇÃO
# =============================================================================

def facies_distribution_array(facies_array, target_grid=None):
    arr = np.asarray(facies_array).astype(int)
    total = arr.size
    if total == 0: return {}, 0

    volumes = _get_cell_volumes(target_grid)
    z_coords = _get_cell_z_coords(target_grid)
    
    # Segurança se tamanhos não baterem
    if volumes.size != total:
        volumes = np.ones(total) 
        z_coords = np.zeros(total)

    vals = np.unique(arr)
    stats = {}
    
    for fac in vals:
        mask = (arr == fac)
        s = _calc_stats_for_subset(mask, volumes, z_coords)
        s["fraction"] = s["cells"] / total
        stats[int(fac)] = s

    return stats, total

def reservoir_facies_distribution_array(facies_array, reservoir_facies, target_grid=None):
    arr = np.asarray(facies_array).astype(int)
    if isinstance(reservoir_facies, (int, np.integer)):
        fac_set = {int(reservoir_facies)}
    else:
        fac_set = {int(f) for f in reservoir_facies}

    mask_res = np.isin(arr, list(fac_set))
    res_total = int(mask_res.sum())
    
    if res_total == 0: return {}, 0

    volumes = _get_cell_volumes(target_grid)
    z_coords = _get_cell_z_coords(target_grid)
    
    if volumes.size != arr.size:
        volumes = np.ones(arr.size)
        z_coords = np.zeros(arr.size)

    arr_res = arr[mask_res]
    vol_res = volumes[mask_res]
    z_res = z_coords[mask_res]
    
    vals = np.unique(arr_res)
    stats = {}
    
    for fac in vals:
        mask_local = (arr_res == fac)
        count = int(mask_local.sum())
        vol = float(vol_res[mask_local].sum())
        
        zs = z_res[mask_local]
        thick = float(zs.max() - zs.min()) if count > 0 else 0.0
        
        stats[int(fac)] = {
            "cells": count,
            "fraction": count / res_total,
            "volume": vol,
            "thickness_gross": thick
        }

    return stats, res_total

# =============================================================================
# MÉTRICAS GLOBAIS
# =============================================================================

def compute_global_metrics(reservoir_facies):
    res_arr = ensure_reservoir(reservoir_facies)
    clusters_arr, _ = ensure_clusters(reservoir_facies)

    total_cells = facies.size
    # Soma de 0s e 1s funciona para contar células, mesmo sendo int
    res_cells = int(res_arr.sum())
    ntg = res_cells / total_cells if total_cells else 0.0

    volumes = _get_cell_volumes()
    
    # 1. Volume Total do Grid (Soma de tudo)
    grid_total_vol = float(volumes.sum())

    # 2. Volume do Reservatório (CORREÇÃO AQUI)
    # Convertemos res_arr para bool para funcionar como máscara de filtro
    mask_bool = res_arr.astype(bool)
    res_vol = float(volumes[mask_bool].sum())

    counts = np.bincount(clusters_arr)
    if counts.size > 0: counts[0] = 0

    n_clusters = (counts > 0).sum()
    largest_label = counts.argmax() if counts.size > 0 else 0
    largest_size = int(counts[largest_label]) if counts.size > 0 else 0
    connected_fraction = largest_size / res_cells if res_cells > 0 else 0.0

    return {
        "total_cells": total_cells,
        "res_cells": res_cells,
        "ntg": ntg,
        "n_clusters": n_clusters,
        "largest_label": largest_label,
        "largest_size": largest_size,
        "connected_fraction": connected_fraction,
        "grid_volume": grid_total_vol,   # Volume fixo do grid
        "reservoir_volume": res_vol,     # Volume dinâmico da seleção
    }

def compute_directional_percolation(reservoir_facies):
    clusters_arr, _ = ensure_clusters(reservoir_facies)
    clusters_xyz = clusters_arr.reshape((nx, ny, nz), order="F")

    # X
    left = clusters_xyz[0, :, :]
    right = clusters_xyz[-1, :, :]
    left_ids = set(np.unique(left)); left_ids.discard(0)
    right_ids = set(np.unique(right)); right_ids.discard(0)
    x_common = left_ids.intersection(right_ids)

    # Y
    front = clusters_xyz[:, 0, :]
    back = clusters_xyz[:, -1, :]
    f_ids = set(np.unique(front)); f_ids.discard(0)
    b_ids = set(np.unique(back)); b_ids.discard(0)
    y_common = f_ids.intersection(b_ids)

    # Z
    top = clusters_xyz[:, :, 0]
    bottom = clusters_xyz[:, :, -1]
    t_ids = set(np.unique(top)); t_ids.discard(0)
    bo_ids = set(np.unique(bottom)); bo_ids.discard(0)
    z_common = t_ids.intersection(bo_ids)

    return {
        "x_perc": bool(x_common), "x_clusters": x_common,
        "y_perc": bool(y_common), "y_clusters": y_common,
        "z_perc": bool(z_common), "z_clusters": z_common,
    }

def compute_global_metrics_for_array(facies_array, reservoir_facies, target_grid=None):
    """
    Calcula métricas globais para um array de fácies específico.
    target_grid: Opcional. Usado para calcular volumes corretos do modelo.
    """
    from load_data import nx, ny, nz
    
    arr = np.asarray(facies_array).astype(int)
    total_cells = arr.size

    if isinstance(reservoir_facies, (int, np.integer)):
        fac_set = {int(reservoir_facies)}
    else:
        fac_set = {int(f) for f in reservoir_facies}

    mask = np.isin(arr, list(fac_set))
    res_cells = int(mask.sum())
    ntg = res_cells / total_cells if total_cells else 0.0
    
    # --- CÁLCULO DE VOLUMES ---
    volumes = _get_cell_volumes(target_grid)
    
    # Se os tamanhos baterem, calcula. Se não, zera (segurança).
    if volumes.size == total_cells:
        grid_volume = float(volumes.sum())
        reservoir_volume = float(volumes[mask].sum())
    else:
        grid_volume = 0.0
        reservoir_volume = 0.0

    if res_cells == 0:
        return ({
            "total_cells": int(total_cells), "res_cells": 0, "ntg": 0.0,
            "n_clusters": 0, "largest_label": 0, "largest_size": 0, "connected_fraction": 0.0,
            "grid_volume": grid_volume, "reservoir_volume": 0.0
        }, {
            "x_perc": False, "x_clusters": set(), "y_perc": False, "y_clusters": set(), "z_perc": False, "z_clusters": set(),
        })

    # Análise de Clusters
    try:
        res_xyz = mask.reshape((nx, ny, nz), order="F")
        structure = generate_binary_structure(3, 1)
        labeled, _ = nd_label(res_xyz, structure=structure)
        clusters_1d = labeled.reshape(-1, order="F")

        counts = np.bincount(clusters_1d)
        if counts.size > 0: counts[0] = 0
        n_clusters = int((counts > 0).sum())
        largest_label = int(counts.argmax()) if counts.size > 0 else 0
        largest_size = int(counts[largest_label]) if counts.size > 0 else 0
        connected_fraction = largest_size / res_cells if res_cells > 0 else 0.0

        # Percolação
        clusters_xyz = labeled
        left_ids = set(np.unique(clusters_xyz[0,:,:])); left_ids.discard(0)
        right_ids = set(np.unique(clusters_xyz[-1,:,:])); right_ids.discard(0)
        x_common = left_ids.intersection(right_ids)

        f_ids = set(np.unique(clusters_xyz[:,0,:])); f_ids.discard(0)
        b_ids = set(np.unique(clusters_xyz[:,-1,:])); b_ids.discard(0)
        y_common = f_ids.intersection(b_ids)

        t_ids = set(np.unique(clusters_xyz[:,:,0])); t_ids.discard(0)
        bo_ids = set(np.unique(clusters_xyz[:,:,-1])); bo_ids.discard(0)
        z_common = t_ids.intersection(bo_ids)
        
        perc = {
            "x_perc": bool(x_common), "x_clusters": x_common,
            "y_perc": bool(y_common), "y_clusters": y_common,
            "z_perc": bool(z_common), "z_clusters": z_common,
        }
    except Exception:
        n_clusters = 0; largest_label=0; largest_size=0; connected_fraction=0.0
        perc = {"x_perc": False, "x_clusters": set(), "y_perc": False, "y_clusters": set(), "z_perc": False, "z_clusters": set()}

    metrics = {
        "total_cells": int(total_cells), "res_cells": res_cells, "ntg": float(ntg),
        "n_clusters": n_clusters, "largest_label": largest_label,
        "largest_size": largest_size, "connected_fraction": float(connected_fraction),
        "grid_volume": grid_volume, "reservoir_volume": reservoir_volume
    }
    
    return metrics, perc

def compute_facies_metrics():
    total_cells = facies.size
    unique_facies = np.unique(facies)
    volumes = _get_cell_volumes()
    z_centers = _get_cell_z_coords()
    results = []

    for fac in unique_facies:
        fac_mask = (facies == fac)
        fac_cells = int(fac_mask.sum())
        fac_fraction = fac_cells / total_cells if total_cells else 0.0
        fac_volume = float(volumes[fac_mask].sum()) if fac_cells > 0 else 0.0

        from scipy.ndimage import label, generate_binary_structure
        is_fac = (facies == fac).astype(np.uint8)
        arr_zyx = is_fac.reshape((nx, ny, nz), order="F").transpose(2, 1, 0)
        structure = generate_binary_structure(3, 1)
        labeled_zyx, n_clusters = label(arr_zyx, structure=structure)
        labeled_xyz = labeled_zyx.transpose(2, 1, 0)
        fac_clusters = labeled_xyz.reshape(-1, order="F").astype(np.int32)

        counts = np.bincount(fac_clusters)
        if counts.size > 0:
            counts[0] = 0
            largest_label = counts.argmax()
            largest_size = int(counts[largest_label]) if largest_label > 0 else 0
        else:
            largest_label = 0; largest_size = 0

        connected_fraction = (largest_size / fac_cells) if fac_cells > 0 else 0.0

        if largest_label > 0:
            largest_mask = (fac_clusters == largest_label)
            largest_volume = float(volumes[largest_mask].sum())
            z_vals = z_centers[largest_mask]
            thickness = float(z_vals.max() - z_vals.min()) if z_vals.size > 0 else 0.0
        else:
            largest_volume = 0.0; thickness = 0.0

        left = labeled_xyz[0, :, :]; right = labeled_xyz[-1, :, :]
        x_p = bool(set(np.unique(left)).intersection(set(np.unique(right))) - {0})
        front = labeled_xyz[:, 0, :]; back = labeled_xyz[:, -1, :]
        y_p = bool(set(np.unique(front)).intersection(set(np.unique(back))) - {0})
        top = labeled_xyz[:, :, 0]; bottom = labeled_xyz[:, :, -1]
        z_p = bool(set(np.unique(top)).intersection(set(np.unique(bottom))) - {0})

        fac_result = {
            "facies": int(fac), "cells": fac_cells, "fraction": fac_fraction,
            "n_clusters": int(n_clusters), "largest_label": int(largest_label),
            "largest_size": int(largest_size), "connected_fraction": float(connected_fraction),
            "volume_total": fac_volume, "volume_largest_cluster": largest_volume,
            "thickness_largest_cluster": thickness,
            "perc": {"x_perc": x_p, "y_perc": y_p, "z_perc": z_p},
        }
        results.append(fac_result)
    return results

def export_facies_metrics_to_excel(output_path=None):
    data = compute_facies_metrics()
    df = pd.DataFrame(data)
    perc_df = df["perc"].apply(pd.Series)
    perc_df = perc_df.rename(columns={"x_perc": "Perc_X", "y_perc": "Perc_Y", "z_perc": "Perc_Z"})
    df = pd.concat([df.drop(columns=["perc"]), perc_df[["Perc_X", "Perc_Y", "Perc_Z"]]], axis=1)

    if output_path is None:
        output_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "facies_metrics.xlsx")

    try:
        df.to_excel(output_path, index=False, sheet_name="Facies Metrics")
        print(f"\n✅ Métricas por fácies exportadas para: {output_path}")
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_path = output_path.replace(".xlsx", f"_{ts}.xlsx")
        df.to_excel(alt_path, index=False, sheet_name="Facies Metrics")
        print(f"\nO arquivo original estava bloqueado. Salvei como: {alt_path}")

def make_thickness_2d_from_grid(array_name_3d="thickness_local", array_name_2d="thickness_2d"):
    if array_name_3d not in grid.cell_data:
        raise ValueError(f"Array {array_name_3d} não existe no grid 3D.")
    arr3d = grid.cell_data[array_name_3d].reshape((nx, ny, nz), order="F")
    thickness_2d = np.full((nx, ny), np.nan, dtype=float)
    for ix in range(nx):
        for iy in range(ny):
            col_vals = arr3d[ix, iy, :]
            col_vals = col_vals[col_vals > 0]
            if col_vals.size > 0: thickness_2d[ix, iy] = col_vals.max()
    x_min, x_max, y_min, y_max, z_min, z_max = grid.bounds
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    xs, ys = np.meshgrid(xs, ys, indexing="ij")
    zs = np.full_like(xs, z_max)
    surf = pv.StructuredGrid(xs, ys, zs)
    surf.cell_data[array_name_2d] = thickness_2d[:nx-1, :ny-1].ravel(order="F")
    return surf

def _vertical_metric_base_names(include_filtered=True):
    names = [
        "Ttot_reservoir",
        "Tenv_reservoir",
        "NTG_col_reservoir",
        "NTG_env_reservoir",
        "n_packages_reservoir",
        "Tpack_max_reservoir",
        "Tpack_mean_reservoir",
        "Cdom_reservoir",
        "ICV_reservoir",
        "ICV_env_reservoir",
        "ICV_col_reservoir",
        "Qv_reservoir",
        "Qv_abs_reservoir",
        "Tgap_sum_reservoir",
        "Tgap_max_reservoir",
        "Gap_env_reservoir",
        "Nswitch_reservoir",
        "Dswitch_col_reservoir",
        "Dswitch_env_reservoir",
        "Wswitch_reservoir",
        "WswitchN_col_reservoir",
        "Npersist_reservoir",
        "Rpersist_reservoir",
        "Wpersist_reservoir",
        "WpersistN_reservoir",
    ]
    if include_filtered:
        names.extend([f"{name}_filt" for name in names])
    return names


VERTICAL_NORMALIZED_BASE_NAMES = (
    "NTG_col_reservoir",
    "NTG_env_reservoir",
    "Cdom_reservoir",
    "ICV_reservoir",
    "ICV_env_reservoir",
    "ICV_col_reservoir",
    "Qv_reservoir",
    "Qv_abs_reservoir",
    "Gap_env_reservoir",
    "Rpersist_reservoir",
)


def is_vertical_metric_normalized_name(name, prefix="vert_"):
    s = str(name or "")
    if not s.startswith(prefix):
        return False
    core = s[len(prefix):]
    return any(core == base or core == f"{base}_filt" for base in VERTICAL_NORMALIZED_BASE_NAMES)


VERTICAL_PRESET_DEFINITIONS = [
    ("Espessura", "Ttot_reservoir", "Espessura total reservatório (m)"),
    ("Espessura no envelope", "Tenv_reservoir", "Espessura no envelope (m)"),
    ("Proporção de fácies (coluna)", "NTG_col_reservoir", "Proporção de fácies (coluna)"),
    ("Proporção de fácies (envelope)", "NTG_env_reservoir", "Proporção de fácies (envelope)"),
    ("Maior pacote", "Tpack_max_reservoir", "Maior pacote (m)"),
    ("Espessura média dos pacotes", "Tpack_mean_reservoir", "Espessura média dos pacotes (m)"),
    ("Nº pacotes", "n_packages_reservoir", "Nº pacotes"),
    ("Dominância do maior pacote", "Cdom_reservoir", "Dominância do maior pacote"),
    ("ICV", "ICV_reservoir", "ICV"),
    ("ICV envelope", "ICV_env_reservoir", "ICV envelope"),
    ("ICV coluna", "ICV_col_reservoir", "ICV coluna"),
    ("Qv", "Qv_reservoir", "Qv"),
    ("Qv absoluto", "Qv_abs_reservoir", "Qv absoluto"),
    ("Soma dos gaps", "Tgap_sum_reservoir", "Soma dos gaps (m)"),
    ("Maior gap", "Tgap_max_reservoir", "Maior gap (m)"),
    ("Fração de gaps no envelope", "Gap_env_reservoir", "Fração de gaps no envelope"),
    ("Nº de trocas", "Nswitch_reservoir", "Nº de trocas"),
    ("Densidade de trocas (coluna)", "Dswitch_col_reservoir", "Densidade de trocas (coluna)"),
    ("Densidade de trocas (envelope)", "Dswitch_env_reservoir", "Densidade de trocas (envelope)"),
    ("Troca ponderada", "Wswitch_reservoir", "Troca ponderada"),
    ("Troca ponderada norm. (coluna)", "WswitchN_col_reservoir", "Troca ponderada norm. (coluna)"),
    ("Permanência bruta", "Npersist_reservoir", "Permanência bruta"),
    ("Permanência relativa", "Rpersist_reservoir", "Permanência relativa"),
    ("Permanência ponderada", "Wpersist_reservoir", "Permanência ponderada"),
    ("Permanência ponderada norm.", "WpersistN_reservoir", "Permanência ponderada norm."),
]


def get_vertical_metric_presets(prefix="vert_", include_filtered=True):
    presets = {}
    for label, base, title in VERTICAL_PRESET_DEFINITIONS:
        presets[label] = (f"{prefix}{base}", title)
        if include_filtered:
            presets[f"{label} (filtrado)"] = (f"{prefix}{base}_filt", f"{title} (filtrado)")
    return presets


def _zero_vertical_metrics_dict():
    return {name: 0.0 for name in _vertical_metric_base_names(include_filtered=False)}


def _clip01(value):
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except Exception:
        return 0.0


def _build_binary_runs(binary_flags, thickness_values):
    flags = np.asarray(binary_flags, dtype=np.int8).ravel()
    th = np.asarray(thickness_values, dtype=float).ravel()
    n = min(flags.size, th.size)
    if n == 0:
        return []

    runs = []
    cur_flag = int(flags[0])
    cur_t = float(th[0])
    cur_n = 1

    for fl, tt in zip(flags[1:n], th[1:n]):
        fl = int(fl)
        tt = float(tt)
        if fl == cur_flag:
            cur_t += tt
            cur_n += 1
        else:
            runs.append((cur_flag, cur_t, cur_n))
            cur_flag = fl
            cur_t = tt
            cur_n = 1
    runs.append((cur_flag, cur_t, cur_n))
    return runs


def _merge_adjacent_same_binary_runs(runs):
    if not runs:
        return []
    out = [tuple(runs[0])]
    for flag, thick, count in runs[1:]:
        pf, pt, pc = out[-1]
        if int(flag) == int(pf):
            out[-1] = (pf, float(pt) + float(thick), int(pc) + int(count))
        else:
            out.append((int(flag), float(thick), int(count)))
    return out


def _merge_thin_binary_runs(runs, t_min):
    if not runs:
        return []
    try:
        t_min = float(t_min)
    except Exception:
        t_min = 0.0
    if t_min <= 0.0 or len(runs) <= 1:
        return _merge_adjacent_same_binary_runs(runs)

    runs = [(int(f), float(t), int(c)) for (f, t, c) in runs]
    runs = _merge_adjacent_same_binary_runs(runs)

    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, (flag, thick, count) in enumerate(runs):
            if float(thick) >= t_min:
                continue

            if i == 0:
                nf, nt, nc = runs[1]
                runs[1] = (nf, nt + thick, nc + count)
                runs.pop(0)
            elif i == len(runs) - 1:
                pf, pt, pc = runs[-2]
                runs[-2] = (pf, pt + thick, pc + count)
                runs.pop(-1)
            else:
                lf, lt, lc = runs[i - 1]
                rf, rt, rc = runs[i + 1]
                if float(lt) >= float(rt):
                    runs[i - 1] = (lf, lt + thick, lc + count)
                    runs.pop(i)
                else:
                    runs[i + 1] = (rf, rt + thick, rc + count)
                    runs.pop(i)

            runs = _merge_adjacent_same_binary_runs(runs)
            changed = True
            break

    return _merge_adjacent_same_binary_runs(runs)


def _expand_binary_runs_to_cells(runs):
    flags = []
    th = []
    for flag, thick, count in runs:
        count = max(int(count), 1)
        per_cell = float(thick) / float(count)
        flags.extend([int(flag)] * count)
        th.extend([per_cell] * count)
    return np.asarray(flags, dtype=np.int8), np.asarray(th, dtype=float)


def _compute_vertical_metrics_from_binary_cells(binary_flags, thickness_values):
    out = _zero_vertical_metrics_dict()

    flags = np.asarray(binary_flags, dtype=np.int8).ravel()
    th = np.asarray(thickness_values, dtype=float).ravel()
    valid = np.isfinite(th) & (th > 0.0)
    flags = flags[valid]
    th = th[valid]

    if flags.size == 0 or th.size == 0:
        return out

    T_col = float(np.sum(th))
    if T_col <= 0.0:
        return out

    target_mask = flags == 1
    if not np.any(target_mask):
        return out

    idx = np.where(target_mask)[0]
    start_idx = int(idx[0])
    end_idx = int(idx[-1])

    T_tot = float(np.sum(th[target_mask]))
    T_env = float(np.sum(th[start_idx:end_idx + 1])) if end_idx >= start_idx else 0.0
    NTG_col = T_tot / T_col if T_col > 0.0 else 0.0
    NTG_env = T_tot / T_env if T_env > 0.0 else 0.0

    runs = _build_binary_runs(flags, th)
    target_runs = [(f, t, c) for (f, t, c) in runs if int(f) == 1]
    n_packages = len(target_runs)
    pack_thicknesses = [float(t) for (_f, t, _c) in target_runs]
    T_pack_max = max(pack_thicknesses) if pack_thicknesses else 0.0
    T_pack_mean = (T_tot / float(n_packages)) if n_packages > 0 else 0.0
    Cdom = (T_pack_max / T_tot) if T_tot > 0.0 else 0.0
    ICV_env = (T_pack_max / T_env) if T_env > 0.0 else 0.0
    ICV_col = (T_pack_max / T_col) if T_col > 0.0 else 0.0
    Qv = NTG_col * ICV_env
    Qv_abs = ICV_env * ICV_col

    target_run_ids = [i for i, (f, _t, _c) in enumerate(runs) if int(f) == 1]
    gaps = []
    if len(target_run_ids) >= 2:
        for run_id in range(target_run_ids[0] + 1, target_run_ids[-1]):
            f_gap, t_gap, _c_gap = runs[run_id]
            if int(f_gap) == 0:
                gaps.append(float(t_gap))
    Tgap_sum = float(sum(gaps)) if gaps else 0.0
    Tgap_max = max(gaps) if gaps else 0.0
    Gap_env = (Tgap_sum / T_env) if T_env > 0.0 else 0.0

    adj_same_target = (flags[1:] == 1) & (flags[:-1] == 1)
    adj_switch = flags[1:] != flags[:-1]

    Nswitch = float(np.sum(adj_switch))
    Dswitch_col = (Nswitch / T_col) if T_col > 0.0 else 0.0
    Dswitch_env = (Nswitch / T_env) if T_env > 0.0 else 0.0
    Wswitch = float(np.sum(0.5 * (th[1:] + th[:-1])[adj_switch])) if np.any(adj_switch) else 0.0
    WswitchN_col = (Wswitch / T_col) if T_col > 0.0 else 0.0

    Npersist = float(np.sum(adj_same_target))
    n_target_cells = int(np.sum(target_mask))
    Rpersist = (Npersist / max(n_target_cells - 1, 1)) if n_target_cells > 0 else 0.0
    Wpersist = float(np.sum(0.5 * (th[1:] + th[:-1])[adj_same_target])) if np.any(adj_same_target) else 0.0
    WpersistN = (Wpersist / T_tot) if T_tot > 0.0 else 0.0

    out.update({
        "Ttot_reservoir": T_tot,
        "Tenv_reservoir": T_env,
        "NTG_col_reservoir": _clip01(NTG_col),
        "NTG_env_reservoir": _clip01(NTG_env),
        "n_packages_reservoir": float(n_packages),
        "Tpack_max_reservoir": T_pack_max,
        "Tpack_mean_reservoir": T_pack_mean,
        "Cdom_reservoir": _clip01(Cdom),
        "ICV_reservoir": _clip01(ICV_env),
        "ICV_env_reservoir": _clip01(ICV_env),
        "ICV_col_reservoir": _clip01(ICV_col),
        "Qv_reservoir": _clip01(Qv),
        "Qv_abs_reservoir": _clip01(Qv_abs),
        "Tgap_sum_reservoir": Tgap_sum,
        "Tgap_max_reservoir": Tgap_max,
        "Gap_env_reservoir": _clip01(Gap_env),
        "Nswitch_reservoir": Nswitch,
        "Dswitch_col_reservoir": Dswitch_col,
        "Dswitch_env_reservoir": Dswitch_env,
        "Wswitch_reservoir": Wswitch,
        "WswitchN_col_reservoir": WswitchN_col,
        "Npersist_reservoir": Npersist,
        "Rpersist_reservoir": _clip01(Rpersist),
        "Wpersist_reservoir": Wpersist,
        "WpersistN_reservoir": WpersistN,
    })
    return out


def compute_vertical_metrics_for_grid(
    target_grid,
    facies_array,
    reservoir_set,
    prefix="vert_",
    thin_lamination_threshold=0.30,
    include_filtered=True,
):
    if target_grid is None or facies_array is None:
        return []
    if target_grid.n_cells != nx * ny * nz:
        return []

    metric_names = _vertical_metric_base_names(include_filtered=include_filtered)
    data_map = {f"{prefix}{name}": np.zeros((nx, ny, nz), dtype=float) for name in metric_names}

    th = _get_cell_thickness(target_grid)
    if th is None:
        return []

    try:
        fac_3d = np.asarray(facies_array).reshape((nx, ny, nz), order="F")
        th_3d = np.asarray(th, dtype=float).reshape((nx, ny, nz), order="F")
    except Exception:
        return []

    res_list = sorted({int(f) for f in (reservoir_set or [])})

    for ix in range(nx):
        for iy in range(ny):
            col_fac = fac_3d[ix, iy, :]
            col_th = th_3d[ix, iy, :]
            valid_th = np.isfinite(col_th) & (col_th > 0.0)
            if not np.any(valid_th):
                continue

            mask = np.isin(col_fac, res_list) & valid_th if res_list else np.zeros_like(valid_th, dtype=bool)
            if not np.any(mask):
                continue

            raw_metrics = _compute_vertical_metrics_from_binary_cells(mask.astype(np.int8), col_th)
            for base_name, value in raw_metrics.items():
                data_map[f"{prefix}{base_name}"][ix, iy, mask] = value

            if include_filtered:
                raw_runs = _build_binary_runs(mask[valid_th].astype(np.int8), col_th[valid_th])
                filt_runs = _merge_thin_binary_runs(raw_runs, thin_lamination_threshold)
                filt_flags, filt_th = _expand_binary_runs_to_cells(filt_runs)
                filt_metrics = _compute_vertical_metrics_from_binary_cells(filt_flags, filt_th)
                for base_name, value in filt_metrics.items():
                    data_map[f"{prefix}{base_name}_filt"][ix, iy, mask] = value

    for name, arr_3d in data_map.items():
        target_grid.cell_data[name] = arr_3d.reshape(-1, order="F")

    return list(data_map.keys())


def add_vertical_facies_metrics(facies_selected, prefix="vert_", thin_lamination_threshold=0.30, include_filtered=True):
    return compute_vertical_metrics_for_grid(
        target_grid=grid,
        facies_array=facies,
        reservoir_set=facies_selected,
        prefix=prefix,
        thin_lamination_threshold=thin_lamination_threshold,
        include_filtered=include_filtered,
    )

# =============================================================================
# ANÁLISE DE POÇOS
# =============================================================================

def sample_grid_along_well(grid, well_df, scalar_name="Facies"):
    """
    Extrai valores do grid ao longo da trajetória do poço.
    grid: objeto PyVista (unstructured ou structured)
    well_df: DataFrame com colunas X, Y, Z
    """
    points = well_df[["X", "Y", "Z"]].values
    
    # PyVista tem uma função rápida para isso: sample_over_line ou probe
    # Mas como temos pontos discretos arbitrários, usamos probe filter
    import pyvista as pv
    
    # Cria um PolyData apenas com os pontos do poço
    well_poly = pv.PolyData(points)
    
    # O filtro 'probe' projeta os dados do grid nestes pontos
    sampled = well_poly.sample(grid, tolerance=1.0) # tolerance depende da escala
    
    # Retorna os valores amostrados
    return sampled.point_data[scalar_name]

def sample_well_from_grid(well_df, grid_source, scalar_name="Facies"):
    """
    Recebe o DataFrame do poço (com X, Y, Z) e o Grid PyVista.
    Retorna array de fácies SIMULADAS para cada ponto do poço.
    """
    if well_df is None or well_df.empty:
        return None
        
    points = well_df[["X", "Y", "Z"]].values
    
    # Cria nuvem de pontos temporária
    import pyvista as pv
    cloud = pv.PolyData(points)
    
    # Sample/Probe: Projeta os valores do grid nestes pontos
    # tolerance: margem de erro caso o ponto caia exatamente na borda
    sampled = cloud.sample(grid_source, tolerance=1.0)
    
    # Retorna o array amostrado. Se cair fora do grid, vem 0 ou NaN.
    return sampled.point_data.get(scalar_name, np.zeros(len(points)))

def calculate_well_accuracy(real_arr, sim_arr):
    """
    Calcula % de acerto ponto a ponto.
    Ignora NaNs (onde não tem log real).
    """
    # Garante arrays numpy
    r = np.asarray(real_arr)
    s = np.asarray(sim_arr)
    
    # Máscara válida (onde real não é NaN e simulado > 0)
    valid = (~np.isnan(r)) & (r != -999.25)
    
    if not np.any(valid):
        return 0.0, 0
        
    r_valid = r[valid]
    s_valid = s[valid]
    
    matches = (r_valid == s_valid).sum()
    total = len(r_valid)
    
    accuracy = matches / total if total > 0 else 0.0
    return accuracy, total

def resample_to_normalized_depth(depth_arr, facies_arr, n_samples=200):
    """
    Normaliza um perfil de poço para um domínio 0.0 (topo) a 1.0 (base)
    e reamostra as fácies usando vizinho mais próximo.
    """
    # Validações básicas
    if len(depth_arr) < 2 or len(facies_arr) == 0:
        return np.zeros(n_samples, dtype=int)
    
    # Garante arrays numpy e remove NaNs
    facies_arr = np.asarray(facies_arr)
    depth_arr = np.asarray(depth_arr)
    valid = ~np.isnan(facies_arr)
    
    d = depth_arr[valid]
    f = facies_arr[valid]
    
    if len(d) == 0: return np.zeros(n_samples, dtype=int)

    # Cria eixo normalizado (0 a 1) baseado na profundidade original
    d_min, d_max = d.min(), d.max()
    if d_max == d_min: return np.full(n_samples, int(f[0]), dtype=int)
    
    d_norm = (d - d_min) / (d_max - d_min) # Vetor de 0 a 1 original
    
    # Eixo alvo uniforme (0 a 1)
    target_axis = np.linspace(0, 1, n_samples)
    
    # Interpolação Nearest Neighbor (busca o índice mais próximo)
    idx = np.searchsorted(d_norm, target_axis, side="left")
    idx = np.clip(idx, 0, len(f) - 1)
    
    return f[idx].astype(int)

def calculate_stratigraphic_accuracy(real_depth, real_facies, sim_depth, sim_facies):
    """
    Calcula acurácia comparando vetores normalizados (ignora espessura absoluta).
    Retorna: acurácia (float), vetor_real_norm, vetor_sim_norm
    """
    # Se não houver dados simulados (poço fora do grid), acurácia é 0
    if len(sim_facies) == 0: return 0.0, [], []

    # 1. Reamostra ambos para 200 'pixels' estratigráficos
    # Usamos 200 bins para ter resolução suficiente
    r_norm = resample_to_normalized_depth(real_depth, real_facies, n_samples=200)
    s_norm = resample_to_normalized_depth(sim_depth, sim_facies, n_samples=200)
    
    # 2. Compara ponto a ponto no domínio normalizado
    matches = (r_norm == s_norm).sum()
    accuracy = matches / len(r_norm)
    
    return accuracy, r_norm, s_norm

def sample_well_from_grid_resampled(
    well_df,
    grid_source,
    step=0.1,
    scalar_name="Facies",
    tolerance=5.0,
    return_cell_ids=False,
):
    import numpy as np
    import pyvista as pv

    if well_df is None or well_df.empty:
        return None, None

    if not all(c in well_df.columns for c in ("DEPT", "X", "Y", "Z")):
        return None, None

    df = well_df.sort_values("DEPT")
    dept = df["DEPT"].to_numpy(dtype=float)
    x = df["X"].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)
    z = df["Z"].to_numpy(dtype=float)

    if len(dept) < 2:
        return None, None

    dmin, dmax = float(dept.min()), float(dept.max())
    step = 0.1 if step <= 0 else float(step)
    depth_res = np.arange(dmin, dmax + 0.5 * step, step, dtype=float)

    x_res = np.interp(depth_res, dept, x)
    y_res = np.interp(depth_res, dept, y)
    z_res = np.interp(depth_res, dept, z)

    cloud = pv.PolyData(np.column_stack([x_res, y_res, z_res]))
    sampled = cloud.sample(grid_source, tolerance=float(tolerance))
    fac = sampled.point_data.get(scalar_name, np.zeros(len(depth_res)))

    cell_ids = None
    if return_cell_ids:
        if "vtkOriginalCellIds" in sampled.point_data:
            cell_ids = np.asarray(sampled.point_data["vtkOriginalCellIds"]).astype(int)
        elif "vtkOriginalCellIds" in sampled.cell_data:
            cell_ids = np.asarray(sampled.cell_data["vtkOriginalCellIds"]).astype(int)
        else:
            cell_ids = np.full(len(depth_res), -1, dtype=int)

    if return_cell_ids:
        return depth_res, fac, cell_ids
    return depth_res, fac



def extract_facies_layers(depth, facies):
    """
    Retorna lista de camadas: [{"facies": int, "top": float, "base": float, "thick": float}, ...]
    Assume depth crescente.
    """
    import numpy as np

    if depth is None or facies is None:
        return []

    depth = np.asarray(depth, dtype=float)
    facies = np.asarray(facies).astype(int)

    if len(depth) < 2 or len(depth) != len(facies):
        return []

    layers = []
    cur = int(facies[0])
    top = float(depth[0])

    for i in range(1, len(facies)):
        if int(facies[i]) != cur:
            base = float(depth[i])
            thick = base - top
            layers.append({"facies": cur, "top": top, "base": base, "thick": thick})
            cur = int(facies[i])
            top = float(depth[i])

    base = float(depth[-1])
    thick = base - top
    layers.append({"facies": cur, "top": top, "base": base, "thick": thick})

    return layers


def print_layers(label, depth, facies, min_thick=0.05):
    from analysis import extract_facies_layers
    layers = extract_facies_layers(depth, facies)
    print(f"\n--- {label} ---")
    tot = 0.0
    for L in layers:
        if L["thick"] >= min_thick:
            print(f"fac {L['facies']:>4} | {L['top']:.2f} -> {L['base']:.2f} | {L['thick']:.2f} m")
            tot += L["thick"]
    print(f"TOTAL (filtrado >= {min_thick} m): {tot:.2f} m")

def estimate_probe_tolerance_from_grid(grid, factor=0.9):
    """
    Estima tolerance baseado no tamanho típico da célula do grid.
    Não depende de nx/ny/nz no state: tenta extrair do próprio grid.
    """
    import numpy as np

    # tenta pegar dimensões estruturadas
    nx = ny = nz = None

    # StructuredGrid / ImageData tem dimensions
    if hasattr(grid, "dimensions") and grid.dimensions is not None:
        dims = tuple(int(v) for v in grid.dimensions)
        # em PyVista, StructuredGrid: (nx, ny, nz) em pontos.
        # para células, seria -1, mas aqui só precisamos escala.
        if len(dims) == 3:
            nx, ny, nz = dims[0], dims[1], dims[2]

    # fallback: tentar inferir a partir do array k_index (se existir)
    if (nx is None or ny is None or nz is None):
        if hasattr(grid, "cell_data") and "k_index" in grid.cell_data and "j_index" in grid.cell_data and "i_index" in grid.cell_data:
            i = grid.cell_data["i_index"]
            j = grid.cell_data["j_index"]
            k = grid.cell_data["k_index"]
            nx = int(np.max(i)) + 1
            ny = int(np.max(j)) + 1
            nz = int(np.max(k)) + 1

    # última tentativa: usa bounds sem dividir (tolerance grande porém seguro)
    b = grid.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    dx = (b[1] - b[0]) / max(nx or 1, 1)
    dy = (b[3] - b[2]) / max(ny or 1, 1)
    dz = (b[5] - b[4]) / max(nz or 1, 1)

    diag = float(np.sqrt(dx*dx + dy*dy + dz*dz))
    # se nx/ny/nz não existiam, diag vira diagonal do bounds inteiro -> tolerance gigante.
    # limita pra não explodir:
    diag = min(diag, max(b[1]-b[0], b[3]-b[2], b[5]-b[4]) * 0.1)

    return max(1e-6, factor * diag)



def _profile_to_runs(depth, facies_values):
    """Converte um perfil (depth, facies) em uma lista de runs (facies, thickness).

    Convenção:
      - Usa diferenças consecutivas de depth como incrementos de espessura.
      - A espessura dz entre (i -> i+1) é atribuída à fácies do ponto i.
      - Ordena por depth crescente e ignora valores não finitos.

    Retorna:
      runs: list[tuple[int, float]]
      t_total: float (soma das espessuras nos runs)
    """
    d = np.asarray(depth, dtype=float).ravel()
    f = np.asarray(facies_values, dtype=float).ravel()
    n = min(d.size, f.size)
    if n < 2:
        return [], 0.0

    d = d[:n]
    f = f[:n]

    mask = np.isfinite(d) & np.isfinite(f)
    d = d[mask]
    f = f[mask].astype(int)

    if d.size < 2:
        return [], 0.0

    # ordena por depth (MD/TVD)
    order = np.argsort(d)
    d = d[order]
    f = f[order]

    dz = np.diff(d)
    dz = np.abs(dz)
    fac_step = f[:-1]

    # remove passos com dz ~ 0
    keep = dz > 1e-12
    dz = dz[keep]
    fac_step = fac_step[keep]

    if dz.size == 0:
        return [], 0.0

    runs = []
    cur_fac = int(fac_step[0])
    cur_t = float(dz[0])

    for ff, tt in zip(fac_step[1:], dz[1:]):
        ff = int(ff)
        tt = float(tt)
        if ff == cur_fac:
            cur_t += tt
        else:
            runs.append((cur_fac, cur_t))
            cur_fac = ff
            cur_t = tt
    runs.append((cur_fac, cur_t))

    t_total = float(sum(t for _, t in runs))
    return runs, t_total


def _merge_adjacent_same_facies(runs):
    if not runs:
        return []
    out = [runs[0]]
    for fac, t in runs[1:]:
        pf, pt = out[-1]
        if int(fac) == int(pf):
            out[-1] = (pf, float(pt) + float(t))
        else:
            out.append((int(fac), float(t)))
    return out


def _merge_thin_runs(runs, t_min):
    """Mescla runs finos (< t_min) no vizinho (preferindo o vizinho mais espesso)."""
    if not runs:
        return []
    try:
        t_min = float(t_min)
    except Exception:
        t_min = 0.0
    if t_min <= 0.0 or len(runs) <= 1:
        return _merge_adjacent_same_facies(runs)

    runs = [(int(f), float(t)) for f, t in runs]
    runs = _merge_adjacent_same_facies(runs)

    changed = True
    # iterativo: cada merge pode criar novos runs finos
    while changed and len(runs) > 1:
        changed = False
        for i, (fac, t) in enumerate(runs):
            if t >= t_min:
                continue

            # escolhe alvo
            if i == 0:
                # merge no próximo
                f2, t2 = runs[1]
                runs[1] = (f2, t2 + t)
                runs.pop(0)
            elif i == len(runs) - 1:
                f1, t1 = runs[-2]
                runs[-2] = (f1, t1 + t)
                runs.pop(-1)
            else:
                fL, tL = runs[i - 1]
                fR, tR = runs[i + 1]
                # vai para o vizinho mais espesso (tL vs tR)
                if tL >= tR:
                    runs[i - 1] = (fL, tL + t)
                    runs.pop(i)
                else:
                    runs[i + 1] = (fR, tR + t)
                    runs.pop(i)

            runs = _merge_adjacent_same_facies(runs)
            changed = True
            break

    return _merge_adjacent_same_facies(runs)


def _runs_to_proportions(runs):
    """Converte runs (facies, thickness) em proporções por espessura."""
    if not runs:
        return {}, 0.0
    tot = float(sum(t for _, t in runs))
    if tot <= 1e-12:
        return {}, 0.0
    acc = {}
    for fac, t in runs:
        fac = int(fac)
        acc[fac] = acc.get(fac, 0.0) + float(t)
    props = {fac: float(th / tot) for fac, th in acc.items()}
    return props, tot


def compute_well_match_score_from_profiles(
    real_depth, real_fac,
    sim_depth,  sim_fac,
    *,
    t_min=0.30,
    ignore_real_zeros=True,
    **_ignored,
):
    """Score 0..1 baseado em PROPORÇÕES por espessura (sem bin-a-bin / sem kappa).

    Pipeline:
      1) Converte REAL e SIM em runs (facies, thickness) usando diferenças de depth.
      2) (Opcional) remove facies 0 no REAL (muito comum ser "sem dado") antes de calcular proporções.
      3) Mescla segmentos finos (< t_min) no vizinho para reduzir ruído estratigráfico.
      4) Calcula proporções p(f) = T_f / T_total e score por distância L1:

         D = 0.5 * sum_f | p_real(f) - p_sim(f) |
         Score = 1 - D

    Retorna dict com:
      score, prop_score, prop_distance, t_real, t_sim, t_real_valid, t_sim_valid,
      runs_real, runs_sim, props_real, props_sim
    """
    # runs "brutos"
    runs_r, t_r = _profile_to_runs(real_depth, real_fac)
    runs_s, t_s = _profile_to_runs(sim_depth,  sim_fac)

    # ignora zeros no REAL (opcional) *antes* de suavizar
    if ignore_real_zeros and runs_r:
        runs_r = [(f, t) for (f, t) in runs_r if int(f) != 0]

    # suavização por espessura mínima
    runs_r_f = _merge_thin_runs(runs_r, t_min=t_min)
    runs_s_f = _merge_thin_runs(runs_s, t_min=t_min)

    props_r, t_rv = _runs_to_proportions(runs_r_f)
    props_s, t_sv = _runs_to_proportions(runs_s_f)

    keys = set(props_r.keys()) | set(props_s.keys())
    l1 = 0.0
    for k in keys:
        l1 += abs(float(props_r.get(k, 0.0)) - float(props_s.get(k, 0.0)))

    prop_distance = 0.5 * float(l1)
    prop_score = float(np.clip(1.0 - prop_distance, 0.0, 1.0))

    # mantém compatibilidade com campos antigos (não usados no novo ranking)
    return {
        "score": prop_score,
        "prop_score": prop_score,
        "prop_distance": prop_distance,
        "props_real": props_r,
        "props_sim": props_s,
        "runs_real": runs_r_f,
        "runs_sim": runs_s_f,
        "t_real": float(t_r),
        "t_sim": float(t_s),
        "t_real_valid": float(t_rv),
        "t_sim_valid": float(t_sv),
        # campos legacy:
        "strat_acc": 0.0,
        "strat_kappa_norm": 0.0,
        "thick_score": 0.0,
        "n_valid_bins": int(max(0.0, t_rv) * 1000.0),  # só pra não virar zero à toa
    }


def compute_well_match_score(
    real_depth, real_facies,
    sim_depth, sim_facies,
    *,
    t_min=0.30,
    ignore_real_zeros=True,
    **kwargs,
):
    """Alias: mantém window.py funcionando, mas agora usa score por proporção."""
    return compute_well_match_score_from_profiles(
        real_depth, real_facies,
        sim_depth, sim_facies,
        t_min=t_min,
        ignore_real_zeros=ignore_real_zeros,
        **kwargs,
    )


def compute_well_fit_score(
    real_depth, real_facies,
    sim_depth, sim_facies,
    *,
    t_min=0.30,
    ignore_real_zeros=True,
    **kwargs,
):
    """Alias do score final (mesma lógica do match score)."""
    return compute_well_match_score(
        real_depth, real_facies,
        sim_depth, sim_facies,
        t_min=t_min,
        ignore_real_zeros=ignore_real_zeros,
        **kwargs,
    )

def generate_detailed_metrics_df(facies_array, target_grid=None):
    """Gera DataFrame detalhado com volumes baseados no grid fornecido."""
    from load_data import nx, ny, nz
    
    arr = np.asarray(facies_array, dtype=int)
    total_cells = arr.size
    
    volumes = _get_cell_volumes(target_grid)
    z_vals = _get_cell_z_coords(target_grid)
    
    # Fallback se dimensão não bater (segurança)
    if volumes.size != total_cells:
        volumes = np.ones(total_cells)
        z_vals = np.zeros(total_cells)
        
    unique_f = np.unique(arr)
    data_list = []
    
    # Tenta criar estrutura para label 3D
    try:
        struct = generate_binary_structure(3, 1)
    except: 
        struct = None
    
    for f in unique_f:
        mask = (arr == f)
        count = int(mask.sum())
        if count == 0: continue
        
        frac = count / total_cells
        vol_tot = float(volumes[mask].sum())
        
        # Clusters e Conectividade
        n_clus = 0; largest_size = 0; vol_largest = 0.0; thick = 0.0
        conn = 0.0
        perc_x = False; perc_y = False; perc_z = False
        
        if struct is not None:
            try:
                # Reshape para 3D (F-order = I, J, K do Eclipse)
                mask_3d = mask.reshape((nx, ny, nz), order="F")
                
                # Labeling (scipy usa C-order, então transpomos)
                lbl_3d, n_clus = nd_label(mask_3d.transpose(2,1,0), structure=struct)
                
                if n_clus > 0:
                    lbl_flat = lbl_3d.transpose(2,1,0).reshape(-1, order="F")
                    counts = np.bincount(lbl_flat)
                    counts[0] = 0 # ignora background 0
                    
                    largest_idx = counts.argmax()
                    largest_size = counts[largest_idx]
                    
                    mask_largest = (lbl_flat == largest_idx)
                    vol_largest = float(volumes[mask_largest].sum())
                    
                    zs = z_vals[mask_largest]
                    if zs.size > 0:
                        thick = float(zs.max() - zs.min())
                        
                    # Percolação (usa o array 3D transposto lbl_3d [z,y,x])
                    # X (última dimensão do lbl_3d)
                    left = lbl_3d[:, :, 0]; right = lbl_3d[:, :, -1]
                    perc_x = bool(np.intersect1d(left[left>0], right[right>0]).size > 0)
                    
                    # Y (dimensão do meio)
                    front = lbl_3d[:, 0, :]; back = lbl_3d[:, -1, :]
                    perc_y = bool(np.intersect1d(front[front>0], back[back>0]).size > 0)
                    
                    # Z (primeira dimensão)
                    top = lbl_3d[0, :, :]; bottom = lbl_3d[-1, :, :]
                    perc_z = bool(np.intersect1d(top[top>0], bottom[bottom>0]).size > 0)
                
                conn = largest_size / count
            except Exception: 
                pass
            
        data_list.append({
            "facies": int(f),
            "cells": count,
            "fraction": frac,
            "n_clusters": n_clus,
            "largest_size": largest_size,
            "connected_fraction": conn,
            "volume_total": vol_tot,
            "volume_largest_cluster": vol_largest,
            "thickness_largest_cluster": thick,
            "Perc_X": perc_x,
            "Perc_Y": perc_y,
            "Perc_Z": perc_z
        })
            
    return pd.DataFrame(data_list)

def compute_facies_entropy_map(list_of_facies_arrays, target_grid=None):
    """
    Calcula a entropia de Shannon célula a célula para Fácies (Discreto).
    H(x) = - sum(p * log(p))
    """
    import numpy as np
    
    if not list_of_facies_arrays:
        if target_grid: return np.zeros(target_grid.n_cells)
        return np.array([])

    try:
        # Stack shape: (N_modelos, N_celulas)
        stack = np.vstack(list_of_facies_arrays)
    except ValueError:
        if target_grid: return np.zeros(target_grid.n_cells)
        return np.array([])

    n_models, n_cells = stack.shape
    if n_models < 1: return np.zeros(n_cells)

    # Identifica todas as classes únicas
    unique_vals = np.unique(stack)
    entropy_map = np.zeros(n_cells, dtype=float)
    
    # Vetorizado: para cada fácies, calcula P(x) e soma -P*log(P)
    for val in unique_vals:
        # Conta ocorrências desta fácies em cada célula
        counts = (stack == val).sum(axis=0)
        probs = counts / n_models
        
        # Evita log(0)
        mask = probs > 0
        p_valid = probs[mask]
        
        entropy_map[mask] -= p_valid * np.log(p_valid)
        
    return entropy_map

def compute_continuous_uncertainty_map(list_of_arrays, target_grid=None, metric="std"):
    """
    Calcula incerteza para variáveis contínuas (ex: Porosidade, NTG).
    metric="std" -> Desvio Padrão (Standard Deviation)
    metric="var" -> Variância
    metric="range" -> Max - Min
    """
    import numpy as np
    
    if not list_of_arrays:
        if target_grid: return np.zeros(target_grid.n_cells)
        return np.array([])

    try:
        stack = np.vstack(list_of_arrays) # (N_modelos, N_celulas)
    except ValueError:
        return np.zeros(target_grid.n_cells) if target_grid else np.array([])

    if metric == "std":
        return np.std(stack, axis=0)
    elif metric == "var":
        return np.var(stack, axis=0)
    elif metric == "range":
        return np.ptp(stack, axis=0) # Peak to peak (Max - Min)
    
    return np.zeros(stack.shape[1])

# =============================================================================
# MÉDIA PONDERADA POR ESPESSURA E INCERTEZA DO ENSEMBLE
# =============================================================================

def _ensemble_stat(stack, metric="std"):
    """
    Calcula estatística entre modelos no eixo 0.

    stack shape:
        (n_modelos, n_amostras)

    metric:
        "mean"  -> média entre cenários
        "std"   -> desvio padrão entre cenários
        "var"   -> variância entre cenários
        "range" -> amplitude: max - min
    """
    import numpy as np

    stack = np.asarray(stack, dtype=float)

    if stack.size == 0:
        return np.array([])

    metric = str(metric or "std").lower()

    if metric in ("mean", "media", "média", "mu"):
        return np.nanmean(stack, axis=0)

    if metric in ("std", "desvio", "desvio_padrao", "desvio_padrão", "sigma"):
        return np.nanstd(stack, axis=0)

    if metric in ("var", "variance", "variancia", "variância"):
        return np.nanvar(stack, axis=0)

    if metric in ("range", "amplitude", "amp"):
        return np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)

    return np.nanstd(stack, axis=0)


def compute_continuous_uncertainty_map(list_of_arrays, target_grid=None, metric="std"):
    """
    Calcula estatísticas entre cenários para variáveis contínuas célula a célula.

    metric:
        "mean"  -> média entre cenários
        "std"   -> desvio padrão
        "var"   -> variância
        "range" -> amplitude
    """
    import numpy as np

    if not list_of_arrays:
        if target_grid is not None:
            return np.zeros(target_grid.n_cells, dtype=float)
        return np.array([], dtype=float)

    try:
        stack = np.vstack([np.asarray(a, dtype=float).ravel() for a in list_of_arrays])
    except Exception:
        return np.zeros(target_grid.n_cells, dtype=float) if target_grid is not None else np.array([], dtype=float)

    return _ensemble_stat(stack, metric=metric)


def compute_thickness_weighted_property_map(
    target_grid,
    scalar_name,
    output_name=None,
    *,
    clip_to_01=False,
):
    """
    Calcula a média ponderada por espessura de uma propriedade por coluna.

    Para cada coluna (i,j):

        mean_h = sum_k(h_ijk * p_ijk) / sum_k(h_ijk)

    O resultado é salvo como cell_data replicado verticalmente na coluna,
    permitindo visualização 3D e redução 2D pelo fluxo atual do programa.

    Retorna:
        output_name, out_2d
    """
    import numpy as np

    if target_grid is None:
        return None, None

    if scalar_name not in target_grid.cell_data:
        return None, None

    if target_grid.n_cells != nx * ny * nz:
        return None, None

    prop = np.asarray(target_grid.cell_data[scalar_name], dtype=float)
    th = _get_cell_thickness(target_grid)

    if prop.size != target_grid.n_cells or th is None or len(th) != target_grid.n_cells:
        return None, None

    try:
        prop_3d = prop.reshape((nx, ny, nz), order="F")
        th_3d = np.asarray(th, dtype=float).reshape((nx, ny, nz), order="F")
    except Exception:
        return None, None

    out_2d = np.full((nx, ny), np.nan, dtype=float)
    out_3d = np.zeros((nx, ny, nz), dtype=float)

    for ix in range(nx):
        for iy in range(ny):
            p_col = prop_3d[ix, iy, :]
            h_col = th_3d[ix, iy, :]

            mask = np.isfinite(p_col) & np.isfinite(h_col) & (h_col > 0.0)
            if not np.any(mask):
                continue

            p = p_col[mask]
            h = h_col[mask]

            if clip_to_01:
                p = np.clip(p, 0.0, 1.0)

            h_sum = float(np.sum(h))
            if h_sum <= 0.0:
                continue

            val = float(np.sum(h * p) / h_sum)

            out_2d[ix, iy] = val

            # replica o valor na coluna para visualização 3D
            out_3d[ix, iy, mask] = val

    if output_name is None:
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in str(scalar_name))
        output_name = f"wmean_th_{safe}"

    target_grid.cell_data[output_name] = out_3d.reshape(-1, order="F")
    return output_name, out_2d


def reduce_grid_scalar_to_column_map(
    target_grid,
    scalar_name,
    *,
    reduction="max",
    clip_to_01=False,
):
    """
    Reduz um campo 3D para mapa 2D por coluna.

    reduction:
        "max"           -> máximo finito da coluna
        "mean"          -> média aritmética
        "sum"           -> soma
        "weighted_mean" -> média ponderada por espessura
        "equivalent"    -> soma h*p, útil para propriedades fracionárias
    """
    import numpy as np

    if target_grid is None:
        return None

    if scalar_name == "__total_column_thickness__":
        th = _get_cell_thickness(target_grid)
        if th is None or len(th) != target_grid.n_cells:
            return None

        try:
            th_3d = np.asarray(th, dtype=float).reshape((nx, ny, nz), order="F")
        except Exception:
            return None

        out = np.full((nx, ny), np.nan, dtype=float)
        for ix in range(nx):
            for iy in range(ny):
                col = th_3d[ix, iy, :]
                mask = np.isfinite(col) & (col > 0.0)
                if np.any(mask):
                    out[ix, iy] = float(np.sum(col[mask]))
        return out

    if scalar_name not in target_grid.cell_data:
        return None

    if target_grid.n_cells != nx * ny * nz:
        return None

    arr = np.asarray(target_grid.cell_data[scalar_name], dtype=float)
    if arr.size != target_grid.n_cells:
        return None

    try:
        arr_3d = arr.reshape((nx, ny, nz), order="F")
    except Exception:
        return None

    reduction = str(reduction or "max").lower()
    out = np.full((nx, ny), np.nan, dtype=float)

    if reduction in ("weighted_mean", "media_ponderada", "média_ponderada", "equivalent", "equivalente"):
        th = _get_cell_thickness(target_grid)
        if th is None or len(th) != target_grid.n_cells:
            return None

        try:
            th_3d = np.asarray(th, dtype=float).reshape((nx, ny, nz), order="F")
        except Exception:
            return None

        for ix in range(nx):
            for iy in range(ny):
                p_col = arr_3d[ix, iy, :]
                h_col = th_3d[ix, iy, :]

                mask = np.isfinite(p_col) & np.isfinite(h_col) & (h_col > 0.0)
                if not np.any(mask):
                    continue

                p = p_col[mask]
                h = h_col[mask]

                if clip_to_01:
                    p = np.clip(p, 0.0, 1.0)

                eq = float(np.sum(h * p))

                if reduction in ("equivalent", "equivalente"):
                    out[ix, iy] = eq
                else:
                    h_sum = float(np.sum(h))
                    out[ix, iy] = eq / h_sum if h_sum > 0.0 else np.nan

        return out

    for ix in range(nx):
        for iy in range(ny):
            col = arr_3d[ix, iy, :]
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                continue

            if reduction == "mean":
                out[ix, iy] = float(np.nanmean(finite))
            elif reduction == "sum":
                out[ix, iy] = float(np.nansum(finite))
            else:
                out[ix, iy] = float(np.nanmax(finite))

    return out


def compute_column_ensemble_stat_map(
    list_of_grids,
    scalar_name,
    *,
    metric="std",
    reduction="weighted_mean",
    clip_to_01=False,
):
    """
    Calcula estatística entre cenários depois de reduzir cada modelo para mapa 2D.

    Exemplo:
        - primeiro calcula Qv(i,j) para cada modelo;
        - depois calcula std(Qv)(i,j), var(Qv)(i,j), range(Qv)(i,j), etc.
    """
    import numpy as np

    maps = []

    for g in list_of_grids or []:
        m2d = reduce_grid_scalar_to_column_map(
            g,
            scalar_name,
            reduction=reduction,
            clip_to_01=clip_to_01,
        )
        if m2d is None:
            continue
        maps.append(np.asarray(m2d, dtype=float).reshape(-1, order="F"))

    if not maps:
        return None

    try:
        stack = np.vstack(maps)
    except Exception:
        return None

    out_flat = _ensemble_stat(stack, metric=metric)

    try:
        return out_flat.reshape((nx, ny), order="F")
    except Exception:
        return None


def expand_column_map_to_cell_data(target_grid, column_map_2d, output_name):
    """
    Expande um mapa 2D (nx,ny) para cell_data 3D replicado em k.
    Útil para visualizar incerteza por coluna no viewer 3D.
    """
    import numpy as np

    if target_grid is None or column_map_2d is None:
        return None

    arr2d = np.asarray(column_map_2d, dtype=float)

    try:
        arr2d = arr2d.reshape((nx, ny), order="F")
    except Exception:
        return None

    out3d = np.zeros((nx, ny, nz), dtype=float)

    for ix in range(nx):
        for iy in range(ny):
            val = arr2d[ix, iy]
            if np.isfinite(val):
                out3d[ix, iy, :] = float(val)

    out1d = out3d.reshape(-1, order="F")
    target_grid.cell_data[output_name] = out1d
    return out1d


def compute_model_level_ensemble_summary(
    list_of_grids,
    model_names,
    scalar_name,
    *,
    reduction="weighted_mean",
    clip_to_01=False,
):
    """
    Resume cada modelo em um valor único e calcula estatísticas do ensemble.

    Para cada modelo:
        valor_modelo = média espacial do mapa reduzido por coluna

    Depois, entre modelos:
        média, desvio padrão, variância e amplitude.
    """
    import numpy as np
    import pandas as pd

    rows = []

    for g, name in zip(list_of_grids or [], model_names or []):
        m2d = reduce_grid_scalar_to_column_map(
            g,
            scalar_name,
            reduction=reduction,
            clip_to_01=clip_to_01,
        )

        if m2d is None:
            continue

        finite = np.asarray(m2d, dtype=float)
        finite = finite[np.isfinite(finite)]

        if finite.size == 0:
            continue

        rows.append({
            "modelo": str(name),
            "valor_medio_espacial": float(np.nanmean(finite)),
            "desvio_espacial": float(np.nanstd(finite)),
            "min_espacial": float(np.nanmin(finite)),
            "max_espacial": float(np.nanmax(finite)),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return {
            "per_model": df,
            "ensemble": {
                "mean": 0.0,
                "std": 0.0,
                "var": 0.0,
                "range": 0.0,
            }
        }

    vals = df["valor_medio_espacial"].to_numpy(dtype=float)

    ens_mean = float(np.nanmean(vals))
    ens_std = float(np.nanstd(vals))
    ens_var = float(np.nanvar(vals))
    ens_range = float(np.nanmax(vals) - np.nanmin(vals))

    df["desvio_abs_da_media_ensemble"] = np.abs(df["valor_medio_espacial"] - ens_mean)

    return {
        "per_model": df,
        "ensemble": {
            "mean": ens_mean,
            "std": ens_std,
            "var": ens_var,
            "range": ens_range,
        }
    }