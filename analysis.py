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
    """Retorna array de volumes das células. Calcula se não existir."""
    g = target_grid if target_grid is not None else grid
    
    # 1) Tenta pegar cache
    for key in ("Volume", "Volume ", "volume", "Volume_"):
        if key in g.cell_data:
            return g.cell_data[key]

    # 2) Calcula
    tmp = g.compute_cell_sizes(length=False, area=False, volume=True)
    vol_arr = None
    for key in ("Volume", "Volume ", "volume", "Volume_"):
        if key in tmp.cell_data:
            vol_arr = tmp.cell_data[key]
            break

    if vol_arr is None:
        return np.ones(g.n_cells) 

    g.cell_data["Volume"] = vol_arr
    return vol_arr

def _get_cell_z_coords(target_grid=None):
    g = target_grid if target_grid is not None else grid
    return g.cell_centers().points[:, 2]

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

def compute_global_metrics_for_array(facies_array, reservoir_facies):
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

    if res_cells == 0:
        return ({
            "total_cells": int(total_cells), "res_cells": 0, "ntg": 0.0,
            "n_clusters": 0, "largest_label": 0, "largest_size": 0, "connected_fraction": 0.0,
        }, {
            "x_perc": False, "x_clusters": set(), "y_perc": False, "y_clusters": set(), "z_perc": False, "z_clusters": set(),
        })

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

    metrics = {
        "total_cells": int(total_cells), "res_cells": res_cells, "ntg": float(ntg),
        "n_clusters": n_clusters, "largest_label": largest_label,
        "largest_size": largest_size, "connected_fraction": float(connected_fraction),
    }
    perc = {
        "x_perc": bool(x_common), "x_clusters": x_common,
        "y_perc": bool(y_common), "y_clusters": y_common,
        "z_perc": bool(z_common), "z_clusters": z_common,
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

def add_vertical_facies_metrics(facies_selected, prefix="vert_"):
    if isinstance(facies_selected, (int, np.integer)):
        facies_set = {int(facies_selected)}
    else:
        facies_set = {int(f) for f in facies_selected}

    facies_xyz = facies.reshape((nx, ny, nz), order="F")
    z_xyz = grid.cell_centers().points[:, 2].reshape((nx, ny, nz), order="F")

    Ttot_3d   = np.zeros((nx, ny, nz), dtype=float)
    Tenv_3d   = np.zeros((nx, ny, nz), dtype=float)
    NTGcol_3d = np.zeros((nx, ny, nz), dtype=float)
    NTGenv_3d = np.zeros((nx, ny, nz), dtype=float)
    Npack_3d     = np.zeros((nx, ny, nz), dtype=float)
    Tpackmax_3d  = np.zeros((nx, ny, nz), dtype=float)
    Tgap_sum_3d  = np.zeros((nx, ny, nz), dtype=float)
    Tgap_max_3d  = np.zeros((nx, ny, nz), dtype=float)
    ICV_3d       = np.zeros((nx, ny, nz), dtype=float)
    Qv_3d        = np.zeros((nx, ny, nz), dtype=float)
    Qvabs_3d     = np.zeros((nx, ny, nz), dtype=float)

    for ix in range(nx):
        for iy in range(ny):
            col_fac = facies_xyz[ix, iy, :]
            col_z   = z_xyz[ix, iy, :]
            z_col_min = float(np.nanmin(col_z))
            z_col_max = float(np.nanmax(col_z))
            T_col_total = abs(z_col_max - z_col_min)
            dz_mean_col = T_col_total / nz if nz > 0 else 0.0
            mask = np.isin(col_fac, list(facies_set))
            if not mask.any() or dz_mean_col == 0.0 or T_col_total == 0.0: continue

            idx = np.where(mask)[0]
            n_tot = idx.size
            n_env = int(idx[-1] - idx[0] + 1)
            T_tot = n_tot * dz_mean_col
            T_env = n_env * dz_mean_col
            NTG_col = T_tot / T_col_total if T_col_total > 0 else 0.0
            NTG_env = T_tot / T_env if T_env > 0 else 0.0

            packages = []
            start = idx[0]; prev = idx[0]
            for k in idx[1:]:
                if k == prev + 1: prev = k
                else: packages.append((start, prev)); start = prev = k
            packages.append((start, prev))

            n_packages = len(packages)
            thickness_packs = []
            for (k0, k1) in packages:
                n_cells = int(k1 - k0 + 1)
                thickness_packs.append(n_cells * dz_mean_col)
            Tpack_max = max(thickness_packs) if thickness_packs else 0.0

            gaps = []
            if n_packages > 1:
                for (s1, e1), (s2, e2) in zip(packages[:-1], packages[1:]):
                    gap_cells = int(s2 - e1 - 1)
                    if gap_cells > 0: gaps.append(gap_cells * dz_mean_col)
            Tgap_sum = float(sum(gaps)) if gaps else 0.0
            Tgap_max = max(gaps) if gaps else 0.0

            ICV = Tpack_max / T_env if T_env > 0 else 0.0
            ICV = max(0.0, min(1.0, ICV))
            Qv = NTG_col * ICV
            frac_pack = Tpack_max / T_col_total if T_col_total > 0 else 0.0
            Qv_abs = ICV * frac_pack

            Ttot_3d[ix, iy, mask]   = T_tot
            Tenv_3d[ix, iy, mask]   = T_env
            NTGcol_3d[ix, iy, mask] = NTG_col
            NTGenv_3d[ix, iy, mask] = NTG_env
            Npack_3d[ix, iy, mask]    = float(n_packages)
            Tpackmax_3d[ix, iy, mask] = Tpack_max
            Tgap_sum_3d[ix, iy, mask] = Tgap_sum
            Tgap_max_3d[ix, iy, mask] = Tgap_max
            ICV_3d[ix, iy, mask]      = ICV
            Qv_3d[ix, iy, mask]       = Qv
            Qvabs_3d[ix, iy, mask]    = Qv_abs

    name_map = {
        "Ttot_reservoir": Ttot_3d, "Tenv_reservoir": Tenv_3d,
        "NTG_col_reservoir": NTGcol_3d, "NTG_env_reservoir": NTGenv_3d,
        "n_packages_reservoir": Npack_3d, "Tpack_max_reservoir": Tpackmax_3d,
        "Tgap_sum_reservoir": Tgap_sum_3d, "Tgap_max_reservoir": Tgap_max_3d,
        "ICV_reservoir": ICV_3d, "Qv_reservoir": Qv_3d, "Qv_abs_reservoir": Qvabs_3d
    }
    for k, v in name_map.items():
        grid.cell_data[prefix + k] = v.reshape(-1, order="F")