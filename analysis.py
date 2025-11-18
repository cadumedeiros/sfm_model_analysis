# analysis.py
import numpy as np
import pandas as pd
import os
from datetime import datetime
import pyvista as pv

from load_data import grid, facies, nx, ny, nz
from derived_fields import ensure_reservoir, ensure_clusters

def compute_global_metrics(reservoir_facies):
    res_arr = ensure_reservoir(reservoir_facies)
    clusters_arr, _ = ensure_clusters(reservoir_facies)

    total_cells = facies.size
    res_cells = int(res_arr.sum())
    ntg = res_cells / total_cells if total_cells else 0.0

    counts = np.bincount(clusters_arr)
    if counts.size > 0:
        counts[0] = 0

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
    }

def compute_directional_percolation(reservoir_facies):
    # usa o mesmo array de clusters que já está no grid
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
        "x_perc": bool(x_common),
        "x_clusters": x_common,
        "y_perc": bool(y_common),
        "y_clusters": y_common,
        "z_perc": bool(z_common),
        "z_clusters": z_common,
    }

def get_cluster_sizes(reservoir_facies):
    """
    Devolve um array 1D só com os tamanhos dos clusters de reservatório.
    """
    _, _ = ensure_reservoir(reservoir_facies), ensure_clusters(reservoir_facies)
    clusters_arr, _ = ensure_clusters(reservoir_facies)

    counts = np.bincount(clusters_arr)
    if counts.size > 0:
        counts[0] = 0  # tira o fundo
    cluster_sizes = counts[counts > 0]
    return cluster_sizes


# --- análise fácie a fácie com volume ---

def _get_cell_centers_z():
    # pega centros das células (x, y, z)
    centers = grid.cell_centers().points
    # z é a terceira coluna
    return centers[:, 2]

def _get_cell_volumes():
    # 1) já tem volume? usa
    for key in ("Volume", "Volume ", "volume", "Volume_"):
        if key in grid.cell_data:
            return grid.cell_data[key]

    # 2) não tem: calcula num grid temporário e copia pro original
    tmp = grid.compute_cell_sizes(length=False, area=False, volume=True)
    vol_arr = None
    for key in ("Volume", "Volume ", "volume", "Volume_"):
        if key in tmp.cell_data:
            vol_arr = tmp.cell_data[key]
            break

    if vol_arr is None:
        raise RuntimeError("Não foi possível obter o volume das células do grid.")

    # cola no grid original pra futuras chamadas
    grid.cell_data["Volume"] = vol_arr
    return vol_arr


def _label_single_facies(facie_id: int):
    is_fac = (facies == facie_id).astype(np.uint8)

    arr_xyz = is_fac.reshape((nx, ny, nz), order="F")   # (x, y, z)
    arr_zyx = arr_xyz.transpose(2, 1, 0)                # (z, y, x)

    from scipy.ndimage import label, generate_binary_structure
    structure = generate_binary_structure(3, 1)
    labeled_zyx, n_clusters = label(arr_zyx, structure=structure)

    labeled_xyz = labeled_zyx.transpose(2, 1, 0)
    clusters_1d = labeled_xyz.reshape(-1, order="F").astype(np.int32)
    return clusters_1d, int(n_clusters)

def _directional_perc_from_clusters(clusters_1d: np.ndarray):
    clusters_xyz = clusters_1d.reshape((nx, ny, nz), order="F")

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
        "x_perc": bool(x_common),
        "x_clusters": x_common,
        "y_perc": bool(y_common),
        "y_clusters": y_common,
        "z_perc": bool(z_common),
        "z_clusters": z_common,
    }

def compute_facies_metrics():
    total_cells = facies.size
    unique_facies = np.unique(facies)

    volumes = _get_cell_volumes()
    volumes = np.abs(volumes)  # garante que não fica negativo por orientação
    z_centers = _get_cell_centers_z()

    results = []

    for fac in unique_facies:
        fac_mask = (facies == fac)
        fac_cells = int(fac_mask.sum())
        fac_fraction = fac_cells / total_cells if total_cells else 0.0

        # volume total da fácies
        fac_volume = float(volumes[fac_mask].sum()) if fac_cells > 0 else 0.0

        # clusters dessa fácies
        fac_clusters, n_clusters = _label_single_facies(fac)

        counts = np.bincount(fac_clusters)
        if counts.size > 0:
            counts[0] = 0
            largest_label = counts.argmax()
            largest_size = int(counts[largest_label]) if largest_label > 0 else 0
        else:
            largest_label = 0
            largest_size = 0

        connected_fraction = (largest_size / fac_cells) if fac_cells > 0 else 0.0

        # volume do maior cluster + espessura
        if largest_label > 0:
            largest_mask = (fac_clusters == largest_label)

            largest_volume = float(volumes[largest_mask].sum())

            # espessura = zmax - zmin das células do maior cluster
            z_vals = z_centers[largest_mask]
            if z_vals.size > 0:
                thickness = float(z_vals.max() - z_vals.min())
            else:
                thickness = 0.0
        else:
            largest_volume = 0.0
            thickness = 0.0

        perc = _directional_perc_from_clusters(fac_clusters)

        fac_result = {
            "facies": int(fac),
            "cells": fac_cells,
            "fraction": fac_fraction,
            "n_clusters": int(n_clusters),
            "largest_label": int(largest_label),
            "largest_size": int(largest_size),
            "connected_fraction": float(connected_fraction),
            "volume_total": fac_volume,
            "volume_largest_cluster": largest_volume,
            "thickness_largest_cluster": thickness,
            "perc": perc,
        }
        results.append(fac_result)

    return results

def export_facies_metrics_to_excel(output_path=None):
    data = compute_facies_metrics()
    df = pd.DataFrame(data)

    # expande percolação
    perc_df = df["perc"].apply(pd.Series)
    perc_df = perc_df.rename(columns={
        "x_perc": "Perc_X",
        "y_perc": "Perc_Y",
        "z_perc": "Perc_Z"
    })
    df = pd.concat([df.drop(columns=["perc"]), perc_df[["Perc_X", "Perc_Y", "Perc_Z"]]], axis=1)

    if output_path is None:
        output_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "facies_metrics.xlsx")

    try:
        df.to_excel(output_path, index=False, sheet_name="Facies Metrics")
        print(f"\n✅ Métricas por fácies exportadas para: {output_path}")
    except PermissionError:
        # provavelmente o arquivo está aberto no Excel
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
            if col_vals.size > 0:
                thickness_2d[ix, iy] = col_vals.max()

    x_min, x_max, y_min, y_max, z_min, z_max = grid.bounds
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    xs, ys = np.meshgrid(xs, ys, indexing="ij")
    zs = np.full_like(xs, z_max)

    surf = pv.StructuredGrid(xs, ys, zs)

    thickness_cells = thickness_2d[:nx-1, :ny-1].ravel(order="F")

    surf.cell_data[array_name_2d] = thickness_cells
    return surf

def add_vertical_facies_metrics(facies_selected, prefix="vert_"):
    """
    Calcula métricas verticais (T_tot, T_env, NTG_col, NTG_env) considerando
    TODAS as fácies selecionadas como parte do reservatório.
    facies_selected: int, list, set
    """

    # normalizar entrada para conjunto de ints
    if isinstance(facies_selected, (int, np.integer)):
        facies_set = {int(facies_selected)}
    else:
        facies_set = {int(f) for f in facies_selected}

    facies_xyz = facies.reshape((nx, ny, nz), order="F")

    # geometria vertical média
    z_min, z_max = grid.bounds[4], grid.bounds[5]
    dz_mean = (z_max - z_min) / nz
    T_col_total = (z_max - z_min)

    # arrays 3D de saída
    Ttot_3d   = np.zeros((nx, ny, nz), dtype=float)
    Tenv_3d   = np.zeros((nx, ny, nz), dtype=float)
    NTGcol_3d = np.zeros((nx, ny, nz), dtype=float)
    NTGenv_3d = np.zeros((nx, ny, nz), dtype=float)

    for ix in range(nx):
        for iy in range(ny):
            col = facies_xyz[ix, iy, :]

            # máscara combinada: todas as fácies selecionadas
            mask = np.isin(col, list(facies_set))

            if not mask.any():
                continue

            z_idx = np.where(mask)[0]
            n_tot = z_idx.size
            n_env = int(z_idx[-1] - z_idx[0] + 1)

            # convertendo para espessuras em metros
            T_tot = n_tot * dz_mean
            T_env = n_env * dz_mean

            NTG_col = T_tot / T_col_total if T_col_total > 0 else 0.0
            NTG_env = T_tot / T_env if T_env > 0 else 0.0

            NTG_col = max(0.0, min(1.0, NTG_col))
            NTG_env = max(0.0, min(1.0, NTG_env))

            # grava os valores em TODAS as células da coluna pertencentes ao conjunto de fácies
            Ttot_3d[ix, iy, mask]   = T_tot
            Tenv_3d[ix, iy, mask]   = T_env
            NTGcol_3d[ix, iy, mask] = NTG_col
            NTGenv_3d[ix, iy, mask] = NTG_env

    # salva arrays — não usa mais id único na chave
    name_Ttot   = prefix + "Ttot_reservoir"
    name_Tenv   = prefix + "Tenv_reservoir"
    name_NTGcol = prefix + "NTG_col_reservoir"
    name_NTGenv = prefix + "NTG_env_reservoir"

    grid.cell_data[name_Ttot]   = Ttot_3d.reshape(-1, order="F")
    grid.cell_data[name_Tenv]   = Tenv_3d.reshape(-1, order="F")
    grid.cell_data[name_NTGcol] = NTGcol_3d.reshape(-1, order="F")
    grid.cell_data[name_NTGenv] = NTGenv_3d.reshape(-1, order="F")

    print("Métricas verticais recalculadas para conjunto de fácies:", facies_set)
    print("  →", name_Ttot)
    print("  →", name_Tenv)
    print("  →", name_NTGcol)
    print("  →", name_NTGenv)


