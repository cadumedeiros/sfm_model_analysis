# analysis.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import pyvista as pv

from load_data import facies, nx, ny, nz, grid
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

def plot_cluster_histogram(reservoir_facies, bins=30):
    """
    Plota o histograma reaproveitando o que tinha no hist_clusters.py
    """
    cluster_sizes = get_cluster_sizes(reservoir_facies)

    print(f"Nº de clusters: {cluster_sizes.size}")
    if cluster_sizes.size:
        print(f"Tamanho mínimo: {cluster_sizes.min()}")
        print(f"Tamanho máximo: {cluster_sizes.max()}")

    plt.figure()
    plt.hist(cluster_sizes, bins=bins)
    plt.xlabel("Tamanho do cluster (nº de células)")
    plt.ylabel("Frequência")
    plt.title("Histograma de tamanhos de cluster")
    plt.tight_layout()
    plt.show()


# --- NOVO: análise fácie a fácie com volume ---

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
        print(f"\n⚠️ O arquivo original estava bloqueado. Salvei como: {alt_path}")

def print_facies_metrics():
    facies_data = compute_facies_metrics()
    print("=== MÉTRICAS POR FÁCIES ===")
    for item in facies_data:
        fac = item["facies"]
        print(f"\nFácies {fac}:")
        print(f"  Células               : {item['cells']}")
        print(f"  % no modelo           : {item['fraction']*100:.2f}%")
        print(f"  Nº de clusters        : {item['n_clusters']}")
        print(f"  Maior cluster (células): {item['largest_size']}")
        print(f"  Fração conectada      : {item['connected_fraction']:.3f}")
        print(f"  Volume total          : {item['volume_total']:.3f}")
        print(f"  Volume maior cluster  : {item['volume_largest_cluster']:.3f}")
        print(f"  Espessura maior clust.: {item['thickness_largest_cluster']:.3f}")
        print(f"  Perc Xmin→Xmax        : {item['perc']['x_perc']}")
        print(f"  Perc Ymin→Ymax        : {item['perc']['y_perc']}")
        print(f"  Perc Topo→Base        : {item['perc']['z_perc']}")


def add_local_thickness_of_facies(facie_id: int, array_name="thickness_local"):
    """
    Calcula a espessura por coluna (x,y) do MAIOR CLUSTER da fácies indicada,
    considerando blocos verticais separados. Se o bloco tiver só 1 célula,
    usa uma espessura mínima aproximada (altura média da camada) pra não dar 0.
    """
    from scipy.ndimage import label, generate_binary_structure
    import numpy as np

    if isinstance(facie_id, (set, list, tuple)):
        if len(facie_id) != 1:
            raise ValueError("Passe apenas UMA fácies por vez para espessura local.")
        facie_id = list(facie_id)[0]
    facie_id = int(facie_id)

    # espessura mínima aproximada (altura média de uma camada)
    dz_default = (grid.bounds[5] - grid.bounds[4]) / nz

    # 1) máscara da fácies
    is_fac = (facies == facie_id).astype(np.uint8)

    # 2) rotula só essa fácies
    arr_xyz = is_fac.reshape((nx, ny, nz), order="F")
    arr_zyx = arr_xyz.transpose(2, 1, 0)
    struct = generate_binary_structure(3, 1)
    labeled_zyx, _ = label(arr_zyx, structure=struct)
    labeled_xyz = labeled_zyx.transpose(2, 1, 0)
    clusters_1d = labeled_xyz.reshape(-1, order="F")

    # 3) maior cluster
    counts = np.bincount(clusters_1d)
    counts[0] = 0
    largest_label = counts.argmax()
    if largest_label == 0:
        print(f"Nenhum cluster encontrado para a fácies {facie_id}")
        grid.cell_data[array_name] = np.zeros(facies.size)
        return

    # 4) z dos centros
    z_centers = grid.cell_centers().points[:, 2]
    z_xyz = z_centers.reshape((nx, ny, nz), order="F")

    # 5) máscara 3D do maior cluster
    cluster_xyz = (clusters_1d == largest_label).reshape((nx, ny, nz), order="F")

    # array de saída
    thickness_per_cell = np.zeros(facies.size, dtype=float)

    # 6) percorre cada coluna
    for ix in range(nx):
        for iy in range(ny):
            col_mask = cluster_xyz[ix, iy, :]
            if not col_mask.any():
                continue

            z_idx = np.where(col_mask)[0]
            z_vals = z_xyz[ix, iy, :][col_mask]

            # separa em blocos contíguos
            blocks = []
            current = [z_idx[0]]
            for prev, cur in zip(z_idx, z_idx[1:]):
                if cur == prev + 1:
                    current.append(cur)
                else:
                    blocks.append(current)
                    current = [cur]
            blocks.append(current)

            max_thick = 0.0
            for block in blocks:
                # z dos centros desse bloco
                mask_block = np.isin(z_idx, block)
                block_z = z_vals[mask_block]

                if block_z.size == 1:
                    dz = dz_default
                else:
                    dz = float(block_z.max() - block_z.min())

                if dz > max_thick:
                    max_thick = dz

            idx_1d = np.ravel_multi_index(
                (np.full(z_idx.size, ix), np.full(z_idx.size, iy), z_idx),
                dims=(nx, ny, nz),
                order="F"
            )
            thickness_per_cell[idx_1d] = max_thick

    grid.cell_data[array_name] = thickness_per_cell
    grid.active_scalars_name = array_name

    print(f"✅ Espessura local (com blocos verticais) da fácies {facie_id} salva em grid.cell_data['{array_name}']")


    import numpy as np
from load_data import grid, facies, nx, ny, nz

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


import numpy as np
from load_data import grid, facies, nx, ny, nz

def add_local_thickness_of_facies_all_clusters(facie_id, array_name="thickness_local"):
    """
    Calcula a espessura local (por coluna, com blocos verticais) para
    TODOS os clusters da fácies indicada e grava diretamente em
    grid.cell_data["thickness_local"] (ou no nome que você passar).
    Assim o visualize.py continua funcionando igual.
    """
    # aceita {23}, [23], (23,) ou 23
    if isinstance(facie_id, (set, list, tuple)):
        if len(facie_id) != 1:
            raise ValueError("Passe apenas UMA fácies por vez.")
        facie_id = list(facie_id)[0]
    facie_id = int(facie_id)

    from scipy.ndimage import label, generate_binary_structure

    # máscara só da fácies
    is_fac = (facies == facie_id).astype(np.uint8)

    # reshapes para rotular
    arr_xyz = is_fac.reshape((nx, ny, nz), order="F")
    arr_zyx = arr_xyz.transpose(2, 1, 0)

    struct = generate_binary_structure(3, 1)
    labeled_zyx, n_clusters = label(arr_zyx, structure=struct)
    labeled_xyz = labeled_zyx.transpose(2, 1, 0)
    clusters_1d = labeled_xyz.reshape(-1, order="F")

    if n_clusters == 0:
        print(f"Nenhum cluster encontrado para a fácies {facie_id}")
        grid.cell_data[array_name] = np.zeros(facies.size, dtype=float)
        grid.active_scalars_name = array_name
        return

    # z dos centros das células
    z_centers = grid.cell_centers().points[:, 2]
    z_xyz = z_centers.reshape((nx, ny, nz), order="F")

    # espessura mínima (1 camada)
    dz_default = (grid.bounds[5] - grid.bounds[4]) / nz

    # array final (mesmo tamanho do modelo)
    thickness_per_cell = np.zeros(facies.size, dtype=float)

    # percorre TODOS os clusters dessa fácies
    for clabel in range(1, n_clusters + 1):
        cluster_xyz = (clusters_1d == clabel).reshape((nx, ny, nz), order="F")
        if not cluster_xyz.any():
            continue

        for ix in range(nx):
            for iy in range(ny):
                col_mask = cluster_xyz[ix, iy, :]
                if not col_mask.any():
                    continue

                z_idx = np.where(col_mask)[0]
                z_vals = z_xyz[ix, iy, :][col_mask]

                # quebra em blocos contíguos em z
                blocks = []
                current = [z_idx[0]]
                for prev, cur in zip(z_idx, z_idx[1:]):
                    if cur == prev + 1:
                        current.append(cur)
                    else:
                        blocks.append(current)
                        current = [cur]
                blocks.append(current)

                # pega a maior espessura entre os blocos
                max_thick = 0.0
                for block in blocks:
                    mask_block = np.isin(z_idx, block)
                    block_z = z_vals[mask_block]
                    if block_z.size == 1:
                        dz = dz_default
                    else:
                        dz = float(block_z.max() - block_z.min())
                    if dz > max_thick:
                        max_thick = dz

                # grava esse valor nas células desse cluster nessa coluna
                idx_1d = np.ravel_multi_index(
                    (np.full(z_idx.size, ix), np.full(z_idx.size, iy), z_idx),
                    dims=(nx, ny, nz),
                    order="F"
                )
                thickness_per_cell[idx_1d] = max_thick

    # salva COM O MESMO NOME de antes
    grid.cell_data[array_name] = thickness_per_cell
    grid.active_scalars_name = array_name
    print(f"✅ Espessura local (todos os clusters) da fácies {facie_id} salva em grid.cell_data['{array_name}']")
