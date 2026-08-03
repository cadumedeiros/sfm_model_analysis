# analysis.py
import numpy as np
import pandas as pd
import pyvista as pv

from load_data import grid, facies, nx, ny, nz
from scipy.ndimage import label as nd_label, generate_binary_structure

# =============================================================================
# HELPERS DE GEOMETRIA
# =============================================================================
def _get_cell_volumes(target_grid=None, *, strict=False):
    """Retorna os volumes físicos absolutos das células.

    ``strict=True`` é usado pela extração científica para impedir que uma falha
    geométrica seja silenciosamente convertida em células de volume unitário.
    O fallback antigo é preservado quando ``strict=False`` para manter a UI
    retrocompatível.
    """
    g = target_grid if target_grid is not None else grid
    
    # 1) Tenta pegar cache
    for key in ("Volume", "Volume ", "volume", "Volume_", "CellVolume", "cell_volume"):
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
        if strict:
            raise ValueError("Não foi possível calcular os volumes das células do grid.")
        return np.ones(g.n_cells)

    # CORREÇÃO CRÍTICA: Volumes físicos devem ser positivos
    vol_arr = np.abs(vol_arr)
    
    g.cell_data["Volume"] = vol_arr
    return vol_arr

def _get_cell_z_coords(target_grid=None):
    g = target_grid if target_grid is not None else grid
    centers = g.cell_centers() if callable(getattr(g, "cell_centers", None)) else g.cell_centers
    return centers.points[:, 2]

def _get_cell_thickness(target_grid=None, *, strict=False):
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

    if strict and not np.any(np.isfinite(thick) & (thick > 0.0)):
        raise ValueError("Não foi possível obter espessuras positivas das células do grid.")

    g.cell_data["cell_thickness"] = thick
    return thick


def _normalize_facies_set(values):
    """Normaliza uma fácies ou coleção de fácies para ``set[int]``."""
    if values is None:
        return set()
    if isinstance(values, (int, np.integer)):
        return {int(values)}
    return {int(value) for value in values}


def _resolve_grid_shape(facies_array, grid_shape=None):
    """Valida e devolve a forma estrutural ``(nx, ny, nz)`` do modelo."""
    arr = np.asarray(facies_array).ravel()
    shape = tuple(int(v) for v in (grid_shape or (nx, ny, nz)))
    if len(shape) != 3 or any(v <= 0 for v in shape):
        raise ValueError(f"Forma de grid inválida: {shape!r}.")
    if int(np.prod(shape)) != arr.size:
        raise ValueError(
            f"Número de células ({arr.size}) incompatível com grid_shape={shape} "
            f"({int(np.prod(shape))} células)."
        )
    return shape


def get_active_cell_mask(
    facies_array,
    target_grid=None,
    *,
    inactive_facies=(0,),
    active_mask=None,
    require_positive_thickness=False,
):
    """Constrói uma máscara única de células ativas para todos os descritores.

    Por padrão, a fácies 0 é tratada como inativa/sem deposição. Uma máscara
    explícita pode ser fornecida para modelos que usam outra convenção.
    """
    fac = np.asarray(facies_array).ravel()
    n_cells = fac.size

    if active_mask is None:
        inactive = _normalize_facies_set(inactive_facies)
        mask = np.ones(n_cells, dtype=bool)
        if inactive:
            mask &= ~np.isin(fac, list(inactive))
    else:
        mask = np.asarray(active_mask, dtype=bool).ravel().copy()
        if mask.size != n_cells:
            raise ValueError("active_mask e facies_array possuem tamanhos diferentes.")

    volumes = _get_cell_volumes(target_grid, strict=True)
    if volumes.size != n_cells:
        raise ValueError("Volumes e facies_array possuem tamanhos diferentes.")
    mask &= np.isfinite(volumes) & (volumes > 0.0)

    if require_positive_thickness:
        thickness = _get_cell_thickness(target_grid, strict=True)
        if thickness.size != n_cells:
            raise ValueError("Espessuras e facies_array possuem tamanhos diferentes.")
        mask &= np.isfinite(thickness) & (thickness > 0.0)

    return mask


def summarize_2d_map(array_2d, prefix=None):
    """Resume um mapa 2D com estatísticas padronizadas para a tabela mestre."""
    arr = np.asarray(array_2d, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        stats = {
            "mean": np.nan,
            "std": np.nan,
            "p10": np.nan,
            "p50": np.nan,
            "p90": np.nan,
            "n_valid": 0,
        }
    else:
        stats = {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "p10": float(np.percentile(finite, 10)),
            "p50": float(np.percentile(finite, 50)),
            "p90": float(np.percentile(finite, 90)),
            "n_valid": int(finite.size),
        }
    if prefix is None:
        return stats
    return {f"{prefix}_{name}": value for name, value in stats.items()}

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

def facies_distribution_array(
    facies_array,
    target_grid=None,
    *,
    inactive_facies=(0,),
    active_mask=None,
):
    arr = np.asarray(facies_array).astype(int).ravel()
    if arr.size == 0:
        return {}, 0

    volumes = _get_cell_volumes(target_grid, strict=True)
    z_coords = _get_cell_z_coords(target_grid)
    active = get_active_cell_mask(
        arr,
        target_grid,
        inactive_facies=inactive_facies,
        active_mask=active_mask,
    )
    total = int(np.sum(active))
    total_volume = float(np.sum(volumes[active])) if total else 0.0
    vals = np.unique(arr[active]) if total else np.array([], dtype=int)
    stats = {}

    for fac in vals:
        mask = active & (arr == fac)
        s = _calc_stats_for_subset(mask, volumes, z_coords)
        s["cell_fraction"] = s["cells"] / total if total else 0.0
        s["volume_fraction"] = s["volume"] / total_volume if total_volume > 0.0 else 0.0
        s["fraction"] = s["volume_fraction"]
        stats[int(fac)] = s

    return stats, total

def reservoir_facies_distribution_array(
    facies_array,
    reservoir_facies,
    target_grid=None,
    *,
    inactive_facies=(0,),
    active_mask=None,
):
    arr = np.asarray(facies_array).astype(int).ravel()
    fac_set = _normalize_facies_set(reservoir_facies)
    active = get_active_cell_mask(
        arr,
        target_grid,
        inactive_facies=inactive_facies,
        active_mask=active_mask,
    )
    mask_res = active & np.isin(arr, list(fac_set))
    res_total = int(mask_res.sum())

    if res_total == 0:
        return {}, 0

    volumes = _get_cell_volumes(target_grid, strict=True)
    z_coords = _get_cell_z_coords(target_grid)

    arr_res = arr[mask_res]
    vol_res = volumes[mask_res]
    z_res = z_coords[mask_res]
    total_volume = float(np.sum(vol_res))
    
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
            "cell_fraction": count / res_total,
            "volume_fraction": vol / total_volume if total_volume > 0.0 else 0.0,
            "fraction": vol / total_volume if total_volume > 0.0 else 0.0,
            "volume": vol,
            "thickness_gross": thick
        }

    return stats, res_total

def compute_target_body_descriptors(
    facies_array,
    target_facies,
    target_grid=None,
    *,
    grid_shape=None,
    inactive_facies=(0,),
    active_mask=None,
    min_body_volume=0.0,
    connectivity=1,
    return_labels=False,
):
    """Calcula conectividade, fragmentação e percolação do alvo em 3D.

    A conectividade principal é volumétrica. As versões por número de células
    são mantidas para comparação com resultados antigos.
    """
    arr = np.asarray(facies_array, dtype=int).ravel()
    shape = _resolve_grid_shape(arr, grid_shape)
    target_set = _normalize_facies_set(target_facies)
    volumes = _get_cell_volumes(target_grid, strict=True)
    active = get_active_cell_mask(
        arr,
        target_grid,
        inactive_facies=inactive_facies,
        active_mask=active_mask,
    )
    target_mask = active & np.isin(arr, list(target_set)) if target_set else np.zeros(arr.size, dtype=bool)
    target_cells = int(np.sum(target_mask))
    target_volume = float(np.sum(volumes[target_mask])) if target_cells else 0.0

    empty_perc = {
        "x_perc": False,
        "x_clusters": set(),
        "y_perc": False,
        "y_clusters": set(),
        "z_perc": False,
        "z_clusters": set(),
    }
    if target_cells == 0:
        result = {
            "target_cells": 0,
            "target_volume": 0.0,
            "n_bodies": 0,
            "largest_body_label": 0,
            "largest_body_cells": 0,
            "largest_body_volume": 0.0,
            "connected_cell_fraction": 0.0,
            "connected_volume_fraction": 0.0,
            "effective_body_count": 0.0,
            "body_volume_p50": 0.0,
            "body_volume_p90": 0.0,
            "body_table": pd.DataFrame(columns=["body_label", "cells", "volume"]),
            "percolation": empty_perc,
        }
        if return_labels:
            result["labels"] = np.zeros(arr.size, dtype=np.int32)
        return result

    connectivity = int(connectivity)
    if connectivity not in (1, 2, 3):
        raise ValueError("connectivity deve ser 1 (faces), 2 (arestas) ou 3 (vértices).")
    structure = generate_binary_structure(3, connectivity)
    target_3d = target_mask.reshape(shape, order="F")
    labels_3d, _ = nd_label(target_3d, structure=structure)
    labels = labels_3d.reshape(-1, order="F").astype(np.int32)

    rows = []
    for body_label in np.unique(labels[labels > 0]):
        body_mask = labels == int(body_label)
        rows.append({
            "body_label": int(body_label),
            "cells": int(np.sum(body_mask)),
            "volume": float(np.sum(volumes[body_mask])),
        })
    body_table = pd.DataFrame(rows)
    if body_table.empty:
        kept = body_table
    else:
        kept = body_table.loc[body_table["volume"] >= float(min_body_volume)].copy()

    if kept.empty:
        largest_label = 0
        largest_cells = 0
        largest_volume = 0.0
        kept_labels = set()
    else:
        largest_row = kept.loc[kept["volume"].idxmax()]
        largest_label = int(largest_row["body_label"])
        largest_cells = int(largest_row["cells"])
        largest_volume = float(largest_row["volume"])
        kept_labels = set(int(v) for v in kept["body_label"].tolist())

    def _boundary_labels(a, b):
        left = set(int(v) for v in np.unique(a) if int(v) in kept_labels)
        right = set(int(v) for v in np.unique(b) if int(v) in kept_labels)
        return left.intersection(right)

    x_common = _boundary_labels(labels_3d[0, :, :], labels_3d[-1, :, :])
    y_common = _boundary_labels(labels_3d[:, 0, :], labels_3d[:, -1, :])
    z_common = _boundary_labels(labels_3d[:, :, 0], labels_3d[:, :, -1])
    perc = {
        "x_perc": bool(x_common),
        "x_clusters": x_common,
        "y_perc": bool(y_common),
        "y_clusters": y_common,
        "z_perc": bool(z_common),
        "z_clusters": z_common,
    }

    all_body_volumes = body_table["volume"].to_numpy(dtype=float)
    if target_volume > 0.0 and all_body_volumes.size:
        shares = all_body_volumes / target_volume
        denom = float(np.sum(shares ** 2))
        effective_n = (1.0 / denom) if denom > 0.0 else 0.0
    else:
        effective_n = 0.0

    result = {
        "target_cells": target_cells,
        "target_volume": target_volume,
        "n_bodies": int(len(kept)),
        "largest_body_label": largest_label,
        "largest_body_cells": largest_cells,
        "largest_body_volume": largest_volume,
        "connected_cell_fraction": (largest_cells / target_cells) if target_cells else 0.0,
        "connected_volume_fraction": (largest_volume / target_volume) if target_volume > 0.0 else 0.0,
        "effective_body_count": float(effective_n),
        "body_volume_p50": float(np.percentile(all_body_volumes, 50)) if all_body_volumes.size else 0.0,
        "body_volume_p90": float(np.percentile(all_body_volumes, 90)) if all_body_volumes.size else 0.0,
        "body_table": body_table,
        "percolation": perc,
    }
    if return_labels:
        result["labels"] = labels
    return result


def compute_global_metrics_for_array(
    facies_array,
    reservoir_facies,
    target_grid=None,
    *,
    grid_shape=None,
    inactive_facies=(0,),
    active_mask=None,
    min_body_volume=0.0,
    connectivity=1,
):
    """Calcula métricas globais consistentes para um modelo.

    Os nomes ``ntg``, ``res_cells`` e ``connected_fraction`` são aliases de
    compatibilidade com a interface. Os novos consumidores devem usar as chaves
    ``target_*`` e ``connected_volume_fraction``.
    """
    arr = np.asarray(facies_array, dtype=int).ravel()
    shape = _resolve_grid_shape(arr, grid_shape)
    volumes = _get_cell_volumes(target_grid, strict=True)
    active = get_active_cell_mask(
        arr,
        target_grid,
        inactive_facies=inactive_facies,
        active_mask=active_mask,
    )
    active_cells = int(np.sum(active))
    active_volume = float(np.sum(volumes[active])) if active_cells else 0.0
    all_grid_volume = float(np.sum(volumes[np.isfinite(volumes) & (volumes > 0.0)]))

    bodies = compute_target_body_descriptors(
        arr,
        reservoir_facies,
        target_grid,
        grid_shape=shape,
        inactive_facies=inactive_facies,
        active_mask=active,
        min_body_volume=min_body_volume,
        connectivity=connectivity,
    )
    target_cells = int(bodies["target_cells"])
    target_volume = float(bodies["target_volume"])
    target_cell_fraction = (target_cells / active_cells) if active_cells else 0.0
    target_volume_fraction = (target_volume / active_volume) if active_volume > 0.0 else 0.0

    metrics = {
        "total_cells": int(arr.size),
        "active_cells": active_cells,
        "target_cells": target_cells,
        "target_cell_fraction": float(target_cell_fraction),
        "target_volume_fraction": float(target_volume_fraction),
        "active_grid_volume": active_volume,
        "all_grid_volume": all_grid_volume,
        "target_volume": target_volume,
        "n_bodies": int(bodies["n_bodies"]),
        "largest_body_label": int(bodies["largest_body_label"]),
        "largest_body_cells": int(bodies["largest_body_cells"]),
        "largest_body_volume": float(bodies["largest_body_volume"]),
        "connected_cell_fraction": float(bodies["connected_cell_fraction"]),
        "connected_volume_fraction": float(bodies["connected_volume_fraction"]),
        "effective_body_count": float(bodies["effective_body_count"]),
        "body_volume_p50": float(bodies["body_volume_p50"]),
        "body_volume_p90": float(bodies["body_volume_p90"]),
        # aliases legados usados por window.py
        "res_cells": target_cells,
        "ntg": float(target_volume_fraction),
        "n_clusters": int(bodies["n_bodies"]),
        "largest_label": int(bodies["largest_body_label"]),
        "largest_size": int(bodies["largest_body_cells"]),
        "connected_fraction": float(bodies["connected_volume_fraction"]),
        "grid_volume": active_volume,
        "reservoir_volume": target_volume,
    }
    return metrics, bodies["percolation"]

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


def compute_vertical_descriptor_maps(
    facies_array,
    target_facies,
    target_grid=None,
    *,
    grid_shape=None,
    inactive_facies=(0,),
    active_mask=None,
    use_filtered=False,
    thin_lamination_threshold=0.30,
):
    """Calcula diretamente os mapas 2D selecionados para a tabela mestre.

    Diferentemente do fluxo de visualização, os valores não são replicados nas
    células 3D. Isso evita que zeros de células não alvo contaminem os resumos
    estatísticos da tabela mestre.
    """
    arr = np.asarray(facies_array, dtype=int).ravel()
    shape = _resolve_grid_shape(arr, grid_shape)
    nx_m, ny_m, nz_m = shape
    target_set = _normalize_facies_set(target_facies)
    thickness = _get_cell_thickness(target_grid, strict=True)
    active = get_active_cell_mask(
        arr,
        target_grid,
        inactive_facies=inactive_facies,
        active_mask=active_mask,
        require_positive_thickness=True,
    )

    fac_3d = arr.reshape(shape, order="F")
    th_3d = np.asarray(thickness, dtype=float).reshape(shape, order="F")
    active_3d = active.reshape(shape, order="F")
    map_names = (
        "Tcolumn",
        "Ttarget",
        "Tenv",
        "target_fraction_col",
        "target_fraction_env",
        "n_packages",
        "Tpack_max",
        "Cdom",
        "gap_fraction_env",
        "Tgap_max",
    )
    maps = {name: np.full((nx_m, ny_m), np.nan, dtype=float) for name in map_names}

    for ix in range(nx_m):
        for iy in range(ny_m):
            valid = active_3d[ix, iy, :]
            if not np.any(valid):
                continue

            fac_col = fac_3d[ix, iy, valid]
            th_col = th_3d[ix, iy, valid]
            flags = np.isin(fac_col, list(target_set)).astype(np.int8) if target_set else np.zeros(fac_col.size, dtype=np.int8)

            if use_filtered:
                runs = _build_binary_runs(flags, th_col)
                runs = _merge_thin_binary_runs(runs, thin_lamination_threshold)
                flags_use, th_use = _expand_binary_runs_to_cells(runs)
            else:
                flags_use, th_use = flags, th_col

            metrics = _compute_vertical_metrics_from_binary_cells(flags_use, th_use)
            maps["Tcolumn"][ix, iy] = float(np.sum(th_use))
            maps["Ttarget"][ix, iy] = float(metrics["Ttot_reservoir"])
            maps["Tenv"][ix, iy] = float(metrics["Tenv_reservoir"])
            maps["target_fraction_col"][ix, iy] = float(metrics["NTG_col_reservoir"])
            maps["target_fraction_env"][ix, iy] = float(metrics["NTG_env_reservoir"])
            maps["n_packages"][ix, iy] = float(metrics["n_packages_reservoir"])
            maps["Tpack_max"][ix, iy] = float(metrics["Tpack_max_reservoir"])
            maps["Cdom"][ix, iy] = float(metrics["Cdom_reservoir"])
            maps["gap_fraction_env"][ix, iy] = float(metrics["Gap_env_reservoir"])
            maps["Tgap_max"][ix, iy] = float(metrics["Tgap_max_reservoir"])

    return maps


def compute_vpc(
    facies_array,
    target_grid=None,
    *,
    grid_shape=None,
    inactive_facies=(0,),
    active_mask=None,
    facies_ids=None,
    model_id=None,
    model_name=None,
):
    """Calcula a Vertical Proportion Curve por proporção areal de camada.

    A área horizontal é estimada por ``volume / espessura estratigráfica``.
    O resultado longo contém todas as fácies em todas as camadas, inclusive
    proporções zero, o que simplifica comparações entre modelos.
    """
    arr = np.asarray(facies_array, dtype=int).ravel()
    shape = _resolve_grid_shape(arr, grid_shape)
    nx_m, ny_m, nz_m = shape
    volumes = _get_cell_volumes(target_grid, strict=True)
    thickness = _get_cell_thickness(target_grid, strict=True)
    active = get_active_cell_mask(
        arr,
        target_grid,
        inactive_facies=inactive_facies,
        active_mask=active_mask,
        require_positive_thickness=True,
    )

    area = np.zeros(arr.size, dtype=float)
    valid_area = active & np.isfinite(volumes) & np.isfinite(thickness) & (thickness > 0.0)
    area[valid_area] = volumes[valid_area] / thickness[valid_area]
    area[~np.isfinite(area) | (area <= 0.0)] = 0.0

    arr_3d = arr.reshape(shape, order="F")
    area_3d = area.reshape(shape, order="F")
    active_3d = active.reshape(shape, order="F")
    z_3d = np.asarray(_get_cell_z_coords(target_grid), dtype=float).reshape(shape, order="F")

    if facies_ids is None:
        ids = sorted(int(v) for v in np.unique(arr[active]))
    else:
        ids = sorted(_normalize_facies_set(facies_ids))

    records = []
    for layer_k in range(nz_m):
        layer_active = active_3d[:, :, layer_k]
        layer_area = area_3d[:, :, layer_k]
        weights = np.where(layer_active, layer_area, 0.0)
        active_area = float(np.sum(weights))
        active_cells = int(np.sum(layer_active))
        z_layer = z_3d[:, :, layer_k]
        z_valid = layer_active & np.isfinite(z_layer) & (weights > 0.0)
        z_weight = float(np.sum(weights[z_valid]))
        if z_weight > 0.0:
            mean_z = float(np.sum(z_layer[z_valid] * weights[z_valid]) / z_weight)
        else:
            mean_z = np.nan
        coordinate = float(layer_k / (nz_m - 1)) if nz_m > 1 else 0.0

        for facies_id in ids:
            facies_area = float(np.sum(weights[arr_3d[:, :, layer_k] == int(facies_id)]))
            proportion = (facies_area / active_area) if active_area > 0.0 else np.nan
            records.append({
                "model_id": model_id,
                "model_name": model_name,
                "layer_k": int(layer_k),
                "stratigraphic_coordinate": coordinate,
                "layer_mean_z": mean_z,
                "facies": int(facies_id),
                "area_proportion": float(proportion) if np.isfinite(proportion) else np.nan,
                "facies_area": facies_area,
                "active_area": active_area,
                "active_cells": active_cells,
            })

    return pd.DataFrame.from_records(records)


def compute_vpc_distance(vpc_df, reference_vpc_df):
    """Calcula a distância RMSE entre duas VPCs no espaço camada-fácies."""
    required = {"layer_k", "facies", "area_proportion"}
    if vpc_df is None or reference_vpc_df is None:
        return np.nan
    if not required.issubset(vpc_df.columns) or not required.issubset(reference_vpc_df.columns):
        raise ValueError("As tabelas VPC não possuem as colunas obrigatórias.")

    left = vpc_df.groupby(["layer_k", "facies"], as_index=False)["area_proportion"].mean()
    right = reference_vpc_df.groupby(["layer_k", "facies"], as_index=False)["area_proportion"].mean()
    merged = left.merge(right, on=["layer_k", "facies"], how="outer", suffixes=("_model", "_reference"))
    a = merged["area_proportion_model"].fillna(0.0).to_numpy(dtype=float)
    b = merged["area_proportion_reference"].fillna(0.0).to_numpy(dtype=float)
    if a.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((a - b) ** 2)))

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

def generate_detailed_metrics_df(
    facies_array,
    target_grid=None,
    *,
    grid_shape=None,
    inactive_facies=(0,),
    active_mask=None,
    min_body_volume=0.0,
    connectivity=1,
):
    """Gera a tabela por fácies usando a mesma convenção das métricas globais."""
    arr = np.asarray(facies_array, dtype=int).ravel()
    shape = _resolve_grid_shape(arr, grid_shape)
    volumes = _get_cell_volumes(target_grid, strict=True)
    z_vals = _get_cell_z_coords(target_grid)
    active = get_active_cell_mask(
        arr,
        target_grid,
        inactive_facies=inactive_facies,
        active_mask=active_mask,
    )
    active_cells = int(np.sum(active))
    active_volume = float(np.sum(volumes[active])) if active_cells else 0.0
    rows = []

    for facies_id in sorted(int(v) for v in np.unique(arr[active])):
        mask = active & (arr == facies_id)
        count = int(np.sum(mask))
        volume_total = float(np.sum(volumes[mask]))
        body = compute_target_body_descriptors(
            arr,
            {facies_id},
            target_grid,
            grid_shape=shape,
            inactive_facies=inactive_facies,
            active_mask=active,
            min_body_volume=min_body_volume,
            connectivity=connectivity,
            return_labels=True,
        )

        largest_label = int(body["largest_body_label"])
        labels = body.get("labels")
        vertical_extent = 0.0
        if labels is not None and largest_label > 0:
            largest_mask = labels == largest_label
            z_body = np.asarray(z_vals, dtype=float)[largest_mask]
            if z_body.size:
                vertical_extent = float(np.nanmax(z_body) - np.nanmin(z_body))

        perc = body["percolation"]
        cell_fraction = (count / active_cells) if active_cells else 0.0
        volume_fraction = (volume_total / active_volume) if active_volume > 0.0 else 0.0
        rows.append({
            "facies": facies_id,
            "cells": count,
            "cell_fraction": float(cell_fraction),
            "volume_fraction": float(volume_fraction),
            "fraction": float(volume_fraction),  # alias legado
            "n_clusters": int(body["n_bodies"]),
            "largest_size": int(body["largest_body_cells"]),
            "connected_cell_fraction": float(body["connected_cell_fraction"]),
            "connected_volume_fraction": float(body["connected_volume_fraction"]),
            "connected_fraction": float(body["connected_volume_fraction"]),  # alias legado
            "effective_body_count": float(body["effective_body_count"]),
            "volume_total": volume_total,
            "volume_largest_cluster": float(body["largest_body_volume"]),
            "largest_body_vertical_extent": vertical_extent,
            "thickness_largest_cluster": vertical_extent,  # alias legado; não é espessura estratigráfica
            "Perc_X": bool(perc["x_perc"]),
            "Perc_Y": bool(perc["y_perc"]),
            "Perc_Z": bool(perc["z_perc"]),
        })

    return pd.DataFrame(rows)

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
