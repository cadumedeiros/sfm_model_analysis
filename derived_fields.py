# derived_fields.py
import numpy as np
from scipy.ndimage import label, generate_binary_structure
from load_data import grid, facies, nx, ny, nz

def ensure_reservoir(reservoir_facies):
    if "Reservoir" in grid.cell_data:
        return grid.cell_data["Reservoir"]
    is_res = np.isin(facies, list(reservoir_facies)).astype(np.int8)
    grid.cell_data["Reservoir"] = is_res
    return is_res

def ensure_clusters(reservoir_facies):
    # se já tem, só retorna
    if "Clusters" in grid.cell_data and "LargestCluster" in grid.cell_data:
        return grid.cell_data["Clusters"], grid.cell_data["LargestCluster"]

    is_res = ensure_reservoir(reservoir_facies)

    # 1D -> 3D
    arr_xyz = is_res.reshape((nx, ny, nz), order="F")
    is_res_3d = arr_xyz.transpose(2, 1, 0)  # (nz, ny, nx)

    structure = generate_binary_structure(3, 1)
    labeled_3d, n_clusters = label(is_res_3d, structure=structure)

    # volta pra 1D
    clusters_xyz = labeled_3d.transpose(2, 1, 0)
    clusters_1d = clusters_xyz.reshape(-1, order="F")
    grid.cell_data["Clusters"] = clusters_1d.astype(np.int32)

    # maior cluster
    counts = np.bincount(labeled_3d.ravel())
    counts[0] = 0
    largest_label = counts.argmax()

    largest_mask_xyz = (labeled_3d == largest_label).transpose(2, 1, 0)
    largest_mask_1d = largest_mask_xyz.reshape(-1, order="F").astype(np.uint8)
    grid.cell_data["LargestCluster"] = largest_mask_1d

    return grid.cell_data["Clusters"], grid.cell_data["LargestCluster"]
