import numpy as np
from scipy.ndimage import label, generate_binary_structure

from load_data import grid, facies, nx, ny, nz  # agora temos o shape
from config import RESERVOIR_FACIES

# quais fácies são reservatório

is_res_1d = np.isin(facies, list(RESERVOIR_FACIES)).astype(np.uint8)

# 1) reshape do 1D -> (nx, ny, nz) no mesmo empacotamento do Eclipse
arr_xyz = is_res_1d.reshape((nx, ny, nz), order="F")  # (x, y, z)
# 2) SciPy costuma trabalhar como (z, y, x), então vamos transpor
is_res_3d = arr_xyz.transpose(2, 1, 0)  # (nz, ny, nx)

# estrutura 3D de 6 vizinhos
structure = generate_binary_structure(3, 1)

labeled, n_clusters = label(is_res_3d, structure=structure)

# conta tamanho dos clusters (labeled é 3D)
counts = np.bincount(labeled.ravel())
counts[0] = 0
largest_label = counts.argmax()
largest_size = counts[largest_label]

total_res = is_res_1d.sum()
connected_fraction = largest_size / total_res

print(f"NTG = {total_res / is_res_1d.size:.3f}")
print(f"Clusters de reservatório: {n_clusters}")
print(f"Maior cluster: {largest_size} células")
print(f"Fração conectada: {connected_fraction:.3f}")

# se quiser voltar pro grid do pyvista:
# desfaz o transpose e o reshape
largest_mask_3d = (labeled == largest_label)
largest_mask_xyz = largest_mask_3d.transpose(2, 1, 0)  # volta pra (x, y, z)
largest_mask_1d = largest_mask_xyz.reshape(-1, order="F")
grid.cell_data["LargestCluster"] = largest_mask_1d.astype(np.uint8)

# 👉 salva TODOS os rótulos no grid
# volta pro formato (x, y, z) e depois 1D
clusters_xyz = labeled.transpose(2, 1, 0)
clusters_1d = clusters_xyz.reshape(-1, order="F")
grid.cell_data["Clusters"] = clusters_1d.astype(np.int32)
