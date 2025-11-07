# local_windows.py
import numpy as np
from load_data import grid, facies, nx, ny, nz

def compute_local_ntg(reservoir_mask: np.ndarray, window=(5, 5, 3)):
    """
    reservoir_mask: array 1D (mesmo tamanho do facies) com 0/1
    window: (wx, wy, wz) em número de células
    """
    wx, wy, wz = window

    arr = reservoir_mask.reshape((nx, ny, nz), order="F")  # (x,y,z)
    ntg_local = np.zeros_like(arr, dtype=float)

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                x0 = max(ix - wx // 2, 0)
                x1 = min(ix + wx // 2 + 1, nx)
                y0 = max(iy - wy // 2, 0)
                y1 = min(iy + wy // 2 + 1, ny)
                z0 = max(iz - wz // 2, 0)
                z1 = min(iz + wz // 2 + 1, nz)

                block = arr[x0:x1, y0:y1, z0:z1]
                total = block.size
                resv = block.sum()
                ntg_local[ix, iy, iz] = resv / total if total else 0.0

    ntg_local_1d = ntg_local.reshape(-1, order="F")
    grid.cell_data["NTG_local"] = ntg_local_1d
    return ntg_local_1d
