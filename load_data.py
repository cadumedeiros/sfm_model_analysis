# load_data.py
import pyvista as pv
import numpy as np
import config as cfg

grdecl_path = "assets/grid_mchaves.grdecl"

grid = pv.read_grdecl(grdecl_path)

def read_keyword_array(path, keyword="Facies"):
    vals = []
    inside = False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_strip = line.strip()
            if not inside and line_strip.upper().startswith(keyword.upper()):
                inside = True
                line_strip = line_strip[len(keyword):].strip()
            if inside:
                if "/" in line_strip:
                    before_slash = line_strip.split("/")[0]
                    if before_slash:
                        vals.extend(before_slash.split())
                    break
                else:
                    if line_strip:
                        vals.extend(line_strip.split())
    return np.array(list(map(int, vals)))

def read_specgrid(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ls = line.strip()
            if ls.upper().startswith("SPECGRID"):
                # linha tipo: SPECGRID
                #              50 40 10  1  F /
                nums = []
                while True:
                    l2 = next(f).strip()
                    if "/" in l2:
                        l2 = l2.split("/")[0]
                        nums.extend(l2.split())
                        break
                    else:
                        nums.extend(l2.split())
                nx, ny, nz = map(int, nums[:3])
                return nx, ny, nz
    raise RuntimeError("SPECGRID não encontrado no GRDECL")

facies = read_keyword_array(grdecl_path, "Facies")
nx, ny, nz = read_specgrid(grdecl_path)

n_cells = grid.n_cells
assert len(facies) == grid.n_cells

facies_3d = facies.reshape((nx, ny, nz), order="F")   # (x, y, z) no empacotamento Eclipse
facies_3d = facies_3d[:, :, ::-1]                     # inverte só o eixo z
facies = facies_3d.reshape(-1, order="F")             # volta pra 1D

grid.cell_data["Facies"] = facies

