# load_data.py
import pyvista as pv
import numpy as np
from config import ANCHOR_Y, APPLY_REFLECTION

# Z Settings (Originais - sem alteração)
Z_FACTOR = 1.0     
Z_SHIFT  = 0.0      
# --------------------------------------

grdecl_path = "grids/_BENCHMARK_MCHAVES_Inferior_2025-1-Tck123_SIM_BaseModel_.grdecl"

print(f"Lendo Grid: {grdecl_path}...")
grid = pv.read_grdecl(grdecl_path)
print(f"Bounds Originais: {grid.bounds}")

# --- APLICAÇÃO DA TRANSFORMAÇÃO ---
if APPLY_REFLECTION:
    grid.points[:, 1] = (2 * ANCHOR_Y) - grid.points[:, 1]
    print(f">>> Grid Refletido Y (Pivô {ANCHOR_Y})")

# Ajuste de Z (apenas se precisar no futuro, por padrão fator=1 não faz nada)
if Z_FACTOR != 1.0 or Z_SHIFT != 0.0:
    grid.points[:, 2] = (grid.points[:, 2] * Z_FACTOR) + Z_SHIFT

print(f"Bounds Finais:    {grid.bounds}")

def load_grid_from_grdecl(path, facies_keyword="Facies"):
    """
    Carrega um GRDECL COMPLETO (geometria + facies), aplicando as mesmas
    transformações usadas no grid base (reflexão Y / ajuste de Z).

    Retorna:
        grid (pyvista.UnstructuredGrid), facies_1d (np.ndarray int)
    """
    # 1) Geometria (isso traz ZCORN/COORD -> pontos corretos)
    g = pv.read_grdecl(path)

    # 2) Aplica as MESMAS transformações do base (pra manter o mesmo referencial)
    if APPLY_REFLECTION:
        g.points[:, 1] = (2 * ANCHOR_Y) - g.points[:, 1]

    if Z_FACTOR != 1.0 or Z_SHIFT != 0.0:
        g.points[:, 2] = (g.points[:, 2] * Z_FACTOR) + Z_SHIFT

    # 3) Lê facies e reorganiza igual ao base
    fac = read_keyword_array(path, facies_keyword)
    nx2, ny2, nz2 = read_specgrid(path)  # retorna iterável -> desempacota ok

    fac_3d = fac.reshape((nx2, ny2, nz2), order="F")
    fac_3d = fac_3d[:, :, ::-1]  # inverte K como você já faz
    fac_1d = fac_3d.reshape(-1, order="F").astype(int)

    # 4) Atribui no grid
    g.cell_data["Facies"] = fac_1d

    return g, fac_1d


# --- FUNÇÕES DE LEITURA DE PROPRIEDADES (Mantidas Iguais) ---
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
                    if before_slash: vals.extend(before_slash.split())
                    break
                else:
                    if line_strip: vals.extend(line_strip.split())
    return np.array([int(float(v)) for v in vals if v], dtype=int)

def read_specgrid(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip().upper().startswith("SPECGRID"):
                nums = []
                while True:
                    l2 = next(f).strip()
                    if "/" in l2: 
                        nums.extend(l2.split("/")[0].split())
                        break
                    nums.extend(l2.split())
                return map(int, nums[:3])
    return 1,1,1

facies = read_keyword_array(grdecl_path, "Facies")
nx, ny, nz = read_specgrid(grdecl_path)

# Se o espelhamento geométrico inverteu o Y físico, 
# o array de dados (facies) também precisa acompanhar a inversão no eixo J?
# Geralmente o PyVista mapeia célula a célula pela ordem dos pontos.
# Vamos manter o padrão. Se as fácies ficarem "espelhadas" visualmente (argila onde devia ter areia),
# precisaremos inverter o eixo 1 aqui: facies_3d[:, ::-1, ::-1]
facies_3d = facies.reshape((nx, ny, nz), order="F")   
facies_3d = facies_3d[:, :, ::-1] # Inverte Z (K)
facies = facies_3d.reshape(-1, order="F")             

grid.cell_data["Facies"] = facies

def load_facies_from_grdecl(path):
    fac = read_keyword_array(path, "Facies")
    nx2, ny2, nz2 = read_specgrid(path)
    fac_3d = fac.reshape((nx2, ny2, nz2), order="F")
    fac_3d = fac_3d[:, :, ::-1]
    fac_1d = fac_3d.reshape(-1, order="F")
    return fac_1d