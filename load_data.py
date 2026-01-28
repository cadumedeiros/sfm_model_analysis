# load_data.py
"""
Centraliza o carregamento de GRDECL e mantém compatibilidade com o resto do projeto.

IMPORTANTE:
- Muitos arquivos importam diretamente: from load_data import grid, facies, nx, ny, nz
  Então este módulo inicializa esses globais no import.
- Facies e StratigraphicThickness são lidos do texto do GRDECL (pv.read_grdecl nem sempre popula cell_data).
- A inversão em Z (K) é controlada por flip_k. O segredo é usar SEMPRE o mesmo flip_k
  para base e para modelos carregados pela UI.
"""

from __future__ import annotations

import re
from typing import Tuple

import numpy as np
import pyvista as pv

from config import ANCHOR_Y, APPLY_REFLECTION


# =========================
# Defaults do projeto
# =========================
DEFAULT_GRDECL_PATH = "grids/_BENCHMARK_MCHAVES_Inferior_2025-1-Tck123_SIM_BaseModel_.grdecl"

# Se suas fácies aparecem invertidas em Z, este é o knob principal.
# Você relatou que: load_grid_from_grdecl(..., flip_k=True) "deu certo"
FLIP_K_DEFAULT = True

# Debug prints
VERBOSE_DEFAULT = True


# =========================
# GRDECL parsing helpers
# =========================

_RE_REPEAT = re.compile(
    r"^(?P<n>\d+)\*(?P<val>[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)$"
)
_RE_KEYWORD_LINE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\s*/?\s*$")

def _tokenize_keyword_block(lines: list[str]) -> list[str]:
    """Coleta tokens após uma keyword até o terminador '/'."""
    tokens: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue

        # corta no terminador
        if "/" in s:
            s = s.split("/", 1)[0].strip()

        if s:
            tokens.extend(s.split())

        if "/" in line:
            break

    return tokens

def _expand_repeats(tokens: list[str]) -> list[str]:
    """Expande tokens do tipo '10*3.5' para ['3.5', ...] (10x)."""
    out: list[str] = []
    for t in tokens:
        m = _RE_REPEAT.match(t)
        if m:
            n = int(m.group("n"))
            val = m.group("val")
            out.extend([val] * n)
        else:
            out.append(t)
    return out

def list_grdecl_keywords(grdecl_path: str) -> list[str]:
    """Lista keywords do GRDECL (apenas linhas com keyword isolada)."""
    keys: list[str] = []
    seen: set[str] = set()

    with open(grdecl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("--"):
                continue

            if _RE_KEYWORD_LINE.match(s):
                key = s.replace("/", "").strip()
                if not key:
                    continue
                key_norm = key.upper()
                if key_norm not in seen:
                    keys.append(key)
                    seen.add(key_norm)

                # Skip block until '/' (if not on same line)
                if "/" in line:
                    continue
                for line in f:
                    if "/" in line:
                        break

    return keys

def discover_numeric_keyword_arrays(
    grdecl_path: str,
    nx: int,
    ny: int,
    nz: int,
    *,
    exclude: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Descobre keywords numericas com tamanho nx*ny*nz e devolve seus arrays 1D."""
    expected = nx * ny * nz
    exclude_norm = {k.upper() for k in (exclude or set())}

    out: dict[str, np.ndarray] = {}
    for key in list_grdecl_keywords(grdecl_path):
        if key.upper() in exclude_norm:
            continue
        try:
            arr = read_keyword_array(grdecl_path, key, dtype=float)
        except Exception:
            continue
        if arr.size != expected:
            continue
        out[key] = arr

    return out

def read_specgrid(grdecl_path: str) -> Tuple[int, int, int]:
    """Lê (nx, ny, nz) a partir de SPECGRID."""
    with open(grdecl_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    m = re.search(r"\bSPECGRID\b", text, flags=re.IGNORECASE)
    if not m:
        raise ValueError("SPECGRID não encontrado no GRDECL.")

    after = text[m.end():].splitlines()
    tokens = _tokenize_keyword_block(after)

    if len(tokens) < 3:
        raise ValueError("SPECGRID inválido (faltam nx, ny, nz).")

    nx, ny, nz = map(int, tokens[:3])
    return nx, ny, nz

def read_keyword_array(grdecl_path: str, keyword: str, dtype=float) -> np.ndarray:
    """Lê um array numérico de uma keyword (suporta 'n*valor')."""
    target = keyword.upper()
    with open(grdecl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("--"):
                continue

            if _RE_KEYWORD_LINE.match(s):
                key = s.replace("/", "").strip()
                if not key:
                    continue

                if key.upper() == target:
                    if "/" in line:
                        tokens = []
                    else:
                        block_lines: list[str] = []
                        for line in f:
                            block_lines.append(line)
                            if "/" in line:
                                break
                        tokens = _tokenize_keyword_block(block_lines)
                    tokens = _expand_repeats(tokens)
                    break
                else:
                    # Skip this block to avoid false keyword matches inside data
                    if "/" in line:
                        continue
                    for line in f:
                        if "/" in line:
                            break
        else:
            raise KeyError(f"Keyword '{keyword}' não encontrado no GRDECL.")

    # Trata expoente Fortran D (1.0D+03)
    def _to_float(tok: str) -> float:
        return float(tok.replace("D", "E").replace("d", "e"))

    if dtype in (float, np.float32, np.float64):
        return np.array([_to_float(t) for t in tokens], dtype=float)

    # int
    return np.array([int(_to_float(t)) for t in tokens], dtype=int)

def _reshape_flat(arr1d: np.ndarray, nx: int, ny: int, nz: int, *, flip_k: bool, dtype) -> np.ndarray:
    """
    Reorganiza o array 1D (GRDECL) para o mesmo ordering do pv.read_grdecl.

    Convenção: GRDECL -> I mais rápido, depois J, depois K.
    A gente reshape para (nz, ny, nx) e ravel em C.
    flip_k=True inverte o eixo K (Z) para corrigir inversão.
    """
    expected = nx * ny * nz
    if arr1d.size != expected:
        raise ValueError(f"Tamanho do array ({arr1d.size}) != nx*ny*nz ({expected}).")

    a = np.asarray(arr1d, dtype=dtype).reshape((nz, ny, nx), order="C")

    if flip_k:
        a = a[::-1, :, :]

    return a.ravel(order="C")


# =========================
# High-level loaders
# =========================

def load_grid_from_grdecl(
    grdecl_path: str,
    *,
    facies_keyword: str = "Facies",
    thickness_keyword: str = "StratigraphicThickness",
    flip_k: bool = FLIP_K_DEFAULT,
    apply_reflection: bool = APPLY_REFLECTION,
    anchor_y: float = ANCHOR_Y,
    verbose: bool = VERBOSE_DEFAULT,
):
    """
    Carrega geometria + Facies + Propriedades Petrofísicas (PORO, PERM...).
    """
    if verbose:
        print(f"\nLendo Grid: {grdecl_path}...")

    # 1. Leitura da Geometria via PyVista
    g = pv.read_grdecl(grdecl_path)

    # Reflexão em Y (se configurado)
    if apply_reflection:
        pts = g.points.copy()
        pts[:, 1] = 2.0 * float(anchor_y) - pts[:, 1]
        g.points = pts
        if verbose:
            print(f">>> Grid Refletido Y (Pivô {anchor_y})")

    if verbose:
        print("Bounds Finais:   ", g.bounds)

    nx_, ny_, nz_ = read_specgrid(grdecl_path)

    # 2. Leitura de Fácies
    try:
        fac_raw = read_keyword_array(grdecl_path, facies_keyword, dtype=float)
        fac_1d = _reshape_flat(fac_raw, nx_, ny_, nz_, flip_k=flip_k, dtype=int)
    except KeyError:
        if verbose: print(f"[WARN] Keyword '{facies_keyword}' não encontrada. Criando array zerado.")
        fac_1d = np.zeros(g.n_cells, dtype=int)
        
    g.cell_data["Facies"] = fac_1d

    # 3. Leitura de Espessura
    try:
        th_raw = read_keyword_array(grdecl_path, thickness_keyword, dtype=float)
        th_1d = _reshape_flat(th_raw, nx_, ny_, nz_, flip_k=flip_k, dtype=float)
        g.cell_data["StratigraphicThickness"] = th_1d
        g.cell_data["cell_thickness"] = th_1d
    except KeyError:
        pass
    except Exception as e:
        if verbose: print(f"[WARN] Falha ao ler '{thickness_keyword}': {e}")

    # 4. --- Auto: Leitura de propriedades numéricas (qualquer keyword com nx*ny*nz) ---
    auto_exclude = {
        "SPECGRID", "COORD", "ZCORN", "MAPAXES", "MAPUNITS", "GRIDUNIT",
        facies_keyword, thickness_keyword, "Facies", "StratigraphicThickness", "cell_thickness",
    }

    auto_arrays = discover_numeric_keyword_arrays(
        grdecl_path, nx_, ny_, nz_, exclude=auto_exclude
    )

    for key, arr_raw in auto_arrays.items():
        try:
            # Reshape e Flip K (importante para alinhar com a geometria)
            arr_1d = _reshape_flat(arr_raw, nx_, ny_, nz_, flip_k=flip_k, dtype=float)
            g.cell_data[key] = arr_1d
            if verbose: print(f"[INFO] Propriedade carregada: {key}")
        except Exception as e:
            if verbose: print(f"[WARN] Erro ao ler propriedade '{key}': {e}")

    return g, fac_1d


def load_facies_from_grdecl(
    grdecl_path: str,
    *,
    facies_keyword: str = "Facies",
    flip_k: bool = FLIP_K_DEFAULT,
):
    """Compat: alguns lugares importam essa função."""
    nx_, ny_, nz_ = read_specgrid(grdecl_path)
    fac_raw = read_keyword_array(grdecl_path, facies_keyword, dtype=float)
    fac_1d = _reshape_flat(fac_raw, nx_, ny_, nz_, flip_k=flip_k, dtype=int)
    return fac_1d, nx_, ny_, nz_


# =========================
# Backward-compatible globals
# =========================

def _init_default_globals():
    g, f = load_grid_from_grdecl(DEFAULT_GRDECL_PATH, flip_k=FLIP_K_DEFAULT, verbose=VERBOSE_DEFAULT)
    nx_, ny_, nz_ = read_specgrid(DEFAULT_GRDECL_PATH)
    return g, f, nx_, ny_, nz_

grid, facies, nx, ny, nz = _init_default_globals()
