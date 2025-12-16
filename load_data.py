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
    with open(grdecl_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    m = re.search(rf"\b{re.escape(keyword)}\b", text, flags=re.IGNORECASE)
    if not m:
        raise KeyError(f"Keyword '{keyword}' não encontrado no GRDECL.")

    after = text[m.end():].splitlines()
    tokens = _tokenize_keyword_block(after)
    tokens = _expand_repeats(tokens)

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
    Carrega geometria + Facies (+ StratigraphicThickness se existir).

    RETORNO (fixo, 2 valores):
        grid, facies_1d
    """
    if verbose:
        print(f"\nLendo Grid: {grdecl_path}...")

    g = pv.read_grdecl(grdecl_path)

    # Reflexão em Y (se seu projeto usa isso)
    if apply_reflection:
        pts = g.points.copy()
        pts[:, 1] = 2.0 * float(anchor_y) - pts[:, 1]
        g.points = pts
        if verbose:
            print(f">>> Grid Refletido Y (Pivô {anchor_y})")

    if verbose:
        print("Bounds Finais:   ", g.bounds)

    nx_, ny_, nz_ = read_specgrid(grdecl_path)

    # Facies
    fac_raw = read_keyword_array(grdecl_path, facies_keyword, dtype=float)
    fac_1d = _reshape_flat(fac_raw, nx_, ny_, nz_, flip_k=flip_k, dtype=int)
    g.cell_data["Facies"] = fac_1d

    # Thickness (opcional)
    try:
        th_raw = read_keyword_array(grdecl_path, thickness_keyword, dtype=float)
        th_1d = _reshape_flat(th_raw, nx_, ny_, nz_, flip_k=flip_k, dtype=float)
        g.cell_data["StratigraphicThickness"] = th_1d
        g.cell_data["cell_thickness"] = th_1d  # alias conveniente
    except KeyError:
        if verbose:
            print(f"[INFO] Keyword '{thickness_keyword}' não existe neste GRDECL.")
    except Exception as e:
        if verbose:
            print(f"[WARN] Falha ao ler '{thickness_keyword}': {e}")

    if verbose:
        print("has StratigraphicThickness?", "StratigraphicThickness" in g.cell_data)
        print("cell_data keys sample:", list(g.cell_data.keys())[:10])
        if "StratigraphicThickness" in g.cell_data:
            a = g.cell_data["StratigraphicThickness"]
            print(f"StratigraphicThickness: n={a.size} min={float(np.nanmin(a))} max={float(np.nanmax(a))}")

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
