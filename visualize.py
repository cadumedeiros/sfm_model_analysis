# visualize_all.py
import numpy as np
import pyvista as pv
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from load_data import grid, facies, nx, ny, nz
from config import load_facies_colors

FACIES_COLORS = load_facies_colors()

# ---------------------------------------------------------
# Índice de camada (k) por célula, contado do topo à base
# ---------------------------------------------------------

k3d = np.zeros((nx, ny, nz), dtype=int)
for k in range(nz):
    # 0 = topo, nz-1 = base
    k3d[:, :, k] = nz - 1 - k   # <<< NOVO

k_index_flat = k3d.reshape(-1, order="F")
grid.cell_data["k_index"] = k_index_flat
N_LAYERS = nz

THICKNESS_SCALAR_NAME = "thickness_local"  # padrão
THICKNESS_SCALAR_TITLE = "Thickness local"

def set_thickness_scalar(name, title=None):
    global THICKNESS_SCALAR_NAME, THICKNESS_SCALAR_TITLE
    THICKNESS_SCALAR_NAME = name
    THICKNESS_SCALAR_TITLE = title or name

def add_facies_legend(plotter, position=(0.87, 0.30)):
    # carrega do seu config
    raw_colors = load_facies_colors()

    # normaliza só se precisar
    facies_colors = {}
    for fac, (r, g, b, a) in raw_colors.items():
        if r > 1 or g > 1 or b > 1:  # veio 0–255
            facies_colors[fac] = (r/255, g/255, b/255)
        else:  # já veio 0–1
            facies_colors[fac] = (r, g, b)

    # monta figura da legenda
    n = len(facies_colors)
    fig_height = max(2, n * 0.28)
    fig, ax = plt.subplots(figsize=(2.4, fig_height))
    ax.axis("off")
    ax.set_facecolor((1, 1, 1, 0.0))

    ax.text(0.0, n * 0.32, "FÁCIES", fontsize=10, fontweight="bold", color="black")

    for i, (fac, color) in enumerate(sorted(facies_colors.items())):
        y = i * 0.3
        ax.add_patch(
            Rectangle((0, y), 0.35, 0.25, facecolor=color, edgecolor="black")
        )
        ax.text(
            0.42, y,
            str(fac),
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="black"
        )

    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, n * 0.32)
    fig.tight_layout(pad=0.2)

    tmpfile = "assets/_facies_legend.png"
    fig.savefig(tmpfile, dpi=200, transparent=True)
    plt.close(fig)

    # põe no canto
    legend = plotter.add_logo_widget(tmpfile, position=position, size=(0.25, 0.55),)
    legend.SetProcessEvents(False)



# ---------- 2. LUTs ----------
def make_facies_lut():
    all_facies = np.unique(facies)
    max_fac = int(all_facies.max())
    lut = pv.LookupTable(n_values=max_fac + 1)
    # default cinza opaco
    for v in range(max_fac + 1):
        lut.SetTableValue(v, 0.8, 0.8, 0.8, 1.0)
    for fac, rgba in FACIES_COLORS.items():
        if fac <= max_fac:
            lut.SetTableValue(fac, *rgba)  # já vem com alpha = 1 do teu config
    return lut, (0, max_fac)


def make_reservoir_lut():
    lut = pv.LookupTable(n_values=2)
    lut.SetTableValue(0, 0.8, 0.8, 0.8, 1.0)  # fundo cinza transparente
    lut.SetTableValue(1, 0.0, 0.5, 1.0, 1.0)   # reservatório sólido
    return lut, (0, 1)


def make_largest_lut():
    lut = pv.LookupTable(n_values=2)
    lut.SetTableValue(0, 0.8, 0.8, 0.8, 1.0)  # fundo
    lut.SetTableValue(1, 1.0, 0.0, 0.0, 1.0)   # maior cluster sólido
    return lut, (0, 1)


def make_clusters_lut(clusters_array):
    n_clusters = int(clusters_array.max()) + 1
    lut = pv.LookupTable(n_values=n_clusters)

    # não vamos mostrar o 0 aqui, mas deixa com alpha 1
    lut.SetTableValue(0, 1, 1, 1, 1.0)

    rng = np.random.RandomState(42)
    for i in range(1, n_clusters):
        r, g, b = rng.rand(3)
        lut.SetTableValue(i, r, g, b, 1.0)

    return lut, (0, n_clusters - 1)


def run(mode="facies", z_exag=15.0, show_scalar_bar=False):
    global MODE, Z_EXAG, SHOW_SCALAR_BAR
    MODE = mode
    Z_EXAG = z_exag
    SHOW_SCALAR_BAR = show_scalar_bar

    # ---------- 3. Grid base exagerado ----------
    grid_base = grid.copy()

    x_min, x_max, y_min, y_max, z_min, z_max = grid_base.bounds

    grid_base.points[:, 1] = y_max - (grid_base.points[:, 1] - y_min)
    grid_base.points[:, 2] *= Z_EXAG

    plotter = pv.Plotter()

    state = {"bg_actor": None, "main_actor": None, "mode": MODE, "k_min": 0,}


    # ---------- 4. cola arrays no recorte ----------
    def attach_cell_data_from_original(clipped, original):
        if "vtkOriginalCellIds" not in clipped.cell_data:
            return clipped
        orig_ids = clipped.cell_data["vtkOriginalCellIds"]
        for name, arr in original.cell_data.items():
            clipped.cell_data[name] = arr[orig_ids]
        return clipped
    
    
    # ---------- 4b. filtro por camada k_min ----------
    def apply_k_filter(mesh):
        """
        Mantém apenas células com k_index >= state['k_min'].
        Funciona tanto no grid original quanto em recortes (clip/threshold),
        usando vtkOriginalCellIds quando disponível.
        """
        kmin = state.get("k_min", 0)
        if kmin <= 0:
            return mesh  # nada a filtrar

        # tenta pegar k_index diretamente
        if "k_index" in mesh.cell_data:
            k_arr = mesh.cell_data["k_index"]
        elif "vtkOriginalCellIds" in mesh.cell_data and "k_index" in grid_base.cell_data:
            orig_ids = mesh.cell_data["vtkOriginalCellIds"]
            k_arr = grid_base.cell_data["k_index"][orig_ids]
        else:
            # não sabe como mapear -> não filtra
            return mesh

        idx = np.where(k_arr >= kmin)[0]
        if idx.size == 0:
            # devolve um mesh vazio compatível
            return mesh.extract_cells([])

        return mesh.extract_cells(idx)



    # ---------- 5. desenha ----------
    def show_mesh(mesh):
        mode = state["mode"]
        
        # aplica o filtro por camada antes de qualquer outra coisa
        mesh = apply_k_filter(mesh)

        if mode == "facies":
            lut, rng = make_facies_lut()
            actor = plotter.add_mesh(
                mesh,
                scalars="Facies",
                show_edges=False,
                name="main",
                reset_camera=False,
            )
            actor.mapper.lookup_table = lut
            actor.mapper.scalar_range = rng

            state["bg_actor"] = None
            state["main_actor"] = actor

            add_facies_legend(plotter)

            plotter.remove_scalar_bar()
            
        elif mode == "thickness_local":
            scalar_name = THICKNESS_SCALAR_NAME
            thr = 1e-6

            mesh = apply_k_filter(mesh)

            bg = mesh.threshold(thr, invert=True, scalars=scalar_name)
            main = mesh.threshold(thr, scalars=scalar_name)

            if bg.n_cells > 0:
                bg_actor = plotter.add_mesh(
                    bg,
                    color=(0.8, 0.8, 0.8),
                    opacity=0.01,
                    show_edges=False,
                    name="bg",
                    reset_camera=False,
                )
            else:
                bg_actor = None

            main_actor = plotter.add_mesh(
                main,
                scalars=scalar_name,
                cmap="plasma",
                show_edges=True,
                name="main",
                reset_camera=False,
                scalar_bar_args={"title": THICKNESS_SCALAR_TITLE},
            )

            # garante que está usando dados de CÉLULA desse campo
            mapper = main_actor.mapper
            mapper.SetScalarModeToUseCellFieldData()
            mapper.SelectColorArray(scalar_name)
            mapper.SetScalarVisibility(True)

            state["bg_actor"] = bg_actor
            state["main_actor"] = main_actor


        elif mode == "reservoir":

            mesh = apply_k_filter(mesh)

            bg = mesh.threshold(0.5, invert=True, scalars="Reservoir")
            main = mesh.threshold(0.5, scalars="Reservoir")

            # fundo cinza transparente
            if bg.n_cells > 0:
                bg_actor = plotter.add_mesh(
                    bg,
                    color=(0.8, 0.8, 0.8),
                    opacity=0.02,
                    show_edges=False,
                    name="bg",
                    reset_camera=False,
                )
            else:
                bg_actor = None

            # reservatório sólido
            main_actor = plotter.add_mesh(
                main,
                color="orange",
                opacity=1.0,
                show_edges=True,
                name="main",
                reset_camera=False,
            )

            state["bg_actor"] = bg_actor
            state["main_actor"] = main_actor

        elif mode == "largest":

            mesh = apply_k_filter(mesh)

            bg = mesh.threshold(0.5, invert=True, scalars="LargestCluster")
            main = mesh.threshold(0.5, scalars="LargestCluster")

            # fundo cinza transparente
            if bg.n_cells > 0:
                bg_actor = plotter.add_mesh(
                    bg,
                    color=(0.8, 0.8, 0.8),
                    opacity=0.02,
                    show_edges=False,
                    name="bg",
                    reset_camera=False,
                )
            else:
                bg_actor = None

            main_actor = plotter.add_mesh(
                main,
                color="orange",
                opacity=1.0,
                show_edges=True,
                name="main",
                reset_camera=False,
            )

            state["bg_actor"] = bg_actor
            state["main_actor"] = main_actor

        elif mode == "clusters":

            mesh = apply_k_filter(mesh)

            bg = mesh.threshold(0.5, invert=True, scalars="Clusters")
            main = mesh.threshold(0.5, scalars="Clusters")

            # fundo cinza transparente
            if bg.n_cells > 0:
                bg_actor = plotter.add_mesh(
                    bg,
                    color=(0.8, 0.8, 0.8),
                    opacity=0.05,
                    show_edges=False,
                    name="bg",
                    reset_camera=False,
                )
            else:
                bg_actor = None

            # clusters sólidos coloridos
            main_actor = plotter.add_mesh(
                main,
                scalars="Clusters",
                show_edges=True,
                name="main",
                reset_camera=False,
            )

            lut, rng = make_clusters_lut(main.cell_data["Clusters"])
            mapper = main_actor.mapper
            mapper.lookup_table = lut
            mapper.scalar_range = rng
            mapper.SetScalarModeToUseCellFieldData()
            mapper.SelectColorArray("Clusters")
            mapper.SetScalarVisibility(True)
            main_actor.prop.opacity = 1.0


            # guarda ambos no state
            state["bg_actor"] = bg_actor
            state["main_actor"] = main_actor

            plotter.remove_scalar_bar()
        
        elif MODE == "ntg_local":
            
            mesh = apply_k_filter(mesh)


            main_actor = plotter.add_mesh(
                mesh,
                scalars="NTG_local",
                cmap="plasma",
                clim=[0.0, 1.0],
                show_edges=True,
                name="main",
                reset_camera=False,
                scalar_bar_args={"title": "NTG local"},
            )


            # guarda ambos no state
            state["bg_actor"] = None
            state["main_actor"] = main_actor

        else:
            raise ValueError(f"Modo desconhecido: {mode}")


        if not SHOW_SCALAR_BAR:
            if hasattr(plotter, "scalar_bars") and plotter.scalar_bars:
                plotter.remove_scalar_bar()


    # ---------- 5b. controle de camada: subir/descer k_min ----------
    def change_k(delta):
        kmin = state.get("k_min", 0)
        new = int(np.clip(kmin + delta, 0, N_LAYERS - 1))
        if new == kmin:
            return

        state["k_min"] = new
        print(f"[visualize] k_min = {new} (0 = topo)")

        # 1. Começa sempre do grid_base
        base = grid_base

        # 2. Se tiver caixa ativa, aplica o clip primeiro
        if state.get("box_widget"):
            box = state["box_widget"].GetBounds()
            base = grid_base.clip_box(box, invert=False, crinkle=True)
            base = attach_cell_data_from_original(base, grid_base)

        # 3. Aplica o filtro de camada
        mesh = apply_k_filter(base)
        mode = state["mode"]

        # 4. Atualiza conforme o modo --------------------------

        if mode == "facies":
            if state["main_actor"] is not None:
                state["main_actor"].mapper.SetInputData(mesh)
                state["main_actor"].mapper.Update()

        elif mode == "thickness_local":
            scalar_name = THICKNESS_SCALAR_NAME
            thr = 1e-6

            bg = mesh.threshold(thr, invert=True, scalars=scalar_name)
            main = mesh.threshold(thr, scalars=scalar_name)

            if state["bg_actor"] is not None:
                state["bg_actor"].mapper.SetInputData(bg)
                state["bg_actor"].mapper.Update()

            if state["main_actor"] is not None:
                actor = state["main_actor"]
                mapper = actor.mapper
                mapper.SetInputData(main)
                mapper.SetScalarModeToUseCellFieldData()
                mapper.SelectColorArray(scalar_name)
                mapper.SetScalarVisibility(True)
                mapper.Update()

        elif mode == "reservoir":
            bg = mesh.threshold(0.5, invert=True, scalars="Reservoir")
            main = mesh.threshold(0.5, scalars="Reservoir")

            if state["bg_actor"] is not None:
                state["bg_actor"].mapper.SetInputData(bg)
                state["bg_actor"].mapper.Update()

            if state["main_actor"] is not None:
                state["main_actor"].mapper.SetInputData(main)
                state["main_actor"].mapper.Update()

        elif mode == "largest":
            bg = mesh.threshold(0.5, invert=True, scalars="LargestCluster")
            main = mesh.threshold(0.5, scalars="LargestCluster")

            if state["bg_actor"] is not None:
                state["bg_actor"].mapper.SetInputData(bg)
                state["bg_actor"].mapper.Update()

            if state["main_actor"] is not None:
                state["main_actor"].mapper.SetInputData(main)
                state["main_actor"].mapper.Update()

        elif mode == "clusters":
            bg = mesh.threshold(0.5, invert=True, scalars="Clusters")
            main = mesh.threshold(0.5, scalars="Clusters")

            if state["bg_actor"] is not None:
                state["bg_actor"].mapper.SetInputData(bg)
                state["bg_actor"].mapper.Update()

            if state["main_actor"] is not None:
                actor = state["main_actor"]
                mapper = actor.mapper
                mapper.SetInputData(main)

                # reconstroi LUT com os clusters presentes
                lut, rng = make_clusters_lut(main.cell_data["Clusters"])
                mapper.lookup_table = lut
                mapper.scalar_range = rng

                mapper.SetScalarModeToUseCellFieldData()
                mapper.SelectColorArray("Clusters")
                mapper.SetScalarVisibility(True)
                mapper.Update()

        elif mode == "ntg_local":
            if state["main_actor"] is not None:
                state["main_actor"].mapper.SetInputData(mesh)
                state["main_actor"].mapper.Update()

        plotter.render()






    # primeiro draw
    show_mesh(grid_base)

     # atalhos: seta para cima/baixo controlam o índice de camada
    plotter.add_key_event("z",   lambda: change_k(-1))  # sobe (mostra mais topo)
    plotter.add_key_event("x", lambda: change_k(+1))  # desce (mostra mais base)


    # ---------- 6. callback do corte (suave) ----------
    def box_callback(box):
        mode = state["mode"]

        if mode == "facies":
            # corta o grid inteiro (porque facies mostra tudo)
            clipped = grid_base.clip_box(box, invert=False, crinkle=True)
            if clipped.n_cells == 0:
                return

            clipped = attach_cell_data_from_original(clipped, grid_base)
            clipped = apply_k_filter(clipped)
            state["main_actor"].mapper.SetInputData(clipped)
            state["main_actor"].mapper.Update()
            return
        
        if mode == "thickness_local":
            scalar_name = THICKNESS_SCALAR_NAME
            thr = 1e-6

            # 1) pega só o reservatório (células com espessura > 0)
            res_only = grid_base.threshold(thr, scalars=scalar_name)

            # 2) corta só o reservatório
            res_clipped = res_only.clip_box(box, invert=False, crinkle=True)
            res_clipped = attach_cell_data_from_original(res_clipped, grid_base)
            res_clipped = apply_k_filter(res_clipped)

            # 3) fundo = modelo inteiro cortado
            bg_clipped = grid_base.clip_box(box, invert=False, crinkle=True)
            bg_clipped = attach_cell_data_from_original(bg_clipped, grid_base)
            bg_clipped = apply_k_filter(bg_clipped)

            # 4) atualiza actors
            if state["bg_actor"] is not None:
                bg_no_res = bg_clipped.threshold(thr, invert=True, scalars=scalar_name)
                state["bg_actor"].mapper.SetInputData(bg_no_res)
                state["bg_actor"].mapper.Update()

            if state["main_actor"] is not None:
                actor = state["main_actor"]
                mapper = actor.mapper
                mapper.SetInputData(res_clipped)
                mapper.SetScalarModeToUseCellFieldData()
                mapper.SelectColorArray(scalar_name)
                mapper.SetScalarVisibility(True)
                mapper.Update()

            return

        if mode == "reservoir":
            res_only = grid_base.threshold(0.5, scalars="Reservoir")

            res_clipped = res_only.clip_box(box, invert=False, crinkle=True)
            res_clipped = attach_cell_data_from_original(res_clipped, grid_base)
            res_clipped = apply_k_filter(res_clipped)

            bg_clipped = grid_base.clip_box(box, invert=False, crinkle=True)
            bg_clipped = attach_cell_data_from_original(bg_clipped, grid_base)
            bg_clipped = apply_k_filter(bg_clipped)

            if state["bg_actor"] is not None:
                bg_no_res = bg_clipped.threshold(0.5, invert=True, scalars="Reservoir")
                state["bg_actor"].mapper.SetInputData(bg_no_res)
                state["bg_actor"].mapper.Update()

            if state["main_actor"] is not None:
                state["main_actor"].mapper.SetInputData(res_clipped)
                state["main_actor"].mapper.Update()

            return

        elif mode == "largest":
            largest_only = grid_base.threshold(0.5, scalars="LargestCluster")
            largest_clipped = largest_only.clip_box(box, invert=False, crinkle=True)
            largest_clipped = attach_cell_data_from_original(largest_clipped, grid_base)
            largest_clipped = apply_k_filter(largest_clipped)

            bg_clipped = grid_base.clip_box(box, invert=False, crinkle=True)
            bg_clipped = attach_cell_data_from_original(bg_clipped, grid_base)
            bg_clipped = apply_k_filter(bg_clipped)

            if state["bg_actor"] is not None:
                bg_no_largest = bg_clipped.threshold(0.5, invert=True, scalars="LargestCluster")
                state["bg_actor"].mapper.SetInputData(bg_no_largest)
                state["bg_actor"].mapper.Update()

            if state["main_actor"] is not None:
                state["main_actor"].mapper.SetInputData(largest_clipped)
                state["main_actor"].mapper.Update()

            return

        elif mode == "clusters":
            clusters_only = grid_base.threshold(0.5, scalars="Clusters")
            clusters_clipped = clusters_only.clip_box(box, invert=False, crinkle=True)
            clusters_clipped = attach_cell_data_from_original(clusters_clipped, grid_base)
            clusters_clipped = apply_k_filter(clusters_clipped)

            bg_clipped = grid_base.clip_box(box, invert=False, crinkle=True)
            bg_clipped = attach_cell_data_from_original(bg_clipped, grid_base)
            bg_clipped = apply_k_filter(bg_clipped)

            if state["bg_actor"] is not None:
                bg_no_clusters = bg_clipped.threshold(0.5, invert=True, scalars="Clusters")
                state["bg_actor"].mapper.SetInputData(bg_no_clusters)
                state["bg_actor"].mapper.Update()

            if state["main_actor"] is not None:
                actor = state["main_actor"]
                mapper = actor.mapper
                mapper.SetInputData(clusters_clipped)

                lut, rng = make_clusters_lut(clusters_clipped.cell_data["Clusters"])
                mapper.lookup_table = lut
                mapper.scalar_range = rng

                mapper.SetScalarModeToUseCellFieldData()
                mapper.SelectColorArray("Clusters")
                mapper.SetScalarVisibility(True)
                mapper.Update()

                actor.prop.opacity = 1.0

            return

        elif mode == "ntg_local":
            clipped = grid_base.clip_box(box, invert=False, crinkle=True)
            if clipped.n_cells == 0:
                return

            clipped = attach_cell_data_from_original(clipped, grid_base)
            clipped = apply_k_filter(clipped)

            if state["main_actor"] is not None:
                state["main_actor"].mapper.SetInputData(clipped)
                state["main_actor"].mapper.Update()

            return



    box_widget = plotter.add_box_widget(
        callback=box_callback,
        bounds=grid_base.bounds,
        rotation_enabled=False,
        interaction_event="always",
    )

    logo = plotter.add_logo_widget("assets/forward_PNG.png", position=(0.02, 0.85), size=(0.12, 0.12))
    logo.SetProcessEvents(False)
    plotter.set_background("white", top="lightblue")

    plotter.enable_anti_aliasing('ssaa')

    box_widget.SetHandleSize(0.01)

    rep = box_widget.GetHandleProperty()
    rep.SetOpacity(0.1)

    shp = box_widget.GetSelectedHandleProperty()
    shp.SetOpacity(0.0)

    op = box_widget.GetOutlineProperty()
    op.SetOpacity(0.05)

    plotter.add_axes()
    plotter.show()


def show_thickness_2d(surf, scalar_name="thickness_2d"):
    # troca valores negativos por NaN pra ficarem brancos
    arr = surf.cell_data[scalar_name]
    arr = np.where(arr < 0, np.nan, arr)
    surf.cell_data[scalar_name] = arr

    p = pv.Plotter()

    p.add_mesh(
        surf,
        scalars=scalar_name,
        cmap="plasma",
        show_edges=True,
        edge_color="black",
        line_width=0.5,
        lighting=False,
        nan_color="white",
        interpolate_before_map=False,  # não suaviza
        preference="cell",             # <<< usa cell_data na sua versão
    )

    p.remove_scalar_bar()

    p.set_background("white")
    p.remove_bounds_axes()
    p.view_xy()
    p.enable_parallel_projection()
    p.enable_terrain_style()
    p.add_scalar_bar(title="Thickness")

    p.show()