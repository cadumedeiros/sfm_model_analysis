# visualize_all.py
import numpy as np
import pyvista as pv

from load_data import grid, facies, nx, ny, nz
from config import load_facies_colors

FACIES_COLORS = load_facies_colors()


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


def run(mode="reservoir", z_exag=15.0, show_scalar_bar=False):
    global MODE, Z_EXAG, SHOW_SCALAR_BAR
    MODE = mode
    Z_EXAG = z_exag
    SHOW_SCALAR_BAR = show_scalar_bar

    # ---------- 3. Grid base exagerado ----------
    grid_base = grid.copy()
    grid_base.points[:, 2] *= Z_EXAG

    plotter = pv.Plotter()

    state = {"bg_actor": None, "main_actor": None, "mode": MODE}


    # ---------- 4. cola arrays no recorte ----------
    def attach_cell_data_from_original(clipped, original):
        if "vtkOriginalCellIds" not in clipped.cell_data:
            return clipped
        orig_ids = clipped.cell_data["vtkOriginalCellIds"]
        for name, arr in original.cell_data.items():
            clipped.cell_data[name] = arr[orig_ids]
        return clipped


    # ---------- 5. desenha ----------
    def show_mesh(mesh):
        mode = state["mode"]

        if mode == "facies":
            lut, rng = make_facies_lut()
            actor = plotter.add_mesh(
                mesh,
                scalars="Facies",
                show_edges=True,
                name="main",
                reset_camera=False,
            )
            actor.mapper.lookup_table = lut
            actor.mapper.scalar_range = rng

            state["bg_actor"] = None
            state["main_actor"] = actor
            


        elif mode == "reservoir":
            bg = mesh.threshold(0.5, invert=True, scalars="Reservoir")
            main = mesh.threshold(0.5, scalars="Reservoir")

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

            # reservatório sólido
            main_actor = plotter.add_mesh(
                main,
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                show_edges=True,
                name="main",
                reset_camera=False,
            )

            state["bg_actor"] = bg_actor
            state["main_actor"] = main_actor

        elif mode == "largest":
            bg = mesh.threshold(0.5, invert=True, scalars="LargestCluster")
            main = mesh.threshold(0.5, scalars="LargestCluster")

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

            main_actor = plotter.add_mesh(
                main,
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                show_edges=True,
                name="main",
                reset_camera=False,
            )

            state["bg_actor"] = bg_actor
            state["main_actor"] = main_actor

        elif mode == "clusters":
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
            main_actor.mapper.lookup_table = lut
            main_actor.mapper.scalar_range = rng
            main_actor.prop.opacity = 1.0


            # guarda ambos no state
            state["bg_actor"] = bg_actor
            state["main_actor"] = main_actor
        
        elif MODE == "ntg_local":
            bg = mesh.threshold(0.5, invert=True, scalars="NTG_local")
            main = mesh.threshold(0.5, scalars="NTG_local")

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



    # primeiro draw
    show_mesh(grid_base)


    # ---------- 6. callback do corte (suave) ----------
    def box_callback(box):
        mode = state["mode"]

        if mode == "facies":
            # corta o grid inteiro (porque facies mostra tudo)
            clipped = grid_base.clip_box(box, invert=False, crinkle=True)
            if clipped.n_cells == 0:
                return

            clipped = attach_cell_data_from_original(clipped, grid_base)
            state["main_actor"].mapper.SetInputData(clipped)
            state["main_actor"].mapper.Update()
            return

        if mode == "reservoir":
            # 1) pega só o reservatório do modelo inteiro
            res_only = grid_base.threshold(0.5, scalars="Reservoir")

            # 2) corta só o reservatório
            res_clipped = res_only.clip_box(box, invert=False, crinkle=True)

            # 3) fundo = modelo inteiro cortado (pra ver a caixa atravessando)
            bg_clipped = grid_base.clip_box(box, invert=False, crinkle=True)
            bg_clipped = attach_cell_data_from_original(bg_clipped, grid_base)

            # 4) atualiza actors existentes
            if state["bg_actor"] is not None:
                # precisa tirar o reservatório do fundo
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

            bg_clipped = grid_base.clip_box(box, invert=False, crinkle=True)
            bg_clipped = attach_cell_data_from_original(bg_clipped, grid_base)

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

            bg_clipped = grid_base.clip_box(box, invert=False, crinkle=True)
            bg_clipped = attach_cell_data_from_original(bg_clipped, grid_base)

            # atualiza fundo
            if state["bg_actor"] is not None:
                bg_no_clusters = bg_clipped.threshold(0.5, invert=True, scalars="Clusters")
                state["bg_actor"].mapper.SetInputData(bg_no_clusters)
                state["bg_actor"].mapper.Update()

            # atualiza o main
            if state["main_actor"] is not None:
                actor = state["main_actor"]
                mapper = actor.mapper

                mapper.SetInputData(clusters_clipped)
                mapper.Update()

                # aplica LUT de novo
                lut, rng = make_clusters_lut(clusters_clipped.cell_data["Clusters"])
                mapper.lookup_table = lut
                mapper.scalar_range = rng

                # 👇 estas 3 linhas são o que estava faltando
                mapper.SetScalarModeToUseCellFieldData()
                mapper.SelectColorArray("Clusters")
                mapper.SetScalarVisibility(True)

                actor.prop.opacity = 1.0
            
            return
        
        elif mode == "ntg_local":
            clipped = grid_base.clip_box(box, invert=False, crinkle=True)
            if clipped.n_cells == 0:
                return

            clipped = attach_cell_data_from_original(clipped, grid_base)
            state["main_actor"].mapper.SetInputData(clipped)
            state["main_actor"].mapper.Update()
            return


    box_widget = plotter.add_box_widget(
        callback=box_callback,
        bounds=grid_base.bounds,
        rotation_enabled=False,
        interaction_event="always",
    )

    plotter.add_axes()
    plotter.show()
