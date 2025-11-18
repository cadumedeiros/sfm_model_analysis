# visualize_all.py
import numpy as np
import pyvista as pv
pv.global_theme.allow_empty_mesh = True
import vtk

from load_data import grid, facies, nx, ny, nz
from config import load_facies_colors
from derived_fields import ensure_reservoir, ensure_clusters
from local_windows import compute_local_ntg
from analysis import add_vertical_facies_metrics, make_thickness_2d_from_grid

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

def show_thickness_2d(surf, scalar_name, title=None):
    """
    Plota o mapa 2D sem duplicar barra de cores.
    """
    p = pv.Plotter(window_size=(1000, 800))
    
    # REMOVE a barra automática: setamos scalar_bar_args=None
    p.add_mesh(
        surf,
        scalars=scalar_name,
        cmap="plasma",
        show_edges=True,
        edge_color="black",
        line_width=0.5,
        scalar_bar_args=None,       # <- isto impede criar a barra automática
        nan_color="white",
        preference="cell",
    )

    p.view_xy()
    p.enable_parallel_projection()
    p.enable_image_style()
    p.set_background("white")
    p.add_scalar_bar(title=title if title else scalar_name)  # apenas UMA barra
    p.show()

def update_2d_plot(plotter, array_name_3d, title="Mapa 2D"):
    """
    Atualiza um plotter PyVista (QtInteractor/BackgroundPlotter) com o mapa 2D
    da métrica vertical 'array_name_3d'.

    - NÃO usa screenshot nem PNG.
    - NÃO duplica barra de cores.
    """
    # Gera a superfície 2D (X, Y) a partir do array 3D
    surf = make_thickness_2d_from_grid(
        array_name_3d,
        array_name_3d + "_2d",
    )
    scalar_name_2d = array_name_3d + "_2d"

    # Valores negativos viram NaN (branco)
    arr = surf.cell_data[scalar_name_2d]
    arr = np.where(arr < 0, np.nan, arr)
    surf.cell_data[scalar_name_2d] = arr

    # Limpa o plotter 2D
    plotter.clear()

    # Desenha o mapa, SEM barra automática
    plotter.add_mesh(
        surf,
        scalars=scalar_name_2d,
        cmap="plasma",
        show_edges=True,
        edge_color="black",
        line_width=0.5,
        scalar_bar_args=None,  # impede barra automática
        lighting=False,
        nan_color="white",
        interpolate_before_map=False,
        preference="cell",
    )

    # Configura a câmera 2D
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.enable_image_style()
    plotter.set_background("white")
    plotter.remove_bounds_axes()

    # Uma única barra de cores, com o título correto
    plotter.remove_scalar_bar()
    plotter.add_scalar_bar(title=title)

    plotter.reset_camera()



def compute_cluster_sizes(clusters_array):
    """
    Retorna um dict {cluster_id: tamanho_em_celulas},
    ignorando o cluster 0 (background).
    """
    arr = np.asarray(clusters_array, dtype=int)
    mask = arr > 0
    labels, counts = np.unique(arr[mask], return_counts=True)
    return {int(l): int(c) for l, c in zip(labels, counts)}


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


def make_clusters_lut(clusters_arr):
    """
    Gera LUT categórica usando paletas Brewer do VTK.
    Cores totalmente distintas, sem tons parecidos.
    """
    import vtk

    labels = np.unique(clusters_arr)
    labels = labels[labels > 0]  # ignora 0 (não reservatório)

    n = len(labels)

    # cria LUT
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(n + 1)
    lut.Build()

    # cor para cluster 0
    lut.SetTableValue(0, 0.2, 0.2, 0.2, 1.0)

    # usar paleta Brewer Set3
    series = vtk.vtkColorSeries()
    series.SetColorScheme(series.BREWER_QUALITATIVE_SET3)

    n_colors = series.GetNumberOfColors()

    # atribuir cores distintas (cíclicas se n > 12)
    for idx, lab in enumerate(labels, start=1):
        color = series.GetColor(idx % n_colors)
        r = color.GetRed()   / 255.0
        g = color.GetGreen() / 255.0
        b = color.GetBlue()  / 255.0
        lut.SetTableValue(idx, r, g, b, 1.0)

    return lut, (0, n + 1)




def run(mode="facies", z_exag=15.0, show_scalar_bar=False, external_plotter=None, external_state=None):
    if external_plotter is not None:
        plotter = external_plotter
    else:
        plotter = pv.Plotter()

    state = external_state if external_state is not None else {}
    state["mode"] = mode
    state["z_exag"] = z_exag
    state["show_scalar_bar"] = show_scalar_bar

    global MODE, Z_EXAG, SHOW_SCALAR_BAR
    MODE = mode
    Z_EXAG = z_exag
    SHOW_SCALAR_BAR = show_scalar_bar

    # ---------- 3. Grid base exagerado ----------
    grid_base = grid.copy()
    x_min, x_max, y_min, y_max, z_min, z_max = grid_base.bounds
    grid_base.points[:, 1] = y_max - (grid_base.points[:, 1] - y_min)
    grid_base.points[:, 2] *= Z_EXAG

    # --------------------------------------------

    # plotter = pv.Plotter()

    # state = {"bg_actor": None, 
    #          "main_actor": None, 
    #          "mode": MODE, 
    #          "k_min": 0,
    #          "box_bounds": grid_base.bounds,
    #          "facies_legend_actor": None,
    #          }

    state.setdefault("bg_actor", None)
    state.setdefault("main_actor", None)
    state.setdefault("k_min", 0)
    state.setdefault("box_bounds", grid_base.bounds)
    state.setdefault("facies_legend_actor", None)
    state.setdefault("clusters_legend_actor", None)

    def init_thickness_presets():
        presets = {}

        if "vert_Ttot_reservoir" in grid_base.cell_data:
            presets["Espessura"] = (
                "vert_Ttot_reservoir",
                "Espessura total reservatório (m)",
            )

        if "vert_NTG_col_reservoir" in grid_base.cell_data:
            presets["NTG coluna"] = (
                "vert_NTG_col_reservoir",
                "NTG coluna (reservatório)",
            )

        if "vert_NTG_env_reservoir" in grid_base.cell_data:
            presets["NTG envelope"] = (
                "vert_NTG_env_reservoir",
                "NTG envelope (reservatório)",
            )

        if "vert_Tpack_max_reservoir" in grid_base.cell_data:
            presets["Maior pacote"] = (
                "vert_Tpack_max_reservoir",
                "Maior pacote vertical (m)",
            )

        if "vert_n_packages_reservoir" in grid_base.cell_data:
            presets["Nº pacotes"] = (
                "vert_n_packages_reservoir",
                "Número de pacotes verticais",
            )

        if "vert_ICV_reservoir" in grid_base.cell_data:
            presets["ICV"] = (
                "vert_ICV_reservoir",
                "Índice de continuidade vertical (ICV)",
            )

        if "vert_Qv_reservoir" in grid_base.cell_data:
            presets["Qv"] = (
                "vert_Qv_reservoir",
                "Índice combinado Qv",
            )

        if "vert_Qv_abs_reservoir" in grid_base.cell_data:
            presets["Qv absoluto"] = (
                "vert_Qv_abs_reservoir",
                "Índice de qualidade vertical absoluta (Qv_abs)",
            )

        return presets


    
    def _sync_from_grid_to_grid_base(names):
        """Copia alguns arrays do grid original para o grid_base exagerado."""
        for name in names:
            if name in grid.cell_data:
                grid_base.cell_data[name] = grid.cell_data[name].copy()

    def update_reservoir_fields(reservoir_facies):
        """
        Recalcula:
        - array Reservoir (0/1)
        - Clusters e LargestCluster
        - NTG_local (janelas deslizantes)
        - métricas verticais de espessura para uma fácies de referência
        e sincroniza tudo para o grid_base.
        """
        # 1) Recalcula reservatório e clusters no grid original
        res_arr = ensure_reservoir(reservoir_facies)
        clusters_arr, largest_arr = ensure_clusters(reservoir_facies)

        grid.cell_data["Reservoir"] = res_arr.astype(np.uint8)
        grid.cell_data["Clusters"] = clusters_arr.astype(np.int32)
        grid.cell_data["LargestCluster"] = largest_arr.astype(np.uint8)

        # 2) NTG local (usa a máscara de reservatório)
        compute_local_ntg(res_arr, window=(1, 1, 5))  # mesmo window do main

        # 3) Espessura vertical: escolhe uma fácies de referência
        fac_set = reservoir_facies
        if isinstance(fac_set, (set, list, tuple)):
            fac_set = set(int(f) for f in fac_set)
        else:
            fac_set = {int(fac_set)}

        add_vertical_facies_metrics(fac_set)

        # 4) Copia arrays importantes pro grid_base
        sync_names = ["Reservoir", "Clusters", "LargestCluster", "NTG_local"]
        for key in grid.cell_data.keys():
            if key.startswith("vert_"):
                sync_names.append(key)


        _sync_from_grid_to_grid_base(sync_names)

        # 5) Recalcula LUT e tamanhos de clusters para o modo "clusters"
        if "Clusters" in grid_base.cell_data:
            full_clusters_arr = grid_base.cell_data["Clusters"]
            state["clusters_lut"], state["clusters_rng"] = make_clusters_lut(full_clusters_arr)
            state["clusters_sizes"] = compute_cluster_sizes(full_clusters_arr)

        # 6) Atualiza o range do thickness_local com o novo scalar
        scalar_name = THICKNESS_SCALAR_NAME
        if scalar_name in grid_base.cell_data:
            arr = grid_base.cell_data[scalar_name]
            vmin = 1e-6
            vmax = arr.max()
            if vmax <= vmin:
                vmax = vmin + 1.0
            state["thickness_clim"] = (vmin, vmax)

    state.setdefault("thickness_presets", init_thickness_presets())
    state.setdefault("thickness_mode", "Espessura")
    state["update_reservoir_fields"] = update_reservoir_fields

    def _update_thickness_from_state():
        """
        Lê state['thickness_mode'] e state['thickness_presets'],
        escolhe o scalar correto e atualiza:
        - THICKNESS_SCALAR_NAME
        - THICKNESS_SCALAR_TITLE
        - state['thickness_clim']
        """
        presets = state.get("thickness_presets") or {}
        mode_label = state.get("thickness_mode", "Espessura")

        if mode_label not in presets:
            if "Espessura" in presets:
                mode_label = "Espessura"
                state["thickness_mode"] = "Espessura"
            else:
                return  # não há nada pra fazer

        scalar_name, title = presets[mode_label]

        # atualiza os globais
        set_thickness_scalar(scalar_name, title)

        # recalcula o clim com base no array 3D completo
        if scalar_name in grid_base.cell_data:
            arr = grid_base.cell_data[scalar_name]
            vmin = 1e-6
            vmax = float(arr.max())
            if vmax <= vmin:
                vmax = vmin + 1.0
            state["thickness_clim"] = (vmin, vmax)

    # já deixa configurado um scalar coerente na inicialização
    _update_thickness_from_state()

    # Para o modo "clusters", pré-calcula a LUT
    if "Clusters" in grid_base.cell_data:
        full_clusters_arr = grid_base.cell_data["Clusters"]
        state["clusters_lut"], state["clusters_rng"] = make_clusters_lut(full_clusters_arr)
        state["clusters_sizes"] = compute_cluster_sizes(full_clusters_arr)

    # Para o modo "thickness_local", pré-calcula o range (clim)
    if MODE == "thickness_local" and THICKNESS_SCALAR_NAME in grid_base.cell_data:
        arr = grid_base.cell_data[THICKNESS_SCALAR_NAME]
        # Usamos 1e-6 como o mínimo (igual ao seu threshold)
        vmin = 1e-6 
        vmax = arr.max()
        if vmax <= vmin:
            vmax = vmin + 1.0 # Garante um range válido
        # Salva o range no 'state'
        state["thickness_clim"] = (vmin, vmax)


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
    

    def _update_facies_mapper(actor, mesh):
        mapper = actor.mapper
        mapper.SetInputData(mesh)
        mapper.Update()

    def _update_simple_mapper(actor, mesh):
        """Atualiza mappers simples ('reservoir', 'largest', 'ntg_local')"""
        mapper = actor.mapper
        mapper.SetInputData(mesh)
        mapper.Update()

    def _update_thickness_mapper(actor, mesh):
        """Atualiza o mapper do modo 'thickness_local'"""
        scalar_name = THICKNESS_SCALAR_NAME
        mapper = actor.mapper
        mapper.SetInputData(mesh)
        mapper.SetScalarModeToUseCellFieldData() # <-- A Chave
        mapper.SelectColorArray(scalar_name)
        if "thickness_clim" in state:
            mapper.scalar_range = state["thickness_clim"]
        mapper.SetScalarVisibility(True)
        mapper.Update()

    def _update_clusters_mapper(actor, mesh):
        """Atualiza o mapper do modo 'clusters' """
        mapper = actor.mapper
        mapper.SetInputData(mesh)
        
        if "clusters_lut" in state:
            lut = state["clusters_lut"]
            rng = state["clusters_rng"]
        else: 
            lut, rng = make_clusters_lut(grid_base.cell_data["Clusters"])
        
        mapper.lookup_table = lut
        mapper.scalar_range = rng
        mapper.SetScalarModeToUseCellFieldData()  
        mapper.SelectColorArray("Clusters")
        mapper.SetScalarVisibility(True)
        mapper.Update()

    # ---------- 5. desenha ----------
    def show_mesh(mesh):
        mode = state["mode"]

        mesh = attach_cell_data_from_original(mesh, grid_base)
        
        # aplica o filtro por camada antes de qualquer outra coisa
        mesh = apply_k_filter(mesh)

         # --- remove atores anteriores para não misturar modos ---
        for key in ("bg_actor", "main_actor"):
            old = state.get(key)
            if old is not None:
                try:
                    plotter.remove_actor(old)
                except Exception:
                    pass
                state[key] = None

        # --- controla visibilidade das legendas (fácies e clusters) ---
        if mode != "facies":
            fl = state.get("facies_legend_actor")
            if fl is not None:
                try:
                    # tenta remover do plotter
                    plotter.remove_actor(fl)
                except Exception:
                    # se for widget, tenta desligar
                    try:
                        fl.Off()
                    except Exception:
                        pass
                state["facies_legend_actor"] = None

        # --- se não estou em Clusters, some com a legenda de clusters ---
        if mode != "clusters":
            cl = state.get("clusters_legend_actor")
            if cl is not None:
                try:
                    plotter.remove_actor(cl)
                except Exception:
                    try:
                        cl.Off()
                    except Exception:
                        pass
                state["clusters_legend_actor"] = None

        if mode == "facies":
            lut, rng = make_facies_lut()
            actor = plotter.add_mesh(
                mesh, scalars="Facies", show_edges=True, name="main", reset_camera=False,
            )
            plotter.reset_camera_clipping_range()
            actor.mapper.lookup_table = lut
            actor.mapper.scalar_range = rng
            state["bg_actor"] = None
            state["main_actor"] = actor
            # state["facies_legend_actor"] = add_facies_legend(plotter)

            plotter.remove_scalar_bar()
            
        elif mode == "thickness_local":
            _update_thickness_from_state()
            scalar_name = THICKNESS_SCALAR_NAME

            # 1) Garante que o array existe no mesh
            if scalar_name not in mesh.cell_data:
                print(f"[thickness_local] Array '{scalar_name}' não encontrado no mesh.")
                print("Arrays disponíveis:", list(mesh.cell_data.keys()))
                state["bg_actor"] = None
                state["main_actor"] = None
                return  # evita crash

            thr = 1e-6

            # 2) Faz o threshold com try/except para não matar a aplicação
            try:
                bg = mesh.threshold(thr, invert=True, scalars=scalar_name)
                main = mesh.threshold(thr, scalars=scalar_name)
            except ValueError as e:
                print(f"[thickness_local] Erro no threshold com '{scalar_name}': {e}")
                state["bg_actor"] = None
                state["main_actor"] = None
                return

            # 3) Adiciona os atores
            bg_actor = None
            if bg.n_cells > 0:
                bg_actor = plotter.add_mesh(
                    bg,
                    color=(0.8, 0.8, 0.8),
                    opacity=0.01,
                    show_edges=False,
                    name="bg",
                    reset_camera=False,
                )

            main_actor = plotter.add_mesh(
                main,
                scalars=scalar_name,
                cmap="plasma",
                clim=state.get("thickness_clim"),
                show_edges=True,
                name="main",
                reset_camera=False,
                scalar_bar_args={"title": THICKNESS_SCALAR_TITLE},
            )

            _update_thickness_mapper(main_actor, main)
            plotter.reset_camera_clipping_range()

            state["bg_actor"] = bg_actor
            state["main_actor"] = main_actor


        elif mode == "reservoir":
            # separa fundo e reservatório usando o array "Reservoir" (0/1)
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

            # reservatório colorido pela FÁCIES (mesma LUT do modo "facies")
            lut, rng = make_facies_lut()
            main_actor = plotter.add_mesh(
                main,
                scalars="Facies",   # <- usa facies para colorir
                opacity=1.0,
                show_edges=True,
                name="main",
                reset_camera=False,
            )
            main_actor.mapper.lookup_table = lut
            main_actor.mapper.scalar_range = rng

            plotter.reset_camera_clipping_range()
            state["bg_actor"] = bg_actor
            state["main_actor"] = main_actor

        elif mode == "largest":
            bg = mesh.threshold(0.5, invert=True, scalars="LargestCluster")
            main = mesh.threshold(0.5, scalars="LargestCluster")
            if bg.n_cells > 0:
                bg_actor = plotter.add_mesh(bg,
                                            color=(0.8, 0.8, 0.8), 
                                            opacity=0.02, 
                                            show_edges=False, 
                                            name="bg", 
                                            reset_camera=False
                                            )
            else: bg_actor = None
            main_actor = plotter.add_mesh(main, 
                                          color="lightcoral", 
                                          opacity=1.0, 
                                          show_edges=True, 
                                          name="main", 
                                          reset_camera=False
                                          )
            plotter.reset_camera_clipping_range()
            state["bg_actor"] = bg_actor
            state["main_actor"] = main_actor

        elif mode == "clusters":
            bg = mesh.threshold(0.5, invert=True, scalars="Clusters")
            main = mesh.threshold(0.5, scalars="Clusters")
            if bg.n_cells > 0:
                bg_actor = plotter.add_mesh(bg, 
                                            color=(0.8, 0.8, 0.8), 
                                            opacity=0.02, 
                                            show_edges=False, 
                                            name="bg", 
                                            reset_camera=False
                                            )
            else: bg_actor = None
            # Adiciona o mesh, mas NÃO passa cmap/clim aqui
            main_actor = plotter.add_mesh(main, 
                                          scalars="Clusters", 
                                          show_edges=True, 
                                          name="main", 
                                          reset_camera=False
                                          )
            
            _update_clusters_mapper(main_actor, main)
            plotter.reset_camera_clipping_range()

            state["bg_actor"] = bg_actor
            state["main_actor"] = main_actor
            plotter.remove_scalar_bar()
        
        elif mode == "ntg_local":
            main_actor = plotter.add_mesh(
                mesh, scalars="NTG_local", cmap="plasma", clim=[0.0, 1.0],
                show_edges=True, name="main", reset_camera=False,
                scalar_bar_args={"title": "NTG local"},
            )
            plotter.reset_camera_clipping_range()
            state["bg_actor"] = None
            state["main_actor"] = main_actor
        else:
            raise ValueError(f"Modo desconhecido: {mode}")

        if mode in ("thickness_local", "ntg_local"):
            if not SHOW_SCALAR_BAR and hasattr(plotter, "scalar_bars") and plotter.scalar_bars:
                plotter.remove_scalar_bar()
        else:
            if hasattr(plotter, "scalar_bars") and plotter.scalar_bars:
                plotter.remove_scalar_bar()


    # ---------- 5b. controle de camada: subir/descer k_min ----------
    def change_k(delta):
        kmin = state.get("k_min", 0)
        new = int(np.clip(kmin + delta, 0, N_LAYERS - 1))
        if new == kmin:
            return

        state["k_min"] = new

        # 1. Começa sempre do grid_base
        base = grid_base
        box = state["box_bounds"]
        base = grid_base.clip_box(box, invert=False, crinkle=True)
        base = attach_cell_data_from_original(base, grid_base)
        mesh = apply_k_filter(base)
        
        mode = state["mode"]

        # 4. Atualiza conforme o modo --------------------------

        if mode == "facies":
            if state["main_actor"]: _update_facies_mapper(state["main_actor"], mesh)

        elif mode == "thickness_local":
            thr = 1e-6
            bg = mesh.threshold(thr, invert=True, scalars=THICKNESS_SCALAR_NAME)
            main = mesh.threshold(thr, scalars=THICKNESS_SCALAR_NAME)
            if state["bg_actor"]: _update_simple_mapper(state["bg_actor"], bg)
            if state["main_actor"]: _update_thickness_mapper(state["main_actor"], main)
                

        elif mode == "reservoir":
            bg = mesh.threshold(0.5, invert=True, scalars="Reservoir")
            main = mesh.threshold(0.5, scalars="Reservoir")
            if state["bg_actor"]: _update_simple_mapper(state["bg_actor"], bg)
            if state["main_actor"]: _update_simple_mapper(state["main_actor"], main)

        elif mode == "largest":
            bg = mesh.threshold(0.5, invert=True, scalars="LargestCluster")
            main = mesh.threshold(0.5, scalars="LargestCluster")
            if state["bg_actor"]: _update_simple_mapper(state["bg_actor"], bg)
            if state["main_actor"]: _update_simple_mapper(state["main_actor"], main)

        elif mode == "clusters":
            bg = mesh.threshold(0.5, invert=True, scalars="Clusters")
            main = mesh.threshold(0.5, scalars="Clusters")
            if state["bg_actor"]: _update_simple_mapper(state["bg_actor"], bg)
            if state["main_actor"]: _update_clusters_mapper(state["main_actor"], main)

        elif mode == "ntg_local":
            if state["main_actor"]:
                mapper = state["main_actor"].mapper
                mapper.SetInputData(mesh)
                mapper.SetScalarModeToUseCellFieldData()
                mapper.SelectColorArray("NTG_local")
                mapper.scalar_range = [0.0, 1.0]
                mapper.SetScalarVisibility(True)
                mapper.Update()

        plotter.render()


    def _update_reservoir_from_state():
        """
        Recalcula Reservoir / Clusters / LargestCluster para o
        conjunto state['reservoir_facies'] e sincroniza em grid_base.
        Também atualiza a LUT e os tamanhos dos clusters no state.
        """
        rf = state.get("reservoir_facies")
        if not rf:
            return

        # recalcula no grid "global"
        res_arr = ensure_reservoir(rf)
        clusters_arr, largest_arr = ensure_clusters(rf)

        # copia os arrays para o grid_base usado na visualização
        grid_base.cell_data["Reservoir"] = res_arr
        grid_base.cell_data["Clusters"] = clusters_arr
        grid_base.cell_data["LargestCluster"] = largest_arr

        # atualiza LUT e tamanhos de clusters no state
        state["clusters_lut"], state["clusters_rng"] = make_clusters_lut(clusters_arr)
        state["clusters_sizes"] = compute_cluster_sizes(clusters_arr)
    

    _update_reservoir_from_state()



    # primeiro draw
    # show_mesh(grid_base)

     # atalhos: seta para cima/baixo controlam o índice de camada
    plotter.add_key_event("z",   lambda: change_k(-1))  # sobe (mostra mais topo)
    plotter.add_key_event("x", lambda: change_k(+1))  # desce (mostra mais base)


    # ---------- 6. callback do corte (suave) ----------
    def box_callback(box):
        state["box_bounds"] = box
        mode = state["mode"]

        base = grid_base
        base = grid_base.clip_box(box, invert=False, crinkle=True)
        base = attach_cell_data_from_original(base, grid_base)
        mesh = apply_k_filter(base)

        if mode == "facies":
            if state["main_actor"]: _update_facies_mapper(state["main_actor"], mesh)
        
        elif mode == "thickness_local":
            thr = 1e-6
            bg = mesh.threshold(thr, invert=True, scalars=THICKNESS_SCALAR_NAME)
            main = mesh.threshold(thr, scalars=THICKNESS_SCALAR_NAME)
            if state["bg_actor"]: _update_simple_mapper(state["bg_actor"], bg)
            if state["main_actor"]: _update_thickness_mapper(state["main_actor"], main)

        elif mode == "reservoir":
            bg = mesh.threshold(0.5, invert=True, scalars="Reservoir")
            main = mesh.threshold(0.5, scalars="Reservoir")
            if state["bg_actor"]: _update_simple_mapper(state["bg_actor"], bg)
            if state["main_actor"]: _update_simple_mapper(state["main_actor"], main)

        elif mode == "largest":
            bg = mesh.threshold(0.5, invert=True, scalars="LargestCluster")
            main = mesh.threshold(0.5, scalars="LargestCluster")
            if state["bg_actor"]: _update_simple_mapper(state["bg_actor"], bg)
            if state["main_actor"]: _update_simple_mapper(state["main_actor"], main)

        elif mode == "clusters":
            bg = mesh.threshold(0.5, invert=True, scalars="Clusters")
            main = mesh.threshold(0.5, scalars="Clusters")
            if state["bg_actor"]: _update_simple_mapper(state["bg_actor"], bg)
            if state["main_actor"]: _update_clusters_mapper(state["main_actor"], main) # Chama a função limpa

        elif mode == "ntg_local":
            if state["main_actor"]:
                mapper = state["main_actor"].mapper
                mapper.SetInputData(mesh)
                mapper.SetScalarModeToUseCellFieldData()
                mapper.SelectColorArray("NTG_local")
                mapper.scalar_range = [0.0, 1.0]
                mapper.SetScalarVisibility(True)
                mapper.Update()

    # Box widget
    box_widget = plotter.add_box_widget(
        callback=box_callback,
        bounds=grid_base.bounds,
        rotation_enabled=False,
        interaction_event="always",
    )

    state["box_widget"] = box_widget

    # Logo e ajustes visuais
    # logo = plotter.add_logo_widget("assets/forward_PNG.png", position=(0.02, 0.85), size=(0.12, 0.12))
    # logo.SetProcessEvents(False)
    plotter.set_background("white", top="lightgray")
    # plotter.enable_anti_aliasing('ssaa')

    box_widget.SetHandleSize(0.01)
    rep = box_widget.GetHandleProperty()
    rep.SetOpacity(0.1)
    shp = box_widget.GetSelectedHandleProperty()
    shp.SetOpacity(0.0)
    op = box_widget.GetOutlineProperty()
    op.SetOpacity(0.05)

    plotter.add_axes()
    change_k(0)

    # --- Primeiro draw CORRETO ---
    box = state["box_bounds"]
    base = grid_base.clip_box(box, invert=False, crinkle=True)
    base = attach_cell_data_from_original(base, grid_base)
    mesh = apply_k_filter(base)
    show_mesh(mesh)
    plotter.reset_camera()    # ESSENCIAL NO QT
    plotter.render()

    def _refresh():
        box = state["box_bounds"]
        base = grid_base.clip_box(box, invert=False, crinkle=True)
        base = attach_cell_data_from_original(base, grid_base)
        mesh = apply_k_filter(base)
        show_mesh(mesh)

    state["update_reservoir"] = _update_reservoir_from_state
    state["refresh"] = _refresh
    state["update_thickness"] = _update_thickness_from_state

    plotter.reset_camera_clipping_range()
    plotter.render()


    # plotter.show()






