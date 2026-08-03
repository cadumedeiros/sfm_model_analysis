"""Configuração da tabela mestre de descritores SFM.

As colunas de parâmetros de calibração não são mais fixadas neste arquivo.
Elas são preservadas dinamicamente a partir do dicionário ``parameters`` de
cada modelo, evitando que novas curvas ou multiplicadores sejam descartados.
"""

IDENTIFICATION_COLUMNS = [
    "simulation_id",
    "simulation_name",
    "study",
    "output_path",
    "of_value",
]

GROUPING_COLUMNS = [
    "simulation_number",
    "family",
    "cluster",
    "group_silhouette",
    "group_distance_to_centroid",
    "group_distance_percentile",
    "representative_role",
    "selected_for_visualization",
    "grouping_match",
]

DEFAULT_INACTIVE_FACIES = (0,)

GLOBAL_TARGET_DESCRIPTORS = [
    "active_cells",
    "active_grid_volume",
    "target_cells",
    "target_volume",
    "target_cell_fraction",
    "target_volume_fraction",
    "n_bodies",
    "largest_body_cells",
    "largest_body_volume",
    "connected_cell_fraction",
    "connected_volume_fraction",
    "effective_body_count",
    "body_volume_p50",
    "body_volume_p90",
    "percolation_x",
    "percolation_y",
    "percolation_z",
]

# Mapas 2D usados como núcleo da análise de descritores.
VERTICAL_MAP_DESCRIPTORS = [
    "Tcolumn",
    "Ttarget",
    "Tenv",
    "target_fraction_col",
    "n_packages",
    "Tpack_max",
    "gap_fraction_env",
]

# Mantidos disponíveis para análises complementares, sem entrar por padrão.
VERTICAL_MAP_DESCRIPTORS_OPTIONAL = [
    "target_fraction_env",
    "Cdom",
    "Tgap_max",
]

MAP_SUMMARY_STATS = ["mean", "std", "p10", "p50", "p90"]

VPC_SCALAR_DESCRIPTORS = [
    "vpc_distance_to_base",
]


def expanded_master_table_columns(
    parameter_columns=None,
    facies_ids=None,
    *,
    include_optional=False,
):
    """Retorna a ordem recomendada das colunas conhecidas da tabela mestre."""
    columns = list(IDENTIFICATION_COLUMNS)
    columns.extend(GROUPING_COLUMNS)
    columns.extend(list(parameter_columns or []))
    columns.extend(GLOBAL_TARGET_DESCRIPTORS)

    for facies_id in sorted(int(v) for v in (facies_ids or [])):
        columns.append(f"facies_{facies_id}_volume_fraction")

    map_names = list(VERTICAL_MAP_DESCRIPTORS)
    if include_optional:
        map_names.extend(VERTICAL_MAP_DESCRIPTORS_OPTIONAL)
    for descriptor in map_names:
        for stat in MAP_SUMMARY_STATS:
            columns.append(f"{descriptor}_{stat}")

    columns.extend(VPC_SCALAR_DESCRIPTORS)
    columns.extend(["processing_status", "processing_message"])
    return columns
