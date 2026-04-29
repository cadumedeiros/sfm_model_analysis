
"""Starter configuration for the SFM master table.

This file separates:
1) input columns from calibration
2) target descriptors to extract from each model
3) summary statistics to save for map-based descriptors

Edit freely as your software evolves.
"""

PARAMETER_COLUMNS = [
    "Carbo_GrainsProdvsTime",
    "Carbo_MudProdvsTime",
    "Carbo_RudProdvsTime",
    "LutitesProdvsTime",
    "S1Supply0",
    "S2Supply0",
]

IDENTIFICATION_COLUMNS = [
    "Simulation_ID",
    "Simulation",
    "OutputPath",
    "OF Value",
]

# Core target-based descriptors (single facies or facies group)
GLOBAL_TARGET_DESCRIPTORS = [
    "target_volume",
    "target_fraction_global",
    "n_clusters",
    "connected_fraction",
    "percolation_x",
    "percolation_y",
    "percolation_z",
]

# Column / map descriptors to summarize with statistics
VERTICAL_MAP_DESCRIPTORS = [
    "Ttot",
    "target_fraction_col",
    "n_packages",
    "Tpack_max",
    "ICV_env",
    "Tgap_sum",
]

# Good second-layer descriptors
VERTICAL_MAP_DESCRIPTORS_OPTIONAL = [
    "Tenv",
    "Tgap_max",
    "target_fraction_env",
    "Qv",
    "Qv_abs",
]

# Whole facies-model descriptors
FACIES_MODEL_DESCRIPTORS = [
    "facies_entropy_global",
    "facies_diversity_global",
    "n_facies_per_column_mean",
]

# Local validation descriptors
WELL_DESCRIPTORS = [
    "well_score_mean",
    "well_score_min",
    "well_score_std",
]

# Default summary statistics for map-like descriptors
MAP_SUMMARY_STATS = ["mean", "std", "p10", "p50", "p90"]

def expanded_master_table_columns():
    cols = []
    cols.extend(IDENTIFICATION_COLUMNS)
    cols.extend(PARAMETER_COLUMNS)
    cols.extend(GLOBAL_TARGET_DESCRIPTORS)
    cols.extend(FACIES_MODEL_DESCRIPTORS)
    cols.extend(WELL_DESCRIPTORS)
    for name in VERTICAL_MAP_DESCRIPTORS:
        for stat in MAP_SUMMARY_STATS:
            cols.append(f"{name}_{stat}")
    return cols
