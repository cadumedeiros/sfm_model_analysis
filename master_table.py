"""Extração reproduzível da tabela mestre e das VPCs dos modelos SFM.

O módulo é independente da interface Qt. Cada registro de entrada pode conter
um grid já carregado ou um ``output_path`` para um arquivo GRDECL.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from master_table_config import (
    DEFAULT_INACTIVE_FACIES,
    GROUPING_COLUMNS,
    IDENTIFICATION_COLUMNS,
    MAP_SUMMARY_STATS,
    VERTICAL_MAP_DESCRIPTORS,
    VERTICAL_MAP_DESCRIPTORS_OPTIONAL,
    expanded_master_table_columns,
)


class MasterTableExtractionCancelled(RuntimeError):
    """Interrupção solicitada pelo consumidor durante a extração em lote."""


GROUPING_ID_CANDIDATES = (
    "Simulation_ID",
    "simulation_id",
    "Simulation ID",
    "model_id",
    "Model_ID",
    "Simulation",
)

GROUPING_COLUMN_ALIASES = {
    "simulation": "simulation_number",
    "family": "family",
    "cluster": "cluster",
    "silhouette": "group_silhouette",
    "distance_to_centroid": "group_distance_to_centroid",
    "distance_percentile": "group_distance_percentile",
    "representative_role": "representative_role",
    "role": "representative_role",
    "selected_for_visualization": "selected_for_visualization",
    "of_value": "of_value",
}


def _column_key(value):
    """Normaliza um cabeçalho somente para localizar aliases conhecidos."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def normalize_simulation_id(value):
    """Cria uma chave estável para associar IDs como Sim549, sim_549 e 549."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (int, np.integer)):
        return f"sim{int(value)}"
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return f"sim{int(value)}"

    text = str(value).strip()
    if not text:
        return None
    stem = os.path.splitext(os.path.basename(text))[0]
    if stem.lower() in {"base", "modelo_base", "model_base"}:
        return "base"

    match = re.search(
        r"(?i)(?:^|[^a-z0-9])sim(?:ulation)?[\s_-]*(\d+)(?:$|[^0-9])",
        stem,
    )
    if match:
        return f"sim{int(match.group(1))}"
    if re.fullmatch(r"\d+(?:\.0+)?", stem):
        return f"sim{int(float(stem))}"
    return stem.casefold()


def _find_grouping_id_column(frame):
    for candidate in GROUPING_ID_CANDIDATES:
        if candidate in frame.columns:
            return candidate
    normalized = {_column_key(column): column for column in frame.columns}
    for candidate in GROUPING_ID_CANDIDATES:
        found = normalized.get(_column_key(candidate))
        if found is not None:
            return found
    raise ValueError(
        "A tabela de agrupamento precisa conter Simulation_ID, simulation_id "
        "ou outra coluna de identificação equivalente."
    )


def load_grouping_table(source):
    """Lê o CSV/Excel do agrupamento ou copia um DataFrame já carregado."""
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if source is None:
        return None

    path = os.path.abspath(os.fspath(source))
    extension = os.path.splitext(path)[1].lower()
    if extension in {".csv", ".txt"}:
        return pd.read_csv(path)
    if extension not in {".xlsx", ".xls", ".xlsm"}:
        raise ValueError("A tabela de agrupamento deve ser CSV ou Excel.")

    workbook = pd.ExcelFile(path)
    preferred = (
        "model_families",
        "Model_Families",
        "04_Modelos",
        "05_Representantes",
        "Models",
    )
    sheet_names = list(dict.fromkeys([*preferred, *workbook.sheet_names]))
    errors = []
    for sheet_name in sheet_names:
        if sheet_name not in workbook.sheet_names:
            continue
        last_error = None
        for header_row in range(0, 8):
            frame = pd.read_excel(
                workbook,
                sheet_name=sheet_name,
                header=header_row,
            )
            try:
                _find_grouping_id_column(frame)
                return frame
            except ValueError as exc:
                last_error = exc
        errors.append(f"{sheet_name}: {last_error}")
    raise ValueError(
        "Nenhuma aba do arquivo de agrupamento possui uma coluna de identificação. "
        + " | ".join(errors)
    )


def _prepare_grouping_table(source):
    frame = load_grouping_table(source)
    if frame is None:
        return None
    frame = frame.copy()
    id_column = _find_grouping_id_column(frame)
    frame["_join_id"] = frame[id_column].map(normalize_simulation_id)
    frame = frame.loc[frame["_join_id"].notna()].copy()

    rename = {}
    for column in frame.columns:
        key = _column_key(column)
        if column == id_column:
            continue
        if key in GROUPING_COLUMN_ALIASES:
            rename[column] = GROUPING_COLUMN_ALIASES[key]
    frame = frame.rename(columns=rename)
    frame = frame.drop(columns=[id_column], errors="ignore")
    data_columns = [
        column
        for column in frame.columns
        if column != "_join_id"
    ]
    if data_columns:
        frame = frame.loc[frame[data_columns].notna().any(axis=1)].copy()

    if frame["_join_id"].duplicated().any():
        duplicates = sorted(
            frame.loc[frame["_join_id"].duplicated(keep=False), "_join_id"]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            "A tabela de agrupamento possui Simulation_ID duplicados: "
            + ", ".join(duplicates[:10])
        )
    return frame


def merge_grouping_results(master_df, vpc_df, log_df, grouping_source):
    """Associa famílias aos descritores e às VPCs por Simulation_ID."""
    grouping = _prepare_grouping_table(grouping_source)
    if grouping is None:
        return master_df, vpc_df, log_df

    master = master_df.copy()
    master["_join_id"] = master["simulation_id"].map(normalize_simulation_id)
    if master["_join_id"].duplicated().any():
        raise ValueError("A tabela mestre possui simulation_id duplicados.")

    grouping_ids = set(grouping["_join_id"])
    master = master.merge(
        grouping,
        on="_join_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "__grouping"),
    )

    grouping_suffix = "__grouping"
    for column in list(master.columns):
        if not column.endswith(grouping_suffix):
            continue
        original = column[: -len(grouping_suffix)]
        if original in master.columns:
            missing = master[original].isna()
            master.loc[missing, original] = master.loc[missing, column]
            master = master.drop(columns=[column])
        else:
            master = master.rename(columns={column: original})

    master["grouping_match"] = master["_join_id"].isin(grouping_ids)
    matched_ids = set(master.loc[master["grouping_match"], "_join_id"])
    unused_grouping_ids = grouping_ids.difference(matched_ids)
    unmatched_nonbase = int(
        ((~master["grouping_match"]) & (master["_join_id"] != "base")).sum()
    )
    master = master.drop(columns=["_join_id"])

    front = [
        column
        for column in [*IDENTIFICATION_COLUMNS, *GROUPING_COLUMNS]
        if column in master.columns
    ]
    remainder = [column for column in master.columns if column not in front]
    master = master.reindex(columns=front + remainder)

    vpc = vpc_df.copy()
    if not vpc.empty:
        compact_columns = [
            column
            for column in (
                "simulation_id",
                "family",
                "cluster",
                "representative_role",
                "selected_for_visualization",
                "grouping_match",
            )
            if column in master.columns
        ]
        model_grouping = master[compact_columns].copy()
        model_grouping["_join_id"] = model_grouping["simulation_id"].map(
            normalize_simulation_id
        )
        model_grouping = model_grouping.drop(columns=["simulation_id"])
        vpc["_join_id"] = vpc["model_id"].map(normalize_simulation_id)
        vpc = vpc.merge(
            model_grouping,
            on="_join_id",
            how="left",
            validate="many_to_one",
        ).drop(columns=["_join_id"])
        grouping_columns = [
            column
            for column in (
                "family",
                "cluster",
                "representative_role",
                "selected_for_visualization",
                "grouping_match",
            )
            if column in vpc.columns
        ]
        leading = [
            column
            for column in ("model_id", "model_name", *grouping_columns)
            if column in vpc.columns
        ]
        vpc = vpc.reindex(
            columns=leading + [column for column in vpc.columns if column not in leading]
        )

    log = log_df.copy()
    if not log.empty:
        log["_join_id"] = log["simulation_id"].map(normalize_simulation_id)
        log["grouping_status"] = np.where(
            log["_join_id"].isin(grouping_ids),
            "matched",
            "unmatched",
        )
        family_by_id = (
            master.assign(
                _join_id=master["simulation_id"].map(normalize_simulation_id)
            )
            .set_index("_join_id")
            .get("family")
        )
        if family_by_id is not None:
            log["family"] = log["_join_id"].map(family_by_id)
        log = log.drop(columns=["_join_id"])

    summary = {
        "master_models": int(len(master)),
        "matched_models": int(master["grouping_match"].sum()),
        "unmatched_master_models": int((~master["grouping_match"]).sum()),
        "unmatched_nonbase_models": unmatched_nonbase,
        "grouping_rows": int(len(grouping)),
        "unused_grouping_models": int(len(unused_grouping_ids)),
    }
    master.attrs["grouping_summary"] = summary
    vpc.attrs["grouping_summary"] = summary
    log.attrs["grouping_summary"] = summary
    return master, vpc, log


def _first_present(record, keys, default=None):
    for key in keys:
        if key not in record:
            continue
        value = record[key]
        if value is None:
            continue
        try:
            missing = pd.isna(value)
            if np.isscalar(missing) and bool(missing):
                continue
        except (TypeError, ValueError):
            pass
        return value
    return default


def _resolve_record_data(record):
    grid = record.get("grid")
    facies = record.get("facies")
    shape = record.get("grid_shape")
    output_path = _first_present(
        record,
        ("output_path", "OutputPath", "Output Path", "grdecl_path"),
    )

    if shape is None and all(record.get(key) is not None for key in ("nx", "ny", "nz")):
        shape = (int(record["nx"]), int(record["ny"]), int(record["nz"]))

    if grid is None or facies is None:
        if not output_path:
            raise ValueError("O registro precisa conter grid/facies ou output_path.")
        from load_data import load_grid_from_grdecl, read_specgrid

        grid, facies = load_grid_from_grdecl(
            str(output_path),
            load_all_properties=False,
            verbose=False,
        )
        shape = read_specgrid(str(output_path))

    return grid, np.asarray(facies, dtype=int).ravel(), shape, output_path


def _extract_parameters(record, parameter_columns=None):
    parameters = dict(record.get("parameters") or {})
    for key in parameter_columns or []:
        if key in record and key not in parameters:
            parameters[key] = record[key]
    return parameters


def complete_vpc_facies(vpc_df, facies_ids=None):
    """Inclui proporções zero para fácies ausentes em uma camada ativa."""
    if vpc_df is None or vpc_df.empty:
        return pd.DataFrame() if vpc_df is None else vpc_df

    if facies_ids is None:
        facies_ids = sorted(
            int(value)
            for value in vpc_df["facies"].dropna().unique()
        )
    else:
        facies_ids = sorted({int(value) for value in facies_ids})

    layer_meta_columns = [
        "model_id",
        "model_name",
        "layer_k",
        "stratigraphic_coordinate",
        "layer_mean_z",
        "active_cells",
        "active_volume",
        "candidate_cells",
        "excluded_cells",
    ]

    layer_meta = (
        vpc_df[layer_meta_columns]
        .drop_duplicates(["model_id", "layer_k"])
    )

    value_columns = [
        "model_id",
        "layer_k",
        "facies",
        "vpc_proportion",
        "facies_cells",
        "volume_proportion",
        "facies_volume",
    ]

    completed = []

    for model_id, meta in layer_meta.groupby(
        "model_id",
        sort=False,
        dropna=False,
    ):
        product = pd.MultiIndex.from_product(
            [
                meta["layer_k"].astype(int).tolist(),
                facies_ids,
            ],
            names=["layer_k", "facies"],
        ).to_frame(index=False)

        product["model_id"] = model_id

        product = product.merge(
            meta,
            on=["model_id", "layer_k"],
            how="left",
        )

        values = vpc_df.loc[
            vpc_df["model_id"].astype(str) == str(model_id),
            value_columns,
        ]

        product = product.merge(
            values,
            on=["model_id", "layer_k", "facies"],
            how="left",
        )

        has_cells = product["active_cells"].fillna(0) > 0
        has_volume = product["active_volume"].fillna(0.0) > 0.0

        product.loc[has_cells, "vpc_proportion"] = (
            product.loc[has_cells, "vpc_proportion"].fillna(0.0)
        )

        product.loc[has_volume, "volume_proportion"] = (
            product.loc[has_volume, "volume_proportion"].fillna(0.0)
        )

        product["facies_cells"] = (
            product["facies_cells"].fillna(0).astype(int)
        )

        product["facies_volume"] = (
            product["facies_volume"].fillna(0.0)
        )

        completed.append(product)

    columns = [
        "model_id",
        "model_name",
        "layer_k",
        "stratigraphic_coordinate",
        "layer_mean_z",
        "facies",
        "vpc_proportion",
        "facies_cells",
        "active_cells",
        "volume_proportion",
        "facies_volume",
        "active_volume",
        "candidate_cells",
        "excluded_cells",
    ]

    return (
        pd.concat(completed, ignore_index=True)
        .reindex(columns=columns)
    )


def complete_master_facies_fractions(master_df, facies_ids=None):
    """Converte ausências reais de uma fácies em proporção volumétrica zero.

    Linhas com erro permanecem como ``NaN`` para não confundir falha de
    processamento com ausência geológica da fácies.
    """
    if master_df is None or master_df.empty:
        return pd.DataFrame() if master_df is None else master_df

    frame = master_df.copy()
    if facies_ids is None:
        columns = [
            column
            for column in frame.columns
            if re.fullmatch(r"facies_-?\d+_volume_fraction", str(column))
        ]
    else:
        columns = [
            f"facies_{int(facies_id)}_volume_fraction"
            for facies_id in sorted({int(v) for v in facies_ids})
        ]
        for column in columns:
            if column not in frame.columns:
                frame[column] = np.nan

    if "processing_status" in frame.columns:
        valid_rows = frame["processing_status"].eq("ok")
    else:
        valid_rows = pd.Series(True, index=frame.index)

    for column in columns:
        missing = valid_rows & frame[column].isna()
        frame.loc[missing, column] = 0.0
    return frame


# Alias privado mantido para compatibilidade com código anterior.
_complete_vpc_facies = complete_vpc_facies


def extract_master_tables(
    model_records: Iterable[Mapping],
    target_facies,
    *,
    base_model_id=None,
    inactive_facies=DEFAULT_INACTIVE_FACIES,
    min_body_volume=0.0,
    connectivity=1,
    include_optional_maps=False,
    use_filtered_vertical=False,
    thin_lamination_threshold=0.30,
    parameter_columns=None,
    progress_callback=None,
):
    """Calcula a tabela mestre, a VPC longa e o log de processamento.

    Cada item de ``model_records`` aceita as chaves:

    - ``simulation_id``, ``simulation_name``, ``output_path`` e ``of_value``;
    - ``parameters``: dicionário com os parâmetros de calibração;
    - ``grid``, ``facies`` e ``grid_shape`` para dados já carregados.
    """
    from analysis import (
        compute_global_metrics_for_array,
        compute_vertical_descriptor_maps,
        compute_vpc,
        compute_vpc_distance,
        generate_detailed_metrics_df,
        summarize_2d_map,
    )

    if isinstance(model_records, pd.DataFrame):
        records = model_records.to_dict(orient="records")
    else:
        records = [dict(record) for record in model_records]
    master_rows = []
    vpc_tables = []
    log_rows = []
    parameter_names = set(parameter_columns or [])

    for index, record in enumerate(records):
        simulation_id = _first_present(
            record,
            ("simulation_id", "Simulation_ID", "Simulation ID", "model_id"),
            default=f"model_{index:04d}",
        )
        simulation_name = _first_present(
            record,
            ("simulation_name", "model_name", "Simulation", "Model", "name"),
            default=str(simulation_id),
        )
        study = _first_present(record, ("study", "Study", "study_name"))
        output_path = _first_present(
            record,
            ("output_path", "OutputPath", "Output Path", "grdecl_path"),
        )
        of_value = _first_present(
            record,
            ("of_value", "OF Value", "objective_function", "objective_value", "OF"),
            default=np.nan,
        )
        parameters = _extract_parameters(record, parameter_columns=parameter_columns)
        parameter_names.update(parameters.keys())
        row = {
            "simulation_id": simulation_id,
            "simulation_name": simulation_name,
            "study": study,
            "output_path": output_path,
            "of_value": of_value,
            **parameters,
        }

        if progress_callback is not None:
            keep_going = progress_callback(
                index,
                len(records),
                simulation_id,
                "processing",
            )
            if keep_going is False:
                raise MasterTableExtractionCancelled("Extração cancelada pelo usuário.")

        try:
            grid, facies, shape, resolved_path = _resolve_record_data(record)
            if resolved_path is not None:
                row["output_path"] = resolved_path

            metrics, percolation = compute_global_metrics_for_array(
                facies,
                target_facies,
                target_grid=grid,
                grid_shape=shape,
                inactive_facies=inactive_facies,
                min_body_volume=min_body_volume,
                connectivity=connectivity,
            )
            for key in (
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
            ):
                row[key] = metrics.get(key, np.nan)
            row["percolation_x"] = bool(percolation["x_perc"])
            row["percolation_y"] = bool(percolation["y_perc"])
            row["percolation_z"] = bool(percolation["z_perc"])

            facies_table = generate_detailed_metrics_df(
                facies,
                target_grid=grid,
                grid_shape=shape,
                inactive_facies=inactive_facies,
                min_body_volume=min_body_volume,
                connectivity=connectivity,
            )
            for _, facies_row in facies_table.iterrows():
                facies_id = int(facies_row["facies"])
                row[f"facies_{facies_id}_volume_fraction"] = float(facies_row["volume_fraction"])

            vertical_maps = compute_vertical_descriptor_maps(
                facies,
                target_facies,
                target_grid=grid,
                grid_shape=shape,
                inactive_facies=inactive_facies,
                use_filtered=use_filtered_vertical,
                thin_lamination_threshold=thin_lamination_threshold,
            )
            selected_maps = list(VERTICAL_MAP_DESCRIPTORS)
            if include_optional_maps:
                selected_maps.extend(VERTICAL_MAP_DESCRIPTORS_OPTIONAL)
            for map_name in selected_maps:
                summary = summarize_2d_map(vertical_maps[map_name], prefix=map_name)
                for stat_name in MAP_SUMMARY_STATS:
                    row[f"{map_name}_{stat_name}"] = summary[f"{map_name}_{stat_name}"]

            vpc = compute_vpc(
                facies,
                target_grid=grid,
                grid_shape=shape,
                inactive_facies=inactive_facies,
                model_id=simulation_id,
                model_name=simulation_name,
            )
            vpc_tables.append(vpc)
            row["processing_status"] = "ok"
            row["processing_message"] = ""
            log_rows.append({
                "simulation_id": simulation_id,
                "status": "ok",
                "message": "",
            })
        except Exception as exc:
            row["processing_status"] = "error"
            row["processing_message"] = str(exc)
            log_rows.append({
                "simulation_id": simulation_id,
                "status": "error",
                "message": str(exc),
            })

        master_rows.append(row)
        if progress_callback is not None:
            keep_going = progress_callback(
                index + 1,
                len(records),
                simulation_id,
                row["processing_status"],
            )
            if keep_going is False and index + 1 < len(records):
                raise MasterTableExtractionCancelled("Extração cancelada pelo usuário.")

    master_df = pd.DataFrame(master_rows)
    vpc_df = pd.concat(vpc_tables, ignore_index=True) if vpc_tables else pd.DataFrame()
    vpc_df = complete_vpc_facies(vpc_df)
    log_df = pd.DataFrame(log_rows)

    if base_model_id is None:
        for record, row in zip(records, master_rows):
            is_base = _first_present(record, ("is_base", "IsBase"), default=False)
            if bool(is_base) or str(row["simulation_id"]).lower() == "base":
                base_model_id = row["simulation_id"]
                break

    master_df["vpc_distance_to_base"] = np.nan
    if base_model_id is not None and not vpc_df.empty:
        reference = vpc_df.loc[vpc_df["model_id"].astype(str) == str(base_model_id)]
        if not reference.empty:
            for idx, model_id in master_df["simulation_id"].items():
                current = vpc_df.loc[vpc_df["model_id"].astype(str) == str(model_id)]
                if not current.empty:
                    master_df.at[idx, "vpc_distance_to_base"] = compute_vpc_distance(current, reference)

    facies_ids = []
    if not vpc_df.empty:
        facies_ids = sorted(int(v) for v in vpc_df["facies"].dropna().unique())
    master_df = complete_master_facies_fractions(master_df, facies_ids)
    ordered = expanded_master_table_columns(
        parameter_columns=sorted(parameter_names),
        facies_ids=facies_ids,
        include_optional=include_optional_maps,
    )
    ordered_present = [column for column in ordered if column in master_df.columns]
    extras = [column for column in master_df.columns if column not in ordered_present]
    master_df = master_df.reindex(columns=ordered_present + extras)
    return master_df, vpc_df, log_df


def export_master_tables(
    output_path,
    model_records,
    target_facies,
    *,
    grouping_table=None,
    **kwargs,
):
    """Calcula, integra o agrupamento e grava o Excel com três abas."""
    master_df, vpc_df, log_df = extract_master_tables(
        model_records,
        target_facies,
        **kwargs,
    )
    if grouping_table is not None:
        master_df, vpc_df, log_df = merge_grouping_results(
            master_df,
            vpc_df,
            log_df,
            grouping_table,
        )
    output_path = os.path.abspath(os.fspath(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        master_df.to_excel(writer, sheet_name="Master_Table", index=False)
        vpc_df.to_excel(writer, sheet_name="VPC_Long", index=False)
        log_df.to_excel(writer, sheet_name="Processing_Log", index=False)
    return output_path, master_df, vpc_df, log_df
