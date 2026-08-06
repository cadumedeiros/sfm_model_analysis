"""Extração em lote de descritores e VPCs sem abrir a interface gráfica.

O programa lê um manifesto produzido pelo agrupamento, associa cada
``Simulation_ID`` a um arquivo GRDECL, processa um grid por vez e grava
checkpoints por modelo. Uma execução interrompida pode ser retomada usando o
mesmo diretório de saída.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


# Impede que analysis.py mantenha o grid padrão carregado durante o lote.
os.environ["GRID_VIEW_ANALYSIS_SKIP_DEFAULT_GRID"] = "1"

from master_table import (  # noqa: E402
    complete_master_facies_fractions,
    complete_vpc_facies,
    extract_master_tables,
    merge_grouping_results,
    normalize_simulation_id,
)


BATCH_FORMAT_VERSION = 1
DEFAULT_GRID_GLOB = "*.grdecl"

ID_COLUMN_CANDIDATES = (
    "Simulation_ID",
    "simulation_id",
    "Simulation ID",
    "model_id",
    "Model_ID",
)
SIMULATION_COLUMN_CANDIDATES = (
    "Simulation",
    "simulation",
    "simulation_number",
)
OF_COLUMN_CANDIDATES = (
    "OF Value",
    "of_value",
    "OF",
    "objective_function",
    "objective_value",
)
PATH_COLUMN_CANDIDATES = (
    "output_path",
    "OutputPath",
    "Output Path",
    "grdecl_path",
    "grid_path",
)

NON_PARAMETER_COLUMN_KEYS = {
    "simulation_id",
    "simulation",
    "simulation_number",
    "model_id",
    "of",
    "of_value",
    "objective_function",
    "objective_value",
    "family",
    "cluster",
    "silhouette",
    "distance_to_centroid",
    "distance_percentile",
    "group_silhouette",
    "group_distance_to_centroid",
    "group_distance_percentile",
    "representative_role",
    "role",
    "selected_for_visualization",
    "expected_grid_token",
    "output_path",
    "grdecl_path",
    "grid_path",
    "is_base",
    "study",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _column_key(value) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def _find_column(frame: pd.DataFrame, candidates) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    normalized = {_column_key(column): column for column in frame.columns}
    for candidate in candidates:
        found = normalized.get(_column_key(candidate))
        if found is not None:
            return found
    return None


def load_batch_manifest(source) -> pd.DataFrame:
    """Lê e valida o manifesto CSV/Excel do conjunto aceito."""
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Manifesto não encontrado: {path}")

    if path.suffix.lower() in {".csv", ".txt"}:
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
        workbook = pd.ExcelFile(path)
        preferred = (
            "batch_extraction_manifest",
            "model_families",
            "04_Modelos",
            "Models",
            "Planilha1",
        )
        frame = None
        for sheet_name in (*preferred, *workbook.sheet_names):
            if sheet_name not in workbook.sheet_names:
                continue
            candidate = pd.read_excel(workbook, sheet_name=sheet_name)
            if (
                _find_column(candidate, ID_COLUMN_CANDIDATES) is not None
                or _find_column(
                    candidate,
                    SIMULATION_COLUMN_CANDIDATES,
                ) is not None
            ):
                frame = candidate
                break
        if frame is None:
            raise ValueError(
                "Nenhuma aba do manifesto contém Simulation_ID ou Simulation."
            )
    else:
        raise ValueError("O manifesto deve ser CSV ou Excel.")

    frame = frame.copy()
    id_column = _find_column(frame, ID_COLUMN_CANDIDATES)
    simulation_column = _find_column(
        frame,
        SIMULATION_COLUMN_CANDIDATES,
    )
    if id_column is None and simulation_column is None:
        raise ValueError(
            "O manifesto precisa conter Simulation_ID ou Simulation."
        )
    if id_column is None:
        frame["Simulation_ID"] = frame[simulation_column].map(
            normalize_simulation_id
        )
        id_column = "Simulation_ID"

    frame["_batch_id"] = frame[id_column].map(normalize_simulation_id)
    if frame["_batch_id"].isna().any():
        bad_rows = (frame.index[frame["_batch_id"].isna()] + 2).tolist()
        raise ValueError(
            "Há identificadores vazios ou inválidos nas linhas: "
            + ", ".join(map(str, bad_rows[:20]))
        )
    if frame["_batch_id"].duplicated().any():
        duplicates = sorted(
            frame.loc[
                frame["_batch_id"].duplicated(keep=False),
                "_batch_id",
            ].unique()
        )
        raise ValueError(
            "O manifesto possui Simulation_ID duplicados: "
            + ", ".join(duplicates[:20])
        )
    if (frame["_batch_id"] == "base").any():
        raise ValueError(
            "O modelo-base não deve estar no manifesto; use --base-grid."
        )
    return frame.reset_index(drop=True)


def infer_parameter_columns(frame: pd.DataFrame) -> list[str]:
    """Identifica parâmetros numéricos sem fixar nomes da calibração."""
    parameters = []
    for column in frame.columns:
        key = _column_key(column)
        if key.startswith("_") or key in NON_PARAMETER_COLUMN_KEYS:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() == frame[column].notna().sum() and values.notna().any():
            parameters.append(column)
    return parameters


def simulation_id_from_grid_path(path) -> str | None:
    """Extrai ``simNNN`` ou ``base`` de nomes usuais de arquivos GRDECL."""
    stem = Path(path).stem
    if re.search(r"(?i)base[\s_-]*model|model[\s_-]*base", stem):
        return "base"
    identifier = normalize_simulation_id(stem)
    if identifier and re.fullmatch(r"sim\d+", identifier):
        return identifier
    return None


def canonical_simulation_id(value) -> str:
    normalized = normalize_simulation_id(value)
    if normalized == "base":
        return "base"
    match = re.fullmatch(r"sim(\d+)", normalized or "")
    if match:
        return f"Sim{int(match.group(1))}"
    return str(value)


def index_grid_files(
    grids_dir,
    *,
    grid_glob: str = DEFAULT_GRID_GLOB,
) -> dict[str, list[Path]]:
    """Indexa recursivamente os GRDECLs por identificador de simulação."""
    directory = Path(grids_dir).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(
            f"A pasta de grids não foi encontrada: {directory}"
        )
    index: dict[str, list[Path]] = {}
    for path in sorted(directory.rglob(grid_glob)):
        if not path.is_file():
            continue
        identifier = simulation_id_from_grid_path(path)
        if identifier is None:
            continue
        index.setdefault(identifier, []).append(path.resolve())
    return index


def _resolve_explicit_path(row, grids_dir: Path) -> Path | None:
    for column in PATH_COLUMN_CANDIDATES:
        if column not in row:
            continue
        value = row[column]
        if value is None or bool(pd.isna(value)):
            continue
        candidate = Path(str(value)).expanduser()
        candidates = [candidate]
        if not candidate.is_absolute():
            candidates.insert(0, grids_dir / candidate)
        for path in candidates:
            if path.is_file() and path.suffix.lower() == ".grdecl":
                return path.resolve()
    return None


def resolve_model_records(
    manifest: pd.DataFrame,
    grids_dir,
    *,
    base_grid=None,
    grid_glob: str = DEFAULT_GRID_GLOB,
    allow_missing: bool = False,
) -> tuple[list[dict], pd.DataFrame]:
    """Associa todas as linhas do manifesto a arquivos GRDECL inequívocos."""
    directory = Path(grids_dir).expanduser().resolve()
    grid_index = index_grid_files(directory, grid_glob=grid_glob)

    if base_grid is None:
        base_candidates = grid_index.get("base", [])
        if len(base_candidates) != 1:
            raise ValueError(
                "Informe --base-grid ou mantenha exatamente um arquivo "
                "BaseModel na pasta de grids."
            )
        base_path = base_candidates[0]
    else:
        base_path = Path(base_grid).expanduser().resolve()
        if not base_path.is_file():
            raise FileNotFoundError(
                f"Grid do modelo-base não encontrado: {base_path}"
            )

    records = [
        {
            "simulation_id": "base",
            "simulation_name": "Modelo-base",
            "output_path": str(base_path),
            "is_base": True,
        }
    ]
    resolution_rows = [
        {
            "simulation_id": "base",
            "status": "resolved",
            "output_path": str(base_path),
            "message": "",
        }
    ]
    missing = []
    ambiguous = []

    for _, row in manifest.iterrows():
        identifier = row["_batch_id"]
        display_id = canonical_simulation_id(identifier)
        explicit = _resolve_explicit_path(row, directory)
        candidates = [explicit] if explicit is not None else grid_index.get(
            identifier,
            [],
        )
        candidates = sorted({path.resolve() for path in candidates})
        message = ""
        if len(candidates) == 1:
            resolved_path = candidates[0]
            status = "resolved"
        elif not candidates:
            resolved_path = None
            status = "missing"
            message = "Nenhum GRDECL correspondente foi encontrado."
            missing.append(identifier)
        else:
            resolved_path = None
            status = "ambiguous"
            message = "Mais de um GRDECL corresponde ao mesmo ID: " + " | ".join(
                str(path) for path in candidates
            )
            ambiguous.append(identifier)

        record = row.drop(labels=["_batch_id"]).to_dict()
        record["simulation_id"] = display_id
        record["simulation_name"] = display_id
        record["output_path"] = (
            str(resolved_path) if resolved_path is not None else None
        )
        records.append(record)
        resolution_rows.append(
            {
                "simulation_id": display_id,
                "status": status,
                "output_path": record["output_path"],
                "message": message,
            }
        )

    if ambiguous:
        raise ValueError(
            "Há IDs associados a múltiplos grids: "
            + ", ".join(ambiguous[:20])
        )
    if missing and not allow_missing:
        raise FileNotFoundError(
            f"{len(missing)} modelos do manifesto não possuem grid. "
            "Primeiros IDs: "
            + ", ".join(missing[:20])
        )
    return records, pd.DataFrame(resolution_rows)


def _parse_id_filter(values: str | None) -> set[str] | None:
    if not values:
        return None
    identifiers = {
        normalize_simulation_id(value)
        for value in re.split(r"[,;\s]+", values)
        if value.strip()
    }
    identifiers.discard(None)
    identifiers.discard("base")
    return identifiers


def select_manifest_models(
    manifest: pd.DataFrame,
    *,
    only_ids: Iterable[str] | None = None,
    max_models: int | None = None,
) -> pd.DataFrame:
    selected = manifest.copy()
    if only_ids is not None:
        normalized = {
            normalize_simulation_id(value)
            for value in only_ids
        }
        selected = selected.loc[selected["_batch_id"].isin(normalized)]
        missing = normalized.difference(set(selected["_batch_id"]))
        if missing:
            raise ValueError(
                "IDs solicitados não estão no manifesto: "
                + ", ".join(sorted(missing))
            )
    if max_models is not None:
        if max_models < 1:
            raise ValueError("--max-models precisa ser pelo menos 1.")
        selected = selected.head(max_models)
    if selected.empty:
        raise ValueError("Nenhum modelo foi selecionado para o lote.")
    return selected.reset_index(drop=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_fingerprint(
    manifest_path: Path,
    records: list[dict],
    *,
    target_facies,
    extraction_options: dict,
) -> str:
    payload = {
        "format_version": BATCH_FORMAT_VERSION,
        "manifest_sha256": _file_sha256(manifest_path),
        "records": [
            {
                "simulation_id": record["simulation_id"],
                "output_path": record.get("output_path"),
            }
            for record in records
        ],
        "target_facies": sorted(int(value) for value in target_facies),
        "extraction_options": extraction_options,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _shard_name(simulation_id) -> str:
    normalized = normalize_simulation_id(simulation_id) or str(simulation_id)
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", normalized)


def _load_status(status_path: Path) -> dict | None:
    if not status_path.exists():
        return None
    return json.loads(status_path.read_text(encoding="utf-8"))


def _basic_error_tables(record: dict, message: str, elapsed: float):
    simulation_id = record["simulation_id"]
    master = pd.DataFrame(
        [
            {
                "simulation_id": simulation_id,
                "simulation_name": record.get(
                    "simulation_name",
                    simulation_id,
                ),
                "output_path": record.get("output_path"),
                "processing_status": "error",
                "processing_message": message,
                "processing_seconds": elapsed,
            }
        ]
    )
    log = pd.DataFrame(
        [
            {
                "simulation_id": simulation_id,
                "status": "error",
                "message": message,
                "processing_seconds": elapsed,
            }
        ]
    )
    return master, pd.DataFrame(), log


def _process_one_record(
    record: dict,
    *,
    target_facies,
    parameter_columns,
    extraction_options,
    extractor: Callable,
):
    started_at = _utc_now()
    start = time.perf_counter()
    try:
        master, vpc, log = extractor(
            [record],
            target_facies,
            base_model_id=None,
            parameter_columns=parameter_columns,
            **extraction_options,
        )
        elapsed = time.perf_counter() - start
        if master.empty:
            raise RuntimeError("O extrator não devolveu uma linha de resultado.")
        master["processing_seconds"] = elapsed
        if log.empty:
            log = pd.DataFrame(
                [
                    {
                        "simulation_id": record["simulation_id"],
                        "status": master.iloc[0].get(
                            "processing_status",
                            "error",
                        ),
                        "message": master.iloc[0].get(
                            "processing_message",
                            "",
                        ),
                    }
                ]
            )
        log["processing_seconds"] = elapsed
    except Exception as exc:
        elapsed = time.perf_counter() - start
        master, vpc, log = _basic_error_tables(
            record,
            str(exc),
            elapsed,
        )

    finished_at = _utc_now()
    log["output_path"] = record.get("output_path")
    log["started_at_utc"] = started_at
    log["finished_at_utc"] = finished_at
    status = str(master.iloc[0].get("processing_status", "error"))
    message = str(master.iloc[0].get("processing_message", ""))
    return master, vpc, log, {
        "simulation_id": record["simulation_id"],
        "status": status,
        "message": message,
        "processing_seconds": float(elapsed),
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
    }


def _read_shards(records: list[dict], shards_dir: Path):
    master_frames = []
    vpc_frames = []
    log_frames = []
    for record in records:
        directory = shards_dir / _shard_name(record["simulation_id"])
        master_path = directory / "master.csv"
        vpc_path = directory / "vpc.csv"
        log_path = directory / "log.csv"
        if master_path.exists():
            master_frames.append(pd.read_csv(master_path))
        if vpc_path.exists() and vpc_path.stat().st_size > 0:
            try:
                vpc_frames.append(pd.read_csv(vpc_path))
            except pd.errors.EmptyDataError:
                pass
        if log_path.exists():
            log_frames.append(pd.read_csv(log_path))
    master = (
        pd.concat(master_frames, ignore_index=True, sort=False)
        if master_frames
        else pd.DataFrame()
    )
    vpc = (
        pd.concat(vpc_frames, ignore_index=True, sort=False)
        if vpc_frames
        else pd.DataFrame()
    )
    log = (
        pd.concat(log_frames, ignore_index=True, sort=False)
        if log_frames
        else pd.DataFrame()
    )
    return master, vpc, log


def _add_vpc_distance_to_base(
    master: pd.DataFrame,
    vpc: pd.DataFrame,
) -> pd.DataFrame:
    frame = master.copy()
    frame["vpc_distance_to_base"] = np.nan
    if frame.empty or vpc.empty:
        return frame

    values = vpc.copy()
    values["_batch_id"] = values["model_id"].map(normalize_simulation_id)

    # Permite consolidar tanto resultados novos quanto tabelas antigas.
    proportion_column = (
        "vpc_proportion"
        if "vpc_proportion" in values.columns
        else "area_proportion"
    )

    matrix = values.pivot_table(
        index="_batch_id",
        columns=["layer_k", "facies"],
        values=proportion_column,
        aggfunc="mean",
    ).fillna(0.0)

    if "base" not in matrix.index:
        return frame
    differences = matrix.subtract(matrix.loc["base"], axis="columns")
    distances = np.sqrt((differences ** 2).mean(axis=1))
    frame["vpc_distance_to_base"] = frame["simulation_id"].map(
        lambda value: distances.get(
            normalize_simulation_id(value),
            np.nan,
        )
    )
    return frame


def consolidate_batch_results(
    records: list[dict],
    manifest: pd.DataFrame,
    *,
    output_dir: Path,
    shards_dir: Path,
    resolution_log: pd.DataFrame,
    write_excel: bool = True,
) -> dict[str, Path]:
    """Consolida os checkpoints sem recalcular nenhum grid."""
    master, vpc, log = _read_shards(records, shards_dir)
    if master.empty:
        raise RuntimeError("Nenhum resultado foi encontrado para consolidar.")

    if not vpc.empty:
        facies_ids = sorted(
            int(value) for value in vpc["facies"].dropna().unique()
        )
        vpc = complete_vpc_facies(vpc, facies_ids)
    else:
        facies_ids = []
    master = complete_master_facies_fractions(master, facies_ids)
    master = _add_vpc_distance_to_base(master, vpc)
    grouping = manifest.drop(columns=["_batch_id"], errors="ignore")
    master, vpc, log = merge_grouping_results(
        master,
        vpc,
        log,
        grouping,
    )

    order = {
        normalize_simulation_id(record["simulation_id"]): position
        for position, record in enumerate(records)
    }
    master["_batch_order"] = master["simulation_id"].map(
        lambda value: order.get(normalize_simulation_id(value), len(order))
    )
    master = master.sort_values("_batch_order").drop(
        columns=["_batch_order"]
    ).reset_index(drop=True)
    if not log.empty:
        log["_batch_order"] = log["simulation_id"].map(
            lambda value: order.get(
                normalize_simulation_id(value),
                len(order),
            )
        )
        log = log.sort_values("_batch_order").drop(
            columns=["_batch_order"]
        ).reset_index(drop=True)

    master_path = output_dir / "Master_Table.csv"
    vpc_path = output_dir / "VPC_Long.csv"
    log_path = output_dir / "Processing_Log.csv"
    resolution_path = output_dir / "Grid_Resolution_Log.csv"
    _atomic_csv(master, master_path)
    _atomic_csv(vpc, vpc_path)
    _atomic_csv(log, log_path)
    _atomic_csv(resolution_log, resolution_path)

    outputs = {
        "master_table": master_path,
        "vpc_long": vpc_path,
        "processing_log": log_path,
        "resolution_log": resolution_path,
    }
    if write_excel:
        excel_path = output_dir / "tabela_mestre_descritores_lote.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            master.to_excel(writer, sheet_name="Master_Table", index=False)
            vpc.to_excel(writer, sheet_name="VPC_Long", index=False)
            log.to_excel(writer, sheet_name="Processing_Log", index=False)
            resolution_log.to_excel(
                writer,
                sheet_name="Grid_Resolution",
                index=False,
            )
        outputs["excel"] = excel_path
    return outputs


def run_batch_extraction(
    manifest_path,
    grids_dir,
    base_grid,
    target_facies,
    output_dir,
    *,
    grid_glob: str = DEFAULT_GRID_GLOB,
    only_ids: Iterable[str] | None = None,
    max_models: int | None = None,
    allow_missing: bool = False,
    retry_errors: bool = False,
    write_excel: bool = True,
    extractor: Callable = extract_master_tables,
    **extraction_options,
) -> dict:
    """Executa ou retoma o lote e devolve um resumo reproduzível."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "_model_checkpoints"
    shards_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "extraction_checkpoint.json"

    manifest = load_batch_manifest(manifest_path)
    manifest = select_manifest_models(
        manifest,
        only_ids=only_ids,
        max_models=max_models,
    )
    parameter_columns = infer_parameter_columns(manifest)
    records, resolution_log = resolve_model_records(
        manifest,
        grids_dir,
        base_grid=base_grid,
        grid_glob=grid_glob,
        allow_missing=allow_missing,
    )

    target_facies = tuple(sorted({int(value) for value in target_facies}))
    if not target_facies:
        raise ValueError("Informe pelo menos uma fácies-alvo.")
    fingerprint = _run_fingerprint(
        manifest_path,
        records,
        target_facies=target_facies,
        extraction_options=extraction_options,
    )
    checkpoint = {
        "format_version": BATCH_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "manifest_path": str(manifest_path),
        "grids_dir": str(Path(grids_dir).expanduser().resolve()),
        "base_grid": str(Path(base_grid).expanduser().resolve()),
        "target_facies": list(target_facies),
        "planned_models": len(records),
        "completed_models": 0,
        "ok_models": 0,
        "error_models": 0,
        "last_model": None,
    }
    if checkpoint_path.exists():
        previous = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != fingerprint:
            raise RuntimeError(
                "O diretório de saída contém um lote com configuração "
                "diferente. Use outro diretório para preservar os resultados."
            )
        checkpoint.update(previous)
    else:
        _atomic_json(checkpoint_path, checkpoint)

    for position, record in enumerate(records, start=1):
        identifier = record["simulation_id"]
        shard_dir = shards_dir / _shard_name(identifier)
        status_path = shard_dir / "status.json"
        previous_status = _load_status(status_path)
        if previous_status is not None:
            previous_value = previous_status.get("status")
            if previous_value == "ok":
                print(
                    f"[{position}/{len(records)}] {identifier}: "
                    "checkpoint existente"
                )
                continue
            if previous_value == "error" and not retry_errors:
                print(
                    f"[{position}/{len(records)}] {identifier}: "
                    "erro anterior preservado"
                )
                continue

        print(f"[{position}/{len(records)}] {identifier}: processando")
        shard_dir.mkdir(parents=True, exist_ok=True)
        master, vpc, log, status = _process_one_record(
            record,
            target_facies=target_facies,
            parameter_columns=parameter_columns,
            extraction_options=extraction_options,
            extractor=extractor,
        )
        _atomic_csv(master, shard_dir / "master.csv")
        _atomic_csv(vpc, shard_dir / "vpc.csv")
        _atomic_csv(log, shard_dir / "log.csv")
        _atomic_json(status_path, status)

        statuses = [
            _load_status(
                shards_dir
                / _shard_name(item["simulation_id"])
                / "status.json"
            )
            for item in records
        ]
        statuses = [item for item in statuses if item is not None]
        checkpoint.update(
            {
                "updated_at_utc": _utc_now(),
                "completed_models": len(statuses),
                "ok_models": sum(
                    item.get("status") == "ok" for item in statuses
                ),
                "error_models": sum(
                    item.get("status") != "ok" for item in statuses
                ),
                "last_model": identifier,
            }
        )
        _atomic_json(checkpoint_path, checkpoint)
        print(
            f"[{position}/{len(records)}] {identifier}: "
            f"{status['status']} em {status['processing_seconds']:.2f} s"
        )
        del master, vpc, log
        gc.collect()

    outputs = consolidate_batch_results(
        records,
        manifest,
        output_dir=output_dir,
        shards_dir=shards_dir,
        resolution_log=resolution_log,
        write_excel=write_excel,
    )
    statuses = [
        _load_status(
            shards_dir
            / _shard_name(record["simulation_id"])
            / "status.json"
        )
        for record in records
    ]
    statuses = [item for item in statuses if item is not None]
    summary = {
        "finished_at_utc": _utc_now(),
        "planned_models": len(records),
        "completed_models": len(statuses),
        "ok_models": sum(
            item.get("status") == "ok" for item in statuses
        ),
        "error_models": sum(
            item.get("status") != "ok" for item in statuses
        ),
        "parameter_columns": parameter_columns,
        "target_facies": list(target_facies),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    _atomic_json(output_dir / "run_summary.json", summary)
    checkpoint.update(summary)
    checkpoint["updated_at_utc"] = _utc_now()
    _atomic_json(checkpoint_path, checkpoint)
    return summary


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = {
        int(token)
        for token in re.split(r"[,;\s]+", value)
        if token.strip()
    }
    return tuple(sorted(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extrai descritores e VPCs de muitos GRDECLs sem abrir a "
            "interface do Grid View Analysis."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--grids-dir", required=True)
    parser.add_argument("--base-grid", required=True)
    parser.add_argument("--target-facies", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--grid-glob", default=DEFAULT_GRID_GLOB)
    parser.add_argument(
        "--only",
        help="IDs separados por vírgula, por exemplo Sim549,Sim1469.",
    )
    parser.add_argument("--max-models", type=int)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--no-excel", action="store_true")
    parser.add_argument("--include-optional-maps", action="store_true")
    parser.add_argument("--use-filtered-vertical", action="store_true")
    parser.add_argument("--thin-lamination-threshold", type=float, default=0.30)
    parser.add_argument("--min-body-volume", type=float, default=0.0)
    parser.add_argument("--connectivity", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--inactive-facies", default="0")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    only_ids = _parse_id_filter(args.only)
    target_facies = _parse_int_list(args.target_facies)
    inactive_facies = _parse_int_list(args.inactive_facies)
    summary = run_batch_extraction(
        args.manifest,
        args.grids_dir,
        args.base_grid,
        target_facies,
        args.output_dir,
        grid_glob=args.grid_glob,
        only_ids=only_ids,
        max_models=args.max_models,
        allow_missing=args.allow_missing,
        retry_errors=args.retry_errors,
        write_excel=not args.no_excel,
        inactive_facies=inactive_facies,
        min_body_volume=args.min_body_volume,
        connectivity=args.connectivity,
        include_optional_maps=args.include_optional_maps,
        use_filtered_vertical=args.use_filtered_vertical,
        thin_lamination_threshold=args.thin_lamination_threshold,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["error_models"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
