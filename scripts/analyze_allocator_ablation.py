"""Analyze allocator ablation outputs and write a CSV summary."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from allocator.diagnostics import compute_diagnostics, paired_comparison  # noqa: E402


SETTING_ORDER = ["B0", "B1", "B2", "B3", "B4", "G2", "G3"]


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    run_dirs = _find_run_dirs(args.input_dir)
    if not run_dirs:
        raise SystemExit(f"No allocator run directories found under {args.input_dir}")
    _validate_target_consistency(run_dirs)
    _validate_seed_leakage(run_dirs)
    _validate_namespaced_memory_ids(run_dirs)

    rows = []
    records_by_group: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for run_dir in run_dirs:
        manifest = _read_json(run_dir / "manifest.json")
        records = _read_jsonl(run_dir / "allocation_records.jsonl")
        diagnostics = _read_json(run_dir / "diagnostics.json") if (run_dir / "diagnostics.json").exists() else compute_diagnostics(records)
        setting = manifest.get("setting")
        group = str(run_dir.parent)
        records_by_group.setdefault(group, {})[setting] = records
        rows.append(_summary_row(run_dir, manifest, records, diagnostics))

    pair_stats: dict[str, dict[str, dict[str, int]]] = {
        group: paired_comparison(setting_records)
        for group, setting_records in records_by_group.items()
    }
    for row in rows:
        group = str(Path(row["run_dir"]).parent)
        setting = row["setting"]
        previous = _previous_setting(setting)
        if previous:
            stats = pair_stats.get(group, {}).get(f"{setting}_vs_{previous}", {})
            row["paired_vs_previous"] = previous
            row["paired_shared"] = stats.get("shared", 0)
            row["paired_improved"] = stats.get("improved", 0)
            row["paired_worse"] = stats.get("worse", 0)
            row["paired_unchanged"] = stats.get("unchanged", 0)
        else:
            row["paired_vs_previous"] = ""
            row["paired_shared"] = 0
            row["paired_improved"] = 0
            row["paired_worse"] = 0
            row["paired_unchanged"] = 0

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with args.output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["group"], SETTING_ORDER.index(item["setting"]) if item["setting"] in SETTING_ORDER else 99)):
            writer.writerow(row)
    print(args.output_file)
    return args.output_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze allocator ablation outputs.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    return parser.parse_args(argv)


def _summary_row(run_dir: Path, manifest: dict[str, Any], records: list[dict[str, Any]], diagnostics: dict[str, Any]) -> dict[str, Any]:
    retrieval = diagnostics.get("retrieval", {})
    masking = diagnostics.get("masking", {})
    selection = diagnostics.get("selection", {})
    realization = diagnostics.get("realization", {})
    allowed_cap = diagnostics.get("allowed_cap", {})
    task = diagnostics.get("task", {})
    return {
        "group": str(run_dir.parent),
        "run_dir": str(run_dir),
        "setting": manifest.get("setting", run_dir.name),
        "dataset": manifest.get("dataset") or _infer_dataset(records),
        "num_targets": len(records),
        "success_rate": task.get("success_rate", 0.0),
        "average_score": task.get("average_score", 0.0),
        "allocator_error_rate": diagnostics.get("allocator_error_rate", 0.0),
        "avg_retrieved_memory_count": retrieval.get("avg_retrieved_memory_count", 0.0),
        "masking_reject_ratio": retrieval.get("masking_reject_ratio", 0.0),
        "mask_density": masking.get("mask_density", 0.0),
        "all_zero_memory_rate": masking.get("all_zero_memory_rate", 0.0),
        "all_one_memory_rate": masking.get("all_one_memory_rate", 0.0),
        "avg_agents_per_memory": masking.get("avg_agents_per_memory", 0.0),
        "avg_allowed_memories_per_agent": masking.get("avg_allowed_memories_per_agent", 0.0),
        "avg_final_memory_count_per_agent": selection.get("avg_final_memory_count_per_agent", 0.0),
        "selected_allowed_ratio": selection.get("selected_allowed_ratio", 0.0),
        "simple_redundancy_rate": selection.get("simple_redundancy_rate", 0.0),
        "avg_memory_block_length": selection.get("avg_memory_block_length", 0.0),
        "avg_realized_memory_length": realization.get("avg_realized_memory_length", 0.0),
        "allowed_cap_dropped_count": allowed_cap.get("allowed_cap_dropped_count", 0),
        "allowed_cap_dropped_ratio": allowed_cap.get("allowed_cap_dropped_ratio", 0.0),
        "avg_allowed_before_cap": allowed_cap.get("avg_allowed_before_cap", 0.0),
        "avg_allowed_after_cap": allowed_cap.get("avg_allowed_after_cap", 0.0),
        "failure_type_distribution": json.dumps(task.get("failure_type_distribution", {}), ensure_ascii=False),
        "reason_tag_counts": json.dumps(masking.get("reason_tag_counts", {}), ensure_ascii=False),
        "realization_type_counts": json.dumps(realization.get("realization_type_counts", {}), ensure_ascii=False),
    }


def _find_run_dirs(input_dir: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in input_dir.rglob("manifest.json")
        if (path.parent / "allocation_records.jsonl").exists()
    )


def _validate_target_consistency(run_dirs: list[Path]) -> None:
    groups: dict[Path, dict[str, list[str]]] = {}
    for run_dir in run_dirs:
        manifest = _read_json(run_dir / "manifest.json")
        groups.setdefault(run_dir.parent, {})[str(manifest.get("setting"))] = list(manifest.get("target_example_ids") or [])
    for group, ids_by_setting in groups.items():
        if len(ids_by_setting) < 2:
            continue
        first_setting, first_ids = next(iter(ids_by_setting.items()))
        for setting, ids in ids_by_setting.items():
            if ids != first_ids:
                raise SystemExit(
                    f"Target example ids differ in {group}: {setting} does not match {first_setting}."
                )


def _validate_seed_leakage(run_dirs: list[Path]) -> None:
    for run_dir in run_dirs:
        manifest = _read_json(run_dir / "manifest.json")
        target_ids = set(manifest.get("target_example_ids") or [])
        seed_ids = set()
        for seed_run in manifest.get("memory_source_seed_runs") or []:
            examples_path = Path(seed_run) / "examples.jsonl"
            if not examples_path.exists():
                continue
            for row in _read_jsonl(examples_path):
                if row.get("example_id"):
                    seed_ids.add(str(row["example_id"]))
        overlap = target_ids & seed_ids
        if overlap:
            raise SystemExit(f"Memory leakage detected in {run_dir}: target ids overlap seed ids: {sorted(overlap)[:5]}")


def _validate_namespaced_memory_ids(run_dirs: list[Path]) -> None:
    for run_dir in run_dirs:
        for record in _read_jsonl(run_dir / "allocation_records.jsonl"):
            for memory in record.get("retrieved_memories") or []:
                memory_id = str(memory.get("memory_id") or "")
                if memory_id and ":" not in memory_id:
                    raise SystemExit(f"Un-namespaced memory id in {run_dir}: {memory_id}")


def _previous_setting(setting: str) -> str | None:
    if setting == "G2":
        return "B2"
    if setting == "G3":
        return "B3"
    if setting not in SETTING_ORDER:
        return None
    index = SETTING_ORDER.index(setting)
    if index == 0:
        return None
    return SETTING_ORDER[index - 1]


def _infer_dataset(records: list[dict[str, Any]]) -> str:
    for record in records:
        result = record.get("task_result") or {}
        target = result.get("target") or {}
        if "normalized_answer" in target:
            return "strategyqa"
    return ""


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
