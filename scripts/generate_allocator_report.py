"""Generate a markdown report for allocator ablation results."""

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


SETTING_ORDER = ["B0", "B1", "B2", "B3", "B4", "G2", "G3"]


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    rows = _read_summary(args.summary_file)
    records_by_setting = _load_records_by_setting(args.input_dir)
    report = build_report(rows, records_by_setting)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(report, encoding="utf-8")
    print(args.output_file)
    return args.output_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate allocator ablation markdown report.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--summary_file", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    return parser.parse_args(argv)


def build_report(rows: list[dict[str, str]], records_by_setting: dict[str, list[dict[str, Any]]]) -> str:
    lines = ["# Allocator Ablation Analysis Report", ""]
    lines.extend(_overall_comparison_table(rows))
    lines.extend(["", "## Dataset-Level Comparison", ""])
    lines.extend(_dataset_comparison_table(rows))
    lines.extend(["", "## Step Interpretation", ""])
    lines.extend(_step_interpretation(rows))
    lines.extend(["", "## Diagnostics", ""])
    lines.extend(_diagnostics(rows))
    lines.extend(["", "## Likely Bottleneck", ""])
    lines.extend(_bottleneck(rows))
    lines.extend(["", "## Typical Failure Cases", ""])
    lines.extend(_failure_cases(records_by_setting))
    lines.extend(["", "## Recommendations", ""])
    lines.extend(_recommendations(rows))
    lines.append("")
    return "\n".join(lines)


def _overall_comparison_table(rows: list[dict[str, str]]) -> list[str]:
    by_setting = _aggregate_rows_by_setting(rows)
    output = [
        "## B0-B4 Comparison",
        "",
        "| Setting | Success Rate | Average Score | Allocator Error Rate | Avg Retrieved | Mask Density | Avg Final Memories/Agent | Cap Drop Ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for setting in SETTING_ORDER:
        row = by_setting.get(setting, {})
        output.append(
            "| {setting} | {success:.3f} | {score:.3f} | {error:.3f} | {retrieved:.2f} | {mask:.3f} | {selected:.2f} | {cap:.3f} |".format(
                setting=setting,
                success=_float(row.get("success_rate")),
                score=_float(row.get("average_score")),
                error=_float(row.get("allocator_error_rate")),
                retrieved=_float(row.get("avg_retrieved_memory_count")),
                mask=_float(row.get("mask_density")),
                selected=_float(row.get("avg_final_memory_count_per_agent")),
                cap=_float(row.get("allowed_cap_dropped_ratio")),
            )
        )
    return output


def _dataset_comparison_table(rows: list[dict[str, str]]) -> list[str]:
    output = [
        "| Dataset | Setting | Success Rate | Average Score | Allocator Error Rate | Avg Retrieved | Mask Density | Avg Final Memories/Agent | Cap Drop Ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            row.get("dataset", ""),
            SETTING_ORDER.index(row.get("setting", "")) if row.get("setting") in SETTING_ORDER else 99,
        ),
    )
    for row in sorted_rows:
        output.append(
            "| {dataset} | {setting} | {success:.3f} | {score:.3f} | {error:.3f} | {retrieved:.2f} | {mask:.3f} | {selected:.2f} | {cap:.3f} |".format(
                dataset=row.get("dataset") or row.get("group") or "",
                setting=row.get("setting") or "",
                success=_float(row.get("success_rate")),
                score=_float(row.get("average_score")),
                error=_float(row.get("allocator_error_rate")),
                retrieved=_float(row.get("avg_retrieved_memory_count")),
                mask=_float(row.get("mask_density")),
                selected=_float(row.get("avg_final_memory_count_per_agent")),
                cap=_float(row.get("allowed_cap_dropped_ratio")),
            )
        )
    return output


def _step_interpretation(rows: list[dict[str, str]]) -> list[str]:
    by_setting = _aggregate_rows_by_setting(rows)
    output: list[str] = []
    for left, right, message in (
        ("B0", "B1", "If B1 < B0, direct sharing may cause negative transfer or context pollution."),
        ("B1", "B2", "If B2 > B1, memory-agent masking is useful."),
        ("B2", "B3", "If B3 > B2, selection, budget, or redundancy is an important bottleneck."),
        ("B3", "B4", "If B4 > B3, realization form is an important bottleneck."),
        ("B2", "G2", "If G2 > B2, retrieval recall may be a bottleneck; if G2 < B2, global masking adds noise."),
        ("B3", "G3", "If G3 > B3, retrieval-limited selection missed useful memories; otherwise retrieval remains useful."),
        ("G2", "G3", "If G3 > G2, selection/budget after global masking is useful."),
    ):
        delta = _float(by_setting.get(right, {}).get("success_rate")) - _float(by_setting.get(left, {}).get("success_rate"))
        output.append(f"- {right} - {left}: success delta `{delta:.3f}`. {message}")
    return output


def _diagnostics(rows: list[dict[str, str]]) -> list[str]:
    by_setting = _aggregate_rows_by_setting(rows)
    output = []
    for setting in SETTING_ORDER:
        row = by_setting.get(setting)
        if not row:
            continue
        output.append(
            "- {setting}: retrieval={retrieved:.2f}, reject_ratio={reject:.3f}, all_zero={zero:.3f}, "
            "selected_allowed={ratio:.3f}, redundancy={redundancy:.3f}, avg_block_len={block:.1f}, avg_realized_len={realized:.1f}, cap_drop={cap:.3f}".format(
                setting=setting,
                retrieved=_float(row.get("avg_retrieved_memory_count")),
                reject=_float(row.get("masking_reject_ratio")),
                zero=_float(row.get("all_zero_memory_rate")),
                ratio=_float(row.get("selected_allowed_ratio")),
                redundancy=_float(row.get("simple_redundancy_rate")),
                block=_float(row.get("avg_memory_block_length")),
                realized=_float(row.get("avg_realized_memory_length")),
                cap=_float(row.get("allowed_cap_dropped_ratio")),
            )
        )
    return output


def _bottleneck(rows: list[dict[str, str]]) -> list[str]:
    by_setting = _aggregate_rows_by_setting(rows)
    deltas = []
    for left, right, name in (
        ("B0", "B1", "Retrieval/direct sharing"),
        ("B1", "B2", "Masking"),
        ("B2", "B3", "Selection"),
        ("B3", "B4", "Realization"),
        ("B2", "G2", "Global masking vs retrieval"),
        ("B3", "G3", "Global masking plus selection vs retrieval plus selection"),
    ):
        deltas.append((name, _float(by_setting.get(right, {}).get("success_rate")) - _float(by_setting.get(left, {}).get("success_rate"))))
    best_name, best_delta = max(deltas, key=lambda item: item[1])
    worst_name, worst_delta = min(deltas, key=lambda item: item[1])
    output = [f"- Largest observed improvement: `{best_name}` with delta `{best_delta:.3f}`."]
    if worst_delta < 0:
        output.append(f"- Largest degradation: `{worst_name}` with delta `{worst_delta:.3f}`, which is the first place to inspect for negative transfer.")
    if best_delta <= 0:
        output.append("- No stage improved success rate; retrieval quality, allocator JSON stability, or memory usefulness may be the current bottleneck.")
    return output


def _failure_cases(records_by_setting: dict[str, list[dict[str, Any]]]) -> list[str]:
    failures = []
    for setting in SETTING_ORDER:
        for record in records_by_setting.get(setting, []):
            result = record.get("task_result") or {}
            if result.get("success"):
                continue
            failures.append((setting, record))
            if len(failures) >= 5:
                break
        if len(failures) >= 5:
            break
    if not failures:
        return ["- No failed cases found in the available records."]
    output = []
    for setting, record in failures:
        result = record.get("task_result") or {}
        output.append(
            f"- `{setting}` target `{record.get('target_task_id')}` failed; "
            f"prediction=`{result.get('prediction')}`, allocator_errors={len(record.get('allocator_errors') or [])}, "
            f"retrieved={len(record.get('retrieved_memories') or [])}."
        )
    return output


def _recommendations(rows: list[dict[str, str]]) -> list[str]:
    by_setting = _aggregate_rows_by_setting(rows)
    output = [
        "- If B1 underperforms B0, reduce retrieval_top_k or improve retrieval before adding more routing logic.",
        "- If B2 improves B1 but allocator_error_rate is high, harden the mask JSON protocol or use a stronger allocator LLM only for masking.",
        "- If B3 improves B2, tune per-agent memory budget and add redundancy-aware selection.",
        "- If B4 improves B3, keep raw memories for evidence but realize concise agent-specific warnings for prompt injection.",
    ]
    if _float(by_setting.get("B1", {}).get("success_rate")) < _float(by_setting.get("B0", {}).get("success_rate")):
        output.append("- Direct sharing appears risky in this run; treat over-sharing/context pollution as a primary hypothesis.")
    if "G2" in by_setting or "G3" in by_setting:
        output.append("- Compare G2/G3 against B2/B3 to decide whether retrieval recall or global masking noise is the stronger bottleneck.")
    return output


def _aggregate_rows_by_setting(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    by_setting: dict[str, dict[str, str]] = {}
    for setting in SETTING_ORDER:
        matches = [row for row in rows if row.get("setting") == setting]
        total = sum(int(float(row.get("num_targets") or 0)) for row in matches)
        if not matches or total == 0:
            continue
        aggregate: dict[str, str] = {"setting": setting, "num_targets": str(total)}
        weighted_fields = [
            "success_rate",
            "average_score",
            "allocator_error_rate",
            "avg_retrieved_memory_count",
            "masking_reject_ratio",
            "mask_density",
            "all_zero_memory_rate",
            "selected_allowed_ratio",
            "simple_redundancy_rate",
            "avg_memory_block_length",
            "avg_realized_memory_length",
            "avg_final_memory_count_per_agent",
            "allowed_cap_dropped_ratio",
            "avg_allowed_before_cap",
            "avg_allowed_after_cap",
        ]
        for field in weighted_fields:
            value = sum(
                _float(row.get(field)) * int(float(row.get("num_targets") or 0))
                for row in matches
            ) / total
            aggregate[field] = str(value)
        by_setting[setting] = aggregate
    return by_setting


def _load_records_by_setting(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    for manifest_path in input_dir.rglob("manifest.json"):
        run_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        setting = str(manifest.get("setting"))
        path = run_dir / "allocation_records.jsonl"
        if not path.exists():
            continue
        records.setdefault(setting, []).extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    return records


def _read_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
