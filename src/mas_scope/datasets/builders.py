"""Split-aware dataset builders for no-memory MAS baselines."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from mas_scope.core.registry import registry
from mas_scope.core.types import TaskExample
from mas_scope.datasets.validators import normalize_yes_no


def _read_json_or_jsonl(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "examples", "tasks", "records"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    raise ValueError(f"Unsupported dataset payload in {path}")


def _first_present(raw: dict, keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in raw and raw[key] not in (None, ""):
            return raw[key]
    return default


class DatasetBuilder(ABC):
    dataset_name: str
    task_type: str
    task_class: str
    environment_class: str | None = None

    def __init__(self, data_path: str | Path, **kwargs) -> None:
        self.data_path = Path(data_path)
        self.options = kwargs

    @abstractmethod
    def build(self, split: str, limit: int | None = None) -> list[TaskExample]:
        ...

    def available_splits(self) -> list[str]:
        if self.data_path.is_dir():
            splits = [path.name for path in self.data_path.iterdir() if path.is_dir()]
            return sorted(splits) if splits else ["default"]
        return ["train", "dev", "validation", "test"]

    def _limit(self, examples: list[TaskExample], limit: int | None) -> list[TaskExample]:
        return examples[:limit] if limit is not None else examples

    def _metadata(self, source_path: Path, split: str, **extra) -> dict:
        return {
            "task_class": self.task_class,
            "environment": self.environment_class,
            "source_path": str(source_path),
            "split": split,
            **{key: value for key, value in extra.items() if value is not None},
        }


@registry.register_dataset_builder("hotpotqa")
class HotpotQABuilder(DatasetBuilder):
    dataset_name = "hotpotqa"
    task_type = "qa"
    task_class = "qa"

    def build(self, split: str, limit: int | None = None) -> list[TaskExample]:
        source = self._resolve_split_file(split)
        records = _read_json_or_jsonl(source)
        examples = [self._normalize_record(raw, split, source, index) for index, raw in enumerate(records)]
        return self._limit(examples, limit)

    def _resolve_split_file(self, split: str) -> Path:
        if self.data_path.is_file():
            return self.data_path
        candidates = {
            "train": ["hotpot_train_v1.1.json", "train.json", "hotpotqa_train.json"],
            "dev": ["hotpot_dev_distractor_v1.json", "dev.json", "validation.json"],
            "validation": ["hotpot_dev_distractor_v1.json", "validation.json", "dev.json"],
            "test": ["hotpot_test_fullwiki_v1.json", "test.json"],
        }.get(split, [f"{split}.json", f"{split}.jsonl"])
        for candidate in candidates:
            path = self.data_path / candidate
            if path.exists():
                return path
        raise FileNotFoundError(f"No HotpotQA file found for split '{split}' under {self.data_path}")

    def _normalize_record(self, raw: dict, split: str, source: Path, index: int) -> TaskExample:
        question = raw.get("question")
        if not question:
            raise ValueError(f"HotpotQA record {index} is missing question")
        example_id = str(raw.get("_id") or raw.get("id") or f"hotpotqa_{split}_{index}")
        supporting_facts = self._normalize_supporting_facts(raw.get("supporting_facts"))
        context = self._normalize_context(raw.get("context"))
        target = {}
        if "answer" in raw:
            target["answer"] = raw.get("answer")
        if supporting_facts is not None:
            target["supporting_facts"] = supporting_facts
        return TaskExample(
            example_id=example_id,
            dataset_name=self.dataset_name,
            task_type=self.task_type,
            split=split,
            input={"question": question, "context": context, "supporting_facts": supporting_facts},
            target=target,
            metadata=self._metadata(
                source,
                split,
                raw_id=example_id,
                level=raw.get("level"),
                question_type=raw.get("type"),
                benchmark_format="hotpotqa",
            ),
        )

    def _normalize_context(self, context):
        if isinstance(context, dict) and "title" in context and "sentences" in context:
            return [[title, sentences] for title, sentences in zip(context["title"], context["sentences"])]
        return context

    def _normalize_supporting_facts(self, supporting_facts):
        if isinstance(supporting_facts, dict) and "title" in supporting_facts and "sent_id" in supporting_facts:
            return [[title, sent_id] for title, sent_id in zip(supporting_facts["title"], supporting_facts["sent_id"])]
        return supporting_facts


@registry.register_dataset_builder("strategyqa")
class StrategyQABuilder(DatasetBuilder):
    dataset_name = "strategyqa"
    task_type = "qa"
    task_class = "qa"

    def build(self, split: str, limit: int | None = None) -> list[TaskExample]:
        source = self._resolve_split_file(split)
        records = _read_json_or_jsonl(source)
        examples = [self._normalize_record(raw, split, source, index) for index, raw in enumerate(records)]
        return self._limit(examples, limit)

    def _resolve_split_file(self, split: str) -> Path:
        if self.data_path.is_file():
            return self.data_path
        candidates = [f"{split}.json", f"{split}.jsonl", f"strategyqa_{split}.json", f"strategyqa_{split}.jsonl"]
        if split == "validation":
            candidates.extend(["dev.json", "dev.jsonl"])
        for candidate in candidates:
            path = self.data_path / candidate
            if path.exists():
                return path
        raise FileNotFoundError(f"No StrategyQA file found for split '{split}' under {self.data_path}")

    def _normalize_record(self, raw: dict, split: str, source: Path, index: int) -> TaskExample:
        question = raw.get("question")
        if not question:
            raise ValueError(f"StrategyQA record {index} is missing question")
        example_id = str(raw.get("qid") or raw.get("id") or f"strategyqa_{split}_{index}")
        normalized_answer = normalize_yes_no(raw.get("answer")) if "answer" in raw else None
        target = {}
        if "answer" in raw:
            target = {"answer": raw.get("answer"), "normalized_answer": normalized_answer}
        return TaskExample(
            example_id=example_id,
            dataset_name=self.dataset_name,
            task_type=self.task_type,
            split=split,
            input={
                "question": question,
                "facts": raw.get("facts"),
                "decomposition": raw.get("decomposition"),
                "evidence": raw.get("evidence") or raw.get("paragraphs"),
            },
            target=target,
            metadata=self._metadata(source, split, raw_id=example_id, term=raw.get("term"), benchmark_format="strategyqa"),
        )


@registry.register_dataset_builder("pddl")
class PDDLBuilder(DatasetBuilder):
    dataset_name = "pddl"
    task_type = "formal_planning"
    task_class = "formal_planning"

    def build(self, split: str, limit: int | None = None) -> list[TaskExample]:
        if self.data_path.is_file():
            records = _read_json_or_jsonl(self.data_path)
            examples = [self._normalize_json_record(raw, split, self.data_path, index) for index, raw in enumerate(records)]
            return self._limit(examples, limit)
        examples = self._build_from_directory(split)
        return self._limit(examples, limit)

    def available_splits(self) -> list[str]:
        if self.data_path.is_dir():
            if (self.data_path / "instances").exists() or (self.data_path / "domain.pddl").exists():
                return ["default"]
        return super().available_splits()

    def _normalize_json_record(self, raw: dict, split: str, source: Path, index: int) -> TaskExample:
        domain_pddl = raw.get("domain_pddl")
        problem_pddl = raw.get("problem_pddl")
        if not domain_pddl or not problem_pddl:
            raise ValueError(f"PDDL record {index} requires domain_pddl and problem_pddl")
        goal = raw.get("goal") or raw.get("instruction")
        return TaskExample(
            example_id=str(raw.get("id") or f"pddl_{split}_{index}"),
            dataset_name=self.dataset_name,
            task_type=self.task_type,
            split=split,
            input={"instruction": raw.get("instruction") or goal or "Generate a valid plan.", "domain_pddl": domain_pddl, "problem_pddl": problem_pddl},
            target={"reference_plan": raw.get("reference_plan"), "goal": goal},
            metadata=self._metadata(source, split, raw_id=raw.get("id"), benchmark_format="pddl-json"),
        )

    def _build_from_directory(self, split: str) -> list[TaskExample]:
        root = self.data_path / split if (self.data_path / split).is_dir() else self.data_path
        instances_dir = root / "instances"
        problem_paths = sorted(instances_dir.glob("*.pddl")) if instances_dir.exists() else sorted(root.glob("problem*.pddl"))
        domain_default = root / "domain.pddl"
        examples: list[TaskExample] = []
        for index, problem_path in enumerate(problem_paths):
            domain_path = self._domain_for_problem(root, domain_default, problem_path)
            problem_pddl = problem_path.read_text(encoding="utf-8")
            domain_pddl = domain_path.read_text(encoding="utf-8")
            plan_path = root / "plans" / f"{problem_path.stem}.plan"
            reference_plan = plan_path.read_text(encoding="utf-8").splitlines() if plan_path.exists() else None
            examples.append(
                TaskExample(
                    example_id=f"{root.name}_{problem_path.stem}",
                    dataset_name=self.dataset_name,
                    task_type=self.task_type,
                    split=split,
                    input={
                        "instruction": f"Generate a plan for {problem_path.name}.",
                        "domain_pddl": domain_pddl,
                        "problem_pddl": problem_pddl,
                    },
                    target={"reference_plan": reference_plan, "goal": self._extract_goal(problem_pddl)},
                    metadata=self._metadata(
                        problem_path,
                        split,
                        domain_path=str(domain_path),
                        problem_path=str(problem_path),
                        domain_name=root.name,
                        benchmark_format="ipc-directory",
                    ),
                )
            )
        return examples

    def _domain_for_problem(self, root: Path, domain_default: Path, problem_path: Path) -> Path:
        domains_dir = root / "domains"
        candidates = [domains_dir / f"{problem_path.stem}.pddl", domains_dir / problem_path.name, domain_default]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No domain PDDL found for problem {problem_path}")

    def _extract_goal(self, problem_pddl: str) -> str | None:
        marker = "(:goal"
        start = problem_pddl.lower().find(marker)
        return problem_pddl[start:].strip() if start >= 0 else None


@registry.register_dataset_builder("alfworld")
class ALFWorldBuilder(DatasetBuilder):
    dataset_name = "alfworld"
    task_type = "interactive"
    task_class = "interactive"
    environment_class = "alfworld"

    def build(self, split: str, limit: int | None = None) -> list[TaskExample]:
        split_dir = self._split_dir(split)
        traj_paths = sorted(split_dir.glob("**/traj_data.json"))
        examples = [self._normalize_trial(path, split, index) for index, path in enumerate(traj_paths)]
        return self._limit(examples, limit)

    def available_splits(self) -> list[str]:
        root = self.data_path
        if (root / "json_2.1.1").exists():
            root = root / "json_2.1.1"
        return sorted(path.name for path in root.iterdir() if path.is_dir()) if root.exists() else []

    def _split_dir(self, split: str) -> Path:
        root = self.data_path
        if (root / "json_2.1.1").exists():
            root = root / "json_2.1.1"
        split_dir = root / split
        if split_dir.exists():
            return split_dir
        if root.name == split:
            return root
        raise FileNotFoundError(f"No ALFWorld split directory '{split}' under {self.data_path}")

    def _normalize_trial(self, traj_path: Path, split: str, index: int) -> TaskExample:
        with traj_path.open("r", encoding="utf-8") as handle:
            traj = json.load(handle)
        trial_dir = traj_path.parent
        task_type = traj.get("task_type") or trial_dir.parent.name.split("-")[0]
        instruction = self._instruction_from_traj(traj) or trial_dir.parent.name.replace("-", " ")
        game_file = trial_dir / "game.tw-pddl"
        initial_state = trial_dir / "initial_state.pddl"
        return TaskExample(
            example_id=str(traj.get("task_id") or trial_dir.name or f"alfworld_{split}_{index}"),
            dataset_name=self.dataset_name,
            task_type=self.task_type,
            split=split,
            input={"instruction": instruction, "game_file": str(game_file), "traj_file": str(traj_path)},
            target={"success": True},
            metadata=self._metadata(
                traj_path,
                split,
                environment="alfworld",
                trial_dir=str(trial_dir),
                game_file=str(game_file),
                initial_state_pddl=str(initial_state) if initial_state.exists() else None,
                alfworld_task_type=task_type,
                benchmark_format="alfworld-json_2.1.1",
            ),
        )

    def _instruction_from_traj(self, traj: dict) -> str | None:
        anns = traj.get("turk_annotations", {}).get("anns", [])
        if anns and isinstance(anns[0], dict):
            return anns[0].get("task_desc")
        return traj.get("task_desc") or traj.get("goal")


@registry.register_dataset_builder("scienceworld")
class ScienceWorldBuilder(DatasetBuilder):
    dataset_name = "scienceworld"
    task_type = "interactive"
    task_class = "interactive"
    environment_class = "scienceworld"

    def build(self, split: str, limit: int | None = None) -> list[TaskExample]:
        source = self._resolve_manifest(split)
        records = _read_json_or_jsonl(source)
        filtered = [raw for raw in records if raw.get("split", split) == split]
        examples = [self._normalize_record(raw, split, source, index) for index, raw in enumerate(filtered)]
        return self._limit(examples, limit)

    def available_splits(self) -> list[str]:
        if self.data_path.is_file():
            records = _read_json_or_jsonl(self.data_path)
            return sorted({raw.get("split", "default") for raw in records})
        return super().available_splits()

    def _resolve_manifest(self, split: str) -> Path:
        if self.data_path.is_file():
            return self.data_path
        for candidate in (f"{split}.json", f"{split}.jsonl", "manifest.json", "manifest.jsonl"):
            path = self.data_path / candidate
            if path.exists():
                return path
        raise FileNotFoundError(f"No ScienceWorld manifest found for split '{split}' under {self.data_path}")

    def _normalize_record(self, raw: dict, split: str, source: Path, index: int) -> TaskExample:
        task_id = _first_present(raw, ("task_id", "task_num", "task_number"))
        task_name = _first_present(raw, ("task_name", "name"))
        variation_id = _first_present(raw, ("variation_id", "variation", "variation_idx"), 0)
        instruction = raw.get("instruction") or raw.get("task_description") or f"Complete ScienceWorld task {task_name or task_id} variation {variation_id}."
        example_id = str(raw.get("id") or f"scienceworld_{task_id or task_name}_{variation_id}")
        return TaskExample(
            example_id=example_id,
            dataset_name=self.dataset_name,
            task_type=self.task_type,
            split=split,
            input={"instruction": instruction, "task_id": task_id, "task_name": task_name, "variation_id": variation_id},
            target={"success": True},
            metadata=self._metadata(
                source,
                split,
                environment="scienceworld",
                task_id=task_id,
                task_name=task_name,
                variation_id=variation_id,
                simplification=raw.get("simplification") or raw.get("simplifications_preset"),
                benchmark_format="scienceworld-manifest",
            ),
        )
