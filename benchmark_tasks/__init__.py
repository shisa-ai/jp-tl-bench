from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / "benchmark_tasks"
JUDGE_PROFILES_DIR = REPO_ROOT / "judge_profiles"
DEFAULT_TASK_FILE = "translation_ja_en_bidirectional_v1.yaml"
DEFAULT_JUDGE_PROFILE_FILE = "default.yaml"


@dataclass(frozen=True)
class DatasetRef:
    repo: str
    config: str
    split: str
    revision: str
    hf_token_env: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "config": self.config,
            "split": self.split,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class ComparePromptProfile:
    profile_id: str
    prompt_path: Path


@dataclass(frozen=True)
class DirectionConfig:
    key: str
    source_language: str
    target_language: str
    display_name: str
    summary_language_label: str
    translation_prompts: dict[str, Path]
    legacy_english: bool | None = None

    def prompt_path(self, variant: str) -> Path:
        if variant not in self.translation_prompts:
            raise KeyError(f"Prompt variant '{variant}' is not configured for direction '{self.key}'")
        return self.translation_prompts[variant]


@dataclass(frozen=True)
class JudgeProfile:
    profile_id: str
    compare_prompt_profile: str
    parser_id: str
    path: Path
    raw: dict[str, Any]

    def resolve_request_settings(self, model_name: str) -> dict[str, Any]:
        settings = dict(self.raw.get("request_defaults", {}))
        matches = []
        for override in self.raw.get("model_request_overrides", []):
            needle = override.get("contains")
            if needle and needle.lower() in model_name.lower():
                matches.append(override)
        for override in sorted(matches, key=lambda item: len(item.get("contains", ""))):
            settings.update(override.get("settings", {}))
        return settings


class TaskConfig:
    def __init__(self, path: Path, raw: dict[str, Any]):
        self.path = path
        self.raw = raw
        self.task_id = raw["task_id"]
        self.task_type = raw["task_type"]
        self.task_version = raw["task_version"]
        self.language_names = raw.get("language_names", {})
        dataset = raw["dataset"]
        self.dataset = DatasetRef(
            repo=dataset["repo"],
            config=dataset["config"],
            split=dataset["split"],
            revision=dataset["revision"],
            hf_token_env=dataset.get("hf_token_env"),
        )
        self.compatibility = raw.get("compatibility", {})
        self.compare_prompt_profile = raw["compare_prompt_profile"]
        self.compare_prompt_profiles = {
            profile_id: ComparePromptProfile(
                profile_id=profile_id,
                prompt_path=_resolve_repo_path(profile["prompt_path"]),
            )
            for profile_id, profile in raw["compare_prompt_profiles"].items()
        }
        self.directions = [
            DirectionConfig(
                key=direction["key"],
                source_language=direction["source_language"],
                target_language=direction["target_language"],
                display_name=direction["display_name"],
                summary_language_label=direction.get("summary_language_label", direction["key"]),
                translation_prompts={
                    variant: _resolve_repo_path(prompt_path)
                    for variant, prompt_path in direction["translation_prompts"].items()
                },
                legacy_english=direction.get("legacy_english"),
            )
            for direction in raw["directions"]
        ]
        self._direction_by_key = {direction.key: direction for direction in self.directions}
        self._direction_by_pair = {
            (direction.source_language, direction.target_language): direction
            for direction in self.directions
        }
        self._direction_by_legacy_english = {
            direction.legacy_english: direction
            for direction in self.directions
            if direction.legacy_english is not None
        }
        self._language_alias_to_code = {
            code.lower(): code
            for code in self.language_names
        }
        self._language_alias_to_code.update(
            {
                name.lower(): code
                for code, name in self.language_names.items()
                if isinstance(name, str) and name.strip()
            }
        )
        self.scoring_direction_order = raw.get(
            "scoring", {}
        ).get("direction_order", [direction.key for direction in self.directions])
        self.task_config_digest = _compute_digest(raw)

    @property
    def emit_legacy_english(self) -> bool:
        return bool(self.compatibility.get("emit_legacy_english", False))

    @property
    def name_alias_mode(self) -> str:
        return self.compatibility.get("name_alias_mode", "legacy_passthrough")

    def get_prompt_variant(self, low_context: bool = False, ultra_low_context: bool = False) -> str:
        if ultra_low_context:
            return "ultra_low_context"
        if low_context:
            return "low_context"
        return "default"

    def direction_for_languages(self, source_language: str, target_language: str) -> DirectionConfig:
        try:
            return self._direction_by_pair[(source_language, target_language)]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported direction for task '{self.task_id}': {source_language}->{target_language}"
            ) from exc

    def direction_by_key(self, key: str) -> DirectionConfig:
        try:
            return self._direction_by_key[key]
        except KeyError as exc:
            raise ValueError(f"Unknown direction key '{key}' for task '{self.task_id}'") from exc

    def canonicalize_language(self, value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        if not normalized:
            return None
        return self._language_alias_to_code.get(normalized, normalized)

    def supports_record(self, record: dict[str, Any]) -> bool:
        source_language = self.canonicalize_language(
            record.get("source_language") or record.get("language")
        )
        target_language = self.canonicalize_language(record.get("target_language"))
        if source_language and target_language:
            return (source_language, target_language) in self._direction_by_pair
        if source_language:
            return any(
                direction.source_language == source_language
                for direction in self.directions
            )
        if "english" in record and record.get("english") in (True, False):
            return bool(record["english"]) in self._direction_by_legacy_english
        return False

    def direction_for_record(self, record: dict[str, Any]) -> DirectionConfig:
        source_language = self.canonicalize_language(
            record.get("source_language") or record.get("language")
        )
        target_language = self.canonicalize_language(record.get("target_language"))
        if source_language and target_language:
            return self.direction_for_languages(source_language, target_language)
        if source_language and not target_language:
            matches = [
                direction
                for direction in self.directions
                if direction.source_language == source_language
            ]
            if len(matches) == 1:
                return matches[0]
            if not matches:
                raise ValueError(
                    f"Task '{self.task_id}' does not support source language '{source_language}' for record {record!r}"
                )
            raise ValueError(
                f"Task '{self.task_id}' cannot infer a unique target language from source language '{source_language}' for record {record!r}"
            )
        if "english" in record and record.get("english") in (True, False):
            try:
                return self._direction_by_legacy_english[bool(record["english"])]
            except KeyError as exc:
                raise ValueError(
                    f"Task '{self.task_id}' does not define a legacy english mapping for record {record!r}"
                ) from exc
        raise ValueError(
            f"Record is missing source/target language metadata and legacy english mapping: {record!r}"
        )

    def get_prompt_path(
        self,
        source_language: str,
        target_language: str,
        *,
        low_context: bool = False,
        ultra_low_context: bool = False,
    ) -> Path:
        direction = self.direction_for_languages(source_language, target_language)
        variant = self.get_prompt_variant(low_context=low_context, ultra_low_context=ultra_low_context)
        return direction.prompt_path(variant)

    def get_language_name(self, code: str) -> str:
        return self.language_names.get(code, code.upper())

    def normalize_record(
        self,
        raw: dict[str, Any],
        *,
        require_source_text: bool = False,
    ) -> dict[str, Any]:
        normalized = dict(raw)
        item_id = raw.get("item_id") or raw.get("name")
        if not item_id:
            raise ValueError(f"Task record is missing both item_id and name: {raw!r}")

        direction = self.direction_for_record(raw)
        source_text = raw.get("source_text", raw.get("text"))
        if require_source_text and source_text is None:
            raise ValueError(f"Task record is missing source_text/text: {raw!r}")

        if self.name_alias_mode == "item_id_alias":
            canonical_name = item_id
        else:
            canonical_name = raw.get("name") or item_id

        normalized.update(
            {
                "item_id": item_id,
                "name": canonical_name,
                "task_id": raw.get("task_id", self.task_id),
                "task_type": raw.get("task_type", self.task_type),
                "task_version": raw.get("task_version", self.task_version),
                "source_language": direction.source_language,
                "target_language": direction.target_language,
            }
        )
        if source_text is not None:
            normalized["source_text"] = source_text

        if self.emit_legacy_english and direction.legacy_english is not None:
            normalized["english"] = direction.legacy_english
        else:
            normalized.pop("english", None)

        return normalized


def _resolve_repo_path(path_str: str) -> Path:
    return (REPO_ROOT / path_str).resolve()


def _compute_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _is_hub_commit_sha(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{40}", value))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_config_path(spec: str | os.PathLike[str] | None, *, base_dir: Path, default_file: str, env_var: str) -> Path:
    if spec is None:
        spec = os.getenv(env_var, default_file)
    path = Path(spec)
    if path.exists():
        return path.resolve()
    filename = path.name
    if not filename.endswith(".yaml"):
        filename = f"{filename}.yaml"
    candidate = base_dir / filename
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve config '{spec}' under {base_dir}")


def load_task_config(task: str | os.PathLike[str] | None = None) -> TaskConfig:
    path = _resolve_config_path(
        task,
        base_dir=TASKS_DIR,
        default_file=DEFAULT_TASK_FILE,
        env_var="TASK_CONFIG",
    )
    return TaskConfig(path=path, raw=_load_yaml(path))


def load_judge_profile(profile: str | os.PathLike[str] | None = None) -> JudgeProfile:
    path = _resolve_config_path(
        profile,
        base_dir=JUDGE_PROFILES_DIR,
        default_file=DEFAULT_JUDGE_PROFILE_FILE,
        env_var="JUDGE_PROFILE",
    )
    raw = _load_yaml(path)
    return JudgeProfile(
        profile_id=raw["judge_profile_id"],
        compare_prompt_profile=raw["compare_prompt_profile"],
        parser_id=raw["parser_id"],
        path=path,
        raw=raw,
    )


def resolve_dataset_ref(
    task_config: TaskConfig,
    *,
    token: str | None = None,
    api: HfApi | None = None,
) -> dict[str, Any]:
    dataset = task_config.dataset
    resolved_revision = dataset.revision
    if not _is_hub_commit_sha(dataset.revision):
        info = (api or HfApi()).dataset_info(
            dataset.repo,
            revision=dataset.revision,
            token=token,
        )
        resolved_revision = getattr(info, "sha", None) or dataset.revision
        if not _is_hub_commit_sha(resolved_revision):
            raise RuntimeError(
                f"Could not resolve an immutable Hub commit SHA for {dataset.repo}@{dataset.revision}"
            )
    return {
        "repo": dataset.repo,
        "config": dataset.config,
        "split": dataset.split,
        "revision": dataset.revision,
        "resolved_revision": resolved_revision,
    }


def resolve_compare_prompt_path(task_config: TaskConfig, judge_profile: JudgeProfile | None = None) -> Path:
    profile_id = judge_profile.compare_prompt_profile if judge_profile else task_config.compare_prompt_profile
    if profile_id not in task_config.compare_prompt_profiles:
        raise KeyError(
            f"Compare prompt profile '{profile_id}' is not defined for task '{task_config.task_id}'"
        )
    return task_config.compare_prompt_profiles[profile_id].prompt_path
