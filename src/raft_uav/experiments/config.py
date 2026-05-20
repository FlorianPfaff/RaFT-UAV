"""Typed experiment configuration and provenance capture."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ExperimentConfig:
    """Serializable configuration for one experiment family."""

    name: str
    dataset_root: str
    output_dir: str
    flights: tuple[str, ...] = ()
    methods: tuple[str, ...] = ()
    options: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    calibration_artifacts: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ExperimentConfig":
        return cls(
            name=str(payload.get("name", "experiment")),
            dataset_root=str(payload.get("dataset_root", "")),
            output_dir=str(payload.get("output_dir", "outputs/experiment")),
            flights=tuple(str(item) for item in payload.get("flights", ()) or ()),
            methods=tuple(str(item) for item in payload.get("methods", ()) or ()),
            options=tuple(str(item) for item in payload.get("options", ()) or ()),
            environment={str(k): str(v) for k, v in dict(payload.get("environment", {}) or {}).items()},
            calibration_artifacts={
                str(k): str(v)
                for k, v in dict(payload.get("calibration_artifacts", {}) or {}).items()
            },
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        source = Path(path)
        if source.suffix.lower() == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
        elif source.suffix.lower() == ".toml":
            if tomllib is None:  # pragma: no cover
                raise RuntimeError("TOML config support requires Python 3.11+")
            payload = tomllib.loads(source.read_text(encoding="utf-8"))
        else:
            raise ValueError("experiment config must be .json or .toml")
        if not isinstance(payload, Mapping):
            raise ValueError("experiment config root must be a mapping")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["flights"] = list(self.flights)
        payload["methods"] = list(self.methods)
        payload["options"] = list(self.options)
        payload["environment"] = dict(self.environment)
        payload["calibration_artifacts"] = dict(self.calibration_artifacts)
        payload["metadata"] = dict(self.metadata)
        return payload

    def merged_environment(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ if base is None else base)
        env.update({str(k): str(v) for k, v in self.environment.items()})
        return env


def write_resolved_experiment_config(
    destination: str | Path,
    *,
    config: ExperimentConfig | None = None,
    argv: list[str] | None = None,
    env_prefixes: tuple[str, ...] = ("RAFT_UAV_",),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a resolved, auditable experiment configuration JSON."""

    payload: dict[str, Any] = {
        "argv": list(sys.argv if argv is None else argv),
        "python": sys.version,
        "platform": platform.platform(),
        "git": git_snapshot(Path.cwd()),
        "environment": filtered_environment(env_prefixes),
    }
    if config is not None:
        payload["config"] = config.to_dict()
    if extra:
        payload["extra"] = dict(extra)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def filtered_environment(prefixes: tuple[str, ...] = ("RAFT_UAV_",)) -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if any(key.startswith(prefix) for prefix in prefixes)
    }


def git_snapshot(cwd: Path) -> dict[str, Any]:
    commit = _run_git(["rev-parse", "HEAD"], cwd)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    status = _run_git(["status", "--porcelain"], cwd)
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_porcelain": status,
    }


def _run_git(args: list[str], cwd: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""
