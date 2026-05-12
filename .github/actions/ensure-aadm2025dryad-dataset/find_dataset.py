from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

RF_DIR_NAMES = {"RF Sensor and Radar", "RF_Sensor_and_Radar"}
TRANSIENT_DATASET_PREFIX = "AADM2025Dryad.tmp."


def is_transient_dataset_path(path: Path) -> bool:
    return any(part.startswith(TRANSIENT_DATASET_PREFIX) for part in path.parts)


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        try:
            expanded = path.expanduser()
        except RuntimeError:
            continue
        if expanded in seen:
            continue
        seen.add(expanded)
        out.append(expanded)
    return out


def find_rf_root(root: Path, max_depth: int, *, allow_transient: bool = False) -> Path | None:
    if not allow_transient and is_transient_dataset_path(root):
        return None
    if not root.exists() or not root.is_dir():
        return None
    if root.name in RF_DIR_NAMES:
        return root
    for name in RF_DIR_NAMES:
        direct = root / name
        if direct.is_dir():
            return direct

    root_depth = len(root.parts)
    for dirpath, dirnames, _ in os.walk(root):
        current = Path(dirpath)
        if not allow_transient:
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if not dirname.startswith(TRANSIENT_DATASET_PREFIX)
            ]
        depth = len(current.parts) - root_depth
        if depth >= max_depth:
            dirnames[:] = []
        if current.name in RF_DIR_NAMES:
            return current
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-dataset-root-file", required=True)
    parser.add_argument("--resolved-rf-root-file", required=True)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--allow-transient", action="store_true")
    args = parser.parse_args()

    roots = [Path(root) for root in args.root if root]
    searched: list[str] = []
    for root in unique_paths(roots):
        searched.append(str(root))
        rf_root = find_rf_root(root, args.max_depth, allow_transient=args.allow_transient)
        if rf_root is None:
            continue

        dataset_root = rf_root.parent
        Path(args.resolved_dataset_root_file).write_text(str(dataset_root), encoding="utf-8")
        Path(args.resolved_rf_root_file).write_text(str(rf_root), encoding="utf-8")
        print(f"Resolved dataset_root={dataset_root}")
        print(f"Resolved RF Sensor and Radar root={rf_root}")
        return 0

    print("Could not locate an RF Sensor and Radar directory.", file=sys.stderr)
    print("Searched roots:", file=sys.stderr)
    for path in searched:
        print(f"- {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
