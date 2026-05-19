"""Compare online radar association choices with an oracle diagnostic selector."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raft_uav.io.aerpaw import normalize_truth, read_truth, select_flight  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", default="Opt1")
    parser.add_argument(
        "--online-association",
        choices=[
            "prediction-nis",
            "rf-anchored-nis",
            "rf-gated-nis",
            "track-continuity",
            "geometry-score",
        ],
        default="prediction-nis",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/association_failure_analysis"),
    )
    parser.add_argument("--smoother", choices=["none", "fixed-lag", "rts"], default="fixed-lag")
    parser.add_argument("--smoother-lag-s", type=float, default=20.0)
    parser.add_argument("--rf-gate-prob", type=float, default=0.99)
    parser.add_argument("--radar-gate-prob", type=float, default=0.99)
    parser.add_argument("--radar-catprob-threshold", type=float, default=0.5)
    parser.add_argument("--rf-inflation-alpha", type=float, default=0.5)
    parser.add_argument("--radar-inflation-alpha", type=float, default=0.5)
    parser.add_argument("--geometry-velocity-std", type=float, default=12.0)
    parser.add_argument("--geometry-velocity-weight", type=float, default=0.25)
    parser.add_argument("--geometry-switch-penalty", type=float, default=4.0)
    parser.add_argument("--geometry-catprob-weight", type=float, default=2.0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    run_root = args.output_dir / args.flight
    online_dir = run_root / f"online_{args.online_association}"
    oracle_dir = run_root / "oracle_nearest_truth"
    for association, destination in (
        (args.online_association, online_dir),
        ("oracle-nearest-truth", oracle_dir),
    ):
        _run_one(args=args, association=association, output_dir=destination)

    flight = select_flight(args.dataset_root, args.flight)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")
    truth, _, _ = normalize_truth(read_truth(flight.truth_txt))

    online = _read_selected_radar(online_dir / args.flight / "selected_radar.csv")
    oracle = _read_selected_radar(oracle_dir / args.flight / "selected_radar.csv")
    online = _append_truth_errors(online, truth)
    oracle = _append_truth_errors(oracle, truth)

    comparison = _compare_online_to_oracle(online, oracle)
    summary = _summary(args.flight, args.online_association, comparison, online, oracle)

    analysis_dir = run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = analysis_dir / "association_comparison.csv"
    summary_path = analysis_dir / "association_summary.json"
    figure_path = analysis_dir / "association_diagnostic.png"
    comparison.to_csv(comparison_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_diagnostic_plot(figure_path, comparison, args.flight, args.online_association)

    print(f"flight={args.flight}")
    print(f"online_association={args.online_association}")
    print(f"selected_rows={summary['selected_rows']}")
    print(f"oracle_rows={summary['oracle_rows']}")
    print(f"compared_frames={summary['compared_frames']}")
    print(f"track_match_rate={summary['track_match_rate']:.3f}")
    print(f"selected_error_p95_m={summary['selected_truth_error_m']['p95_m']:.3f}")
    print(f"oracle_error_p95_m={summary['oracle_truth_error_m']['p95_m']:.3f}")
    print(f"oracle_gap_p95_m={summary['oracle_gap_m']['p95_m']:.3f}")
    print(f"comparison_csv={comparison_path}")
    print(f"summary_json={summary_path}")
    print(f"figure_png={figure_path}")
    return 0


def _run_one(*, args: argparse.Namespace, association: str, output_dir: Path) -> None:
    metrics_path = output_dir / args.flight / "metrics.json"
    selected_path = output_dir / args.flight / "selected_radar.csv"
    if args.skip_existing and metrics_path.exists() and selected_path.exists():
        return

    command = [
        sys.executable,
        "-m",
        "raft_uav.cli",
        "run-baseline",
        str(args.dataset_root),
        "--flight",
        args.flight,
        "--output-dir",
        str(output_dir),
        "--radar-association",
        association,
        "--robust-update",
        "nis-inflate",
        "--rf-gate-prob",
        str(args.rf_gate_prob),
        "--radar-gate-prob",
        str(args.radar_gate_prob),
        "--radar-catprob-threshold",
        str(args.radar_catprob_threshold),
        "--rf-inflation-alpha",
        str(args.rf_inflation_alpha),
        "--radar-inflation-alpha",
        str(args.radar_inflation_alpha),
        "--smoother",
        args.smoother,
        "--geometry-velocity-std",
        str(args.geometry_velocity_std),
        "--geometry-velocity-weight",
        str(args.geometry_velocity_weight),
        "--geometry-switch-penalty",
        str(args.geometry_switch_penalty),
        "--geometry-catprob-weight",
        str(args.geometry_catprob_weight),
    ]
    if args.smoother == "fixed-lag":
        command.extend(["--smoother-lag-s", str(args.smoother_lag_s)])
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, env=_subprocess_env())


def _read_selected_radar(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing selected radar artifact: {path}")
    return pd.read_csv(path)


def _append_truth_errors(selected: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    out = selected.copy()
    if out.empty:
        out["truth_error_m"] = []
        out["truth_time_delta_s"] = []
        return out
    truth_times = truth["time_s"].to_numpy(dtype=float)
    query_times = out["time_s"].to_numpy(dtype=float)
    truth_indices = _nearest_time_indices(truth_times, query_times)
    truth_positions = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)[truth_indices]
    selected_positions = out[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    out["truth_error_m"] = np.linalg.norm(selected_positions - truth_positions, axis=1)
    out["truth_time_delta_s"] = np.abs(truth_times[truth_indices] - query_times)
    return out


def _compare_online_to_oracle(online: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    online_compact = _compact_selected_rows(online, "selected")
    oracle_compact = _compact_selected_rows(oracle, "oracle")
    if "frame_index" in online_compact.columns and "frame_index" in oracle_compact.columns:
        comparison = online_compact.merge(oracle_compact, on="frame_index", how="outer")
    else:
        online_compact["time_key_s"] = online_compact["selected_time_s"].round(3)
        oracle_compact["time_key_s"] = oracle_compact["oracle_time_s"].round(3)
        comparison = online_compact.merge(oracle_compact, on="time_key_s", how="outer")

    comparison["time_s"] = comparison["selected_time_s"].combine_first(
        comparison["oracle_time_s"]
    )
    comparison["has_selected"] = comparison["selected_track_id"].notna()
    comparison["has_oracle"] = comparison["oracle_track_id"].notna()
    comparison["selected_matches_oracle"] = (
        comparison["has_selected"]
        & comparison["has_oracle"]
        & (comparison["selected_track_id"] == comparison["oracle_track_id"])
    )
    comparison["oracle_gap_m"] = (
        comparison["selected_truth_error_m"] - comparison["oracle_truth_error_m"]
    )
    return comparison.sort_values("time_s").reset_index(drop=True)


def _compact_selected_rows(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    columns = {
        "frame_index": "frame_index",
        "time_s": f"{prefix}_time_s",
        "track_id": f"{prefix}_track_id",
        "cat_prob_uav": f"{prefix}_cat_prob_uav",
        "association_nis": f"{prefix}_association_nis",
        "association_score": f"{prefix}_association_score",
        "association_candidate_rows": f"{prefix}_candidate_rows",
        "range_m": f"{prefix}_range_m",
        "radial_velocity_mps": f"{prefix}_radial_velocity_mps",
        "num_inliers": f"{prefix}_num_inliers",
        "truth_error_m": f"{prefix}_truth_error_m",
        "truth_time_delta_s": f"{prefix}_truth_time_delta_s",
    }
    present = {column: renamed for column, renamed in columns.items() if column in frame.columns}
    compact = frame.loc[:, list(present)].rename(columns=present).copy()
    for column in columns.values():
        if column not in compact.columns and column != "frame_index":
            compact[column] = np.nan
    return compact


def _summary(
    flight: str,
    online_association: str,
    comparison: pd.DataFrame,
    online: pd.DataFrame,
    oracle: pd.DataFrame,
) -> dict[str, Any]:
    paired = comparison.loc[comparison["has_selected"] & comparison["has_oracle"]].copy()
    gaps = paired["oracle_gap_m"].to_numpy(dtype=float)
    gaps = gaps[np.isfinite(gaps)]
    match_rate = float(paired["selected_matches_oracle"].mean()) if len(paired) else float("nan")
    return {
        "flight": flight,
        "online_association": online_association,
        "selected_rows": int(len(online)),
        "oracle_rows": int(len(oracle)),
        "compared_frames": int(len(paired)),
        "track_match_rate": match_rate,
        "selected_truth_error_m": _summarize(paired["selected_truth_error_m"]),
        "oracle_truth_error_m": _summarize(paired["oracle_truth_error_m"]),
        "oracle_gap_m": _summarize(pd.Series(gaps)),
        "frames_gap_gt_100m": int(np.sum(gaps > 100.0)),
        "frames_gap_gt_250m": int(np.sum(gaps > 250.0)),
        "worst_frames": _worst_frames(paired),
    }


def _summarize(values: pd.Series) -> dict[str, float]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if array.size == 0:
        return {
            "mean_m": float("nan"),
            "p50_m": float("nan"),
            "p95_m": float("nan"),
            "max_m": float("nan"),
        }
    return {
        "mean_m": float(np.mean(array)),
        "p50_m": float(np.percentile(array, 50)),
        "p95_m": float(np.percentile(array, 95)),
        "max_m": float(np.max(array)),
    }


def _worst_frames(paired: pd.DataFrame, top_k: int = 8) -> list[dict[str, Any]]:
    if paired.empty:
        return []
    worst = paired.sort_values("oracle_gap_m", ascending=False).head(top_k)
    columns = [
        "frame_index",
        "time_s",
        "selected_track_id",
        "oracle_track_id",
        "selected_truth_error_m",
        "oracle_truth_error_m",
        "oracle_gap_m",
        "selected_association_nis",
        "selected_association_score",
        "selected_cat_prob_uav",
        "selected_range_m",
        "selected_radial_velocity_mps",
    ]
    present = [column for column in columns if column in worst.columns]
    rows = []
    for row in worst[present].to_dict(orient="records"):
        rows.append({key: _json_scalar(value) for key, value in row.items()})
    return rows


def _write_diagnostic_plot(
    path: Path,
    comparison: pd.DataFrame,
    flight: str,
    online_association: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paired = comparison.loc[comparison["has_selected"] & comparison["has_oracle"]]
    if paired.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.5), sharex=True, constrained_layout=True)
    axes[0].plot(
        paired["time_s"],
        paired["selected_truth_error_m"],
        color="#386cb0",
        linewidth=1.1,
        label=online_association,
    )
    axes[0].plot(
        paired["time_s"],
        paired["oracle_truth_error_m"],
        color="#1b9e77",
        linewidth=1.1,
        label="oracle nearest truth",
    )
    axes[0].set_ylabel("truth error [m]")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, color="#dddddd", linewidth=0.7)

    axes[1].plot(paired["time_s"], paired["oracle_gap_m"], color="#d95f02", linewidth=1.0)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("oracle gap [m]")
    axes[1].grid(True, color="#dddddd", linewidth=0.7)

    axes[2].scatter(
        paired["time_s"],
        paired["selected_track_id"],
        color="#386cb0",
        s=10,
        alpha=0.75,
        label=online_association,
    )
    axes[2].scatter(
        paired["time_s"],
        paired["oracle_track_id"],
        color="#1b9e77",
        s=10,
        alpha=0.75,
        label="oracle",
    )
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("track id")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, color="#dddddd", linewidth=0.7)
    fig.suptitle(f"{flight} radar association diagnostic")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _nearest_time_indices(reference_times_s: np.ndarray, query_times_s: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference_times_s, dtype=float).reshape(-1)
    query = np.asarray(query_times_s, dtype=float).reshape(-1)
    insertion = np.searchsorted(reference, query)
    right = np.clip(insertion, 0, reference.size - 1)
    left = np.clip(insertion - 1, 0, reference.size - 1)
    use_right = np.abs(reference[right] - query) < np.abs(reference[left] - query)
    return np.where(use_right, right, left)


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not current else os.pathsep.join([src_path, current])
    return env


def _json_scalar(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
