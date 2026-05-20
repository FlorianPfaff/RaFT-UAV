import json

import numpy as np

from raft_uav.coordinates import LocalENUProjector
from raft_uav.io.aerpaw import (
    normalize_radar,
    normalize_rf,
    normalize_truth,
    read_radar_tracks_json,
    read_rf_csv,
    read_truth,
    select_radar_measurement_rows,
)


def test_enu_origin_maps_to_near_zero():
    projector = LocalENUProjector(35.7274895, -78.696216, 2.717)

    enu = projector.transform(35.7274895, -78.696216, 2.717)

    np.testing.assert_allclose(enu, np.zeros(3), atol=1e-6)


def test_timestamp_normalization_is_monotonic_and_truth_relative(tmp_path):
    truth_path = tmp_path / "vehicleOut.txt"
    truth_path.write_text(
        "\n".join(
            [
                '1,-78.696216,35.7274895,2.717,"(0,0,0)","(0,0,0)",49.6,2025-10-07 15:42:19.000000,4,27',
                '2,-78.6962159,35.7274895,2.727,"(0,0,0)","(0,0,0)",49.6,2025-10-07 15:42:20.000000,4,27',
            ]
        ),
        encoding="utf-8",
    )
    rf_path = tmp_path / "rf.csv"
    rf_path.write_text(
        "\n".join(
            [
                "Meas ID,Algorithm,Center Frequency,Sample Rate,Num Samples,Latitude,Longitude,Elevation,RHO,CEP,Total Sensors,Valid Sensors,Overloaded Sensors,Sensor Names,Time,Comment",
                "1,tdoa,1,1,1,35.72749,-78.69621,0,0,75,4,4,0,s,2025-10-07 19:42:20.000,",
            ]
        ),
        encoding="utf-8",
    )

    truth, projector, origin_time = normalize_truth(read_truth(truth_path))
    rf = normalize_rf(read_rf_csv(rf_path), projector, origin_time)

    assert truth["time_s"].tolist() == [0.0, 1.0]
    assert truth["time_s"].is_monotonic_increasing
    np.testing.assert_allclose(rf["time_s"].to_numpy(), np.array([1.0]), atol=1e-6)


def test_rf_and_radar_clock_offsets_are_independent(tmp_path):
    truth_path = tmp_path / "vehicleOut.txt"
    truth_path.write_text(
        '1,-78.696216,35.7274895,2.717,"(0,0,0)","(0,0,0)",49.6,2025-10-07 15:42:20.000000,4,27',
        encoding="utf-8",
    )
    rf_path = tmp_path / "rf.csv"
    rf_path.write_text(
        "\n".join(
            [
                "Meas ID,Algorithm,Center Frequency,Sample Rate,Num Samples,Latitude,Longitude,Elevation,RHO,CEP,Total Sensors,Valid Sensors,Overloaded Sensors,Sensor Names,Time,Comment",
                "1,tdoa,1,1,1,35.72749,-78.69621,0,0,75,4,4,0,s,2025-10-07 19:42:25.000,",
            ]
        ),
        encoding="utf-8",
    )
    radar_path = tmp_path / "radar.json"
    radar_path.write_text(
        json.dumps(
            {
                "params": {"globalTime": 1759851745.0},
                "trackData": [
                    {
                        "id": 1,
                        "lla": [35.72749, -78.69621, 30.0],
                        "globalTime": 1759851745.0,
                        "catProb": [0.8, 0.1],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    _, projector, origin_time = normalize_truth(read_truth(truth_path))
    rf = normalize_rf(
        read_rf_csv(rf_path),
        projector,
        origin_time,
        clock_offset_s=-4.0 * 60.0 * 60.0,
    )
    radar = normalize_radar(
        read_radar_tracks_json(radar_path),
        projector,
        origin_time,
        clock_offset_s=0.0,
    )
    radar_with_rf_offset = normalize_radar(
        read_radar_tracks_json(radar_path),
        projector,
        origin_time,
        clock_offset_s=-4.0 * 60.0 * 60.0,
    )

    np.testing.assert_allclose(rf["time_s"].to_numpy(), np.array([5.0]), atol=1e-6)
    np.testing.assert_allclose(radar["time_s"].to_numpy(), np.array([5.0]), atol=1e-6)
    np.testing.assert_allclose(
        radar_with_rf_offset["time_s"].to_numpy(), np.array([-14395.0]), atol=1e-6
    )


def test_radar_jsonl_reader_skips_non_object_frames(tmp_path):
    radar_path = tmp_path / "radar.json"
    radar_path.write_text(
        "\n".join(
            [
                json.dumps({"trackData": []}),
                json.dumps(["not", "an", "object"]),
            ]
        ),
        encoding="utf-8",
    )

    assert read_radar_tracks_json(radar_path).empty


def test_radar_jsonl_reader_skips_non_object_payload(tmp_path):
    radar_path = tmp_path / "radar.json"
    radar_path.write_text(json.dumps([]), encoding="utf-8")

    assert read_radar_tracks_json(radar_path).empty


def test_radar_jsonl_reader_and_catprob_selection(tmp_path):
    radar_path = tmp_path / "radar.json"
    frames = [
        {"error": {"code": 200, "message": ""}},
        {
            "params": {"globalTime": 1759866140.0},
            "trackData": [
                {
                    "id": 1,
                    "lla": [35.72749, -78.69621, 30.0],
                    "globalTime": 1759866140.0,
                    "catProb": [0.6, 0.1],
                },
                {
                    "id": 2,
                    "lla": [35.727, -78.696, 30.0],
                    "globalTime": 1759866140.0,
                    "catProb": [0.1, 0.1],
                },
                {
                    "id": 3,
                    "lla": [35.7271, -78.6961, 30.0],
                    "globalTime": 1759866140.0,
                    "catProb": [0.55, 0.1],
                },
            ],
        },
    ]
    radar_path.write_text("\n".join(json.dumps(frame) for frame in frames), encoding="utf-8")
    truth_path = tmp_path / "vehicleOut.txt"
    truth_path.write_text(
        '1,-78.696216,35.7274895,2.717,"(0,0,0)","(0,0,0)",49.6,2025-10-07 15:42:20.000000,4,27',
        encoding="utf-8",
    )

    truth, projector, origin_time = normalize_truth(read_truth(truth_path))
    radar = normalize_radar(read_radar_tracks_json(radar_path), projector, origin_time)
    selected = select_radar_measurement_rows(radar, selection="catprob", catprob_threshold=0.5)
    selected_all = select_radar_measurement_rows(
        radar,
        selection="catprob-all",
        catprob_threshold=0.5,
    )

    assert len(radar) == 3
    assert selected["track_id"].tolist() == [1]
    assert selected_all["track_id"].tolist() == [1, 3]
    assert len(truth) == 1


def test_radar_jsonl_reader_skips_malformed_records(tmp_path):
    radar_path = tmp_path / "radar.json"
    frames = [
        [],
        {"params": ["not", "a", "mapping"], "trackData": "not-a-list"},
        {
            "params": {"globalTime": 123.0},
            "trackData": [
                None,
                "not-a-track",
                {
                    "id": 7,
                    "lla": [35.72749, -78.69621, 30.0],
                    "catProb": [0.9, 0.1],
                },
            ],
        },
    ]
    radar_path.write_text(
        "\n".join(json.dumps(frame) for frame in frames),
        encoding="utf-8",
    )

    radar = read_radar_tracks_json(radar_path)

    assert len(radar) == 1
    row = radar.iloc[0]
    assert row["track_id"] == 7
    assert row["global_time_raw_s"] == 123.0
    assert row["cat_prob_uav"] == 0.9
