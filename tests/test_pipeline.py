import csv
from pathlib import Path
from unittest.mock import patch

from qos_ml.drift import AdaptiveDriftDetector, ErrorRateDriftDetector
from qos_ml.qos_policy import policy_for
from network.analyze_results import read_iperf_result
from network.enforcer import TcEnforcer
from traffic.feature_extractor import extract_features, label_from_filename
from traffic.flow_builder import build_flows


def _packet(timestamp, src_port=1000, dst_port=443):
    return {
        "timestamp": timestamp,
        "size": 100,
        "proto": "TCP",
        "src_ip": "10.0.0.1",
        "dst_ip": "10.0.0.2",
        "src_port": src_port,
        "dst_port": dst_port,
        "source_file": "skype_audio.pcap",
    }


def test_flow_builder_keeps_capture_files_separate():
    packets = [_packet(0), {**_packet(1), "source_file": "email.pcap"}]
    assert len(build_flows(packets)) == 2


def test_feature_extraction_and_labels():
    flow = build_flows([_packet(0), _packet(0.1)])[0]
    row = extract_features(flow)
    assert row["packet_count"] == 2
    assert row["label"] == "voip"
    assert label_from_filename("vpn_email2a.pcap") == ("email", True)


def test_drift_detector():
    detector = ErrorRateDriftDetector(window_size=3, threshold=2 / 3)
    assert not detector.update("voip", "voip")
    assert not detector.update("voip", "chat")
    assert detector.update("voip", "email")
    assert detector.error_rate == 2 / 3


def test_parent_drift_detector_detects_network_degradation():
    detector = AdaptiveDriftDetector(rtt_multiplier=2.0, drop_delta=1)
    assert not detector.update_metrics({"rtt_ms": 10, "dropped_packets": 0})
    assert detector.update_metrics({"rtt_ms": 25, "dropped_packets": 0})
    assert detector.last_reason == "rtt_spike"
    assert detector.update_metrics({"rtt_ms": 10, "dropped_packets": 1})
    assert detector.last_reason == "queue_drops"


def test_unknown_qos_class_is_best_effort():
    assert policy_for("unknown") == {"dscp": 0, "classid": "1:30"}


def test_enforcer_dry_run_records_policy():
    enforcer = TcEnforcer(dry_run=True)
    result = enforcer.apply("voip", {"proto": "UDP", "dst_port": 5001})
    assert result["dscp"] == 46
    assert enforcer.applied == [result]


def test_enforcer_supports_running_inside_namespace():
    enforcer = TcEnforcer(namespace=None, dry_run=True)
    result = enforcer.apply("voip", {"proto": "UDP", "dst_port": 5001})
    assert result["dry_run"] is True


def test_monitor_history_is_bounded():
    from network.monitor import QoSMonitor

    def runner(command):
        return type("Result", (), {"stdout": "time=1 ms dropped 0", "returncode": 0})()

    monitor = QoSMonitor(namespace=None, runner=runner, max_history=2)
    monitor.sample()
    monitor.sample()
    monitor.sample()
    assert len(monitor.history) == 2


def test_iperf_result_parser():
    path = Path("network/results/qos_udp.json")
    metrics = read_iperf_result(path)
    assert metrics["throughput_mbps"] >= 0


def test_api_server_cli_passes_retraining_options():
    from api import server
    import qos_ml.controller

    with patch("api.server.serve") as mocked_serve, patch(
        "qos_ml.controller.LiveController"
    ) as mocked_controller, patch(
        "sys.argv",
        [
            "api.server",
            "models/random_forest.joblib",
            "--dataset",
            "data/processed/features_balanced.csv",
            "--output-dir",
            "models",
            "--port",
            "8099",
            "--monitor-interval",
            "10",
        ],
    ):
        server.main()

    mocked_controller.assert_called_once_with(
        "models/random_forest.joblib",
        dataset_path="data/processed/features_balanced.csv",
        retrain_output="models",
        min_confidence=0.7,
        latency_budget_ms=None,
        fallback_model=None,
    )
    mocked_serve.assert_called_once_with(
        mocked_controller.return_value,
        "127.0.0.1",
        8099,
        10.0,
    )


def _synthetic_rows(files_per_class=4, per_file=25):
    import random

    random.seed(0)
    centres = {"voip": 120, "chat": 300, "email": 700, "streaming": 1200}
    rows = []
    for label, centre in centres.items():
        for f in range(files_per_class):
            for _ in range(per_file):
                size = centre + random.uniform(-40, 40)
                rows.append(
                    {
                        "proto": "UDP", "flow_duration": random.uniform(1, 5), "packet_count": 20,
                        "total_bytes": size * 20, "mean_packet_size": size, "std_packet_size": 5,
                        "mean_iat": 0.1, "std_iat": 0.02, "num_bursts": 2, "mean_burst_size": 10,
                        "is_vpn": 0, "src_ip": "10.0.0.1", "src_port": 1, "dst_ip": "10.0.0.2",
                        "dst_port": 2, "source_file": f"{label}_{f}.pcap", "label": label,
                    }
                )
    return rows


def _write_dataset(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_split_keeps_each_recording_on_one_side():
    from qos_ml.train import split_rows

    rows = _synthetic_rows()
    train_idx, test_idx, method = split_rows(rows)
    assert method == "by_recording"
    train_files = {rows[i]["source_file"] for i in train_idx}
    test_files = {rows[i]["source_file"] for i in test_idx}
    assert train_files.isdisjoint(test_files)
    assert {rows[i]["label"] for i in test_idx} <= {rows[i]["label"] for i in train_idx}


def test_min_packets_drops_short_flows(tmp_path):
    from qos_ml.train import _load_rows

    rows = _synthetic_rows(per_file=5)
    rows[0]["packet_count"] = 2
    _write_dataset(tmp_path / "d.csv", rows)
    assert len(_load_rows(tmp_path / "d.csv", min_packets=10)) == len(rows) - 1


def _controller(tmp_path, **kwargs):
    from qos_ml.controller import LiveController
    from qos_ml.train import train_models

    _write_dataset(tmp_path / "data.csv", _synthetic_rows())
    train_models(tmp_path / "data.csv", tmp_path, model_names=["random_forest"])
    return LiveController(
        tmp_path / "random_forest.joblib", dataset_path=tmp_path / "data.csv", **kwargs
    )


def test_low_confidence_prediction_gets_default_policy(tmp_path):
    flow = {k: v for k, v in _synthetic_rows()[0].items()}
    confident = _controller(tmp_path, min_confidence=0.0).classify(flow)
    assert confident["label"] == "voip" and confident["enforcement"]["dscp"] == 46
    strict = _controller(tmp_path, min_confidence=1.0)
    flow["mean_packet_size"] = 500  # between classes -> the forest is split
    event = strict.classify(flow)
    assert event["confidence"] is not None
    if event["confidence"] < 1.0:
        assert event["label"] == "unknown" and event["enforcement"]["dscp"] == 0


def test_feedback_is_saved_only_with_a_valid_flow_and_label(tmp_path):
    controller = _controller(tmp_path)
    flow = _synthetic_rows()[0]
    controller.feedback("voip", "chat")  # no flow: counted for drift, not saved
    controller.feedback("voip", "not_a_class", flow)  # invalid label: not saved
    assert not controller.feedback_path.exists()
    controller.feedback("voip", "chat", flow)
    saved = list(csv.DictReader(controller.feedback_path.open()))
    assert len(saved) == 1 and saved[0]["label"] == "chat"
    combined = controller._training_file()
    assert sum(1 for _ in csv.DictReader(combined.open())) == len(_synthetic_rows()) + controller.feedback_repeat


def test_retrain_promotes_only_a_model_that_is_not_worse(tmp_path):
    controller = _controller(tmp_path)
    controller.retrain()
    assert controller.last_retrain["promoted"] is True
    assert (tmp_path / "random_forest.prev.joblib").exists()

    class Bad:
        classes_ = ["voip"]

        def predict(self, frame):
            return ["voip"] * len(frame)

    controller.model = Bad()
    controller.retrain()
    assert controller.last_retrain["promoted"] is True  # candidate beats the bad model
    assert controller.model is not Bad

    import joblib

    good = joblib.load(tmp_path / "random_forest.joblib")
    controller.model = good
    controller.max_regression = -0.5  # demand a big improvement that cannot happen
    before = controller.model
    controller.retrain()
    assert controller.last_retrain["promoted"] is False and controller.model is before


def test_early_packet_study_reports_each_packet_count():
    from traffic.early_packet_study import run_study

    flows = []
    for label, size in (("voip", 100), ("chat", 400), ("email", 900), ("streaming", 1300)):
        for f in range(4):
            packets = [
                {**_packet(i * 0.05), "size": size + i, "source_file": f"{label}_{f}.pcap"}
                for i in range(30)
            ]
            flows.append(
                {"src_ip": "10.0.0.1", "src_port": 1, "dst_ip": "10.0.0.2", "dst_port": 2,
                 "proto": "TCP", "packets": packets, "source_file": f"{label}_{f}.pcap"}
            )
    results = run_study(flows, [3, 10, 100], model_names=("random_forest",))
    assert [r["packets"] for r in results] == [3, 10]  # 100 packets: no flow is long enough
    assert all(r["split"] == "random_flows" or r["split"] == "by_recording" for r in results)


def test_noise_port_flows_are_excluded_and_stay_best_effort(tmp_path):
    from qos_ml.features import is_noise_flow
    from qos_ml.train import _load_rows

    rows = _synthetic_rows(per_file=5)
    rows[0]["dst_port"] = 5355
    _write_dataset(tmp_path / "d.csv", rows)
    assert is_noise_flow(rows[0]) and not is_noise_flow(rows[1])
    assert len(_load_rows(tmp_path / "d.csv", min_packets=0)) == len(rows) - 1
    assert len(_load_rows(tmp_path / "d.csv", min_packets=0, drop_noise_ports=False)) == len(rows)

    event = _controller(tmp_path).classify(dict(rows[0]))
    assert event["label"] == "unknown" and event["enforcement"]["dscp"] == 0


def test_queue_metrics_and_report(tmp_path):
    from qos_ml.report import build_report, confidence_table, recommend_threshold
    import pandas as pd

    _write_dataset(tmp_path / "d.csv", _synthetic_rows())
    summary = build_report(tmp_path / "d.csv", tmp_path / "out", folds=4, seeds=1)
    assert summary["queue_accuracy"] >= 0.9 and summary["class_macro_f1"] >= 0.9
    for name in ("per_class.csv", "per_queue.csv", "confidence_curve.csv", "summary.csv"):
        assert (tmp_path / "out" / name).exists()
    table = confidence_table(pd.DataFrame(
        {"true": ["voip", "voip", "email"], "predicted": ["voip", "chat", "email"], "confidence": [0.9, 0.4, 0.95]}
    ))
    assert recommend_threshold(table, 1.0) == 0.3  # voip/chat share a queue, so all 3 are right
    assert table.loc[table.min_confidence == 0.5, "flows_answered"].item() == round(2 / 3, 3)


def test_synthetic_traffic_has_all_classes_and_labels_from_filenames():
    from traffic.synthetic import CLASSES, generate_flows

    flows = generate_flows(2, 3, seed=1)
    labels = {extract_features(flow)["label"] for flow in flows}
    assert labels == set(CLASSES)
    assert all(flow["source_file"].startswith("synthetic_") for flow in flows)
    assert all(len(flow["packets"]) >= 14 for flow in flows)


def test_synthetic_pcaps_round_trip_through_the_real_parser(tmp_path):
    import pytest

    pytest.importorskip("scapy")
    from traffic.capture import parse_folder
    from traffic.synthetic import generate_flows, write_pcaps

    flows = generate_flows(1, 2, seed=2, classes=("voip", "file_transfer"))
    write_pcaps(flows, tmp_path)
    parsed = build_flows(parse_folder(str(tmp_path)))
    assert sum(len(f["packets"]) for f in parsed) == sum(len(f["packets"]) for f in flows)
    assert {extract_features(f)["label"] for f in parsed} == {"voip", "file_transfer"}


def test_drift_demo_recovers_after_feedback_retraining(tmp_path):
    from demo.drift_demo import run

    result = run(drift=0.8, feedback=120, workdir=tmp_path)
    assert result["regime_a"] > 0.9
    assert result["regime_b_before"] < result["regime_a"] - 0.15
    assert result["drift_detected"] and result["last_retrain"]["promoted"]
    assert result["regime_b_after"] > result["regime_b_before"] + 0.1


def test_ping_summary_parser_on_recorded_logs():
    from network.analyze_benchmark import parse_ping

    best = parse_ping(Path("network/results/mixed_best_effort_ping.txt").read_text())
    codel = parse_ping(Path("network/results/mixed_fq_codel_ping.txt").read_text())
    htb = parse_ping(Path("network/results/mixed_qos_ping.txt").read_text())
    assert best["avg_ms"] == 0.034 and best["loss_percent"] == 0
    assert codel["avg_ms"] == 4.523 and codel["loss_percent"] == 10
    assert htb["avg_ms"] == 400.408 and htb["max_ms"] == 545.05


def test_benchmark_summary_writes_one_row_per_scenario(tmp_path):
    from network.analyze_benchmark import summarize

    for name in ("baseline", "qos"):
        (tmp_path / f"{name}_ping.txt").write_text(
            "3 packets transmitted, 3 received, 0% packet loss, time 2ms\n"
            "rtt min/avg/max/mdev = 1.0/2.0/3.0/0.5 ms\n"
        )
    rows = summarize(tmp_path, tmp_path / "summary.csv")
    assert [row["Scenario"] for row in rows] == ["baseline", "qos"]
    assert rows[0]["Ping_Avg_ms"] == 2.0 and rows[0]["Voice_Jitter_ms"] is None


def test_api_rejects_invalid_feedback_and_accepts_valid():
    import json
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from api.server import create_handler

    class Fake:
        def feedback(self, predicted, actual, flow=None):
            return False

        def status(self):
            return {}

    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(Fake()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/api/feedback"

    def post(payload):
        request = Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"}, method="POST")
        try:
            return urlopen(request, timeout=5).status
        except HTTPError as error:
            return error.code

    try:
        assert post({"predicted": "voip", "actual": "chat"}) == 200
        assert post({"predicted": "voip", "actual": "x" * 500}) == 400
        assert post({"predicted": "voip", "actual": 5}) == 400
        assert post({"predicted": "voip", "actual": "chat", "flow": "not-an-object"}) == 400
        assert post({"predicted": "voip"}) == 400
    finally:
        server.shutdown()


def test_every_policy_class_maps_to_a_known_queue():
    from qos_ml.qos_policy import CLASS_POLICY

    assert {p["classid"] for p in CLASS_POLICY.values()} <= {"1:10", "1:20", "1:30"}
    assert policy_for("voip")["dscp"] > policy_for("p2p")["dscp"]


def test_port_baseline_and_best_effort_baseline():
    from qos_ml.baselines import classify_best_effort, classify_by_port

    assert classify_by_port({"proto": "TCP", "src_port": 40000, "dst_port": 993}) == "email"
    assert classify_by_port({"proto": "UDP", "src_port": 40000, "dst_port": 3478}) == "voip"
    assert classify_by_port({"proto": "UDP", "src_port": 50000, "dst_port": 17000}) == "voip"
    assert classify_by_port({"proto": "TCP", "src_port": 50000, "dst_port": 443}) == "streaming"
    assert classify_by_port({"proto": "TCP", "src_port": 50000, "dst_port": 9999}) == "unknown"
    assert classify_best_effort({"dst_port": 443}) == "unknown"


def test_model_selection_trades_accuracy_against_latency():
    from qos_ml.model_selection import select_model

    candidates = [
        {"name": "big", "macro_f1": 0.60, "inference_ms": 20.0},
        {"name": "small", "macro_f1": 0.55, "inference_ms": 1.0},
    ]
    assert select_model(candidates, beta=0.0)[0] == "big"  # accuracy only
    assert select_model(candidates, beta=0.02)[0] == "small"  # latency matters
    assert select_model(candidates, budget_ms=5.0, beta=0.0)[0] == "small"  # budget excludes big
    assert select_model(candidates, budget_ms=0.1)[0] == "small"  # nothing fits: fastest wins
    assert select_model([])[0] is None


def test_f1_drift_metric_matches_lambda_of_algorithm_1():
    from qos_ml.drift import ErrorRateDriftDetector, weighted_f1

    assert weighted_f1([("a", "a"), ("b", "b")]) == 1.0
    assert weighted_f1([("a", "b"), ("b", "a")]) == 0.0
    detector = ErrorRateDriftDetector(window_size=4, threshold=0.5, metric="f1")
    results = [detector.update(p, a) for p, a in [("a", "a"), ("a", "b"), ("b", "a"), ("b", "a")]]
    assert results == [False, False, False, True]
    assert 0 < detector.error_rate <= 1
    try:
        ErrorRateDriftDetector(metric="bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_controller_falls_back_to_lightweight_model_over_budget(tmp_path):
    import joblib
    from qos_ml.controller import LiveController
    from qos_ml.train import train_models

    _write_dataset(tmp_path / "data.csv", _synthetic_rows())
    train_models(tmp_path / "data.csv", tmp_path, model_names=["random_forest", "decision_tree"])
    controller = LiveController(
        tmp_path / "random_forest.joblib",
        latency_budget_ms=0.0001,  # impossible budget: any real prediction exceeds it
        fallback_model=tmp_path / "decision_tree.joblib",
        min_confidence=0.0,
    )
    flow = _synthetic_rows()[0]
    assert controller.status()["using_fallback_model"] is False
    controller.classify(flow)
    assert controller.status()["using_fallback_model"] is True
    assert controller.classify(flow)["predicted_label"] == "voip"  # still answers, via the compact model


def test_controller_stays_on_primary_model_within_budget(tmp_path):
    from qos_ml.controller import LiveController

    controller = _controller(tmp_path, latency_budget_ms=10_000, min_confidence=0.0)
    controller.fallback = controller.model
    controller.classify(_synthetic_rows()[0])
    assert controller.status()["using_fallback_model"] is False
    assert controller.status()["inference_ms"] is not None


def test_report_compares_baselines_and_selects_a_model(tmp_path):
    from qos_ml.report import build_report

    _write_dataset(tmp_path / "d.csv", _synthetic_rows())
    summary = build_report(tmp_path / "d.csv", tmp_path / "out", folds=4, seeds=1)
    assert summary["selected_model"] in {"random_forest", "svm", "hist_gb", "decision_tree"}
    baselines = list(csv.DictReader((tmp_path / "out" / "baseline_comparison.csv").open()))
    assert {"port-based", "no classification (all best-effort)"} <= {row["method"] for row in baselines}
    assert (tmp_path / "out" / "model_selection.csv").exists()
