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
    )
    mocked_serve.assert_called_once_with(
        mocked_controller.return_value,
        "127.0.0.1",
        8099,
        10.0,
    )
