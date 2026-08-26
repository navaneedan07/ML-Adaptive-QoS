import json
import csv


def read_udp_result(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    end = data["end"]
    udp = end["sum"]

    return {
        "throughput_mbps": udp["bits_per_second"] / 1_000_000,
        "jitter_ms": udp["jitter_ms"],
        "lost_packets": udp["lost_packets"],
        "packets": udp["packets"],
        "packet_loss_percent": udp["lost_percent"]
    }


best_effort = read_udp_result(
    "results/best_effort_udp.json"
)

qos = read_udp_result(
    "results/qos_udp.json"
)


with open("results/comparison.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Scenario",
        "Throughput_Mbps",
        "Jitter_ms",
        "Lost_Packets",
        "Packets",
        "Packet_Loss_Percent"
    ])

    writer.writerow([
        "Best Effort",
        best_effort["throughput_mbps"],
        best_effort["jitter_ms"],
        best_effort["lost_packets"],
        best_effort["packets"],
        best_effort["packet_loss_percent"]
    ])

    writer.writerow([
        "QoS",
        qos["throughput_mbps"],
        qos["jitter_ms"],
        qos["lost_packets"],
        qos["packets"],
        qos["packet_loss_percent"]
    ])


print("Results written to results/comparison.csv")
