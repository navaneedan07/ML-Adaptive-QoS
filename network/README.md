# Network QoS benchmark

These scripts run the Linux-only part of the project in two network
namespaces. They require `iproute2`, `iperf3`, `ping`, and `sudo`; they are
intended for Linux or WSL with the required kernel capabilities.

## Run

```bash
./setup_network.sh
iperf3 -s -B 10.0.0.2 -p 5001 --daemon
./setup_qos.sh
./measure_qos.sh results/qos_udp.json
python3 analyze_results.py results/best_effort_udp.json results/qos_udp.json results/comparison.csv
./remove_qos.sh
./cleanup_network.sh
```

`setup_qos.sh` creates HTB classes with `fq_codel` child queues and maps UDP
port 5001 to the latency-sensitive class. Environment variables `NAMESPACE`
and `DEVICE` can override the defaults.

The Python measurement and analysis modules are import-safe and accept explicit
paths, so they can also be used from CI or from a notebook.
