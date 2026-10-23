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

## Fair benchmark (baseline vs QoS)

`run_benchmark.sh` runs two scenarios on the **same 10 Mbit bottleneck** with the
same load (4 bulk TCP streams) and measures a voice-like UDP probe plus ping:

```bash
DURATION=30 ./run_benchmark.sh results/benchmark   # writes summary.csv
```

- `baseline` (`setup_baseline.sh`): one FIFO queue, no classes.
- `qos` (`setup_qos.sh`): priority classes 1:10 / 1:20 / 1:30 with `fq_codel`;
  ping and UDP 5001 go to the latency-sensitive class.

Compare `Ping_Avg_ms`, `Ping_Max_ms` and `Voice_Jitter_ms` between the rows.

### Reading the older `results/mixed_*` files

Review 1 (Table III) reports three mixed-contention configurations: best effort,
HTB alone (RTT 432 ms, bufferbloat) and HTB + `fq_codel` (RTT 4.03 ms, UDP loss
14.89%, because CoDel sheds non-responsive UDP on purpose). Matching the logs here:

- `mixed_qos_ping.txt` (about 400 ms) is the **HTB-only** run.
- `mixed_fq_codel_ping.txt` (about 4.5 ms, 10% ping loss) is the **HTB + fq_codel** run.
- `mixed_best_effort_ping.txt` (0.03 ms) does **not** match the paper's best-effort
  RTT of 42.86 ms. It looks like an unloaded run.

The committed iperf files do not reproduce Table III either (recomputed from
`mixed_*_udp.json` / `mixed_*_tcp.json`):

| | Table III | committed files |
|---|---|---|
| best effort | loss 0.27%, jitter 0.641 ms, TCP 1.710 Mbps | loss 14.07%, jitter 0.463 ms, TCP 3.180 Mbps |
| HTB | loss 0%, jitter 1.966 ms, TCP 2.881 Mbps | loss 0%, jitter 2.034 ms, TCP 3.133 Mbps |
| HTB + fq_codel | loss 14.89%, jitter 0.330 ms, TCP 2.866 Mbps | no iperf files committed |

So the paper's numbers come from runs whose raw files are not in this repository.
Either add those raw files, or re-run `run_benchmark.sh` and update the paper's
table from the fresh results. The scripts were written on macOS and syntax-checked
only; they have not been executed.
