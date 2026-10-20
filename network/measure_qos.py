"""Measure latency and optionally run an iperf3 JSON benchmark."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def measure_ping(namespace: str = "ns1", host: str = "10.0.0.2", count: int = 20) -> float | None:
    result = run_command(["sudo", "ip", "netns", "exec", namespace, "ping", "-c", str(count), host])
    match = re.search(r"rtt .* = [\d.]+/([\d.]+)/", result.stdout)
    if result.returncode != 0 or not match:
        return None
    return float(match.group(1))


def measure_iperf(
    output_path: str | Path,
    *,
    namespace: str = "ns1",
    host: str = "10.0.0.2",
    port: int = 5001,
    duration: int = 10,
    udp: bool = True,
) -> dict:
    command = [
        "sudo", "ip", "netns", "exec", namespace, "iperf3", "-c", host,
        "-p", str(port), "-t", str(duration), "-J",
    ]
    if udp:
        command.append("-u")
    result = run_command(command)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "iperf3 failed")
    data = json.loads(result.stdout)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="ns1")
    parser.add_argument("--host", default="10.0.0.2")
    parser.add_argument("--ping-count", type=int, default=20)
    parser.add_argument("--iperf-output", type=Path)
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--tcp", action="store_true")
    args = parser.parse_args()
    print(f"Average latency: {measure_ping(args.namespace, args.host, args.ping_count)} ms")
    if args.iperf_output:
        measure_iperf(
            args.iperf_output,
            namespace=args.namespace,
            host=args.host,
            port=args.port,
            duration=args.duration,
            udp=not args.tcp,
        )
        print(f"Wrote iperf3 results to {args.iperf_output}")


if __name__ == "__main__":
    main()
