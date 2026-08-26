import subprocess
import json
import statistics
import re


def run_command(command):
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )

    return result.stdout


def measure_ping():
    command = (
        "sudo ip netns exec ns1 "
        "ping -c 20 10.0.0.2"
    )

    output = run_command(command)

    match = re.search(
        r"rtt .* = [\d.]+/([\d.]+)/",
        output
    )

    if match:
        return float(match.group(1))

    return None


print("Average latency:", measure_ping(), "ms")
