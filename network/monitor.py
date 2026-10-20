"""QoS monitoring primitives for the live controller."""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable


Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _default_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


@dataclass
class QoSMonitor:
    namespace: str | None = "ns1"
    interface: str = "veth1"
    runner: Runner = _default_runner
    history: list[dict[str, float | int | str | None]] = field(default_factory=list)
    max_history: int = 1000

    def _namespace_prefix(self) -> list[str]:
        return ["sudo", "ip", "netns", "exec", self.namespace] if self.namespace else []

    def sample(self, host: str = "10.0.0.2") -> dict[str, float | int | str | None]:
        ping = self.runner([*self._namespace_prefix(), "ping", "-c", "1", "-W", "1", host])
        match = re.search(r"time[=<]([\d.]+)\s*ms", ping.stdout)
        rtt = float(match.group(1)) if match else None
        tc = self.runner([
            *self._namespace_prefix(), "tc", "-s", "qdisc", "show",
            "dev", self.interface,
        ])
        dropped = sum(int(value) for value in re.findall(r"dropped\s+(\d+)", tc.stdout))
        sample = {
            "timestamp": time.time(),
            "rtt_ms": rtt,
            "dropped_packets": dropped,
            "tc_command_ok": int(tc.returncode == 0),
        }
        self.history.append(sample)
        del self.history[:-self.max_history]
        return sample
