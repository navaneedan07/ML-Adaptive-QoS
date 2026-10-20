"""Linux tc/iptables enforcement adapter used by the live controller.

The adapter is deliberately command-list based (never a shell string). Use
``dry_run=True`` for development and Windows; set ``dry_run=False`` only on a
Linux host where the configured interface and privileges are available.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Callable

from qos_ml.qos_policy import policy_for


Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _default_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


@dataclass
class TcEnforcer:
    namespace: str | None = "ns1"
    interface: str = "veth1"
    dry_run: bool = True
    runner: Runner = _default_runner
    applied: list[dict[str, object]] = field(default_factory=list)

    def _run(self, command: list[str]) -> None:
        if self.dry_run:
            return
        result = self.runner(command)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"command failed: {' '.join(command)}")

    def _namespace_prefix(self) -> list[str]:
        return ["sudo", "ip", "netns", "exec", self.namespace] if self.namespace else []

    def apply(self, label: str, flow: dict[str, object]) -> dict[str, object]:
        """Mark and classify one flow using its protocol and destination port."""
        policy = policy_for(label)
        proto = str(flow.get("proto", "")).lower()
        dst_port = int(flow.get("dst_port", 0))
        dscp = int(policy["dscp"])
        classid = str(policy["classid"])
        match = ["protocol", proto, "dst_port", str(dst_port)] if proto in {"tcp", "udp"} else []

        tc = [
            *self._namespace_prefix(), "tc", "filter", "replace",
            "dev", self.interface, "protocol", "ip", "parent", "1:", "prio", "10",
            "flower", *match, "flowid", classid,
        ]
        self._run(tc)

        # DSCP is applied in the namespace's mangle table. -C makes repeated
        # classifications idempotent instead of adding an unbounded rule set.
        if match:
            iptables_match = [
                *self._namespace_prefix(), "iptables", "-t", "mangle",
                "-C", "POSTROUTING", *match, "-j", "DSCP", "--set-dscp", str(dscp),
            ]
            check = self.runner(iptables_match) if not self.dry_run else None
            if self.dry_run or (check and check.returncode != 0):
                add = iptables_match.copy()
                add[add.index("-C")] = "-A"
                self._run(add)

        result = {"label": label, "dscp": dscp, "classid": classid, "dry_run": self.dry_run}
        self.applied.append(result)
        return result
