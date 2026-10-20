"""Class-to-DSCP policy used by Linux tc enforcement."""

from __future__ import annotations

CLASS_POLICY = {
    "voip": {"dscp": 46, "classid": "1:10"},
    "chat": {"dscp": 34, "classid": "1:10"},
    "streaming": {"dscp": 26, "classid": "1:20"},
    "email": {"dscp": 10, "classid": "1:20"},
    "p2p": {"dscp": 8, "classid": "1:30"},
    "file_transfer": {"dscp": 8, "classid": "1:30"},
}
DEFAULT_POLICY = {"dscp": 0, "classid": "1:30"}


def policy_for(label: str) -> dict[str, int | str]:
    """Return a safe default policy for an unknown or unsupported label."""
    return CLASS_POLICY.get(label, DEFAULT_POLICY).copy()

