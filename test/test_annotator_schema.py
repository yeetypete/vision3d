"""The annotator's SceneUpdate schema must agree with the ROS definitions.

A bag carries a copy of every message definition it uses. When two schemas in
one file define the same type differently, a reader (Foxglove, for one) reports
a conflict and may fall back to defaults for the whole type -- so a missing
field default in the hand-assembled SceneUpdate schema breaks a recording that
is otherwise fine. The definitions below are the upstream ones, verbatim.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SCHEMA = (
    Path(__file__).parent.parent
    / "tools"
    / "annotator"
    / "schemas"
    / "foxglove_SceneUpdate.msg"
)

#: Upstream field lists for the types the schema shares with other ROS
#: messages, defaults included. From ``geometry_msgs`` and
#: ``builtin_interfaces``.
CANONICAL = {
    "geometry_msgs/Quaternion": [
        "float64 x 0",
        "float64 y 0",
        "float64 z 0",
        "float64 w 1",
    ],
    "geometry_msgs/Vector3": ["float64 x", "float64 y", "float64 z"],
    "geometry_msgs/Point": ["float64 x", "float64 y", "float64 z"],
    "geometry_msgs/Pose": [
        "Point position",
        "Quaternion orientation",
    ],
    "builtin_interfaces/Time": ["int32 sec", "uint32 nanosec"],
    "builtin_interfaces/Duration": ["int32 sec", "uint32 nanosec"],
}


def definitions() -> dict[str, list[str]]:
    """Parse the concatenated schema into field lists per type.

    Returns:
        Type name to its field lines, comments and blanks removed.
    """
    out: dict[str, list[str]] = {}
    for part in re.split(r"^=+\s*$", SCHEMA.read_text(), flags=re.MULTILINE):
        match = re.match(r"\s*MSG:\s*(\S+)\s*\n(.*)", part, re.DOTALL)
        if match is None:
            continue
        fields = [
            line.strip()
            for line in match.group(2).splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        out[match.group(1)] = fields
    return out


def resolved(fields: list[str]) -> list[str]:
    """Drop package prefixes from field types.

    ``geometry_msgs/Point position`` and ``Point position`` are the same
    declaration -- a bare name resolves within the message's own package -- and
    a reader compares the resolved form. Defaults are kept, since those are
    what actually differ between definitions.

    Args:
        fields: Field lines from one type.

    Returns:
        The same lines with any ``package/`` prefix removed.
    """
    out = []
    for field in fields:
        head, _, tail = field.partition(" ")
        out.append(f"{head.rsplit('/', 1)[-1]} {tail}".strip())
    return out


@pytest.mark.parametrize("name", sorted(CANONICAL))
def test_shared_type_matches_upstream(name: str) -> None:
    parsed = definitions()
    assert name in parsed, f"{name} is not defined in {SCHEMA.name}"
    # Field defaults count: dropping "float64 w 1" is what made Foxglove reject
    # a bag whose tf messages carry the full definition.
    assert resolved(parsed[name]) == resolved(CANONICAL[name])


def test_every_referenced_type_is_defined() -> None:
    """A type used but not defined leaves the schema unparseable."""
    parsed = definitions()
    known = set(parsed) | {n.split("/")[-1] for n in parsed}
    primitives = {
        "bool",
        "byte",
        "char",
        "float32",
        "float64",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "string",
    }
    for owner, fields in parsed.items():
        for field in fields:
            type_name = field.split()[0].removesuffix("[]")
            type_name = re.sub(r"\[\d*\]$", "", type_name)
            assert type_name in primitives or type_name in known, (
                f"{owner} references undefined type {type_name}"
            )
