"""The design the stored format-4 record in ``tests/format_4`` ran under, kept in a module of
its own so its document loads from bytes: one source asking a ``noul`` of the oracle ``o``."""

from ommat import Design, Plan, Policy, Question, ReadView, Source, Text
from ommat import types as t


def plan(r: ReadView) -> Plan:
    return Plan("inform", reply="asked")


DESIGN = Design(
    name="format-4",
    state=t.record(text=t.string),
    sources={"a": Source({"q:a/yes": Question("noul", Text("Is it so?"))}, oracle="o")},
    policy=Policy(plan, version="1", reads=[]),
)
