"""The design stored in ``tests/format_8``, a format-8 document as the step-19 engine wrote it,
kept in a module of its own so its document loads from bytes. It is all tables, so its hash
rests on no function's source, and holds no ``known`` cell, which format 8 does not have."""

from ommat import Design, Origin, Parameter, PlanDeclaration, PolicyTable, Step, Table, Text
from ommat import types as t

DESIGN = Design(
    name="format-8",
    state=t.record(score=t.number(0, 1), name=t.string),
    parameters={"bar": Parameter(t.number(0, 1), 0.8)},
    rules={
        "level": Table(
            "d:level",
            t.label("high", "low"),
            "F",
            ["t:state.score"],
            [[">= bar", "high"], ["-", "low"]],
        ),
    },
    tools=["notify"],
    plans={
        "page": PlanDeclaration("execute", [Step("notify", {"who": Origin(ref="t:state.name")})]),
        "tell": PlanDeclaration("inform", reply=Text("Nothing urgent.")),
    },
    policy=PolicyTable(["d:level"], [["high", "page"], ["-", "tell"]]),
)
