"""The how-to's designs: a selector that picks a workflow, an email workflow that reads the
selector's estimates through two ports, and the combined design under a top policy."""

from ommat import Design, Helper, Input, Parameter, Question, Source, Table, Text
from ommat import types as t
from ommat.helpers import Part, combine

STATE = t.record(message=t.string)
ROUTES = ("email", "other")
YES_NO = t.label("yes", "no")
BAR = t.number(0, 1)


def bars(at: str) -> dict[str, Parameter]:
    """The bar and margin under ``at``."""
    return {f"{at}/bar": Parameter(BAR, 0.6), f"{at}/margin": Parameter(BAR, 0.2)}


def picked(ref: str, outputs: str, at: str) -> Table:
    """A data rule picking the answer at ``ref`` by the bar and margin under ``at``."""
    return Table(
        "d:" + ref.removeprefix("q:"),
        outputs,
        "F",
        inputs=[Helper("pick_with_margin", ref, bar=f"{at}/bar", margin=f"{at}/margin")],
        rows=[["-", Input(ref)]],
    )


# --- the selector ------------------------------------------------------------------------
SELECTOR = Design(
    name="selector",
    state=STATE,
    sources={
        "router": Source(
            {
                "q:route": Question("choice", Text("Which workflow handles it?"), ROUTES),
                "q:urgent": Question("noul", Text("Is it urgent?")),
            },
            needs=["t:state"],
        )
    },
    rules={
        "route": picked("q:route", t.label(*ROUTES), "selector"),
        "urgent": picked("q:urgent", YES_NO, "selector"),
    },
    parameters=bars("selector"),
    policy="d:route",
)

# --- the email workflow, with its ports ---------------------------------------------------
EMAIL = Part(
    Design(
        name="email",
        state=STATE,
        sources={
            "email/ask": Source(
                {"q:email/folder_named": Question("noul", Text("Does it name a folder?"))},
                needs=["t:state"],
            )
        },
        rules={
            "email/folder_named": picked("q:email/folder_named", YES_NO, "email"),
            "email/outcome": Table(
                "d:email/outcome",
                t.label("skip", "flag", "file", "ask_folder"),
                "F",
                ["d:email/chosen", "d:email/urgent", "d:email/folder_named"],
                [
                    [False, "-", "-", "skip"],
                    [True, "yes", "-", "flag"],
                    [True, "no", "yes", "file"],
                    [True, "no", "no", "ask_folder"],
                ],
            ),
        },
        parameters=bars("email"),
        policy="d:email/outcome",
    ),
    ports={"d:email/chosen": t.boolean, "d:email/urgent": YES_NO},
)

# --- the combined design -----------------------------------------------------------------
CHOSEN = Table("d:email/chosen", t.boolean, "F", ["d:route"], [["email", True], ["Known", False]])
ACT = Table(
    "d:act",
    t.label("pass", "skip", "flag", "file", "ask_folder"),
    "F",
    ["d:route", "d:email/outcome"],
    [["email", "-", Input("d:email/outcome")], ["other", "-", "pass"]],
)

DESIGN = combine(
    Part(SELECTOR),
    EMAIL,
    bind={"d:email/chosen": CHOSEN, "d:email/urgent": "d:urgent"},
    policy=ACT,
    name="assistant",
)
