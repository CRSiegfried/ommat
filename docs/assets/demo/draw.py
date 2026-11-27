"""Draw docs/assets/demo.svg, the animation at the top of the README.

``python draw.py > ../demo.svg`` decides the three requests of ``assistant.py`` with the
assistant design and draws what happened: the questions each round sends to the model, the ones
never asked, the values and markers the rules settle, and the records.

The calls, the questions in them, the answers, the skipped questions, the values and the markers
are read from the engine's observer and its records. The timing is the
drawing's own. The output is the same on every run.
"""

import html
import math
from typing import Any

from assistant import FIND, MODEL, REMIND, VAGUE
from assistant_design import DESIGN

from ommat import decide_sync, load
from ommat.bindings import fixed

W, H = 1100, 540
BG, TEXT, DIM, LINE = "#1e1e2e", "#cdd6f4", "#7f849c", "#45475a"
BLUE, GREEN, AMBER, MAUVE, FILL = "#89b4fa", "#a6e3a1", "#f9e2af", "#cba6f7", "#313244"
CHOICES = (BLUE, "#fab387", "#f5c2e7")
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
CARD = (30, 46, 320, 54)
ROW_Y, ROW_GAP, HOME_X, HOME_GAP = 196, 54, 190, 52
OMMAT = (565, 86)
BOX_X, BOX_W, BOX_H, CHIP_H, CHIP_FITS, FLOW_Y = 410, 310, 90, 32, 38, 168
MODEL_AT = (935, 190)
RECORD_X, RECORD_GAP, RECORD_Y = 860, 75, 384
Point = tuple[float, float]
Turn = tuple[str, list[dict[str, Any]], Any]


def run() -> list[Turn]:
    """Each request with what the observer reported and its record."""
    loaded = load(DESIGN)
    turns: list[Turn] = []
    for request in (FIND, REMIND, VAGUE):
        answers = {
            ref: {"yes": a, "no": round(1 - a, 2)} if isinstance(a, float) else a
            for ref, a in MODEL[request].items()
        }
        items: list[dict[str, Any]] = []
        decision = decide_sync(
            loaded,
            {"request": request},
            {"default": fixed(answers)},
            observer=lambda _, item, seen=items: seen.append(item),
        )
        turns.append((request, items, decision.record))
    return turns


def n(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def hexagon(r: float, cx: float = 0.0, cy: float = 0.0) -> str:
    points = [
        (cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
        for a in range(0, 360, 60)
    ]
    return " ".join(f"{n(x)},{n(y)}" for x, y in points)


def idia(r: float, stroke: str, look: int) -> str:
    """A one-eyed hexagon, drawn around the origin, its pupil turned to one of eight
    directions: ``look`` counts eighths of a turn, clockwise from the right."""
    eye, pupil = r * 0.42, r * 0.17
    turn = math.radians(45 * (look % 8))
    return (
        f'<polygon points="{hexagon(r)}" fill="{FILL}" stroke="{stroke}" stroke-width="2"/>'
        f'<circle r="{n(eye)}" fill="{TEXT}"/>'
        f'<circle cx="{n(math.cos(turn) * eye * 0.5)}" cy="{n(math.sin(turn) * eye * 0.5)}" '
        f'r="{n(pupil)}" fill="{BG}"/>'
    )


OMMAT_R, OMMAT_CELL = 60.0, 15.0
OMMAT_EYES = ((-16.0, 6.0, 21.0), (24.0, -17.0, 13.0), (21.0, 21.0, 8.5))


def blob(r: float) -> str:
    """The path of a closed, smooth, uneven outline around the origin, about ``r`` across."""
    points: list[tuple[float, float]] = []
    for step in range(24):
        a = math.radians(15 * step)
        reach = r * (1 + 0.1 * math.sin(2 * a + 0.4) + 0.07 * math.sin(3 * a + 2.3))
        reach += r * 0.04 * math.sin(5 * a + 4.0)
        points.append((reach * math.cos(a), reach * math.sin(a)))
    path = f"M{n(points[0][0])} {n(points[0][1])}"
    for i in range(len(points)):
        before, here, there, after = (points[(i + k) % len(points)] for k in (-1, 0, 1, 2))
        path += " C" + " ".join(
            f"{n(x)} {n(y)}"
            for x, y in (
                (here[0] + (there[0] - before[0]) / 6, here[1] + (there[1] - before[1]) / 6),
                (there[0] - (after[0] - here[0]) / 6, there[1] - (after[1] - here[1]) / 6),
                there,
            )
        )
    return path + "Z"


def ommat(pupil_classes: list[str]) -> str:
    """A compound eye, drawn around the origin: a soft, uneven body faceted with small
    hexagons, and three eyes of three sizes, each with a glinting pupil that carries its class
    of ``pupil_classes``."""
    step = OMMAT_CELL * math.sqrt(3)
    facets = "".join(
        f'<polygon points="{hexagon(OMMAT_CELL, 1.5 * OMMAT_CELL * i, step * (j + i / 2))}"/>'
        for i in range(-5, 6)
        for j in range(-7, 8)
        if math.hypot(1.5 * OMMAT_CELL * i, step * (j + i / 2)) < OMMAT_R * 1.25 + OMMAT_CELL
    )
    body = blob(OMMAT_R)
    return (
        f'<clipPath id="ommat"><path d="{body}"/></clipPath>'
        f'<path d="{body}" fill="{FILL}"/>'
        f'<g clip-path="url(#ommat)" fill="none" stroke="{MAUVE}" stroke-opacity="0.4" '
        f'stroke-width="1.2">{facets}</g>'
        f'<path d="{body}" fill="none" stroke="{MAUVE}" stroke-width="2.5"/>'
        + "".join(
            f'<g transform="translate({n(x)},{n(y)})"><circle r="{n(r)}" fill="{TEXT}"/>'
            f'<g class="{pupil_class}"><circle r="{n(r * 0.48)}" fill="{BG}"/>'
            f'<circle cx="{n(-r * 0.17)}" cy="{n(-r * 0.17)}" r="{n(r * 0.14)}" fill="{TEXT}"/>'
            f"</g></g>"
            for (x, y, r), pupil_class in zip(OMMAT_EYES, pupil_classes, strict=True)
        )
    )


def words(
    x: float,
    y: float,
    what: str,
    colour: str = DIM,
    size: int = 15,
    anchor: str = "middle",
    mono: bool = False,
) -> str:
    family = f' font-family="{MONO}"' if mono else ""
    return (
        f'<text x="{n(x)}" y="{n(y)}" fill="{colour}" font-size="{size}" '
        f'text-anchor="{anchor}"{family}>{html.escape(what, quote=False)}</text>'
    )


def page(x: float, y: float, w: float, h: float) -> str:
    """A small sheet with lines on it, its top left at (x, y)."""
    lines = "".join(
        f'<rect x="{n(x + w * 0.18)}" y="{n(y + h * (0.2 + 0.18 * i))}" width="{n(w * part)}" '
        f'height="4" rx="2" fill="{DIM}"/>'
        for i, part in enumerate((0.64, 0.52, 0.64, 0.36))
    )
    return (
        f'<rect x="{n(x)}" y="{n(y)}" width="{n(w)}" height="{n(h)}" rx="6" fill="{FILL}" '
        f'stroke="{LINE}"/>' + lines
    )


class Drawing:
    """SVG elements, each with the keyframes that animate it over one loop."""

    def __init__(self) -> None:
        self.tracks: list[tuple[str, str, list[tuple[float, str]]]] = []
        self.body: list[str] = []

    def track(self, prop: str, first: str, changes: list[tuple[float, str]], ease: float) -> str:
        """A class whose ``prop`` starts at ``first`` and, at each (seconds, value) of
        ``changes``, moves to that value over ``ease`` seconds."""
        name = f"a{len(self.tracks)}"
        points, now = [(0.0, first)], first
        for t, value in sorted(changes, key=lambda change: change[0]):
            points += [(t, now), (t + ease, value)]
            now = value
        self.tracks.append((name, prop, points))
        return name

    def shown(self, start: float, end: float | None = None) -> str:
        """A class that fades in at ``start`` and out at ``end``, or stays to the loop's end."""
        changes = [(start, "1")] if end is None else [(start, "1"), (end, "0")]
        return self.track("opacity", "0", changes, 0.4)

    def add(self, classes: str, inner: str) -> None:
        self.body.append(f'<g class="{classes}">{inner}</g>')

    def dot(self, start: float, a: Point, b: Point, colour: str) -> None:
        """A dot that travels from ``a`` to ``b`` in 0.6 seconds."""
        seen = self.track("opacity", "0", [(start, "1"), (start + 0.6, "0")], 0.08)
        here, there = (f"translate({n(x)}px,{n(y)}px)" for x, y in (a, b))
        moving = self.track("transform", here, [(start, there)], 0.6)
        self.body.append(f'<g class="{seen}"><circle class="{moving}" r="7" fill="{colour}"/></g>')

    def style(self, length: float) -> str:
        rules = []
        for name, prop, points in self.tracks:
            frames = " ".join(
                f"{n(100 * t / length)}% {{{prop}:{v}}}"
                for t, v in [*points, (length, points[-1][1])]
            )
            rules.append(
                f"@keyframes {name} {{{frames}}} "
                f".{name} {{animation:{name} {n(length)}s ease-in-out infinite}}"
            )
        return " ".join(rules)


def answer(x: float, y: float, value: dict[str, Any], colour: str) -> str:
    """What came back for one question, under its idia at (x, y)."""
    if "probabilities" in value:
        yes = value["probabilities"]["yes"]
        return (
            f'<rect x="{n(x - 24)}" y="{n(y + 26)}" width="48" height="7" rx="3.5" '
            f'fill="{LINE}"/>'
            f'<rect x="{n(x - 24)}" y="{n(y + 26)}" width="{n(max(48 * yes, 7))}" height="7" '
            f'rx="3.5" fill="{colour}"/>' + words(x, y + 50, f"yes {yes:.2f}", DIM, 12)
        )
    found = value["value"]
    if found is None:
        return words(x, y + 40, "nothing given", DIM, 13)
    return words(x, y + 40, ", ".join(found) if isinstance(found, list) else str(found), TEXT, 13)


def settled(entry: dict[str, Any]) -> tuple[str, str]:
    """An entry's value or its marker's kind, and the colour to write it in."""
    if "marker" in entry:
        return entry["marker"]["kind"], AMBER
    return str(entry["value"]), GREEN


def chip(y: float, label: str, entry: dict[str, Any]) -> str:
    """A settled value in the middle column, its top at ``y``, named on the left. A value too
    long for one line breaks after a comma."""
    value, colour = settled(entry)
    lines = [value]
    if len(value) > CHIP_FITS and ", " in value[:CHIP_FITS]:
        cut = value.rindex(", ", 0, CHIP_FITS) + 1
        lines = [value[:cut], value[cut + 1 :]]
    height = CHIP_H + 18 * (len(lines) - 1)
    return (
        f'<rect x="{BOX_X}" y="{n(y)}" width="{BOX_W}" height="{height}" rx="9" fill="{FILL}" '
        f'stroke="{colour}" stroke-width="1.5"/>'
        + words(BOX_X - 14, y + 21, label, DIM, 14, "end", mono=label.startswith("d:"))
        + "".join(
            words(BOX_X + BOX_W / 2, y + 21 + 18 * i, line, colour, 13, mono=True)
            for i, line in enumerate(lines)
        )
    )


def draw(turns: list[Turn]) -> str:
    d = Drawing()
    sources = {name: list(source.provides) for name, source in DESIGN.sources.items()}
    refs = [ref for provides in sources.values() for ref in provides]
    home = {
        ref: (float(HOME_X + i * HOME_GAP), float(ROW_Y + row * ROW_GAP))
        for row, provides in enumerate(sources.values())
        for i, ref in enumerate(provides)
    }
    row_of = {ref: row for row, provides in enumerate(sources.values()) for ref in provides}
    looks = {ref: (3 * i + 1) % 8 for i, ref in enumerate(refs)}
    # A colour for each source that a question is named after: its own questions wear it, and
    # so does the question that decides whether they are asked.
    named = [name for name in sources if any(ref.endswith("/" + name) for ref in refs)]
    shade = {name: CHOICES[i % len(CHOICES)] for i, name in enumerate(named)}
    colour_of = {
        ref: shade.get(ref.rpartition("/")[2], shade.get(name, TEXT))
        for name, provides in sources.items()
        for ref in provides
    }
    place: dict[str, list[tuple[float, str]]] = {ref: [] for ref in refs}
    faded: dict[str, list[tuple[float, str]]] = {ref: [] for ref in refs}
    gaze: list[list[tuple[float, str]]] = [[] for _ in OMMAT_EYES]
    by_x = sorted(range(len(OMMAT_EYES)), key=lambda eye: OMMAT_EYES[eye][0])
    notes: list[tuple[float, float, str]] = []

    def look(t: float, *at: Point) -> None:
        """Turn the ommat's pupils toward ``at``, or back to the middle when there is nowhere
        to look. Given several places, the eyes take one each, left to right, and the eyes
        left over take the last."""
        for turn, eye in enumerate(by_x):
            ex, ey, r = OMMAT_EYES[eye]
            x = y = 0.0
            if at:
                goal = at[min(turn, len(at) - 1)]
                dx, dy = goal[0] - OMMAT[0] - ex, goal[1] - OMMAT[1] - ey
                far = math.hypot(dx, dy)
                x, y = dx / far * r * 0.42, dy / far * r * 0.42
            gaze[eye].append((t, f"translate({n(x)}px,{n(y)}px)"))

    def to(x: float, y: float) -> str:
        return f"translate({n(x)}px,{n(y)}px)"

    # The design, on the left: every question an idia, in a row per source.
    d.add(d.shown(0.3), words(175, 152, f"the design: {len(refs)} questions"))
    for row, name in enumerate(sources):
        d.add(
            d.shown(0.3),
            words(40, ROW_Y + row * ROW_GAP + 5, name, shade.get(name, DIM), 13, "start"),
        )

    record_at = [(RECORD_X + RECORD_GAP * i, RECORD_Y) for i in range(len(turns))]

    t = 0.6
    for index, (request, items, record) in enumerate(turns):
        scene: list[tuple[float, str]] = []
        start = t
        pace = 1.0 if index == 0 else 0.85
        card_x, card_y, card_w, card_h = CARD
        scene.append(
            (
                t,
                words(card_x + card_w / 2, card_y - 14, "the request")
                + f'<rect x="{card_x}" y="{card_y}" width="{card_w}" height="{card_h}" rx="10" '
                f'fill="{FILL}" stroke="{LINE}"/>'
                + words(card_x + card_w / 2, card_y + 32, f"“{request}”", TEXT, 14),
            )
        )
        card_edge = (card_x + card_w + 10, card_y + card_h / 2)
        look(t + 0.2 * pace, card_edge)
        d.dot(t + 0.7 * pace, card_edge, (OMMAT[0] - 72, OMMAT[1]), TEXT)
        arrives = (
            f"a request arrives; the design holds {len(refs)} questions"
            if index == 0
            else "another request arrives"
        )
        notes.append((t, t + 1.8 * pace, arrives))
        t += 1.8 * pace

        entries = [item for item in items if item["item"] == "entry"]
        last = [e["ref"] for e in entries if e["ref"].startswith("d:")][-1]
        unsure = {ref for e in entries if "marker" in e for ref in e["marker"]["cause"]["refs"]}
        asked = sum(1 for e in entries if e["ref"] in home and "value" in e)
        calls = 0
        flow = float(FLOW_Y)
        causes: set[tuple[str, ...]] = set()
        done: set[str] = set()
        position = 0
        while position < len(items):
            item = items[position]
            position += 1
            if item["item"] == "call":
                calls += 1
                group = list(item["questions"])
                rest = items[position:]
                stop = next((i for i, x in enumerate(rest) if x["item"] == "call"), len(rest))
                skipped = [
                    e["ref"]
                    for e in rest[:stop]
                    if e["item"] == "entry" and e["ref"] in home and "marker" in e
                ]
                box = (
                    f'<rect x="{BOX_X}" y="{n(flow)}" width="{BOX_W}" height="{BOX_H}" rx="12" '
                    f'fill="none" stroke="{LINE}" stroke-width="2" stroke-dasharray="6 6"/>'
                    + words(BOX_X - 14, flow + BOX_H / 2 + 5, f"call {calls}", DIM, 14, "end")
                )
                scene.append((t, box))
                slots = {
                    ref: (BOX_X + BOX_W * (i + 0.5) / len(group), flow + 30)
                    for i, ref in enumerate(group)
                }
                for i, ref in enumerate(group):
                    place[ref].append((t + 0.3 + 0.12 * i, to(*slots[ref])))
                # An eye on each question, following it from its row into the call.
                look(t, *(home[ref] for ref in group))
                halfway = [
                    ((home[ref][0] + slots[ref][0]) / 2, (home[ref][1] + slots[ref][1]) / 2)
                    for ref in group
                ]
                look(t + 0.45, *halfway)
                look(t + 0.75, *(slots[ref] for ref in group))
                for ref in skipped:
                    faded[ref].append((t + 0.3, "0.28"))
                for row in sorted({row_of[ref] for ref in skipped}):
                    tag_x = max(home[ref][0] for ref in skipped if row_of[ref] == row) + 30
                    tag = words(tag_x, ROW_Y + row * ROW_GAP + 4, "not asked", DIM, 12, "start")
                    scene.append((t + 0.4, tag))
                near = (BOX_X + BOX_W + 12, flow + BOX_H / 2)
                far = (MODEL_AT[0] - 74, MODEL_AT[1] + (near[1] - MODEL_AT[1]) * 0.25)
                wire = (
                    f'<line x1="{n(near[0])}" y1="{n(near[1])}" x2="{n(far[0])}" '
                    f'y2="{n(far[1])}" stroke="{LINE}" stroke-width="2"/>'
                )
                scene.append((t + 1.0, wire))
                look(t + 1.4 * pace, MODEL_AT)
                d.dot(t + 1.5 * pace, near, far, TEXT)
                d.dot(t + 2.2 * pace, far, near, GREEN)
                for e in rest[:stop]:
                    if e["item"] == "entry" and e["ref"] in slots and "value" in e:
                        colour = AMBER if e["ref"] in unsure else GREEN
                        scene.append((t + 2.8 * pace, answer(*slots[e["ref"]], e["value"], colour)))
                round_ = item["round"]
                if calls == 1:
                    note = f"round {round_}: the {len(group)} questions that are ready go in one"
                    note += " model call"
                else:
                    note = f"round {round_}: {len(group)} more questions are asked"
                    if skipped:
                        note += f", and the other {len(skipped)} never are"
                notes.append((t, t + 3.2 * pace, note))
                done.update(skipped)
                flow += BOX_H + 12
                t += 3.2 * pace
            elif item["item"] == "entry" and item["ref"] in done:
                continue
            elif item["item"] == "entry" and item["ref"] in home and "marker" in item:
                skipped = [item["ref"]]
                while (
                    position < len(items)
                    and items[position]["item"] == "entry"
                    and items[position]["ref"] in home
                    and "marker" in items[position]
                ):
                    skipped.append(items[position]["ref"])
                    position += 1
                rows = sorted({row_of[ref] for ref in skipped})
                look(t, *((HOME_X + HOME_GAP / 2, ROW_Y + row * ROW_GAP) for row in rows))
                for ref in skipped:
                    faded[ref].append((t + 0.2, "0.28"))
                for row in sorted({row_of[ref] for ref in skipped}):
                    tag_x = max(home[ref][0] for ref in skipped if row_of[ref] == row) + 30
                    tag = words(tag_x, ROW_Y + row * ROW_GAP + 4, "not asked", DIM, 12, "start")
                    scene.append((t + 0.3, tag))
                took = 1.8 * pace
                notes.append((t, t + took, f"so the other {len(skipped)} questions are not asked"))
                t += took
            elif item["item"] == "entry" and item["ref"].startswith("d:"):
                ref = item["ref"]
                look(t, (BOX_X + BOX_W / 2, flow + CHIP_H / 2 + 60))
                d.dot(t + 0.1, (OMMAT[0], OMMAT[1] + 70), (BOX_X + BOX_W / 2, flow - 6), MAUVE)
                scene.append((t + 0.7, chip(flow, ref, item)))
                if "marker" in item:
                    cause = tuple(item["marker"]["cause"]["refs"])
                    kind = item["marker"]["kind"]
                    if cause in causes:
                        note = f"{ref} carries the same {kind} and its cause to the caller"
                    else:
                        note = f"no clear answer at {cause[0]}: {ref} is the marker {kind}"
                    causes.add(cause)
                else:
                    note = f"a rule reads the answers and settles {ref}"
                took = 1.8 * pace
                if ref == last:
                    # The record of the turn.
                    spot = record_at[index]
                    took = 2.8 * pace
                    look(t + 0.9 * pace, spot)
                    d.dot(t + 1.0 * pace, (OMMAT[0] + 50, OMMAT[1] + 40), spot, MAUVE)
                    d.add(
                        d.shown(t + 1.0 * pace + 0.6),
                        page(spot[0] - 20, spot[1], 40, 50)
                        + words(spot[0], spot[1] + 68, f"record {index + 1}", DIM, 12),
                    )
                    plural = "call" if calls == 1 else "calls"
                    note = f"asked {asked} of {len(refs)} questions in {calls} model {plural}; "
                    note += (
                        "the turn is saved as a record"
                        if "value" in record.entries[last]
                        else "the caller hands this turn on"
                    )
                notes.append((t, t + took, note))
                flow += CHIP_H + 14
                t += took

        for at, inner in scene:
            d.add(d.shown(at, t), inner)
        for ref in refs:
            if place[ref] and place[ref][-1][0] > start:
                place[ref].append((t, to(*home[ref])))
            if faded[ref] and faded[ref][-1][0] > start:
                faded[ref].append((t, "1"))
        look(t)
        t += 0.7

    length = t

    # The questions, the ommat and the model, which stay through the loop.
    for i, ref in enumerate(refs):
        moving = d.track("transform", to(*home[ref]), place[ref], 0.8)
        fading = d.track("opacity", "1", faded[ref], 0.4)
        d.add(
            d.shown(0.3 + 0.04 * i),
            f'<g class="{fading}"><g class="{moving}">'
            f"{idia(17, colour_of[ref], looks[ref])}</g></g>",
        )
    pupils = [d.track("transform", to(0, 0), turns_of, 0.3) for turns_of in gaze]
    d.add(
        d.shown(0.2),
        f'<g transform="translate({OMMAT[0]},{OMMAT[1]})">{ommat(pupils)}</g>'
        + words(OMMAT[0] + 80, OMMAT[1] - 2, "ommat", MAUVE, 16, "start")
        + words(OMMAT[0] + 80, OMMAT[1] + 17, "decides what to ask next", DIM, 13, "start"),
    )
    d.add(
        d.shown(0.4),
        f'<g transform="translate({MODEL_AT[0]},{MODEL_AT[1]})">{idia(58, GREEN, 4)}</g>'
        + words(MODEL_AT[0], MODEL_AT[1] + 90, "the model"),
    )
    for start, end, what in notes:
        d.add(d.shown(start + 0.1, end - 0.4), words(W / 2, H - 28, what, TEXT, 19))

    whole = d.track("opacity", "0", [(0.0, "1"), (length - 0.6, "0")], 0.5)
    style = "text{font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif} "
    title = (
        f"An ommat, a compound eye, decides {len(turns)} requests with a design of "
        f"{len(refs)} questions: each round it sends the model only the questions that are "
        f"ready, and rules settle a value or a marker from the answers"
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img">\n'
        f"<title>{title}</title>\n<style>{style}{d.style(length)}</style>\n"
        f'<rect width="{W}" height="{H}" rx="14" fill="{BG}"/>\n'
        f'<g class="{whole}">\n' + "\n".join(d.body) + "\n</g>\n</svg>\n"
    )


if __name__ == "__main__":
    print(draw(run()), end="")
