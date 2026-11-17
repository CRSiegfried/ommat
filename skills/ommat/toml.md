# Authoring a design as TOML

Read this when you write, change or review a design, or a slice of one, as a `*.toml` file instead of Python. `ommat.authoring` is on `main`, not in a tagged release, and carries no stability promise before 1.0.

A file holds a design's declarations: its state, sources, questions, labels, parameters, targets, decision tables, and the names of its rule functions. Rule logic stays in Python. The reader builds the `Design` you would have written, before `load` and outside the hash, so a TOML design and its Python twin load to one design hash.

```python
import tomllib
from ommat import load
from ommat.authoring import design_from, slice_from

with open("route.toml", "rb") as f:
    doc = tomllib.load(f)  # the reader takes a parsed mapping and reads no file
loaded = load(design_from(doc, version="3", values={"route/bar": 0.7}))
```

`slice_from(doc, ...)` returns a `Slice(sources, rules, parameters)`: pieces written together that do not load alone. Merge its three mappings into the `sources`, `rules` and `parameters` of the `Design` you build. A design or slice is wholly TOML or wholly Python; designs of either form still combine with `combine`.

## Trust and errors

The reader imports each rule's `fn`, which runs that module's top-level code. A document is as trusted as source code: never read one you would not import. A slip it cannot build raises `AuthoringError`, a `ValueError` whose message names the path, such as `rules.op: unknown key 'read'`. Everything about the design itself (an undeclared read, a missing `provides`, a bad type) is `load`'s to refuse, with `InvalidDesign`.

## What a document holds

Top level: `module`, `version`, `[types]`, `[labels]`, `[parameters]`, `[sources.*]`, `[rules.*]`; a whole design adds `state` and `policy` (both required), `name` and `[targets]`. Any other key is refused.

- **module.** `fn = "op"` is a name in the top-level `module`; `fn = "pkg.mod:name"` is a full path. A rename or move changes the design hash, which covers the function's module, qualname and source.
- **version.** Every function rule's version unless a rule gives its own; a table has none. Give it in the file or from the caller (`version=`), never both.
- **[types].** Names for type strings; `$name` inside a rule's `type`, a source's `fetch` or a parameter's `type` expands it, nested. Write the engine's canonical text, `label{"a", "b"}` sorted; `load` reports a non-canonical one with the right spelling. `{ required = true }` leaves the string to the caller's `types={...}`.
- **[parameters].** `"path" = { type = "$unit", value = 0.9 }` writes the value; omitting `value` leaves it to the caller's `values={"path": 0.7}`. A value has one owner: a path the file gives may not be in `values`, and one it leaves must be.
- **Source.** `oracle`, `needs`, `only_when = { "d:x" = ["v"] }` (asked only when that rule holds one of the values), `grounded`, `timeout`, `cost`, `ask` (the questions) and `fetch` (`"t:ref" = "$type"`).
- **Question.** One key names the kind and holds the wording: `noul`, `choice`, `score` or `extraction`. Then `labels` (a list of `{ name, description }`, or the name of a `[labels]` entry), `target`, `latent = { "tg:x" = "role" }`, `value_type`, `is_list`, `text` and `reference_date`. `text = "t:state.request"` is wrapped for you; write `reference_date = { ref = "t:state.today" }`. `slot = "candidate"` fills `{candidate}` once per candidate of the source's `grounded` rule; `slot = { name = "d:rule" }` names another rule.
- **Rule.** `fn`, `type` (what the function returns), `reads`, `params`, `candidates`, `grounded`, `target`, `provides` (default `d:<rule id>`), `version`. A `params` entry may be a pattern, `"route/bar/*"`, matched against the file's own parameters; one that matches nothing is refused.
- **Target.** `"tg:x" = { statement = "...", values = "..." }`, whole designs only.

A wording that holds both quote kinds needs a basic string with escapes. A TOML 1.0 inline table is one line, so a long declaration takes a `[table]` header of its own.

## Decision tables

A table is a rule with `rows` in place of `fn`: named inputs, then sparse rows. A row names only the inputs it tests (an omitted input matches anything) and returns `then`; the first matching row wins.

```toml
[rules.short_circuit]
type   = 'label{"list_folders", "list_rules", "none"}'
target = "tg:operation"
hit    = "F"
inputs.folders = { mass = "q:intent", labels = "email/sc/list_folders" }
inputs.rules   = { mass = "q:intent", labels = "email/sc/list_rules" }
rows = [
  { folders = ">= email/bar/short_circuit_list_folders", then = "list_folders" },
  { rules   = ">= email/bar/short_circuit_list_rules",   then = "list_rules" },
  { then = "none" },
]
```

- **Inputs** keep their order, which is hashed: a ref (`"t:account.x"`), or a helper call whose one helper key (`mass`, `pick_with_margin`, `pick_with_thresholds`, `size`, `extract`) holds the ref it reads and whose other keys are parameter paths.
- **Cells** are the engine's forms as strings: `"Unasked"`, `"Unsure or Unavailable"`, `"Known"`, `">= param/path"`, `"< param/path"`, `"in d:ref"`, a set `"{done, empty}"`, `"-"` (any), `"Undetermined"` (any marker), an interval `"[0, 5]"`, an integer operand `">= 3"`; native `true`, `false` and integers; a set as `{ one_of = [...] }`. A bare string is a label. A label spelt like one of those forms is `{ is = "label" }`. `hit` is `"F"` only.
- **then** is a literal, or `{ ref = "q:intent" }` to return what a helper over that ref returns.
- **Order is the logic.** Comment each row that needs a reason. End with `{ then = ... }` if the table should answer when nothing matches. Use `Known`, not `-`, when a marker must pass on instead of matching; `Known` over an `Unavailable` input is a plain no-match, it does not escalate. Where every failure must escalate, use `{ one_of = [true, false] }` over a boolean.
- **A copy** is `like = "other_table"` plus only what differs: `provides`, `type`, `target` or `inputs.<name>`. It cannot write rows, add an input or copy a copy. Leave a table that differs inside a row's cell written out twice, with a comment; there is no loop, any-of row or template, because each hides the rows the engine reports by position.
- Tables the form does not spell: hit policies other than `F`, the `Param` and `Unsure` outputs, `order`, `candidates`, `grounded`.

## Writing one

1. Start from the Python. Move declarations (sources, questions, labels, parameters, targets, types) into the file; leave rule functions where they are, and name each in `fn`.
2. Copy each function's `reads` and `params` by hand; the engine refuses an undeclared read at decide time, and nothing flags one declared and never read.
3. Check before you delete the Python: the design hash must equal the Python's. A slice cannot be checked alone, so compare the merged designs.
4. A comment moved into the file does not move the hash; a docstring edited inside a rule function does.
5. To read a design built from slices, print `ommat.outline` over the loaded design.

The caller owns `version`, `values` and `types`, passed as arguments; never edit a copy of the document, because then the file no longer says what the design is. Write generated entries (one parameter per label) out, and let `params` patterns cover repeats; a design whose entries are really generated is wholly Python.

## Not spelt yet

Write these in Python: rules generated from arguments, a source's `when`, `payload` and `feedback`, a design's `groups`, a `fetch` with a candidates mapping, a rule's `boundaries`, parameters built from a model dump, a question built per call, and a reference to a Python object for a wording or target.

More: [the spec's section on the reader](https://github.com/CRSiegfried/ommat/blob/main/docs/spec/README.md#the-authoring-reader).
