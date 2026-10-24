# Contributing

Thanks for your interest in ommat. The design and the spec are still young, so contributions start with an issue.

## Issues first

- **Bugs:** open an issue with a minimal reproduction, the ommat and Python versions, and what you expected.
- **Changes and features:** open a proposal issue before writing code. Say what problem it solves and how it bears on the spec's guarantees.
- **Pull requests** are welcome once an issue is agreed. Link it in the PR. A PR without an agreed issue may be closed, however good the code is.
- **Security problems:** do not open an issue; follow [SECURITY.md](SECURITY.md).

## Development

The `idia` extra's development source is a checkout of idia beside this one, at `../idia`, so clone it first:

```sh
git clone https://github.com/CRSiegfried/idia.git ../idia
uv sync
uv run ruff check
uv run ruff format --check
uv run pyright
uv run pytest
```

CI runs the same checks on Python 3.11–3.14 on Linux, macOS and Windows. On pushes to main it also runs `scripts/check_subjects.py`, which checks that commit subjects describe library changes.

## Ground rules for changes

- **The core imports only the standard library.** Integrations with third-party SDKs go in optional extras and are imported lazily.
- **The spec is normative.** A change that alters behavior the spec describes changes the spec in the same PR, with tests.
- **The public API is what `ommat.__all__` lists, and the `__all__` of `ommat.types`, `ommat.helpers`, `ommat.evaluation` and `ommat.bindings`.** Modules and names starting with an underscore are internal. Removing or changing a public name needs a deprecation first: one minor release with a `DeprecationWarning`. The conformance suite, `ommat.bindings.conformance`, the conversation renderers, `ommat.conversation`, and the shipped bindings, such as `ommat.bindings.idia` and its `oracle_binding`, are outside it: a shipped binding follows its provider library while that library is 0.x, so it may change in any minor release when the library changes, and the changelog says so.
- **Record user-visible changes** under `Unreleased` in [CHANGELOG.md](CHANGELOG.md).

## Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
