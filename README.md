# ommat

A blackboard engine that turns a decision model's answers into one checked plan. Its one operation is **decide**, which runs two stages:

- **Interpret.** Sources fetch model answers and other outside values onto a blackboard in rounds, through bindings the caller supplies; rules, which are functions with declared reads, turn them into estimates. An unknown stays unknown, as a marker (`Unsure`, `Unavailable` or `Unasked`), never a quiet no or a `None`.
- **Compile.** One policy reads the estimates and writes the plan: execute these steps, clarify, inform, or escalate. The guard then checks every requirement the design declares for the plan and every requirement each step declares, refuses a step that declares none unless the design turns that off, and passes or refuses the plan.

Decide returns the plan, the guard's verdict and a record of the turn, and calls no tool. The caller runs the plan's steps with its own workflow engine, or with the small reference runner ommat ships. Nothing after decide consults the model, and the record replays the turn with no model call.

New to ommat? Start with the tutorial, [Write your first design](docs/tutorials/write-a-design.md). Then [Decide with real models through idia](docs/tutorials/decide-with-idia.md) binds a design to model providers.

**The name.** Ommat and Idia come from *ommatidia*, the thousands of small hexagonal eyes that make up a compound eye and combine into one picture: *ommat-* is the Greek stem for "eye", *-idia* "little ones". Many small, cheap questions add up to one full picture. Ommat is the engine that asks them; Idia puts them to the models.

## Install

ommat is installed from its git tag until it is on a package index:

```sh
pip install "ommat @ git+https://github.com/CRSiegfried/ommat@v0.4.0"
```

`ommat[idia]` is the binding ommat ships to decision-model providers, through the [idia](https://github.com/CRSiegfried/idia) library; install idia from its tag first, with the extras for the providers you use, as the [tutorial](docs/tutorials/decide-with-idia.md) shows:

```sh
pip install "idia[openai,cloudflare] @ git+https://github.com/CRSiegfried/idia@v0.1.0"
pip install "ommat[idia] @ git+https://github.com/CRSiegfried/ommat@v0.4.0"
```

## Status

Pre-release, at 0.4.0. `load` checks a design and refuses an invalid one. `decide` runs interpret over rounds of sources and rules, then compile with the policy and the guard, and returns the plan, the verdict and a record that `replay` reruns with no model call. Rules and the policy are written as Python functions. A question is a `noul`, `choice` or `score`, answered with a probability per label (a label may carry a description), or an `extraction`, a typed value read from a text. The record document carries its own hash, and `ommat.evaluation.use_answers` reruns one turn with another record's answers, to compare two models. The escalation and redacted views, the helpers, two design lints and the reference runner ship with it. It writes format-4 records and design documents, and still reads formats 1 to 3, which earlier releases wrote. `ommat[idia]` binds an oracle to decision-model providers through idia. The core imports only the Python standard library ([ADR-0001](docs/adr/0001-write-the-core-in-python-with-no-runtime-dependencies.md)).

0.4.0 leaves out decision tables, policy tables, data rules, plan declarations, candidate sets, candidate-valued label sets, and grounded declarations on a source or rule; `load` refuses a design that uses one. No state carries from one turn to the next: each turn stands alone. The written spec is partial, and its conformance fixtures come later ([docs/spec](docs/spec/README.md)). [CHANGELOG.md](CHANGELOG.md) lists each release.

## Contributing

Issues first: see [CONTRIBUTING.md](CONTRIBUTING.md). Report vulnerabilities privately, as described in [SECURITY.md](SECURITY.md). This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## License

MIT. See [LICENSE](LICENSE).
