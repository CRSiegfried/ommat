# Security policy

## Supported versions

Until 1.0, only the latest release receives fixes. After 1.0, the latest minor release of each supported major version does.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub: open the repository's **Security** tab and choose **Report a vulnerability** (<https://github.com/CRSiegfried/ommat/security/advisories/new>). Do not open a public issue.

Include what you found, how to reproduce it, and the versions affected. You can expect an acknowledgement within a week. Once a fix is ready, the advisory is published with credit to you unless you prefer otherwise.

## What counts

ommat runs code you supply (oracle bindings, function `when`s, rules, the policy, the codec, the observer, and the `functions` mapping given to `load`) in your process; it does not sandbox it. Reports in scope include ways ommat itself breaks a guarantee it documents, for example:

- decide returning anything but the policy's entry as the decision, or a marker entry not read as undecided
- the model being consulted after decide returns
- a rule reading something it did not declare
- a redacted view showing in clear a value digested under the applied policy, or a digest that discloses the value

The record holds every value of a turn and is as sensitive as the turn's inputs; redaction is a separate view of it. The core makes no network calls and has no runtime dependencies. Issues in optional extras' third-party SDKs belong with those projects.
