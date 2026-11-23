# Contributing to Preflight

Thanks for helping. Bug reports, questions, new results and fixes are all welcome.

## Before you start

- **Questions and bugs:** open an issue. Include your OS, Python version, the command you ran and
  its output.
- **Bigger changes:** open an issue first, so we can agree on the approach before you write code.

## Development

```sh
make smoke   # install, offline test suite, demo checks: the same check CI runs on every PR
make demo    # serve the demo at http://localhost:8000
```

Requirements are in the [README](README.md#quick-start). The real cohorts are only needed for
real-data runs (`make data`).

## Rules for changes

1. **Every number has a source.** A new result is a committed file under `bench/results/` or
   `results/`, and any number quoted in the docs is added to `results/claims.json`.
   `python3 -I scripts/check_claims.py` must print `ok`.
2. **Every number has an evidence label:** MEASURED, ESTIMATE, ANALYTIC, SYNTHETIC or UNMEASURED.
3. **Kernels match the reference.** A new or changed re-imaging kernel needs a test against
   `core/reimage_ref.py`.
4. **Frozen means frozen.** The pattern grammar, the locked held-out split and the pre-registered
   rule ([docs/PREREGISTRATION.md](docs/PREREGISTRATION.md)) don't change; new versions are added
   alongside.
5. **Tests with the change.** A bug fix comes with a test that fails without it.

## Pull requests

Keep each PR to one change, describe what it changes and how you checked it, and make sure
`make smoke` passes. By contributing you agree that your work is released under the
[MIT License](LICENSE).

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md).
