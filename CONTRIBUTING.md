# Contributing to Biopsy Scan

Thanks for your interest. Questions, bug reports, new results and code are all welcome.

## Questions about study design or the statistics?

If you're using Biopsy Scan and want to ask "how do I do X?" or "what does this number mean?", the
best place is [Discussions](https://github.com/atiladeokegab/biopsy-scan/discussions). General
survival-analysis questions also do well at [stats.stackexchange.com](https://stats.stackexchange.com/).

## Reporting bugs

Please open an [issue](https://github.com/atiladeokegab/biopsy-scan/issues/new/choose). Include
the command you ran, its output, your OS and Python version, and the commit you're on
(`git rev-parse --short HEAD`). A small reproducing snippet helps most.

## Requesting features

Open an issue with as much context as you can: the study you're planning, what Biopsy Scan can't
answer yet, and links to other tools that already do it.

## Submitting code or other changes

Please open an issue first so we can agree on the approach, then send a pull request. Keep each PR
to one change, and say what it changes and how you checked it.

## Development

### Setting up

You need Python 3.11, [uv](https://docs.astral.sh/uv/) and Node.js (for the demo tests). From the
repository root:

```sh
make install   # uv sync --dev
```

The real cohorts are only needed for real-data runs: `make data` downloads them. The test suite
runs offline on a small invented fixture.

### Running the tests

```sh
make test      # the Python test suite
make smoke     # tests + demo checks: the same check CI runs on every pull request
```

### Building the documentation

```sh
uv run --no-project --with-requirements docs/requirements.txt mkdocs serve
```

Pages live in `docs/` and the repository root (README, RESULTS, ARCHITECTURE); `mkdocs.yml` sets
the navigation.

### Rules for changes

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

By contributing you agree that your work is released under the [MIT License](LICENSE). Please
follow our [Code of Conduct](CODE_OF_CONDUCT.md).
