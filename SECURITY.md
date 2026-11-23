# Security policy

## Reporting a vulnerability

Please don't open a public issue. Report it privately through GitHub's
[private vulnerability reporting](https://github.com/atiladeokegab/sota-parallelism-nov21/security/advisories/new).
We aim to reply within a week.

## Scope

Preflight is research software: it runs locally on public, de-identified data and has no server
component in this repository other than the static demo. The most relevant issues are unsafe file
handling, dependency vulnerabilities, and anything in `deploy/` or the Modal jobs that could expose
credentials.

## Data

The repository contains no patient-identifiable data. The cohorts are public and are downloaded
by `make data`; they are never committed.
