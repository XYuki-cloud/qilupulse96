# Local release checklist

This checklist records the gate for maintaining the private preview repository
and for any later public release. A private-preview push may update the
existing reviewed remote, but it must not upload real runtime inputs, reports,
or model artifacts.

## Repository boundary

- [ ] The final remote operation is explicitly authorized and uses an ordinary
      push; no force-push is needed for this data-package release.
- [ ] Public history contains only clean-release commits.
- [ ] The original private production repository remains unchanged.
- [ ] `git status --short --branch` is clean.
- [ ] The staged file list has been reviewed against the inclusion/exclusion
      policy.

## Code and legal provenance

- [ ] Every retained source row in `docs/PROVENANCE.md` has an
      author/source/license decision or maintainer publication confirmation.
- [ ] Any retained third-party notice is recorded in
      `THIRD_PARTY_NOTICES.md`.
- [ ] `XYuki` authorship or relicensing authority is confirmed; it is not
      inferred only from the configured Git name or email.
- [ ] No production weights are present unless the model release gate passes.

## Security and data boundary

- [ ] Secret scan is clean.
- [ ] Absolute path, username, internal hostname, and private-file-name scans
      are clean.
- [ ] Original market/weather workbooks, private runtime inputs, reports,
      ledgers, checkpoints, and generated output are absent; only the four
      allowlisted field-minimized research workbooks are present under
      `data/public/`.
- [ ] `data/public/MANIFEST.json` hashes, sheet names, fields, row counts, and
      coverage match the four committed workbooks.
- [ ] Public workbook metadata, hidden sheets, formulas, comments, hyperlinks,
      external links, embedded objects, and private text markers are clean.
- [ ] Large-file and binary review is clean.

## Experimental status and result disclosure

- [ ] `README.md` and `README.zh-CN.md` state clearly that the project is
      experimental and not suitable for production use.
- [ ] Tests, synthetic fixtures, and local backtests are not described as proof
      of live-market accuracy, trading performance, or production readiness.
- [ ] Any disclosed aggregate metrics are rounded, tied to a stated validation
      window and method, and reviewed for data and model redistribution rights.
- [ ] The feature audit is labeled `exploratory_backend_numeric_drift` when
      backend parity is not certified; it is not presented as a public
      benchmark or causal evidence.
- [ ] No single-day private prediction values, extrema, probabilities, run
      identifiers, or report descriptions are included in tracked documents.
- [ ] No private path, checksum, raw market value, model weight, CSV, JSON
      result, or report package is copied into tracked documentation.

## Reproducibility

- [ ] `uv sync --locked --dev` succeeds in a clean environment.
- [ ] `pytest` succeeds without real data, API keys, or production weights.
- [ ] `compileall`, import smoke tests, wheel build, and clean wheel install
      succeed.
- [ ] The production CLI fails clearly when required inputs are missing.
- [ ] The public workbook importer passes `--check-only` and writes canonical
      parquet only under the ignored runtime root.
- [ ] `git diff --check` succeeds.

## Publication decision

The current local staging result may be called **ready for private technical
review** only after the engineering checks pass. It may be called **ready to
publish publicly** only after the provenance and model/data gates above are
also checked. Neither status means that the experimental model is suitable for
production deployment.
