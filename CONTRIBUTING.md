# Contributing

Thank you for helping improve QiluPulse-96.

Before opening a change:

1. Read `AGENTS.md`, `docs/ARCHITECTURE.md`, and `docs/PROVENANCE.md`.
2. Keep changes within the public Shandong production scope.
3. Do not add real data, credentials, local paths, model weights, or generated
   reports.
4. Add or update tests for behavior changes.
5. Run the local checks:

   ```powershell
   uv run pytest -q
   uv run python -m compileall -q src scripts
   git diff --check
   ```

Use focused commits and explain any data-contract or license impact in the
pull request. Contributions are accepted under the repository license unless
a separate written agreement says otherwise.
