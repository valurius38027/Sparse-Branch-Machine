# GPAF structural-call PR conflict-resolution branch

Status: integration branch note for the GPAF structural-call work.

## Purpose

The previous GPAF structural-call PR was difficult to merge because it bundled
code, tests, checkpoint format changes, C API/JSON diagnostics, and several
status-document updates. This branch keeps the repository changes intact and
avoids resolving the conflict by rolling back GPAF implementation state.

## Resolution policy

- Preserve the GPAF implementation and tests already present on this branch.
- Do not accept conflict resolutions that revert GPAF status documentation to the
  older adaptive-computation-only snapshot.
- Keep `PROGRESS.md` as a current-state summary, not the authority for design
  conflicts.
- Keep the canonical GPAF design and implementation plan in:
  - `docs/superpowers/specs/2026-06-30-global-predictive-address-field-design.md`
  - `docs/superpowers/plans/2026-06-30-global-predictive-address-field.md`
- If the target branch has newer non-GPAF documentation edits, merge those edits
  around the GPAF status rather than deleting the GPAF sections.

## Verified local state

At the time this note was added, the working tree had no textual conflict
markers and the full CTest suite passed on the new branch after restoring the
GPAF documentation state instead of committing the staged documentation rollback.

## Remaining integration risk

This branch does not reduce the conceptual size of the original GPAF patch. If
the hosting platform still reports conflicts, resolve them by replaying this
branch onto the target branch file-by-file, prioritizing the actual source and
test files over progress-summary churn.
