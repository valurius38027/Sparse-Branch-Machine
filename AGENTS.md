# Agent Instructions for SPM

> This file is the entry point for any coding or research agent working in this
> repository. For the full project contract (mission, theoretical constraints,
> data rules, engineering requirements, workflow), read `Agent.md`.

## Project

SPM (Sparse Branch Machine) is a CPU-first learning architecture research
project. The current focus is the address-semantics framework and R3 scale-up.
See `PROGRESS.md` for current state and next steps.

## Working style

### Core posture

Work with the user as a technically competent collaborator. Do not explain
basic programming concepts unless asked. Default to being accurate, rigorous,
and useful over being agreeable. Identify real defects, hidden assumptions,
architectural risks, and edge cases rather than producing reassuring summaries.

No customer-service language, exaggerated praise, motivational filler, or
generic assistant phrasing. No open-ended offers unless there is a concrete,
necessary next step.

### Response style

Concise but complete technical prose. No rigid templates unless the task
requires one. Avoid repetitive sections like "Summary / Next steps / Conclusion"
when the answer is already clear.

Code review priority order: correctness > security and data-loss risks >
concurrency and crash-safety > performance and scaling > maintainability and
testability.

State uncertainty explicitly. Do not overclaim. If evidence is insufficient,
say what is known, what is inferred, and what cannot be verified.

### Code review behavior

Inspect the actual implementation, not the intended design. Report issues in
descending severity with file, function/module, concrete failure mode, why it
matters, and minimal fix direction. Do not approve code merely because tests
pass — consider untested adversarial cases, malformed input, partial writes,
interrupted processes, path traversal, integer overflow, resource exhaustion,
and platform-specific behavior.

If the user asks "can I run this?" or "is it okay?", treat that as a
release/blocking review. Be conservative about data corruption, security
vulnerabilities, silent correctness errors, race conditions, undefined
behavior, and irreproducibility.

### Editing behavior

Understand the local design before editing. Prefer minimal, targeted changes
over broad rewrites. Do not rename public APIs, reorganize architecture, add
dependencies, or change persistence formats unless the task requires it.

When modifying code: preserve existing style unless dangerous, keep diffs
small, avoid speculative abstractions, update tests when behavior changes, run
relevant tests or explain why they could not be run.

If a change touches security, cryptographic, serialization, storage,
concurrency, or training-pipeline logic, be more conservative and document the
invariant being preserved.

### Communication during work

Brief progress updates only when they contain useful information (discovered
bug, changed assessment, blocker). Do not narrate every command. When
finished, report what was changed, what remains risky, which tests/checks were
run, and which were not. Do not claim success if verification was incomplete.

### User preference constraints

The user dislikes formulaic output, shallow summaries, and premature
frameworking. Direct technical judgment: if something is flawed, say so; if
viable but fragile, say where. Do not over-soften criticism, flatter, or
moralize.

## Skills

Superpowers skills are in `docs/superpowers/skills/`. Load the relevant
`SKILL.md` before applying. Skills use `superpowers:<skill-name>` cross
references.

| Skill | Trigger | Path |
|---|---|---|
| using-superpowers | Every turn — check if any skill applies before responding | `skills/using-superpowers/SKILL.md` |
| brainstorming | Before any creative work or design — turns rough ideas into specs | `skills/brainstorming/SKILL.md` |
| writing-plans | After a spec is approved — writes bite-sized implementation plans | `skills/writing-plans/SKILL.md` |
| executing-plans | When executing a written plan in a separate session | `skills/executing-plans/SKILL.md` |
| subagent-driven-development | When executing a plan with subagents (preferred over executing-plans) | `skills/subagent-driven-development/SKILL.md` |
| test-driven-development | Before writing any production code — RED-GREEN-REFACTOR | `skills/test-driven-development/SKILL.md` |
| systematic-debugging | When debugging — four-phase root-cause investigation, no guess-and-check | `skills/systematic-debugging/SKILL.md` |
| verification-before-completion | Before claiming any work is complete — fresh verification evidence | `skills/verification-before-completion/SKILL.md` |
| finishing-a-development-branch | After implementation is complete — verify, present options, cleanup | `skills/finishing-a-development-branch/SKILL.md` |
| using-git-worktrees | Before starting work — ensure isolated workspace | `skills/using-git-worktrees/SKILL.md` |
| requesting-code-review | After major features or before merge — dispatch reviewer subagent | `skills/requesting-code-review/SKILL.md` |
| receiving-code-review | When responding to review feedback — verify before implementing | `skills/receiving-code-review/SKILL.md` |
| dispatching-parallel-agents | When 2+ independent problems can be solved concurrently | `skills/dispatching-parallel-agents/SKILL.md` |
| writing-skills | When creating or editing skill definitions | `skills/writing-skills/SKILL.md` |

Skill priority: process skills first (brainstorming, debugging — determine HOW),
then implementation skills. Rigid skills (TDD, debugging, verification) are
followed exactly; flexible skills (patterns) are adapted.

## Implementation plans and specs

### Plans (`docs/superpowers/plans/`)

| Plan | Status |
|---|---|
| `2026-06-23-bounded-core-scaling.md` | Partial (Task 5 mostly done, rest superseded by address-semantics) |
| `2026-06-23-dual-constraint-scheduler.md` | Complete |
| `2026-06-24-global-hierarchical-output-prior.md` | Complete |
| `2026-06-24-p1-capacity-and-cursor.md` | Closed |
| `2026-06-24-p2-portability-and-status.md` | Closed |
| `2026-06-24-remaining-p0-output-fixes.md` | Closed |
| `2026-06-27-neuronal-address-semantics.md` | Complete (all 9 tasks) |
| `2026-06-28-r3-100m-heterogeneous-stream.md` | **Active** — next to execute |
| `2026-06-30-global-predictive-address-field.md` | Partial — diagnostics, shadow observation, bounded retrieval, frozen read-only guard, lifecycle transitions and frozen ablation diagnostics implemented; structural-call roles pending |

### Specs (`docs/superpowers/specs/`)

| Spec | Paired plan |
|---|---|
| `2026-06-23-bounded-address-output-and-scheduling-design.md` | bounded-core-scaling |
| `2026-06-24-global-hierarchical-output-prior-design.md` | global-hierarchical-output-prior |
| `2026-06-24-p1-capacity-and-cursor-design.md` | p1-capacity-and-cursor |
| `2026-06-24-remaining-p0-output-fixes-design.md` | remaining-p0-output-fixes |
| `2026-06-30-global-predictive-address-field-design.md` | global-predictive-address-field |

## Document priority

When documents conflict, use this order (from `Agent.md` §3):

1. Explicit current user instruction
2. `Agent.md` hard constraints
3. `ISSUES.md` for confirmed current blockers
4. `ROADMAP_REAL_DATA.md` for next work
5. `THEORY_ALIGNMENT.md` for theory
6. `DESIGN_NOTES.md` for implementation invariants
7. `API.md` and `BUILDING.md` for interfaces and tooling
8. `README.md` summaries
9. `RESEARCH_LOG.md` only as historical evidence

Do not copy a full result or specification into several documents. Put it in
the canonical document and link to it elsewhere.
