# Global Predictive Address Field Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a next-generation sparse addressing layer whose global influence emerges from predictive-role reuse and held-out codelength evidence rather than token-ID or token-signature similarity.

**Architecture:** Keep the existing address-program system as a bootstrap and control. Add GPAF in conservative stages: first diagnostics, then shadow role-key observation, then bounded candidate retrieval, then lifecycle and frozen ablation. No dense router, no global scan, no linguistic labels and no target leakage.

**Tech Stack:** C++20 shared libraries (`sbm_machine`, `sbm_experiment`, `sbm_api`), C ABI, Python `ctypes` runtime, CTest, existing corpus runner scripts.

---

## File structure

- Modify `include/sbm/types.hpp`: runtime config fields, GPAF metrics structures and role-key enums.
- Modify `include/sbm/machine.hpp`: GPAF storage fields and helper declarations.
- Create `src/modules/global_predictive_address.cpp`: GPAF slot storage, role-key hashing, shadow observation, bounded lookup and lifecycle helpers.
- Modify `src/modules/machine_routing.cpp`: candidate-source accounting and disabled-by-default GPAF candidate injection.
- Modify `src/modules/token_objective.cpp`: post-prediction role observation and target-safe writer updates.
- Modify `src/modules/machine_topology.cpp`: slot lifecycle/frozen-ablation integration after shadow mode is stable.
- Modify `src/modules/experiment.cpp` and `src/modules/token_experiment.cpp`: JSON diagnostics and result summaries.
- Modify `src/api/c_api.cpp`: config schema and public diagnostics exposure.
- Modify `python/sbm_runtime.py` and `python/sbm_presets.py`: runtime parameter access and conservative presets.
- Modify `tests/test_sbm.cpp`, `tests/test_c_api.cpp`, `tests/test_scaling.cpp`, `tests/test_presets.py`, and `tests/test_resumable_runner.py`: correctness, schema, bounded-work, frozen immutability and resume tests.
- Update `DESIGN_NOTES.md`, `PROGRESS.md` and `RESEARCH_LOG.md` only after implementation evidence exists.

## Task 1: Baseline route-source and token-similarity diagnostics

Status: implemented in commit following this plan update; commit step intentionally remains unchecked until the final task commit is created.

**Files:**
- Modify: `include/sbm/types.hpp`
- Modify: `include/sbm/machine.hpp`
- Modify: `src/modules/machine_routing.cpp`
- Modify: `src/modules/experiment.cpp`
- Modify: `src/modules/token_experiment.cpp`
- Modify: `src/api/c_api.cpp`
- Test: `tests/test_c_api.cpp`

- [x] **Step 1: Add failing diagnostics assertions**

Add test coverage that result JSON exposes candidate counts by source and route-score component totals. Expected fields:

```json
"candidate_source_exact_bucket"
"candidate_source_control_edge"
"candidate_source_neighbor_bucket"
"route_score_hamming_sum"
"route_score_exact_sum"
"route_score_edge_prior_sum"
```

Run:

```bash
cmake --build build-fast --target sbm_tests sbm_c_api_tests -j2
ctest --test-dir build-fast --output-on-failure -R "sbm_cpp_api|sbm_c_api"
```

Expected: FAIL because the fields do not exist.

- [x] **Step 2: Add metric fields and accounting**

Add monotonically increasing counters for candidate sources and score components. Do not change routing behavior.

- [x] **Step 3: Emit metrics through summaries and C API JSON**

Expose the new counters in machine summaries and experiment result JSON.

- [x] **Step 4: Verify behavior-preserving diagnostics**

Run:

```bash
cmake --build build-fast --target sbm_api sbm_tests sbm_c_api_tests -j2
ctest --test-dir build-fast --output-on-failure -R "sbm_cpp_api|sbm_c_api"
```

Expected: PASS. Existing NLL and route behavior should remain unchanged for fixed seeds except for extra diagnostics fields.

- [ ] **Step 5: Commit**

```bash
git add include/sbm/types.hpp include/sbm/machine.hpp src/modules/machine_routing.cpp src/modules/experiment.cpp src/modules/token_experiment.cpp src/api/c_api.cpp tests/test_c_api.cpp
git commit -m "diagnostics: expose token-similarity routing dependence"
```

## Task 2: Shadow GPAF role-key observation

Status: shadow-only observation is implemented for token cross-entropy paths. It records role-key counters and has no candidate-retrieval effect; commit step intentionally remains unchecked until the final task commit is created.

**Files:**
- Modify: `include/sbm/types.hpp`
- Modify: `include/sbm/machine.hpp`
- Modify: `CMakeLists.txt`
- Modify: `src/modules/machine_topology.cpp`
- Modify: `src/modules/token_objective.cpp`
- Modify: `src/modules/token_sparse_output.cpp`
- Modify: `src/modules/experiment.cpp`
- Test: `tests/test_sbm.cpp`
- Test: `tests/test_scaling.cpp`

- [x] **Step 1: Add failing shadow-mode tests**

Add tests asserting that enabling `gpaf_shadow_observation=true` records role-key observations but does not change prediction, active node count or candidates used for routing.

Run:

```bash
cmake --build build-fast --target sbm_tests sbm_scaling_tests -j2
ctest --test-dir build-fast --output-on-failure -R "sbm_cpp_api|sbm_scaling"
```

Expected: FAIL because GPAF types and config fields do not exist.

- [x] **Step 2: Define GPAF config and metrics**

Add disabled-by-default config fields:

```text
gpaf_shadow_observation=false
gpaf_candidate_retrieval=false
gpaf_query_keys_per_step=0
gpaf_slots=0
gpaf_residents_per_slot=0
```

Add metrics:

```text
gpaf_role_observations
gpaf_unique_role_keys
gpaf_slots_allocated
gpaf_shadow_updates
gpaf_slots_probed
gpaf_candidates_returned
```

- [x] **Step 3: Implement role-key hashing in shadow mode**

Create a structural-role shadow key from active channel program kind, arity, lineage edge kinds and generation. Do not use raw target tokens, and do not inject candidates.

- [x] **Step 4: Verify no behavior change**

Run fixed-seed before/after smoke and assert predicted tokens, active nodes and candidates examined match when `gpaf_candidate_retrieval=false`.

Run:

```bash
ctest --test-dir build-fast --output-on-failure -R "sbm_cpp_api|sbm_scaling"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt include/sbm/types.hpp include/sbm/machine.hpp src/modules/machine_topology.cpp src/modules/token_objective.cpp src/modules/token_sparse_output.cpp src/modules/experiment.cpp tests/test_scaling.cpp
git commit -m "feat: add shadow global predictive address field"
```

## Task 3: Bounded GPAF candidate retrieval behind a flag

Status: implemented behind `gpaf_candidate_retrieval=false` by default. Lookup is
quota-bounded, score-conservative and covered by same-format checkpoint/resume
tests. Slot lifecycle and frozen ablation are still Task 4 work.

**Files:**
- Modify: `include/sbm/types.hpp`
- Modify: `include/sbm/machine.hpp`
- Modify: `src/modules/global_predictive_address.cpp`
- Modify: `src/modules/machine_routing.cpp`
- Modify: `tests/test_scaling.cpp`
- Modify: `tests/test_c_api.cpp`

- [x] **Step 1: Add failing bounded-work tests**

Assert that with candidate retrieval enabled:

```text
gpaf_slots_probed_per_token <= gpaf_query_keys_per_step
gpaf_candidates_returned_per_token <= gpaf_query_keys_per_step * gpaf_residents_per_slot
active_nodes <= beam_width
```

Run:

```bash
cmake --build build-fast --target sbm_scaling_tests sbm_c_api_tests -j2
ctest --test-dir build-fast --output-on-failure -R "sbm_scaling|sbm_c_api"
```

Expected: FAIL until retrieval accounting exists.

- [x] **Step 2: Implement quota-based slot lookup**

Probe at most `gpaf_query_keys_per_step` slots and return at most `gpaf_residents_per_slot` residents from each slot. Candidate merge must preserve separate source quotas.

- [x] **Step 3: Keep GPAF score conservative**

Initially GPAF introduces candidates but does not add a large direct score boost. The candidate must still pass reliability, dependency and route-score selection.

- [x] **Step 4: Verify bounded retrieval**

Run:

```bash
ctest --test-dir build-fast --output-on-failure -R "sbm_scaling|sbm_c_api"
```

Expected: PASS with bounded candidate diagnostics.

- [x] **Step 5: Commit**

```bash
git add include/sbm/types.hpp include/sbm/machine.hpp src/modules/global_predictive_address.cpp src/modules/machine_routing.cpp tests/test_scaling.cpp tests/test_c_api.cpp
git commit -m "feat: add bounded GPAF candidate retrieval"
```

## Task 4: Slot lifecycle and frozen ablation

Status: partial. Frozen evaluation now may read existing GPAF residents for
routing but does not mutate GPAF role observations, slot/probe counters or
returned-candidate counters. Observed GPAF role slots are now explicitly tracked
as Probe slots and reported/persisted by phase. Active admission,
quarantine/recoverable-retirement transitions and frozen ablation accounting
remain pending.

**Files:**
- Modify: `include/sbm/types.hpp`
- Modify: `include/sbm/machine.hpp`
- Modify: `src/modules/global_predictive_address.cpp`
- Modify: `src/modules/machine_topology.cpp`
- Modify: `src/modules/token_experiment.cpp`
- Test: `tests/test_sbm.cpp`
- Test: `tests/test_resumable_runner.py`

- [ ] **Step 1: Add failing lifecycle tests**

Partial coverage added for frozen read-only behavior and Probe slot phase
diagnostic/checkpoint preservation; phase-transition tests for Probe -> Active,
Probe -> Quarantined and RecoverableRetired -> Active are still pending.

Add tests that frozen evaluation does not update GPAF slots, slot credit or role-key statistics. Add tests for Probe -> Active, Probe -> Quarantined and RecoverableRetired -> Active transitions.

Run:

```bash
cmake --build build-fast --target sbm_tests sbm_api -j2
ctest --test-dir build-fast --output-on-failure -R "sbm_cpp_api|sbm_resumable_runner"
```

Expected: FAIL until lifecycle state exists.

- [ ] **Step 2: Implement slot phases**

Partial implementation: role slots enter the explicit Probe phase, Probe/Active
are the only routable phases, and phase counts are exported. Active admission
and retirement transitions are not implemented yet.

Use phases:

```text
Probe
Active
Quarantined
RecoverableRetired
PhysicallyErased
```

Only Probe and Active slots may route. Frozen evaluation may read Active slots but cannot mutate any GPAF state.

- [ ] **Step 3: Implement exact slot ablation accounting**

For each candidate Active/Probe slot, compute held-out codelength with and without that slot's candidates under the same prediction examples. Record codelength gain, false-positive cost, description cost, execution cost and reuse count.

- [ ] **Step 4: Verify freeze and resume invariants**

Run:

```bash
ctest --test-dir build-fast --output-on-failure -R "sbm_cpp_api|sbm_resumable_runner"
```

Expected: PASS; checkpoint/resume preserves GPAF state exactly for same-format runs.

- [ ] **Step 5: Commit**

```bash
git add include/sbm/types.hpp include/sbm/machine.hpp src/modules/global_predictive_address.cpp src/modules/machine_topology.cpp src/modules/token_experiment.cpp tests/test_sbm.cpp tests/test_resumable_runner.py
git commit -m "feat: add GPAF lifecycle and frozen ablation"
```

## Task 5: Structural-call role keys

**Files:**
- Modify: `include/sbm/types.hpp`
- Modify: `src/modules/global_predictive_address.cpp`
- Modify: `src/modules/machine_topology.cpp`
- Modify: `src/modules/machine_routing.cpp`
- Test: `tests/test_scaling.cpp`
- Test: `tests/test_sbm.cpp`

- [ ] **Step 1: Add failing structural-call tests**

Assert that structural-call keys are produced only from accepted programs and compatible dependency states, and that disabling the producer dependency disables the call slot.

Run:

```bash
cmake --build build-fast --target sbm_tests sbm_scaling_tests -j2
ctest --test-dir build-fast --output-on-failure -R "sbm_cpp_api|sbm_scaling"
```

Expected: FAIL until structural-call roles exist.

- [ ] **Step 2: Implement StructuralCallRole generation**

Generate keys from accepted program identity, output state kind, dependency edge kind, call-depth bucket, reuse bucket and execution-cost bucket. Do not include linguistic labels or raw target tokens.

- [ ] **Step 3: Enforce dependency compatibility**

A structural-call slot can route only when the required producer/dependency state is active and compatible. If the producer is quarantined or recoverably retired, the dependent call slot is effectively blocked.

- [ ] **Step 4: Verify dependency closure**

Run:

```bash
ctest --test-dir build-fast --output-on-failure -R "sbm_cpp_api|sbm_scaling"
```

Expected: PASS with dependency-blocked diagnostics.

- [ ] **Step 5: Commit**

```bash
git add include/sbm/types.hpp src/modules/global_predictive_address.cpp src/modules/machine_topology.cpp src/modules/machine_routing.cpp tests/test_scaling.cpp tests/test_sbm.cpp
git commit -m "feat: add GPAF structural call roles"
```

## Task 6: Experiment gates, presets and documentation

**Files:**
- Modify: `python/sbm_presets.py`
- Modify: `scripts/run_corpus_training.py`
- Modify: `scripts/run_resumable_corpus_training.py`
- Modify: `tests/test_presets.py`
- Modify: `DESIGN_NOTES.md`
- Modify: `PROGRESS.md`
- Append: `RESEARCH_LOG.md`

- [ ] **Step 1: Add failing preset tests**

Add presets:

```text
gpaf-shadow-v1
gpaf-retrieval-v1
gpaf-structural-call-v1
```

The first preset must not change predictions; the latter two must be explicitly experimental and disabled unless requested.

Run:

```bash
python tests/test_presets.py
```

Expected: FAIL until presets exist.

- [ ] **Step 2: Add experiment reporting fields**

Corpus runners must emit GPAF config, role-key counts, slot lifecycle counts, source-quota diagnostics, frozen slot ablation and model byte split.

- [ ] **Step 3: Update canonical docs after evidence exists**

Update `DESIGN_NOTES.md` with the adopted invariant summary, `PROGRESS.md` with current handoff status and `RESEARCH_LOG.md` with accepted or negative experiment results. Do not claim language semantics unless real-data gates pass.

- [ ] **Step 4: Run full relevant checks**

Run:

```bash
cmake --build build-fast --target sbm_api sbm_tests sbm_c_api_tests sbm_scaling_tests -j2
ctest --test-dir build-fast --output-on-failure
python tests/test_presets.py
```

Expected: PASS or documented environment limitation.

- [ ] **Step 5: Commit**

```bash
git add python/sbm_presets.py scripts/run_corpus_training.py scripts/run_resumable_corpus_training.py tests/test_presets.py DESIGN_NOTES.md PROGRESS.md RESEARCH_LOG.md
git commit -m "research: add GPAF experiment gates and documentation"
```

## Self-review checklist

- Spec coverage: diagnostics, shadow observation, retrieval, lifecycle, structural calls, controls and docs are mapped to tasks.
- Placeholder scan: this plan intentionally contains no incomplete placeholder markers.
- Type consistency: GPAF naming uses `gpaf_*` runtime fields and role names `CoPredictionRole`, `BindingReuseRole`, `StructuralCallRole` consistently.
