# Global Predictive Address Field Design

## Status

Draft architecture specification for the next-generation sparse addressing mechanism.
This document is not an implementation claim. It defines the intended direction,
constraints, admission gates and documentation responsibilities for a large
addressing-system change.

## Motivation

The current address system is useful as a bounded execution substrate, but its
strongest locality signal is still surface token identity or token-derived
signature similarity. `ContentMatch` and `ContentFollow` are bounded and generic,
but they mainly discover recurrence over equal tokens and their successors. That
is too narrow for language and semantics.

Language-level structure should not be hand-coded as subject, object, entity,
coreference, topic, syntax or semantic labels. Those categories are high-level
abstractions that should emerge only if the sparse machine discovers reusable
predictive structure under held-out validation. The next mechanism must therefore
avoid replacing token heuristics with linguistic heuristics. It must instead make
room for global reusable sparse computation to emerge from prediction, binding
reuse, structural calls and codelength evidence.

## Core claim

A context should not retrieve a node because its token ID pattern is similar to a
stored prototype. A context should retrieve a node or program because historical
frozen evidence says that this context and the stored structure share a reusable
predictive role.

The proposed mechanism is the **Global Predictive Address Field** (GPAF): a
bounded, auditable, sparse global address layer whose keys are learned from
prediction roles and structural reuse rather than from token similarity.

## Non-goals

- Do not introduce dense embeddings or an ANN nearest-neighbor router as the
  central mechanism.
- Do not scan all nodes, programs, slots or parameters per token.
- Do not add linguistic supervision or hard-coded linguistic role labels.
- Do not let target-dependent information affect the prediction being scored for
  the same sample.
- Do not claim language understanding, semantics or variable binding from
  synthetic or compatibility-only results.

## Current mechanism being replaced as the primary signal

The existing system should remain as a bootstrap and compatibility control. Its
important limitations are:

1. Address signatures pack current tokens, lagged tokens, content match bits and
   matched successors into a compact signature.
2. Candidate retrieval prioritizes exact per-channel buckets, previous-route
   control edges and neighboring bucket probes.
3. Route scoring gives substantial weight to Hamming similarity between node
   prototypes and current signatures.
4. Content matching is bounded and auditable, but it is still primarily exact
   token recurrence.

These are acceptable engineering supports. They are not sufficient as the main
research path for global sparse addressing.

## Design principle: emergent roles, not prior semantic labels

GPAF keys must represent **operational roles** that can be measured without
naming a linguistic abstraction:

- which accepted program or local structure produced predictive gain;
- which binding state kind was produced or consumed;
- which dependency/call pattern was satisfied;
- which coarse output decision path or residual-correction class was improved;
- how often the role reused across documents, seeds and shards;
- whether the role survived frozen ablation after description and execution
  costs.

These roles may later align with semantic or syntactic phenomena, but the system
must not require that alignment in advance.

## Architecture

### Global address slot

A GPAF slot is a bounded global sparse cell:

```text
GlobalPredictiveSlot {
    key: PredictiveRoleKey
    phase: Probe | Active | Quarantined | RecoverableRetired | PhysicallyErased
    residents: bounded list of NodeId
    programs: bounded list of AddressProgram/channel references
    writer_rule: PredictiveRoleWriter
    reader_rule: PredictiveRoleReader
    stats: PredictiveSlotStats
}
```

The slot owns no unbounded vector and performs no dense comparison. It stores a
small resident list, small program reference list and scalar audit statistics.

### Predictive role key

A predictive role key is a typed sparse key. Version 1 should support these key
families:

1. **CoPredictionRole** — contexts that receive similar sparse output-tree or
   residual-logit corrections after prediction is fixed.
2. **BindingReuseRole** — contexts that produce or consume the same kind of
   binding/call state, independent of raw token identity.
3. **StructuralCallRole** — contexts that can call an accepted program whose
   held-out contribution has transferred beyond the local context that created
   it.

Each key family must be generic and non-linguistic. Keys may include quantized
loss/codelength buckets, binding kind, address-program op kind, dependency edge
kind, call depth, reuse count bucket, execution-cost bucket and output-tree
region. Keys must not include subject/object/entity/coreference labels.

### Read path

Prediction-time read path:

```text
history/window
    -> existing seed address programs
    -> first-pass bounded local route
    -> local route emits at most Q predictive role query keys
    -> GPAF probes at most Q slots
    -> each slot returns at most R residents/program references
    -> candidate merge under fixed source quotas
    -> final route selection
    -> prediction fixed
```

The read path uses only state available before the target. Query count, returned
residents, global candidates and final active nodes are all hard-bounded.

### Write path

Training-time write path:

```text
prediction fixed
    -> target observed
    -> local nodes and output residuals update
    -> counterfactual/codelength contribution is computed
    -> writer rules propose or update predictive role slots
    -> slot statistics and lifecycle evidence update
```

A target can affect future queries only after the current prediction has already
been scored. Frozen validation/test performs no slot writes, no slot lifecycle
changes and no topology-credit updates.

### Candidate merge

Candidates are merged by source quota rather than by one global pool:

```text
local_exact_quota
control_edge_quota
gpaf_role_quota
structural_call_quota
exploration_quota
```

The first implementation should keep GPAF quotas small. GPAF may introduce
candidates, but it should not directly dominate final scoring until slot-level
frozen evidence exists.

### Scoring

The route score should move away from token-signature Hamming similarity as a
primary term. A GPAF-aware score should be based on:

```text
expected held-out codelength gain
+ reliability
+ dependency satisfaction
+ binding reuse value
+ calibrated edge or slot prior
- false-positive cost
- execution cost
- stale-reference penalty
```

Token-signature similarity may remain as a compatibility fallback and ablation
control, not the central global retrieval signal.

## Lifecycle and admission

GPAF slots must use the same scientific discipline as address programs:

1. **Shadow observation** — collect role-key and counterfactual evidence without
   affecting predictions.
2. **Probe** — allow bounded candidate injection during a training adaptation
   period.
3. **Frozen validation** — disable learning and measure exact slot ablation on a
   held-out tail.
4. **Admission** — accept only after codelength gain exceeds description and
   execution costs.
5. **Quarantine or recoverable retirement** — mask harmful or stale slots while
   retaining audit state.
6. **Physical erase** — remove slot state only under an explicit lifecycle event.

Admission must be based on held-out codelength, structural description cost,
execution cost, reuse evidence and false-positive cost. Raw training loss is not
a valid acceptance criterion.

## Diagnostics

Every GPAF experiment must report:

- persistent node count;
- edge count;
- GPAF slot count by phase;
- active nodes per token;
- candidates inspected per token by source;
- GPAF query keys per token;
- GPAF slots probed per token;
- GPAF residents returned per token;
- output work per token;
- slot-level held-out codelength gain;
- false-positive cost;
- unique documents and shards touched by each active slot;
- train and frozen-evaluation throughput;
- model bytes split by node store, output store, topology and GPAF.

## Controls and gates

A GPAF result is not accepted unless it beats all relevant controls under the
same active-work budget:

- unigram/current-token controls;
- short-context table or fixed-lag controls;
- current local-only SPM route;
- shuffled GPAF role keys;
- random GPAF slots with the same quota;
- GPAF shadow-only run;
- fixed `[1,2,4]` and current accepted-address controls when applicable.

Accepted evidence requires multi-seed stability, document-distribution evidence,
strict frozen-evaluation immutability and shard-transfer checks. A gain that
comes from proportional growth in per-token candidate work is rejected.

## Implementation sequence

1. Add diagnostics that quantify current token-signature and Hamming-similarity
   dependence without changing behavior.
2. Add shadow GPAF role-key generation and slot statistics; no prediction change.
3. Add bounded GPAF candidate retrieval behind a disabled-by-default runtime
   flag.
4. Add slot lifecycle, frozen ablation and JSON diagnostics.
5. Add structural-call role keys only after CoPredictionRole and
   BindingReuseRole have passed controls.
6. Run compatibility-corpus smoke tests, then R3-scale admissible gates only if
   data provenance is sufficient for the claim being made.

## Documentation ownership

- This specification owns the next-generation GPAF architecture intent.
- `docs/superpowers/plans/2026-06-30-global-predictive-address-field.md` owns
  the implementation task sequence.
- `DESIGN_NOTES.md` owns the architectural invariant summary after the design is
  adopted.
- `PROGRESS.md` owns the current handoff state for agents resuming work.
- `ROADMAP_REAL_DATA.md` remains the authority for admissible real-data claims.
- `RESEARCH_LOG.md` records experimental outcomes, including negative results.

## Open questions

1. Which role-key family should be admitted first: CoPredictionRole or
   BindingReuseRole? The default recommendation is CoPredictionRole because it
   can be shadow-measured against existing output diagnostics before it affects
   routing.
2. How much slot-level evidence is enough for activation? The first threshold
   should be conservative and derived from codelength net of description and
   execution cost.
3. Whether structural-call slots can be made stable without false-positive
   propagation is unknown and must not be assumed.
