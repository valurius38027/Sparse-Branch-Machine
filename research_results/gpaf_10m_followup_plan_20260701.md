# GPAF 10M Follow-up Plan (2026-07-01)

## Sync status

This checkout has no configured git remote, so `git fetch --all --prune` had no
remote branch to synchronize and `git push -u origin ...` cannot work until a
remote is configured. The analysis below is therefore based on the 10M evidence
currently present in this checkout, especially
`research_results/adaptive_computation_upgrade_10m_20260630.md`.

## Evidence used

The available 10M FineWeb-Edu result shows that `upgrade-v1` is the best current
base: eval NLL 6.2305 ± 0.0001, 895 MB, 17,000 tok/s, and one accepted topology
channel. It improves over `r3-baseline` by 0.111 nats/token while reducing model
size from 2.16 GB to 895 MB.

The same result shows that adaptive beam width and iterative refinement are not
the right default base for GPAF yet: `upgrade-v1-adaptive` is worse than
`upgrade-v1` by about 0.026 nats/token and slower at 12,444 tok/s.

## Research interpretation

GPAF should be evaluated on top of `upgrade-v1`, not on top of the adaptive beam
width/refinement variant. Momentum with bias correction changes the topology
regime: fewer channels are accepted, but local sparse decisions learn more
strongly. That is useful for GPAF because a shadow or retrieval slot should be
judged against a strong local learner, not against a weak baseline that leaves
obvious local gains unclaimed.

The first 10M GPAF gate should be observational. Run `gpaf-shadow-v1` first and
compare it to `upgrade-v1` with identical seeds and corpus split. This should
prove whether role keys have reuse, whether structural-call keys appear at
meaningful rates, and whether route-source diagnostics still show token-signature
or exact-bucket dominance. It must not change predictions.

Only if shadow diagnostics show reusable keys should `gpaf-retrieval-v1` or
`gpaf-structural-call-v1` be run. Those presets intentionally use small bounded
quotas so that any NLL change can be attributed against active work, candidate
source, and frozen ablation diagnostics.

## Recommended next experiments

1. `upgrade-v1` reproduction on the latest 10M manifest and seeds.
2. `gpaf-shadow-v1` on the same manifest and seeds; required comparisons:
   - eval NLL equality or near-equality with `upgrade-v1`;
   - `gpaf_role_observations`, `gpaf_unique_role_keys`, and structural-call key
     counts;
   - route-source counters and Hamming/exact score contribution totals;
   - model bytes and throughput.
3. `gpaf-retrieval-v1` only if shadow keys show reuse; required comparisons:
   - eval NLL and per-key frozen GPAF ablation;
   - `gpaf_slots_probed` and `gpaf_candidates_returned` bounded by quota;
   - shuffled/random-slot controls before claiming a mechanism gain.
4. `gpaf-structural-call-v1` only after dependency-bearing accepted channels are
   observed in the same run family; required comparisons:
   - structural-call observations and returned candidates;
   - dependency-blocked lookup diagnostics after producer retirement;
   - per-key ablation gain versus false-positive cost.

## Non-claims

This plan does not claim GPAF improves language modeling quality or captures
semantics. It only defines the next real-corpus mechanism gates that should be
run once the latest remote 10M results are available in this checkout.
