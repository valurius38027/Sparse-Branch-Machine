# SPM Project Progress

> This file tracks current project state for any agent resuming work.
> It is a summary, not a log — for full chronological detail see RESEARCH_LOG.md.

## Active objective

持续进行正式的工程推进，保持和计划文件同步，饱和式推进，保持连贯性。
基础设施充分时避免保守增量，允许临时粗糙边缘，但最终状态必须经过验证。

当前新增架构方向：下一代 **Global Predictive Address Field (GPAF)** 已开始实施。当前完成的是工程安全的前置层：路由来源/token-signature 依赖诊断、不改变预测的 shadow-only GPAF role-key 观测、默认关闭的有界 GPAF candidate retrieval、frozen evaluation 的 GPAF read-only 保护、Probe/Active/Quarantined/RecoverableRetired phase 统计/持久化、基于重复观测和 resident 多样性的 Probe -> Active promotion、显式 quarantine / recoverable-retire / restore 生命周期转换，以及冻结评估中的 GPAF aggregate/per-role-key codelength ablation 诊断。structural-call role-key routing 的第一版已实现；description/execution cost attribution、自动实验准入和真实语料验证尚未实现。

## Branch state

- **Branch:** `work`
- **Status:** contains post-upgrade commits plus GPAF diagnostics/shadow implementation and bounded retrieval work
- **Current documentation additions:**
  - Spec: `docs/superpowers/specs/2026-06-30-global-predictive-address-field-design.md`
  - Plan: `docs/superpowers/plans/2026-06-30-global-predictive-address-field.md`
  - Architecture note: `DESIGN_NOTES.md` next-generation GPAF section
  - Implemented: route-source score diagnostics, shadow-only GPAF role-key counters, bounded GPAF candidate retrieval, frozen read-only GPAF retrieval, GPAF slot phase diagnostics, Probe -> Active promotion, explicit quarantine/recoverable-retire/restore transitions, frozen GPAF aggregate/per-role-key codelength ablation diagnostics, structural-call role-key routing/diagnostics and same-format checkpoint persistence

## Completed: Adaptive Computation Upgrade (plan `2026-06-30-adaptive-computation-upgrade.md`)

### Implemented

| Item | Status | Commit |
|---|---|---|
| U2.1 Per-entry Adam-like momentum | Done | `8094de0` |
| U1.1 Adaptive beam width | Done | `81abd27` |
| U1.2 Iterative refinement | Done | `59c1955` |
| Config presets (`r3-baseline`, `upgrade-v1`, `upgrade-v1-adaptive`) | Done | `3e9b661` |
| Adam bias correction fix | Done | `6a89d21` |
| 10M validation and result report | Done | `bfc0af3` |

### 10M FineWeb-Edu Results

| Configuration | eval NLL | Topology accepted | Bytes | Tok/s |
|---|---:|---:|---:|---:|
| r3-baseline | 6.3412 | 5 | 2.16 GB | 15,756 |
| **upgrade-v1 (momentum)** | **6.2305 ± 0.0001** | 1 | 895 MB | 17,000 |
| upgrade-v1-adaptive | 6.2559 ± 0.0002 | 1 | 896 MB | 12,444 |

Key finding: **momentum with bias correction improves NLL by 0.111 nats/token**, reduces model size by 59%, and increases throughput by 8%. Adaptive beam width + refinement are neutral/slightly negative in the current configuration.

Full report: `research_results/adaptive_computation_upgrade_10m_20260630.md`

## Next work queue

1. Finish or explicitly supersede the active R3 100M heterogeneous stream gate: `docs/superpowers/plans/2026-06-28-r3-100m-heterogeneous-stream.md`.
2. Continue GPAF from `docs/superpowers/plans/2026-06-30-global-predictive-address-field.md`: next pending work is description/execution cost attribution, automatic experiment gates and presets; structural-call role keys are implemented but still experimental and disabled by default through GPAF retrieval config.
3. Do not claim language semantics from GPAF unless real-data provenance, frozen validation, multi-seed stability, shard transfer and strong controls pass.

## Open theoretical gates

GPAF is intended to create room for global sparse retrieval to emerge from predictive role reuse. It does not by itself solve content-conditioned variable binding, relation-following, task-comparable topology value, long-horizon credit or stable cross-domain language structure.
