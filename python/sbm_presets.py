"""Named configuration presets for SPM experiments.

Presets are declarative dictionaries of runtime parameter overrides. They are
applied on top of the library defaults, so adding a new preset never requires
code changes outside this file unless the parameter itself is new.
"""

from __future__ import annotations

from typing import Any


PRESETS: dict[str, dict[str, Any]] = {
    "r3-baseline": {
        "adaptive_topology": True,
        "address_lags": 1,
        "topology_enable_delta": False,
        "max_sparse_decisions_per_node": 512,
        "classification_learning_rate": 0.8,
        "classification_mature_learning_rate": 0.2,
        "record_channel_attribution": True,
        "topology_accept_uses_structural_value": True,
    },
    "upgrade-v1": {
        # Proven 10M improvement over r3-baseline: -0.111 nats/token.
        # Adam-like momentum on sparse decisions with bias correction.
        "adaptive_topology": True,
        "address_lags": 1,
        "topology_enable_delta": False,
        "max_sparse_decisions_per_node": 512,
        "classification_learning_rate": 0.8,
        "classification_mature_learning_rate": 0.2,
        "record_channel_attribution": True,
        "topology_accept_uses_structural_value": True,
        "use_momentum": True,
        "topology_enable_content_follow_multi": True,
    },
    "upgrade-v1-adaptive": {
        # Experimental: adds adaptive beam width and iterative refinement.
        # 10M result is slightly worse than upgrade-v1 but still beats baseline.
        "adaptive_topology": True,
        "address_lags": 1,
        "topology_enable_delta": False,
        "max_sparse_decisions_per_node": 512,
        "classification_learning_rate": 0.8,
        "classification_mature_learning_rate": 0.2,
        "record_channel_attribution": True,
        "topology_accept_uses_structural_value": True,
        "use_momentum": True,
        "topology_enable_content_follow_multi": True,
        "beam_width": 8,
        "beam_width_min": 2,
        "confidence_threshold": 0.7,
        "max_refinement_rounds": 2,
        "refinement_confidence_threshold": 0.5,
    },
    "gpaf-shadow-v1": {
        # 10M follow-up gate: observe GPAF role-key reuse on the proven
        # upgrade-v1 base without changing routing or resident retrieval.
        "adaptive_topology": True,
        "address_lags": 1,
        "topology_enable_delta": False,
        "max_sparse_decisions_per_node": 512,
        "classification_learning_rate": 0.8,
        "classification_mature_learning_rate": 0.2,
        "record_channel_attribution": True,
        "topology_accept_uses_structural_value": True,
        "use_momentum": True,
        "topology_enable_content_follow_multi": True,
        "gpaf_shadow_observation": True,
        "gpaf_candidate_retrieval": False,
        "gpaf_query_keys_per_step": 0,
        "gpaf_slots": 65536,
        "gpaf_residents_per_slot": 0,
    },
    "gpaf-retrieval-v1": {
        # Experimental: bounded GPAF candidates on top of upgrade-v1. Keep
        # quotas small until 10M shadow diagnostics show reusable role keys.
        "adaptive_topology": True,
        "address_lags": 1,
        "topology_enable_delta": False,
        "max_sparse_decisions_per_node": 512,
        "classification_learning_rate": 0.8,
        "classification_mature_learning_rate": 0.2,
        "record_channel_attribution": True,
        "topology_accept_uses_structural_value": True,
        "use_momentum": True,
        "topology_enable_content_follow_multi": True,
        "gpaf_shadow_observation": True,
        "gpaf_candidate_retrieval": True,
        "gpaf_query_keys_per_step": 2,
        "gpaf_slots": 65536,
        "gpaf_residents_per_slot": 4,
        "gpaf_probe_min_observations": 64,
        "gpaf_probe_min_residents": 1,
    },
    "gpaf-structural-call-v1": {
        # Experimental: emphasizes structural-call GPAF roles after the
        # dependency-bearing channel path is already accepted and auditable.
        "adaptive_topology": True,
        "address_lags": 1,
        "topology_enable_delta": False,
        "max_sparse_decisions_per_node": 512,
        "classification_learning_rate": 0.8,
        "classification_mature_learning_rate": 0.2,
        "record_channel_attribution": True,
        "topology_accept_uses_structural_value": True,
        "use_momentum": True,
        "topology_enable_content_follow_multi": True,
        "gpaf_shadow_observation": True,
        "gpaf_candidate_retrieval": True,
        "gpaf_query_keys_per_step": 4,
        "gpaf_slots": 65536,
        "gpaf_residents_per_slot": 4,
        "gpaf_probe_min_observations": 32,
        "gpaf_probe_min_residents": 2,
    },
}


def apply_preset(preset_name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a config dictionary for the named preset with optional overrides.

    Later overrides take precedence over preset values.
    """
    if preset_name not in PRESETS:
        raise KeyError(f"unknown preset {preset_name!r}; available: {sorted(PRESETS)}")
    result = dict(PRESETS[preset_name])
    if overrides:
        result.update(overrides)
    return result
