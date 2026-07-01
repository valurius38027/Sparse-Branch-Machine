#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from sbm_presets import apply_preset  # noqa: E402


def main() -> None:
    r3 = apply_preset("r3-baseline")
    assert r3["adaptive_topology"] is True
    assert r3["address_lags"] == 1

    upgrade = apply_preset("upgrade-v1")
    assert upgrade["use_momentum"] is True
    assert "beam_width_min" not in upgrade

    adaptive = apply_preset("upgrade-v1-adaptive")
    assert adaptive["use_momentum"] is True
    assert adaptive["beam_width_min"] == 2
    assert adaptive["beam_width"] == 8
    assert adaptive["max_refinement_rounds"] == 2


    gpaf_shadow = apply_preset("gpaf-shadow-v1")
    assert gpaf_shadow["use_momentum"] is True
    assert gpaf_shadow["gpaf_shadow_observation"] is True
    assert gpaf_shadow["gpaf_candidate_retrieval"] is False
    assert gpaf_shadow["gpaf_slots"] == 65536
    assert gpaf_shadow["gpaf_residents_per_slot"] == 0

    gpaf_retrieval = apply_preset("gpaf-retrieval-v1")
    assert gpaf_retrieval["use_momentum"] is True
    assert gpaf_retrieval["gpaf_shadow_observation"] is True
    assert gpaf_retrieval["gpaf_candidate_retrieval"] is True
    assert gpaf_retrieval["gpaf_query_keys_per_step"] == 2
    assert gpaf_retrieval["gpaf_residents_per_slot"] == 4
    assert gpaf_retrieval["gpaf_probe_min_observations"] == 64

    gpaf_structural = apply_preset("gpaf-structural-call-v1")
    assert gpaf_structural["use_momentum"] is True
    assert gpaf_structural["gpaf_shadow_observation"] is True
    assert gpaf_structural["gpaf_candidate_retrieval"] is True
    assert gpaf_structural["gpaf_query_keys_per_step"] == 4
    assert gpaf_structural["gpaf_residents_per_slot"] == 4
    assert gpaf_structural["gpaf_probe_min_observations"] == 32
    assert gpaf_structural["gpaf_probe_min_residents"] == 2

    overridden = apply_preset("upgrade-v1-adaptive", {"beam_width": 12})
    assert overridden["beam_width"] == 12
    assert overridden["use_momentum"] is True

    try:
        apply_preset("nonexistent")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass

    print("preset tests passed")


if __name__ == "__main__":
    main()
