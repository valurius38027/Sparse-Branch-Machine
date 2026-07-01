#include "sbm/machine.hpp"

#include "sbm/detail/random.hpp"
#include "sbm/detail/vector_ops.hpp"
#include "sbm/math.hpp"

#include <algorithm>
#include <cmath>

namespace sbm {

void SparseBranchMachine::erase_slot(std::size_t slot) {
    const NodeId victim = ids_[slot];
    const std::size_t last = ids_.size() - 1U;
    if (slot != last) {
        ids_[slot] = ids_[last];
        prototypes_[slot] = prototypes_[last];
        visits_[slot] = visits_[last];
        address_visits_[slot] = address_visits_[last];
        utility_ema_[slot] = utility_ema_[last];
        loss_ema_[slot] = loss_ema_[last];
        address_loss_ema_[slot] = address_loss_ema_[last];
        phases_[slot] = phases_[last];
        channels_[slot] = channels_[last];
        hot_indexed_[slot] = hot_indexed_[last];
        parents_[slot] = parents_[last];
        if (!uses_sparse_token_output()) {
            std::copy_n(output_vectors_.data() + last * config_.vector_dim,
                        config_.vector_dim,
                        output_vectors_.data() + slot * config_.vector_dim);
        }
        sparse_outputs_[slot] = std::move(sparse_outputs_[last]);
        sparse_admission_[slot] = std::move(sparse_admission_[last]);
        sparse_output_evicted_masks_[slot] = sparse_output_evicted_masks_[last];
        edges_[slot] = std::move(edges_[last]);
        id_to_slot_[ids_[slot]] = static_cast<std::uint32_t>(slot);
    }
    ids_.pop_back();
    prototypes_.pop_back();
    visits_.pop_back();
    address_visits_.pop_back();
    utility_ema_.pop_back();
    loss_ema_.pop_back();
    address_loss_ema_.pop_back();
    phases_.pop_back();
    channels_.pop_back();
    hot_indexed_.pop_back();
    parents_.pop_back();
    if (!uses_sparse_token_output()) {
        output_vectors_.resize(ids_.size() * config_.vector_dim);
    }
    sparse_outputs_.pop_back();
    sparse_admission_.pop_back();
    sparse_output_evicted_masks_.pop_back();
    edges_.pop_back();
    id_to_slot_[victim] = UINT32_MAX;
}

std::size_t SparseBranchMachine::prune(std::uint32_t min_visits,
                                       float utility_threshold) {
    std::size_t pruned = 0;
    for (std::size_t slot = ids_.size(); slot-- > 0;) {
        if (visits_[slot] < min_visits && utility_ema_[slot] < utility_threshold) {
            erase_slot(slot);
            ++pruned;
        }
    }
    total_pruned_ += pruned;
    if (pruned != 0) rebuild_indexes();
    return pruned;
}

double SparseBranchMachine::output_distance(std::size_t a,
                                             std::size_t b) const noexcept {
    if (!uses_sparse_token_output()) {
        return std::sqrt(detail::squared_distance(
            output_vectors_.data() + a * config_.vector_dim,
            output_vectors_.data() + b * config_.vector_dim,
            config_.vector_dim) / static_cast<float>(config_.vector_dim));
    }
    const auto& left = sparse_outputs_[a];
    const auto& right = sparse_outputs_[b];
    std::size_t i = 0U;
    std::size_t j = 0U;
    double squared = 0.0;
    std::size_t count = 0U;
    while (i < left.size() || j < right.size()) {
        if (j >= right.size() || (i < left.size() && left[i].decision < right[j].decision)) {
            squared += static_cast<double>(left[i].logit) * left[i].logit;
            ++i;
        } else if (i >= left.size() || right[j].decision < left[i].decision) {
            squared += static_cast<double>(right[j].logit) * right[j].logit;
            ++j;
        } else {
            const double delta = static_cast<double>(left[i].logit) - right[j].logit;
            squared += delta * delta;
            ++i;
            ++j;
        }
        ++count;
    }
    return count == 0U ? 0.0 : std::sqrt(squared / static_cast<double>(count));
}

void SparseBranchMachine::absorb_node(std::size_t survivor_slot,
                                      std::size_t victim_slot) {
    const float survivor_weight = static_cast<float>(std::max(1U, visits_[survivor_slot]));
    const float victim_weight = static_cast<float>(std::max(1U, visits_[victim_slot]));
    const float inverse = 1.0F / (survivor_weight + victim_weight);
    if (!uses_sparse_token_output()) {
        float* survivor = output_vectors_.data() + survivor_slot * config_.vector_dim;
        const float* victim = output_vectors_.data() + victim_slot * config_.vector_dim;
        for (std::uint32_t component = 0; component < config_.vector_dim; ++component) {
            survivor[component] = (survivor_weight * survivor[component] +
                                   victim_weight * victim[component]) * inverse;
        }
    } else {
        for (const auto& entry : sparse_outputs_[victim_slot]) {
            auto& destination = *mutable_sparse_entry(
                survivor_slot, entry.decision, true);
            const auto destination_visits = destination.visits;
            const std::uint64_t combined_visits =
                static_cast<std::uint64_t>(destination_visits) + entry.visits;
            const float destination_weight = static_cast<float>(
                std::max(1U, destination_visits));
            const float entry_weight = static_cast<float>(std::max(1U, entry.visits));
            const float evidence_inverse = 1.0F / (destination_weight + entry_weight);
            destination.logit = (destination_weight * destination.logit +
                                 entry_weight * entry.logit) * evidence_inverse;
            destination.gain_ema =
                (destination_weight * destination.gain_ema +
                 entry_weight * entry.gain_ema) * evidence_inverse;
            destination.visits = static_cast<std::uint32_t>(std::min<std::uint64_t>(
                combined_visits, std::numeric_limits<std::uint32_t>::max()));
            destination.last_update_step = std::max(
                destination.last_update_step, entry.last_update_step);
        }
        sparse_output_evicted_masks_[survivor_slot] |=
            sparse_output_evicted_masks_[victim_slot];
    }
    visits_[survivor_slot] += visits_[victim_slot];
    address_visits_[survivor_slot] += address_visits_[victim_slot];
    utility_ema_[survivor_slot] = (survivor_weight * utility_ema_[survivor_slot] +
        victim_weight * utility_ema_[victim_slot]) * inverse;
    loss_ema_[survivor_slot] = (survivor_weight * loss_ema_[survivor_slot] +
        victim_weight * loss_ema_[victim_slot]) * inverse;
    address_loss_ema_[survivor_slot] =
        (survivor_weight * address_loss_ema_[survivor_slot] +
         victim_weight * address_loss_ema_[victim_slot]) * inverse;
    for (const auto& edge : edges_[victim_slot]) {
        reinforce_edge(ids_[survivor_slot], edge.dst, edge.weight);
    }
    const NodeId victim_id = ids_[victim_slot];
    const NodeId survivor_id = ids_[survivor_slot];
    for (auto& list : edges_) {
        for (auto& edge : list) {
            if (edge.dst == victim_id) edge.dst = survivor_id;
        }
    }
    for (auto& id : previous_route_) if (id == victim_id) id = survivor_id;
    for (auto& frame : trace_) {
        for (auto& id : frame.route) if (id == victim_id) id = survivor_id;
    }
    erase_slot(victim_slot);
}

std::size_t SparseBranchMachine::merge_redundant(std::size_t max_merges) {
    std::size_t merged = 0;
    for (std::size_t a = 0; a < ids_.size() && merged < max_merges; ++a) {
        for (std::size_t b = a + 1U; b < ids_.size() && merged < max_merges; ++b) {
            if (channels_[a] != channels_[b]) continue;
            if (hamming_similarity(prototypes_[a], prototypes_[b]) <
                config_.merge_similarity) continue;
            if (output_distance(a, b) > config_.merge_vector_distance) continue;
            if (visits_[b] > visits_[a]) absorb_node(b, a);
            else absorb_node(a, b);
            ++merged;
            break;
        }
    }
    total_merged_ += merged;
    if (merged != 0) rebuild_indexes();
    return merged;
}

void SparseBranchMachine::rebuild_indexes() {
    auto previous = std::move(bucket_directory_);
    bucket_directory_.clear();
    for (std::size_t slot = 0; slot < ids_.size(); ++slot) {
        const auto index = bucket_index(channels_[slot], prototypes_[slot]);
        auto& state = ensure_bucket(index);
        if (state.residents.empty()) {
            const auto found = previous.find(index);
            if (found != previous.end()) {
                state.last_split_step = found->second.last_split_step;
            }
        }
        state.residents.push_back(ids_[slot]);
        if (state.cold.size() < config_.bucket_scan_limit) {
            state.cold.push_back(ids_[slot]);
        }
    }
    for (std::size_t slot = 0; slot < ids_.size(); ++slot) {
        const bool was_hot = hot_indexed_[slot] != 0U;
        hot_indexed_[slot] = 0U;
        if (was_hot) mark_hot(ids_[slot]);
    }
    for (auto& list : edges_) {
        list.erase(std::remove_if(list.begin(), list.end(), [&](const Edge& edge) {
            return !contains(edge.dst);
        }), list.end());
    }
}

void SparseBranchMachine::prefill_distractors(std::size_t count) {
    std::vector<float> value(uses_sparse_token_output() ? 0U : config_.vector_dim);
    for (std::size_t index = 0; index < count; ++index) {
        for (auto& component : value) {
            const auto sample = static_cast<int>(detail::next_random(rng_state_) % 2001U) - 1000;
            component = static_cast<float>(sample) / 1000.0F;
        }
        std::vector<std::uint8_t> enabled_channels;
        for (std::uint8_t channel = 0; channel < topology_.size(); ++channel) {
            if (channel_enabled(channel)) enabled_channels.push_back(channel);
        }
        if (enabled_channels.empty()) break;
        const auto channel = enabled_channels[static_cast<std::size_t>(
            detail::next_random(rng_state_) % enabled_channels.size())];
        (void)new_node(detail::next_random(rng_state_), value, kInvalidNode, channel);
    }
}

Diagnostics SparseBranchMachine::diagnostics() const noexcept {
    std::uint64_t edge_count = 0;
    std::uint64_t edge_capacity = 0;
    std::uint64_t bucket_capacity = 0;
    std::uint64_t occupied_buckets = 0;
    std::uint64_t full_buckets = 0;
    std::uint64_t max_bucket_residents = 0;
    std::uint64_t cold = 0;
    std::uint64_t warm = 0;
    std::uint64_t mature = 0;
    std::uint64_t dormant = 0;
    std::uint64_t anchor = 0;
    std::uint64_t residual = 0;
    for (const auto& list : edges_) {
        edge_count += list.size();
        edge_capacity += list.capacity();
    }
    for (const auto& [key, state] : bucket_directory_) {
        (void)key;
        if (!state.residents.empty()) ++occupied_buckets;
        if (state.residents.size() >= config_.max_specializations_per_bucket) {
            ++full_buckets;
        }
        max_bucket_residents = std::max<std::uint64_t>(
            max_bucket_residents, state.residents.size());
        bucket_capacity += state.residents.capacity();
        bucket_capacity += state.hot.capacity();
        bucket_capacity += state.cold.capacity();
    }
    for (std::size_t slot = 0; slot < ids_.size(); ++slot) {
        switch (phase_of_slot(slot)) {
            case NodePhase::Cold: ++cold; break;
            case NodePhase::Warm: ++warm; break;
            case NodePhase::Mature: ++mature; break;
            case NodePhase::Dormant: ++dormant; break;
        }
        if (channels_[slot] == 0U) ++anchor;
        else ++residual;
    }
    std::uint64_t sparse_entries = 0U;
    std::uint64_t sparse_capacity = 0U;
    std::uint64_t sparse_admission_capacity = 0U;
    std::uint64_t max_sparse_entries = 0U;
    std::uint64_t saturated_sparse_nodes = 0U;
    std::uint64_t max_sparse_decision_visits = 0U;
    for (const auto& entries : sparse_outputs_) {
        sparse_entries += entries.size();
        sparse_capacity += entries.capacity();
        max_sparse_entries = std::max<std::uint64_t>(max_sparse_entries, entries.size());
        if (entries.size() >= config_.max_sparse_decisions_per_node) {
            ++saturated_sparse_nodes;
        }
        for (const auto& entry : entries) {
            max_sparse_decision_visits = std::max<std::uint64_t>(
                max_sparse_decision_visits, entry.visits);
        }
    }
    for (const auto& candidates : sparse_admission_) {
        sparse_admission_capacity += candidates.capacity();
    }
    const std::uint64_t address_index_bytes =
        bucket_directory_.bucket_count() * sizeof(void*) +
        bucket_directory_.size() *
            (sizeof(std::pair<const std::size_t, BucketState>) + 2U * sizeof(void*)) +
        bucket_capacity * sizeof(NodeId);
    const std::uint64_t output_structure_bytes =
        (implicit_output_.has_value() ? sizeof(detail::ImplicitOutputTree) : 0U) +
        token_path_scratch_.capacity() * sizeof(detail::ImplicitDecision);
    const std::uint64_t global_output_prior_bytes =
        (global_output_total_.capacity() + global_output_right_.capacity()) *
            sizeof(std::uint64_t) +
        global_output_logit_cache_.capacity() * sizeof(float);
    const auto denominator = std::max<std::uint64_t>(1, total_steps_);
    const std::uint64_t bytes =
        ids_.capacity() * sizeof(NodeId) +
        prototypes_.capacity() * sizeof(std::uint64_t) +
        visits_.capacity() * sizeof(std::uint32_t) +
        address_visits_.capacity() * sizeof(std::uint32_t) +
        utility_ema_.capacity() * sizeof(float) +
        loss_ema_.capacity() * sizeof(float) +
        address_loss_ema_.capacity() * sizeof(float) +
        phases_.capacity() * sizeof(std::uint8_t) +
        channels_.capacity() * sizeof(std::uint8_t) +
        hot_indexed_.capacity() * sizeof(std::uint8_t) +
        parents_.capacity() * sizeof(NodeId) +
        output_vectors_.capacity() * sizeof(float) +
        sparse_capacity * sizeof(detail::SparseOutputEntry) +
        sparse_admission_.capacity() * sizeof(std::vector<SparseAdmissionCandidate>) +
        sparse_admission_capacity * sizeof(SparseAdmissionCandidate) +
        sparse_output_evicted_masks_.capacity() * sizeof(std::uint64_t) +
        id_to_slot_.capacity() * sizeof(std::uint32_t) +
        edge_capacity * sizeof(Edge) +
        address_index_bytes + output_structure_bytes + global_output_prior_bytes;
    std::uint64_t seed_channels = 0U;
    std::uint64_t probe_channels = 0U;
    std::uint64_t active_channels = 0U;
    std::uint64_t retired_channels = 0U;
    std::uint64_t quarantined_channels = 0U;
    std::uint64_t recoverable_retired_channels = 0U;
    std::uint64_t dependency_blocked_channels = 0U;
    std::uint64_t topology_restored = 0U;
    for (const auto& event : topology_events_) {
        if (event.decision == TopologyDecision::Restored) {
            ++topology_restored;
        }
    }
    for (std::size_t channel = 0U; channel < topology_.size(); ++channel) {
        const auto& state = topology_[channel];
        switch (state.phase) {
            case ChannelPhase::Seed: ++seed_channels; break;
            case ChannelPhase::Probe: ++probe_channels; break;
            case ChannelPhase::Active: ++active_channels; break;
            case ChannelPhase::Retired: ++retired_channels; break;
            case ChannelPhase::Quarantined: ++quarantined_channels; break;
            case ChannelPhase::RecoverableRetired:
                ++recoverable_retired_channels;
                break;
        }
        const bool phase_enabled = state.phase == ChannelPhase::Seed ||
            state.phase == ChannelPhase::Probe ||
            state.phase == ChannelPhase::Active;
        if (phase_enabled && !channel_enabled(channel)) {
            ++dependency_blocked_channels;
        }
    }
    Diagnostics result;
    result.steps = total_steps_;
    result.live_nodes = ids_.size();
    result.logical_ids_issued = next_id_;
    result.edges = edge_count;
    result.avg_active =
        static_cast<double>(total_active_) / static_cast<double>(denominator);
    result.avg_candidates =
        static_cast<double>(total_candidates_) / static_cast<double>(denominator);
    result.created_total = total_created_;
    result.merged_total = total_merged_;
    result.pruned_total = total_pruned_;
    result.cold_nodes = cold;
    result.warm_nodes = warm;
    result.mature_nodes = mature;
    result.dormant_nodes = dormant;
    result.anchor_nodes = anchor;
    result.residual_nodes = residual;
    result.stale_bucket_refs_skipped = stale_bucket_refs_skipped_;
    result.stale_edge_refs_skipped = stale_edge_refs_skipped_;
    result.candidate_source_exact_bucket = candidate_source_exact_bucket_;
    result.candidate_source_control_edge = candidate_source_control_edge_;
    result.candidate_source_neighbor_bucket = candidate_source_neighbor_bucket_;
    result.route_score_hamming_sum = route_score_hamming_sum_;
    result.route_score_exact_sum = route_score_exact_sum_;
    result.route_score_edge_prior_sum = route_score_edge_prior_sum_;
    result.gpaf_role_observations = gpaf_role_observations_total_;
    result.gpaf_unique_role_keys = gpaf_role_observations_.size();
    result.gpaf_slots_allocated = gpaf_slot_phases_.size();
    for (const auto& [key, phase] : gpaf_slot_phases_) {
        (void)key;
        switch (static_cast<GpafSlotPhase>(phase)) {
        case GpafSlotPhase::Probe:
            ++result.gpaf_probe_slots;
            break;
        case GpafSlotPhase::Active:
            ++result.gpaf_active_slots;
            break;
        case GpafSlotPhase::Quarantined:
            ++result.gpaf_quarantined_slots;
            break;
        case GpafSlotPhase::RecoverableRetired:
            ++result.gpaf_recoverable_retired_slots;
            break;
        case GpafSlotPhase::PhysicallyErased:
            ++result.gpaf_physically_erased_slots;
            break;
        }
    }
    result.gpaf_shadow_updates = gpaf_shadow_updates_;
    result.gpaf_slots_probed = gpaf_slots_probed_;
    result.gpaf_candidates_returned = gpaf_candidates_returned_;
    result.estimated_bytes = bytes;
    result.sparse_output_entries = sparse_entries;
    result.topology_proposals = topology_proposals_;
    result.topology_accepted = topology_accepted_;
    result.topology_rejected = topology_rejected_;
    result.topology_pruned = topology_pruned_;
    result.topology_restored = topology_restored;
    result.seed_channels = seed_channels;
    result.probe_channels = probe_channels;
    result.active_channels = active_channels;
    result.retired_channels = retired_channels;
    result.dependency_blocked_channels = dependency_blocked_channels;
    result.quarantined_channels = quarantined_channels;
    result.recoverable_retired_channels = recoverable_retired_channels;
    result.address_execution_frames = address_execution_frames_;
    result.address_binding_hits = address_binding_hits_;
    result.address_binding_misses = address_binding_misses_;
    result.binding_reuse_observations = binding_reuse_observations_;
    for (const auto& channel_records : binding_reuse_) {
        result.binding_reuse_unique_keys += channel_records.size();
    }
    result.binding_reuse_events = binding_reuse_events_;
    result.address_binding_kind_frames = address_binding_kind_frames_;
    result.address_binding_kind_hits = address_binding_kind_hits_;
    result.address_binding_kind_distance_sum =
        address_binding_kind_distance_sum_;
    result.address_binding_kind_pattern_span_sum =
        address_binding_kind_pattern_span_sum_;
    result.structural_description_cost = structural_description_cost_;
    result.structural_execution_cost = structural_execution_cost_;
    result.simd_enabled = simd_available();
    result.address_index_bytes = address_index_bytes;
    result.address_occupied_buckets = occupied_buckets;
    result.address_full_buckets = full_buckets;
    result.address_max_bucket_residents = max_bucket_residents;
    result.address_capacity_blocked_splits = address_capacity_blocked_splits_;
    result.output_structure_bytes = output_structure_bytes;
    result.max_bucket_candidates_inspected = max_bucket_candidates_inspected_;
    result.max_sparse_entries_per_node = max_sparse_entries;
    result.global_output_prior_bytes = global_output_prior_bytes;
    result.global_output_prior_updates = global_output_prior_updates_;
    result.sparse_output_insertions = sparse_output_insertions_;
    result.sparse_output_evictions = sparse_output_evictions_;
    result.sparse_output_probable_reconstructions =
        sparse_output_probable_reconstructions_;
    result.sparse_output_saturated_nodes = saturated_sparse_nodes;
    result.max_sparse_decision_visits = max_sparse_decision_visits;
    result.sparse_output_admission_rejections =
        sparse_output_admission_rejections_;
    result.sparse_output_admission_promotions =
        sparse_output_admission_promotions_;
    result.max_responsibility_mass_error = max_responsibility_mass_error_;
    return result;
}

} // namespace sbm
