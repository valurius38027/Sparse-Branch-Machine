#include "sbm/machine.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace sbm {
namespace {

constexpr std::array<char, 8> kMagic{'S', 'B', 'M', 'C', 'K', 'P', 'T', 'A'};

template <class T>
void write_scalar(std::ostream& out, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("failed to write checkpoint scalar");
}

template <class T>
T read_scalar(std::istream& in) {
    static_assert(std::is_trivially_copyable_v<T>);
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("failed to read checkpoint scalar");
    return value;
}

template <class T>
void write_vector(std::ostream& out, const std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    write_scalar<std::uint64_t>(out, values.size());
    if (!values.empty()) {
        out.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(T)));
        if (!out) throw std::runtime_error("failed to write checkpoint vector");
    }
}

template <class T>
std::vector<T> read_vector(std::istream& in) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto size = read_scalar<std::uint64_t>(in);
    if (size > static_cast<std::uint64_t>(SIZE_MAX / sizeof(T))) {
        throw std::runtime_error("checkpoint vector is too large");
    }
    std::vector<T> values(static_cast<std::size_t>(size));
    if (!values.empty()) {
        in.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(values.size() * sizeof(T)));
        if (!in) throw std::runtime_error("failed to read checkpoint vector");
    }
    return values;
}

template <class T>
void write_nested_vector(std::ostream& out, const std::vector<std::vector<T>>& values) {
    write_scalar<std::uint64_t>(out, values.size());
    for (const auto& value : values) write_vector(out, value);
}

template <class T>
std::vector<std::vector<T>> read_nested_vector(std::istream& in) {
    const auto size = read_scalar<std::uint64_t>(in);
    std::vector<std::vector<T>> values;
    values.reserve(static_cast<std::size_t>(size));
    for (std::uint64_t index = 0; index < size; ++index) {
        values.push_back(read_vector<T>(in));
    }
    return values;
}

void write_u64_map(std::ostream& out,
                   const std::unordered_map<std::uint64_t, std::uint64_t>& values) {
    write_scalar<std::uint64_t>(out, values.size());
    for (const auto& [key, value] : values) {
        write_scalar(out, key);
        write_scalar(out, value);
    }
}

std::unordered_map<std::uint64_t, std::uint64_t> read_u64_map(std::istream& in) {
    const auto size = read_scalar<std::uint64_t>(in);
    std::unordered_map<std::uint64_t, std::uint64_t> values;
    values.reserve(static_cast<std::size_t>(size));
    for (std::uint64_t index = 0; index < size; ++index) {
        const auto key = read_scalar<std::uint64_t>(in);
        const auto value = read_scalar<std::uint64_t>(in);
        values.emplace(key, value);
    }
    return values;
}

void write_u64_u8_map(std::ostream& out,
                      const std::unordered_map<std::uint64_t, std::uint8_t>& values) {
    write_scalar<std::uint64_t>(out, values.size());
    for (const auto& [key, value] : values) {
        write_scalar(out, key);
        write_scalar(out, value);
    }
}

std::unordered_map<std::uint64_t, std::uint8_t> read_u64_u8_map(std::istream& in) {
    const auto size = read_scalar<std::uint64_t>(in);
    std::unordered_map<std::uint64_t, std::uint8_t> values;
    values.reserve(static_cast<std::size_t>(size));
    for (std::uint64_t index = 0; index < size; ++index) {
        const auto key = read_scalar<std::uint64_t>(in);
        const auto value = read_scalar<std::uint8_t>(in);
        values.emplace(key, value);
    }
    return values;
}

void write_resident_map(
    std::ostream& out,
    const std::unordered_map<std::uint64_t, std::vector<NodeId>>& values) {
    write_scalar<std::uint64_t>(out, values.size());
    for (const auto& [key, residents] : values) {
        write_scalar(out, key);
        write_vector(out, residents);
    }
}

std::unordered_map<std::uint64_t, std::vector<NodeId>> read_resident_map(
    std::istream& in) {
    const auto size = read_scalar<std::uint64_t>(in);
    std::unordered_map<std::uint64_t, std::vector<NodeId>> values;
    values.reserve(static_cast<std::size_t>(size));
    for (std::uint64_t index = 0; index < size; ++index) {
        const auto key = read_scalar<std::uint64_t>(in);
        values.emplace(key, read_vector<NodeId>(in));
    }
    return values;
}

void write_config(std::ostream& out, const Config& config) {
    write_scalar(out, config.objective);
    write_scalar(out, config.token_alphabet);
    write_scalar(out, config.vector_dim);
    write_scalar(out, config.context_width);
    write_scalar(out, config.bucket_bits);
    write_scalar(out, config.beam_width);
    write_scalar(out, config.beam_width_min);
    write_scalar(out, config.confidence_threshold);
    write_scalar(out, config.max_refinement_rounds);
    write_scalar(out, config.refinement_confidence_threshold);
    write_scalar(out, config.bucket_scan_limit);
    write_scalar(out, config.edge_scan_limit);
    write_scalar(out, config.max_edges_per_node);
    write_scalar(out, config.edge_reinforce_width);
    write_scalar(out, config.max_specializations_per_bucket);
    write_scalar(out, config.split_min_visits);
    write_scalar(out, config.split_cooldown);
    write_scalar(out, config.split_context_similarity);
    write_scalar(out, config.split_loss_threshold);
    write_scalar(out, config.allow_growth_when_frozen);
    write_scalar(out, config.trace_horizon);
    write_scalar(out, config.trace_decay);
    write_scalar(out, config.edge_decay);
    write_scalar(out, config.edge_learning_rate);
    write_scalar(out, config.edge_score_weight);
    write_scalar(out, config.edge_min_contribution);
    write_scalar(out, config.responsibility_temperature);
    write_scalar(out, config.exact_region_mass);
    write_scalar(out, config.min_update_responsibility);
    write_vector(out, config.address_lags);
    write_scalar(out, config.adaptive_topology);
    write_scalar(out, config.max_address_channels);
    write_scalar(out, config.topology_max_lag);
    write_scalar(out, config.topology_max_arity);
    write_scalar(out, config.topology_enable_delta);
    write_scalar(out, config.topology_enable_content_match);
    write_scalar(out, config.topology_enable_content_follow_multi);
    write_scalar(out, config.topology_probe_interval);
    write_scalar(out, config.topology_probe_warmup);
    write_scalar(out, config.topology_probe_steps);
    write_scalar(out, config.topology_validation_steps);
    write_scalar(out, config.topology_min_observations);
    write_scalar(out, config.topology_accept_credit);
    write_scalar(out, config.topology_credit_decay);
    write_scalar(out, config.topology_prune_patience);
    write_scalar(out, config.topology_prune_credit);
    write_scalar(out, config.address_execution_mode);
    write_scalar(out, config.accepted_channel_retirement);
    write_scalar(out, config.structural_description_cost_weight);
    write_scalar(out, config.structural_execution_cost_weight);
    write_scalar(out, config.binding_reuse_value_weight);
    write_scalar(out, config.topology_accept_uses_structural_value);
    write_scalar(out, config.residual_channel_gain);
    write_scalar(out, config.residual_learning_rate);
    write_scalar(out, config.residual_mature_learning_rate);
    write_scalar(out, config.residual_recency_pseudocount);
    write_scalar(out, config.warm_visits);
    write_scalar(out, config.mature_visits);
    write_scalar(out, config.dormant_utility);
    write_scalar(out, config.node_learning_rate);
    write_scalar(out, config.mature_learning_rate);
    write_scalar(out, config.merge_similarity);
    write_scalar(out, config.merge_vector_distance);
    write_scalar(out, config.classification_learning_rate);
    write_scalar(out, config.classification_mature_learning_rate);
    write_scalar(out, config.label_smoothing);
    write_scalar(out, config.softmax_temperature);
    write_scalar(out, config.logit_decay);
    write_scalar(out, config.sparse_token_output);
    write_scalar(out, config.sparse_output_topk);
    write_scalar(out, config.sparse_output_beam_width);
    write_scalar(out, config.max_sparse_decisions_per_node);
    write_scalar(out, config.decode_token_ranking_during_training);
    write_scalar(out, config.use_momentum);
    write_scalar(out, config.momentum_beta1);
    write_scalar(out, config.momentum_beta2);
    write_scalar(out, config.momentum_eps);
    write_scalar(out, config.record_channel_attribution);
    write_scalar(out, config.max_binding_reuse_records_per_channel);
    write_scalar(out, config.gpaf_shadow_observation);
    write_scalar(out, config.gpaf_candidate_retrieval);
    write_scalar(out, config.gpaf_query_keys_per_step);
    write_scalar(out, config.gpaf_slots);
    write_scalar(out, config.gpaf_residents_per_slot);
    write_scalar(out, config.output_tree_seed);
    write_scalar(out, config.seed);
}

Config read_config(std::istream& in) {
    Config config;
    config.objective = read_scalar<ObjectiveKind>(in);
    config.token_alphabet = read_scalar<std::uint32_t>(in);
    config.vector_dim = read_scalar<std::uint32_t>(in);
    config.context_width = read_scalar<std::uint32_t>(in);
    config.bucket_bits = read_scalar<std::uint32_t>(in);
    config.beam_width = read_scalar<std::uint32_t>(in);
    config.beam_width_min = read_scalar<std::uint32_t>(in);
    config.confidence_threshold = read_scalar<float>(in);
    config.max_refinement_rounds = read_scalar<std::uint32_t>(in);
    config.refinement_confidence_threshold = read_scalar<float>(in);
    config.bucket_scan_limit = read_scalar<std::uint32_t>(in);
    config.edge_scan_limit = read_scalar<std::uint32_t>(in);
    config.max_edges_per_node = read_scalar<std::uint32_t>(in);
    config.edge_reinforce_width = read_scalar<std::uint32_t>(in);
    config.max_specializations_per_bucket = read_scalar<std::uint32_t>(in);
    config.split_min_visits = read_scalar<std::uint32_t>(in);
    config.split_cooldown = read_scalar<std::uint32_t>(in);
    config.split_context_similarity = read_scalar<double>(in);
    config.split_loss_threshold = read_scalar<float>(in);
    config.allow_growth_when_frozen = read_scalar<bool>(in);
    config.trace_horizon = read_scalar<std::uint32_t>(in);
    config.trace_decay = read_scalar<float>(in);
    config.edge_decay = read_scalar<float>(in);
    config.edge_learning_rate = read_scalar<float>(in);
    config.edge_score_weight = read_scalar<float>(in);
    config.edge_min_contribution = read_scalar<float>(in);
    config.responsibility_temperature = read_scalar<float>(in);
    config.exact_region_mass = read_scalar<float>(in);
    config.min_update_responsibility = read_scalar<float>(in);
    config.address_lags = read_vector<std::uint32_t>(in);
    config.adaptive_topology = read_scalar<bool>(in);
    config.max_address_channels = read_scalar<std::uint32_t>(in);
    config.topology_max_lag = read_scalar<std::uint32_t>(in);
    config.topology_max_arity = read_scalar<std::uint32_t>(in);
    config.topology_enable_delta = read_scalar<bool>(in);
    config.topology_enable_content_match = read_scalar<bool>(in);
    config.topology_enable_content_follow_multi = read_scalar<bool>(in);
    config.topology_probe_interval = read_scalar<std::uint32_t>(in);
    config.topology_probe_warmup = read_scalar<std::uint32_t>(in);
    config.topology_probe_steps = read_scalar<std::uint32_t>(in);
    config.topology_validation_steps = read_scalar<std::uint32_t>(in);
    config.topology_min_observations = read_scalar<std::uint32_t>(in);
    config.topology_accept_credit = read_scalar<float>(in);
    config.topology_credit_decay = read_scalar<float>(in);
    config.topology_prune_patience = read_scalar<std::uint32_t>(in);
    config.topology_prune_credit = read_scalar<float>(in);
    config.address_execution_mode = read_scalar<AddressExecutionMode>(in);
    config.accepted_channel_retirement = read_scalar<AcceptedChannelRetirement>(in);
    config.structural_description_cost_weight = read_scalar<float>(in);
    config.structural_execution_cost_weight = read_scalar<float>(in);
    config.binding_reuse_value_weight = read_scalar<float>(in);
    config.topology_accept_uses_structural_value = read_scalar<bool>(in);
    config.residual_channel_gain = read_scalar<float>(in);
    config.residual_learning_rate = read_scalar<float>(in);
    config.residual_mature_learning_rate = read_scalar<float>(in);
    config.residual_recency_pseudocount = read_scalar<float>(in);
    config.warm_visits = read_scalar<std::uint32_t>(in);
    config.mature_visits = read_scalar<std::uint32_t>(in);
    config.dormant_utility = read_scalar<float>(in);
    config.node_learning_rate = read_scalar<float>(in);
    config.mature_learning_rate = read_scalar<float>(in);
    config.merge_similarity = read_scalar<float>(in);
    config.merge_vector_distance = read_scalar<float>(in);
    config.classification_learning_rate = read_scalar<float>(in);
    config.classification_mature_learning_rate = read_scalar<float>(in);
    config.label_smoothing = read_scalar<float>(in);
    config.softmax_temperature = read_scalar<float>(in);
    config.logit_decay = read_scalar<float>(in);
    config.sparse_token_output = read_scalar<bool>(in);
    config.sparse_output_topk = read_scalar<std::uint32_t>(in);
    config.sparse_output_beam_width = read_scalar<std::uint32_t>(in);
    config.max_sparse_decisions_per_node = read_scalar<std::uint32_t>(in);
    config.decode_token_ranking_during_training = read_scalar<bool>(in);
    config.use_momentum = read_scalar<bool>(in);
    config.momentum_beta1 = read_scalar<float>(in);
    config.momentum_beta2 = read_scalar<float>(in);
    config.momentum_eps = read_scalar<float>(in);
    config.record_channel_attribution = read_scalar<bool>(in);
    config.max_binding_reuse_records_per_channel = read_scalar<std::uint32_t>(in);
    config.gpaf_shadow_observation = read_scalar<bool>(in);
    config.gpaf_candidate_retrieval = read_scalar<bool>(in);
    config.gpaf_query_keys_per_step = read_scalar<std::uint32_t>(in);
    config.gpaf_slots = read_scalar<std::uint32_t>(in);
    config.gpaf_residents_per_slot = read_scalar<std::uint32_t>(in);
    config.output_tree_seed = read_scalar<std::uint64_t>(in);
    config.seed = read_scalar<std::uint64_t>(in);
    return config;
}

} // namespace

struct CheckpointAccess {
static void write_trace(std::ostream& out,
                        const std::deque<SparseBranchMachine::TraceFrame>& trace) {
    write_scalar<std::uint64_t>(out, trace.size());
    for (const auto& frame : trace) {
        write_vector(out, frame.route);
        write_vector(out, frame.contribution);
        write_scalar(out, frame.loss);
    }
}

static std::deque<SparseBranchMachine::TraceFrame> read_trace(std::istream& in) {
    const auto size = read_scalar<std::uint64_t>(in);
    std::deque<SparseBranchMachine::TraceFrame> trace;
    for (std::uint64_t index = 0; index < size; ++index) {
        SparseBranchMachine::TraceFrame frame;
        frame.route = read_vector<NodeId>(in);
        frame.contribution = read_vector<float>(in);
        frame.loss = read_scalar<float>(in);
        trace.push_back(std::move(frame));
    }
    return trace;
}

static void write_buckets(
    std::ostream& out,
    const std::unordered_map<std::size_t, SparseBranchMachine::BucketState>& buckets) {
    write_scalar<std::uint64_t>(out, buckets.size());
    for (const auto& [key, state] : buckets) {
        write_scalar(out, key);
        write_vector(out, state.residents);
        write_vector(out, state.hot);
        write_vector(out, state.cold);
        write_scalar(out, state.last_split_step);
    }
}

static std::unordered_map<std::size_t, SparseBranchMachine::BucketState>
read_buckets(std::istream& in) {
    const auto size = read_scalar<std::uint64_t>(in);
    std::unordered_map<std::size_t, SparseBranchMachine::BucketState> buckets;
    buckets.reserve(static_cast<std::size_t>(size));
    for (std::uint64_t index = 0; index < size; ++index) {
        const auto key = read_scalar<std::size_t>(in);
        SparseBranchMachine::BucketState state;
        state.residents = read_vector<NodeId>(in);
        state.hot = read_vector<NodeId>(in);
        state.cold = read_vector<NodeId>(in);
        state.last_split_step = read_scalar<std::uint64_t>(in);
        buckets.emplace(key, std::move(state));
    }
    return buckets;
}
};

void save_checkpoint(const SparseBranchMachine& machine, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("failed to open checkpoint for writing");
    out.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    write_config(out, machine.config_);
    write_vector(out, machine.topology_);
    write_scalar(out, machine.proposal_cursor_);
    write_vector(out, machine.proposed_program_keys_);
    write_scalar(out, machine.topology_generation_counter_);
    write_scalar(out, machine.next_probe_step_);
    write_scalar(out, machine.topology_proposals_);
    write_scalar(out, machine.topology_accepted_);
    write_scalar(out, machine.topology_rejected_);
    write_scalar(out, machine.topology_pruned_);
    write_vector(out, machine.topology_events_);
    write_scalar(out, machine.rng_state_);
    write_vector(out, machine.ids_);
    write_vector(out, machine.prototypes_);
    write_vector(out, machine.visits_);
    write_vector(out, machine.address_visits_);
    write_vector(out, machine.utility_ema_);
    write_vector(out, machine.loss_ema_);
    write_vector(out, machine.address_loss_ema_);
    write_vector(out, machine.phases_);
    write_vector(out, machine.channels_);
    write_vector(out, machine.hot_indexed_);
    write_vector(out, machine.parents_);
    write_nested_vector(out, machine.binding_reuse_);
    write_u64_map(out, machine.gpaf_role_observations_);
    write_u64_u8_map(out, machine.gpaf_slot_phases_);
    write_resident_map(out, machine.gpaf_residents_);
    write_vector(out, machine.output_vectors_);
    write_nested_vector(out, machine.sparse_outputs_);
    write_nested_vector(out, machine.sparse_admission_);
    write_vector(out, machine.sparse_output_evicted_masks_);
    write_vector(out, machine.global_output_total_);
    write_vector(out, machine.global_output_right_);
    write_vector(out, machine.global_output_logit_cache_);
    write_scalar(out, machine.global_output_prior_updates_);
    write_nested_vector(out, machine.edges_);
    write_vector(out, machine.id_to_slot_);
    write_scalar(out, machine.next_id_);
    CheckpointAccess::write_buckets(out, machine.bucket_directory_);
    write_vector(out, machine.previous_route_);
    write_vector(out, machine.previous_responsibilities_);
    CheckpointAccess::write_trace(out, machine.trace_);
    write_vector(out, machine.history_);
    write_vector(out, machine.prediction_buffer_);
    write_scalar(out, machine.total_steps_);
    write_scalar(out, machine.total_candidates_);
    write_scalar(out, machine.total_active_);
    write_scalar(out, machine.total_created_);
    write_scalar(out, machine.total_merged_);
    write_scalar(out, machine.total_pruned_);
    write_scalar(out, machine.candidate_source_exact_bucket_);
    write_scalar(out, machine.candidate_source_control_edge_);
    write_scalar(out, machine.candidate_source_neighbor_bucket_);
    write_scalar(out, machine.route_score_hamming_sum_);
    write_scalar(out, machine.route_score_exact_sum_);
    write_scalar(out, machine.route_score_edge_prior_sum_);
    write_scalar(out, machine.gpaf_role_observations_total_);
    write_scalar(out, machine.gpaf_shadow_updates_);
    write_scalar(out, machine.gpaf_slots_probed_);
    write_scalar(out, machine.gpaf_candidates_returned_);
    write_scalar(out, machine.stale_bucket_refs_skipped_);
    write_scalar(out, machine.stale_edge_refs_skipped_);
    write_scalar(out, machine.max_bucket_candidates_inspected_);
    write_scalar(out, machine.address_capacity_blocked_splits_);
    write_scalar(out, machine.address_execution_frames_);
    write_scalar(out, machine.address_binding_hits_);
    write_scalar(out, machine.address_binding_misses_);
    write_scalar(out, machine.binding_reuse_observations_);
    write_scalar(out, machine.binding_reuse_events_);
    write_scalar(out, machine.address_binding_kind_frames_);
    write_scalar(out, machine.address_binding_kind_hits_);
    write_scalar(out, machine.address_binding_kind_distance_sum_);
    write_scalar(out, machine.address_binding_kind_pattern_span_sum_);
    write_scalar(out, machine.structural_description_cost_);
    write_scalar(out, machine.structural_execution_cost_);
    write_scalar(out, machine.sparse_output_insertions_);
    write_scalar(out, machine.sparse_output_evictions_);
    write_scalar(out, machine.sparse_output_probable_reconstructions_);
    write_scalar(out, machine.sparse_output_admission_rejections_);
    write_scalar(out, machine.sparse_output_admission_promotions_);
    write_scalar(out, machine.max_responsibility_mass_error_);
}

SparseBranchMachine load_checkpoint(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open checkpoint for reading");
    std::array<char, 8> magic{};
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!in || magic != kMagic) throw std::runtime_error("invalid checkpoint header");
    SparseBranchMachine machine(read_config(in));
    machine.topology_ = read_vector<SparseBranchMachine::AddressChannelState>(in);
    machine.proposal_cursor_ = read_scalar<std::uint64_t>(in);
    machine.proposed_program_keys_ = read_vector<std::uint64_t>(in);
    machine.topology_generation_counter_ = read_scalar<std::uint64_t>(in);
    machine.next_probe_step_ = read_scalar<std::uint64_t>(in);
    machine.topology_proposals_ = read_scalar<std::uint64_t>(in);
    machine.topology_accepted_ = read_scalar<std::uint64_t>(in);
    machine.topology_rejected_ = read_scalar<std::uint64_t>(in);
    machine.topology_pruned_ = read_scalar<std::uint64_t>(in);
    machine.topology_events_ = read_vector<TopologyEvent>(in);
    machine.rng_state_ = read_scalar<std::uint64_t>(in);
    machine.ids_ = read_vector<NodeId>(in);
    machine.prototypes_ = read_vector<std::uint64_t>(in);
    machine.visits_ = read_vector<std::uint32_t>(in);
    machine.address_visits_ = read_vector<std::uint32_t>(in);
    machine.utility_ema_ = read_vector<float>(in);
    machine.loss_ema_ = read_vector<float>(in);
    machine.address_loss_ema_ = read_vector<float>(in);
    machine.phases_ = read_vector<std::uint8_t>(in);
    machine.channels_ = read_vector<std::uint8_t>(in);
    machine.hot_indexed_ = read_vector<std::uint8_t>(in);
    machine.parents_ = read_vector<NodeId>(in);
    machine.binding_reuse_ =
        read_nested_vector<SparseBranchMachine::BindingReuseRecord>(in);
    machine.gpaf_role_observations_ = read_u64_map(in);
    machine.gpaf_slot_phases_ = read_u64_u8_map(in);
    machine.gpaf_residents_ = read_resident_map(in);
    machine.output_vectors_ = read_vector<float>(in);
    machine.sparse_outputs_ = read_nested_vector<detail::SparseOutputEntry>(in);
    machine.sparse_admission_ =
        read_nested_vector<SparseBranchMachine::SparseAdmissionCandidate>(in);
    machine.sparse_output_evicted_masks_ = read_vector<std::uint64_t>(in);
    machine.global_output_total_ = read_vector<std::uint64_t>(in);
    machine.global_output_right_ = read_vector<std::uint64_t>(in);
    machine.global_output_logit_cache_ = read_vector<float>(in);
    machine.global_output_prior_updates_ = read_scalar<std::uint64_t>(in);
    machine.edges_ = read_nested_vector<Edge>(in);
    machine.id_to_slot_ = read_vector<std::uint32_t>(in);
    machine.next_id_ = read_scalar<NodeId>(in);
    machine.bucket_directory_ = CheckpointAccess::read_buckets(in);
    machine.previous_route_ = read_vector<NodeId>(in);
    machine.previous_responsibilities_ = read_vector<float>(in);
    machine.trace_ = CheckpointAccess::read_trace(in);
    machine.history_ = read_vector<std::uint32_t>(in);
    machine.prediction_buffer_ = read_vector<float>(in);
    machine.total_steps_ = read_scalar<std::uint64_t>(in);
    machine.total_candidates_ = read_scalar<std::uint64_t>(in);
    machine.total_active_ = read_scalar<std::uint64_t>(in);
    machine.total_created_ = read_scalar<std::uint64_t>(in);
    machine.total_merged_ = read_scalar<std::uint64_t>(in);
    machine.total_pruned_ = read_scalar<std::uint64_t>(in);
    machine.candidate_source_exact_bucket_ = read_scalar<std::uint64_t>(in);
    machine.candidate_source_control_edge_ = read_scalar<std::uint64_t>(in);
    machine.candidate_source_neighbor_bucket_ = read_scalar<std::uint64_t>(in);
    machine.route_score_hamming_sum_ = read_scalar<double>(in);
    machine.route_score_exact_sum_ = read_scalar<double>(in);
    machine.route_score_edge_prior_sum_ = read_scalar<double>(in);
    machine.gpaf_role_observations_total_ = read_scalar<std::uint64_t>(in);
    machine.gpaf_shadow_updates_ = read_scalar<std::uint64_t>(in);
    machine.gpaf_slots_probed_ = read_scalar<std::uint64_t>(in);
    machine.gpaf_candidates_returned_ = read_scalar<std::uint64_t>(in);
    machine.stale_bucket_refs_skipped_ = read_scalar<std::uint64_t>(in);
    machine.stale_edge_refs_skipped_ = read_scalar<std::uint64_t>(in);
    machine.max_bucket_candidates_inspected_ = read_scalar<std::uint64_t>(in);
    machine.address_capacity_blocked_splits_ = read_scalar<std::uint64_t>(in);
    machine.address_execution_frames_ = read_scalar<std::uint64_t>(in);
    machine.address_binding_hits_ = read_scalar<std::uint64_t>(in);
    machine.address_binding_misses_ = read_scalar<std::uint64_t>(in);
    machine.binding_reuse_observations_ = read_scalar<std::uint64_t>(in);
    machine.binding_reuse_events_ = read_scalar<std::uint64_t>(in);
    machine.address_binding_kind_frames_ =
        read_scalar<std::array<std::uint64_t, kAddressBindingKindCount>>(in);
    machine.address_binding_kind_hits_ =
        read_scalar<std::array<std::uint64_t, kAddressBindingKindCount>>(in);
    machine.address_binding_kind_distance_sum_ =
        read_scalar<std::array<double, kAddressBindingKindCount>>(in);
    machine.address_binding_kind_pattern_span_sum_ =
        read_scalar<std::array<double, kAddressBindingKindCount>>(in);
    machine.structural_description_cost_ = read_scalar<double>(in);
    machine.structural_execution_cost_ = read_scalar<double>(in);
    machine.sparse_output_insertions_ = read_scalar<std::uint64_t>(in);
    machine.sparse_output_evictions_ = read_scalar<std::uint64_t>(in);
    machine.sparse_output_probable_reconstructions_ = read_scalar<std::uint64_t>(in);
    machine.sparse_output_admission_rejections_ = read_scalar<std::uint64_t>(in);
    machine.sparse_output_admission_promotions_ = read_scalar<std::uint64_t>(in);
    machine.max_responsibility_mass_error_ = read_scalar<double>(in);
    if (in.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("checkpoint has trailing bytes");
    }
    return machine;
}

} // namespace sbm
