#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sbm {

using NodeId = std::uint32_t;
inline constexpr NodeId kInvalidNode = UINT32_MAX;
inline constexpr std::uint8_t kInvalidChannel = UINT8_MAX;
inline constexpr std::size_t kMaxAddressChannels = 8U;
inline constexpr std::size_t kMaxGpafAblationKeys = 8U;
inline constexpr std::size_t kMaxAddressProgramArity = 2U;
inline constexpr std::size_t kAddressBindingKindCount = 4U;

enum class NodePhase : std::uint8_t { Cold, Warm, Mature, Dormant };
enum class ChannelPhase : std::uint8_t {
    Seed,
    Probe,
    Active,
    Retired,
    Quarantined,
    RecoverableRetired,
};
enum class TopologyDecision : std::uint8_t {
    Proposed,
    Accepted,
    Rejected,
    Pruned,
    Restored,
};

[[nodiscard]] inline const char* topology_decision_name(
    TopologyDecision decision) noexcept {
    switch (decision) {
    case TopologyDecision::Proposed: return "Proposed";
    case TopologyDecision::Accepted: return "Accepted";
    case TopologyDecision::Rejected: return "Rejected";
    case TopologyDecision::Pruned: return "Pruned";
    case TopologyDecision::Restored: return "Restored";
    }
    return "Unknown";
}
enum class AddressOp : std::uint8_t {
    Tuple = 0U,
    DeltaMod = 1U,
    ContentMatch = 2U,
    ContentFollow = 3U,
    ContentFollowMulti2 = 4U,
    ContentFollowMulti3 = 5U,
};

enum class AddressExecutionMode : std::uint8_t {
    LegacySignature = 0U,
    InterpretedFrames = 1U,
};

enum class AcceptedChannelRetirement : std::uint8_t {
    Preserve = 0U,
    Quarantine = 1U,
    PhysicalErase = 2U,
    RecoverableRetire = 3U,
};

enum class AddressBindingKind : std::uint8_t {
    None = 0U,
    Positional = 1U,
    ContentMatch = 2U,
    ContentFollow = 3U,
};

enum class AddressStateKind : std::uint8_t {
    None = 0U,
    TokenWindow = 1U,
    PositionalSignature = 2U,
    ContentBinding = 3U,
    FollowBinding = 4U,
};

enum class AddressGraphEdgeKind : std::uint8_t {
    None = 0U,
    Prefix = 1U,
    PositionalDependency = 2U,
    ContentMatchDependency = 3U,
    ContentFollowCall = 4U,
};

struct AddressBindingState {
    std::uint32_t current_token{};
    std::uint32_t matched_token{};
    std::uint32_t matched_successor{};
    std::uint32_t matched_index{};
    std::uint32_t matched_distance{};
    std::uint32_t pattern_span{};
    std::uint64_t binding_key{};
    std::uint8_t pattern_terms{};
    bool matched{};
};

struct AddressProgram {
    std::array<std::uint32_t, kMaxAddressProgramArity> lags{};
    std::uint8_t arity{1U};
    AddressOp op{AddressOp::Tuple};

    [[nodiscard]] bool operator==(const AddressProgram&) const noexcept = default;
};

struct AddressExecutionFrame {
    AddressProgram program{};
    AddressStateKind input_state{AddressStateKind::None};
    AddressStateKind output_state{AddressStateKind::None};
    AddressBindingKind binding{AddressBindingKind::None};
    AddressBindingKind required_dependency_binding{AddressBindingKind::None};
    AddressBindingState binding_state{};
    std::uint8_t channel{kInvalidChannel};
    std::uint8_t parent_channel{kInvalidChannel};
    std::uint8_t dependency_channel{kInvalidChannel};
    std::uint32_t source_index{};
    std::uint32_t matched_index{};
    std::uint32_t successor{};
    std::uint32_t dependency{};
    std::uint64_t dependency_signature{};
    std::uint64_t dependency_binding_key{};
    std::uint64_t call_key{};
    std::uint64_t signature{};
    float description_cost{};
    float execution_cost{};
    bool matched{};
    bool call_matched{};
};

[[nodiscard]] inline AddressProgram singleton_address_program(std::uint32_t lag) noexcept {
    AddressProgram program;
    program.lags[0] = lag;
    program.arity = 1U;
    return program;
}

[[nodiscard]] inline std::uint64_t address_program_key(
    const AddressProgram& program) noexcept {
    std::uint64_t key = static_cast<std::uint64_t>(program.arity) << 56U;
    key |= static_cast<std::uint64_t>(program.op) << 48U;
    for (std::size_t index = 0; index < program.arity; ++index) {
        key |= (static_cast<std::uint64_t>(program.lags[index]) & 0x0FFFFFFFULL)
               << (28U * index);
    }
    return key;
}

[[nodiscard]] inline bool is_content_follow_op(AddressOp op) noexcept {
    return op == AddressOp::ContentFollow ||
           op == AddressOp::ContentFollowMulti2 ||
           op == AddressOp::ContentFollowMulti3;
}

[[nodiscard]] inline std::uint8_t content_follow_hop_count(AddressOp op) noexcept {
    switch (op) {
    case AddressOp::ContentFollowMulti2: return 2U;
    case AddressOp::ContentFollowMulti3: return 3U;
    case AddressOp::ContentFollow:
    default: return 1U;
    }
}

[[nodiscard]] inline AddressStateKind address_program_input_state(
    const AddressProgram& program) noexcept {
    return is_content_follow_op(program.op)
        ? AddressStateKind::ContentBinding
        : AddressStateKind::TokenWindow;
}

[[nodiscard]] inline AddressStateKind address_program_output_state(
    const AddressProgram& program) noexcept {
    switch (program.op) {
    case AddressOp::ContentMatch:
        return AddressStateKind::ContentBinding;
    case AddressOp::ContentFollow:
    case AddressOp::ContentFollowMulti2:
    case AddressOp::ContentFollowMulti3:
        return AddressStateKind::FollowBinding;
    case AddressOp::Tuple:
    case AddressOp::DeltaMod:
    default:
        return AddressStateKind::PositionalSignature;
    }
}

[[nodiscard]] inline AddressBindingKind address_program_required_dependency_binding(
    const AddressProgram& program) noexcept {
    return is_content_follow_op(program.op)
        ? AddressBindingKind::ContentMatch
        : AddressBindingKind::None;
}

[[nodiscard]] inline bool address_dependency_satisfied(
    const AddressProgram& program,
    AddressBindingKind dependency_binding) noexcept {
    const auto required = address_program_required_dependency_binding(program);
    return required == AddressBindingKind::None || dependency_binding == required;
}

[[nodiscard]] inline AddressGraphEdgeKind address_dependency_edge_kind(
    const AddressProgram& program) noexcept {
    switch (program.op) {
    case AddressOp::ContentMatch:
        return AddressGraphEdgeKind::ContentMatchDependency;
    case AddressOp::ContentFollow:
    case AddressOp::ContentFollowMulti2:
    case AddressOp::ContentFollowMulti3:
        return AddressGraphEdgeKind::ContentFollowCall;
    case AddressOp::Tuple:
    case AddressOp::DeltaMod:
    default:
        return program.arity > 1U
            ? AddressGraphEdgeKind::Prefix
            : AddressGraphEdgeKind::PositionalDependency;
    }
}

[[nodiscard]] inline AddressGraphEdgeKind address_parent_edge_kind(
    const AddressProgram& program) noexcept {
    if (program.arity > 1U) return AddressGraphEdgeKind::Prefix;
    return address_dependency_edge_kind(program);
}

struct TopologyEvent {
    std::uint64_t step{};
    std::uint64_t channel_generation{};
    AddressProgram program{};
    TopologyDecision decision{TopologyDecision::Proposed};
    float credit{};
    float structural_value_without_reuse{};
    float binding_reuse_bonus{};
    std::uint64_t binding_reuse_observations{};
    std::uint64_t binding_reuse_unique_keys{};
    std::uint64_t binding_reuse_events{};
    std::uint8_t channel{kInvalidChannel};
    std::uint8_t parent_channel{kInvalidChannel};
    std::uint8_t dependency_channel{kInvalidChannel};
    AddressGraphEdgeKind parent_edge_kind{AddressGraphEdgeKind::None};
    AddressGraphEdgeKind dependency_edge_kind{AddressGraphEdgeKind::None};
};

enum class ObjectiveKind : std::uint8_t {
    VectorRegression = 0U,
    TokenCrossEntropy = 1U,
};

struct Edge {
    NodeId dst{kInvalidNode};
    float weight{};
    float eligibility{};
};

struct Config {
    ObjectiveKind objective{ObjectiveKind::VectorRegression};
    std::uint32_t token_alphabet{64};
    std::uint32_t vector_dim{16};
    std::uint32_t context_width{12};
    std::uint32_t bucket_bits{12};
    std::uint32_t beam_width{6};
    // Adaptive beam width: on high-confidence tokens active set is truncated
    // to beam_width_min; beam_width remains the hard upper bound. Both are
    // O(1) with respect to stored capacity.
    std::uint32_t beam_width_min{6};
    float confidence_threshold{0.8F};
    // Iterative refinement: when the initial route has low maximum
    // responsibility, repeat candidate selection with a larger neighbor radius
    // up to max_refinement_rounds. All refinement stays before the target.
    std::uint32_t max_refinement_rounds{0};
    float refinement_confidence_threshold{0.5F};
    std::uint32_t bucket_scan_limit{32};
    std::uint32_t edge_scan_limit{8};
    std::uint32_t max_edges_per_node{32};
    std::uint32_t edge_reinforce_width{2};
    std::uint32_t max_specializations_per_bucket{4};
    std::uint32_t split_min_visits{48};
    std::uint32_t split_cooldown{128};

    double split_context_similarity{0.78};
    float split_loss_threshold{0.72F};
    bool allow_growth_when_frozen{false};

    std::uint32_t trace_horizon{8};
    float trace_decay{0.80F};
    float edge_decay{0.999F};
    float edge_learning_rate{0.08F};
    float edge_score_weight{0.32F};
    float edge_min_contribution{0.004F};
    float responsibility_temperature{3.0F};
    float exact_region_mass{0.88F};
    float min_update_responsibility{0.01F};
    // Seed address views. In adaptive mode these are only the initial topology;
    // additional temporal views are proposed and retained by measured predictive credit.
    std::vector<std::uint32_t> address_lags{1U};
    bool adaptive_topology{true};
    std::uint32_t max_address_channels{6};
    std::uint32_t topology_max_lag{16};
    std::uint32_t topology_max_arity{2};
    bool topology_enable_delta{true};
    bool topology_enable_content_match{true};
    bool topology_enable_content_follow_multi{false};
    std::uint32_t topology_probe_interval{2048};
    std::uint32_t topology_probe_warmup{512};
    std::uint32_t topology_probe_steps{4096};
    std::uint32_t topology_validation_steps{1024};
    std::uint32_t topology_min_observations{512};
    float topology_accept_credit{0.0005F};
    float topology_credit_decay{0.995F};
    std::uint32_t topology_prune_patience{UINT32_MAX};
    float topology_prune_credit{-0.01F};
    AddressExecutionMode address_execution_mode{AddressExecutionMode::InterpretedFrames};
    AcceptedChannelRetirement accepted_channel_retirement{
        AcceptedChannelRetirement::Preserve};
    float structural_description_cost_weight{1.0F};
    float structural_execution_cost_weight{0.0F};
    float binding_reuse_value_weight{0.0F};
    bool topology_accept_uses_structural_value{false};
    bool gpaf_shadow_observation{false};
    bool gpaf_candidate_retrieval{false};
    std::uint32_t gpaf_query_keys_per_step{0};
    std::uint32_t gpaf_slots{0};
    std::uint32_t gpaf_residents_per_slot{0};
    std::uint32_t gpaf_probe_min_observations{64};
    std::uint32_t gpaf_probe_min_residents{1};
    float residual_channel_gain{1.0F};
    float residual_learning_rate{0.10F};
    float residual_mature_learning_rate{0.030F};
    float residual_recency_pseudocount{0.75F};
    std::uint32_t warm_visits{12};
    std::uint32_t mature_visits{96};
    float dormant_utility{-0.30F};
    float node_learning_rate{0.12F};
    float mature_learning_rate{0.035F};
    float merge_similarity{0.95F};
    float merge_vector_distance{0.08F};

    // Token-cross-entropy objective.  vector_dim is set to the output
    // vocabulary size by the token experiment so the same node storage can
    // hold dense local logits without introducing a separate matrix path.
    float classification_learning_rate{0.35F};
    float classification_mature_learning_rate{0.08F};
    float label_smoothing{0.01F};
    float softmax_temperature{1.0F};
    float logit_decay{0.0001F};
    // Exact hierarchical softmax.  Every address node stores only binary
    // decisions encountered on observed target paths, so token training and
    // scoring scale with log(vocabulary) instead of vocabulary size.
    bool sparse_token_output{true};
    std::uint32_t sparse_output_topk{5};
    std::uint32_t sparse_output_beam_width{16};
    std::uint32_t max_sparse_decisions_per_node{64};
    bool decode_token_ranking_during_training{false};
    // Per-entry Adam-like momentum for sparse decision logits. When enabled,
    // each SparseOutputEntry accumulates momentum and gradient variance,
    // adapting its effective learning rate per decision.
    bool use_momentum{false};
    float momentum_beta1{0.9F};
    float momentum_beta2{0.999F};
    float momentum_eps{1e-8F};
    // Optional R2 diagnostic. When enabled, frozen token evaluation records
    // per-address-channel counterfactual codelength contribution. Normal runs
    // leave it disabled to avoid attribution overhead.
    bool record_channel_attribution{false};
    std::uint32_t max_binding_reuse_records_per_channel{4096};

    std::uint64_t output_tree_seed{7};
    std::uint64_t seed{7};
};

struct StepStats {
    float normalized_mse{};
    float cosine{};
    std::uint32_t active_nodes{};
    std::uint32_t candidates_examined{};
    std::uint32_t live_nodes{};
    std::uint32_t created{};
    std::vector<NodeId> route;
    float cross_entropy{};
    float target_probability{};
    bool top1_correct{};
    bool top5_correct{};
    bool ranking_available{true};
    std::uint32_t predicted_token{};
    std::uint8_t channel_credit_count{};
    std::array<float, kMaxAddressChannels> channel_credit{};
    std::array<float, kMaxAddressChannels> channel_responsibility_mass{};
    bool channel_subset_available{};
    float seed_only_cross_entropy{};
    float active_only_cross_entropy{};
    float content_only_cross_entropy{};
    float tuple_only_cross_entropy{};
    bool gpaf_ablation_available{};
    float gpaf_removed_cross_entropy{};
    float gpaf_codelength_gain{};
    float gpaf_false_positive_cost{};
    std::uint32_t gpaf_ablation_nodes{};
    std::uint8_t gpaf_ablation_key_count{};
    std::array<std::uint64_t, kMaxGpafAblationKeys> gpaf_ablation_keys{};
    std::array<std::uint32_t, kMaxGpafAblationKeys> gpaf_ablation_key_nodes{};
    std::array<float, kMaxGpafAblationKeys> gpaf_ablation_key_removed_cross_entropy{};
    std::array<float, kMaxGpafAblationKeys> gpaf_ablation_key_gain{};
    std::array<float, kMaxGpafAblationKeys> gpaf_ablation_key_false_positive_cost{};
    std::array<std::uint32_t, kMaxAddressChannels> channel_dependency{};
    std::array<std::uint8_t, kMaxAddressChannels> channel_parent_channel{};
    std::array<std::uint8_t, kMaxAddressChannels> channel_dependency_channel{};
    std::array<std::uint8_t, kMaxAddressChannels> channel_input_state{};
    std::array<std::uint8_t, kMaxAddressChannels> channel_output_state{};
    std::array<std::uint8_t, kMaxAddressChannels> channel_required_dependency_binding{};
    std::array<float, kMaxAddressChannels> channel_caller_removed_credit{};
    std::array<float, kMaxAddressChannels> channel_dependency_retained_credit{};
    std::array<float, kMaxAddressChannels> channel_dependency_removed_credit{};
    std::array<std::uint8_t, kMaxAddressChannels> channel_binding_kind{};
    std::array<std::uint8_t, kMaxAddressChannels> channel_binding_matched{};
    std::array<std::uint32_t, kMaxAddressChannels> channel_binding_current_token{};
    std::array<std::uint32_t, kMaxAddressChannels> channel_binding_matched_token{};
    std::array<std::uint32_t, kMaxAddressChannels> channel_binding_successor{};
    std::array<std::uint32_t, kMaxAddressChannels> channel_binding_distance{};
    std::array<std::uint32_t, kMaxAddressChannels> channel_binding_pattern_span{};
    std::array<std::uint64_t, kMaxAddressChannels> channel_binding_key{};
    std::array<std::uint64_t, kMaxAddressChannels> channel_dependency_binding_key{};
    std::array<std::uint64_t, kMaxAddressChannels> channel_call_key{};
    std::array<float, kMaxAddressChannels> channel_description_cost{};
    std::array<float, kMaxAddressChannels> channel_execution_cost{};
};

struct Diagnostics {
    std::uint64_t steps{};
    std::uint64_t live_nodes{};
    std::uint64_t logical_ids_issued{};
    std::uint64_t edges{};
    double avg_active{};
    double avg_candidates{};
    std::uint64_t created_total{};
    std::uint64_t merged_total{};
    std::uint64_t pruned_total{};
    std::uint64_t cold_nodes{};
    std::uint64_t warm_nodes{};
    std::uint64_t mature_nodes{};
    std::uint64_t dormant_nodes{};
    std::uint64_t anchor_nodes{};
    std::uint64_t residual_nodes{};
    std::uint64_t stale_bucket_refs_skipped{};
    std::uint64_t stale_edge_refs_skipped{};
    std::uint64_t candidate_source_exact_bucket{};
    std::uint64_t candidate_source_control_edge{};
    std::uint64_t candidate_source_neighbor_bucket{};
    double route_score_hamming_sum{};
    double route_score_exact_sum{};
    double route_score_edge_prior_sum{};
    std::uint64_t gpaf_role_observations{};
    std::uint64_t gpaf_unique_role_keys{};
    std::uint64_t gpaf_slots_allocated{};
    std::uint64_t gpaf_probe_slots{};
    std::uint64_t gpaf_active_slots{};
    std::uint64_t gpaf_quarantined_slots{};
    std::uint64_t gpaf_recoverable_retired_slots{};
    std::uint64_t gpaf_physically_erased_slots{};
    std::uint64_t gpaf_slot_promotions{};
    std::uint64_t gpaf_slot_quarantines{};
    std::uint64_t gpaf_slot_recoverable_retires{};
    std::uint64_t gpaf_slot_restores{};
    std::uint64_t gpaf_shadow_updates{};
    std::uint64_t gpaf_slots_probed{};
    std::uint64_t gpaf_candidates_returned{};
    std::uint64_t gpaf_structural_call_observations{};
    std::uint64_t gpaf_structural_call_keys{};
    std::uint64_t gpaf_structural_call_candidates_returned{};
    std::uint64_t gpaf_structural_call_blocked{};
    std::uint64_t estimated_bytes{};
    std::uint64_t sparse_output_entries{};
    std::uint64_t topology_proposals{};
    std::uint64_t topology_accepted{};
    std::uint64_t topology_rejected{};
    std::uint64_t topology_pruned{};
    std::uint64_t topology_restored{};
    std::uint64_t seed_channels{};
    std::uint64_t probe_channels{};
    std::uint64_t active_channels{};
    std::uint64_t retired_channels{};
    std::uint64_t dependency_blocked_channels{};
    std::uint64_t address_execution_frames{};
    std::uint64_t address_binding_hits{};
    std::uint64_t address_binding_misses{};
    std::uint64_t binding_reuse_observations{};
    std::uint64_t binding_reuse_unique_keys{};
    std::uint64_t binding_reuse_events{};
    std::array<std::uint64_t, kAddressBindingKindCount> address_binding_kind_frames{};
    std::array<std::uint64_t, kAddressBindingKindCount> address_binding_kind_hits{};
    std::array<double, kAddressBindingKindCount> address_binding_kind_distance_sum{};
    std::array<double, kAddressBindingKindCount> address_binding_kind_pattern_span_sum{};
    std::uint64_t quarantined_channels{};
    std::uint64_t recoverable_retired_channels{};
    double structural_value_nats{};
    double structural_description_cost{};
    double structural_execution_cost{};
    bool simd_enabled{};
    std::uint64_t address_index_bytes{};
    std::uint64_t address_occupied_buckets{};
    std::uint64_t address_full_buckets{};
    std::uint64_t address_max_bucket_residents{};
    std::uint64_t address_capacity_blocked_splits{};
    std::uint64_t output_structure_bytes{};
    std::uint64_t max_bucket_candidates_inspected{};
    std::uint64_t max_sparse_entries_per_node{};
    std::uint64_t global_output_prior_bytes{};
    std::uint64_t global_output_prior_updates{};
    std::uint64_t sparse_output_insertions{};
    std::uint64_t sparse_output_evictions{};
    std::uint64_t sparse_output_probable_reconstructions{};
    std::uint64_t sparse_output_saturated_nodes{};
    std::uint64_t max_sparse_decision_visits{};
    std::uint64_t sparse_output_admission_rejections{};
    std::uint64_t sparse_output_admission_promotions{};
    double max_responsibility_mass_error{};
};

struct ProgramAttribution {
    std::uint8_t channel{};
    std::uint8_t parent_channel{kInvalidChannel};
    std::uint8_t dependency_channel{kInvalidChannel};
    std::uint64_t channel_generation{};
    AddressGraphEdgeKind parent_edge_kind{AddressGraphEdgeKind::None};
    AddressGraphEdgeKind dependency_edge_kind{AddressGraphEdgeKind::None};
    AddressStateKind input_state{AddressStateKind::None};
    AddressStateKind output_state{AddressStateKind::None};
    AddressBindingKind required_dependency_binding{AddressBindingKind::None};
    AddressProgram program{};
    std::uint32_t dependency{};
    double credit_sum{};
    double caller_removed_credit_sum{};
    double dependency_retained_credit_sum{};
    double dependency_removed_credit_sum{};
    double binding_distance_sum{};
    double binding_pattern_span_sum{};
    std::uint64_t binding_matches{};
    std::uint64_t unique_binding_keys{};
    std::uint64_t binding_key_reuse_events{};
    std::uint64_t call_matches{};
    std::uint64_t unique_call_keys{};
    std::uint64_t call_key_reuse_events{};
    double description_cost{};
    double execution_cost{};
    std::uint64_t observations{};
    std::uint64_t positive{};
};

} // namespace sbm
