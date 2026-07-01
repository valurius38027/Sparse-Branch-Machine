#pragma once

#include "sbm/types.hpp"
#include "sbm/detail/implicit_output.hpp"
#include "sbm/detail/sparse_output.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sbm {

struct CheckpointAccess;

class SparseBranchMachine {
public:
    explicit SparseBranchMachine(Config config);

    [[nodiscard]] StepStats step(std::uint32_t token, std::span<const float> target,
                                 bool learn = true);
    [[nodiscard]] StepStats step_token(std::uint32_t token,
                                       std::uint32_t target_token,
                                       bool learn = true);
    void reset_sequence();
    void freeze_topology();
    [[nodiscard]] bool retire_channel(std::size_t channel,
                                      AcceptedChannelRetirement policy);
    [[nodiscard]] bool restore_channel(std::size_t channel);
    [[nodiscard]] std::size_t restore_dependency_closure(std::size_t channel);
    [[nodiscard]] std::size_t quarantine_gpaf_slots();
    [[nodiscard]] std::size_t recoverably_retire_gpaf_slots();
    [[nodiscard]] std::size_t restore_gpaf_slots();
    [[nodiscard]] std::size_t prune(std::uint32_t min_visits = 8,
                                    float utility_threshold = -0.05F);
    [[nodiscard]] std::size_t merge_redundant(std::size_t max_merges = 64);
    void rebuild_indexes();
    void prefill_distractors(std::size_t count);

    [[nodiscard]] Diagnostics diagnostics() const noexcept;
    [[nodiscard]] const Config& config() const noexcept { return config_; }
    [[nodiscard]] bool contains(NodeId id) const noexcept;
    [[nodiscard]] std::size_t live_nodes() const noexcept { return ids_.size(); }
    [[nodiscard]] NodePhase phase(NodeId id) const noexcept;
    [[nodiscard]] std::span<const float> last_prediction() const noexcept {
        return prediction_buffer_;
    }
    [[nodiscard]] std::vector<std::uint32_t> learned_address_lags() const;
    [[nodiscard]] std::vector<AddressProgram> learned_address_programs() const;
    [[nodiscard]] std::vector<float> learned_channel_credit() const;
    [[nodiscard]] std::vector<std::uint8_t> learned_channel_phase() const;
    [[nodiscard]] std::vector<std::uint8_t>
        learned_channel_effective_enabled() const;
    [[nodiscard]] std::vector<std::uint8_t> learned_channel_parent() const;
    [[nodiscard]] std::vector<std::uint8_t> learned_channel_dependency() const;
    [[nodiscard]] std::vector<std::uint64_t> learned_channel_generation() const;
    [[nodiscard]] std::vector<std::uint8_t> learned_channel_parent_edge_kind() const;
    [[nodiscard]] std::vector<std::uint8_t> learned_channel_dependency_edge_kind() const;
    [[nodiscard]] std::span<const AddressExecutionFrame> last_execution_frames()
        const noexcept {
        return execution_frames_;
    }
    [[nodiscard]] const std::vector<TopologyEvent>& topology_events() const noexcept {
        return topology_events_;
    }

    friend void save_checkpoint(const SparseBranchMachine& machine,
                                const std::string& path);
    friend SparseBranchMachine load_checkpoint(const std::string& path);
    friend struct CheckpointAccess;

private:
    enum class CandidateSource : std::uint8_t {
        ExactBucket,
        ControlEdge,
        NeighborBucket,
        GpafRole,
    };
    enum class GpafSlotPhase : std::uint8_t {
        Probe,
        Active,
        Quarantined,
        RecoverableRetired,
        PhysicallyErased,
    };
    struct CandidateNode {
        NodeId id{kInvalidNode};
        float edge_prior{};
        CandidateSource source{CandidateSource::ExactBucket};
        std::uint64_t gpaf_key{UINT64_MAX};
    };
    struct ScoredNode {
        double score{};
        NodeId id{kInvalidNode};
        float responsibility{};
        float contribution{};
        bool exact_region{};
        std::uint8_t channel{};
        CandidateSource source{CandidateSource::ExactBucket};
        std::uint64_t gpaf_key{UINT64_MAX};
    };
    struct AddressChannelState {
        AddressProgram program{};
        ChannelPhase phase{ChannelPhase::Retired};
        float credit_ema{};
        double credit_sum{};
        std::uint64_t observations{};
        std::uint64_t born_step{};
        std::uint64_t generation{};
        std::uint8_t parent_channel{kInvalidChannel};
        std::uint8_t dependency_channel{kInvalidChannel};
        AddressGraphEdgeKind parent_edge_kind{AddressGraphEdgeKind::None};
        AddressGraphEdgeKind dependency_edge_kind{AddressGraphEdgeKind::None};
    };
    using SparseOutputEntry = detail::SparseOutputEntry;
    struct SparseAdmissionCandidate {
        std::uint32_t decision{};
        std::uint8_t sightings{};
        std::uint64_t last_seen_step{};
    };
    struct BindingReuseRecord {
        std::uint64_t key{};
        std::uint64_t observations{};
        std::uint64_t last_seen_step{};
    };
    struct BindingReuseSummary {
        std::uint64_t observations{};
        std::uint64_t unique_keys{};
        std::uint64_t events{};
    };
    struct TraceFrame {
        std::vector<NodeId> route;
        std::vector<float> contribution;
        float loss{};
    };
    struct BucketState {
        std::vector<NodeId> residents;
        std::vector<NodeId> hot;
        std::vector<NodeId> cold;
        std::uint64_t last_split_step{};
    };

    [[nodiscard]] std::uint32_t bucket(std::uint64_t signature) const noexcept;
    [[nodiscard]] bool uses_sparse_token_output() const noexcept;
    [[nodiscard]] float sparse_logit(std::size_t slot, std::uint32_t decision) const noexcept;
    SparseOutputEntry* mutable_sparse_entry(std::size_t slot,
                                            std::uint32_t decision,
                                            bool force_admission = false);
    [[nodiscard]] float aggregate_sparse_logit(std::span<const ScoredNode> active,
                                               std::uint32_t decision) const noexcept;
    [[nodiscard]] float aggregate_sparse_logit_masked(
        std::span<const ScoredNode> active,
        std::uint32_t decision,
        std::uint32_t channel_mask,
        float mass_scale = 1.0F) const noexcept;
    [[nodiscard]] float global_output_logit(std::uint32_t decision) const noexcept;
    void observe_global_output_path(
        std::span<const detail::ImplicitDecision> path);
    [[nodiscard]] StepStats step_token_dense(std::uint32_t token,
                                             std::uint32_t target_token,
                                             bool learn);
    [[nodiscard]] StepStats step_token_sparse(std::uint32_t token,
                                              std::uint32_t target_token,
                                              bool learn);
    [[nodiscard]] bool channel_enabled(std::size_t channel) const noexcept;
    [[nodiscard]] bool channel_learning_enabled(std::size_t channel) const noexcept;
    [[nodiscard]] std::span<const AddressExecutionFrame> execute_address_programs(
        std::span<const std::uint32_t> window, bool learn);
    [[nodiscard]] std::span<const std::uint64_t> make_signatures(
        std::span<const std::uint32_t> window, bool learn);
    void observe_binding_reuse(std::size_t channel, std::uint64_t key);
    void observe_gpaf_shadow_roles(std::span<const ScoredNode> active);
    [[nodiscard]] std::uint64_t gpaf_role_key_for_channel(
        std::uint8_t channel) const noexcept;
    [[nodiscard]] std::uint64_t gpaf_structural_call_key_for_channel(
        std::uint8_t channel) const noexcept;
    [[nodiscard]] BindingReuseSummary binding_reuse_summary(
        std::size_t channel) const noexcept;
    [[nodiscard]] double binding_reuse_bonus(
        const BindingReuseSummary& summary) const noexcept;
    void maybe_begin_topology_probe(bool learn);
    void maybe_finalize_topology_probe();
    void observe_topology_credit(std::span<const float> channel_credit);
    [[nodiscard]] std::uint8_t find_channel_for_program(
        const AddressProgram& program) const noexcept;
    [[nodiscard]] std::uint8_t find_positional_channel_for_lag(
        std::uint32_t lag) const noexcept;
    [[nodiscard]] std::uint8_t find_prefix_channel(
        const AddressProgram& program) const noexcept;
    [[nodiscard]] std::pair<std::uint8_t, std::uint8_t> resolve_channel_lineage(
        const AddressProgram& program) const noexcept;
    void quarantine_channel(std::size_t channel);
    void recoverably_retire_channel(std::size_t channel);
    void physically_erase_channel(std::size_t channel);
    [[nodiscard]] std::size_t bucket_index(std::uint8_t channel,
                                           std::uint64_t signature) const noexcept;
    [[nodiscard]] const BucketState* find_bucket(std::size_t index) const noexcept;
    [[nodiscard]] BucketState& ensure_bucket(std::size_t index);
    [[nodiscard]] std::span<const NodeId> bounded_bucket_nodes(
        std::size_t index, std::size_t limit);
    void mark_hot(NodeId id);
    [[nodiscard]] NodeId new_node(std::uint64_t signature, std::span<const float> initial,
                                  NodeId parent = kInvalidNode,
                                  std::uint8_t channel = 0U);
    [[nodiscard]] std::size_t slot_of(NodeId id) const noexcept;
    [[nodiscard]] std::span<const CandidateNode> candidate_ids(
        std::span<const std::uint64_t> signatures,
        std::int64_t max_radius = 2,
        bool update_gpaf_state = true);
    [[nodiscard]] double score(std::size_t slot,
                               std::span<const std::uint64_t> signatures,
                               float edge_prior) const noexcept;
    [[nodiscard]] std::pair<std::span<ScoredNode>, std::uint32_t>
        select_route(std::span<const std::uint64_t> signatures,
                     std::int64_t max_radius = 2,
                     bool update_gpaf_state = true);
    void assign_responsibilities(std::span<ScoredNode> active) const;
    void aggregate(std::span<const ScoredNode> active, std::span<float> output) const;
    void compute_counterfactual_contributions(std::span<ScoredNode> active,
                                              std::span<const float> target,
                                              float target_energy,
                                              float full_loss) const;
    [[nodiscard]] double output_distance(std::size_t a, std::size_t b) const noexcept;
    [[nodiscard]] NodePhase phase_of_slot(std::size_t slot) const noexcept;

    void reinforce_edge(NodeId source, NodeId destination, float delta);
    void decay_edges(NodeId source);
    void apply_trace_credit(float normalized_loss);
    void update_phase(std::size_t slot);
    void absorb_node(std::size_t survivor_slot, std::size_t victim_slot);
    void erase_slot(std::size_t slot);
    [[nodiscard]] bool push_candidate(std::vector<CandidateNode>& values, NodeId id,
                                      float edge_prior,
                                      CandidateSource source,
                                      std::uint64_t gpaf_key = UINT64_MAX) const noexcept;

    Config config_;
    std::vector<AddressChannelState> topology_;
    std::uint64_t proposal_cursor_{};
    std::vector<std::uint64_t> proposed_program_keys_;
    std::uint64_t topology_generation_counter_{};
    std::uint64_t next_probe_step_{};
    std::uint64_t topology_proposals_{};
    std::uint64_t topology_accepted_{};
    std::uint64_t topology_rejected_{};
    std::uint64_t topology_pruned_{};
    std::vector<TopologyEvent> topology_events_;
    std::uint64_t rng_state_{};

    std::vector<NodeId> ids_;
    std::vector<std::uint64_t> prototypes_;
    std::vector<std::uint32_t> visits_;
    std::vector<std::uint32_t> address_visits_;
    std::vector<float> utility_ema_;
    std::vector<float> loss_ema_;
    std::vector<float> address_loss_ema_;
    std::vector<std::uint8_t> phases_;
    std::vector<std::uint8_t> channels_;
    std::vector<std::uint8_t> hot_indexed_;
    std::vector<NodeId> parents_;
    std::vector<std::vector<BindingReuseRecord>> binding_reuse_;
    std::unordered_map<std::uint64_t, std::uint64_t> gpaf_role_observations_;
    std::unordered_map<std::uint64_t, std::uint64_t>
        gpaf_structural_call_observations_;
    std::unordered_map<std::uint64_t, std::uint8_t> gpaf_slot_phases_;
    std::unordered_map<std::uint64_t, std::vector<NodeId>> gpaf_residents_;
    std::vector<float> output_vectors_;
    std::vector<std::vector<detail::SparseOutputEntry>> sparse_outputs_;
    std::vector<std::vector<SparseAdmissionCandidate>> sparse_admission_;
    std::vector<std::uint64_t> sparse_output_evicted_masks_;
    std::optional<detail::ImplicitOutputTree> implicit_output_;
    std::vector<detail::ImplicitDecision> token_path_scratch_;
    std::vector<std::uint64_t> global_output_total_;
    std::vector<std::uint64_t> global_output_right_;
    std::vector<float> global_output_logit_cache_;
    std::uint64_t global_output_prior_updates_{};
    std::vector<std::vector<Edge>> edges_;

    std::vector<std::uint32_t> id_to_slot_;
    NodeId next_id_{};

    std::unordered_map<std::size_t, BucketState> bucket_directory_;
    std::vector<NodeId> previous_route_;
    std::vector<float> previous_responsibilities_;
    std::deque<TraceFrame> trace_;
    std::vector<std::uint32_t> history_;
    std::vector<float> prediction_buffer_;
    std::vector<float> logit_buffer_;
    std::vector<float> zero_output_buffer_;
    std::vector<float> channel_output_buffer_;
    std::vector<float> channel_credit_buffer_;
    std::vector<std::uint64_t> signature_buffer_;
    std::vector<AddressExecutionFrame> execution_frames_;
    std::vector<CandidateNode> candidate_scratch_;
    std::vector<ScoredNode> scored_scratch_;
    std::vector<ScoredNode> selected_scratch_;
    std::vector<std::uint8_t> chosen_scratch_;
    std::vector<std::size_t> order_scratch_;
    std::vector<NodeId> bucket_node_scratch_;

    std::uint64_t total_steps_{};
    std::uint64_t total_candidates_{};
    std::uint64_t total_active_{};
    std::uint64_t total_created_{};
    std::uint64_t total_merged_{};
    std::uint64_t total_pruned_{};
    std::uint64_t candidate_source_exact_bucket_{};
    std::uint64_t candidate_source_control_edge_{};
    std::uint64_t candidate_source_neighbor_bucket_{};
    double route_score_hamming_sum_{};
    double route_score_exact_sum_{};
    double route_score_edge_prior_sum_{};
    std::uint64_t gpaf_role_observations_total_{};
    std::uint64_t gpaf_slot_promotions_{};
    std::uint64_t gpaf_slot_quarantines_{};
    std::uint64_t gpaf_slot_recoverable_retires_{};
    std::uint64_t gpaf_slot_restores_{};
    std::uint64_t gpaf_shadow_updates_{};
    std::uint64_t gpaf_slots_probed_{};
    std::uint64_t gpaf_candidates_returned_{};
    std::uint64_t gpaf_structural_call_candidates_returned_{};
    std::uint64_t gpaf_structural_call_blocked_{};
    mutable std::uint64_t stale_bucket_refs_skipped_{};
    mutable std::uint64_t stale_edge_refs_skipped_{};
    std::uint64_t max_bucket_candidates_inspected_{};
    std::uint64_t address_capacity_blocked_splits_{};
    std::uint64_t address_execution_frames_{};
    std::uint64_t address_binding_hits_{};
    std::uint64_t address_binding_misses_{};
    std::uint64_t binding_reuse_observations_{};
    std::uint64_t binding_reuse_events_{};
    std::array<std::uint64_t, kAddressBindingKindCount>
        address_binding_kind_frames_{};
    std::array<std::uint64_t, kAddressBindingKindCount>
        address_binding_kind_hits_{};
    std::array<double, kAddressBindingKindCount>
        address_binding_kind_distance_sum_{};
    std::array<double, kAddressBindingKindCount>
        address_binding_kind_pattern_span_sum_{};
    double structural_description_cost_{};
    double structural_execution_cost_{};
    std::uint64_t sparse_output_insertions_{};
    std::uint64_t sparse_output_evictions_{};
    std::uint64_t sparse_output_probable_reconstructions_{};
    std::uint64_t sparse_output_admission_rejections_{};
    std::uint64_t sparse_output_admission_promotions_{};
    mutable double max_responsibility_mass_error_{};
};

void save_checkpoint(const SparseBranchMachine& machine, const std::string& path);
[[nodiscard]] SparseBranchMachine load_checkpoint(const std::string& path);

} // namespace sbm
