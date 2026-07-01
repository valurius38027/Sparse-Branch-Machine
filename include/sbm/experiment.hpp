#pragma once

#include "sbm/dataset.hpp"
#include "sbm/types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <span>
#include <vector>

namespace sbm {

struct VectorMetrics {
    double normalized_rmse{};
    double cosine{};
    double r2{};
};

struct TokenMetrics {
    double cross_entropy{};
    double bits_per_token{};
    double perplexity{};
    double top1_accuracy{};
    double top5_accuracy{};
    double mean_target_probability{};
    bool ranking_available{true};
};

struct ChannelAttribution {
    std::uint8_t channel{};
    AddressProgram program{};
    std::uint8_t phase{};
    std::uint64_t eval_observations{};
    std::uint64_t eval_positive{};
    std::uint64_t eval_documents{};
    std::uint64_t eval_positive_documents{};
    double eval_credit_sum{};
    double eval_mean_credit{};
    double eval_positive_fraction{};
    double eval_positive_document_fraction{};
};

struct ChannelDependencySummary {
    std::uint8_t channel{};
    AddressProgram program{};
    std::uint8_t phase{};
    std::uint8_t parent_channel{kInvalidChannel};
    std::uint8_t dependency_channel{kInvalidChannel};
    std::uint64_t channel_generation{};
    AddressGraphEdgeKind parent_edge_kind{AddressGraphEdgeKind::None};
    AddressGraphEdgeKind dependency_edge_kind{AddressGraphEdgeKind::None};
    AddressStateKind input_state{AddressStateKind::None};
    AddressStateKind output_state{AddressStateKind::None};
    AddressBindingKind required_dependency_binding{AddressBindingKind::None};
    std::uint8_t effective_enabled{1U};
    std::uint8_t dependency_available{1U};
    std::uint64_t direct_caller_count{};
    std::uint64_t effective_direct_caller_count{};
    std::uint64_t blocked_direct_caller_count{};
    std::uint64_t own_observations{};
    std::uint64_t own_binding_matches{};
    std::uint64_t unique_binding_keys{};
    std::uint64_t binding_key_reuse_events{};
    std::uint64_t own_call_matches{};
    std::uint64_t unique_call_keys{};
    std::uint64_t call_key_reuse_events{};
    double own_credit_sum{};
    double own_caller_removed_credit_sum{};
    std::uint64_t downstream_call_matches{};
    double downstream_caller_removed_credit_sum{};
    double downstream_dependency_removed_credit_sum{};
};

struct ChannelDependencyEdge {
    std::uint8_t caller_channel{kInvalidChannel};
    std::uint8_t dependency_channel{kInvalidChannel};
    std::uint64_t caller_generation{};
    std::uint64_t dependency_generation{};
    AddressGraphEdgeKind edge_kind{AddressGraphEdgeKind::None};
    AddressStateKind caller_input_state{AddressStateKind::None};
    AddressStateKind dependency_output_state{AddressStateKind::None};
    AddressBindingKind required_dependency_binding{AddressBindingKind::None};
    std::uint64_t observations{};
    std::uint64_t call_matches{};
    std::uint64_t unique_call_keys{};
    std::uint64_t call_key_reuse_events{};
    double caller_removed_credit_sum{};
    double dependency_removed_credit_sum{};
};

struct ExperimentResult {
    Diagnostics diagnostics;
    VectorMetrics train;
    VectorMetrics eval;
    VectorMetrics mean_baseline_eval;
    VectorMetrics token_baseline_eval;
    VectorMetrics pair_baseline_eval;
    VectorMetrics pair_lag2_baseline_eval;
    VectorMetrics multiscale_baseline_eval;
    VectorMetrics seen_context_eval;
    VectorMetrics unseen_context_eval;
    std::uint64_t seen_context_vectors{};
    std::uint64_t unseen_context_vectors{};
    double steps_per_second{};
    double elapsed_seconds{};
    bool strict_freeze{};
    std::uint64_t dataset_hash{};
    std::uint32_t vector_dim{};
    std::uint32_t token_alphabet{};
    std::vector<std::uint32_t> address_lags;
    std::vector<std::uint32_t> learned_address_lags;
    std::vector<AddressProgram> learned_address_programs;
    std::vector<float> learned_channel_credit;
    std::vector<std::uint8_t> learned_channel_phase;
    std::vector<std::uint8_t> learned_channel_effective_enabled;
    std::vector<std::uint8_t> learned_channel_parent;
    std::vector<std::uint8_t> learned_channel_dependency;
    std::vector<std::uint64_t> learned_channel_generation;
    std::vector<std::uint8_t> learned_channel_parent_edge_kind;
    std::vector<std::uint8_t> learned_channel_dependency_edge_kind;
    std::vector<ChannelAttribution> eval_channel_attribution;
    std::vector<TopologyEvent> topology_events;
    float exact_region_mass{};
    float residual_channel_gain{};
    float residual_recency_pseudocount{};
    float edge_score_weight{};
};

struct TokenExperimentResult {
    std::string task{"mathematical_next_token_cross_entropy"};
    Diagnostics diagnostics;
    TokenMetrics train;
    TokenMetrics eval;
    TokenMetrics unigram_baseline_eval;
    TokenMetrics current_token_baseline_eval;
    TokenMetrics pair_context_baseline_eval;
    TokenMetrics multiscale_baseline_eval;
    TokenMetrics interpolated_multiscale_baseline_eval;
    TokenMetrics seed_only_eval;
    TokenMetrics active_channels_only_eval;
    TokenMetrics content_channels_only_eval;
    TokenMetrics tuple_channels_only_eval;
    double oracle_cross_entropy{};
    double excess_cross_entropy{};
    std::uint64_t gpaf_ablation_examples{};
    std::uint64_t gpaf_ablation_nodes{};
    double gpaf_ablation_mean_removed_cross_entropy{};
    double gpaf_ablation_mean_gain{};
    double gpaf_ablation_false_positive_cost{};
    std::uint8_t gpaf_ablation_key_count{};
    std::array<std::uint64_t, kMaxGpafAblationKeys> gpaf_ablation_keys{};
    std::array<std::uint64_t, kMaxGpafAblationKeys> gpaf_ablation_key_examples{};
    std::array<std::uint64_t, kMaxGpafAblationKeys> gpaf_ablation_key_nodes{};
    std::array<double, kMaxGpafAblationKeys>
        gpaf_ablation_key_mean_removed_cross_entropy{};
    std::array<double, kMaxGpafAblationKeys> gpaf_ablation_key_mean_gain{};
    std::array<double, kMaxGpafAblationKeys>
        gpaf_ablation_key_false_positive_cost{};
    double steps_per_second{};
    double elapsed_seconds{};
    double baseline_elapsed_seconds{};
    bool strict_freeze{};
    std::uint64_t dataset_hash{};
    std::uint32_t vocab_size{};
    std::uint64_t train_examples{};
    std::uint64_t eval_examples{};
    std::uint64_t sequence_count{};
    std::vector<std::uint32_t> address_lags;
    std::vector<std::uint32_t> learned_address_lags;
    std::vector<AddressProgram> learned_address_programs;
    std::vector<float> learned_channel_credit;
    std::vector<std::uint8_t> learned_channel_phase;
    std::vector<std::uint8_t> learned_channel_effective_enabled;
    std::vector<std::uint8_t> learned_channel_parent;
    std::vector<std::uint8_t> learned_channel_dependency;
    std::vector<std::uint64_t> learned_channel_generation;
    std::vector<std::uint8_t> learned_channel_parent_edge_kind;
    std::vector<std::uint8_t> learned_channel_dependency_edge_kind;
    std::vector<ChannelAttribution> eval_channel_attribution;
    std::vector<double> eval_channel_mean_responsibility;
    std::vector<ProgramAttribution> eval_program_attribution;
    std::vector<ChannelDependencySummary> address_dependency_graph;
    std::vector<ChannelDependencyEdge> address_dependency_edges;
    std::vector<TopologyEvent> topology_events;
    float exact_region_mass{};
    float edge_score_weight{};
    float softmax_temperature{};
    float label_smoothing{};
    bool sparse_token_output{};
    std::uint64_t output_tree_seed{};
    float structural_description_cost_weight{};
    float structural_execution_cost_weight{};
    float binding_reuse_value_weight{};
};

[[nodiscard]] ExperimentResult run_experiment(
    const VectorDataset& dataset,
    std::size_t warmup,
    std::uint64_t seed = 7,
    bool strict_freeze = true,
    std::size_t prefill = 0,
    std::size_t prune_interval = 0,
    std::size_t merge_interval = 0);

[[nodiscard]] ExperimentResult run_experiment(
    const VectorDataset& dataset,
    std::size_t warmup,
    Config config,
    bool strict_freeze = true,
    std::size_t prefill = 0,
    std::size_t prune_interval = 0,
    std::size_t merge_interval = 0);

[[nodiscard]] TokenExperimentResult run_token_experiment(
    const TokenDataset& dataset,
    std::size_t warmup_examples,
    Config config,
    bool strict_freeze = true,
    std::size_t prefill = 0,
    std::size_t prune_interval = 0,
    std::size_t merge_interval = 0);

[[nodiscard]] TokenExperimentResult run_token_experiment(
    const MappedTokenShard& shard,
    std::size_t warmup_examples,
    Config config,
    bool strict_freeze = true,
    std::size_t prefill = 0,
    std::size_t prune_interval = 0,
    std::size_t merge_interval = 0);

[[nodiscard]] TokenExperimentResult run_token_corpus_experiment(
    std::span<const MappedTokenShard* const> train_shards,
    std::span<const MappedTokenShard* const> eval_shards,
    Config config,
    bool strict_freeze = true,
    std::size_t prefill = 0,
    std::size_t prune_interval = 0,
    std::size_t merge_interval = 0);

[[nodiscard]] TokenExperimentResult run_token_corpus_experiment_limited(
    std::span<const MappedTokenShard* const> train_shards,
    std::span<const MappedTokenShard* const> eval_shards,
    std::size_t max_train_examples,
    std::size_t max_eval_examples,
    Config config,
    bool strict_freeze = true,
    std::size_t prefill = 0,
    std::size_t prune_interval = 0,
    std::size_t merge_interval = 0);

[[nodiscard]] std::string to_json(const ExperimentResult& result);
[[nodiscard]] std::string to_json(const TokenExperimentResult& result);

} // namespace sbm
