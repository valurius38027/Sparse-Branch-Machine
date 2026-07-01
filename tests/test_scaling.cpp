#include "sparse_branch_machine.hpp"
#include "sbm/detail/implicit_output.hpp"
#include "sbm/detail/sparse_output.hpp"

#include <cassert>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

sbm::Config sparse_config(std::uint32_t bucket_bits, std::uint32_t vocabulary) {
    sbm::Config config;
    config.objective = sbm::ObjectiveKind::TokenCrossEntropy;
    config.token_alphabet = vocabulary;
    config.vector_dim = vocabulary;
    config.bucket_bits = bucket_bits;
    config.address_lags = {1U};
    config.adaptive_topology = false;
    config.max_address_channels = 1U;
    config.sparse_token_output = true;
    config.seed = 17U;
    return config;
}

std::uint64_t address_bytes(std::uint32_t bucket_bits) {
    sbm::SparseBranchMachine machine(sparse_config(bucket_bits, 4096U));
    return machine.diagnostics().address_index_bytes;
}

std::uint64_t output_bytes(std::uint32_t vocabulary) {
    sbm::SparseBranchMachine machine(sparse_config(4U, vocabulary));
    return machine.diagnostics().output_structure_bytes;
}

double address_p95_microseconds(std::size_t nodes) {
    auto config = sparse_config(10U, 128U);
    config.bucket_scan_limit = 16U;
    config.max_edges_per_node = 1U;
    config.edge_scan_limit = 1U;
    sbm::SparseBranchMachine machine(config);
    machine.prefill_distractors(nodes);
    machine.reset_sequence();
    for (std::uint32_t index = 0U; index < 64U; ++index) {
        (void)machine.step_token(index % 128U, (index + 1U) % 128U, false);
    }
    constexpr std::uint32_t batch_steps = 256U;
    std::vector<double> samples;
    samples.reserve(64U);
    std::uint32_t token_index = 0U;
    for (std::uint32_t batch = 0U; batch < 64U; ++batch) {
        const auto start = std::chrono::steady_clock::now();
        for (std::uint32_t index = 0U; index < batch_steps; ++index, ++token_index) {
            (void)machine.step_token(
                token_index % 128U, (token_index + 1U) % 128U, false);
        }
        const auto elapsed = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count() /
            static_cast<double>(batch_steps);
        samples.push_back(elapsed);
    }
    std::sort(samples.begin(), samples.end());
    return samples[static_cast<std::size_t>(samples.size() * 95U / 100U)];
}

void verify_implicit_output(std::uint32_t vocabulary) {
    sbm::detail::ImplicitOutputTree tree(vocabulary, 29U);
    std::vector<sbm::detail::ImplicitDecision> path;
    const std::uint32_t stride = std::max(1U, vocabulary / 997U);
    for (std::uint32_t token = 0U; token < vocabulary; token += stride) {
        const auto rank = tree.rank_from_token(token);
        assert(rank < vocabulary);
        assert(tree.token_from_rank(rank) == token);
        tree.target_path(token, path);
        std::uint32_t lo = 0U;
        std::uint32_t hi = vocabulary;
        for (const auto& decision : path) {
            const auto split = tree.split(lo, hi);
            assert(decision.id == split.decision_id);
            assert(decision.id < vocabulary - 1U);
            if (decision.right) lo = split.middle;
            else hi = split.middle;
        }
        assert(hi - lo == 1U);
        assert(lo == rank);
    }
}

void verify_output_seed_decoupled() {
    auto left_config = sparse_config(4U, 5U);
    left_config.seed = 7U;
    left_config.output_tree_seed = 29U;
    auto right_config = left_config;
    right_config.seed = 19U;
    sbm::SparseBranchMachine left(left_config);
    sbm::SparseBranchMachine right(right_config);
    for (std::uint32_t target = 0U; target < 5U; ++target) {
        left.reset_sequence();
        right.reset_sequence();
        const auto left_step = left.step_token(0U, target, false);
        const auto right_step = right.step_token(0U, target, false);
        assert(std::abs(left_step.target_probability -
                        right_step.target_probability) < 1e-7F);
    }
}

void verify_global_output_prior() {
    auto config = sparse_config(4U, 4U);
    config.output_tree_seed = 7U;
    config.min_update_responsibility = 2.0F;
    sbm::SparseBranchMachine machine(config);

    const auto first = machine.step_token(0U, 1U, true);
    assert(std::abs(first.target_probability - 0.25F) < 1e-6F);
    (void)machine.step_token(0U, 1U, true);
    (void)machine.step_token(0U, 1U, true);
    (void)machine.step_token(0U, 0U, true);

    const auto trained = machine.diagnostics();
    assert(trained.global_output_prior_updates == 4U);
    assert(trained.global_output_prior_bytes ==
           (config.vector_dim - 1U) *
               (2U * sizeof(std::uint64_t) + sizeof(float)));

    const auto frozen_first = machine.step_token(0U, 1U, false);
    const auto frozen_second = machine.step_token(0U, 1U, false);
    assert(std::abs(frozen_first.target_probability -
                    frozen_second.target_probability) < 1e-7F);
    assert(frozen_first.predicted_token == 1U);
    assert(machine.diagnostics().global_output_prior_updates == 4U);

    sbm::detail::ImplicitOutputTree tree(config.vector_dim, config.output_tree_seed);
    std::vector<sbm::detail::ImplicitDecision> path;
    tree.target_path(1U, path);
    std::uint64_t total[3]{};
    std::uint64_t right[3]{};
    for (const auto target : {1U, 1U, 1U, 0U}) {
        tree.target_path(target, path);
        for (const auto& decision : path) {
            ++total[decision.id];
            right[decision.id] += decision.right ? 1U : 0U;
        }
    }
    tree.target_path(1U, path);
    double expected_probability = 1.0;
    for (const auto& decision : path) {
        const double right_probability =
            (static_cast<double>(right[decision.id]) + 0.5) /
            (static_cast<double>(total[decision.id]) + 1.0);
        expected_probability *= decision.right
            ? right_probability : 1.0 - right_probability;
    }
    assert(std::abs(static_cast<double>(frozen_first.target_probability) -
                    expected_probability) < 1e-6);
}

void verify_sparse_output_policy() {
    using sbm::detail::SparseOutputEntry;
    {
        std::vector<SparseOutputEntry> entries{
            {1U, 0.0F, 100U, 0.05F, 0U},
            {2U, 8.0F, 1U, 0.0F, 0U},
        };
        assert(sbm::detail::select_sparse_output_victim(entries, 100U) == 1U);
    }
    {
        std::vector<SparseOutputEntry> entries{
            {1U, 0.0F, 100U, 0.05F, 0U},
            {2U, 0.0F, 0U, 0.0F, 99U},
        };
        assert(sbm::detail::select_sparse_output_victim(entries, 100U) == 0U);
    }
    {
        std::vector<SparseOutputEntry> entries{
            {2U, 0.0F, 4U, 0.0F, 10U},
            {5U, 0.0F, 4U, 0.0F, 10U},
        };
        assert(sbm::detail::select_sparse_output_victim(entries, 100U) == 1U);
    }
    SparseOutputEntry fresh{1U, 0.0F, 0U, 0.0F, 0U};
    SparseOutputEntry mature{1U, 0.0F, 96U, 0.0F, 0U};
    assert(std::abs(sbm::detail::sparse_decision_learning_rate(
                        fresh, 0.35F, 0.08F, 96U) - 0.35F) < 1e-7F);
    assert(std::abs(sbm::detail::sparse_decision_learning_rate(
                        mature, 0.35F, 0.08F, 96U) -
                    0.08F / std::sqrt(97.0F)) < 1e-7F);
}

void verify_fresh_decision_learning_in_mature_node() {
    auto full_config = sparse_config(4U, 4U);
    full_config.max_specializations_per_bucket = 1U;
    full_config.split_min_visits = UINT32_MAX;
    auto prior_config = full_config;
    prior_config.min_update_responsibility = 2.0F;
    sbm::SparseBranchMachine full(full_config);
    sbm::SparseBranchMachine prior_only(prior_config);
    for (std::uint32_t step = 0U; step < 256U; ++step) {
        (void)full.step_token(0U, 0U, true);
        (void)prior_only.step_token(0U, 0U, true);
    }
    const float full_before = full.step_token(0U, 3U, false).target_probability;
    const float prior_before = prior_only.step_token(0U, 3U, false).target_probability;
    (void)full.step_token(0U, 3U, true);
    (void)prior_only.step_token(0U, 3U, true);
    const float full_after = full.step_token(0U, 3U, false).target_probability;
    const float prior_after = prior_only.step_token(0U, 3U, false).target_probability;
    const double local_log_gain =
        std::log(static_cast<double>(full_after) / full_before) -
        std::log(static_cast<double>(prior_after) / prior_before);
    assert(local_log_gain > 0.02);
}

void verify_refinement_rounds() {
    auto refine_config = sparse_config(4U, 8U);
    refine_config.max_specializations_per_bucket = 1U;
    refine_config.split_min_visits = UINT32_MAX;
    refine_config.max_refinement_rounds = 2U;
    refine_config.refinement_confidence_threshold = 0.5F;

    sbm::SparseBranchMachine refine_machine(refine_config);

    float min_target_prob = 1.0F;
    for (std::uint32_t step = 0U; step < 128U; ++step) {
        const auto stats = refine_machine.step_token(2U, 5U, true);
        min_target_prob = std::min(min_target_prob, stats.target_probability);
        assert(stats.active_nodes > 0U);
        assert(stats.active_nodes <= refine_config.beam_width);
    }
    assert(min_target_prob > 0.0F);
}

void verify_adaptive_beam_width() {
    auto adaptive_config = sparse_config(4U, 8U);
    adaptive_config.max_specializations_per_bucket = 1U;
    adaptive_config.split_min_visits = UINT32_MAX;
    adaptive_config.beam_width_min = 2U;
    adaptive_config.beam_width = 8U;
    adaptive_config.confidence_threshold = 0.8F;

    auto fixed_config = sparse_config(4U, 8U);
    fixed_config.max_specializations_per_bucket = 1U;
    fixed_config.split_min_visits = UINT32_MAX;
    fixed_config.beam_width = 8U;

    sbm::SparseBranchMachine adaptive_machine(adaptive_config);
    sbm::SparseBranchMachine fixed_machine(fixed_config);

    std::uint32_t min_adaptive_active = 999U;
    std::uint32_t max_adaptive_active = 0U;
    for (std::uint32_t step = 0U; step < 256U; ++step) {
        const auto adaptive_stats = adaptive_machine.step_token(2U, 5U, true);
        const auto fixed_stats = fixed_machine.step_token(2U, 5U, true);
        min_adaptive_active = std::min(min_adaptive_active, adaptive_stats.active_nodes);
        max_adaptive_active = std::max(max_adaptive_active, adaptive_stats.active_nodes);
        assert(adaptive_stats.active_nodes <= fixed_stats.active_nodes);
    }

    assert(min_adaptive_active <= 2U);
    assert(max_adaptive_active > 0U);
    assert(max_adaptive_active <= 8U);
}

void verify_momentum_learning() {
    auto base_config = sparse_config(4U, 8U);
    base_config.max_specializations_per_bucket = 1U;
    base_config.split_min_visits = UINT32_MAX;
    base_config.classification_learning_rate = 0.35F;
    base_config.classification_mature_learning_rate = 0.08F;

    auto momentum_config = base_config;
    momentum_config.use_momentum = true;

    auto plain_config = base_config;
    plain_config.use_momentum = false;

    sbm::SparseBranchMachine momentum_machine(momentum_config);
    sbm::SparseBranchMachine plain_machine(plain_config);

    for (std::uint32_t step = 0U; step < 128U; ++step) {
        (void)momentum_machine.step_token(2U, 5U, true);
        (void)plain_machine.step_token(2U, 5U, true);
    }

    const float momentum_prob =
        momentum_machine.step_token(2U, 5U, false).target_probability;
    const float plain_prob =
        plain_machine.step_token(2U, 5U, false).target_probability;

    assert(momentum_prob > plain_prob);
    assert(momentum_prob > 0.02F);
}

void verify_conserved_channel_mass() {
    auto single_left_config = sparse_config(6U, 32U);
    single_left_config.residual_channel_gain = 0.5F;
    auto single_right_config = single_left_config;
    single_right_config.residual_channel_gain = 2.0F;
    sbm::SparseBranchMachine single_left(single_left_config);
    sbm::SparseBranchMachine single_right(single_right_config);
    const auto left = single_left.step_token(3U, 7U, true);
    const auto right = single_right.step_token(3U, 7U, true);
    assert(std::abs(left.target_probability - right.target_probability) < 1e-7F);

    auto multi_config = sparse_config(6U, 32U);
    multi_config.address_lags = {1U, 2U};
    multi_config.max_address_channels = 2U;
    multi_config.residual_channel_gain = 2.0F;
    sbm::SparseBranchMachine multi(multi_config);
    for (std::uint32_t step = 0U; step < 64U; ++step) {
        (void)multi.step_token(step % 32U, (step + 1U) % 32U, true);
    }
    assert(multi.diagnostics().max_responsibility_mass_error < 1e-6);
}

void verify_multi_channel_token_residual_learning() {
    auto single_config = sparse_config(4U, 4U);
    single_config.max_specializations_per_bucket = 1U;
    single_config.split_min_visits = UINT32_MAX;
    single_config.classification_learning_rate = 0.8F;
    single_config.classification_mature_learning_rate = 0.2F;
    auto multi_config = single_config;
    multi_config.address_lags = {1U, 2U};
    multi_config.max_address_channels = 2U;

    auto single_prior_config = single_config;
    single_prior_config.min_update_responsibility = 2.0F;
    auto multi_prior_config = multi_config;
    multi_prior_config.min_update_responsibility = 2.0F;

    sbm::SparseBranchMachine single(single_config);
    sbm::SparseBranchMachine multi(multi_config);
    sbm::SparseBranchMachine single_prior(single_prior_config);
    sbm::SparseBranchMachine multi_prior(multi_prior_config);
    for (std::uint32_t step = 0U; step < 128U; ++step) {
        const auto token = step & 1U;
        (void)single.step_token(token, 0U, true);
        (void)multi.step_token(token, 0U, true);
        (void)single_prior.step_token(token, 0U, true);
        (void)multi_prior.step_token(token, 0U, true);
    }

    const auto token = 1U;
    const float single_before = single.step_token(token, 3U, false).target_probability;
    const float multi_before = multi.step_token(token, 3U, false).target_probability;
    const float single_prior_before =
        single_prior.step_token(token, 3U, false).target_probability;
    const float multi_prior_before =
        multi_prior.step_token(token, 3U, false).target_probability;

    (void)single.step_token(token, 3U, true);
    (void)multi.step_token(token, 3U, true);
    (void)single_prior.step_token(token, 3U, true);
    (void)multi_prior.step_token(token, 3U, true);

    const float single_after = single.step_token(token, 3U, false).target_probability;
    const float multi_after = multi.step_token(token, 3U, false).target_probability;
    const float single_prior_after =
        single_prior.step_token(token, 3U, false).target_probability;
    const float multi_prior_after =
        multi_prior.step_token(token, 3U, false).target_probability;

    const double single_local_gain =
        std::log(static_cast<double>(single_after) / single_before) -
        std::log(static_cast<double>(single_prior_after) / single_prior_before);
    const double multi_local_gain =
        std::log(static_cast<double>(multi_after) / multi_before) -
        std::log(static_cast<double>(multi_prior_after) / multi_prior_before);
    assert(single_local_gain > 0.02);
    assert(multi_local_gain > 0.50 * single_local_gain);
}

void verify_address_capacity_pressure() {
    auto config = sparse_config(1U, 32U);
    config.max_specializations_per_bucket = 1U;
    config.split_min_visits = 1U;
    config.split_cooldown = 0U;
    config.split_context_similarity = 1.01;
    config.split_loss_threshold = 0.0F;
    sbm::SparseBranchMachine machine(config);
    for (std::uint32_t step = 0U; step < 128U; ++step) {
        (void)machine.step_token(step % 32U, (step + 7U) % 32U, true);
    }
    const auto diagnostics = machine.diagnostics();
    assert(diagnostics.address_occupied_buckets > 0U);
    assert(diagnostics.address_full_buckets > 0U);
    assert(diagnostics.address_max_bucket_residents == 1U);
    assert(diagnostics.address_capacity_blocked_splits > 0U);
}

void verify_address_execution_frames() {
    auto config = sparse_config(8U, 4096U);
    config.context_width = 16U;
    config.address_execution_mode = sbm::AddressExecutionMode::InterpretedFrames;
    config.adaptive_topology = true;
    config.max_address_channels = 6U;
    config.beam_width = 6U;
    sbm::SparseBranchMachine machine(config);
    for (std::uint32_t step = 0U; step < 20000U; ++step) {
        (void)machine.step_token(
            step % config.token_alphabet,
            (step + 1U) % config.token_alphabet,
            true);
    }
    const auto diagnostics = machine.diagnostics();
    assert(diagnostics.address_execution_frames > 0U);
    assert(diagnostics.address_binding_hits + diagnostics.address_binding_misses ==
           diagnostics.address_execution_frames);
    assert(diagnostics.structural_description_cost > 0.0);
    assert(diagnostics.structural_execution_cost > 0.0);
    assert(diagnostics.max_bucket_candidates_inspected <=
           config.max_address_channels * config.bucket_scan_limit);
    assert(diagnostics.avg_active <= static_cast<double>(config.beam_width));
}

void verify_dependency_attribution_json() {
    auto dataset = sbm::generate_math_token_process(6U, 256U, 64U, 23U, 0.8F, 0.2F);
    auto config = sparse_config(8U, 64U);
    config.adaptive_topology = true;
    config.max_address_channels = 4U;
    config.beam_width = 4U;
    config.address_lags = {1U, 2U};
    config.topology_enable_delta = false;
    config.max_sparse_decisions_per_node = 128U;
    config.record_channel_attribution = true;
    const auto result = sbm::run_token_experiment(dataset, 768U, config, true, 0U, 0U, 0U);
    const auto json = sbm::to_json(result);
    assert(json.find("\"eval_program_attribution\"") != std::string::npos);
    assert(json.find("\"dependency\"") != std::string::npos);
    assert(json.find("\"parent_channel\"") != std::string::npos);
    assert(json.find("\"dependency_channel\"") != std::string::npos);
    assert(json.find("\"channel_generation\"") != std::string::npos);
    assert(json.find("\"parent_edge_kind\"") != std::string::npos);
    assert(json.find("\"dependency_edge_kind\"") != std::string::npos);
    assert(json.find("\"caller_removed_credit\"") != std::string::npos);
    assert(json.find("\"dependency_retained_credit\"") != std::string::npos);
    assert(json.find("\"dependency_removed_credit\"") != std::string::npos);
    assert(json.find("\"binding_matches\"") != std::string::npos);
    assert(json.find("\"unique_binding_keys\"") != std::string::npos);
    assert(json.find("\"binding_key_reuse_events\"") != std::string::npos);
    assert(json.find("\"call_matches\"") != std::string::npos);
    assert(json.find("\"unique_call_keys\"") != std::string::npos);
    assert(json.find("\"call_key_reuse_events\"") != std::string::npos);
    assert(json.find("\"call_match_fraction\"") != std::string::npos);
    assert(json.find("\"input_state\"") != std::string::npos);
    assert(json.find("\"output_state\"") != std::string::npos);
    assert(json.find("\"required_dependency_binding\"") != std::string::npos);
    assert(json.find("\"binding_reuse_bonus\"") != std::string::npos);
    assert(json.find("\"structural_value_without_reuse\"") != std::string::npos);
    assert(json.find("\"topology_restored\"") != std::string::npos);
    assert(json.find("\"decision_name\"") != std::string::npos);
    assert(json.find("\"mean_binding_distance\"") != std::string::npos);
    assert(json.find("\"mean_binding_pattern_span\"") != std::string::npos);
    assert(json.find("\"address_binding_by_kind\"") != std::string::npos);
    assert(json.find("\"learned_channel_effective_enabled\"") !=
           std::string::npos);
    assert(json.find("\"address_dependency_graph\"") != std::string::npos);
    assert(json.find("\"address_dependency_edges\"") != std::string::npos);
    assert(json.find("\"caller_channel\"") != std::string::npos);
    assert(json.find("\"caller_generation\"") != std::string::npos);
    assert(json.find("\"dependency_generation\"") != std::string::npos);
    assert(json.find("\"edge_kind\"") != std::string::npos);
    assert(json.find("\"direct_caller_count\"") != std::string::npos);
    assert(json.find("\"effective_direct_caller_count\"") !=
           std::string::npos);
    assert(json.find("\"blocked_direct_caller_count\"") !=
           std::string::npos);
    assert(json.find("\"dependency_available\"") != std::string::npos);
    assert(json.find("\"effective_enabled\"") != std::string::npos);
    assert(json.find("\"own_call_matches\"") != std::string::npos);
    assert(json.find("\"downstream_call_matches\"") != std::string::npos);
    assert(json.find("\"downstream_dependency_removed_credit\"") !=
           std::string::npos);
    assert(json.find("\"structural_value\"") != std::string::npos);
    assert(result.address_dependency_graph.size() ==
           result.learned_address_programs.size());
    assert(result.learned_channel_effective_enabled.size() ==
           result.learned_channel_phase.size());
    assert(!result.address_dependency_edges.empty());
    for (const auto& edge : result.address_dependency_edges) {
        assert(edge.caller_channel < result.learned_address_programs.size());
        assert(edge.dependency_channel < result.learned_address_programs.size());
        assert(edge.caller_channel != edge.dependency_channel);
        assert(edge.caller_generation != 0U);
    }
    for (const auto& summary : result.address_dependency_graph) {
        assert(summary.channel < result.learned_address_programs.size());
        assert(summary.effective_enabled <= 1U);
        assert(summary.dependency_available <= 1U);
        assert(summary.effective_direct_caller_count <=
               summary.direct_caller_count);
        assert(summary.blocked_direct_caller_count <=
               summary.direct_caller_count);
        assert(summary.effective_direct_caller_count +
               summary.blocked_direct_caller_count <=
               summary.direct_caller_count);
        assert(summary.unique_binding_keys <= summary.own_binding_matches);
        assert(summary.unique_call_keys <= summary.own_call_matches);
    }
    assert(result.diagnostics.structural_description_cost >= 0.0);
    assert(result.diagnostics.structural_execution_cost >= 0.0);
}

void verify_gpaf_shadow_observation_is_read_only() {
    auto baseline_config = sparse_config(4U, 16U);
    baseline_config.adaptive_topology = false;
    baseline_config.decode_token_ranking_during_training = true;
    auto shadow_config = baseline_config;
    shadow_config.gpaf_shadow_observation = true;
    shadow_config.gpaf_slots = 128U;

    sbm::SparseBranchMachine baseline(baseline_config);
    sbm::SparseBranchMachine shadow(shadow_config);
    for (std::uint32_t step = 0U; step < 256U; ++step) {
        const std::uint32_t token = step % 16U;
        const std::uint32_t target = (step * 5U + 3U) % 16U;
        const auto baseline_stats = baseline.step_token(token, target, true);
        const auto shadow_stats = shadow.step_token(token, target, true);
        assert(baseline_stats.predicted_token == shadow_stats.predicted_token);
        assert(baseline_stats.active_nodes == shadow_stats.active_nodes);
        assert(baseline_stats.candidates_examined ==
               shadow_stats.candidates_examined);
    }

    const auto baseline_diag = baseline.diagnostics();
    const auto shadow_diag = shadow.diagnostics();
    assert(baseline_diag.gpaf_role_observations == 0U);
    assert(baseline_diag.gpaf_slots_allocated == 0U);
    assert(shadow_diag.gpaf_role_observations > 0U);
    assert(shadow_diag.gpaf_unique_role_keys > 0U);
    assert(shadow_diag.gpaf_slots_allocated > 0U);
    assert(shadow_diag.gpaf_slots_probed == 0U);
    assert(shadow_diag.gpaf_candidates_returned == 0U);
    assert(shadow_diag.avg_active == baseline_diag.avg_active);
    assert(shadow_diag.avg_candidates == baseline_diag.avg_candidates);
}

void verify_gpaf_candidate_retrieval_is_bounded() {
    auto config = sparse_config(4U, 16U);
    config.adaptive_topology = false;
    config.gpaf_shadow_observation = true;
    config.gpaf_candidate_retrieval = true;
    config.gpaf_query_keys_per_step = 2U;
    config.gpaf_slots = 64U;
    config.gpaf_residents_per_slot = 3U;
    sbm::SparseBranchMachine machine(config);

    for (std::uint32_t step = 0U; step < 512U; ++step) {
        const std::uint32_t token = step % 16U;
        const std::uint32_t target = (step * 7U + 5U) % 16U;
        (void)machine.step_token(token, target, true);
    }

    const auto diagnostics = machine.diagnostics();
    assert(diagnostics.gpaf_role_observations > 0U);
    assert(diagnostics.gpaf_slots_probed > 0U);
    assert(diagnostics.gpaf_candidates_returned > 0U);
    assert(diagnostics.gpaf_slots_probed <=
           diagnostics.steps * config.gpaf_query_keys_per_step);
    assert(diagnostics.gpaf_candidates_returned <=
           diagnostics.gpaf_slots_probed * config.gpaf_residents_per_slot);
    assert(diagnostics.avg_active <= static_cast<double>(config.beam_width));
}

void verify_gpaf_checkpoint_resume_preserves_retrieval_state() {
    auto config = sparse_config(4U, 16U);
    config.adaptive_topology = false;
    config.gpaf_shadow_observation = true;
    config.gpaf_candidate_retrieval = true;
    config.gpaf_query_keys_per_step = 2U;
    config.gpaf_slots = 64U;
    config.gpaf_residents_per_slot = 3U;
    sbm::SparseBranchMachine machine(config);

    for (std::uint32_t step = 0U; step < 256U; ++step) {
        const std::uint32_t token = step % 16U;
        const std::uint32_t target = (step * 7U + 5U) % 16U;
        (void)machine.step_token(token, target, true);
    }
    const auto before = machine.diagnostics();
    assert(before.gpaf_slots_probed > 0U);
    assert(before.gpaf_candidates_returned > 0U);

    const std::string path = "gpaf_checkpoint_resume_test.sbm";
    sbm::save_checkpoint(machine, path);
    auto resumed = sbm::load_checkpoint(path);
    (void)std::remove(path.c_str());

    const auto loaded = resumed.diagnostics();
    assert(resumed.config().gpaf_shadow_observation);
    assert(resumed.config().gpaf_candidate_retrieval);
    assert(resumed.config().gpaf_query_keys_per_step == config.gpaf_query_keys_per_step);
    assert(resumed.config().gpaf_slots == config.gpaf_slots);
    assert(resumed.config().gpaf_residents_per_slot == config.gpaf_residents_per_slot);
    assert(loaded.gpaf_role_observations == before.gpaf_role_observations);
    assert(loaded.gpaf_unique_role_keys == before.gpaf_unique_role_keys);
    assert(loaded.gpaf_slots_allocated == before.gpaf_slots_allocated);
    assert(loaded.gpaf_shadow_updates == before.gpaf_shadow_updates);
    assert(loaded.gpaf_slots_probed == before.gpaf_slots_probed);
    assert(loaded.gpaf_candidates_returned == before.gpaf_candidates_returned);

    const auto next_machine = machine.step_token(3U, 10U, true);
    const auto next_resumed = resumed.step_token(3U, 10U, true);
    assert(next_machine.predicted_token == next_resumed.predicted_token);
    assert(next_machine.active_nodes == next_resumed.active_nodes);
    assert(next_machine.candidates_examined == next_resumed.candidates_examined);
    assert(next_resumed.candidates_examined > 0U);
    const auto after = resumed.diagnostics();
    assert(after.gpaf_slots_probed > loaded.gpaf_slots_probed);
    assert(after.gpaf_candidates_returned >= loaded.gpaf_candidates_returned);
}

void verify_gpaf_frozen_retrieval_is_read_only() {
    auto config = sparse_config(4U, 16U);
    config.adaptive_topology = false;
    config.gpaf_shadow_observation = true;
    config.gpaf_candidate_retrieval = true;
    config.gpaf_query_keys_per_step = 2U;
    config.gpaf_slots = 64U;
    config.gpaf_residents_per_slot = 3U;
    sbm::SparseBranchMachine machine(config);

    for (std::uint32_t step = 0U; step < 256U; ++step) {
        const std::uint32_t token = step % 16U;
        const std::uint32_t target = (step * 7U + 5U) % 16U;
        (void)machine.step_token(token, target, true);
    }
    const auto before = machine.diagnostics();
    assert(before.gpaf_role_observations > 0U);
    assert(before.gpaf_slots_probed > 0U);
    assert(before.gpaf_candidates_returned > 0U);

    const auto frozen = machine.step_token(3U, 10U, false);
    assert(frozen.active_nodes > 0U);
    const auto after = machine.diagnostics();
    assert(after.gpaf_role_observations == before.gpaf_role_observations);
    assert(after.gpaf_unique_role_keys == before.gpaf_unique_role_keys);
    assert(after.gpaf_slots_allocated == before.gpaf_slots_allocated);
    assert(after.gpaf_shadow_updates == before.gpaf_shadow_updates);
    assert(after.gpaf_slots_probed == before.gpaf_slots_probed);
    assert(after.gpaf_candidates_returned == before.gpaf_candidates_returned);
}

void verify_gpaf_probe_slot_phase_diagnostics_resume() {
    auto config = sparse_config(4U, 16U);
    config.adaptive_topology = false;
    config.gpaf_shadow_observation = true;
    config.gpaf_candidate_retrieval = true;
    config.gpaf_query_keys_per_step = 2U;
    config.gpaf_slots = 64U;
    config.gpaf_residents_per_slot = 3U;
    sbm::SparseBranchMachine machine(config);

    for (std::uint32_t step = 0U; step < 128U; ++step) {
        const std::uint32_t token = step % 16U;
        const std::uint32_t target = (step * 7U + 5U) % 16U;
        (void)machine.step_token(token, target, true);
    }
    const auto before = machine.diagnostics();
    assert(before.gpaf_unique_role_keys > 0U);
    assert(before.gpaf_probe_slots == before.gpaf_unique_role_keys);
    assert(before.gpaf_active_slots == 0U);
    assert(before.gpaf_quarantined_slots == 0U);
    assert(before.gpaf_recoverable_retired_slots == 0U);
    assert(before.gpaf_physically_erased_slots == 0U);

    const std::string path = "gpaf_probe_phase_resume_test.sbm";
    sbm::save_checkpoint(machine, path);
    auto resumed = sbm::load_checkpoint(path);
    (void)std::remove(path.c_str());
    const auto after = resumed.diagnostics();
    assert(after.gpaf_probe_slots == before.gpaf_probe_slots);
    assert(after.gpaf_active_slots == before.gpaf_active_slots);
    assert(after.gpaf_quarantined_slots == before.gpaf_quarantined_slots);
    assert(after.gpaf_recoverable_retired_slots ==
           before.gpaf_recoverable_retired_slots);
    assert(after.gpaf_physically_erased_slots == before.gpaf_physically_erased_slots);
}

} // namespace

int main(int argc, char** argv) {
    const std::string_view mode = argc > 1 ? argv[1] : "all";
    if (mode == "latency") {
        const auto p95_100k = address_p95_microseconds(100000U);
        const auto p95_1m = address_p95_microseconds(1000000U);
        std::cout << "address_p95_us_100k=" << p95_100k
                  << " address_p95_us_1m=" << p95_1m
                  << " ratio=" << p95_1m / p95_100k << '\n';
        return 0;
    }
    if (mode == "implicit") {
        for (const auto vocabulary : {2U, 3U, 4096U, 50257U, 250000U}) {
            verify_implicit_output(vocabulary);
        }
        verify_output_seed_decoupled();
        std::cout << "implicit output tests passed\n";
        return 0;
    }
    if (mode == "capacity") {
        verify_sparse_output_policy();
        verify_fresh_decision_learning_in_mature_node();
        verify_momentum_learning();
        verify_adaptive_beam_width();
        verify_refinement_rounds();
        verify_address_capacity_pressure();
        auto config = sparse_config(4U, 257U);
        config.max_sparse_decisions_per_node = 8U;
        config.max_specializations_per_bucket = 1U;
        config.split_min_visits = UINT32_MAX;
        sbm::SparseBranchMachine machine(config);
        for (std::uint32_t step = 0U; step < 4096U; ++step) {
            (void)machine.step_token(3U, step % 257U, true);
        }
        const auto diagnostics = machine.diagnostics();
        assert(diagnostics.max_sparse_entries_per_node <= 8U);
        assert(diagnostics.sparse_output_insertions > 8U);
        assert(diagnostics.sparse_output_evictions > 0U);
        assert(diagnostics.sparse_output_admission_rejections > 0U);
        assert(diagnostics.sparse_output_admission_promotions > 0U);
        assert(diagnostics.sparse_output_saturated_nodes > 0U);
        assert(diagnostics.sparse_output_evictions < diagnostics.steps);
        std::cout << "bounded decision capacity passed\n";
        return 0;
    }
    if (mode == "prior") {
        verify_global_output_prior();
        std::cout << "global output prior tests passed\n";
        return 0;
    }
    if (mode == "channels") {
        verify_conserved_channel_mass();
        verify_multi_channel_token_residual_learning();
        std::cout << "channel mass conservation passed\n";
        return 0;
    }
    if (mode == "address_frames") {
        verify_address_execution_frames();
        std::cout << "address execution frames passed\n";
        return 0;
    }
    if (mode == "dependency_attribution") {
        verify_dependency_attribution_json();
        std::cout << "dependency attribution passed\n";
        return 0;
    }
    if (mode == "gpaf_shadow") {
        verify_gpaf_shadow_observation_is_read_only();
        std::cout << "GPAF shadow observation passed\n";
        return 0;
    }
    if (mode == "gpaf_retrieval") {
        verify_gpaf_candidate_retrieval_is_bounded();
        std::cout << "GPAF candidate retrieval passed\n";
        return 0;
    }
    if (mode == "gpaf_checkpoint") {
        verify_gpaf_checkpoint_resume_preserves_retrieval_state();
        std::cout << "GPAF checkpoint resume passed\n";
        return 0;
    }
    if (mode == "gpaf_frozen") {
        verify_gpaf_frozen_retrieval_is_read_only();
        std::cout << "GPAF frozen retrieval read-only passed\n";
        return 0;
    }
    if (mode == "gpaf_lifecycle") {
        verify_gpaf_probe_slot_phase_diagnostics_resume();
        std::cout << "GPAF lifecycle diagnostics passed\n";
        return 0;
    }
    const auto address_10 = address_bytes(10U);
    const auto address_16 = address_bytes(16U);
    const auto address_20 = address_bytes(20U);
    const auto output_4k = output_bytes(4096U);
    const auto output_50k = output_bytes(50257U);
    const auto output_250k = output_bytes(250000U);
    std::cerr << "address_bytes=" << address_10 << ',' << address_16 << ','
              << address_20 << " output_bytes=" << output_4k << ',' << output_50k
              << ',' << output_250k << '\n';
    auto collision_config = sparse_config(1U, 128U);
    collision_config.bucket_scan_limit = 16U;
    sbm::SparseBranchMachine collision_machine(collision_config);
    sbm::SparseBranchMachine collision_control(collision_config);
    collision_machine.prefill_distractors(20000U);
    collision_control.prefill_distractors(20000U);
    collision_machine.reset_sequence();
    collision_control.reset_sequence();
    const auto collision_step = collision_machine.step_token(7U, 11U, true);
    const auto control_step = collision_control.step_token(7U, 11U, true);
    assert(collision_step.route == control_step.route);
    const auto collision = collision_machine.diagnostics();
    std::cerr << "max_bucket_scan="
              << collision.max_bucket_candidates_inspected << '\n';

    if (mode == "address" || mode == "all") {
        assert(address_16 <= address_10 * 105U / 100U + 4096U);
        assert(address_20 <= address_10 * 105U / 100U + 4096U);
        assert(collision.max_bucket_candidates_inspected <=
               collision_config.bucket_scan_limit);
        const auto live_before_prune = collision_machine.live_nodes();
        const auto pruned = collision_machine.prune(UINT32_MAX, 1.0F);
        assert(pruned == live_before_prune);
        assert(collision_machine.live_nodes() == 0U);
        collision_machine.prefill_distractors(1024U);
        (void)collision_machine.step_token(7U, 11U, false);
        assert(collision_machine.diagnostics().max_bucket_candidates_inspected <=
               collision_config.bucket_scan_limit);
    }
    if (mode == "output" || mode == "all") {
        constexpr std::uint64_t one_mebibyte = 1024ULL * 1024ULL;
        assert(output_50k <= output_4k + one_mebibyte);
        assert(output_250k <= output_4k + one_mebibyte);
    }

    std::cout << "scaling " << mode << " completed\n";
}
