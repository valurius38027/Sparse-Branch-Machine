#include "sbm/experiment.hpp"
#include "sbm/machine.hpp"
#include "sbm/detail/vector_ops.hpp"
#include "sbm/math.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace sbm {
ExperimentResult run_experiment(const VectorDataset& dataset,
                                      std::size_t warmup,
                                      Config config,
                                      bool strict,
                                      std::size_t prefill,
                                      std::size_t prune_interval,
                                      std::size_t merge_interval) {
    if (dataset.tokens.size() < 2 ||
        dataset.targets.size() != dataset.tokens.size() * dataset.vector_dim) {
        throw std::invalid_argument("invalid dataset");
    }
    if (warmup == 0 || warmup >= dataset.tokens.size()) {
        throw std::invalid_argument("warmup must leave non-empty train and eval partitions");
    }

    config.token_alphabet = dataset.token_alphabet;
    config.vector_dim = dataset.vector_dim;
    config.allow_growth_when_frozen = !strict;
    SparseBranchMachine model(config);
    model.prefill_distractors(prefill);

    const std::size_t dim = dataset.vector_dim;
    const std::size_t alphabet = dataset.token_alphabet;
    std::vector<double> global_sum(dim, 0.0);
    std::vector<double> token_sum(alphabet * dim, 0.0);
    std::vector<std::uint64_t> token_count(alphabet, 0);
    std::vector<double> pair_sum(alphabet * alphabet * dim, 0.0);
    std::vector<std::uint64_t> pair_count(alphabet * alphabet, 0);
    std::unordered_set<std::uint64_t> training_contexts;
    training_contexts.reserve(warmup * 2U);

    const auto context_key = [&](std::size_t index) noexcept {
        std::uint64_t key = mix64(dataset.tokens[index]);
        if (index > 0) key ^= std::rotl(mix64(dataset.tokens[index - 1U]), 21);
        if (index > 1) key ^= std::rotl(mix64(dataset.tokens[index - 2U]), 42);
        return mix64(key);
    };

    for (std::size_t t = 0; t < warmup; ++t) {
        const auto token = static_cast<std::size_t>(dataset.tokens[t]);
        const auto target = dataset.target(t);
        ++token_count[token];
        for (std::size_t j = 0; j < dim; ++j) {
            global_sum[j] += target[j];
            token_sum[token * dim + j] += target[j];
        }
        if (t > 0) {
            const auto previous = static_cast<std::size_t>(dataset.tokens[t - 1U]);
            const std::size_t pair = previous * alphabet + token;
            ++pair_count[pair];
            for (std::size_t j = 0; j < dim; ++j) pair_sum[pair * dim + j] += target[j];
        }
        training_contexts.insert(context_key(t));
    }

    std::vector<float> global_mean(dim, 0.0F);
    std::vector<float> token_mean(alphabet * dim, 0.0F);
    std::vector<float> pair_mean(alphabet * alphabet * dim, 0.0F);
    for (std::size_t j = 0; j < dim; ++j) {
        global_mean[j] = static_cast<float>(global_sum[j] / static_cast<double>(warmup));
    }
    for (std::size_t token = 0; token < alphabet; ++token) {
        for (std::size_t j = 0; j < dim; ++j) {
            token_mean[token * dim + j] = token_count[token] == 0
                ? global_mean[j]
                : static_cast<float>(token_sum[token * dim + j] /
                                     static_cast<double>(token_count[token]));
        }
    }
    for (std::size_t previous = 0; previous < alphabet; ++previous) {
        for (std::size_t token = 0; token < alphabet; ++token) {
            const std::size_t pair = previous * alphabet + token;
            for (std::size_t j = 0; j < dim; ++j) {
                pair_mean[pair * dim + j] = pair_count[pair] == 0
                    ? token_mean[token * dim + j]
                    : static_cast<float>(pair_sum[pair * dim + j] /
                                         static_cast<double>(pair_count[pair]));
            }
        }
    }

    // Strong fixed-table control matching the model's generic 1/2/4 temporal
    // views.  Each later table learns only the residual left by earlier views.
    std::vector<double> lag2_sum(alphabet * alphabet * dim, 0.0);
    std::vector<std::uint64_t> lag2_count(alphabet * alphabet, 0);
    for (std::size_t t = 2; t < warmup; ++t) {
        const std::size_t current = dataset.tokens[t];
        const std::size_t previous = dataset.tokens[t - 1U];
        const std::size_t lagged = dataset.tokens[t - 2U];
        const std::size_t base_key = previous * alphabet + current;
        const std::size_t residual_key = lagged * alphabet + current;
        ++lag2_count[residual_key];
        const auto target = dataset.target(t);
        for (std::size_t j = 0; j < dim; ++j) {
            lag2_sum[residual_key * dim + j] +=
                static_cast<double>(target[j] - pair_mean[base_key * dim + j]);
        }
    }
    std::vector<float> lag2_mean(alphabet * alphabet * dim, 0.0F);
    for (std::size_t key = 0; key < alphabet * alphabet; ++key) {
        if (lag2_count[key] == 0) continue;
        for (std::size_t j = 0; j < dim; ++j) {
            lag2_mean[key * dim + j] = static_cast<float>(
                lag2_sum[key * dim + j] / static_cast<double>(lag2_count[key]));
        }
    }

    std::vector<double> lag4_sum(alphabet * alphabet * dim, 0.0);
    std::vector<std::uint64_t> lag4_count(alphabet * alphabet, 0);
    for (std::size_t t = 4; t < warmup; ++t) {
        const std::size_t current = dataset.tokens[t];
        const std::size_t previous = dataset.tokens[t - 1U];
        const std::size_t lag2 = dataset.tokens[t - 2U];
        const std::size_t lag4 = dataset.tokens[t - 4U];
        const std::size_t base_key = previous * alphabet + current;
        const std::size_t lag2_key = lag2 * alphabet + current;
        const std::size_t lag4_key = lag4 * alphabet + current;
        ++lag4_count[lag4_key];
        const auto target = dataset.target(t);
        for (std::size_t j = 0; j < dim; ++j) {
            const float earlier = pair_mean[base_key * dim + j] +
                                  lag2_mean[lag2_key * dim + j];
            lag4_sum[lag4_key * dim + j] +=
                static_cast<double>(target[j] - earlier);
        }
    }
    std::vector<float> lag4_mean(alphabet * alphabet * dim, 0.0F);
    for (std::size_t key = 0; key < alphabet * alphabet; ++key) {
        if (lag4_count[key] == 0) continue;
        for (std::size_t j = 0; j < dim; ++j) {
            lag4_mean[key * dim + j] = static_cast<float>(
                lag4_sum[key * dim + j] / static_cast<double>(lag4_count[key]));
        }
    }

    struct Accumulator {
        double squared_error{};
        double target_energy{};
        double cosine_sum{};
        double centered_energy{};
        std::uint64_t vectors{};
    };
    auto add = [&](Accumulator& acc, std::span<const float> prediction,
                   std::span<const float> target) {
        acc.squared_error += detail::squared_distance(prediction.data(), target.data(), dim);
        acc.target_energy += detail::dot(target.data(), target.data(), dim);
        acc.cosine_sum += detail::cosine(prediction, target);
        for (std::size_t j = 0; j < dim; ++j) {
            const double centered = static_cast<double>(target[j]) - global_mean[j];
            acc.centered_energy += centered * centered;
        }
        ++acc.vectors;
    };
    const auto metrics = [](const Accumulator& acc) {
        VectorMetrics result;
        if (acc.vectors == 0) return result;
        result.normalized_rmse = std::sqrt(acc.squared_error /
                                           std::max(acc.target_energy, 1e-12));
        result.cosine = acc.cosine_sum / static_cast<double>(acc.vectors);
        result.r2 = 1.0 - acc.squared_error / std::max(acc.centered_energy, 1e-12);
        return result;
    };

    Accumulator train_acc;
    Accumulator eval_acc;
    Accumulator mean_acc;
    Accumulator token_acc;
    Accumulator pair_acc;
    Accumulator pair_lag2_acc;
    Accumulator multiscale_acc;
    Accumulator seen_acc;
    Accumulator unseen_acc;

    const auto start = std::chrono::steady_clock::now();
    std::vector<float> fixed_prediction(dim, 0.0F);
    for (std::size_t t = 0; t < dataset.tokens.size(); ++t) {
        const bool learn = t < warmup;
        if (strict && t == warmup) model.freeze_topology();
        const auto target = dataset.target(t);
        const auto stats = model.step(dataset.tokens[t], target, learn);
        (void)stats;
        const auto prediction = model.last_prediction();
        add(learn ? train_acc : eval_acc, prediction, target);

        if (!learn) {
            add(mean_acc, global_mean, target);
            const std::size_t token = dataset.tokens[t];
            add(token_acc,
                std::span<const float>(token_mean.data() + token * dim, dim), target);
            if (t > 0) {
                const std::size_t previous = dataset.tokens[t - 1U];
                const std::size_t pair = previous * alphabet + token;
                add(pair_acc,
                    std::span<const float>(pair_mean.data() + pair * dim, dim), target);
            } else {
                add(pair_acc, global_mean, target);
            }
            if (t >= 2) {
                const std::size_t current = dataset.tokens[t];
                const std::size_t previous = dataset.tokens[t - 1U];
                const std::size_t lagged = dataset.tokens[t - 2U];
                const std::size_t base_key = previous * alphabet + current;
                const std::size_t lag2_key = lagged * alphabet + current;
                for (std::size_t j = 0; j < dim; ++j) {
                    fixed_prediction[j] = pair_mean[base_key * dim + j] +
                                          lag2_mean[lag2_key * dim + j];
                }
                add(pair_lag2_acc, fixed_prediction, target);
                if (t >= 4) {
                    const std::size_t lag4 = dataset.tokens[t - 4U];
                    const std::size_t lag4_key = lag4 * alphabet + current;
                    for (std::size_t j = 0; j < dim; ++j) {
                        fixed_prediction[j] += lag4_mean[lag4_key * dim + j];
                    }
                    add(multiscale_acc, fixed_prediction, target);
                } else {
                    add(multiscale_acc, fixed_prediction, target);
                }
            } else {
                add(pair_lag2_acc, global_mean, target);
                add(multiscale_acc, global_mean, target);
            }
            if (training_contexts.contains(context_key(t))) add(seen_acc, prediction, target);
            else add(unseen_acc, prediction, target);
        }

        if (learn && prune_interval != 0 && (t + 1U) % prune_interval == 0) {
            (void)model.prune();
        }
        if (learn && merge_interval != 0 && (t + 1U) % merge_interval == 0) {
            (void)model.merge_redundant();
        }
    }
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();

    ExperimentResult result;
    result.diagnostics = model.diagnostics();
    result.train = metrics(train_acc);
    result.eval = metrics(eval_acc);
    result.mean_baseline_eval = metrics(mean_acc);
    result.token_baseline_eval = metrics(token_acc);
    result.pair_baseline_eval = metrics(pair_acc);
    result.pair_lag2_baseline_eval = metrics(pair_lag2_acc);
    result.multiscale_baseline_eval = metrics(multiscale_acc);
    result.seen_context_eval = metrics(seen_acc);
    result.unseen_context_eval = metrics(unseen_acc);
    result.seen_context_vectors = seen_acc.vectors;
    result.unseen_context_vectors = unseen_acc.vectors;
    result.steps_per_second = static_cast<double>(dataset.tokens.size()) / elapsed;
    result.elapsed_seconds = elapsed;
    result.strict_freeze = strict;
    result.dataset_hash = hash_dataset(dataset);
    result.vector_dim = dataset.vector_dim;
    result.token_alphabet = dataset.token_alphabet;
    result.address_lags = config.address_lags;
    result.learned_address_lags = model.learned_address_lags();
    result.learned_address_programs = model.learned_address_programs();
    result.learned_channel_credit = model.learned_channel_credit();
    result.learned_channel_phase = model.learned_channel_phase();
    result.learned_channel_effective_enabled =
        model.learned_channel_effective_enabled();
    result.learned_channel_parent = model.learned_channel_parent();
    result.learned_channel_dependency = model.learned_channel_dependency();
    result.learned_channel_generation = model.learned_channel_generation();
    result.learned_channel_parent_edge_kind =
        model.learned_channel_parent_edge_kind();
    result.learned_channel_dependency_edge_kind =
        model.learned_channel_dependency_edge_kind();
    result.topology_events = model.topology_events();
    result.exact_region_mass = config.exact_region_mass;
    result.residual_channel_gain = config.residual_channel_gain;
    result.residual_recency_pseudocount = config.residual_recency_pseudocount;
    result.edge_score_weight = config.edge_score_weight;
    return result;
}

ExperimentResult run_experiment(const VectorDataset& dataset,
                                std::size_t warmup,
                                std::uint64_t seed,
                                bool strict,
                                std::size_t prefill,
                                std::size_t prune_interval,
                                std::size_t merge_interval) {
    Config config;
    config.seed = seed;
    return run_experiment(dataset, warmup, config, strict, prefill,
                          prune_interval, merge_interval);
}
std::string to_json(const ExperimentResult& result) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(8);
    const auto emit_metrics = [&](const char* prefix, const VectorMetrics& metrics) {
        out << "  \"" << prefix << "_nrmse\": " << metrics.normalized_rmse << ",\n"
            << "  \"" << prefix << "_cosine\": " << metrics.cosine << ",\n"
            << "  \"" << prefix << "_r2\": " << metrics.r2 << ",\n";
    };
    out << "{\n"
        << "  \"task\": \"multiscale_token_vector_prediction\",\n"
        << "  \"steps\": " << result.diagnostics.steps << ",\n"
        << "  \"vector_dim\": " << result.vector_dim << ",\n"
        << "  \"token_alphabet\": " << result.token_alphabet << ",\n"
        << "  \"address_lags\": [";
    for (std::size_t i = 0; i < result.address_lags.size(); ++i) {
        if (i != 0U) out << ", ";
        out << result.address_lags[i];
    }
    out << "],\n  \"learned_address_lags\": [";
    for (std::size_t i = 0; i < result.learned_address_lags.size(); ++i) {
        if (i != 0U) out << ", ";
        out << result.learned_address_lags[i];
    }
    out << "],\n  \"learned_address_programs\": [";
    for (std::size_t i = 0; i < result.learned_address_programs.size(); ++i) {
        if (i != 0U) out << ", ";
        out << '[';
        const auto& program = result.learned_address_programs[i];
        for (std::size_t j = 0; j < program.arity; ++j) {
            if (j != 0U) out << ", ";
            out << program.lags[j];
        }
        out << ']';
    }
    out << "],\n  \"learned_address_operations\": [";
    for (std::size_t i = 0; i < result.learned_address_programs.size(); ++i) {
        if (i != 0U) out << ", ";
        out << static_cast<unsigned>(result.learned_address_programs[i].op);
    }
    out << "],\n  \"learned_channel_credit\": [";
    for (std::size_t i = 0; i < result.learned_channel_credit.size(); ++i) {
        if (i != 0U) out << ", ";
        out << result.learned_channel_credit[i];
    }
    out << "],\n  \"learned_channel_phase\": [";
    for (std::size_t i = 0; i < result.learned_channel_phase.size(); ++i) {
        if (i != 0U) out << ", ";
        out << static_cast<unsigned>(result.learned_channel_phase[i]);
    }
    out << "],\n  \"learned_channel_effective_enabled\": [";
    for (std::size_t i = 0; i < result.learned_channel_effective_enabled.size(); ++i) {
        if (i != 0U) out << ", ";
        out << static_cast<unsigned>(result.learned_channel_effective_enabled[i]);
    }
    out << "],\n  \"learned_channel_parent\": [";
    for (std::size_t i = 0; i < result.learned_channel_parent.size(); ++i) {
        if (i != 0U) out << ", ";
        out << static_cast<unsigned>(result.learned_channel_parent[i]);
    }
    out << "],\n  \"learned_channel_dependency\": [";
    for (std::size_t i = 0; i < result.learned_channel_dependency.size(); ++i) {
        if (i != 0U) out << ", ";
        out << static_cast<unsigned>(result.learned_channel_dependency[i]);
    }
    out << "],\n  \"learned_channel_generation\": [";
    for (std::size_t i = 0; i < result.learned_channel_generation.size(); ++i) {
        if (i != 0U) out << ", ";
        out << result.learned_channel_generation[i];
    }
    out << "],\n  \"learned_channel_parent_edge_kind\": [";
    for (std::size_t i = 0; i < result.learned_channel_parent_edge_kind.size(); ++i) {
        if (i != 0U) out << ", ";
        out << static_cast<unsigned>(result.learned_channel_parent_edge_kind[i]);
    }
    out << "],\n  \"learned_channel_dependency_edge_kind\": [";
    for (std::size_t i = 0; i < result.learned_channel_dependency_edge_kind.size(); ++i) {
        if (i != 0U) out << ", ";
        out << static_cast<unsigned>(result.learned_channel_dependency_edge_kind[i]);
    }
    out << "],\n  \"topology_decision_name_map\": [";
    for (unsigned decision = 0U;
         decision <= static_cast<unsigned>(TopologyDecision::Restored);
         ++decision) {
        if (decision != 0U) out << ", ";
        const auto value = static_cast<TopologyDecision>(decision);
        out << "{\"decision\":" << decision << ",\"decision_name\":\""
            << topology_decision_name(value) << "\"}";
    }
    out << "],\n  \"topology_events\": [";
    for (std::size_t i = 0; i < result.topology_events.size(); ++i) {
        if (i != 0U) out << ", ";
        const auto& event = result.topology_events[i];
        out << "{\"step\":" << event.step
            << ",\"channel_generation\":" << event.channel_generation
            << ",\"lags\":[";
        for (std::size_t j = 0; j < event.program.arity; ++j) {
            if (j != 0U) out << ',';
            out << event.program.lags[j];
        }
        out << "],\"op\":" << static_cast<unsigned>(event.program.op)
            << ",\"decision\":" << static_cast<unsigned>(event.decision)
            << ",\"decision_name\":\""
            << topology_decision_name(event.decision) << "\""
            << ",\"credit\":" << event.credit
            << ",\"channel\":" << static_cast<unsigned>(event.channel)
            << ",\"parent_channel\":"
            << static_cast<unsigned>(event.parent_channel)
            << ",\"dependency_channel\":"
            << static_cast<unsigned>(event.dependency_channel)
            << ",\"parent_edge_kind\":"
            << static_cast<unsigned>(event.parent_edge_kind)
            << ",\"dependency_edge_kind\":"
            << static_cast<unsigned>(event.dependency_edge_kind) << "}";
    }
    out << "],\n"
        << "  \"exact_region_mass\": " << result.exact_region_mass << ",\n"
        << "  \"residual_channel_gain\": " << result.residual_channel_gain << ",\n"
        << "  \"residual_recency_pseudocount\": "
        << result.residual_recency_pseudocount << ",\n"
        << "  \"edge_score_weight\": " << result.edge_score_weight << ",\n"
        << "  \"live_nodes\": " << result.diagnostics.live_nodes << ",\n"
        << "  \"edges\": " << result.diagnostics.edges << ",\n"
        << "  \"avg_active\": " << result.diagnostics.avg_active << ",\n"
        << "  \"avg_candidates\": " << result.diagnostics.avg_candidates << ",\n"
        << "  \"candidate_source_exact_bucket\": "
        << result.diagnostics.candidate_source_exact_bucket << ",\n"
        << "  \"candidate_source_control_edge\": "
        << result.diagnostics.candidate_source_control_edge << ",\n"
        << "  \"candidate_source_neighbor_bucket\": "
        << result.diagnostics.candidate_source_neighbor_bucket << ",\n"
        << "  \"route_score_hamming_sum\": "
        << result.diagnostics.route_score_hamming_sum << ",\n"
        << "  \"route_score_exact_sum\": "
        << result.diagnostics.route_score_exact_sum << ",\n"
        << "  \"route_score_edge_prior_sum\": "
        << result.diagnostics.route_score_edge_prior_sum << ",\n"
        << "  \"gpaf_role_observations\": "
        << result.diagnostics.gpaf_role_observations << ",\n"
        << "  \"gpaf_unique_role_keys\": "
        << result.diagnostics.gpaf_unique_role_keys << ",\n"
        << "  \"gpaf_slots_allocated\": "
        << result.diagnostics.gpaf_slots_allocated << ",\n"
        << "  \"gpaf_probe_slots\": "
        << result.diagnostics.gpaf_probe_slots << ",\n"
        << "  \"gpaf_active_slots\": "
        << result.diagnostics.gpaf_active_slots << ",\n"
        << "  \"gpaf_quarantined_slots\": "
        << result.diagnostics.gpaf_quarantined_slots << ",\n"
        << "  \"gpaf_recoverable_retired_slots\": "
        << result.diagnostics.gpaf_recoverable_retired_slots << ",\n"
        << "  \"gpaf_physically_erased_slots\": "
        << result.diagnostics.gpaf_physically_erased_slots << ",\n"
        << "  \"gpaf_shadow_updates\": "
        << result.diagnostics.gpaf_shadow_updates << ",\n"
        << "  \"gpaf_slots_probed\": "
        << result.diagnostics.gpaf_slots_probed << ",\n"
        << "  \"gpaf_candidates_returned\": "
        << result.diagnostics.gpaf_candidates_returned << ",\n"
        << "  \"created_total\": " << result.diagnostics.created_total << ",\n"
        << "  \"merged_total\": " << result.diagnostics.merged_total << ",\n"
        << "  \"pruned_total\": " << result.diagnostics.pruned_total << ",\n"
        << "  \"anchor_nodes\": " << result.diagnostics.anchor_nodes << ",\n"
        << "  \"residual_nodes\": " << result.diagnostics.residual_nodes << ",\n"
        << "  \"estimated_bytes\": " << result.diagnostics.estimated_bytes << ",\n"
        << "  \"address_index_bytes\": " << result.diagnostics.address_index_bytes << ",\n"
        << "  \"address_occupied_buckets\": " << result.diagnostics.address_occupied_buckets << ",\n"
        << "  \"address_full_buckets\": " << result.diagnostics.address_full_buckets << ",\n"
        << "  \"address_max_bucket_residents\": " << result.diagnostics.address_max_bucket_residents << ",\n"
        << "  \"address_capacity_blocked_splits\": " << result.diagnostics.address_capacity_blocked_splits << ",\n"
        << "  \"output_structure_bytes\": "
        << result.diagnostics.output_structure_bytes << ",\n"
        << "  \"max_bucket_candidates_inspected\": "
        << result.diagnostics.max_bucket_candidates_inspected << ",\n"
        << "  \"max_sparse_entries_per_node\": "
        << result.diagnostics.max_sparse_entries_per_node << ",\n"
        << "  \"global_output_prior_bytes\": "
        << result.diagnostics.global_output_prior_bytes << ",\n"
        << "  \"global_output_prior_updates\": "
        << result.diagnostics.global_output_prior_updates << ",\n"
        << "  \"sparse_output_insertions\": "
        << result.diagnostics.sparse_output_insertions << ",\n"
        << "  \"sparse_output_evictions\": "
        << result.diagnostics.sparse_output_evictions << ",\n"
        << "  \"sparse_output_admission_rejections\": "
        << result.diagnostics.sparse_output_admission_rejections << ",\n"
        << "  \"sparse_output_admission_promotions\": "
        << result.diagnostics.sparse_output_admission_promotions << ",\n"
        << "  \"sparse_output_probable_reconstructions\": "
        << result.diagnostics.sparse_output_probable_reconstructions << ",\n"
        << "  \"sparse_output_saturated_nodes\": "
        << result.diagnostics.sparse_output_saturated_nodes << ",\n"
        << "  \"max_sparse_decision_visits\": "
        << result.diagnostics.max_sparse_decision_visits << ",\n"
        << "  \"max_responsibility_mass_error\": "
        << result.diagnostics.max_responsibility_mass_error << ",\n"
        << "  \"topology_proposals\": " << result.diagnostics.topology_proposals << ",\n"
        << "  \"topology_accepted\": " << result.diagnostics.topology_accepted << ",\n"
        << "  \"topology_rejected\": " << result.diagnostics.topology_rejected << ",\n"
        << "  \"topology_pruned\": " << result.diagnostics.topology_pruned << ",\n"
        << "  \"topology_restored\": " << result.diagnostics.topology_restored << ",\n"
        << "  \"active_channels\": " << result.diagnostics.active_channels << ",\n"
        << "  \"dependency_blocked_channels\": "
        << result.diagnostics.dependency_blocked_channels << ",\n"
        << "  \"probe_channels\": " << result.diagnostics.probe_channels << ",\n"
        << "  \"simd_enabled\": " << (result.diagnostics.simd_enabled ? "true" : "false") << ",\n";
    emit_metrics("train", result.train);
    emit_metrics("eval", result.eval);
    emit_metrics("mean_baseline_eval", result.mean_baseline_eval);
    emit_metrics("token_baseline_eval", result.token_baseline_eval);
    emit_metrics("pair_baseline_eval", result.pair_baseline_eval);
    emit_metrics("pair_lag2_baseline_eval", result.pair_lag2_baseline_eval);
    emit_metrics("multiscale_baseline_eval", result.multiscale_baseline_eval);
    emit_metrics("seen_context_eval", result.seen_context_eval);
    emit_metrics("unseen_context_eval", result.unseen_context_eval);
    out << "  \"seen_context_vectors\": " << result.seen_context_vectors << ",\n"
        << "  \"unseen_context_vectors\": " << result.unseen_context_vectors << ",\n"
        << "  \"steps_per_second\": " << result.steps_per_second << ",\n"
        << "  \"elapsed_seconds\": " << result.elapsed_seconds << ",\n"
        << "  \"strict_freeze\": " << (result.strict_freeze ? "true" : "false") << ",\n"
        << "  \"dataset_hash\": " << result.dataset_hash << "\n"
        << "}\n";
    return out.str();
}

} // namespace sbm
