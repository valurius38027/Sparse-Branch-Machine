#include "sbm/experiment.hpp"

#include "sbm/machine.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sbm {
namespace {

struct TokenAccumulator {
    double cross_entropy{};
    double target_probability{};
    std::uint64_t top1{};
    std::uint64_t top5{};
    std::uint64_t examples{};
    bool ranking_available{true};
};

struct ChannelAttributionAccumulator {
    double credit_sum{};
    std::uint64_t observations{};
    std::uint64_t positive{};
    std::uint64_t documents{};
    std::uint64_t positive_documents{};
};

struct ProgramAttributionAccumulator {
    std::uint8_t channel{};
    std::uint8_t parent_channel{kInvalidChannel};
    std::uint8_t dependency_channel{kInvalidChannel};
    std::uint64_t channel_generation{};
    AddressGraphEdgeKind parent_edge_kind{AddressGraphEdgeKind::None};
    AddressGraphEdgeKind dependency_edge_kind{AddressGraphEdgeKind::None};
    AddressStateKind input_state{AddressStateKind::None};
    AddressStateKind output_state{AddressStateKind::None};
    AddressBindingKind required_dependency_binding{AddressBindingKind::None};
    std::uint32_t dependency{};
    double credit_sum{};
    double caller_removed_credit_sum{};
    double dependency_retained_credit_sum{};
    double dependency_removed_credit_sum{};
    double binding_distance_sum{};
    double binding_pattern_span_sum{};
    std::uint64_t binding_matches{};
    std::unordered_set<std::uint64_t> binding_keys;
    std::uint64_t call_matches{};
    std::unordered_set<std::uint64_t> call_keys;
    double description_cost{};
    double execution_cost{};
    std::uint64_t observations{};
    std::uint64_t positive{};
};

void add_step(TokenAccumulator& accumulator, const StepStats& stats) {
    accumulator.cross_entropy += static_cast<double>(stats.cross_entropy);
    accumulator.target_probability += static_cast<double>(stats.target_probability);
    if (stats.ranking_available) {
        accumulator.top1 += stats.top1_correct ? 1U : 0U;
        accumulator.top5 += stats.top5_correct ? 1U : 0U;
    } else {
        accumulator.ranking_available = false;
    }
    ++accumulator.examples;
}

void add_target_probability(TokenAccumulator& accumulator, double probability) {
    const double bounded = std::max(probability, 1e-12);
    accumulator.cross_entropy -= std::log(bounded);
    accumulator.target_probability += bounded;
    ++accumulator.examples;
    accumulator.ranking_available = false;
}

void add_cross_entropy(TokenAccumulator& accumulator, double cross_entropy) {
    accumulator.cross_entropy += cross_entropy;
    accumulator.target_probability += std::exp(-std::min(cross_entropy, 80.0));
    ++accumulator.examples;
    accumulator.ranking_available = false;
}

TokenMetrics finish(const TokenAccumulator& accumulator) {
    TokenMetrics result;
    if (accumulator.examples == 0U) return result;
    const double inverse = 1.0 / static_cast<double>(accumulator.examples);
    result.cross_entropy = accumulator.cross_entropy * inverse;
    result.bits_per_token = result.cross_entropy / std::log(2.0);
    result.perplexity = std::exp(std::min(result.cross_entropy, 80.0));
    result.top1_accuracy = static_cast<double>(accumulator.top1) * inverse;
    result.top5_accuracy = static_cast<double>(accumulator.top5) * inverse;
    result.mean_target_probability = accumulator.target_probability * inverse;
    result.ranking_available = accumulator.ranking_available;
    return result;
}

std::vector<ChannelAttribution> finish_channel_attribution(
    std::span<const ChannelAttributionAccumulator> accumulators,
    std::span<const AddressProgram> programs,
    std::span<const std::uint8_t> phases) {
    std::vector<ChannelAttribution> result;
    const auto count = std::min({accumulators.size(), programs.size(), phases.size()});
    result.reserve(count);
    for (std::size_t channel = 0U; channel < count; ++channel) {
        const auto& accumulator = accumulators[channel];
        ChannelAttribution attribution;
        attribution.channel = static_cast<std::uint8_t>(channel);
        attribution.program = programs[channel];
        attribution.phase = phases[channel];
        attribution.eval_observations = accumulator.observations;
        attribution.eval_positive = accumulator.positive;
        attribution.eval_documents = accumulator.documents;
        attribution.eval_positive_documents = accumulator.positive_documents;
        attribution.eval_credit_sum = accumulator.credit_sum;
        if (accumulator.observations != 0U) {
            attribution.eval_mean_credit = accumulator.credit_sum /
                static_cast<double>(accumulator.observations);
            attribution.eval_positive_fraction =
                static_cast<double>(accumulator.positive) /
                static_cast<double>(accumulator.observations);
        }
        if (accumulator.documents != 0U) {
            attribution.eval_positive_document_fraction =
                static_cast<double>(accumulator.positive_documents) /
                static_cast<double>(accumulator.documents);
        }
        result.push_back(attribution);
    }
    return result;
}

[[nodiscard]] std::uint64_t program_attribution_key(std::uint8_t channel,
                                                    std::uint32_t dependency) noexcept {
    return (static_cast<std::uint64_t>(channel) << 32U) |
           static_cast<std::uint64_t>(dependency);
}

std::vector<ProgramAttribution> finish_program_attribution(
    const std::unordered_map<std::uint64_t, ProgramAttributionAccumulator>& accumulators,
    std::span<const AddressProgram> programs,
    std::span<const std::uint64_t> generations,
    std::span<const std::uint8_t> parent_edge_kinds,
    std::span<const std::uint8_t> dependency_edge_kinds) {
    std::vector<ProgramAttribution> result;
    result.reserve(accumulators.size());
    for (const auto& [key, accumulator] : accumulators) {
        (void)key;
        if (accumulator.channel >= programs.size()) continue;
        ProgramAttribution attribution;
        attribution.channel = accumulator.channel;
        attribution.parent_channel = accumulator.parent_channel;
        attribution.dependency_channel = accumulator.dependency_channel;
        if (accumulator.channel < generations.size()) {
            attribution.channel_generation = generations[accumulator.channel];
        }
        if (accumulator.channel < parent_edge_kinds.size()) {
            attribution.parent_edge_kind =
                static_cast<AddressGraphEdgeKind>(
                    parent_edge_kinds[accumulator.channel]);
        }
        if (accumulator.channel < dependency_edge_kinds.size()) {
            attribution.dependency_edge_kind =
                static_cast<AddressGraphEdgeKind>(
                    dependency_edge_kinds[accumulator.channel]);
        }
        attribution.input_state = accumulator.input_state;
        attribution.output_state = accumulator.output_state;
        attribution.required_dependency_binding =
            accumulator.required_dependency_binding;
        attribution.program = programs[accumulator.channel];
        attribution.dependency = accumulator.dependency;
        attribution.credit_sum = accumulator.credit_sum;
        attribution.caller_removed_credit_sum =
            accumulator.caller_removed_credit_sum;
        attribution.dependency_retained_credit_sum =
            accumulator.dependency_retained_credit_sum;
        attribution.dependency_removed_credit_sum =
            accumulator.dependency_removed_credit_sum;
        attribution.binding_distance_sum = accumulator.binding_distance_sum;
        attribution.binding_pattern_span_sum =
            accumulator.binding_pattern_span_sum;
        attribution.binding_matches = accumulator.binding_matches;
        attribution.unique_binding_keys = accumulator.binding_keys.size();
        attribution.binding_key_reuse_events =
            attribution.binding_matches > attribution.unique_binding_keys
                ? attribution.binding_matches - attribution.unique_binding_keys
                : 0U;
        attribution.call_matches = accumulator.call_matches;
        attribution.unique_call_keys = accumulator.call_keys.size();
        attribution.call_key_reuse_events =
            attribution.call_matches > attribution.unique_call_keys
                ? attribution.call_matches - attribution.unique_call_keys
                : 0U;
        attribution.description_cost = accumulator.description_cost;
        attribution.execution_cost = accumulator.execution_cost;
        attribution.observations = accumulator.observations;
        attribution.positive = accumulator.positive;
        result.push_back(attribution);
    }
    std::sort(result.begin(), result.end(),
              [](const ProgramAttribution& left,
                 const ProgramAttribution& right) {
                  if (left.channel != right.channel) {
                      return left.channel < right.channel;
                  }
                  return left.dependency < right.dependency;
              });
    return result;
}

std::vector<ChannelDependencySummary> finish_dependency_graph(
    std::span<const AddressProgram> programs,
    std::span<const std::uint8_t> phases,
    std::span<const std::uint8_t> effective_enabled,
    std::span<const std::uint8_t> parents,
    std::span<const std::uint8_t> dependencies,
    std::span<const std::uint64_t> generations,
    std::span<const std::uint8_t> parent_edge_kinds,
    std::span<const std::uint8_t> dependency_edge_kinds,
    std::span<const ProgramAttribution> attributions) {
    const auto count = std::min({programs.size(), phases.size(),
                                 effective_enabled.size(), parents.size(),
                                 dependencies.size(), generations.size(),
                                 parent_edge_kinds.size(),
                                 dependency_edge_kinds.size()});
    std::vector<ChannelDependencySummary> result;
    result.reserve(count);
    for (std::size_t channel = 0U; channel < count; ++channel) {
        ChannelDependencySummary summary;
        summary.channel = static_cast<std::uint8_t>(channel);
        summary.program = programs[channel];
        summary.phase = phases[channel];
        summary.parent_channel = parents[channel];
        summary.dependency_channel = dependencies[channel];
        summary.channel_generation = generations[channel];
        summary.parent_edge_kind =
            static_cast<AddressGraphEdgeKind>(parent_edge_kinds[channel]);
        summary.dependency_edge_kind =
            static_cast<AddressGraphEdgeKind>(dependency_edge_kinds[channel]);
        summary.input_state = address_program_input_state(programs[channel]);
        summary.output_state = address_program_output_state(programs[channel]);
        summary.required_dependency_binding =
            address_program_required_dependency_binding(programs[channel]);
        summary.effective_enabled = effective_enabled[channel] != 0U ? 1U : 0U;
        summary.dependency_available =
            dependencies[channel] != kInvalidChannel &&
            dependencies[channel] < effective_enabled.size()
                ? (effective_enabled[dependencies[channel]] != 0U ? 1U : 0U)
                : 1U;
        for (std::size_t caller = 0U; caller < count; ++caller) {
            if (caller != channel && dependencies[caller] == channel) {
                ++summary.direct_caller_count;
                if (effective_enabled[caller] != 0U) {
                    ++summary.effective_direct_caller_count;
                } else if (phases[caller] ==
                    static_cast<std::uint8_t>(ChannelPhase::Seed) ||
                    phases[caller] ==
                    static_cast<std::uint8_t>(ChannelPhase::Probe) ||
                    phases[caller] ==
                    static_cast<std::uint8_t>(ChannelPhase::Active)) {
                    ++summary.blocked_direct_caller_count;
                }
            }
        }
        for (const auto& attribution : attributions) {
            if (attribution.channel == channel) {
                summary.own_observations += attribution.observations;
                summary.own_binding_matches += attribution.binding_matches;
                summary.unique_binding_keys += attribution.unique_binding_keys;
                summary.binding_key_reuse_events +=
                    attribution.binding_key_reuse_events;
                summary.own_call_matches += attribution.call_matches;
                summary.unique_call_keys += attribution.unique_call_keys;
                summary.call_key_reuse_events +=
                    attribution.call_key_reuse_events;
                summary.own_credit_sum += attribution.credit_sum;
                summary.own_caller_removed_credit_sum +=
                    attribution.caller_removed_credit_sum;
            }
            if (attribution.dependency_channel == channel &&
                attribution.channel != channel) {
                summary.downstream_call_matches += attribution.call_matches;
                summary.downstream_caller_removed_credit_sum +=
                    attribution.caller_removed_credit_sum;
                summary.downstream_dependency_removed_credit_sum +=
                    attribution.dependency_removed_credit_sum;
            }
        }
        result.push_back(summary);
    }
    return result;
}

std::vector<ChannelDependencyEdge> finish_dependency_edges(
    std::span<const AddressProgram> programs,
    std::span<const std::uint8_t> parents,
    std::span<const std::uint8_t> dependencies,
    std::span<const std::uint64_t> generations,
    std::span<const std::uint8_t> parent_edge_kinds,
    std::span<const std::uint8_t> dependency_edge_kinds,
    std::span<const ProgramAttribution> attributions) {
    const auto count = std::min({programs.size(), parents.size(),
                                 dependencies.size(), generations.size(),
                                 parent_edge_kinds.size(),
                                 dependency_edge_kinds.size()});
    std::vector<ChannelDependencyEdge> result;
    result.reserve(count * 2U);
    const auto append_edge =
        [&](std::size_t caller, std::uint8_t target, AddressGraphEdgeKind kind) {
        if (target == kInvalidChannel || target >= count || target == caller ||
            kind == AddressGraphEdgeKind::None) {
            return;
        }
        ChannelDependencyEdge edge;
        edge.caller_channel = static_cast<std::uint8_t>(caller);
        edge.dependency_channel = target;
        edge.caller_generation = generations[caller];
        edge.dependency_generation = generations[target];
        edge.edge_kind = kind;
        edge.caller_input_state = address_program_input_state(programs[caller]);
        edge.dependency_output_state =
            address_program_output_state(programs[target]);
        edge.required_dependency_binding =
            address_program_required_dependency_binding(programs[caller]);
        for (const auto& attribution : attributions) {
            if (attribution.channel != caller ||
                attribution.dependency_channel != target) {
                continue;
            }
            edge.observations += attribution.observations;
            edge.call_matches += attribution.call_matches;
            edge.unique_call_keys += attribution.unique_call_keys;
            edge.call_key_reuse_events += attribution.call_key_reuse_events;
            edge.caller_removed_credit_sum +=
                attribution.caller_removed_credit_sum;
            edge.dependency_removed_credit_sum +=
                attribution.dependency_removed_credit_sum;
        }
        result.push_back(edge);
    };
    for (std::size_t caller = 0U; caller < count; ++caller) {
        const auto parent = parents[caller];
        const auto dependency = dependencies[caller];
        if (parent == dependency) {
            append_edge(caller, dependency,
                        static_cast<AddressGraphEdgeKind>(
                            dependency_edge_kinds[caller]));
            continue;
        }
        append_edge(caller, parent,
                    static_cast<AddressGraphEdgeKind>(
                        parent_edge_kinds[caller]));
        append_edge(caller, dependency,
                    static_cast<AddressGraphEdgeKind>(
                        dependency_edge_kinds[caller]));
    }
    return result;
}

struct CountRow {
    std::uint64_t total{};
    std::unordered_map<std::uint32_t, std::uint32_t> counts;
};

class ConditionalTable {
public:
    void observe(std::uint64_t key, std::uint32_t target) {
        auto [iterator, inserted] = rows_.try_emplace(key);
        (void)inserted;
        ++iterator->second.total;
        ++iterator->second.counts[target];
    }

    [[nodiscard]] double target_probability(std::uint64_t key,
                                            std::uint32_t target,
                                            double fallback,
                                            double smoothing = 0.5) const {
        const auto iterator = rows_.find(key);
        if (iterator == rows_.end()) return fallback;
        const auto& row = iterator->second;
        const double denominator = static_cast<double>(row.total) + smoothing;
        const auto count = row.counts.find(target);
        const auto observed = count == row.counts.end() ? 0U : count->second;
        return (static_cast<double>(observed) + smoothing * fallback) / denominator;
    }

private:
    std::unordered_map<std::uint64_t, CountRow> rows_;
};

std::uint64_t pair_key(std::uint32_t previous, std::uint32_t current) {
    return (static_cast<std::uint64_t>(previous) << 32U) | current;
}

struct TokenDataView {
    std::uint32_t vocab_size{};
    std::span<const std::uint32_t> tokens;
    std::span<const std::uint64_t> sequence_offsets;
    std::span<const float> oracle_nll;
    std::uint64_t dataset_hash{};
    const char* task{};

    [[nodiscard]] std::size_t sequence_count() const noexcept {
        return sequence_offsets.empty() ? 0U : sequence_offsets.size() - 1U;
    }
    [[nodiscard]] std::size_t example_count() const noexcept {
        return tokens.size() - sequence_count();
    }
};

std::uint32_t token_at_lag(const TokenDataView& dataset,
                           std::size_t sequence_start,
                           std::size_t position,
                           std::size_t lag) {
    return position >= sequence_start + lag
        ? dataset.tokens[position - lag]
        : dataset.tokens[sequence_start];
}

void emit_progress(std::size_t processed,
                   std::size_t total,
                   std::size_t warmup,
                   const TokenAccumulator& train,
                   const TokenAccumulator& eval,
                   const SparseBranchMachine& model,
                   std::chrono::steady_clock::time_point started) {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::max(
        std::chrono::duration<double>(now - started).count(), 1e-9);
    const double speed = static_cast<double>(processed) / elapsed;
    const double eta = speed <= 0.0
        ? std::numeric_limits<double>::infinity()
        : static_cast<double>(total - processed) / speed;
    const auto diagnostics = model.diagnostics();
    const auto train_nll = train.examples == 0U
        ? std::numeric_limits<double>::quiet_NaN()
        : train.cross_entropy / static_cast<double>(train.examples);
    const auto eval_nll = eval.examples == 0U
        ? std::numeric_limits<double>::quiet_NaN()
        : eval.cross_entropy / static_cast<double>(eval.examples);
    std::cerr << '\r' << std::fixed << std::setprecision(2)
              << "[sbm-progress] phase="
              << (processed < warmup ? "train" : "eval")
              << " examples=" << processed << '/' << total
              << " speed=" << speed << "/s"
              << " eta=" << eta << "s"
              << " train_nll=" << train_nll
              << " eval_nll=" << eval_nll
              << " live_nodes=" << diagnostics.live_nodes
              << " state_mb="
              << static_cast<double>(diagnostics.estimated_bytes) /
                     (1024.0 * 1024.0)
              << "        ";
    if (processed == total) {
        std::cerr << '\n';
    }
    std::cerr.flush();
}

} // namespace

static TokenExperimentResult run_token_views(std::span<const TokenDataView> datasets,
                                             std::size_t warmup_examples,
                                             Config config,
                                             bool strict_freeze,
                                             std::size_t prefill,
                                             std::size_t prune_interval,
                                             std::size_t merge_interval,
                                             const char* task) {
    if (datasets.empty() || datasets.front().vocab_size < 2U) {
        throw std::invalid_argument("invalid token dataset");
    }
    const auto vocabulary = datasets.front().vocab_size;
    std::size_t total_examples = 0U;
    std::size_t total_sequences = 0U;
    std::uint64_t combined_hash = datasets.size() == 1U
        ? datasets.front().dataset_hash : 0x53424D434F525055ULL;
    for (std::size_t index = 0U; index < datasets.size(); ++index) {
        const auto& dataset = datasets[index];
        if (dataset.vocab_size != vocabulary || dataset.sequence_offsets.size() < 2U) {
            throw std::invalid_argument("token shards have incompatible vocabularies");
        }
        total_examples += dataset.example_count();
        total_sequences += dataset.sequence_count();
        if (datasets.size() != 1U) {
            combined_hash ^= std::rotl(dataset.dataset_hash,
                static_cast<int>((index * 13U) & 63U));
        }
    }
    if (total_examples == 0U || warmup_examples == 0U || warmup_examples >= total_examples) {
        throw std::invalid_argument("warmup must split non-empty token train/eval sets");
    }

    config.objective = ObjectiveKind::TokenCrossEntropy;
    config.token_alphabet = vocabulary;
    config.vector_dim = vocabulary;
    config.allow_growth_when_frozen = !strict_freeze;
    SparseBranchMachine model(config);
    if (prefill != 0U) model.prefill_distractors(prefill);

    std::vector<std::uint64_t> unigram_counts(vocabulary, 0U);
    std::uint64_t unigram_total = 0U;
    ConditionalTable current_table;
    ConditionalTable pair_table;
    ConditionalTable lag2_table;
    ConditionalTable lag4_table;

    const auto baseline_training_start = std::chrono::steady_clock::now();
    std::size_t training_index = 0U;
    for (const auto& dataset : datasets) {
      for (std::size_t sequence = 0; sequence < dataset.sequence_count(); ++sequence) {
        const auto start = static_cast<std::size_t>(dataset.sequence_offsets[sequence]);
        const auto end = static_cast<std::size_t>(dataset.sequence_offsets[sequence + 1U]);
        for (std::size_t position = start; position + 1U < end; ++position) {
            if (training_index >= warmup_examples) break;
            const auto current = dataset.tokens[position];
            const auto target = dataset.tokens[position + 1U];
            const auto lag1 = token_at_lag(dataset, start, position, 1U);
            const auto lag2 = token_at_lag(dataset, start, position, 2U);
            const auto lag4 = token_at_lag(dataset, start, position, 4U);
            ++unigram_counts[target];
            ++unigram_total;
            current_table.observe(current, target);
            pair_table.observe(pair_key(lag1, current), target);
            lag2_table.observe(pair_key(lag2, current), target);
            lag4_table.observe(pair_key(lag4, current), target);
            ++training_index;
        }
        if (training_index >= warmup_examples) break;
      }
      if (training_index >= warmup_examples) break;
    }
    double baseline_elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - baseline_training_start).count();

    std::vector<float> unigram(vocabulary, 0.0F);
    constexpr double prior = 0.5;
    const double unigram_denominator = static_cast<double>(unigram_total) +
        prior * static_cast<double>(vocabulary);
    for (std::uint32_t token = 0; token < vocabulary; ++token) {
        unigram[token] = static_cast<float>(
            (static_cast<double>(unigram_counts[token]) + prior) /
            unigram_denominator);
    }

    TokenAccumulator train_accumulator;
    TokenAccumulator eval_accumulator;
    TokenAccumulator unigram_accumulator;
    TokenAccumulator current_accumulator;
    TokenAccumulator pair_accumulator;
    TokenAccumulator interpolated_multiscale_accumulator;
    TokenAccumulator seed_only_accumulator;
    TokenAccumulator active_channels_only_accumulator;
    TokenAccumulator content_channels_only_accumulator;
    TokenAccumulator tuple_channels_only_accumulator;
    std::array<ChannelAttributionAccumulator, kMaxAddressChannels>
        eval_channel_attribution{};
    std::unordered_map<std::uint64_t, ProgramAttributionAccumulator>
        eval_program_attribution{};
    std::array<double, kMaxAddressChannels> eval_channel_responsibility_sum{};
    double oracle_total = 0.0;
    std::uint64_t oracle_examples = 0U;

    std::size_t example_index = 0U;
    double model_elapsed = 0.0;
    const bool progress_enabled = total_examples >= 1000000U;
    const auto progress_started = std::chrono::steady_clock::now();
    auto next_progress = progress_started;
    if (progress_enabled) {
        std::cerr << '\r' << "[sbm-progress] start examples=" << total_examples
                  << " train=" << warmup_examples
                  << " eval=" << (total_examples - warmup_examples)
                  << " vocab=" << vocabulary << "        ";
        std::cerr.flush();
        next_progress += std::chrono::seconds(120);
    }
    for (const auto& dataset : datasets) {
      for (std::size_t sequence = 0; sequence < dataset.sequence_count(); ++sequence) {
        const auto start = static_cast<std::size_t>(dataset.sequence_offsets[sequence]);
        const auto end = static_cast<std::size_t>(dataset.sequence_offsets[sequence + 1U]);
        model.reset_sequence();
        std::array<double, kMaxAddressChannels> sequence_channel_credit{};
        std::size_t sequence_channel_credit_count = 0U;
        bool sequence_has_eval_attribution = false;
        for (std::size_t position = start; position + 1U < end; ++position) {
            const bool learn = example_index < warmup_examples;
            if (strict_freeze && example_index == warmup_examples) {
                model.freeze_topology();
            }
            const auto current = dataset.tokens[position];
            const auto target = dataset.tokens[position + 1U];
            const auto model_start = std::chrono::steady_clock::now();
            const auto stats = model.step_token(current, target, learn);
            model_elapsed += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - model_start).count();
            add_step(learn ? train_accumulator : eval_accumulator, stats);

            if (!learn) {
                if (stats.channel_credit_count != 0U) {
                    const auto count = std::min<std::size_t>(
                        stats.channel_credit_count, eval_channel_attribution.size());
                    for (std::size_t channel = 0U; channel < count; ++channel) {
                        auto& accumulator = eval_channel_attribution[channel];
                        const double credit = stats.channel_credit[channel];
                        accumulator.credit_sum += credit;
                        ++accumulator.observations;
                        if (credit > 0.0) ++accumulator.positive;
                        const auto dependency = stats.channel_dependency[channel];
                        const auto key = program_attribution_key(
                            static_cast<std::uint8_t>(channel), dependency);
                        auto [program_iterator, inserted] =
                            eval_program_attribution.try_emplace(key);
                        auto& program_accumulator = program_iterator->second;
                        if (inserted) {
                            program_accumulator.channel =
                                static_cast<std::uint8_t>(channel);
                            program_accumulator.parent_channel =
                                stats.channel_parent_channel[channel];
                            program_accumulator.dependency_channel =
                                stats.channel_dependency_channel[channel];
                            program_accumulator.input_state =
                                static_cast<AddressStateKind>(
                                    stats.channel_input_state[channel]);
                            program_accumulator.output_state =
                                static_cast<AddressStateKind>(
                                    stats.channel_output_state[channel]);
                            program_accumulator.required_dependency_binding =
                                static_cast<AddressBindingKind>(
                                    stats.channel_required_dependency_binding[channel]);
                            program_accumulator.dependency = dependency;
                        }
                        program_accumulator.credit_sum += credit;
                        program_accumulator.caller_removed_credit_sum +=
                            static_cast<double>(
                                stats.channel_caller_removed_credit[channel]);
                        program_accumulator.dependency_retained_credit_sum +=
                            static_cast<double>(
                                stats.channel_dependency_retained_credit[channel]);
                        program_accumulator.dependency_removed_credit_sum +=
                            static_cast<double>(
                                stats.channel_dependency_removed_credit[channel]);
                        if (stats.channel_binding_matched[channel] != 0U) {
                            ++program_accumulator.binding_matches;
                            if (stats.channel_binding_key[channel] != 0U) {
                                program_accumulator.binding_keys.insert(
                                    stats.channel_binding_key[channel]);
                            }
                            program_accumulator.binding_distance_sum +=
                                static_cast<double>(
                                    stats.channel_binding_distance[channel]);
                            program_accumulator.binding_pattern_span_sum +=
                                static_cast<double>(
                                    stats.channel_binding_pattern_span[channel]);
                        }
                        if (stats.channel_call_key[channel] != 0U) {
                            ++program_accumulator.call_matches;
                            program_accumulator.call_keys.insert(
                                stats.channel_call_key[channel]);
                        }
                        program_accumulator.description_cost +=
                            static_cast<double>(stats.channel_description_cost[channel]);
                        program_accumulator.execution_cost +=
                            static_cast<double>(stats.channel_execution_cost[channel]);
                        ++program_accumulator.observations;
                        if (credit > 0.0) ++program_accumulator.positive;
                        sequence_channel_credit[channel] += credit;
                        eval_channel_responsibility_sum[channel] +=
                            stats.channel_responsibility_mass[channel];
                    }
                    sequence_channel_credit_count =
                        std::max(sequence_channel_credit_count, count);
                    sequence_has_eval_attribution = true;
                }
                if (stats.channel_subset_available) {
                    add_cross_entropy(seed_only_accumulator,
                                      stats.seed_only_cross_entropy);
                    add_cross_entropy(active_channels_only_accumulator,
                                      stats.active_only_cross_entropy);
                    add_cross_entropy(content_channels_only_accumulator,
                                      stats.content_only_cross_entropy);
                    add_cross_entropy(tuple_channels_only_accumulator,
                                      stats.tuple_only_cross_entropy);
                }
                const auto baseline_start = std::chrono::steady_clock::now();
                const double unigram_probability = unigram[target];
                add_target_probability(unigram_accumulator, unigram_probability);
                const double current_probability = current_table.target_probability(
                    current, target, unigram_probability);
                add_target_probability(current_accumulator, current_probability);

                const auto lag1 = token_at_lag(dataset, start, position, 1U);
                const auto lag2 = token_at_lag(dataset, start, position, 2U);
                const auto lag4 = token_at_lag(dataset, start, position, 4U);
                const double pair_probability = pair_table.target_probability(
                    pair_key(lag1, current), target, current_probability);
                const double lag2_probability = lag2_table.target_probability(
                    pair_key(lag2, current), target, current_probability);
                const double lag4_probability = lag4_table.target_probability(
                    pair_key(lag4, current), target, current_probability);
                add_target_probability(pair_accumulator, pair_probability);
                add_target_probability(
                    interpolated_multiscale_accumulator,
                    (pair_probability + lag2_probability + lag4_probability) / 3.0);
                baseline_elapsed += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - baseline_start).count();

                if (!dataset.oracle_nll.empty()) {
                    oracle_total += dataset.oracle_nll[position + 1U];
                    ++oracle_examples;
                }
            }

            if (learn && prune_interval != 0U &&
                (example_index + 1U) % prune_interval == 0U) {
                (void)model.prune();
            }
            if (learn && merge_interval != 0U &&
                (example_index + 1U) % merge_interval == 0U) {
                (void)model.merge_redundant();
            }
            ++example_index;
            if (progress_enabled) {
                const auto now = std::chrono::steady_clock::now();
                if (now >= next_progress || example_index == warmup_examples ||
                    example_index == total_examples) {
                    emit_progress(example_index, total_examples, warmup_examples,
                                  train_accumulator, eval_accumulator, model,
                                  progress_started);
                    next_progress = now + std::chrono::seconds(120);
                }
            }
        }
        if (sequence_has_eval_attribution) {
            for (std::size_t channel = 0U; channel < sequence_channel_credit_count;
                 ++channel) {
                auto& accumulator = eval_channel_attribution[channel];
                ++accumulator.documents;
                if (sequence_channel_credit[channel] > 0.0) {
                    ++accumulator.positive_documents;
                }
            }
        }
      }
    }
    const double elapsed = std::max(model_elapsed, 1e-12);

    TokenExperimentResult result;
    result.task = task;
    result.diagnostics = model.diagnostics();
    result.train = finish(train_accumulator);
    result.eval = finish(eval_accumulator);
    result.unigram_baseline_eval = finish(unigram_accumulator);
    result.current_token_baseline_eval = finish(current_accumulator);
    result.pair_context_baseline_eval = finish(pair_accumulator);
    result.interpolated_multiscale_baseline_eval =
        finish(interpolated_multiscale_accumulator);
    result.multiscale_baseline_eval = result.interpolated_multiscale_baseline_eval;
    result.seed_only_eval = finish(seed_only_accumulator);
    result.active_channels_only_eval = finish(active_channels_only_accumulator);
    result.content_channels_only_eval = finish(content_channels_only_accumulator);
    result.tuple_channels_only_eval = finish(tuple_channels_only_accumulator);
    result.oracle_cross_entropy = oracle_examples == 0U
        ? std::numeric_limits<double>::quiet_NaN()
        : oracle_total / static_cast<double>(oracle_examples);
    result.excess_cross_entropy = oracle_examples == 0U
        ? std::numeric_limits<double>::quiet_NaN()
        : result.eval.cross_entropy - result.oracle_cross_entropy;
    result.steps_per_second = static_cast<double>(total_examples) / elapsed;
    result.elapsed_seconds = elapsed;
    result.baseline_elapsed_seconds = baseline_elapsed;
    result.strict_freeze = strict_freeze;
    result.dataset_hash = combined_hash;
    result.vocab_size = vocabulary;
    result.train_examples = warmup_examples;
    result.eval_examples = total_examples - warmup_examples;
    result.sequence_count = total_sequences;
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
    if (config.record_channel_attribution) {
        result.eval_channel_attribution = finish_channel_attribution(
            eval_channel_attribution, result.learned_address_programs,
            result.learned_channel_phase);
        result.eval_program_attribution = finish_program_attribution(
            eval_program_attribution, result.learned_address_programs,
            result.learned_channel_generation,
            result.learned_channel_parent_edge_kind,
            result.learned_channel_dependency_edge_kind);
        result.eval_channel_mean_responsibility.reserve(
            result.eval_channel_attribution.size());
        for (std::size_t channel = 0U;
             channel < result.eval_channel_attribution.size(); ++channel) {
            const auto observations =
                result.eval_channel_attribution[channel].eval_observations;
            result.eval_channel_mean_responsibility.push_back(
                observations == 0U ? 0.0 :
                    eval_channel_responsibility_sum[channel] /
                    static_cast<double>(observations));
        }
    }
    result.address_dependency_graph = finish_dependency_graph(
        result.learned_address_programs, result.learned_channel_phase,
        result.learned_channel_effective_enabled,
        result.learned_channel_parent, result.learned_channel_dependency,
        result.learned_channel_generation,
        result.learned_channel_parent_edge_kind,
        result.learned_channel_dependency_edge_kind,
        result.eval_program_attribution);
    result.address_dependency_edges = finish_dependency_edges(
        result.learned_address_programs, result.learned_channel_parent,
        result.learned_channel_dependency,
        result.learned_channel_generation,
        result.learned_channel_parent_edge_kind,
        result.learned_channel_dependency_edge_kind,
        result.eval_program_attribution);
    result.topology_events = model.topology_events();
    result.exact_region_mass = config.exact_region_mass;
    result.edge_score_weight = config.edge_score_weight;
    result.softmax_temperature = config.softmax_temperature;
    result.label_smoothing = config.label_smoothing;
    result.sparse_token_output = config.sparse_token_output;
    result.output_tree_seed = config.output_tree_seed;
    result.structural_description_cost_weight =
        config.structural_description_cost_weight;
    result.structural_execution_cost_weight =
        config.structural_execution_cost_weight;
    result.binding_reuse_value_weight = config.binding_reuse_value_weight;
    return result;
}

TokenExperimentResult run_token_experiment(const TokenDataset& dataset,
                                           std::size_t warmup_examples,
                                           Config config,
                                           bool strict_freeze,
                                           std::size_t prefill,
                                           std::size_t prune_interval,
                                           std::size_t merge_interval) {
    const TokenDataView view{
        dataset.vocab_size, dataset.tokens, dataset.sequence_offsets,
        dataset.oracle_nll, hash_dataset(dataset),
        "mathematical_next_token_cross_entropy"};
    return run_token_views(std::span<const TokenDataView>(&view, 1U), warmup_examples,
                           std::move(config), strict_freeze, prefill, prune_interval,
                           merge_interval, "mathematical_next_token_cross_entropy");
}

TokenExperimentResult run_token_experiment(const MappedTokenShard& shard,
                                           std::size_t warmup_examples,
                                           Config config,
                                           bool strict_freeze,
                                           std::size_t prefill,
                                           std::size_t prune_interval,
                                           std::size_t merge_interval) {
    const TokenDataView view{
        shard.vocab_size(), shard.tokens(), shard.sequence_offsets(), {},
        shard.dataset_hash(), "real_corpus_next_token_cross_entropy"};
    return run_token_views(std::span<const TokenDataView>(&view, 1U), warmup_examples,
                           std::move(config), strict_freeze, prefill, prune_interval,
                           merge_interval, "real_corpus_next_token_cross_entropy");
}

TokenExperimentResult run_token_corpus_experiment(
    std::span<const MappedTokenShard* const> train_shards,
    std::span<const MappedTokenShard* const> eval_shards,
    Config config,
    bool strict_freeze,
    std::size_t prefill,
    std::size_t prune_interval,
    std::size_t merge_interval) {
    if (train_shards.empty() || eval_shards.empty()) {
        throw std::invalid_argument("token corpus requires train and eval shards");
    }
    std::vector<TokenDataView> views;
    views.reserve(train_shards.size() + eval_shards.size());
    std::size_t warmup_examples = 0U;
    const auto append = [&](const MappedTokenShard* shard, bool training) {
        if (shard == nullptr) throw std::invalid_argument("null token shard");
        views.push_back(TokenDataView{
            shard->vocab_size(), shard->tokens(), shard->sequence_offsets(), {},
            shard->dataset_hash(), "real_corpus_next_token_cross_entropy"});
        if (training) warmup_examples += views.back().example_count();
    };
    for (const auto* shard : train_shards) append(shard, true);
    for (const auto* shard : eval_shards) append(shard, false);
    return run_token_views(views, warmup_examples, std::move(config), strict_freeze,
                           prefill, prune_interval, merge_interval,
                           "real_corpus_next_token_cross_entropy");
}

namespace {

TokenDataset copy_limited_examples(
    std::span<const MappedTokenShard* const> shards,
    std::size_t max_examples) {
    if (shards.empty()) {
        throw std::invalid_argument("limited token corpus requires shards");
    }
    TokenDataset result;
    result.sequence_offsets.push_back(0U);
    std::size_t copied_examples = 0U;
    for (const auto* shard : shards) {
        if (shard == nullptr) throw std::invalid_argument("null token shard");
        if (result.vocab_size == 0U) result.vocab_size = shard->vocab_size();
        if (result.vocab_size != shard->vocab_size()) {
            throw std::invalid_argument("limited corpus vocabularies differ");
        }
        const auto tokens = shard->tokens();
        const auto offsets = shard->sequence_offsets();
        for (std::size_t sequence = 0U;
             sequence + 1U < offsets.size() && copied_examples < max_examples;
             ++sequence) {
            const auto start = static_cast<std::size_t>(offsets[sequence]);
            const auto end = static_cast<std::size_t>(offsets[sequence + 1U]);
            if (end <= start + 1U) continue;
            const auto available_examples = end - start - 1U;
            const auto take_examples =
                std::min<std::size_t>(available_examples,
                                      max_examples - copied_examples);
            const auto take_tokens = take_examples + 1U;
            result.tokens.insert(result.tokens.end(),
                                 tokens.begin() + static_cast<std::ptrdiff_t>(start),
                                 tokens.begin() +
                                     static_cast<std::ptrdiff_t>(start + take_tokens));
            result.sequence_offsets.push_back(result.tokens.size());
            copied_examples += take_examples;
        }
        if (copied_examples >= max_examples) break;
    }
    if (result.vocab_size == 0U) {
        throw std::invalid_argument("limited corpus has no vocabulary");
    }
    return result;
}

} // namespace

TokenExperimentResult run_token_corpus_experiment_limited(
    std::span<const MappedTokenShard* const> train_shards,
    std::span<const MappedTokenShard* const> eval_shards,
    std::size_t max_train_examples,
    std::size_t max_eval_examples,
    Config config,
    bool strict_freeze,
    std::size_t prefill,
    std::size_t prune_interval,
    std::size_t merge_interval) {
    const auto train = copy_limited_examples(train_shards, max_train_examples);
    const auto eval = copy_limited_examples(eval_shards, max_eval_examples);
    const TokenDataView views[] = {
        {train.vocab_size, train.tokens, train.sequence_offsets, {},
         hash_dataset(train), "real_corpus_next_token_cross_entropy"},
        {eval.vocab_size, eval.tokens, eval.sequence_offsets, {},
         hash_dataset(eval), "real_corpus_next_token_cross_entropy"},
    };
    return run_token_views(views, train.example_count(), std::move(config),
                           strict_freeze, prefill, prune_interval, merge_interval,
                           "real_corpus_next_token_cross_entropy");
}

std::string to_json(const TokenExperimentResult& result) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(8);
    const auto emit_metrics = [&](const char* prefix, const TokenMetrics& metrics,
                                  bool trailing = true) {
        out << "  \"" << prefix << "_cross_entropy\": " << metrics.cross_entropy << ",\n"
            << "  \"" << prefix << "_bits_per_token\": " << metrics.bits_per_token << ",\n"
            << "  \"" << prefix << "_perplexity\": " << metrics.perplexity << ",\n"
            << "  \"" << prefix << "_top1_accuracy\": ";
        if (metrics.ranking_available) out << metrics.top1_accuracy;
        else out << "null";
        out << ",\n  \"" << prefix << "_top5_accuracy\": ";
        if (metrics.ranking_available) out << metrics.top5_accuracy;
        else out << "null";
        out << ",\n"
            << "  \"" << prefix << "_mean_target_probability\": "
            << metrics.mean_target_probability;
        if (trailing) out << ',';
        out << '\n';
    };
    out << "{\n"
        << "  \"task\": \"" << result.task << "\",\n"
        << "  \"objective\": \"token_cross_entropy\",\n"
        << "  \"vocab_size\": " << result.vocab_size << ",\n"
        << "  \"sequence_count\": " << result.sequence_count << ",\n"
        << "  \"train_examples\": " << result.train_examples << ",\n"
        << "  \"eval_examples\": " << result.eval_examples << ",\n"
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
    out << "],\n  \"address_dependency_graph\": [";
    for (std::size_t i = 0; i < result.address_dependency_graph.size(); ++i) {
        if (i != 0U) out << ", ";
        const auto& summary = result.address_dependency_graph[i];
        out << "{\"channel\":" << static_cast<unsigned>(summary.channel)
            << ",\"lags\":[";
        for (std::size_t j = 0; j < summary.program.arity; ++j) {
            if (j != 0U) out << ',';
            out << summary.program.lags[j];
        }
        const double own_mean_credit = summary.own_observations == 0U ? 0.0 :
            summary.own_credit_sum /
                static_cast<double>(summary.own_observations);
        const double own_binding_match_fraction =
            summary.own_observations == 0U ? 0.0 :
                static_cast<double>(summary.own_binding_matches) /
                    static_cast<double>(summary.own_observations);
        const double own_call_match_fraction =
            summary.own_observations == 0U ? 0.0 :
                static_cast<double>(summary.own_call_matches) /
                    static_cast<double>(summary.own_observations);
        out << "],\"op\":" << static_cast<unsigned>(summary.program.op)
            << ",\"phase\":" << static_cast<unsigned>(summary.phase)
            << ",\"parent_channel\":"
            << static_cast<unsigned>(summary.parent_channel)
            << ",\"dependency_channel\":"
            << static_cast<unsigned>(summary.dependency_channel)
            << ",\"channel_generation\":"
            << summary.channel_generation
            << ",\"parent_edge_kind\":"
            << static_cast<unsigned>(summary.parent_edge_kind)
            << ",\"dependency_edge_kind\":"
            << static_cast<unsigned>(summary.dependency_edge_kind)
            << ",\"input_state\":"
            << static_cast<unsigned>(summary.input_state)
            << ",\"output_state\":"
            << static_cast<unsigned>(summary.output_state)
            << ",\"required_dependency_binding\":"
            << static_cast<unsigned>(summary.required_dependency_binding)
            << ",\"effective_enabled\":"
            << static_cast<unsigned>(summary.effective_enabled)
            << ",\"dependency_available\":"
            << static_cast<unsigned>(summary.dependency_available)
            << ",\"direct_caller_count\":"
            << summary.direct_caller_count
            << ",\"effective_direct_caller_count\":"
            << summary.effective_direct_caller_count
            << ",\"blocked_direct_caller_count\":"
            << summary.blocked_direct_caller_count
            << ",\"own_observations\":" << summary.own_observations
            << ",\"own_binding_matches\":"
            << summary.own_binding_matches
            << ",\"unique_binding_keys\":"
            << summary.unique_binding_keys
            << ",\"binding_key_reuse_events\":"
            << summary.binding_key_reuse_events
            << ",\"own_binding_match_fraction\":"
            << own_binding_match_fraction
            << ",\"own_call_matches\":" << summary.own_call_matches
            << ",\"unique_call_keys\":" << summary.unique_call_keys
            << ",\"call_key_reuse_events\":"
            << summary.call_key_reuse_events
            << ",\"own_call_match_fraction\":"
            << own_call_match_fraction
            << ",\"own_credit_sum\":" << summary.own_credit_sum
            << ",\"own_mean_credit\":" << own_mean_credit
            << ",\"own_caller_removed_credit\":"
            << summary.own_caller_removed_credit_sum
            << ",\"downstream_call_matches\":"
            << summary.downstream_call_matches
            << ",\"downstream_caller_removed_credit\":"
            << summary.downstream_caller_removed_credit_sum
            << ",\"downstream_dependency_removed_credit\":"
            << summary.downstream_dependency_removed_credit_sum << "}";
    }
    out << "],\n  \"address_dependency_edges\": [";
    for (std::size_t i = 0; i < result.address_dependency_edges.size(); ++i) {
        if (i != 0U) out << ", ";
        const auto& edge = result.address_dependency_edges[i];
        const double call_match_fraction = edge.observations == 0U ? 0.0 :
            static_cast<double>(edge.call_matches) /
                static_cast<double>(edge.observations);
        out << "{\"caller_channel\":"
            << static_cast<unsigned>(edge.caller_channel)
            << ",\"dependency_channel\":"
            << static_cast<unsigned>(edge.dependency_channel)
            << ",\"caller_generation\":" << edge.caller_generation
            << ",\"dependency_generation\":"
            << edge.dependency_generation
            << ",\"edge_kind\":"
            << static_cast<unsigned>(edge.edge_kind)
            << ",\"caller_input_state\":"
            << static_cast<unsigned>(edge.caller_input_state)
            << ",\"dependency_output_state\":"
            << static_cast<unsigned>(edge.dependency_output_state)
            << ",\"required_dependency_binding\":"
            << static_cast<unsigned>(edge.required_dependency_binding)
            << ",\"observations\":" << edge.observations
            << ",\"call_matches\":" << edge.call_matches
            << ",\"unique_call_keys\":" << edge.unique_call_keys
            << ",\"call_key_reuse_events\":"
            << edge.call_key_reuse_events
            << ",\"call_match_fraction\":"
            << call_match_fraction
            << ",\"caller_removed_credit_sum\":"
            << edge.caller_removed_credit_sum
            << ",\"dependency_removed_credit_sum\":"
            << edge.dependency_removed_credit_sum << "}";
    }
    out << "],\n  \"eval_channel_attribution\": [";
    for (std::size_t i = 0; i < result.eval_channel_attribution.size(); ++i) {
        if (i != 0U) out << ", ";
        const auto& attribution = result.eval_channel_attribution[i];
        out << "{\"channel\":" << static_cast<unsigned>(attribution.channel)
            << ",\"lags\":[";
        for (std::size_t j = 0; j < attribution.program.arity; ++j) {
            if (j != 0U) out << ',';
            out << attribution.program.lags[j];
        }
        out << "],\"op\":" << static_cast<unsigned>(attribution.program.op)
            << ",\"phase\":" << static_cast<unsigned>(attribution.phase)
            << ",\"eval_observations\":" << attribution.eval_observations
            << ",\"eval_positive\":" << attribution.eval_positive
            << ",\"eval_documents\":" << attribution.eval_documents
            << ",\"eval_positive_documents\":"
            << attribution.eval_positive_documents
            << ",\"eval_credit_sum\":" << attribution.eval_credit_sum
            << ",\"eval_mean_credit\":" << attribution.eval_mean_credit
            << ",\"eval_positive_fraction\":"
            << attribution.eval_positive_fraction
            << ",\"eval_positive_document_fraction\":"
            << attribution.eval_positive_document_fraction << "}";
    }
    out << "],\n  \"eval_channel_mean_responsibility\": [";
    for (std::size_t i = 0; i < result.eval_channel_mean_responsibility.size(); ++i) {
        if (i != 0U) out << ", ";
        out << result.eval_channel_mean_responsibility[i];
    }
    out << "],\n  \"eval_program_attribution\": [";
    for (std::size_t i = 0; i < result.eval_program_attribution.size(); ++i) {
        if (i != 0U) out << ", ";
        const auto& attribution = result.eval_program_attribution[i];
        const double mean_credit = attribution.observations == 0U ? 0.0 :
            attribution.credit_sum / static_cast<double>(attribution.observations);
        const double caller_removed_mean = attribution.observations == 0U ? 0.0 :
            attribution.caller_removed_credit_sum /
                static_cast<double>(attribution.observations);
        const double dependency_retained_mean = attribution.observations == 0U ? 0.0 :
            attribution.dependency_retained_credit_sum /
                static_cast<double>(attribution.observations);
        const double dependency_removed_mean = attribution.observations == 0U ? 0.0 :
            attribution.dependency_removed_credit_sum /
                static_cast<double>(attribution.observations);
        const double binding_match_fraction = attribution.observations == 0U ? 0.0 :
            static_cast<double>(attribution.binding_matches) /
                static_cast<double>(attribution.observations);
        const double call_match_fraction = attribution.observations == 0U ? 0.0 :
            static_cast<double>(attribution.call_matches) /
                static_cast<double>(attribution.observations);
        const double mean_binding_distance = attribution.binding_matches == 0U ? 0.0 :
            attribution.binding_distance_sum /
                static_cast<double>(attribution.binding_matches);
        const double mean_binding_pattern_span =
            attribution.binding_matches == 0U ? 0.0 :
                attribution.binding_pattern_span_sum /
                    static_cast<double>(attribution.binding_matches);
        const double positive_fraction = attribution.observations == 0U ? 0.0 :
            static_cast<double>(attribution.positive) /
                static_cast<double>(attribution.observations);
        const double structural_value = attribution.credit_sum -
            static_cast<double>(result.structural_description_cost_weight) *
                attribution.description_cost -
            static_cast<double>(result.structural_execution_cost_weight) *
                attribution.execution_cost;
        const double binding_reuse_fraction = attribution.binding_matches == 0U ? 0.0 :
            static_cast<double>(attribution.binding_key_reuse_events) /
                static_cast<double>(attribution.binding_matches);
        const double binding_reuse_bonus =
            static_cast<double>(result.binding_reuse_value_weight) *
            std::log1p(static_cast<double>(attribution.binding_key_reuse_events)) *
            binding_reuse_fraction;
        const double reuse_aware_structural_value =
            structural_value + binding_reuse_bonus;
        out << "{\"channel\":" << static_cast<unsigned>(attribution.channel)
            << ",\"lags\":[";
        for (std::size_t j = 0; j < attribution.program.arity; ++j) {
            if (j != 0U) out << ',';
            out << attribution.program.lags[j];
        }
        out << "],\"op\":" << static_cast<unsigned>(attribution.program.op)
            << ",\"parent_channel\":"
            << static_cast<unsigned>(attribution.parent_channel)
            << ",\"dependency_channel\":"
            << static_cast<unsigned>(attribution.dependency_channel)
            << ",\"channel_generation\":"
            << attribution.channel_generation
            << ",\"parent_edge_kind\":"
            << static_cast<unsigned>(attribution.parent_edge_kind)
            << ",\"dependency_edge_kind\":"
            << static_cast<unsigned>(attribution.dependency_edge_kind)
            << ",\"input_state\":"
            << static_cast<unsigned>(attribution.input_state)
            << ",\"output_state\":"
            << static_cast<unsigned>(attribution.output_state)
            << ",\"required_dependency_binding\":"
            << static_cast<unsigned>(attribution.required_dependency_binding)
            << ",\"dependency\":" << attribution.dependency
            << ",\"observations\":" << attribution.observations
            << ",\"positive\":" << attribution.positive
            << ",\"credit_sum\":" << attribution.credit_sum
            << ",\"caller_removed_credit\":"
            << attribution.caller_removed_credit_sum
            << ",\"dependency_retained_credit\":"
            << attribution.dependency_retained_credit_sum
            << ",\"dependency_removed_credit\":"
            << attribution.dependency_removed_credit_sum
            << ",\"binding_matches\":" << attribution.binding_matches
            << ",\"unique_binding_keys\":"
            << attribution.unique_binding_keys
            << ",\"binding_key_reuse_events\":"
            << attribution.binding_key_reuse_events
            << ",\"binding_match_fraction\":" << binding_match_fraction
            << ",\"call_matches\":" << attribution.call_matches
            << ",\"unique_call_keys\":" << attribution.unique_call_keys
            << ",\"call_key_reuse_events\":"
            << attribution.call_key_reuse_events
            << ",\"call_match_fraction\":" << call_match_fraction
            << ",\"mean_binding_distance\":" << mean_binding_distance
            << ",\"mean_binding_pattern_span\":"
            << mean_binding_pattern_span
            << ",\"mean_credit\":" << mean_credit
            << ",\"caller_removed_mean_credit\":" << caller_removed_mean
            << ",\"dependency_retained_mean_credit\":"
            << dependency_retained_mean
            << ",\"dependency_removed_mean_credit\":"
            << dependency_removed_mean
            << ",\"positive_fraction\":" << positive_fraction
            << ",\"description_cost\":" << attribution.description_cost
            << ",\"execution_cost\":" << attribution.execution_cost
            << ",\"structural_value_without_reuse\":" << structural_value
            << ",\"binding_reuse_bonus\":" << binding_reuse_bonus
            << ",\"structural_value\":" << reuse_aware_structural_value << "}";
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
            << ",\"structural_value_without_reuse\":"
            << event.structural_value_without_reuse
            << ",\"binding_reuse_bonus\":" << event.binding_reuse_bonus
            << ",\"binding_reuse_observations\":"
            << event.binding_reuse_observations
            << ",\"binding_reuse_unique_keys\":"
            << event.binding_reuse_unique_keys
            << ",\"binding_reuse_events\":" << event.binding_reuse_events
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
        << "  \"edge_score_weight\": " << result.edge_score_weight << ",\n"
        << "  \"softmax_temperature\": " << result.softmax_temperature << ",\n"
        << "  \"label_smoothing\": " << result.label_smoothing << ",\n"
        << "  \"sparse_token_output\": "
        << (result.sparse_token_output ? "true" : "false") << ",\n"
        << "  \"output_tree_seed\": " << result.output_tree_seed << ",\n"
        << "  \"structural_description_cost_weight\": "
        << result.structural_description_cost_weight << ",\n"
        << "  \"structural_execution_cost_weight\": "
        << result.structural_execution_cost_weight << ",\n"
        << "  \"binding_reuse_value_weight\": "
        << result.binding_reuse_value_weight << ",\n"
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
        << "  \"sparse_output_entries\": "
        << result.diagnostics.sparse_output_entries << ",\n"
        << "  \"topology_proposals\": " << result.diagnostics.topology_proposals << ",\n"
        << "  \"topology_accepted\": " << result.diagnostics.topology_accepted << ",\n"
        << "  \"topology_rejected\": " << result.diagnostics.topology_rejected << ",\n"
        << "  \"topology_pruned\": " << result.diagnostics.topology_pruned << ",\n"
        << "  \"topology_restored\": " << result.diagnostics.topology_restored << ",\n"
        << "  \"active_channels\": " << result.diagnostics.active_channels << ",\n"
        << "  \"dependency_blocked_channels\": "
        << result.diagnostics.dependency_blocked_channels << ",\n"
        << "  \"probe_channels\": " << result.diagnostics.probe_channels << ",\n"
        << "  \"address_execution_frames\": "
        << result.diagnostics.address_execution_frames << ",\n"
        << "  \"address_binding_hits\": "
        << result.diagnostics.address_binding_hits << ",\n"
        << "  \"address_binding_misses\": "
        << result.diagnostics.address_binding_misses << ",\n"
        << "  \"binding_reuse_observations\": "
        << result.diagnostics.binding_reuse_observations << ",\n"
        << "  \"binding_reuse_unique_keys\": "
        << result.diagnostics.binding_reuse_unique_keys << ",\n"
        << "  \"binding_reuse_events\": "
        << result.diagnostics.binding_reuse_events << ",\n"
        << "  \"address_binding_by_kind\": [";
    for (std::size_t index = 0U; index < kAddressBindingKindCount; ++index) {
        if (index != 0U) out << ", ";
        const auto frames = result.diagnostics.address_binding_kind_frames[index];
        const auto hits = result.diagnostics.address_binding_kind_hits[index];
        const double hit_rate = frames == 0U ? 0.0 :
            static_cast<double>(hits) / static_cast<double>(frames);
        const double mean_distance = hits == 0U ? 0.0 :
            result.diagnostics.address_binding_kind_distance_sum[index] /
                static_cast<double>(hits);
        const double mean_pattern_span = hits == 0U ? 0.0 :
            result.diagnostics.address_binding_kind_pattern_span_sum[index] /
                static_cast<double>(hits);
        out << "{\"kind\":" << index
            << ",\"frames\":" << frames
            << ",\"hits\":" << hits
            << ",\"hit_rate\":" << hit_rate
            << ",\"mean_distance\":" << mean_distance
            << ",\"mean_pattern_span\":" << mean_pattern_span << "}";
    }
    out << "],\n"
        << "  \"quarantined_channels\": "
        << result.diagnostics.quarantined_channels << ",\n"
        << "  \"recoverable_retired_channels\": "
        << result.diagnostics.recoverable_retired_channels << ",\n"
        << "  \"structural_value_nats\": "
        << result.diagnostics.structural_value_nats << ",\n"
        << "  \"structural_description_cost\": "
        << result.diagnostics.structural_description_cost << ",\n"
        << "  \"structural_execution_cost\": "
        << result.diagnostics.structural_execution_cost << ",\n"
        << "  \"simd_enabled\": "
        << (result.diagnostics.simd_enabled ? "true" : "false") << ",\n";
    emit_metrics("train", result.train);
    emit_metrics("eval", result.eval);
    emit_metrics("unigram_baseline_eval", result.unigram_baseline_eval);
    emit_metrics("current_token_baseline_eval", result.current_token_baseline_eval);
    emit_metrics("pair_context_baseline_eval", result.pair_context_baseline_eval);
    emit_metrics("multiscale_baseline_eval", result.multiscale_baseline_eval);
    emit_metrics("interpolated_multiscale_baseline_eval",
                 result.interpolated_multiscale_baseline_eval);
    emit_metrics("seed_only_eval", result.seed_only_eval);
    emit_metrics("active_channels_only_eval", result.active_channels_only_eval);
    emit_metrics("content_channels_only_eval", result.content_channels_only_eval);
    emit_metrics("tuple_channels_only_eval", result.tuple_channels_only_eval);
    if (std::isfinite(result.oracle_cross_entropy)) {
        out << "  \"oracle_cross_entropy\": " << result.oracle_cross_entropy << ",\n"
            << "  \"excess_cross_entropy\": " << result.excess_cross_entropy << ",\n";
    } else {
        out << "  \"oracle_cross_entropy\": null,\n"
            << "  \"excess_cross_entropy\": null,\n";
    }
    out << "  \"steps_per_second\": " << result.steps_per_second << ",\n"
        << "  \"elapsed_seconds\": " << result.elapsed_seconds << ",\n"
        << "  \"baseline_elapsed_seconds\": "
        << result.baseline_elapsed_seconds << ",\n"
        << "  \"strict_freeze\": " << (result.strict_freeze ? "true" : "false") << ",\n"
        << "  \"dataset_hash\": " << result.dataset_hash << "\n"
        << "}\n";
    return out.str();
}

} // namespace sbm
