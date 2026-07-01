#include "sbm/machine.hpp"
#include "sbm/math.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sbm {
namespace {

[[nodiscard]] float softplus(float value) noexcept {
    if (value > 20.0F) return value;
    if (value < -20.0F) return std::exp(value);
    return std::log1p(std::exp(value));
}

[[nodiscard]] float branch_loss(float normalized_logit, bool right) noexcept {
    return right ? softplus(-normalized_logit) : softplus(normalized_logit);
}

[[nodiscard]] float log_branch_probability(float normalized_logit,
                                           bool right) noexcept {
    return -branch_loss(normalized_logit, right);
}

[[nodiscard]] float sigmoid(float value) noexcept {
    if (value >= 0.0F) {
        const float inverse = std::exp(-value);
        return 1.0F / (1.0F + inverse);
    }
    const float exponential = std::exp(value);
    return exponential / (1.0F + exponential);
}

[[nodiscard]] std::uint64_t decision_mask(std::uint32_t decision) noexcept {
    const auto mixed = static_cast<std::uint64_t>(decision) * 0x9E3779B97F4A7C15ULL;
    return 1ULL << (mixed >> 58U);
}

} // namespace

float SparseBranchMachine::sparse_logit(std::size_t slot,
                                        std::uint32_t decision) const noexcept {
    if (slot >= sparse_outputs_.size()) return 0.0F;
    const auto& entries = sparse_outputs_[slot];
    const auto found = std::lower_bound(
        entries.begin(), entries.end(), decision,
        [](const SparseOutputEntry& entry, std::uint32_t value) {
            return entry.decision < value;
        });
    return found != entries.end() && found->decision == decision
        ? found->logit : 0.0F;
}

SparseBranchMachine::SparseOutputEntry* SparseBranchMachine::mutable_sparse_entry(
    std::size_t slot,
    std::uint32_t decision,
    bool force_admission) {
    auto& entries = sparse_outputs_.at(slot);
    auto found = std::lower_bound(
        entries.begin(), entries.end(), decision,
        [](const SparseOutputEntry& entry, std::uint32_t value) {
            return entry.decision < value;
        });
    if (found != entries.end() && found->decision == decision) {
        return &*found;
    }
    if (!force_admission && entries.size() >= config_.max_sparse_decisions_per_node) {
        constexpr std::size_t candidate_limit = 8U;
        constexpr std::uint8_t required_sightings = 2U;
        auto& candidates = sparse_admission_.at(slot);
        auto candidate = std::find_if(
            candidates.begin(), candidates.end(),
            [decision](const SparseAdmissionCandidate& value) {
                return value.decision == decision;
            });
        if (candidate == candidates.end()) {
            if (candidates.size() >= candidate_limit) {
                candidate = std::min_element(
                    candidates.begin(), candidates.end(),
                    [](const SparseAdmissionCandidate& left,
                       const SparseAdmissionCandidate& right) {
                        return left.sightings < right.sightings ||
                            (left.sightings == right.sightings &&
                             left.last_seen_step < right.last_seen_step);
                    });
                *candidate = {decision, 1U, total_steps_};
            } else {
                candidates.push_back({decision, 1U, total_steps_});
            }
            ++sparse_output_admission_rejections_;
            return nullptr;
        }
        candidate->last_seen_step = total_steps_;
        if (candidate->sightings < required_sightings) ++candidate->sightings;
        if (candidate->sightings < required_sightings) {
            ++sparse_output_admission_rejections_;
            return nullptr;
        }
        candidates.erase(candidate);
        ++sparse_output_admission_promotions_;
    }
    const auto mask = decision_mask(decision);
    if ((sparse_output_evicted_masks_[slot] & mask) != 0U) {
        ++sparse_output_probable_reconstructions_;
    }
    ++sparse_output_insertions_;
    if (entries.size() >= config_.max_sparse_decisions_per_node) {
        const auto victim_index = detail::select_sparse_output_victim(entries, total_steps_);
        const auto victim = entries.begin() + static_cast<std::ptrdiff_t>(victim_index);
        sparse_output_evicted_masks_[slot] |= decision_mask(victim->decision);
        entries.erase(victim);
        ++sparse_output_evictions_;
        found = std::lower_bound(
            entries.begin(), entries.end(), decision,
            [](const SparseOutputEntry& entry, std::uint32_t value) {
                return entry.decision < value;
            });
    }
    return &*entries.insert(found, {decision, 0.0F, 0U, 0.0F, total_steps_});
}

float SparseBranchMachine::aggregate_sparse_logit(
    std::span<const ScoredNode> active,
    std::uint32_t decision) const noexcept {
    float value = 0.0F;
    for (const auto& node : active) {
        if (std::abs(node.responsibility) < 1e-8F) continue;
        const auto slot = slot_of(node.id);
        if (slot == SIZE_MAX) continue;
        value += node.responsibility * sparse_logit(slot, decision);
    }
    return value;
}

float SparseBranchMachine::aggregate_sparse_logit_masked(
    std::span<const ScoredNode> active,
    std::uint32_t decision,
    std::uint32_t channel_mask,
    float mass_scale) const noexcept {
    float value = 0.0F;
    for (const auto& node : active) {
        if (node.channel >= kMaxAddressChannels ||
            (channel_mask & (1U << node.channel)) == 0U ||
            std::abs(node.responsibility) < 1e-8F) {
            continue;
        }
        const auto slot = slot_of(node.id);
        if (slot == SIZE_MAX) continue;
        value += mass_scale * node.responsibility * sparse_logit(slot, decision);
    }
    return value;
}

float SparseBranchMachine::global_output_logit(
    std::uint32_t decision) const noexcept {
    return decision < global_output_logit_cache_.size()
        ? global_output_logit_cache_[decision] : 0.0F;
}

void SparseBranchMachine::observe_global_output_path(
    std::span<const detail::ImplicitDecision> path) {
    if (global_output_prior_updates_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("global output prior update count overflow");
    }
    for (const auto& step : path) {
        auto& total = global_output_total_.at(step.id);
        auto& right = global_output_right_.at(step.id);
        if (total == std::numeric_limits<std::uint64_t>::max() ||
            (step.right && right == std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error("global output prior branch count overflow");
        }
        ++total;
        if (step.right) ++right;
        constexpr double pseudocount = 0.5;
        const double left = static_cast<double>(total - right);
        global_output_logit_cache_[step.id] = static_cast<float>(std::log(
            (static_cast<double>(right) + pseudocount) / (left + pseudocount)));
    }
    ++global_output_prior_updates_;
}

StepStats SparseBranchMachine::step_token(std::uint32_t token,
                                          std::uint32_t target_token,
                                          bool learn) {
    return uses_sparse_token_output()
        ? step_token_sparse(token, target_token, learn)
        : step_token_dense(token, target_token, learn);
}

StepStats SparseBranchMachine::step_token_sparse(std::uint32_t token,
                                                 std::uint32_t target_token,
                                                 bool learn) {
    if (config_.objective != ObjectiveKind::TokenCrossEntropy) {
        throw std::logic_error("step_token requires the token cross-entropy objective");
    }
    if (token >= config_.token_alphabet || target_token >= config_.vector_dim) {
        throw std::out_of_range("invalid input or target token");
    }

    if (history_.size() == config_.context_width) {
        std::move(history_.begin() + 1, history_.end(), history_.begin());
        history_.back() = token;
    } else {
        history_.push_back(token);
    }
    maybe_begin_topology_probe(learn);
    const auto signatures = make_signatures(history_, learn);

    const bool can_grow = learn || config_.allow_growth_when_frozen;
    std::uint32_t created = 0U;
    const std::span<const float> empty_initial;
    for (std::uint8_t channel = 0; channel < signatures.size(); ++channel) {
        if (!channel_enabled(channel)) continue;
        const auto signature = signatures[channel];
        const auto exact_bucket = bucket_index(channel, signature);
        const auto* bucket_state = find_bucket(exact_bucket);
        const std::size_t exact_count = bucket_state == nullptr
            ? 0U : bucket_state->residents.size();
        NodeId nearest_exact = kInvalidNode;
        double nearest_similarity = -1.0;
        float nearest_persistent_loss = std::numeric_limits<float>::infinity();
        std::uint32_t nearest_visits = 0U;
        for (const NodeId id : bounded_bucket_nodes(
                 exact_bucket, config_.bucket_scan_limit)) {
            const auto slot = slot_of(id);
            if (slot == SIZE_MAX || channels_[slot] != channel) continue;
            const double similarity = hamming_similarity(prototypes_[slot], signature);
            if (similarity > nearest_similarity) {
                nearest_similarity = similarity;
                nearest_exact = id;
                nearest_visits = address_visits_[slot];
                nearest_persistent_loss = address_loss_ema_[slot];
            }
        }
        const bool cooldown_ready = total_steps_ >=
            (bucket_state == nullptr ? 0U : bucket_state->last_split_step) +
                config_.split_cooldown;
        const bool conflict_without_capacity = nearest_exact != kInvalidNode &&
            nearest_visits >= config_.split_min_visits &&
            cooldown_ready && nearest_similarity < config_.split_context_similarity &&
            nearest_persistent_loss > config_.split_loss_threshold;
        const bool persistent_conflict = conflict_without_capacity &&
            exact_count < config_.max_specializations_per_bucket;
        if (can_grow && channel_learning_enabled(channel) &&
            conflict_without_capacity &&
            exact_count >= config_.max_specializations_per_bucket) {
            ++address_capacity_blocked_splits_;
        }
        if (can_grow && channel_learning_enabled(channel) &&
            (exact_count == 0U || persistent_conflict)) {
            const NodeId id = new_node(signature, empty_initial, nearest_exact, channel);
            const auto slot = slot_of(id);
            if (slot != SIZE_MAX) {
                loss_ema_[slot] = 1.0F;
                address_loss_ema_[slot] = 1.0F;
                utility_ema_[slot] = 0.0F;
                mark_hot(id);
            }
            ensure_bucket(exact_bucket).last_split_step = total_steps_;
            ++created;
        }
    }

    auto [active, examined] = select_route(signatures, 2, learn);

    // Iterative refinement: if responsibility is not concentrated, expand the
    // neighbor radius and re-select. All rounds complete before the target is
    // used, preserving causal ordering (§4.5).
    if (config_.max_refinement_rounds > 0) {
        float max_responsibility = 0.0F;
        for (const auto& node : active) {
            max_responsibility = std::max(max_responsibility, node.responsibility);
        }
        for (std::uint32_t round = 0;
             round < config_.max_refinement_rounds &&
             max_responsibility < config_.refinement_confidence_threshold;
             ++round) {
            const std::int64_t radius = 2 + static_cast<std::int64_t>(round) + 1;
            auto [refined, examined_refined] = select_route(signatures, radius, learn);
            examined += examined_refined;
            active = refined;
            max_responsibility = 0.0F;
            for (const auto& node : active) {
                max_responsibility = std::max(max_responsibility,
                                              node.responsibility);
            }
        }
    }

    auto& output_tree = *implicit_output_;
    output_tree.target_path(target_token, token_path_scratch_);
    const float temperature = std::max(config_.softmax_temperature, 1e-5F);
    std::vector<float> path_logits;
    path_logits.reserve(token_path_scratch_.size());
    float cross_entropy = 0.0F;
    for (const auto& step : token_path_scratch_) {
        const float logit = global_output_logit(step.id) +
            aggregate_sparse_logit(active, step.id);
        path_logits.push_back(logit);
        cross_entropy += branch_loss(logit / temperature, step.right);
    }
    const float target_probability = std::exp(-std::min(cross_entropy, 80.0F));
    const float normalized_loss = cross_entropy /
        std::max(std::log(static_cast<float>(config_.vector_dim)), 1e-5F);

    bool gpaf_ablation_available = false;
    float gpaf_removed_cross_entropy = 0.0F;
    float gpaf_codelength_gain = 0.0F;
    float gpaf_false_positive_cost = 0.0F;
    std::uint32_t gpaf_ablation_nodes = 0U;
    std::uint8_t gpaf_ablation_key_count = 0U;
    std::array<std::uint64_t, kMaxGpafAblationKeys> gpaf_ablation_keys{};
    std::array<std::uint32_t, kMaxGpafAblationKeys> gpaf_ablation_key_nodes{};
    std::array<float, kMaxGpafAblationKeys> gpaf_ablation_key_removed_cross_entropy{};
    std::array<float, kMaxGpafAblationKeys> gpaf_ablation_key_gain{};
    std::array<float, kMaxGpafAblationKeys> gpaf_ablation_key_false_positive_cost{};
    gpaf_ablation_keys.fill(UINT64_MAX);
    if (!learn) {
        for (const auto& node : active) {
            if (node.source != CandidateSource::GpafRole ||
                node.gpaf_key == UINT64_MAX) {
                continue;
            }
            ++gpaf_ablation_nodes;
            std::size_t key_index = SIZE_MAX;
            for (std::size_t i = 0; i < gpaf_ablation_key_count; ++i) {
                if (gpaf_ablation_keys[i] == node.gpaf_key) {
                    key_index = i;
                    break;
                }
            }
            if (key_index == SIZE_MAX &&
                gpaf_ablation_key_count < kMaxGpafAblationKeys) {
                key_index = gpaf_ablation_key_count++;
                gpaf_ablation_keys[key_index] = node.gpaf_key;
            }
            if (key_index != SIZE_MAX) ++gpaf_ablation_key_nodes[key_index];
        }
        if (gpaf_ablation_nodes != 0U) {
            gpaf_ablation_available = true;
            std::size_t path_position = 0U;
            for (const auto& step : token_path_scratch_) {
                float removed = 0.0F;
                std::array<float, kMaxGpafAblationKeys> removed_by_key{};
                for (const auto& node : active) {
                    if (node.source != CandidateSource::GpafRole ||
                        node.gpaf_key == UINT64_MAX) {
                        continue;
                    }
                    const auto slot = slot_of(node.id);
                    if (slot == SIZE_MAX) continue;
                    const float node_logit =
                        node.responsibility * sparse_logit(slot, step.id);
                    removed += node_logit;
                    for (std::size_t i = 0; i < gpaf_ablation_key_count; ++i) {
                        if (gpaf_ablation_keys[i] == node.gpaf_key) {
                            removed_by_key[i] += node_logit;
                            break;
                        }
                    }
                }
                gpaf_removed_cross_entropy += branch_loss(
                    (path_logits[path_position] - removed) / temperature, step.right);
                for (std::size_t i = 0; i < gpaf_ablation_key_count; ++i) {
                    gpaf_ablation_key_removed_cross_entropy[i] += branch_loss(
                        (path_logits[path_position] - removed_by_key[i]) / temperature,
                        step.right);
                }
                ++path_position;
            }
            gpaf_codelength_gain = gpaf_removed_cross_entropy - cross_entropy;
            gpaf_false_positive_cost = std::max(0.0F, -gpaf_codelength_gain);
            for (std::size_t i = 0; i < gpaf_ablation_key_count; ++i) {
                gpaf_ablation_key_gain[i] =
                    gpaf_ablation_key_removed_cross_entropy[i] - cross_entropy;
                gpaf_ablation_key_false_positive_cost[i] =
                    std::max(0.0F, -gpaf_ablation_key_gain[i]);
            }
        }
    }

    const bool measure_channel_credit = learn || config_.record_channel_attribution;
    std::uint32_t seed_channel_mask = 0U;
    std::uint32_t active_channel_mask = 0U;
    std::uint32_t content_channel_mask = 0U;
    std::uint32_t tuple_channel_mask = 0U;
    std::array<float, kMaxAddressChannels> attribution_channel_mass{};
    std::array<std::uint32_t, kMaxAddressChannels> attribution_dependency{};
    std::array<std::uint8_t, kMaxAddressChannels> attribution_parent_channel{};
    std::array<std::uint8_t, kMaxAddressChannels> attribution_dependency_channel{};
    std::array<std::uint8_t, kMaxAddressChannels> attribution_input_state{};
    std::array<std::uint8_t, kMaxAddressChannels> attribution_output_state{};
    std::array<std::uint8_t, kMaxAddressChannels> attribution_required_dependency_binding{};
    std::array<float, kMaxAddressChannels> attribution_caller_removed_credit{};
    std::array<float, kMaxAddressChannels> attribution_dependency_retained_credit{};
    std::array<float, kMaxAddressChannels> attribution_dependency_removed_credit{};
    std::array<std::uint8_t, kMaxAddressChannels> attribution_binding_kind{};
    std::array<std::uint8_t, kMaxAddressChannels> attribution_binding_matched{};
    std::array<std::uint32_t, kMaxAddressChannels> attribution_binding_current_token{};
    std::array<std::uint32_t, kMaxAddressChannels> attribution_binding_matched_token{};
    std::array<std::uint32_t, kMaxAddressChannels> attribution_binding_successor{};
    std::array<std::uint32_t, kMaxAddressChannels> attribution_binding_distance{};
    std::array<std::uint32_t, kMaxAddressChannels> attribution_binding_pattern_span{};
    std::array<std::uint64_t, kMaxAddressChannels> attribution_binding_key{};
    std::array<std::uint64_t, kMaxAddressChannels> attribution_dependency_binding_key{};
    std::array<std::uint64_t, kMaxAddressChannels> attribution_call_key{};
    std::array<float, kMaxAddressChannels> attribution_description_cost{};
    std::array<float, kMaxAddressChannels> attribution_execution_cost{};
    attribution_parent_channel.fill(kInvalidChannel);
    attribution_dependency_channel.fill(kInvalidChannel);
    const bool measure_channel_subsets = config_.record_channel_attribution && !learn;
    if (measure_channel_subsets) {
        const auto frames = last_execution_frames();
        std::size_t frame_index = 0U;
        for (const auto& node : active) {
            if (node.channel >= kMaxAddressChannels) continue;
            attribution_channel_mass[node.channel] += node.responsibility;
        }
        for (std::size_t channel = 0U; channel < topology_.size() &&
             channel < kMaxAddressChannels; ++channel) {
            if (!channel_enabled(channel)) continue;
            if (frame_index < frames.size()) {
                const auto& frame = frames[frame_index];
                ++frame_index;
                attribution_dependency[channel] = frame.dependency;
                attribution_parent_channel[channel] = frame.parent_channel;
                attribution_dependency_channel[channel] = frame.dependency_channel;
                attribution_input_state[channel] =
                    static_cast<std::uint8_t>(frame.input_state);
                attribution_output_state[channel] =
                    static_cast<std::uint8_t>(frame.output_state);
                attribution_required_dependency_binding[channel] =
                    static_cast<std::uint8_t>(
                        frame.required_dependency_binding);
                attribution_binding_kind[channel] =
                    static_cast<std::uint8_t>(frame.binding);
                attribution_binding_matched[channel] =
                    frame.binding_state.matched ? 1U : 0U;
                attribution_binding_current_token[channel] =
                    frame.binding_state.current_token;
                attribution_binding_matched_token[channel] =
                    frame.binding_state.matched_token;
                attribution_binding_successor[channel] =
                    frame.binding_state.matched_successor;
                attribution_binding_distance[channel] =
                    frame.binding_state.matched_distance;
                attribution_binding_pattern_span[channel] =
                    frame.binding_state.pattern_span;
                attribution_binding_key[channel] =
                    frame.binding_state.binding_key;
                attribution_dependency_binding_key[channel] =
                    frame.dependency_binding_key;
                attribution_call_key[channel] = frame.call_key;
                attribution_description_cost[channel] = frame.description_cost;
                attribution_execution_cost[channel] = frame.execution_cost;
            }
            const auto bit = 1U << channel;
            if (channel == 0U) seed_channel_mask |= bit;
            if (topology_[channel].phase == ChannelPhase::Active) {
                active_channel_mask |= bit;
                const auto op = topology_[channel].program.op;
                if (op == AddressOp::ContentMatch || is_content_follow_op(op)) {
                    content_channel_mask |= bit;
                } else if (op == AddressOp::Tuple) {
                    tuple_channel_mask |= bit;
                }
            }
        }
    }

    const auto masked_loss = [&](std::uint32_t channel_mask) {
        float selected_mass = 0.0F;
        for (const auto& node : active) {
            if (node.channel < kMaxAddressChannels &&
                (channel_mask & (1U << node.channel)) != 0U) {
                selected_mass += node.responsibility;
            }
        }
        const float mass_scale = selected_mass > 1e-6F ? 1.0F / selected_mass : 1.0F;
        float loss = 0.0F;
        for (const auto& step : token_path_scratch_) {
            const float logit = global_output_logit(step.id) +
                aggregate_sparse_logit_masked(active, step.id, channel_mask, mass_scale);
            loss += branch_loss(logit / temperature, step.right);
        }
        return loss;
    };

    const auto removed_loss = [&](std::uint32_t removed_channel_mask) {
        float loss = 0.0F;
        std::size_t path_position = 0U;
        for (const auto& step : token_path_scratch_) {
            float removed = 0.0F;
            for (const auto& node : active) {
                if (node.channel >= kMaxAddressChannels ||
                    (removed_channel_mask & (1U << node.channel)) == 0U) {
                    continue;
                }
                const auto slot = slot_of(node.id);
                if (slot == SIZE_MAX) continue;
                removed += node.responsibility * sparse_logit(slot, step.id);
            }
            loss += branch_loss(
                (path_logits[path_position] - removed) / temperature, step.right);
            ++path_position;
        }
        return loss;
    };

    std::uint32_t predicted = 0U;
    bool top5 = false;
    const bool ranking_available = !learn || config_.decode_token_ranking_during_training;
    if (ranking_available) {
        struct SearchItem {
            float log_probability{};
            std::uint32_t lo{};
            std::uint32_t hi{};
        };
        std::vector<SearchItem> frontier{{0.0F, 0U, config_.vector_dim}};
        const auto decoder_beam = std::max(config_.sparse_output_topk,
                                           config_.sparse_output_beam_width);
        const auto maximum_depth = static_cast<std::uint32_t>(
            std::bit_width(config_.vector_dim - 1U) + 2U);
        for (std::uint32_t depth = 0U; depth < maximum_depth; ++depth) {
            bool all_leaves = true;
            std::vector<SearchItem> expanded;
            expanded.reserve(frontier.size() * 2U);
            for (const auto& item : frontier) {
                if (item.hi - item.lo == 1U) {
                    expanded.push_back(item);
                    continue;
                }
                all_leaves = false;
                const auto decision = output_tree.split(item.lo, item.hi);
                const float logit = (global_output_logit(decision.decision_id) +
                    aggregate_sparse_logit(active, decision.decision_id)) / temperature;
                expanded.push_back({item.log_probability +
                                        log_branch_probability(logit, false),
                                    item.lo, decision.middle});
                expanded.push_back({item.log_probability +
                                        log_branch_probability(logit, true),
                                    decision.middle, item.hi});
            }
            if (all_leaves) break;
            const auto keep = std::min<std::size_t>(decoder_beam, expanded.size());
            std::partial_sort(expanded.begin(), expanded.begin() + keep, expanded.end(),
                              [](const SearchItem& left, const SearchItem& right) {
                                  return left.log_probability > right.log_probability;
                              });
            expanded.resize(keep);
            frontier = std::move(expanded);
        }
        std::sort(frontier.begin(), frontier.end(),
                  [](const SearchItem& left, const SearchItem& right) {
                      return left.log_probability > right.log_probability;
                  });
        std::vector<std::uint32_t> top_tokens;
        top_tokens.reserve(config_.sparse_output_topk);
        for (const auto& item : frontier) {
            if (item.hi - item.lo != 1U) continue;
            top_tokens.push_back(output_tree.token_from_rank(item.lo));
            if (top_tokens.size() >= config_.sparse_output_topk) break;
        }

        predicted = top_tokens.empty() ? 0U : top_tokens.front();
        top5 = std::find(top_tokens.begin(), top_tokens.end(), target_token) !=
               top_tokens.end();
    }

    if (measure_channel_credit) {
        std::fill(channel_credit_buffer_.begin(), channel_credit_buffer_.end(), 0.0F);
        std::array<std::uint8_t, kMaxAddressChannels> channel_members{};
        for (const auto& node : active) {
            if (node.channel < channel_members.size() &&
                std::abs(node.responsibility) >= 1e-7F) {
                ++channel_members[node.channel];
            }
        }

        for (std::size_t channel = 0; channel < topology_.size(); ++channel) {
            if (!channel_enabled(channel) || channel_members[channel] == 0U) continue;
            float without_loss = 0.0F;
            std::size_t path_position = 0U;
            for (const auto& step : token_path_scratch_) {
                float removed = 0.0F;
                for (const auto& node : active) {
                    if (node.channel != channel) continue;
                    const auto slot = slot_of(node.id);
                    if (slot == SIZE_MAX) continue;
                    removed += node.responsibility * sparse_logit(slot, step.id);
                }
                without_loss += branch_loss(
                    (path_logits[path_position] - removed) / temperature, step.right);
                ++path_position;
            }
            channel_credit_buffer_[channel] = without_loss - cross_entropy;
        }
        if (measure_channel_subsets) {
            for (std::size_t channel = 0U; channel < topology_.size() &&
                 channel < kMaxAddressChannels; ++channel) {
                if (!channel_enabled(channel)) continue;
                const auto caller_mask = 1U << channel;
                const float caller_removed =
                    removed_loss(caller_mask) - cross_entropy;
                attribution_caller_removed_credit[channel] = caller_removed;
                attribution_dependency_retained_credit[channel] = caller_removed;
                const auto dependency_channel = attribution_dependency_channel[channel];
                if (dependency_channel != kInvalidChannel &&
                    dependency_channel < kMaxAddressChannels &&
                    dependency_channel != channel) {
                    const auto joint_mask = caller_mask | (1U << dependency_channel);
                    attribution_dependency_removed_credit[channel] =
                        removed_loss(joint_mask) - cross_entropy;
                }
            }
        }

        for (auto& node : active) {
            const auto slot = slot_of(node.id);
            if (slot == SIZE_MAX || std::abs(node.responsibility) < 1e-7F) {
                node.contribution = 0.0F;
                continue;
            }
            if (channel_members[node.channel] == 1U) {
                node.contribution = channel_credit_buffer_[node.channel];
                continue;
            }
            float without_loss = 0.0F;
            std::size_t path_position = 0U;
            for (const auto& step : token_path_scratch_) {
                const float removed = node.responsibility *
                    sparse_logit(slot, step.id);
                without_loss += branch_loss(
                    (path_logits[path_position] - removed) / temperature, step.right);
                ++path_position;
            }
            node.contribution = without_loss - cross_entropy;
        }
        if (learn) {
            observe_topology_credit(std::span<const float>(channel_credit_buffer_.data(),
                                                            topology_.size()));
        }
    }

    std::vector<NodeId> route;
    std::vector<float> contributions;
    route.reserve(active.size());
    contributions.reserve(active.size());
    for (const auto& node : active) {
        route.push_back(node.id);
        contributions.push_back(node.contribution);
    }

    if (learn) observe_gpaf_shadow_roles(active);
    if (learn) observe_global_output_path(token_path_scratch_);

    if (learn) {
        apply_trace_credit(normalized_loss);
        std::array<float, kMaxAddressChannels> channel_responsibility_mass{};
        for (const auto& node : active) {
            if (node.channel < channel_responsibility_mass.size() &&
                std::abs(node.responsibility) >= 1e-7F) {
                channel_responsibility_mass[node.channel] += node.responsibility;
            }
        }
        for (const auto& node : active) {
            if (!node.exact_region ||
                node.responsibility < config_.min_update_responsibility) continue;
            const auto slot = slot_of(node.id);
            if (slot == SIZE_MAX || !channel_learning_enabled(node.channel)) continue;
            if (visits_[slot] == 0U && !hot_indexed_[slot]) {
                mark_hot(ids_[slot]);
            }
            ++visits_[slot];
            ++address_visits_[slot];
            std::size_t path_position = 0U;
            for (const auto& step : token_path_scratch_) {
                const float probability_right = sigmoid(path_logits[path_position] / temperature);
                const float target_right = step.right
                    ? 1.0F - 0.5F * config_.label_smoothing
                    : 0.5F * config_.label_smoothing;
                auto* entry_pointer = mutable_sparse_entry(slot, step.id);
                if (entry_pointer == nullptr) {
                    ++path_position;
                    continue;
                }
                auto& entry = *entry_pointer;
                const float removed = node.responsibility * entry.logit;
                const float without_loss = branch_loss(
                    (path_logits[path_position] - removed) / temperature, step.right);
                const float full_branch_loss = branch_loss(
                    path_logits[path_position] / temperature, step.right);
                const float gain = std::clamp(
                    without_loss - full_branch_loss, -1.0F, 1.0F);
                const float channel_mass = node.channel < channel_responsibility_mass.size()
                    ? channel_responsibility_mass[node.channel] : node.responsibility;
                const float within_channel_responsibility = std::clamp(
                    node.responsibility / std::max(channel_mass, 1e-6F), 0.0F, 1.0F);
                const float rate = detail::sparse_decision_learning_rate(
                    entry,
                    config_.classification_learning_rate,
                    config_.classification_mature_learning_rate,
                    config_.mature_visits) * within_channel_responsibility / temperature;
                const float grad = target_right - probability_right;
                if (config_.use_momentum) {
                    entry.momentum = config_.momentum_beta1 * entry.momentum +
                        (1.0F - config_.momentum_beta1) * grad;
                    entry.variance = config_.momentum_beta2 * entry.variance +
                        (1.0F - config_.momentum_beta2) * grad * grad;
                    // Adam bias correction: first few updates would otherwise
                    // have an inflated effective learning rate.
                    const std::uint32_t step_count = entry.visits + 1U;
                    const float bias_correction1 =
                        1.0F - std::pow(config_.momentum_beta1,
                                        static_cast<float>(step_count));
                    const float bias_correction2 =
                        1.0F - std::pow(config_.momentum_beta2,
                                        static_cast<float>(step_count));
                    const float m_hat = entry.momentum /
                        std::max(bias_correction1, 1e-8F);
                    const float v_hat = entry.variance /
                        std::max(bias_correction2, 1e-8F);
                    const float adaptive_rate = rate /
                        (std::sqrt(v_hat) + config_.momentum_eps);
                    entry.logit = (1.0F - config_.logit_decay) * entry.logit +
                        adaptive_rate * m_hat;
                } else {
                    entry.logit = (1.0F - config_.logit_decay) * entry.logit +
                        rate * grad;
                }
                entry.gain_ema = 0.99F * entry.gain_ema + 0.01F * gain;
                if (entry.visits != std::numeric_limits<std::uint32_t>::max()) {
                    ++entry.visits;
                }
                entry.last_update_step = total_steps_;
                ++path_position;
            }
            loss_ema_[slot] = 0.96F * loss_ema_[slot] + 0.04F * normalized_loss;
            address_loss_ema_[slot] = 0.96F * address_loss_ema_[slot] +
                                      0.04F * normalized_loss;
            utility_ema_[slot] = 0.985F * utility_ema_[slot] +
                0.015F * std::clamp(node.contribution, -1.0F, 1.0F);
            update_phase(slot);
        }

        const std::size_t source_limit = std::min<std::size_t>(
            previous_route_.size(), config_.edge_reinforce_width);
        const std::size_t destination_limit = std::min<std::size_t>(
            active.size(), config_.edge_reinforce_width);
        for (std::size_t source_index = 0; source_index < source_limit; ++source_index) {
            decay_edges(previous_route_[source_index]);
            const float source_responsibility = source_index < previous_responsibilities_.size()
                ? previous_responsibilities_[source_index] : 1.0F;
            for (std::size_t destination_index = 0;
                 destination_index < destination_limit; ++destination_index) {
                const float helpful = std::max(0.0F, active[destination_index].contribution);
                if (helpful < config_.edge_min_contribution) continue;
                reinforce_edge(previous_route_[source_index],
                               active[destination_index].id,
                               source_responsibility *
                                   active[destination_index].responsibility * helpful);
            }
        }
        trace_.push_back({route, contributions, normalized_loss});
        while (trace_.size() > config_.trace_horizon) trace_.pop_front();
    }

    previous_route_ = route;
    previous_responsibilities_.clear();
    previous_responsibilities_.reserve(active.size());
    for (const auto& node : active) {
        previous_responsibilities_.push_back(node.responsibility);
    }
    ++total_steps_;
    total_candidates_ += examined;
    total_active_ += route.size();

    StepStats stats;
    stats.normalized_mse = normalized_loss;
    stats.active_nodes = static_cast<std::uint32_t>(route.size());
    stats.candidates_examined = examined;
    stats.live_nodes = static_cast<std::uint32_t>(ids_.size());
    stats.created = created;
    stats.route = std::move(route);
    stats.cross_entropy = cross_entropy;
    stats.target_probability = target_probability;
    stats.gpaf_ablation_available = gpaf_ablation_available;
    stats.gpaf_removed_cross_entropy = gpaf_removed_cross_entropy;
    stats.gpaf_codelength_gain = gpaf_codelength_gain;
    stats.gpaf_false_positive_cost = gpaf_false_positive_cost;
    stats.gpaf_ablation_nodes = gpaf_ablation_nodes;
    stats.gpaf_ablation_key_count = gpaf_ablation_key_count;
    stats.gpaf_ablation_keys = gpaf_ablation_keys;
    stats.gpaf_ablation_key_nodes = gpaf_ablation_key_nodes;
    stats.gpaf_ablation_key_removed_cross_entropy =
        gpaf_ablation_key_removed_cross_entropy;
    stats.gpaf_ablation_key_gain = gpaf_ablation_key_gain;
    stats.gpaf_ablation_key_false_positive_cost =
        gpaf_ablation_key_false_positive_cost;
    stats.top1_correct = ranking_available && predicted == target_token;
    stats.top5_correct = ranking_available && top5;
    stats.ranking_available = ranking_available;
    stats.predicted_token = predicted;
    if (measure_channel_subsets) {
        stats.channel_credit_count = static_cast<std::uint8_t>(
            std::min<std::size_t>(topology_.size(), kMaxAddressChannels));
        for (std::size_t channel = 0U; channel < stats.channel_credit_count; ++channel) {
            stats.channel_credit[channel] = channel_credit_buffer_[channel];
            stats.channel_responsibility_mass[channel] =
                attribution_channel_mass[channel];
            stats.channel_dependency[channel] = attribution_dependency[channel];
            stats.channel_parent_channel[channel] = attribution_parent_channel[channel];
            stats.channel_dependency_channel[channel] =
                attribution_dependency_channel[channel];
            stats.channel_input_state[channel] = attribution_input_state[channel];
            stats.channel_output_state[channel] = attribution_output_state[channel];
            stats.channel_required_dependency_binding[channel] =
                attribution_required_dependency_binding[channel];
            stats.channel_caller_removed_credit[channel] =
                attribution_caller_removed_credit[channel];
            stats.channel_dependency_retained_credit[channel] =
                attribution_dependency_retained_credit[channel];
            stats.channel_dependency_removed_credit[channel] =
                attribution_dependency_removed_credit[channel];
            stats.channel_binding_kind[channel] = attribution_binding_kind[channel];
            stats.channel_binding_matched[channel] =
                attribution_binding_matched[channel];
            stats.channel_binding_current_token[channel] =
                attribution_binding_current_token[channel];
            stats.channel_binding_matched_token[channel] =
                attribution_binding_matched_token[channel];
            stats.channel_binding_successor[channel] =
                attribution_binding_successor[channel];
            stats.channel_binding_distance[channel] =
                attribution_binding_distance[channel];
            stats.channel_binding_pattern_span[channel] =
                attribution_binding_pattern_span[channel];
            stats.channel_binding_key[channel] =
                attribution_binding_key[channel];
            stats.channel_dependency_binding_key[channel] =
                attribution_dependency_binding_key[channel];
            stats.channel_call_key[channel] = attribution_call_key[channel];
            stats.channel_description_cost[channel] =
                attribution_description_cost[channel];
            stats.channel_execution_cost[channel] = attribution_execution_cost[channel];
        }
        stats.channel_subset_available = true;
        stats.seed_only_cross_entropy = masked_loss(seed_channel_mask);
        stats.active_only_cross_entropy = active_channel_mask == 0U
            ? cross_entropy : masked_loss(active_channel_mask);
        stats.content_only_cross_entropy = content_channel_mask == 0U
            ? cross_entropy : masked_loss(content_channel_mask);
        stats.tuple_only_cross_entropy = tuple_channel_mask == 0U
            ? cross_entropy : masked_loss(tuple_channel_mask);
    }
    return stats;
}

} // namespace sbm
