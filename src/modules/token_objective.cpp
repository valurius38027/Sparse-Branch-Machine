#include "sbm/machine.hpp"

#include "sbm/detail/vector_ops.hpp"
#include "sbm/math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sbm {
namespace {

void softmax(std::span<const float> logits, std::span<float> probabilities,
             float temperature) {
    const float inverse_temperature = 1.0F / temperature;
    float maximum = -std::numeric_limits<float>::infinity();
    for (const float value : logits) maximum = std::max(maximum, value * inverse_temperature);
    double normalizer = 0.0;
    for (std::size_t index = 0; index < logits.size(); ++index) {
        probabilities[index] = std::exp(logits[index] * inverse_temperature - maximum);
        normalizer += probabilities[index];
    }
    const float inverse = static_cast<float>(1.0 / std::max(normalizer, 1e-30));
    for (auto& value : probabilities) value *= inverse;
}

float ablated_cross_entropy(std::span<const float> logits,
                            const float* removed,
                            float removed_scale,
                            std::uint32_t target,
                            float temperature) {
    const float inverse_temperature = 1.0F / temperature;
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t index = 0; index < logits.size(); ++index) {
        const float value = (logits[index] - removed_scale * removed[index]) *
                            inverse_temperature;
        maximum = std::max(maximum, value);
    }
    double normalizer = 0.0;
    for (std::size_t index = 0; index < logits.size(); ++index) {
        const float value = (logits[index] - removed_scale * removed[index]) *
                            inverse_temperature;
        normalizer += std::exp(value - maximum);
    }
    const float target_logit =
        (logits[target] - removed_scale * removed[target]) * inverse_temperature;
    return maximum + static_cast<float>(std::log(std::max(normalizer, 1e-30))) -
           target_logit;
}

float centered_logit_mean(std::span<const float> logits) {
    double total = 0.0;
    for (const float value : logits) total += value;
    return static_cast<float>(total / static_cast<double>(logits.size()));
}

} // namespace

StepStats SparseBranchMachine::step_token_dense(std::uint32_t token,
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
    const std::span<const float> zero_logits(zero_output_buffer_.data(),
                                             config_.vector_dim);

    // Address allocation depends only on the observed context, never on the
    // target token.  New residents therefore make a uniform prediction on
    // their first use instead of leaking the label into the current metric.
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
            cooldown_ready &&
            nearest_similarity < config_.split_context_similarity &&
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
            const NodeId id = new_node(signature, zero_logits, nearest_exact, channel);
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
    aggregate(active, logit_buffer_);
    softmax(logit_buffer_, prediction_buffer_, config_.softmax_temperature);

    const float target_probability = std::max(prediction_buffer_[target_token], 1e-12F);
    const float cross_entropy = -std::log(target_probability);
    const float normalized_loss = cross_entropy /
        std::max(std::log(static_cast<float>(config_.vector_dim)), 1e-5F);
    const auto predicted = static_cast<std::uint32_t>(std::distance(
        prediction_buffer_.begin(),
        std::max_element(prediction_buffer_.begin(), prediction_buffer_.end())));

    const bool measure_channel_credit = learn || config_.record_channel_attribution;
    if (measure_channel_credit) {
        // Counterfactual credit is computed from log-sum-exp directly.  This
        // avoids allocating and normalizing a complete probability vector for
        // every node and every address program.
        const std::size_t channel_values = topology_.size() * config_.vector_dim;
        std::fill(channel_output_buffer_.begin(),
                  channel_output_buffer_.begin() + static_cast<std::ptrdiff_t>(channel_values),
                  0.0F);
        for (const auto& node : active) {
            const auto slot = slot_of(node.id);
            if (slot == SIZE_MAX || std::abs(node.responsibility) < 1e-7F) continue;
            detail::axpy(channel_output_buffer_.data() +
                             static_cast<std::size_t>(node.channel) * config_.vector_dim,
                         output_vectors_.data() + slot * config_.vector_dim,
                         node.responsibility, config_.vector_dim);
        }

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
            const float* removed = channel_output_buffer_.data() +
                channel * config_.vector_dim;
            const float without_loss = ablated_cross_entropy(
                logit_buffer_, removed, 1.0F, target_token,
                config_.softmax_temperature);
            channel_credit_buffer_[channel] = without_loss - cross_entropy;
        }

        // A channel containing one active node has exactly the same node and
        // channel ablation.  Reusing the exact channel result avoids another
        // full-vocabulary log-sum-exp sweep without changing the credit rule.
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
            const float without_loss = ablated_cross_entropy(
                logit_buffer_, output_vectors_.data() + slot * config_.vector_dim,
                node.responsibility, target_token, config_.softmax_temperature);
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

    if (learn) {
        apply_trace_credit(normalized_loss);
        std::array<float, kMaxAddressChannels> channel_responsibility_mass{};
        for (const auto& node : active) {
            if (node.channel < channel_responsibility_mass.size() &&
                std::abs(node.responsibility) >= 1e-7F) {
                channel_responsibility_mass[node.channel] += node.responsibility;
            }
        }
        const float smoothing = config_.label_smoothing;
        const float off_target = config_.vector_dim > 1U
            ? smoothing / static_cast<float>(config_.vector_dim - 1U)
            : 0.0F;

        for (const auto& node : active) {
            if (!node.exact_region ||
                node.responsibility < config_.min_update_responsibility) {
                continue;
            }
            const auto slot = slot_of(node.id);
            if (slot == SIZE_MAX || !channel_learning_enabled(node.channel)) continue;
            if (visits_[slot] == 0U && !hot_indexed_[slot]) {
                mark_hot(ids_[slot]);
            }
            ++visits_[slot];
            ++address_visits_[slot];
            const bool mature = phase_of_slot(slot) == NodePhase::Mature;
            const float base_rate = mature
                ? config_.classification_mature_learning_rate
                : config_.classification_learning_rate;
            const float schedule = 1.0F /
                std::sqrt(static_cast<float>(std::max(1U, address_visits_[slot])));
            const float channel_mass = node.channel < channel_responsibility_mass.size()
                ? channel_responsibility_mass[node.channel] : node.responsibility;
            const float within_channel_responsibility = std::clamp(
                node.responsibility / std::max(channel_mass, 1e-6F), 0.0F, 1.0F);
            const float rate = base_rate * schedule * within_channel_responsibility /
                std::max(config_.softmax_temperature, 1e-5F);
            float* logits = output_vectors_.data() + slot * config_.vector_dim;
            const float retain = 1.0F - config_.logit_decay;
            // The uniform part of label smoothing adds the same value to every
            // logit and therefore cannot change softmax probabilities.  Drop
            // that null-space component and update the dense negative gradient
            // through the SIMD kernel.
            detail::scale_axpy(logits, prediction_buffer_.data(), retain, -rate,
                               config_.vector_dim);
            logits[target_token] += rate * (1.0F - smoothing - off_target);

            // The update is centered analytically; recenter only occasionally
            // to remove accumulated floating-point drift.
            if ((address_visits_[slot] & 255U) == 0U) {
                const float mean = centered_logit_mean(
                    std::span<const float>(logits, config_.vector_dim));
                for (std::uint32_t candidate = 0; candidate < config_.vector_dim;
                     ++candidate) {
                    logits[candidate] -= mean;
                }
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
                ? previous_responsibilities_[source_index]
                : 1.0F;
            for (std::size_t destination_index = 0;
                 destination_index < destination_limit;
                 ++destination_index) {
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
    stats.cosine = 0.0F;
    stats.active_nodes = static_cast<std::uint32_t>(route.size());
    stats.candidates_examined = examined;
    stats.live_nodes = static_cast<std::uint32_t>(ids_.size());
    stats.created = created;
    stats.route = std::move(route);
    stats.cross_entropy = cross_entropy;
    stats.target_probability = target_probability;
    stats.top1_correct = predicted == target_token;
    std::size_t better = 0U;
    for (std::size_t candidate = 0; candidate < prediction_buffer_.size(); ++candidate) {
        if (candidate != target_token &&
            prediction_buffer_[candidate] > target_probability) ++better;
    }
    stats.top5_correct = better < std::min<std::size_t>(5U, prediction_buffer_.size());
    stats.predicted_token = predicted;
    if (measure_channel_credit) {
        stats.channel_credit_count = static_cast<std::uint8_t>(
            std::min<std::size_t>(topology_.size(), kMaxAddressChannels));
        for (std::size_t channel = 0U; channel < stats.channel_credit_count; ++channel) {
            stats.channel_credit[channel] = channel_credit_buffer_[channel];
        }
    }
    return stats;
}

} // namespace sbm
