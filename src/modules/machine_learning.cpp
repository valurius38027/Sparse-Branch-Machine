#include "sbm/machine.hpp"

#include "sbm/detail/vector_ops.hpp"
#include "sbm/math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sbm {

void SparseBranchMachine::reinforce_edge(NodeId source, NodeId destination, float delta) {
    const auto slot = slot_of(source);
    if (slot == SIZE_MAX || !contains(destination) || delta <= 0.0F) return;
    auto& list = edges_[slot];
    const auto found = std::find_if(list.begin(), list.end(), [&](const Edge& edge) {
        return edge.dst == destination;
    });
    if (found == list.end()) {
        list.push_back({destination, std::min(delta, 1.0F), delta});
    } else {
        found->eligibility = config_.trace_decay * found->eligibility + delta;
        const float target = std::min(found->eligibility, 4.0F);
        found->weight = config_.edge_decay * found->weight +
                        config_.edge_learning_rate * (target - found->weight);
        found->weight = std::clamp(found->weight, 0.0F, 4.0F);
    }
    std::sort(list.begin(), list.end(), [](const Edge& a, const Edge& b) {
        return a.weight == b.weight ? a.dst < b.dst : a.weight > b.weight;
    });
    if (list.size() > config_.max_edges_per_node) {
        list.resize(config_.max_edges_per_node);
    }
}

void SparseBranchMachine::decay_edges(NodeId source) {
    const auto slot = slot_of(source);
    if (slot == SIZE_MAX) return;
    auto& list = edges_[slot];
    for (auto& edge : list) {
        edge.weight *= config_.edge_decay;
        edge.eligibility *= config_.trace_decay;
    }
    list.erase(std::remove_if(list.begin(), list.end(), [](const Edge& edge) {
        return edge.weight < 1e-4F && edge.eligibility < 1e-4F;
    }), list.end());
}

void SparseBranchMachine::update_phase(std::size_t slot) {
    NodePhase phase = NodePhase::Cold;
    if (utility_ema_[slot] <= config_.dormant_utility) phase = NodePhase::Dormant;
    else if (visits_[slot] >= config_.mature_visits) phase = NodePhase::Mature;
    else if (visits_[slot] >= config_.warm_visits) phase = NodePhase::Warm;
    phases_[slot] = static_cast<std::uint8_t>(phase);
}

void SparseBranchMachine::apply_trace_credit(float loss) {
    const float global_credit = 1.0F - std::min(loss, 2.0F);
    float decay = 1.0F;
    for (auto frame = trace_.rbegin(); frame != trace_.rend(); ++frame) {
        for (std::size_t index = 0; index < frame->route.size(); ++index) {
            const auto slot = slot_of(frame->route[index]);
            if (slot == SIZE_MAX) continue;
            const float local = index < frame->contribution.size()
                ? std::clamp(frame->contribution[index], -1.0F, 1.0F)
                : 0.0F;
            utility_ema_[slot] = 0.97F * utility_ema_[slot] +
                0.03F * (0.65F * global_credit + 0.35F * local) * decay;
            update_phase(slot);
        }
        decay *= config_.trace_decay;
    }
}

StepStats SparseBranchMachine::step(std::uint32_t token,
                                    std::span<const float> target,
                                    bool learn) {
    if (config_.objective != ObjectiveKind::VectorRegression) {
        throw std::logic_error("step requires the vector-regression objective");
    }
    if (token >= config_.token_alphabet || target.size() != config_.vector_dim) {
        throw std::out_of_range("invalid token or target dimension");
    }

    if (history_.size() == config_.context_width) {
        std::move(history_.begin() + 1, history_.end(), history_.begin());
        history_.back() = token;
    } else {
        history_.push_back(token);
    }
    maybe_begin_topology_probe(learn);
    const auto signatures = make_signatures(history_, learn);
    auto [active, examined] = select_route(signatures, 2, learn);

    // Prediction and all metrics are fixed before the target can modify any
    // node, including newly allocated address residents.
    aggregate(active, prediction_buffer_);
    const float dimension = static_cast<float>(target.size());
    const float target_energy = detail::dot(target.data(), target.data(), target.size()) /
                                dimension;
    const float mse = detail::squared_distance(prediction_buffer_.data(), target.data(),
                                               target.size()) / dimension;
    const float normalized_mse = mse / std::max(target_energy, 1e-5F);
    const float cosine = detail::cosine(prediction_buffer_, target);
    if (learn) {
        compute_counterfactual_contributions(active, target, target_energy, normalized_mse);
    }

    const bool can_grow = learn || config_.allow_growth_when_frozen;
    std::uint32_t created = 0;
    std::vector<float> creation_prefix(config_.vector_dim, 0.0F);
    std::vector<float> initial_value(config_.vector_dim, 0.0F);

    // Allocate each temporal view as a sequential residual stage.  New nodes
    // receive the first unbiased sample of the quantity they are responsible
    // for, rather than starting residual channels at zero.
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
        std::uint32_t nearest_visits = 0;
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
            nearest_persistent_loss > config_.split_loss_threshold &&
            normalized_mse > config_.split_loss_threshold;
        const bool persistent_conflict = conflict_without_capacity &&
            exact_count < config_.max_specializations_per_bucket;
        if (can_grow && channel_learning_enabled(channel) &&
            conflict_without_capacity &&
            exact_count >= config_.max_specializations_per_bucket) {
            ++address_capacity_blocked_splits_;
        }

        NodeId representative = nearest_exact;
        if (can_grow && channel_learning_enabled(channel) &&
            (exact_count == 0 || persistent_conflict)) {
            const float channel_mass = channel == 0U ? 1.0F : config_.residual_channel_gain;
            for (std::uint32_t component = 0; component < config_.vector_dim; ++component) {
                initial_value[component] =
                    (target[component] - creation_prefix[component]) /
                    std::max(channel_mass, 1e-5F);
            }
            const NodeId id = new_node(signature, initial_value, nearest_exact, channel);
            const auto slot = slot_of(id);
            if (slot != SIZE_MAX) {
                visits_[slot] = 1;
                address_visits_[slot] = 1;
                loss_ema_[slot] = normalized_mse;
                address_loss_ema_[slot] = normalized_mse;
                utility_ema_[slot] = 0.0F;
                mark_hot(id);
                update_phase(slot);
            }
            ensure_bucket(exact_bucket).last_split_step = total_steps_;
            representative = id;
            ++created;
        }

        const auto representative_slot = slot_of(representative);
        if (representative_slot != SIZE_MAX) {
            const float channel_mass = channel == 0U ? 1.0F : config_.residual_channel_gain;
            detail::axpy(creation_prefix.data(),
                         output_vectors_.data() + representative_slot * config_.vector_dim,
                         channel_mass,
                         config_.vector_dim);
        }
    }

    if (learn) {
        std::vector<float> channel_credit(signatures.size(), 0.0F);
        std::vector<float> without_prediction(config_.vector_dim, 0.0F);
        for (std::size_t channel = 0; channel < signatures.size(); ++channel) {
            if (!channel_enabled(channel)) continue;
            std::copy(prediction_buffer_.begin(), prediction_buffer_.end(),
                      without_prediction.begin());
            bool removed = false;
            for (const auto& node : active) {
                if (node.channel != channel || std::abs(node.responsibility) < 1e-7F) continue;
                const auto slot = slot_of(node.id);
                if (slot == SIZE_MAX) continue;
                detail::axpy(without_prediction.data(),
                             output_vectors_.data() + slot * config_.vector_dim,
                             -node.responsibility, config_.vector_dim);
                removed = true;
            }
            if (!removed) continue;
            const float without_mse = detail::squared_distance(
                without_prediction.data(), target.data(), target.size()) / dimension;
            const float without_normalized = without_mse / std::max(target_energy, 1e-5F);
            channel_credit[channel] = without_normalized - normalized_mse;
        }
        observe_topology_credit(channel_credit);
    }

    std::vector<NodeId> route;
    std::vector<float> contributions;
    route.reserve(active.size());
    contributions.reserve(active.size());
    for (const auto& node : active) {
        route.push_back(node.id);
        contributions.push_back(node.contribution);
    }

    if (learn) {
        apply_trace_credit(normalized_mse);
        std::vector<float> error(config_.vector_dim, 0.0F);
        for (std::uint32_t component = 0; component < config_.vector_dim; ++component) {
            error[component] = target[component] - prediction_buffer_[component];
        }

        // Exact residents form an ordered additive decomposition.  Each
        // address stores an online mean of its own target/residual, using 1/n
        // rather than a fixed EMA that is biased when a bucket has few samples.
        std::vector<float> prefix(config_.vector_dim, 0.0F);
        std::vector<float> desired(config_.vector_dim, 0.0F);
        for (std::uint8_t channel = 0; channel < signatures.size(); ++channel) {
            const float channel_mass = channel == 0U ? 1.0F : config_.residual_channel_gain;
            std::vector<const ScoredNode*> exact_nodes;
            for (const auto& node : active) {
                if (node.channel == channel && node.exact_region &&
                    node.responsibility >= config_.min_update_responsibility) {
                    exact_nodes.push_back(&node);
                }
            }
            for (std::uint32_t component = 0; component < config_.vector_dim; ++component) {
                desired[component] = (target[component] - prefix[component]) /
                                     std::max(channel_mass, 1e-5F);
            }
            for (const auto* node : exact_nodes) {
                const auto slot = slot_of(node->id);
                if (slot == SIZE_MAX || !channel_learning_enabled(node->channel)) continue;
                if (visits_[slot] == 0 && !hot_indexed_[slot]) {
                    mark_hot(ids_[slot]);
                }
                ++visits_[slot];
                ++address_visits_[slot];
                const float within_channel = std::clamp(
                    node->responsibility / std::max(channel_mass, 1e-5F),
                    0.0F, 1.0F);
                const float count = static_cast<float>(std::max(1U, address_visits_[slot]));
                const float pseudo = std::max(0.0F, config_.residual_recency_pseudocount);
                const float schedule = channel == 0U ? (1.0F / count)
                    : ((1.0F + pseudo) / (count + pseudo));
                const float mean_rate = within_channel * schedule;
                detail::lerp(output_vectors_.data() + slot * config_.vector_dim,
                             desired.data(), mean_rate, config_.vector_dim);
                loss_ema_[slot] = 0.96F * loss_ema_[slot] +
                                  0.04F * normalized_mse;
                address_loss_ema_[slot] = 0.96F * address_loss_ema_[slot] +
                                          0.04F * normalized_mse;
                utility_ema_[slot] = 0.985F * utility_ema_[slot] +
                    0.015F * std::clamp(node->contribution, -1.0F, 1.0F);
                update_phase(slot);
            }

            // Later residual stages observe the current post-update estimate of
            // all earlier stages, matching their inference-time composition.
            for (const auto* node : exact_nodes) {
                const auto slot = slot_of(node->id);
                if (slot == SIZE_MAX) continue;
                detail::axpy(prefix.data(),
                             output_vectors_.data() + slot * config_.vector_dim,
                             node->responsibility,
                             config_.vector_dim);
            }
        }

        // Non-exact candidates remain read-only.  Their local vector belongs
        // to another address context; only the scalar control edge may learn
        // from a helpful counterfactual contribution.

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
                const float delta = source_responsibility *
                    active[destination_index].responsibility * helpful;
                reinforce_edge(previous_route_[source_index],
                               active[destination_index].id,
                               delta);
            }
        }
        trace_.push_back({route, contributions, normalized_mse});
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
    return {normalized_mse, cosine, static_cast<std::uint32_t>(route.size()),
            examined, static_cast<std::uint32_t>(ids_.size()), created,
            std::move(route)};
}

} // namespace sbm
