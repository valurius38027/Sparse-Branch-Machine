#include "sbm/machine.hpp"

#include "sbm/detail/address_interpreter.hpp"
#include "sbm/math.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>

namespace sbm {
namespace {

std::optional<AddressProgram> proposal_at(std::uint64_t ordinal,
                                          std::uint32_t max_lag,
                                          std::uint32_t max_arity,
                                          bool enable_delta,
                                          bool enable_content_match,
                                          bool enable_content_follow_multi) {
    for (std::uint32_t outer = 2U; outer <= max_lag; ++outer) {
        const auto visit = [&](AddressOp op) -> std::optional<AddressProgram> {
            AddressProgram singleton = singleton_address_program(outer);
            singleton.op = op;
            if (ordinal == 0U) return singleton;
            --ordinal;
            if (max_arity < 2U) return std::nullopt;
            for (std::uint32_t inner = 1U; inner < outer; ++inner) {
                AddressProgram program;
                program.lags[0] = inner;
                program.lags[1] = outer;
                program.arity = 2U;
                program.op = op;
                if (ordinal == 0U) return program;
                --ordinal;
            }
            return std::nullopt;
        };
        if (auto value = visit(AddressOp::Tuple); value.has_value()) return value;
        if (enable_delta) {
            if (auto value = visit(AddressOp::DeltaMod); value.has_value()) return value;
        }
        if (enable_content_match) {
            AddressProgram content = singleton_address_program(outer);
            content.op = AddressOp::ContentMatch;
            if (ordinal == 0U) return content;
            --ordinal;
            if (max_arity >= 2U) {
                for (std::uint32_t inner = 1U; inner < outer; ++inner) {
                    AddressProgram follow;
                    follow.lags[0] = inner;
                    follow.lags[1] = outer;
                    follow.arity = 2U;
                    follow.op = AddressOp::ContentFollow;
                    if (ordinal == 0U) return follow;
                    --ordinal;
                    if (enable_content_follow_multi) {
                        AddressProgram follow2;
                        follow2.lags[0] = inner;
                        follow2.lags[1] = outer;
                        follow2.arity = 2U;
                        follow2.op = AddressOp::ContentFollowMulti2;
                        if (ordinal == 0U) return follow2;
                        --ordinal;
                        AddressProgram follow3;
                        follow3.lags[0] = inner;
                        follow3.lags[1] = outer;
                        follow3.arity = 2U;
                        follow3.op = AddressOp::ContentFollowMulti3;
                        if (ordinal == 0U) return follow3;
                        --ordinal;
                    }
                }
            }
        }
    }
    return std::nullopt;
}

[[nodiscard]] double program_description_cost(const AddressProgram& program) noexcept {
    return 1.0 + static_cast<double>(program.arity);
}

[[nodiscard]] double program_execution_cost(const AddressProgram& program) noexcept {
    if (program.op == AddressOp::ContentMatch) {
        return program.arity == 0U ? 1.0 :
            1.0 + static_cast<double>(program.lags[program.arity - 1U]);
    }
    if (is_content_follow_op(program.op)) {
        const double hops = static_cast<double>(content_follow_hop_count(program.op));
        return program.arity == 0U ? 1.0 :
            1.0 + hops * static_cast<double>(program.lags[program.arity - 1U]);
    }
    return 1.0 + static_cast<double>(program.arity);
}

} // namespace

bool SparseBranchMachine::channel_enabled(std::size_t channel) const noexcept {
    if (channel >= topology_.size()) return false;
    const auto& state = topology_[channel];
    const auto phase = state.phase;
    const bool phase_enabled = phase == ChannelPhase::Seed ||
        phase == ChannelPhase::Probe || phase == ChannelPhase::Active;
    if (!phase_enabled) return false;
    if (address_program_required_dependency_binding(state.program) ==
        AddressBindingKind::None) {
        return true;
    }
    const auto dependency_channel =
        static_cast<std::size_t>(state.dependency_channel);
    if (dependency_channel >= topology_.size() || dependency_channel == channel) {
        return false;
    }
    const auto dependency_phase = topology_[dependency_channel].phase;
    return dependency_phase == ChannelPhase::Seed ||
        dependency_phase == ChannelPhase::Probe ||
        dependency_phase == ChannelPhase::Active;
}

bool SparseBranchMachine::channel_learning_enabled(std::size_t channel) const noexcept {
    if (!channel_enabled(channel)) return false;
    const auto& state = topology_[channel];
    if (state.phase != ChannelPhase::Probe) return true;
    const auto age = total_steps_ - state.born_step;
    const auto validation = std::min(config_.topology_validation_steps,
                                     config_.topology_probe_steps);
    return age + validation < config_.topology_probe_steps;
}

std::span<const AddressExecutionFrame> SparseBranchMachine::execute_address_programs(
    std::span<const std::uint32_t> window,
    bool learn) {
    execution_frames_.clear();
    std::fill(signature_buffer_.begin(), signature_buffer_.end(), 0U);
    if (binding_reuse_.size() < topology_.size()) {
        binding_reuse_.resize(topology_.size());
    }
    std::array<std::size_t, kMaxAddressChannels> frame_index_by_channel{};
    frame_index_by_channel.fill(SIZE_MAX);
    for (std::size_t channel = 0; channel < topology_.size(); ++channel) {
        if (!channel_enabled(channel)) continue;
        const auto& program = topology_[channel].program;
        AddressExecutionFrame frame{};
        const auto seed = config_.seed ^ mix64(address_program_key(program) +
                                              0x9E3779B97F4A7C15ULL);
        (void)execute_address_program(
            window, config_.token_alphabet, program, seed, frame);
        frame.channel = static_cast<std::uint8_t>(channel);
        frame.parent_channel = topology_[channel].parent_channel;
        frame.dependency_channel = topology_[channel].dependency_channel;
        frame.input_state = address_program_input_state(program);
        frame.output_state = address_program_output_state(program);
        frame.required_dependency_binding =
            address_program_required_dependency_binding(program);
        if (channel < frame_index_by_channel.size()) {
            frame_index_by_channel[channel] = execution_frames_.size();
        }
        execution_frames_.push_back(frame);
    }
    for (auto& frame : execution_frames_) {
        const auto channel = static_cast<std::size_t>(frame.channel);
        const auto dependency_channel = static_cast<std::size_t>(frame.dependency_channel);
        if (dependency_channel < frame_index_by_channel.size()) {
            const auto dependency_index = frame_index_by_channel[dependency_channel];
            if (dependency_index != SIZE_MAX && dependency_index < execution_frames_.size()) {
                const auto& dependency_frame = execution_frames_[dependency_index];
                frame.dependency_signature = dependency_frame.signature;
                frame.dependency_binding_key =
                    dependency_frame.binding_state.binding_key;
                if (dependency_frame.matched &&
                    address_dependency_satisfied(frame.program,
                                                 dependency_frame.binding) &&
                    dependency_frame.binding_state.binding_key != 0U) {
                    frame.call_key = mix64(
                        frame.signature ^
                        mix64(dependency_frame.signature + 0x517CC1B727220A95ULL) ^
                        mix64(dependency_frame.binding_state.binding_key +
                              address_program_key(frame.program)));
                    frame.signature = mix64(frame.signature ^ frame.call_key ^
                                            0xA24BAED4963EE407ULL);
                    frame.call_matched = true;
                    frame.execution_cost += 1.0F;
                }
            }
        }
        if (channel < signature_buffer_.size()) {
            signature_buffer_[channel] = frame.signature;
        }
        ++address_execution_frames_;
        if (frame.matched) ++address_binding_hits_;
        else ++address_binding_misses_;
        const auto binding_index = static_cast<std::size_t>(frame.binding);
        if (binding_index < kAddressBindingKindCount) {
            ++address_binding_kind_frames_[binding_index];
            if (frame.binding_state.matched) {
                ++address_binding_kind_hits_[binding_index];
                address_binding_kind_distance_sum_[binding_index] +=
                    static_cast<double>(frame.binding_state.matched_distance);
                address_binding_kind_pattern_span_sum_[binding_index] +=
                    static_cast<double>(frame.binding_state.pattern_span);
            }
        }
        if (learn && frame.binding_state.binding_key != 0U) {
            observe_binding_reuse(frame.channel, frame.binding_state.binding_key);
        }
        structural_description_cost_ += static_cast<double>(frame.description_cost);
        structural_execution_cost_ += static_cast<double>(frame.execution_cost);
    }
    return std::span<const AddressExecutionFrame>(
        execution_frames_.data(), execution_frames_.size());
}

std::span<const std::uint64_t> SparseBranchMachine::make_signatures(
    std::span<const std::uint32_t> window,
    bool learn) {
    if (config_.address_execution_mode == AddressExecutionMode::LegacySignature) {
        std::fill(signature_buffer_.begin(), signature_buffer_.end(), 0U);
        for (std::size_t channel = 0; channel < topology_.size(); ++channel) {
            if (!channel_enabled(channel)) continue;
            const auto& program = topology_[channel].program;
            signature_buffer_[channel] = address_program_signature(
                window, config_.token_alphabet,
                std::span<const std::uint32_t>(program.lags.data(), program.arity),
                program.op,
                config_.seed ^ mix64(address_program_key(program) +
                                     0x9E3779B97F4A7C15ULL));
        }
        return std::span<const std::uint64_t>(
            signature_buffer_.data(), topology_.size());
    }
    (void)execute_address_programs(window, learn);
    return std::span<const std::uint64_t>(signature_buffer_.data(), topology_.size());
}

void SparseBranchMachine::observe_binding_reuse(std::size_t channel,
                                                std::uint64_t key) {
    if (channel >= topology_.size() || key == 0U) return;
    if (binding_reuse_.size() < topology_.size()) binding_reuse_.resize(topology_.size());
    auto& records = binding_reuse_[channel];
    ++binding_reuse_observations_;
    for (auto& record : records) {
        if (record.key != key) continue;
        ++record.observations;
        record.last_seen_step = total_steps_;
        ++binding_reuse_events_;
        return;
    }
    const auto cap = config_.max_binding_reuse_records_per_channel;
    if (cap == 0U) return;
    if (records.size() < cap) {
        records.push_back({key, 1U, total_steps_});
        return;
    }
    auto victim = std::min_element(
        records.begin(), records.end(),
        [](const BindingReuseRecord& left, const BindingReuseRecord& right) {
            if (left.observations != right.observations) {
                return left.observations < right.observations;
            }
            return left.last_seen_step < right.last_seen_step;
        });
    if (victim != records.end()) {
        *victim = {key, 1U, total_steps_};
    }
}

std::uint64_t SparseBranchMachine::gpaf_role_key_for_channel(
    std::uint8_t channel_index) const noexcept {
    if (channel_index >= topology_.size() || config_.gpaf_slots == 0U) return 0U;
    const auto& channel = topology_[channel_index];
    std::uint64_t key = mix64(
        (static_cast<std::uint64_t>(channel.program.op) << 56U) ^
        (static_cast<std::uint64_t>(channel.program.arity) << 48U) ^
        (static_cast<std::uint64_t>(channel_index) << 40U) ^
        (static_cast<std::uint64_t>(channel.dependency_edge_kind) << 32U) ^
        (static_cast<std::uint64_t>(channel.parent_edge_kind) << 24U) ^
        mix64(channel.generation + 0xD1B54A32D192ED03ULL));
    key %= std::max<std::uint32_t>(1U, config_.gpaf_slots);
    return key;
}

void SparseBranchMachine::observe_gpaf_shadow_roles(
    std::span<const ScoredNode> active) {
    if (!config_.gpaf_shadow_observation || config_.gpaf_slots == 0U) return;
    for (const auto& node : active) {
        if (node.channel >= topology_.size() || node.id == kInvalidNode) continue;
        const std::uint64_t key = gpaf_role_key_for_channel(node.channel);
        ++gpaf_role_observations_[key];
        gpaf_slot_phases_.try_emplace(
            key, static_cast<std::uint8_t>(GpafSlotPhase::Probe));
        ++gpaf_role_observations_total_;
        ++gpaf_shadow_updates_;
        if (config_.gpaf_residents_per_slot == 0U) continue;
        auto& residents = gpaf_residents_[key];
        if (std::find(residents.begin(), residents.end(), node.id) != residents.end()) {
            continue;
        }
        if (residents.size() < config_.gpaf_residents_per_slot) {
            residents.push_back(node.id);
            continue;
        }
        residents[total_steps_ % residents.size()] = node.id;
    }
}

SparseBranchMachine::BindingReuseSummary SparseBranchMachine::binding_reuse_summary(
    std::size_t channel) const noexcept {
    BindingReuseSummary result;
    if (channel >= binding_reuse_.size()) return result;
    const auto& records = binding_reuse_[channel];
    result.unique_keys = records.size();
    for (const auto& record : records) {
        result.observations += record.observations;
        if (record.observations > 1U) {
            result.events += record.observations - 1U;
        }
    }
    return result;
}

double SparseBranchMachine::binding_reuse_bonus(
    const BindingReuseSummary& summary) const noexcept {
    if (config_.binding_reuse_value_weight <= 0.0F || summary.observations == 0U ||
        summary.events == 0U) {
        return 0.0;
    }
    const double repeated_fraction =
        static_cast<double>(summary.events) /
        static_cast<double>(summary.observations);
    return static_cast<double>(config_.binding_reuse_value_weight) *
           std::log1p(static_cast<double>(summary.events)) *
           repeated_fraction;
}

void SparseBranchMachine::quarantine_channel(std::size_t channel) {
    if (channel >= topology_.size() || channel == 0U) return;
    topology_[channel].phase = ChannelPhase::Quarantined;
}

void SparseBranchMachine::recoverably_retire_channel(std::size_t channel) {
    if (channel >= topology_.size() || channel == 0U) return;
    topology_[channel].phase = ChannelPhase::RecoverableRetired;
}

bool SparseBranchMachine::retire_channel(
    std::size_t channel,
    AcceptedChannelRetirement policy) {
    if (channel >= topology_.size() || channel == 0U ||
        policy == AcceptedChannelRetirement::Preserve) {
        return false;
    }
    auto& state = topology_[channel];
    if (state.phase != ChannelPhase::Seed &&
        state.phase != ChannelPhase::Probe &&
        state.phase != ChannelPhase::Active) {
        return false;
    }
    const auto reuse_summary = binding_reuse_summary(channel);
    const double reuse_bonus = binding_reuse_bonus(reuse_summary);
    topology_events_.push_back({total_steps_, state.generation,
                                state.program,
                                TopologyDecision::Pruned,
                                state.credit_ema + static_cast<float>(reuse_bonus),
                                state.credit_ema,
                                static_cast<float>(reuse_bonus),
                                reuse_summary.observations,
                                reuse_summary.unique_keys,
                                reuse_summary.events,
                                static_cast<std::uint8_t>(channel),
                                state.parent_channel,
                                state.dependency_channel,
                                state.parent_edge_kind,
                                state.dependency_edge_kind});
    if (policy == AcceptedChannelRetirement::Quarantine) {
        quarantine_channel(channel);
    } else if (policy == AcceptedChannelRetirement::RecoverableRetire) {
        recoverably_retire_channel(channel);
    } else {
        physically_erase_channel(channel);
    }
    ++topology_pruned_;
    return true;
}

bool SparseBranchMachine::restore_channel(std::size_t channel) {
    if (channel >= topology_.size() || channel == 0U) return false;
    auto& state = topology_[channel];
    if (state.phase != ChannelPhase::RecoverableRetired &&
        state.phase != ChannelPhase::Quarantined) {
        return false;
    }
    const auto reuse_summary = binding_reuse_summary(channel);
    const double reuse_bonus = binding_reuse_bonus(reuse_summary);
    topology_events_.push_back({total_steps_, state.generation,
                                state.program,
                                TopologyDecision::Restored,
                                state.credit_ema + static_cast<float>(reuse_bonus),
                                state.credit_ema,
                                static_cast<float>(reuse_bonus),
                                reuse_summary.observations,
                                reuse_summary.unique_keys,
                                reuse_summary.events,
                                static_cast<std::uint8_t>(channel),
                                state.parent_channel,
                                state.dependency_channel,
                                state.parent_edge_kind,
                                state.dependency_edge_kind});
    state.phase = ChannelPhase::Active;
    state.born_step = total_steps_;
    state.observations = 0U;
    return true;
}

std::size_t SparseBranchMachine::restore_dependency_closure(std::size_t channel) {
    if (channel >= topology_.size() || channel == 0U) return 0U;
    std::size_t restored = 0U;
    std::vector<std::size_t> stack;
    std::vector<std::uint8_t> visited(topology_.size(), 0U);
    stack.push_back(channel);
    while (!stack.empty()) {
        const auto current = stack.back();
        stack.pop_back();
        if (current >= topology_.size() || visited[current] != 0U) continue;
        visited[current] = 1U;
        if (restore_channel(current)) ++restored;
        if (!channel_enabled(current)) continue;
        for (std::size_t candidate = 1U; candidate < topology_.size(); ++candidate) {
            if (visited[candidate] != 0U) continue;
            if (topology_[candidate].dependency_channel != current) continue;
            const auto phase = topology_[candidate].phase;
            if (phase == ChannelPhase::Quarantined ||
                phase == ChannelPhase::RecoverableRetired ||
                phase == ChannelPhase::Seed ||
                phase == ChannelPhase::Probe ||
                phase == ChannelPhase::Active) {
                stack.push_back(candidate);
            }
        }
    }
    return restored;
}

void SparseBranchMachine::physically_erase_channel(std::size_t channel) {
    if (channel >= topology_.size() || channel == 0U) return;
    topology_[channel].phase = ChannelPhase::Retired;
    if (channel < binding_reuse_.size()) binding_reuse_[channel].clear();

    for (std::size_t slot = ids_.size(); slot-- > 0;) {
        if (channels_[slot] == channel) erase_slot(slot);
    }
    std::vector<NodeId> filtered_route;
    std::vector<float> filtered_responsibility;
    filtered_route.reserve(previous_route_.size());
    filtered_responsibility.reserve(previous_responsibilities_.size());
    for (std::size_t index = 0; index < previous_route_.size(); ++index) {
        const auto slot = slot_of(previous_route_[index]);
        if (slot == SIZE_MAX || channels_[slot] == channel) continue;
        filtered_route.push_back(previous_route_[index]);
        filtered_responsibility.push_back(index < previous_responsibilities_.size()
            ? previous_responsibilities_[index] : 1.0F);
    }
    previous_route_ = std::move(filtered_route);
    previous_responsibilities_ = std::move(filtered_responsibility);
    trace_.clear();
    rebuild_indexes();
}

std::uint8_t SparseBranchMachine::find_channel_for_program(
    const AddressProgram& program) const noexcept {
    for (std::size_t channel = 0U; channel < topology_.size(); ++channel) {
        if (!channel_enabled(channel)) continue;
        if (topology_[channel].program == program) {
            return static_cast<std::uint8_t>(channel);
        }
    }
    return kInvalidChannel;
}

std::uint8_t SparseBranchMachine::find_positional_channel_for_lag(
    std::uint32_t lag) const noexcept {
    for (std::size_t channel = 0U; channel < topology_.size(); ++channel) {
        if (!channel_enabled(channel)) continue;
        const auto& program = topology_[channel].program;
        if (program.op == AddressOp::Tuple && program.arity == 1U &&
            program.lags[0] == lag) {
            return static_cast<std::uint8_t>(channel);
        }
    }
    for (std::size_t channel = 0U; channel < topology_.size(); ++channel) {
        if (!channel_enabled(channel)) continue;
        const auto& program = topology_[channel].program;
        if ((program.op == AddressOp::Tuple || program.op == AddressOp::DeltaMod) &&
            program.arity != 0U && program.lags[program.arity - 1U] == lag) {
            return static_cast<std::uint8_t>(channel);
        }
    }
    return kInvalidChannel;
}

std::uint8_t SparseBranchMachine::find_prefix_channel(
    const AddressProgram& program) const noexcept {
    if (program.arity == 0U) return kInvalidChannel;
    if (program.arity > 1U) {
        AddressProgram prefix;
        prefix.op = program.op;
        prefix.arity = static_cast<std::uint8_t>(program.arity - 1U);
        for (std::size_t index = 0U; index < prefix.arity; ++index) {
            prefix.lags[index] = program.lags[index];
        }
        if (const auto exact = find_channel_for_program(prefix);
            exact != kInvalidChannel) {
            return exact;
        }
    }
    const auto max_lag = program.lags[program.arity - 1U];
    if (const auto positional = find_positional_channel_for_lag(max_lag);
        positional != kInvalidChannel) {
        return positional;
    }
    return topology_.empty() || !channel_enabled(0U)
        ? kInvalidChannel
        : static_cast<std::uint8_t>(0U);
}

std::pair<std::uint8_t, std::uint8_t> SparseBranchMachine::resolve_channel_lineage(
    const AddressProgram& program) const noexcept {
    if (program.arity == 0U) return {kInvalidChannel, kInvalidChannel};
    const auto max_lag = program.lags[program.arity - 1U];
    if (program.op == AddressOp::ContentMatch) {
        const auto dependency = find_positional_channel_for_lag(max_lag);
        return {dependency, dependency};
    }
    if (is_content_follow_op(program.op)) {
        AddressProgram match = singleton_address_program(max_lag);
        match.op = AddressOp::ContentMatch;
        auto dependency = find_channel_for_program(match);
        if (dependency == kInvalidChannel) {
            dependency = find_positional_channel_for_lag(max_lag);
        }
        return {dependency, dependency};
    }
    const auto parent = find_prefix_channel(program);
    return {parent, parent};
}

void SparseBranchMachine::maybe_finalize_topology_probe() {
    if (!config_.adaptive_topology) return;
    for (std::size_t channel = 0; channel < topology_.size(); ++channel) {
        auto& state = topology_[channel];
        if (state.phase != ChannelPhase::Probe) continue;
        const auto age = total_steps_ - state.born_step;
        if (age < config_.topology_probe_steps ||
            state.observations < config_.topology_min_observations) {
            return;
        }
        const double mean_credit = state.credit_sum /
            static_cast<double>(std::max<std::uint64_t>(1U, state.observations));
        const double description_penalty =
            static_cast<double>(config_.structural_description_cost_weight) *
            program_description_cost(state.program) /
            static_cast<double>(std::max<std::uint64_t>(1U, state.observations));
        const double execution_penalty =
            static_cast<double>(config_.structural_execution_cost_weight) *
            program_execution_cost(state.program);
        const double structural_value =
            mean_credit - description_penalty - execution_penalty;
        const auto reuse_summary = binding_reuse_summary(channel);
        const double reuse_bonus = binding_reuse_bonus(reuse_summary);
        const double reuse_aware_structural_value = structural_value + reuse_bonus;
        const double decision_value = config_.topology_accept_uses_structural_value
            ? reuse_aware_structural_value : mean_credit;
    if (decision_value >= static_cast<double>(config_.topology_accept_credit)) {
            topology_events_.push_back({total_steps_, state.generation,
                                        state.program,
                                        TopologyDecision::Accepted,
                                        static_cast<float>(reuse_aware_structural_value),
                                        static_cast<float>(structural_value),
                                        static_cast<float>(reuse_bonus),
                                        reuse_summary.observations,
                                        reuse_summary.unique_keys,
                                        reuse_summary.events,
                                        static_cast<std::uint8_t>(channel),
                                        state.parent_channel,
                                        state.dependency_channel,
                                        state.parent_edge_kind,
                                        state.dependency_edge_kind});
            state.phase = ChannelPhase::Active;
            state.born_step = total_steps_;
            state.observations = 0U;
            state.credit_sum = 0.0;
            ++topology_accepted_;
        } else {
            topology_events_.push_back({total_steps_, state.generation,
                                        state.program,
                                        TopologyDecision::Rejected,
                                        static_cast<float>(reuse_aware_structural_value),
                                        static_cast<float>(structural_value),
                                        static_cast<float>(reuse_bonus),
                                        reuse_summary.observations,
                                        reuse_summary.unique_keys,
                                        reuse_summary.events,
                                        static_cast<std::uint8_t>(channel),
                                        state.parent_channel,
                                        state.dependency_channel,
                                        state.parent_edge_kind,
                                        state.dependency_edge_kind});
            physically_erase_channel(channel);
            ++topology_rejected_;
        }
        next_probe_step_ = total_steps_ + config_.topology_probe_interval;
        return;
    }
    for (std::size_t channel = 1U; channel < topology_.size(); ++channel) {
        auto& state = topology_[channel];
        if (state.phase != ChannelPhase::Active) continue;
        const auto age = total_steps_ - state.born_step;
        if (age < config_.topology_prune_patience ||
            state.observations < config_.topology_prune_patience ||
            state.credit_ema >= config_.topology_prune_credit) {
            continue;
        }
        if (config_.accepted_channel_retirement ==
            AcceptedChannelRetirement::Preserve) {
            continue;
        }
        const auto reuse_summary = binding_reuse_summary(channel);
        const double reuse_bonus = binding_reuse_bonus(reuse_summary);
        topology_events_.push_back({total_steps_, state.generation,
                                    state.program,
                                    TopologyDecision::Pruned,
                                    state.credit_ema + static_cast<float>(reuse_bonus),
                                    state.credit_ema,
                                    static_cast<float>(reuse_bonus),
                                    reuse_summary.observations,
                                    reuse_summary.unique_keys,
                                    reuse_summary.events,
                                    static_cast<std::uint8_t>(channel),
                                    state.parent_channel,
                                    state.dependency_channel,
                                    state.parent_edge_kind,
                                    state.dependency_edge_kind});
        if (config_.accepted_channel_retirement ==
            AcceptedChannelRetirement::Quarantine) {
            quarantine_channel(channel);
        } else if (config_.accepted_channel_retirement ==
                   AcceptedChannelRetirement::RecoverableRetire) {
            recoverably_retire_channel(channel);
        } else {
            physically_erase_channel(channel);
        }
        ++topology_pruned_;
        next_probe_step_ = total_steps_ + config_.topology_probe_interval;
        return;
    }
}


void SparseBranchMachine::freeze_topology() {
    for (std::size_t channel = 1U; channel < topology_.size(); ++channel) {
        auto& state = topology_[channel];
        if (state.phase != ChannelPhase::Probe) continue;
        const double mean_credit = state.observations == 0U
            ? 0.0
            : state.credit_sum / static_cast<double>(state.observations);
        const auto reuse_summary = binding_reuse_summary(channel);
        const double reuse_bonus = binding_reuse_bonus(reuse_summary);
        topology_events_.push_back({total_steps_, state.generation,
                                    state.program,
                                    TopologyDecision::Rejected,
                                    static_cast<float>(mean_credit + reuse_bonus),
                                    static_cast<float>(mean_credit),
                                    static_cast<float>(reuse_bonus),
                                    reuse_summary.observations,
                                    reuse_summary.unique_keys,
                                    reuse_summary.events,
                                    static_cast<std::uint8_t>(channel),
                                    state.parent_channel,
                                    state.dependency_channel,
                                    state.parent_edge_kind,
                                    state.dependency_edge_kind});
        physically_erase_channel(channel);
        ++topology_rejected_;
    }
}

void SparseBranchMachine::maybe_begin_topology_probe(bool learn) {
    if (!learn) return;
    maybe_finalize_topology_probe();
    if (!config_.adaptive_topology || total_steps_ < next_probe_step_) return;

    for (const auto& state : topology_) {
        if (state.phase == ChannelPhase::Probe) return;
    }

    std::size_t enabled = 0U;
    for (const auto& state : topology_) {
        if (state.phase == ChannelPhase::Seed || state.phase == ChannelPhase::Active ||
            state.phase == ChannelPhase::Probe) {
            ++enabled;
        }
    }
    if (enabled >= config_.max_address_channels) return;

    std::optional<AddressProgram> proposal;
    while (true) {
        const auto candidate = proposal_at(proposal_cursor_++, config_.topology_max_lag,
                                           config_.topology_max_arity,
                                           config_.topology_enable_delta,
                                           config_.topology_enable_content_match,
                                           config_.topology_enable_content_follow_multi);
        if (!candidate.has_value()) break;
        const auto key = address_program_key(*candidate);
        if (std::find(proposed_program_keys_.begin(), proposed_program_keys_.end(), key) ==
            proposed_program_keys_.end()) {
            proposal = candidate;
            proposed_program_keys_.push_back(key);
            break;
        }
    }
    if (!proposal.has_value()) return;

    std::size_t slot = topology_.size();
    for (std::size_t index = 1U; index < topology_.size(); ++index) {
        if (topology_[index].phase == ChannelPhase::Retired) {
            slot = index;
            break;
        }
    }
    if (slot == topology_.size()) {
        if (topology_.size() >= config_.max_address_channels) return;
        topology_.push_back({});
        binding_reuse_.push_back({});
    }

    const auto [parent_channel, dependency_channel] =
        resolve_channel_lineage(*proposal);
    const auto parent_edge_kind = parent_channel == kInvalidChannel
        ? AddressGraphEdgeKind::None
        : address_parent_edge_kind(*proposal);
    const auto dependency_edge_kind = dependency_channel == kInvalidChannel
        ? AddressGraphEdgeKind::None
        : address_dependency_edge_kind(*proposal);
    topology_[slot] = {*proposal, ChannelPhase::Probe, 0.0F, 0.0, 0U,
                       total_steps_, ++topology_generation_counter_,
                       parent_channel, dependency_channel,
                       parent_edge_kind, dependency_edge_kind};
    if (binding_reuse_.size() <= slot) binding_reuse_.resize(slot + 1U);
    binding_reuse_[slot].clear();
    topology_events_.push_back({total_steps_, topology_[slot].generation,
                                *proposal,
                                TopologyDecision::Proposed, 0.0F,
                                0.0F, 0.0F, 0U, 0U, 0U,
                                static_cast<std::uint8_t>(slot),
                                parent_channel, dependency_channel,
                                parent_edge_kind, dependency_edge_kind});
    ++topology_proposals_;
    next_probe_step_ = total_steps_ + config_.topology_probe_steps;
}

void SparseBranchMachine::observe_topology_credit(
    std::span<const float> channel_credit) {
    if (!config_.adaptive_topology) return;
    for (std::size_t channel = 0; channel < topology_.size(); ++channel) {
        auto& state = topology_[channel];
        if (!channel_enabled(channel) || channel >= channel_credit.size()) continue;
        const float credit = channel_credit[channel];
        state.credit_ema = config_.topology_credit_decay * state.credit_ema +
            (1.0F - config_.topology_credit_decay) * credit;
        if (state.phase == ChannelPhase::Active) ++state.observations;
        if (state.phase == ChannelPhase::Probe) {
            const auto age = total_steps_ - state.born_step;
            const auto validation = std::min(config_.topology_validation_steps,
                                             config_.topology_probe_steps);
            const auto validation_start = config_.topology_probe_steps - validation;
            if (age >= validation_start && age >= config_.topology_probe_warmup) {
                state.credit_sum += static_cast<double>(credit);
                ++state.observations;
            }
        }
    }
}

std::vector<std::uint32_t> SparseBranchMachine::learned_address_lags() const {
    std::vector<std::uint32_t> result;
    for (const auto& state : topology_) {
        if ((state.phase == ChannelPhase::Seed || state.phase == ChannelPhase::Probe ||
             state.phase == ChannelPhase::Active) && state.program.arity == 1U) {
            result.push_back(state.program.lags[0]);
        }
    }
    return result;
}

std::vector<AddressProgram> SparseBranchMachine::learned_address_programs() const {
    std::vector<AddressProgram> result;
    for (const auto& state : topology_) {
        if (state.phase == ChannelPhase::Seed || state.phase == ChannelPhase::Probe ||
            state.phase == ChannelPhase::Active) {
            result.push_back(state.program);
        }
    }
    return result;
}

std::vector<float> SparseBranchMachine::learned_channel_credit() const {
    std::vector<float> result;
    result.reserve(topology_.size());
    for (const auto& state : topology_) result.push_back(state.credit_ema);
    return result;
}

std::vector<std::uint8_t> SparseBranchMachine::learned_channel_phase() const {
    std::vector<std::uint8_t> result;
    result.reserve(topology_.size());
    for (const auto& state : topology_) {
        result.push_back(static_cast<std::uint8_t>(state.phase));
    }
    return result;
}

std::vector<std::uint8_t>
SparseBranchMachine::learned_channel_effective_enabled() const {
    std::vector<std::uint8_t> result;
    result.reserve(topology_.size());
    for (std::size_t channel = 0U; channel < topology_.size(); ++channel) {
        result.push_back(channel_enabled(channel) ? 1U : 0U);
    }
    return result;
}

std::vector<std::uint8_t> SparseBranchMachine::learned_channel_parent() const {
    std::vector<std::uint8_t> result;
    result.reserve(topology_.size());
    for (const auto& state : topology_) {
        result.push_back(state.parent_channel);
    }
    return result;
}

std::vector<std::uint8_t> SparseBranchMachine::learned_channel_dependency() const {
    std::vector<std::uint8_t> result;
    result.reserve(topology_.size());
    for (const auto& state : topology_) {
        result.push_back(state.dependency_channel);
    }
    return result;
}

std::vector<std::uint64_t> SparseBranchMachine::learned_channel_generation() const {
    std::vector<std::uint64_t> result;
    result.reserve(topology_.size());
    for (const auto& state : topology_) {
        result.push_back(state.generation);
    }
    return result;
}

std::vector<std::uint8_t> SparseBranchMachine::learned_channel_parent_edge_kind() const {
    std::vector<std::uint8_t> result;
    result.reserve(topology_.size());
    for (const auto& state : topology_) {
        result.push_back(static_cast<std::uint8_t>(state.parent_edge_kind));
    }
    return result;
}

std::vector<std::uint8_t> SparseBranchMachine::learned_channel_dependency_edge_kind() const {
    std::vector<std::uint8_t> result;
    result.reserve(topology_.size());
    for (const auto& state : topology_) {
        result.push_back(static_cast<std::uint8_t>(state.dependency_edge_kind));
    }
    return result;
}

} // namespace sbm
