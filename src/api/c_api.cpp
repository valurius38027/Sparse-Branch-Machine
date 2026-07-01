#include "sbm/api.h"

#include "sbm/dataset.hpp"
#include "sbm/experiment.hpp"
#include "sbm/machine.hpp"
#include "sbm/types.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

struct sbm_config_handle {
    sbm::Config value;
};

struct sbm_dataset_handle {
    std::variant<sbm::VectorDataset, sbm::TokenDataset> value;
};

struct sbm_machine_handle {
    sbm::SparseBranchMachine value;
};

struct sbm_token_shard_handle {
    sbm::MappedTokenShard value;
};

struct sbm_token_corpus_handle {
    std::vector<std::unique_ptr<sbm::MappedTokenShard>> train;
    std::vector<std::unique_ptr<sbm::MappedTokenShard>> eval;
};

namespace {
thread_local std::string g_last_error;

void clear_error() noexcept { g_last_error.clear(); }

void set_error(std::string message) noexcept {
    try {
        g_last_error = std::move(message);
    } catch (...) {
        g_last_error = "unknown API error";
    }
}

template <class Function>
auto guarded(Function&& function) noexcept -> decltype(function()) {
    using Return = decltype(function());
    clear_error();
    try {
        return function();
    } catch (const std::exception& error) {
        set_error(error.what());
    } catch (...) {
        set_error("unknown C++ exception");
    }
    if constexpr (std::is_pointer_v<Return>) return nullptr;
    else return Return{};
}

char* duplicate_string(const std::string& value) {
    auto* result = static_cast<char*>(std::malloc(value.size() + 1U));
    if (result == nullptr) throw std::bad_alloc();
    std::memcpy(result, value.c_str(), value.size() + 1U);
    return result;
}

std::string json_escape(std::string_view value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        switch (ch) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (ch < 0x20U) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<unsigned>(ch) << std::dec;
            } else {
                out << static_cast<char>(ch);
            }
        }
    }
    return out.str();
}

template <class T>
T parse_integer(std::string_view text, const char* name) {
    T value{};
    const auto* begin = text.data();
    const auto* end = text.data() + text.size();
    const auto [position, error] = std::from_chars(begin, end, value);
    if (error != std::errc{} || position != end) {
        throw std::invalid_argument(std::string("invalid integer for ") + name);
    }
    return value;
}

float parse_float(std::string_view text, const char* name) {
    std::string owned(text);
    char* end = nullptr;
    errno = 0;
    const float value = std::strtof(owned.c_str(), &end);
    if (errno != 0 || end != owned.c_str() + owned.size() || !std::isfinite(value)) {
        throw std::invalid_argument(std::string("invalid float for ") + name);
    }
    return value;
}

double parse_double(std::string_view text, const char* name) {
    std::string owned(text);
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(owned.c_str(), &end);
    if (errno != 0 || end != owned.c_str() + owned.size() || !std::isfinite(value)) {
        throw std::invalid_argument(std::string("invalid double for ") + name);
    }
    return value;
}

bool parse_bool(std::string_view text, const char* name) {
    if (text == "1" || text == "true" || text == "TRUE" || text == "yes") return true;
    if (text == "0" || text == "false" || text == "FALSE" || text == "no") return false;
    throw std::invalid_argument(std::string("invalid boolean for ") + name);
}

sbm::AddressExecutionMode parse_address_execution_mode(std::string_view text) {
    if (text == "LegacySignature" || text == "legacy" || text == "legacy_signature" ||
        text == "0") {
        return sbm::AddressExecutionMode::LegacySignature;
    }
    if (text == "InterpretedFrames" || text == "interpreted" ||
        text == "interpreted_frames" || text == "1") {
        return sbm::AddressExecutionMode::InterpretedFrames;
    }
    throw std::invalid_argument("invalid enum for address_execution_mode");
}

sbm::AcceptedChannelRetirement parse_accepted_channel_retirement(
    std::string_view text) {
    if (text == "Preserve" || text == "preserve" || text == "0") {
        return sbm::AcceptedChannelRetirement::Preserve;
    }
    if (text == "Quarantine" || text == "quarantine" || text == "1") {
        return sbm::AcceptedChannelRetirement::Quarantine;
    }
    if (text == "RecoverableRetire" || text == "recoverable_retire" ||
        text == "RecoverableRetired" || text == "recoverable_retired" ||
        text == "2") {
        return sbm::AcceptedChannelRetirement::RecoverableRetire;
    }
    if (text == "PhysicalErase" || text == "physical_erase" ||
        text == "erase" || text == "3") {
        return sbm::AcceptedChannelRetirement::PhysicalErase;
    }
    throw std::invalid_argument("invalid enum for accepted_channel_retirement");
}

std::string_view to_string(sbm::AddressExecutionMode value) noexcept {
    switch (value) {
        case sbm::AddressExecutionMode::LegacySignature: return "LegacySignature";
        case sbm::AddressExecutionMode::InterpretedFrames: return "InterpretedFrames";
    }
    return "InterpretedFrames";
}

std::string_view to_string(sbm::AcceptedChannelRetirement value) noexcept {
    switch (value) {
        case sbm::AcceptedChannelRetirement::Preserve: return "Preserve";
        case sbm::AcceptedChannelRetirement::Quarantine: return "Quarantine";
        case sbm::AcceptedChannelRetirement::RecoverableRetire:
            return "RecoverableRetire";
        case sbm::AcceptedChannelRetirement::PhysicalErase: return "PhysicalErase";
    }
    return "Preserve";
}

std::vector<std::uint32_t> parse_lags(std::string_view text) {
    std::vector<std::uint32_t> result;
    std::size_t offset = 0;
    while (offset <= text.size()) {
        const auto comma = text.find(',', offset);
        const auto part = text.substr(offset, comma == std::string_view::npos
                                              ? std::string_view::npos
                                              : comma - offset);
        if (part.empty()) throw std::invalid_argument("address_lags contains an empty item");
        result.push_back(parse_integer<std::uint32_t>(part, "address_lags"));
        if (result.size() > sbm::kMaxAddressChannels) {
            throw std::invalid_argument("too many address_lags");
        }
        if (comma == std::string_view::npos) break;
        offset = comma + 1U;
    }
    if (result.empty()) throw std::invalid_argument("address_lags must not be empty");
    return result;
}

struct ParameterDescriptor {
    const char* name;
    const char* type;
    const char* default_value;
    const char* minimum;
    const char* maximum;
    const char* scale;
    bool tunable;
    bool search_default;
    bool dataset_bound;
    const char* description;
};

// This table is the single discovery source for the CLI and Python tuner.
constexpr ParameterDescriptor kParameters[] = {
    {"token_alphabet", "uint32", "64", "1", "1048576", "linear", false, false, true, "Dataset token alphabet; overwritten by the dataset at run time."},
    {"vector_dim", "uint32", "16", "1", "4096", "linear", false, false, true, "Target vector dimension; overwritten by the dataset at run time."},
    {"context_width", "uint32", "12", "2", "256", "linear", true, false, false, "Maximum retained token history."},
    {"bucket_bits", "uint32", "12", "6", "18", "linear", true, false, false, "Bits used by each address-channel bucket index."},
    {"beam_width", "uint32", "6", "1", "64", "linear", true, false, false, "Maximum active nodes per step."},
    {"beam_width_min", "uint32", "6", "1", "64", "linear", true, false, false, "Minimum active nodes when prediction is confident; beam_width remains the upper bound."},
    {"confidence_threshold", "float", "0.8", "0.0", "1.0", "linear", true, false, false, "Minimum max-responsibility concentration that triggers truncation to beam_width_min."},
    {"max_refinement_rounds", "uint32", "0", "0", "8", "linear", true, false, false, "Maximum extra routing rounds with expanded neighbor radius for uncertain tokens."},
    {"refinement_confidence_threshold", "float", "0.5", "0.0", "1.0", "linear", true, false, false, "Minimum max-responsibility that stops iterative refinement."},
    {"bucket_scan_limit", "uint32", "32", "4", "128", "log", true, false, false, "Maximum nodes examined from an address bucket."},
    {"edge_scan_limit", "uint32", "8", "1", "32", "linear", true, false, false, "Maximum outgoing edges examined per source node."},
    {"max_edges_per_node", "uint32", "32", "4", "128", "log", true, false, false, "Maximum stored sparse control edges per node."},
    {"edge_reinforce_width", "uint32", "2", "1", "8", "linear", true, false, false, "Number of route positions eligible for edge reinforcement."},
    {"max_specializations_per_bucket", "uint32", "4", "1", "16", "linear", true, false, false, "Maximum conflict-driven specializations per address bucket."},
    {"split_min_visits", "uint32", "48", "8", "256", "log", true, false, false, "Minimum address-local visits before specialization."},
    {"split_cooldown", "uint32", "128", "8", "2048", "log", true, false, false, "Minimum steps between bucket specializations."},
    {"split_context_similarity", "double", "0.78", "0.40", "0.99", "linear", true, false, false, "Maximum context mismatch tolerated by specialization."},
    {"split_loss_threshold", "float", "0.72", "0.10", "2.00", "log", true, false, false, "Address-local normalized-loss threshold for specialization."},
    {"allow_growth_when_frozen", "bool", "false", "", "", "categorical", false, false, false, "Allow structural growth during evaluation."},
    {"trace_horizon", "uint32", "8", "1", "64", "log", true, false, false, "Number of recent routes retained for delayed credit."},
    {"trace_decay", "float", "0.80", "0.30", "0.999", "linear", true, false, false, "Temporal decay for delayed node credit."},
    {"edge_decay", "float", "0.999", "0.95", "0.99999", "linear", true, true, false, "Multiplicative sparse-edge decay."},
    {"edge_learning_rate", "float", "0.08", "0.002", "0.30", "log", true, true, false, "Positive sparse-edge update rate."},
    {"edge_score_weight", "float", "0.32", "0.0", "0.80", "linear", true, true, false, "Routing score weight assigned to learned control edges."},
    {"edge_min_contribution", "float", "0.004", "0.0", "0.05", "linear", true, true, false, "Minimum counterfactual contribution for edge reinforcement."},
    {"responsibility_temperature", "float", "3.0", "0.25", "12.0", "log", true, true, false, "Soft responsibility temperature within a route."},
    {"exact_region_mass", "float", "0.88", "0.50", "1.0", "linear", true, true, false, "Prediction mass reserved for exact address regions."},
    {"min_update_responsibility", "float", "0.01", "0.0", "0.20", "linear", true, true, false, "Minimum local responsibility required for a parameter update."},
    {"address_lags", "uint32[]", "1", "", "", "categorical", false, false, false, "Seed temporal address lags; adaptive topology may add more."},
    {"adaptive_topology", "bool", "true", "", "", "categorical", false, false, false, "Enable predictive-credit topology proposals."},
    {"max_address_channels", "uint32", "6", "1", "8", "linear", true, false, false, "Maximum simultaneous address channels."},
    {"topology_max_lag", "uint32", "16", "2", "256", "log", true, false, false, "Largest temporal lag eligible for proposal."},
    {"topology_max_arity", "uint32", "2", "1", "2", "linear", true, false, false, "Maximum number of history offsets in an address program."},
    {"topology_enable_delta", "bool", "true", "", "", "categorical", false, false, false, "Allow the generic modular-difference address operator to be proposed."},
    {"topology_enable_content_match", "bool", "true", "", "", "categorical", false, false, false, "Allow bounded content-conditioned match/follow address operators to be proposed."},
    {"topology_enable_content_follow_multi", "bool", "false", "", "", "categorical", false, false, false, "Allow bounded multi-hop content-follow address operators to be proposed."},
    {"topology_probe_interval", "uint32", "2048", "128", "16384", "log", true, false, false, "Delay between topology proposals."},
    {"topology_probe_warmup", "uint32", "512", "0", "4096", "linear", true, false, false, "Probe steps ignored before credit collection."},
    {"topology_probe_steps", "uint32", "4096", "512", "32768", "log", true, false, false, "Lifetime of a candidate address channel."},
    {"topology_validation_steps", "uint32", "1024", "128", "8192", "log", true, false, false, "Frozen tail used for out-of-sample topology selection."},
    {"topology_min_observations", "uint32", "512", "64", "8192", "log", true, false, false, "Minimum probe observations before a decision."},
    {"topology_accept_credit", "float", "0.0005", "0.0", "0.05", "linear", true, false, false, "Mean counterfactual NLL gain required to retain a channel."},
    {"topology_credit_decay", "float", "0.995", "0.90", "0.9999", "linear", true, false, false, "EMA decay for address-channel credit."},
    {"topology_prune_patience", "uint32", "4294967295", "512", "4294967295", "log", true, false, false, "Mature observations required before channel retirement; the default preserves accepted channels."},
    {"topology_prune_credit", "float", "-0.01", "-0.05", "0.0", "linear", true, false, false, "Credit threshold for retiring an accepted channel."},
    {"address_execution_mode", "enum", "InterpretedFrames", "", "", "categorical", false, false, false, "Address execution backend: LegacySignature or InterpretedFrames."},
    {"accepted_channel_retirement", "enum", "Preserve", "", "", "categorical", false, false, false, "Lifecycle policy for accepted channels: Preserve, Quarantine, RecoverableRetire or PhysicalErase."},
    {"structural_description_cost_weight", "float", "1.0", "0.0", "10.0", "linear", true, false, false, "Weight applied to program description cost."},
    {"structural_execution_cost_weight", "float", "0.0", "0.0", "10.0", "linear", true, false, false, "Weight applied to measured address execution cost."},
    {"binding_reuse_value_weight", "float", "0.0", "0.0", "10.0", "linear", true, false, false, "Optional structural-value bonus weight for repeatedly observed typed binding keys."},
    {"topology_accept_uses_structural_value", "bool", "false", "", "", "categorical", false, false, false, "Use cost-penalized structural value rather than raw mean credit for topology acceptance."},
    {"residual_channel_gain", "float", "1.0", "0.25", "2.0", "linear", true, true, false, "Gain applied to additive residual channels."},
    {"residual_learning_rate", "float", "0.10", "0.005", "0.50", "log", true, false, false, "Legacy residual update cap used by non-mean paths."},
    {"residual_mature_learning_rate", "float", "0.03", "0.001", "0.20", "log", true, false, false, "Legacy mature residual update cap."},
    {"residual_recency_pseudocount", "float", "0.75", "0.0", "8.0", "linear", true, true, false, "Recency bias for sequential residual means."},
    {"warm_visits", "uint32", "12", "2", "128", "log", true, false, false, "Visits needed to leave the cold lifecycle phase."},
    {"mature_visits", "uint32", "96", "16", "1024", "log", true, false, false, "Visits needed to enter the mature lifecycle phase."},
    {"dormant_utility", "float", "-0.30", "-2.0", "0.0", "linear", true, false, false, "Utility threshold for the dormant lifecycle phase."},
    {"node_learning_rate", "float", "0.12", "0.005", "0.50", "log", true, false, false, "Legacy anchor update cap used by non-mean paths."},
    {"mature_learning_rate", "float", "0.035", "0.001", "0.20", "log", true, false, false, "Legacy mature anchor update cap."},
    {"merge_similarity", "float", "0.95", "0.70", "0.999", "linear", true, false, false, "Minimum address similarity for node merge."},
    {"merge_vector_distance", "float", "0.08", "0.005", "0.50", "log", true, false, false, "Maximum output-vector/logit distance for node merge."},
    {"classification_learning_rate", "float", "0.35", "0.01", "1.0", "log", true, true, false, "Initial local logit learning rate for token cross-entropy."},
    {"classification_mature_learning_rate", "float", "0.08", "0.002", "0.40", "log", true, true, false, "Local logit learning rate for mature token nodes."},
    {"label_smoothing", "float", "0.01", "0.0", "0.20", "linear", true, true, false, "Label smoothing for mathematical next-token cross-entropy."},
    {"softmax_temperature", "float", "1.0", "0.25", "3.0", "log", true, true, false, "Temperature applied to aggregated token logits."},
    {"logit_decay", "float", "0.0001", "0.0", "0.02", "linear", true, true, false, "Per-update decay applied to local token logits."},
    {"sparse_token_output", "bool", "true", "", "", "categorical", false, false, false, "Use exact hierarchical softmax with sparse per-node binary decisions."},
    {"sparse_output_topk", "uint32", "5", "1", "32", "linear", true, false, false, "Number of hierarchical candidates reported by token decoding."},
    {"sparse_output_beam_width", "uint32", "16", "5", "128", "log", true, false, false, "Fixed candidate beam for O(B log V) hierarchical decoding."},
    {"max_sparse_decisions_per_node", "uint32", "64", "8", "4096", "log", true, false, false, "Hard bound on local hierarchical decisions stored by one address node."},
    {"decode_token_ranking_during_training", "bool", "false", "", "", "categorical", false, false, false, "Compute token top-k ranking metrics during training; evaluation ranking is always computed."},
    {"use_momentum", "bool", "false", "", "", "categorical", false, false, false, "Enable per-entry Adam-like momentum for sparse decision logits."},
    {"momentum_beta1", "float", "0.9", "0.0", "0.999", "linear", true, false, false, "Exponential decay rate for the first moment estimate."},
    {"momentum_beta2", "float", "0.999", "0.0", "0.99999", "linear", true, false, false, "Exponential decay rate for the second moment estimate."},
    {"momentum_eps", "float", "1e-8", "1e-12", "1e-3", "log", true, false, false, "Epsilon for numerical stability in adaptive learning rate."},
    {"record_channel_attribution", "bool", "false", "", "", "categorical", false, false, false, "Record frozen-evaluation per-channel counterfactual codelength attribution."},
    {"max_binding_reuse_records_per_channel", "uint32", "4096", "0", "65536", "log", true, false, false, "Bounded per-channel training registry size for reusable binding keys."},
    {"gpaf_shadow_observation", "bool", "false", "", "", "categorical", false, false, false, "Record Global Predictive Address Field role keys without changing routing."},
    {"gpaf_candidate_retrieval", "bool", "false", "", "", "categorical", false, false, false, "Allow GPAF slots to inject bounded candidates; disabled by default."},
    {"gpaf_query_keys_per_step", "uint32", "0", "0", "32", "linear", true, false, false, "Maximum GPAF role query keys per step."},
    {"gpaf_slots", "uint32", "0", "0", "1048576", "log", true, false, false, "Maximum GPAF shadow/address slots."},
    {"gpaf_residents_per_slot", "uint32", "0", "0", "64", "linear", true, false, false, "Maximum resident candidates returned by one GPAF slot."},
    {"output_tree_seed", "uint64", "7", "0", "18446744073709551615", "linear", true, false, false, "Seed for the fixed implicit output decomposition; keep constant across model seeds."},
    {"seed", "uint64", "7", "0", "18446744073709551615", "linear", false, false, false, "Model random seed."},
};

bool set_parameter(sbm::Config& config, std::string_view name, std::string_view value) {
#define SBM_SET_UINT(field) if (name == #field) { config.field = parse_integer<std::uint32_t>(value, #field); return true; }
#define SBM_SET_U64(field) if (name == #field) { config.field = parse_integer<std::uint64_t>(value, #field); return true; }
#define SBM_SET_FLOAT(field) if (name == #field) { config.field = parse_float(value, #field); return true; }
#define SBM_SET_DOUBLE(field) if (name == #field) { config.field = parse_double(value, #field); return true; }
#define SBM_SET_BOOL(field) if (name == #field) { config.field = parse_bool(value, #field); return true; }
    SBM_SET_UINT(token_alphabet)
    SBM_SET_UINT(vector_dim)
    SBM_SET_UINT(context_width)
    SBM_SET_UINT(bucket_bits)
    SBM_SET_UINT(beam_width)
    SBM_SET_UINT(beam_width_min)
    SBM_SET_FLOAT(confidence_threshold)
    SBM_SET_UINT(max_refinement_rounds)
    SBM_SET_FLOAT(refinement_confidence_threshold)
    SBM_SET_UINT(bucket_scan_limit)
    SBM_SET_UINT(edge_scan_limit)
    SBM_SET_UINT(max_edges_per_node)
    SBM_SET_UINT(edge_reinforce_width)
    SBM_SET_UINT(max_specializations_per_bucket)
    SBM_SET_UINT(split_min_visits)
    SBM_SET_UINT(split_cooldown)
    SBM_SET_DOUBLE(split_context_similarity)
    SBM_SET_FLOAT(split_loss_threshold)
    SBM_SET_BOOL(allow_growth_when_frozen)
    SBM_SET_UINT(trace_horizon)
    SBM_SET_FLOAT(trace_decay)
    SBM_SET_FLOAT(edge_decay)
    SBM_SET_FLOAT(edge_learning_rate)
    SBM_SET_FLOAT(edge_score_weight)
    SBM_SET_FLOAT(edge_min_contribution)
    SBM_SET_FLOAT(responsibility_temperature)
    SBM_SET_FLOAT(exact_region_mass)
    SBM_SET_FLOAT(min_update_responsibility)
    if (name == "address_lags") { config.address_lags = parse_lags(value); return true; }
    SBM_SET_BOOL(adaptive_topology)
    SBM_SET_UINT(max_address_channels)
    SBM_SET_UINT(topology_max_lag)
    SBM_SET_UINT(topology_max_arity)
    SBM_SET_BOOL(topology_enable_delta)
    SBM_SET_BOOL(topology_enable_content_match)
    SBM_SET_BOOL(topology_enable_content_follow_multi)
    SBM_SET_UINT(topology_probe_interval)
    SBM_SET_UINT(topology_probe_warmup)
    SBM_SET_UINT(topology_probe_steps)
    SBM_SET_UINT(topology_validation_steps)
    SBM_SET_UINT(topology_min_observations)
    SBM_SET_FLOAT(topology_accept_credit)
    SBM_SET_FLOAT(topology_credit_decay)
    SBM_SET_UINT(topology_prune_patience)
    SBM_SET_FLOAT(topology_prune_credit)
    if (name == "address_execution_mode") {
        config.address_execution_mode = parse_address_execution_mode(value);
        return true;
    }
    if (name == "accepted_channel_retirement") {
        config.accepted_channel_retirement = parse_accepted_channel_retirement(value);
        return true;
    }
    SBM_SET_FLOAT(structural_description_cost_weight)
    SBM_SET_FLOAT(structural_execution_cost_weight)
    SBM_SET_FLOAT(binding_reuse_value_weight)
    SBM_SET_BOOL(topology_accept_uses_structural_value)
    SBM_SET_FLOAT(residual_channel_gain)
    SBM_SET_FLOAT(residual_learning_rate)
    SBM_SET_FLOAT(residual_mature_learning_rate)
    SBM_SET_FLOAT(residual_recency_pseudocount)
    SBM_SET_UINT(warm_visits)
    SBM_SET_UINT(mature_visits)
    SBM_SET_FLOAT(dormant_utility)
    SBM_SET_FLOAT(node_learning_rate)
    SBM_SET_FLOAT(mature_learning_rate)
    SBM_SET_FLOAT(merge_similarity)
    SBM_SET_FLOAT(merge_vector_distance)
    SBM_SET_FLOAT(classification_learning_rate)
    SBM_SET_FLOAT(classification_mature_learning_rate)
    SBM_SET_FLOAT(label_smoothing)
    SBM_SET_FLOAT(softmax_temperature)
    SBM_SET_FLOAT(logit_decay)
    SBM_SET_BOOL(sparse_token_output)
    SBM_SET_UINT(sparse_output_topk)
    SBM_SET_UINT(sparse_output_beam_width)
    SBM_SET_UINT(max_sparse_decisions_per_node)
    SBM_SET_BOOL(decode_token_ranking_during_training)
    SBM_SET_BOOL(use_momentum)
    SBM_SET_FLOAT(momentum_beta1)
    SBM_SET_FLOAT(momentum_beta2)
    SBM_SET_FLOAT(momentum_eps)
    SBM_SET_BOOL(record_channel_attribution)
    SBM_SET_UINT(max_binding_reuse_records_per_channel)
    SBM_SET_BOOL(gpaf_shadow_observation)
    SBM_SET_BOOL(gpaf_candidate_retrieval)
    SBM_SET_UINT(gpaf_query_keys_per_step)
    SBM_SET_UINT(gpaf_slots)
    SBM_SET_UINT(gpaf_residents_per_slot)
    SBM_SET_U64(output_tree_seed)
    SBM_SET_U64(seed)
#undef SBM_SET_UINT
#undef SBM_SET_U64
#undef SBM_SET_FLOAT
#undef SBM_SET_DOUBLE
#undef SBM_SET_BOOL
    return false;
}

std::string config_json(const sbm::Config& c) {
    std::ostringstream out;
    out << std::setprecision(9) << std::boolalpha << "{\n"
        << "  \"token_alphabet\": " << c.token_alphabet << ",\n"
        << "  \"vector_dim\": " << c.vector_dim << ",\n"
        << "  \"context_width\": " << c.context_width << ",\n"
        << "  \"bucket_bits\": " << c.bucket_bits << ",\n"
        << "  \"beam_width\": " << c.beam_width << ",\n"
        << "  \"beam_width_min\": " << c.beam_width_min << ",\n"
        << "  \"confidence_threshold\": " << c.confidence_threshold << ",\n"
        << "  \"max_refinement_rounds\": " << c.max_refinement_rounds << ",\n"
        << "  \"refinement_confidence_threshold\": " << c.refinement_confidence_threshold << ",\n"
        << "  \"bucket_scan_limit\": " << c.bucket_scan_limit << ",\n"
        << "  \"edge_scan_limit\": " << c.edge_scan_limit << ",\n"
        << "  \"max_edges_per_node\": " << c.max_edges_per_node << ",\n"
        << "  \"edge_reinforce_width\": " << c.edge_reinforce_width << ",\n"
        << "  \"max_specializations_per_bucket\": " << c.max_specializations_per_bucket << ",\n"
        << "  \"split_min_visits\": " << c.split_min_visits << ",\n"
        << "  \"split_cooldown\": " << c.split_cooldown << ",\n"
        << "  \"split_context_similarity\": " << c.split_context_similarity << ",\n"
        << "  \"split_loss_threshold\": " << c.split_loss_threshold << ",\n"
        << "  \"allow_growth_when_frozen\": " << c.allow_growth_when_frozen << ",\n"
        << "  \"trace_horizon\": " << c.trace_horizon << ",\n"
        << "  \"trace_decay\": " << c.trace_decay << ",\n"
        << "  \"edge_decay\": " << c.edge_decay << ",\n"
        << "  \"edge_learning_rate\": " << c.edge_learning_rate << ",\n"
        << "  \"edge_score_weight\": " << c.edge_score_weight << ",\n"
        << "  \"edge_min_contribution\": " << c.edge_min_contribution << ",\n"
        << "  \"responsibility_temperature\": " << c.responsibility_temperature << ",\n"
        << "  \"exact_region_mass\": " << c.exact_region_mass << ",\n"
        << "  \"min_update_responsibility\": " << c.min_update_responsibility << ",\n"
        << "  \"address_lags\": [";
    for (std::size_t i = 0; i < c.address_lags.size(); ++i) {
        if (i != 0U) out << ", ";
        out << c.address_lags[i];
    }
    out << "],\n"
        << "  \"adaptive_topology\": " << c.adaptive_topology << ",\n"
        << "  \"max_address_channels\": " << c.max_address_channels << ",\n"
        << "  \"topology_max_lag\": " << c.topology_max_lag << ",\n"
        << "  \"topology_max_arity\": " << c.topology_max_arity << ",\n"
        << "  \"topology_enable_delta\": " << c.topology_enable_delta << ",\n"
        << "  \"topology_enable_content_match\": "
        << c.topology_enable_content_match << ",\n"
        << "  \"topology_enable_content_follow_multi\": "
        << c.topology_enable_content_follow_multi << ",\n"
        << "  \"topology_probe_interval\": " << c.topology_probe_interval << ",\n"
        << "  \"topology_probe_warmup\": " << c.topology_probe_warmup << ",\n"
        << "  \"topology_probe_steps\": " << c.topology_probe_steps << ",\n"
        << "  \"topology_validation_steps\": " << c.topology_validation_steps << ",\n"
        << "  \"topology_min_observations\": " << c.topology_min_observations << ",\n"
        << "  \"topology_accept_credit\": " << c.topology_accept_credit << ",\n"
        << "  \"topology_credit_decay\": " << c.topology_credit_decay << ",\n"
        << "  \"topology_prune_patience\": " << c.topology_prune_patience << ",\n"
        << "  \"topology_prune_credit\": " << c.topology_prune_credit << ",\n"
        << "  \"address_execution_mode\": \""
        << to_string(c.address_execution_mode) << "\",\n"
        << "  \"accepted_channel_retirement\": \""
        << to_string(c.accepted_channel_retirement) << "\",\n"
        << "  \"structural_description_cost_weight\": "
        << c.structural_description_cost_weight << ",\n"
        << "  \"structural_execution_cost_weight\": "
        << c.structural_execution_cost_weight << ",\n"
        << "  \"binding_reuse_value_weight\": "
        << c.binding_reuse_value_weight << ",\n"
        << "  \"topology_accept_uses_structural_value\": "
        << c.topology_accept_uses_structural_value << ",\n"
        << "  \"residual_channel_gain\": " << c.residual_channel_gain << ",\n"
        << "  \"residual_learning_rate\": " << c.residual_learning_rate << ",\n"
        << "  \"residual_mature_learning_rate\": " << c.residual_mature_learning_rate << ",\n"
        << "  \"residual_recency_pseudocount\": " << c.residual_recency_pseudocount << ",\n"
        << "  \"warm_visits\": " << c.warm_visits << ",\n"
        << "  \"mature_visits\": " << c.mature_visits << ",\n"
        << "  \"dormant_utility\": " << c.dormant_utility << ",\n"
        << "  \"node_learning_rate\": " << c.node_learning_rate << ",\n"
        << "  \"mature_learning_rate\": " << c.mature_learning_rate << ",\n"
        << "  \"merge_similarity\": " << c.merge_similarity << ",\n"
        << "  \"merge_vector_distance\": " << c.merge_vector_distance << ",\n"
        << "  \"classification_learning_rate\": " << c.classification_learning_rate << ",\n"
        << "  \"classification_mature_learning_rate\": " << c.classification_mature_learning_rate << ",\n"
        << "  \"label_smoothing\": " << c.label_smoothing << ",\n"
        << "  \"softmax_temperature\": " << c.softmax_temperature << ",\n"
        << "  \"logit_decay\": " << c.logit_decay << ",\n"
        << "  \"sparse_token_output\": " << c.sparse_token_output << ",\n"
        << "  \"sparse_output_topk\": " << c.sparse_output_topk << ",\n"
        << "  \"sparse_output_beam_width\": " << c.sparse_output_beam_width << ",\n"
        << "  \"max_sparse_decisions_per_node\": "
        << c.max_sparse_decisions_per_node << ",\n"
        << "  \"decode_token_ranking_during_training\": "
        << c.decode_token_ranking_during_training << ",\n"
        << "  \"use_momentum\": " << c.use_momentum << ",\n"
        << "  \"momentum_beta1\": " << c.momentum_beta1 << ",\n"
        << "  \"momentum_beta2\": " << c.momentum_beta2 << ",\n"
        << "  \"momentum_eps\": " << c.momentum_eps << ",\n"
        << "  \"record_channel_attribution\": "
        << c.record_channel_attribution << ",\n"
        << "  \"max_binding_reuse_records_per_channel\": "
        << c.max_binding_reuse_records_per_channel << ",\n"
        << "  \"gpaf_shadow_observation\": "
        << c.gpaf_shadow_observation << ",\n"
        << "  \"gpaf_candidate_retrieval\": "
        << c.gpaf_candidate_retrieval << ",\n"
        << "  \"gpaf_query_keys_per_step\": "
        << c.gpaf_query_keys_per_step << ",\n"
        << "  \"gpaf_slots\": " << c.gpaf_slots << ",\n"
        << "  \"gpaf_residents_per_slot\": "
        << c.gpaf_residents_per_slot << ",\n"
        << "  \"output_tree_seed\": " << c.output_tree_seed << ",\n"
        << "  \"seed\": " << c.seed << "\n"
        << "}\n";
    return out.str();
}

std::string_view parameter_tasks(std::string_view name) {
    if (name == "classification_learning_rate" ||
        name == "classification_mature_learning_rate" ||
        name == "label_smoothing" ||
        name == "softmax_temperature" ||
        name == "logit_decay" ||
        name == "sparse_output_topk" ||
        name == "sparse_output_beam_width" ||
        name == "max_sparse_decisions_per_node" ||
        name == "decode_token_ranking_during_training" ||
        name == "use_momentum" ||
        name == "momentum_beta1" ||
        name == "momentum_beta2" ||
        name == "momentum_eps" ||
        name == "record_channel_attribution" ||
        name == "binding_reuse_value_weight" ||
        name == "max_binding_reuse_records_per_channel" ||
        name == "gpaf_shadow_observation" ||
        name == "gpaf_candidate_retrieval" ||
        name == "gpaf_query_keys_per_step" ||
        name == "gpaf_slots" ||
        name == "gpaf_residents_per_slot") {
        return "token-ce";
    }
    if (name == "residual_learning_rate" ||
        name == "residual_mature_learning_rate" ||
        name == "residual_recency_pseudocount" ||
        name == "node_learning_rate" ||
        name == "mature_learning_rate") {
        return "vector";
    }
    return "both";
}

std::string schema_json() {
    std::ostringstream out;
    out << "{\n  \"api_version\": 9,\n  \"parameters\": [\n";
    for (std::size_t i = 0; i < std::size(kParameters); ++i) {
        const auto& p = kParameters[i];
        out << "    {\"name\": \"" << json_escape(p.name)
            << "\", \"type\": \"" << p.type
            << "\", \"default\": \"" << p.default_value
            << "\", \"tunable\": " << (p.tunable ? "true" : "false")
            << ", \"search_default\": " << (p.search_default ? "true" : "false")
            << ", \"dataset_bound\": " << (p.dataset_bound ? "true" : "false")
            << ", \"scale\": \"" << p.scale << "\"";
        if (p.minimum[0] != '\0') out << ", \"minimum\": " << p.minimum;
        if (p.maximum[0] != '\0') out << ", \"maximum\": " << p.maximum;
        const auto tasks = parameter_tasks(p.name);
        out << ", \"tasks\": [";
        if (tasks == "both") out << "\"token-ce\", \"vector\"";
        else out << "\"" << tasks << "\"";
        out << "], \"description\": \"" << json_escape(p.description) << "\"}";
        if (i + 1U != std::size(kParameters)) out << ',';
        out << '\n';
    }
    out << "  ]\n}\n";
    return out.str();
}

const std::string& static_schema() {
    static const std::string value = schema_json();
    return value;
}

} // namespace

extern "C" {

uint32_t sbm_api_version(void) { return 9U; }
const char* sbm_api_version_string(void) { return "9.0.0"; }
const char* sbm_last_error(void) { return g_last_error.c_str(); }
const char* sbm_parameter_schema_json(void) { return static_schema().c_str(); }

sbm_config_handle* sbm_config_create(void) {
    return guarded([] { return new sbm_config_handle{}; });
}

sbm_config_handle* sbm_config_clone(const sbm_config_handle* source) {
    return guarded([&] {
        if (source == nullptr) throw std::invalid_argument("config is null");
        return new sbm_config_handle{source->value};
    });
}

void sbm_config_destroy(sbm_config_handle* config) { delete config; }

int sbm_config_set(sbm_config_handle* config, const char* name, const char* value) {
    clear_error();
    try {
        if (config == nullptr || name == nullptr || value == nullptr) {
            throw std::invalid_argument("config, name and value must be non-null");
        }
        if (!set_parameter(config->value, name, value)) {
            throw std::invalid_argument(std::string("unknown parameter: ") + name);
        }
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return -1;
    } catch (...) {
        set_error("unknown C++ exception");
        return -1;
    }
}

char* sbm_config_get_json(const sbm_config_handle* config) {
    return guarded([&] {
        if (config == nullptr) throw std::invalid_argument("config is null");
        return duplicate_string(config_json(config->value));
    });
}

sbm_machine_handle* sbm_machine_create(const sbm_config_handle* config) {
    return guarded([&] {
        if (config == nullptr) throw std::invalid_argument("config is null");
        auto resolved = config->value;
        resolved.objective = sbm::ObjectiveKind::TokenCrossEntropy;
        resolved.token_alphabet = resolved.vector_dim;
        return new sbm_machine_handle{sbm::SparseBranchMachine(resolved)};
    });
}

sbm_machine_handle* sbm_machine_load_checkpoint(const char* path) {
    return guarded([&] {
        if (path == nullptr) throw std::invalid_argument("path is null");
        return new sbm_machine_handle{sbm::load_checkpoint(path)};
    });
}

int sbm_machine_save_checkpoint(const sbm_machine_handle* machine,
                                const char* path) {
    clear_error();
    try {
        if (machine == nullptr || path == nullptr) {
            throw std::invalid_argument("machine and path must be non-null");
        }
        sbm::save_checkpoint(machine->value, path);
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return -1;
    } catch (...) {
        set_error("unknown C++ exception");
        return -1;
    }
}

void sbm_machine_destroy(sbm_machine_handle* machine) { delete machine; }

int sbm_machine_step_token(sbm_machine_handle* machine,
                           uint32_t input_token,
                           uint32_t target_token,
                           int learn,
                           sbm_step_stats* stats) {
    clear_error();
    try {
        if (machine == nullptr || stats == nullptr) {
            throw std::invalid_argument("machine and stats must be non-null");
        }
        const auto result = machine->value.step_token(
            input_token, target_token, learn != 0);
        stats->cross_entropy = result.cross_entropy;
        stats->target_probability = result.target_probability;
        stats->active_nodes = result.active_nodes;
        stats->candidates_examined = result.candidates_examined;
        stats->live_nodes = result.live_nodes;
        stats->predicted_token = result.predicted_token;
        stats->top1_correct = result.top1_correct ? 1 : 0;
        stats->top5_correct = result.top5_correct ? 1 : 0;
        stats->ranking_available = result.ranking_available ? 1 : 0;
        stats->channel_credit_count = result.channel_credit_count;
        for (std::size_t i = 0; i < sbm::kMaxAddressChannels; ++i) {
            stats->channel_credit[i] = result.channel_credit[i];
            stats->channel_responsibility_mass[i] =
                result.channel_responsibility_mass[i];
            stats->channel_dependency[i] = result.channel_dependency[i];
            stats->channel_parent_channel[i] =
                result.channel_parent_channel[i];
            stats->channel_dependency_channel[i] =
                result.channel_dependency_channel[i];
            stats->channel_input_state[i] = result.channel_input_state[i];
            stats->channel_output_state[i] = result.channel_output_state[i];
            stats->channel_required_dependency_binding[i] =
                result.channel_required_dependency_binding[i];
            stats->channel_caller_removed_credit[i] =
                result.channel_caller_removed_credit[i];
            stats->channel_dependency_retained_credit[i] =
                result.channel_dependency_retained_credit[i];
            stats->channel_dependency_removed_credit[i] =
                result.channel_dependency_removed_credit[i];
            stats->channel_binding_kind[i] = result.channel_binding_kind[i];
            stats->channel_binding_matched[i] =
                result.channel_binding_matched[i];
            stats->channel_binding_current_token[i] =
                result.channel_binding_current_token[i];
            stats->channel_binding_matched_token[i] =
                result.channel_binding_matched_token[i];
            stats->channel_binding_successor[i] =
                result.channel_binding_successor[i];
            stats->channel_binding_distance[i] =
                result.channel_binding_distance[i];
            stats->channel_binding_pattern_span[i] =
                result.channel_binding_pattern_span[i];
            stats->channel_binding_key[i] = result.channel_binding_key[i];
            stats->channel_dependency_binding_key[i] =
                result.channel_dependency_binding_key[i];
            stats->channel_call_key[i] = result.channel_call_key[i];
            stats->channel_description_cost[i] = result.channel_description_cost[i];
            stats->channel_execution_cost[i] = result.channel_execution_cost[i];
        }
        stats->channel_subset_available =
            result.channel_subset_available ? 1 : 0;
        stats->seed_only_cross_entropy = result.seed_only_cross_entropy;
        stats->active_only_cross_entropy = result.active_only_cross_entropy;
        stats->content_only_cross_entropy = result.content_only_cross_entropy;
        stats->tuple_only_cross_entropy = result.tuple_only_cross_entropy;
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return -1;
    } catch (...) {
        set_error("unknown C++ exception");
        return -1;
    }
}

void sbm_machine_reset_sequence(sbm_machine_handle* machine) {
    if (machine != nullptr) machine->value.reset_sequence();
}

void sbm_machine_freeze_topology(sbm_machine_handle* machine) {
    if (machine != nullptr) machine->value.freeze_topology();
}

int sbm_machine_retire_channel(sbm_machine_handle* machine,
                               uint32_t channel,
                               uint32_t retirement_policy) {
    return guarded([&] {
        if (machine == nullptr) throw std::invalid_argument("machine is null");
        if (retirement_policy >
            static_cast<uint32_t>(
                sbm::AcceptedChannelRetirement::RecoverableRetire)) {
            throw std::invalid_argument("invalid retirement policy");
        }
        const auto policy =
            static_cast<sbm::AcceptedChannelRetirement>(retirement_policy);
        return machine->value.retire_channel(channel, policy) ? 1 : 0;
    });
}

int sbm_machine_restore_channel(sbm_machine_handle* machine, uint32_t channel) {
    return guarded([&] {
        if (machine == nullptr) throw std::invalid_argument("machine is null");
        return machine->value.restore_channel(channel) ? 1 : 0;
    });
}

int sbm_machine_restore_dependency_closure(sbm_machine_handle* machine,
                                           uint32_t channel) {
    return guarded([&] {
        if (machine == nullptr) throw std::invalid_argument("machine is null");
        const auto restored = machine->value.restore_dependency_closure(channel);
        if (restored > static_cast<std::size_t>(INT32_MAX)) {
            throw std::overflow_error("restored channel count overflows int");
        }
        return static_cast<int>(restored);
    });
}

char* sbm_machine_diagnostics_json(const sbm_machine_handle* machine) {
    return guarded([&] {
        if (machine == nullptr) throw std::invalid_argument("machine is null");
        const auto diagnostics = machine->value.diagnostics();
        std::ostringstream out;
        out << "{"
            << "\"steps\": " << diagnostics.steps
            << ", \"live_nodes\": " << diagnostics.live_nodes
            << ", \"logical_ids_issued\": " << diagnostics.logical_ids_issued
            << ", \"avg_active\": " << diagnostics.avg_active
            << ", \"avg_candidates\": " << diagnostics.avg_candidates
            << ", \"candidate_source_exact_bucket\": "
            << diagnostics.candidate_source_exact_bucket
            << ", \"candidate_source_control_edge\": "
            << diagnostics.candidate_source_control_edge
            << ", \"candidate_source_neighbor_bucket\": "
            << diagnostics.candidate_source_neighbor_bucket
            << ", \"route_score_hamming_sum\": "
            << diagnostics.route_score_hamming_sum
            << ", \"route_score_exact_sum\": "
            << diagnostics.route_score_exact_sum
            << ", \"route_score_edge_prior_sum\": "
            << diagnostics.route_score_edge_prior_sum
            << ", \"gpaf_role_observations\": "
            << diagnostics.gpaf_role_observations
            << ", \"gpaf_unique_role_keys\": "
            << diagnostics.gpaf_unique_role_keys
            << ", \"gpaf_slots_allocated\": "
            << diagnostics.gpaf_slots_allocated
            << ", \"gpaf_probe_slots\": "
            << diagnostics.gpaf_probe_slots
            << ", \"gpaf_active_slots\": "
            << diagnostics.gpaf_active_slots
            << ", \"gpaf_quarantined_slots\": "
            << diagnostics.gpaf_quarantined_slots
            << ", \"gpaf_recoverable_retired_slots\": "
            << diagnostics.gpaf_recoverable_retired_slots
            << ", \"gpaf_physically_erased_slots\": "
            << diagnostics.gpaf_physically_erased_slots
            << ", \"gpaf_shadow_updates\": "
            << diagnostics.gpaf_shadow_updates
            << ", \"gpaf_slots_probed\": "
            << diagnostics.gpaf_slots_probed
            << ", \"gpaf_candidates_returned\": "
            << diagnostics.gpaf_candidates_returned
            << ", \"address_execution_frames\": "
            << diagnostics.address_execution_frames
            << ", \"address_binding_hits\": " << diagnostics.address_binding_hits
            << ", \"address_binding_misses\": "
            << diagnostics.address_binding_misses
            << ", \"binding_reuse_observations\": "
            << diagnostics.binding_reuse_observations
            << ", \"binding_reuse_unique_keys\": "
            << diagnostics.binding_reuse_unique_keys
            << ", \"binding_reuse_events\": "
            << diagnostics.binding_reuse_events
            << ", \"address_binding_by_kind\": [";
        for (std::size_t index = 0U; index < sbm::kAddressBindingKindCount; ++index) {
            if (index != 0U) out << ", ";
            const auto frames = diagnostics.address_binding_kind_frames[index];
            const auto hits = diagnostics.address_binding_kind_hits[index];
            const double hit_rate = frames == 0U ? 0.0 :
                static_cast<double>(hits) / static_cast<double>(frames);
            const double mean_distance = hits == 0U ? 0.0 :
                diagnostics.address_binding_kind_distance_sum[index] /
                    static_cast<double>(hits);
            const double mean_pattern_span = hits == 0U ? 0.0 :
                diagnostics.address_binding_kind_pattern_span_sum[index] /
                    static_cast<double>(hits);
            out << "{\"kind\":" << index
                << ",\"frames\":" << frames
                << ",\"hits\":" << hits
                << ",\"hit_rate\":" << hit_rate
                << ",\"mean_distance\":" << mean_distance
                << ",\"mean_pattern_span\":" << mean_pattern_span << "}";
        }
        out << "]"
            << ", \"topology_proposals\": " << diagnostics.topology_proposals
            << ", \"topology_accepted\": " << diagnostics.topology_accepted
            << ", \"topology_rejected\": " << diagnostics.topology_rejected
            << ", \"topology_pruned\": " << diagnostics.topology_pruned
            << ", \"topology_restored\": " << diagnostics.topology_restored
            << ", \"dependency_blocked_channels\": "
            << diagnostics.dependency_blocked_channels
            << "}";
        return duplicate_string(out.str());
    });
}

char* sbm_machine_summary_json(const sbm_machine_handle* machine) {
    return guarded([&] {
        if (machine == nullptr) throw std::invalid_argument("machine is null");
        const auto programs = machine->value.learned_address_programs();
        const auto lags = machine->value.learned_address_lags();
        const auto credit = machine->value.learned_channel_credit();
        const auto phase = machine->value.learned_channel_phase();
        const auto effective_enabled =
            machine->value.learned_channel_effective_enabled();
        const auto parent = machine->value.learned_channel_parent();
        const auto dependency = machine->value.learned_channel_dependency();
        const auto generation = machine->value.learned_channel_generation();
        const auto parent_edge_kind =
            machine->value.learned_channel_parent_edge_kind();
        const auto dependency_edge_kind =
            machine->value.learned_channel_dependency_edge_kind();
        std::ostringstream out;
        out << "{";
        out << "\"learned_address_lags\": [";
        for (std::size_t i = 0; i < lags.size(); ++i) {
            if (i != 0U) out << ", ";
            out << lags[i];
        }
        out << "], \"learned_address_programs\": [";
        for (std::size_t i = 0; i < programs.size(); ++i) {
            if (i != 0U) out << ", ";
            out << '[';
            for (std::size_t j = 0; j < programs[i].arity; ++j) {
                if (j != 0U) out << ", ";
                out << programs[i].lags[j];
            }
            out << ']';
        }
        out << "], \"learned_address_operations\": [";
        for (std::size_t i = 0; i < programs.size(); ++i) {
            if (i != 0U) out << ", ";
            out << static_cast<unsigned>(programs[i].op);
        }
        out << "], \"learned_channel_credit\": [";
        for (std::size_t i = 0; i < credit.size(); ++i) {
            if (i != 0U) out << ", ";
            out << credit[i];
        }
        out << "], \"learned_channel_phase\": [";
        for (std::size_t i = 0; i < phase.size(); ++i) {
            if (i != 0U) out << ", ";
            out << static_cast<unsigned>(phase[i]);
        }
        out << "], \"learned_channel_effective_enabled\": [";
        for (std::size_t i = 0; i < effective_enabled.size(); ++i) {
            if (i != 0U) out << ", ";
            out << static_cast<unsigned>(effective_enabled[i]);
        }
        out << "], \"learned_channel_parent\": [";
        for (std::size_t i = 0; i < parent.size(); ++i) {
            if (i != 0U) out << ", ";
            out << static_cast<unsigned>(parent[i]);
        }
        out << "], \"learned_channel_dependency\": [";
        for (std::size_t i = 0; i < dependency.size(); ++i) {
            if (i != 0U) out << ", ";
            out << static_cast<unsigned>(dependency[i]);
        }
        out << "], \"learned_channel_generation\": [";
        for (std::size_t i = 0; i < generation.size(); ++i) {
            if (i != 0U) out << ", ";
            out << generation[i];
        }
        out << "], \"learned_channel_parent_edge_kind\": [";
        for (std::size_t i = 0; i < parent_edge_kind.size(); ++i) {
            if (i != 0U) out << ", ";
            out << static_cast<unsigned>(parent_edge_kind[i]);
        }
        out << "], \"learned_channel_dependency_edge_kind\": [";
        for (std::size_t i = 0; i < dependency_edge_kind.size(); ++i) {
            if (i != 0U) out << ", ";
            out << static_cast<unsigned>(dependency_edge_kind[i]);
        }
        std::vector<std::uint64_t> caller_counts(programs.size(), 0U);
        std::vector<std::uint64_t> effective_caller_counts(programs.size(), 0U);
        std::vector<std::uint64_t> blocked_caller_counts(programs.size(), 0U);
        for (std::size_t caller = 0U; caller < dependency.size(); ++caller) {
            const auto dependency_channel = dependency[caller];
            if (dependency_channel < caller_counts.size() &&
                dependency_channel != caller) {
                ++caller_counts[dependency_channel];
                const bool caller_effective = caller < effective_enabled.size() &&
                    effective_enabled[caller] != 0U;
                const bool caller_committed = caller < phase.size() &&
                    (phase[caller] ==
                         static_cast<std::uint8_t>(sbm::ChannelPhase::Seed) ||
                     phase[caller] ==
                         static_cast<std::uint8_t>(sbm::ChannelPhase::Probe) ||
                     phase[caller] ==
                         static_cast<std::uint8_t>(sbm::ChannelPhase::Active));
                if (caller_effective) {
                    ++effective_caller_counts[dependency_channel];
                } else if (caller_committed) {
                    ++blocked_caller_counts[dependency_channel];
                }
            }
        }
        out << "], \"address_dependency_graph\": [";
        for (std::size_t i = 0; i < programs.size(); ++i) {
            if (i != 0U) out << ", ";
            out << "{\"channel\":" << i << ",\"lags\":[";
            for (std::size_t j = 0; j < programs[i].arity; ++j) {
                if (j != 0U) out << ',';
                out << programs[i].lags[j];
            }
            out << "],\"op\":" << static_cast<unsigned>(programs[i].op)
                << ",\"phase\":"
                << (i < phase.size() ? static_cast<unsigned>(phase[i])
                                      : static_cast<unsigned>(
                                            sbm::ChannelPhase::Retired))
                << ",\"parent_channel\":"
                << (i < parent.size() ? static_cast<unsigned>(parent[i])
                                      : static_cast<unsigned>(
                                            sbm::kInvalidChannel))
                << ",\"dependency_channel\":"
                << (i < dependency.size()
                        ? static_cast<unsigned>(dependency[i])
                        : static_cast<unsigned>(sbm::kInvalidChannel))
                << ",\"channel_generation\":"
                << (i < generation.size() ? generation[i] : 0U)
                << ",\"parent_edge_kind\":"
                << (i < parent_edge_kind.size()
                        ? static_cast<unsigned>(parent_edge_kind[i])
                        : static_cast<unsigned>(sbm::AddressGraphEdgeKind::None))
                << ",\"dependency_edge_kind\":"
                << (i < dependency_edge_kind.size()
                        ? static_cast<unsigned>(dependency_edge_kind[i])
                        : static_cast<unsigned>(sbm::AddressGraphEdgeKind::None))
                << ",\"effective_enabled\":"
                << (i < effective_enabled.size()
                        ? static_cast<unsigned>(effective_enabled[i])
                        : 0U)
                << ",\"dependency_available\":"
                << (i < dependency.size() &&
                    dependency[i] != sbm::kInvalidChannel &&
                    dependency[i] < effective_enabled.size()
                        ? static_cast<unsigned>(
                              effective_enabled[dependency[i]])
                        : 1U)
                << ",\"input_state\":"
                << static_cast<unsigned>(
                       sbm::address_program_input_state(programs[i]))
                << ",\"output_state\":"
                << static_cast<unsigned>(
                       sbm::address_program_output_state(programs[i]))
                << ",\"required_dependency_binding\":"
                << static_cast<unsigned>(
                       sbm::address_program_required_dependency_binding(programs[i]))
                << ",\"direct_caller_count\":" << caller_counts[i]
                << ",\"effective_direct_caller_count\":"
                << effective_caller_counts[i]
                << ",\"blocked_direct_caller_count\":"
                << blocked_caller_counts[i]
                << ",\"own_call_matches\":0"
                << ",\"unique_call_keys\":0"
                << ",\"call_key_reuse_events\":0"
                << ",\"downstream_call_matches\":0}";
        }
        out << "], \"address_dependency_edges\": [";
        bool first_edge = true;
        const auto emit_edge = [&](std::size_t caller,
                                   std::uint8_t target,
                                   sbm::AddressGraphEdgeKind kind) {
            if (target == sbm::kInvalidChannel || target >= programs.size() ||
                target == caller || kind == sbm::AddressGraphEdgeKind::None) {
                return;
            }
            if (!first_edge) out << ", ";
            first_edge = false;
            out << "{\"caller_channel\":" << caller
                << ",\"dependency_channel\":"
                << static_cast<unsigned>(target)
                << ",\"caller_generation\":"
                << (caller < generation.size() ? generation[caller] : 0U)
                << ",\"dependency_generation\":"
                << (target < generation.size() ? generation[target] : 0U)
                << ",\"edge_kind\":"
                << static_cast<unsigned>(kind)
                << ",\"caller_input_state\":"
                << static_cast<unsigned>(
                       sbm::address_program_input_state(programs[caller]))
                << ",\"dependency_output_state\":"
                << static_cast<unsigned>(
                       sbm::address_program_output_state(
                           programs[target]))
                << ",\"required_dependency_binding\":"
                << static_cast<unsigned>(
                       sbm::address_program_required_dependency_binding(
                           programs[caller]))
                << ",\"observations\":0"
                << ",\"call_matches\":0"
                << ",\"unique_call_keys\":0"
                << ",\"call_key_reuse_events\":0"
                << ",\"caller_removed_credit_sum\":0"
                << ",\"dependency_removed_credit_sum\":0}";
        };
        for (std::size_t caller = 0U; caller < programs.size() &&
             caller < dependency.size() && caller < parent.size(); ++caller) {
            const auto parent_channel = parent[caller];
            const auto dependency_channel = dependency[caller];
            if (parent_channel == dependency_channel) {
                emit_edge(caller, dependency_channel,
                          caller < dependency_edge_kind.size()
                              ? static_cast<sbm::AddressGraphEdgeKind>(
                                    dependency_edge_kind[caller])
                              : sbm::AddressGraphEdgeKind::None);
                continue;
            }
            emit_edge(caller, parent_channel,
                      caller < parent_edge_kind.size()
                          ? static_cast<sbm::AddressGraphEdgeKind>(
                                parent_edge_kind[caller])
                          : sbm::AddressGraphEdgeKind::None);
            emit_edge(caller, dependency_channel,
                      caller < dependency_edge_kind.size()
                          ? static_cast<sbm::AddressGraphEdgeKind>(
                                dependency_edge_kind[caller])
                          : sbm::AddressGraphEdgeKind::None);
        }
        out << "], \"topology_decision_name_map\": [";
        for (unsigned decision = 0U;
             decision <= static_cast<unsigned>(sbm::TopologyDecision::Restored);
             ++decision) {
            if (decision != 0U) out << ", ";
            const auto value = static_cast<sbm::TopologyDecision>(decision);
            out << "{\"decision\":" << decision
                << ",\"decision_name\":\""
                << sbm::topology_decision_name(value) << "\"}";
        }
        out << "], \"topology_events\": [";
        const auto& events = machine->value.topology_events();
        for (std::size_t i = 0; i < events.size(); ++i) {
            if (i != 0U) out << ", ";
            const auto& event = events[i];
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
                << sbm::topology_decision_name(event.decision) << "\""
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
        out << "]}";
        return duplicate_string(out.str());
    });
}

sbm_dataset_handle* sbm_dataset_generate(size_t length, uint32_t alphabet,
                                         uint32_t vector_dim, uint32_t hidden_dim,
                                         uint64_t seed, float noise_std) {
    return guarded([&] {
        return new sbm_dataset_handle{sbm::generate_vector_process(
            length, alphabet, vector_dim, hidden_dim, seed, noise_std)};
    });
}

sbm_dataset_handle* sbm_token_dataset_generate_math(size_t sequence_count,
                                                     size_t sequence_length,
                                                     uint32_t vocab_size,
                                                     uint64_t seed,
                                                     float temperature,
                                                     float interaction_strength) {
    return guarded([&] {
        return new sbm_dataset_handle{sbm::generate_math_token_process(
            sequence_count, sequence_length, vocab_size, seed,
            temperature, interaction_strength)};
    });
}

sbm_dataset_handle* sbm_token_dataset_from_ids(const uint32_t* tokens,
                                                size_t token_count,
                                                uint32_t vocab_size,
                                                const uint64_t* sequence_offsets,
                                                size_t offset_count) {
    return guarded([&] {
        if (tokens == nullptr || token_count == 0U) {
            throw std::invalid_argument("tokens must be non-null and non-empty");
        }
        if ((sequence_offsets == nullptr) != (offset_count == 0U)) {
            throw std::invalid_argument(
                "sequence_offsets must be null iff offset_count is zero");
        }
        const std::span<const std::uint32_t> token_span(tokens, token_count);
        const std::span<const std::uint64_t> offset_span = sequence_offsets == nullptr
            ? std::span<const std::uint64_t>{}
            : std::span<const std::uint64_t>(sequence_offsets, offset_count);
        return new sbm_dataset_handle{
            sbm::make_token_dataset(token_span, vocab_size, offset_span)};
    });
}

sbm_dataset_handle* sbm_dataset_load(const char* path) {
    return guarded([&] {
        if (path == nullptr) throw std::invalid_argument("path is null");
        if (sbm::inspect_dataset_kind(path) == sbm::DatasetKind::TokenCrossEntropy) {
            return new sbm_dataset_handle{sbm::load_token_dataset(path)};
        }
        return new sbm_dataset_handle{sbm::load_dataset(path)};
    });
}

int sbm_dataset_save(const sbm_dataset_handle* dataset, const char* path) {
    clear_error();
    try {
        if (dataset == nullptr || path == nullptr) {
            throw std::invalid_argument("dataset and path must be non-null");
        }
        std::visit([&](const auto& value) {
            using Dataset = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Dataset, sbm::VectorDataset>) {
                sbm::save_dataset(value, path);
            } else {
                sbm::save_token_dataset(value, path);
            }
        }, dataset->value);
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return -1;
    } catch (...) {
        set_error("unknown C++ exception");
        return -1;
    }
}

void sbm_dataset_destroy(sbm_dataset_handle* dataset) { delete dataset; }

uint32_t sbm_dataset_kind_of(const sbm_dataset_handle* dataset) {
    if (dataset == nullptr) return 0U;
    return std::holds_alternative<sbm::VectorDataset>(dataset->value)
        ? static_cast<uint32_t>(SBM_DATASET_VECTOR_REGRESSION)
        : static_cast<uint32_t>(SBM_DATASET_TOKEN_CROSS_ENTROPY);
}

uint64_t sbm_dataset_hash(const sbm_dataset_handle* dataset) {
    if (dataset == nullptr) return 0U;
    return std::visit([](const auto& value) { return sbm::hash_dataset(value); },
                      dataset->value);
}

size_t sbm_dataset_length(const sbm_dataset_handle* dataset) {
    if (dataset == nullptr) return 0U;
    return std::visit([](const auto& value) { return value.tokens.size(); },
                      dataset->value);
}

uint32_t sbm_dataset_vector_dim(const sbm_dataset_handle* dataset) {
    if (dataset == nullptr) return 0U;
    if (const auto* vector = std::get_if<sbm::VectorDataset>(&dataset->value)) {
        return vector->vector_dim;
    }
    return 0U;
}

uint32_t sbm_dataset_alphabet(const sbm_dataset_handle* dataset) {
    if (dataset == nullptr) return 0U;
    if (const auto* vector = std::get_if<sbm::VectorDataset>(&dataset->value)) {
        return vector->token_alphabet;
    }
    return std::get<sbm::TokenDataset>(dataset->value).vocab_size;
}

uint32_t sbm_dataset_vocab_size(const sbm_dataset_handle* dataset) {
    if (dataset == nullptr) return 0U;
    if (const auto* token = std::get_if<sbm::TokenDataset>(&dataset->value)) {
        return token->vocab_size;
    }
    return 0U;
}

size_t sbm_dataset_sequence_count(const sbm_dataset_handle* dataset) {
    if (dataset == nullptr) return 0U;
    if (const auto* token = std::get_if<sbm::TokenDataset>(&dataset->value)) {
        return token->sequence_count();
    }
    return 1U;
}

size_t sbm_dataset_example_count(const sbm_dataset_handle* dataset) {
    if (dataset == nullptr) return 0U;
    if (const auto* token = std::get_if<sbm::TokenDataset>(&dataset->value)) {
        return token->example_count();
    }
    return std::get<sbm::VectorDataset>(dataset->value).tokens.size();
}

int sbm_token_shard_write(const sbm_dataset_handle* dataset, const char* path) {
    clear_error();
    try {
        if (dataset == nullptr || path == nullptr) {
            throw std::invalid_argument("dataset and path must be non-null");
        }
        const auto* token = std::get_if<sbm::TokenDataset>(&dataset->value);
        if (token == nullptr) {
            throw std::invalid_argument("token shard requires a token dataset");
        }
        sbm::write_token_shard(*token, path);
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return -1;
    } catch (...) {
        set_error("unknown C++ exception");
        return -1;
    }
}

sbm_token_shard_handle* sbm_token_shard_open(const char* path,
                                              uint64_t shard_index,
                                              int verify_payload) {
    return guarded([&] {
        if (path == nullptr) throw std::invalid_argument("path is null");
        return new sbm_token_shard_handle{
            sbm::MappedTokenShard(path, shard_index, verify_payload != 0)};
    });
}

void sbm_token_shard_destroy(sbm_token_shard_handle* shard) { delete shard; }

uint32_t sbm_token_shard_vocab_size(const sbm_token_shard_handle* shard) {
    return shard == nullptr ? 0U : shard->value.vocab_size();
}

uint64_t sbm_token_shard_token_count(const sbm_token_shard_handle* shard) {
    return shard == nullptr ? 0U : shard->value.token_count();
}

uint64_t sbm_token_shard_sequence_count(const sbm_token_shard_handle* shard) {
    return shard == nullptr ? 0U : shard->value.sequence_count();
}

uint64_t sbm_token_shard_dataset_hash(const sbm_token_shard_handle* shard) {
    return shard == nullptr ? 0U : shard->value.dataset_hash();
}

int sbm_token_shard_next(const sbm_token_shard_handle* shard,
                         sbm_token_shard_cursor* cursor,
                         uint32_t* input,
                         uint32_t* target) {
    clear_error();
    try {
        if (shard == nullptr || cursor == nullptr || input == nullptr || target == nullptr) {
            throw std::invalid_argument("shard, cursor, input and target must be non-null");
        }
        sbm::TokenShardCursor cpp_cursor{
            cursor->shard_index, cursor->sequence_index, cursor->token_offset};
        sbm::TokenExample example{};
        if (!shard->value.next(cpp_cursor, example)) return 0;
        cursor->shard_index = cpp_cursor.shard_index;
        cursor->sequence_index = cpp_cursor.sequence_index;
        cursor->token_offset = cpp_cursor.token_offset;
        *input = example.input;
        *target = example.target;
        return 1;
    } catch (const std::exception& error) {
        set_error(error.what());
        return -1;
    } catch (...) {
        set_error("unknown C++ exception");
        return -1;
    }
}

char* sbm_run_token_shard_experiment_json(const sbm_token_shard_handle* shard,
                                          size_t warmup,
                                          const sbm_config_handle* config,
                                          int strict_freeze,
                                          size_t prefill,
                                          size_t prune_interval,
                                          size_t merge_interval) {
    return guarded([&] {
        if (shard == nullptr || config == nullptr) {
            throw std::invalid_argument("shard and config must be non-null");
        }
        auto resolved = config->value;
        resolved.objective = sbm::ObjectiveKind::TokenCrossEntropy;
        const auto result = sbm::run_token_experiment(
            shard->value, warmup, resolved, strict_freeze != 0,
            prefill, prune_interval, merge_interval);
        return duplicate_string(sbm::to_json(result));
    });
}

sbm_token_corpus_handle* sbm_token_corpus_create(void) {
    return guarded([] { return new sbm_token_corpus_handle{}; });
}

void sbm_token_corpus_destroy(sbm_token_corpus_handle* corpus) { delete corpus; }

int sbm_token_corpus_add_shard(sbm_token_corpus_handle* corpus,
                               const char* path,
                               uint32_t split_kind,
                               uint64_t shard_index,
                               int verify_payload) {
    clear_error();
    try {
        if (corpus == nullptr || path == nullptr) {
            throw std::invalid_argument("corpus and path must be non-null");
        }
        if (split_kind > 1U) throw std::invalid_argument("invalid corpus split kind");
        auto shard = std::make_unique<sbm::MappedTokenShard>(
            path, shard_index, verify_payload != 0);
        (split_kind == 0U ? corpus->train : corpus->eval).push_back(std::move(shard));
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return -1;
    } catch (...) {
        set_error("unknown C++ exception");
        return -1;
    }
}

char* sbm_run_token_corpus_experiment_json(const sbm_token_corpus_handle* corpus,
                                           const sbm_config_handle* config,
                                           int strict_freeze,
                                           size_t prefill,
                                           size_t prune_interval,
                                           size_t merge_interval) {
    return guarded([&] {
        if (corpus == nullptr || config == nullptr) {
            throw std::invalid_argument("corpus and config must be non-null");
        }
        std::vector<const sbm::MappedTokenShard*> train;
        std::vector<const sbm::MappedTokenShard*> eval;
        train.reserve(corpus->train.size());
        eval.reserve(corpus->eval.size());
        for (const auto& shard : corpus->train) train.push_back(shard.get());
        for (const auto& shard : corpus->eval) eval.push_back(shard.get());
        auto resolved = config->value;
        resolved.objective = sbm::ObjectiveKind::TokenCrossEntropy;
        const auto result = sbm::run_token_corpus_experiment(
            train, eval, resolved, strict_freeze != 0,
            prefill, prune_interval, merge_interval);
        return duplicate_string(sbm::to_json(result));
    });
}

char* sbm_run_token_corpus_experiment_limited_json(
    const sbm_token_corpus_handle* corpus,
    size_t max_train_examples,
    size_t max_eval_examples,
    const sbm_config_handle* config,
    int strict_freeze,
    size_t prefill,
    size_t prune_interval,
    size_t merge_interval) {
    return guarded([&] {
        if (corpus == nullptr || config == nullptr) {
            throw std::invalid_argument("corpus and config must be non-null");
        }
        std::vector<const sbm::MappedTokenShard*> train;
        std::vector<const sbm::MappedTokenShard*> eval;
        train.reserve(corpus->train.size());
        eval.reserve(corpus->eval.size());
        for (const auto& shard : corpus->train) train.push_back(shard.get());
        for (const auto& shard : corpus->eval) eval.push_back(shard.get());
        auto resolved = config->value;
        resolved.objective = sbm::ObjectiveKind::TokenCrossEntropy;
        const auto result = sbm::run_token_corpus_experiment_limited(
            train, eval, max_train_examples, max_eval_examples, resolved,
            strict_freeze != 0, prefill, prune_interval, merge_interval);
        return duplicate_string(sbm::to_json(result));
    });
}

char* sbm_run_experiment_json(const sbm_dataset_handle* dataset, size_t warmup,
                              const sbm_config_handle* config, int strict_freeze,
                              size_t prefill, size_t prune_interval,
                              size_t merge_interval) {
    return guarded([&] {
        if (dataset == nullptr || config == nullptr) {
            throw std::invalid_argument("dataset and config must be non-null");
        }
        if (const auto* vector = std::get_if<sbm::VectorDataset>(&dataset->value)) {
            auto resolved = config->value;
            resolved.objective = sbm::ObjectiveKind::VectorRegression;
            const auto result = sbm::run_experiment(
                *vector, warmup, resolved, strict_freeze != 0,
                prefill, prune_interval, merge_interval);
            return duplicate_string(sbm::to_json(result));
        }
        auto resolved = config->value;
        resolved.objective = sbm::ObjectiveKind::TokenCrossEntropy;
        const auto result = sbm::run_token_experiment(
            std::get<sbm::TokenDataset>(dataset->value), warmup, resolved,
            strict_freeze != 0, prefill, prune_interval, merge_interval);
        return duplicate_string(sbm::to_json(result));
    });
}

void sbm_string_free(char* value) { std::free(value); }

} // extern "C"
