#include "sbm/api.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string_view>

int main() {
    assert(sbm_api_version() == 9U);
    assert(std::strlen(sbm_api_version_string()) > 0U);

    const std::string_view schema(sbm_parameter_schema_json());
    assert(schema.find("exact_region_mass") != std::string_view::npos);
    assert(schema.find("classification_learning_rate") != std::string_view::npos);
    assert(schema.find("max_sparse_decisions_per_node") != std::string_view::npos);
    assert(schema.find("output_tree_seed") != std::string_view::npos);
    assert(schema.find("address_execution_mode") != std::string_view::npos);
    assert(schema.find("accepted_channel_retirement") != std::string_view::npos);
    assert(schema.find("topology_accept_uses_structural_value") !=
           std::string_view::npos);
    assert(schema.find("binding_reuse_value_weight") !=
           std::string_view::npos);
    assert(schema.find("max_binding_reuse_records_per_channel") !=
           std::string_view::npos);
    assert(schema.find("search_default") != std::string_view::npos);

    sbm_config_handle* config = sbm_config_create();
    assert(config != nullptr);
    assert(sbm_config_set(config, "exact_region_mass", "0.91") == 0);
    assert(sbm_config_set(config, "address_lags", "1,2,4") == 0);
    assert(sbm_config_set(config, "label_smoothing", "0.02") == 0);
    assert(sbm_config_set(config, "output_tree_seed", "29") == 0);
    assert(sbm_config_set(config, "address_execution_mode", "LegacySignature") == 0);
    assert(sbm_config_set(config, "accepted_channel_retirement", "Quarantine") == 0);
    assert(sbm_config_set(config, "accepted_channel_retirement", "RecoverableRetire") == 0);
    assert(sbm_config_set(config, "structural_description_cost_weight", "0.5") == 0);
    assert(sbm_config_set(config, "structural_execution_cost_weight", "0.25") == 0);
    assert(sbm_config_set(config, "binding_reuse_value_weight", "0.125") == 0);
    assert(sbm_config_set(config, "topology_accept_uses_structural_value", "true") == 0);
    assert(sbm_config_set(config, "does_not_exist", "1") != 0);
    assert(std::strlen(sbm_last_error()) > 0U);

    char* config_json = sbm_config_get_json(config);
    assert(config_json != nullptr);
    assert(std::string_view(config_json).find("classification_learning_rate") !=
           std::string_view::npos);
    assert(std::string_view(config_json).find("\"output_tree_seed\": 29") !=
           std::string_view::npos);
    assert(std::string_view(config_json).find(
        "\"address_execution_mode\": \"LegacySignature\"") != std::string_view::npos);
    assert(std::string_view(config_json).find(
        "\"accepted_channel_retirement\": \"RecoverableRetire\"") !=
        std::string_view::npos);
    assert(std::string_view(config_json).find(
        "\"topology_accept_uses_structural_value\": true") !=
        std::string_view::npos);
    assert(std::string_view(config_json).find(
        "\"binding_reuse_value_weight\": 0.125") != std::string_view::npos);
    sbm_string_free(config_json);

    sbm_dataset_handle* vector_dataset = sbm_dataset_generate(8000, 32, 16, 20, 9, 0.035F);
    assert(vector_dataset != nullptr);
    assert(sbm_dataset_kind_of(vector_dataset) == SBM_DATASET_VECTOR_REGRESSION);
    assert(sbm_dataset_length(vector_dataset) == 8000U);
    assert(sbm_dataset_vector_dim(vector_dataset) == 16U);
    assert(sbm_dataset_alphabet(vector_dataset) == 32U);

    char* vector_result = sbm_run_experiment_json(vector_dataset, 3000, config, 1, 0, 0, 0);
    assert(vector_result != nullptr);
    assert(std::string_view(vector_result).find("\"eval_r2\"") != std::string_view::npos);
    sbm_string_free(vector_result);
    sbm_dataset_destroy(vector_dataset);

    sbm_dataset_handle* token_dataset = sbm_token_dataset_generate_math(
        8, 512, 16, 13, 0.8F, 0.2F);
    assert(token_dataset != nullptr);
    assert(sbm_dataset_kind_of(token_dataset) == SBM_DATASET_TOKEN_CROSS_ENTROPY);
    assert(sbm_dataset_vocab_size(token_dataset) == 16U);
    assert(sbm_dataset_sequence_count(token_dataset) == 8U);
    assert(sbm_dataset_example_count(token_dataset) == 4088U);

    char* token_result = sbm_run_experiment_json(token_dataset, 2500, config, 1, 0, 0, 0);
    assert(token_result != nullptr);
    const std::string_view token_view(token_result);
    assert(token_view.find("\"eval_cross_entropy\"") != std::string_view::npos);
    assert(token_view.find("\"objective\": \"token_cross_entropy\"") !=
           std::string_view::npos);
    assert(token_view.find("\"address_execution_frames\"") != std::string_view::npos);
    assert(token_view.find("\"address_binding_hits\"") != std::string_view::npos);
    assert(token_view.find("\"address_binding_misses\"") != std::string_view::npos);
    assert(token_view.find("\"quarantined_channels\"") != std::string_view::npos);
    assert(token_view.find("\"recoverable_retired_channels\"") !=
           std::string_view::npos);
    assert(token_view.find("\"dependency_blocked_channels\"") !=
           std::string_view::npos);
    assert(token_view.find("\"topology_restored\"") != std::string_view::npos);
    assert(token_view.find("\"learned_channel_effective_enabled\"") !=
           std::string_view::npos);
    assert(token_view.find("\"decision_name\"") != std::string_view::npos);
    assert(token_view.find("\"structural_value_nats\"") != std::string_view::npos);
    assert(token_view.find("\"structural_description_cost\"") !=
           std::string_view::npos);
    assert(token_view.find("\"structural_execution_cost\"") !=
           std::string_view::npos);
    assert(token_view.find("\"candidate_source_exact_bucket\"") !=
           std::string_view::npos);
    assert(token_view.find("\"candidate_source_control_edge\"") !=
           std::string_view::npos);
    assert(token_view.find("\"candidate_source_neighbor_bucket\"") !=
           std::string_view::npos);
    assert(token_view.find("\"route_score_hamming_sum\"") !=
           std::string_view::npos);
    assert(token_view.find("\"route_score_exact_sum\"") !=
           std::string_view::npos);
    assert(token_view.find("\"route_score_edge_prior_sum\"") !=
           std::string_view::npos);
    assert(token_view.find("\"gpaf_probe_slots\"") != std::string_view::npos);
    assert(token_view.find("\"gpaf_active_slots\"") != std::string_view::npos);
    assert(token_view.find("\"gpaf_quarantined_slots\"") != std::string_view::npos);
    assert(token_view.find("\"gpaf_recoverable_retired_slots\"") !=
           std::string_view::npos);
    assert(token_view.find("\"gpaf_physically_erased_slots\"") !=
           std::string_view::npos);
    sbm_string_free(token_result);

    sbm_config_handle* machine_config = sbm_config_create();
    assert(machine_config != nullptr);
    assert(sbm_config_set(machine_config, "objective", "TokenCrossEntropy") != 0);
    assert(sbm_config_set(machine_config, "token_alphabet", "16") == 0);
    assert(sbm_config_set(machine_config, "vector_dim", "16") == 0);
    assert(sbm_config_set(machine_config, "bucket_bits", "8") == 0);
    assert(sbm_config_set(machine_config, "seed", "13") == 0);
    assert(sbm_config_set(machine_config, "address_lags", "1,2") == 0);
    assert(sbm_config_set(machine_config, "topology_probe_interval", "8") == 0);
    char* machine_config_json = sbm_config_get_json(machine_config);
    assert(machine_config_json != nullptr);
    assert(std::string_view(machine_config_json).find("\"address_lags\": [1, 2]") !=
           std::string_view::npos);
    sbm_string_free(machine_config_json);
    sbm_machine_handle* left = sbm_machine_create(machine_config);
    sbm_machine_handle* source = sbm_machine_create(machine_config);
    assert(left != nullptr);
    assert(source != nullptr);
    sbm_step_stats left_stats{};
    sbm_step_stats source_stats{};
    for (uint32_t i = 0; i < 128U; ++i) {
        const uint32_t input_token = i % 16U;
        const uint32_t target_token = (i + 1U) % 16U;
        assert(sbm_machine_step_token(left, input_token, target_token, 1,
                                      &left_stats) == 0);
        assert(sbm_machine_step_token(source, input_token, target_token, 1,
                                      &source_stats) == 0);
    }
    const char* machine_checkpoint_path = "sbm_c_api_machine.sbc";
    std::remove(machine_checkpoint_path);
    assert(sbm_machine_save_checkpoint(source, machine_checkpoint_path) == 0);
    sbm_machine_handle* resumed = sbm_machine_load_checkpoint(machine_checkpoint_path);
    assert(resumed != nullptr);
    std::remove(machine_checkpoint_path);
    for (uint32_t i = 128U; i < 192U; ++i) {
        const uint32_t input_token = i % 16U;
        const uint32_t target_token = (i + 1U) % 16U;
        assert(sbm_machine_step_token(left, input_token, target_token, 1,
                                      &left_stats) == 0);
        assert(sbm_machine_step_token(resumed, input_token, target_token, 1,
                                      &source_stats) == 0);
        assert(left_stats.predicted_token == source_stats.predicted_token);
        assert(left_stats.live_nodes == source_stats.live_nodes);
        assert(std::abs(left_stats.cross_entropy - source_stats.cross_entropy) < 1e-6F);
    }
    sbm_machine_freeze_topology(resumed);
    assert(sbm_machine_retire_channel(resumed, 0U, 3U) == 0);
    assert(sbm_machine_restore_channel(resumed, 0U) == 0);
    assert(sbm_machine_restore_dependency_closure(resumed, 0U) == 0);
    assert(sbm_machine_step_token(resumed, 1U, 3U, 0, &source_stats) == 0);
    assert(source_stats.channel_credit_count <= 8U);
    if (source_stats.channel_subset_available) {
        assert(std::isfinite(source_stats.seed_only_cross_entropy));
        assert(std::isfinite(source_stats.active_only_cross_entropy));
        assert(source_stats.channel_binding_kind[0] <= 3U);
        assert(source_stats.channel_input_state[0] <= 4U);
        assert(source_stats.channel_output_state[0] <= 4U);
        assert(source_stats.channel_required_dependency_binding[0] <= 3U);
        assert(source_stats.channel_binding_matched[0] <= 1U);
        assert(source_stats.channel_binding_current_token[0] < 16U);
        assert(source_stats.channel_binding_matched[0] == 0U ||
               source_stats.channel_binding_key[0] != 0U);
        assert(source_stats.channel_call_key[0] == 0U ||
               source_stats.channel_dependency_binding_key[0] != 0U);
    }
    char* machine_diag = sbm_machine_diagnostics_json(resumed);
    assert(machine_diag != nullptr);
    assert(std::string_view(machine_diag).find("\"steps\": 193") !=
           std::string_view::npos);
    assert(std::string_view(machine_diag).find("\"address_binding_by_kind\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_diag).find("\"binding_reuse_observations\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_diag).find("\"dependency_blocked_channels\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_diag).find("\"topology_restored\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_diag).find("\"candidate_source_exact_bucket\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_diag).find("\"route_score_hamming_sum\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_diag).find("\"gpaf_probe_slots\"") !=
           std::string_view::npos);
    sbm_string_free(machine_diag);
    char* machine_summary = sbm_machine_summary_json(resumed);
    assert(machine_summary != nullptr);
    assert(std::string_view(machine_summary).find("\"learned_address_programs\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find("\"learned_channel_phase\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find(
               "\"learned_channel_effective_enabled\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find("\"topology_events\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find("\"binding_reuse_bonus\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find("\"learned_channel_parent\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find("\"learned_channel_dependency\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find("\"learned_channel_generation\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find("\"learned_channel_parent_edge_kind\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find("\"learned_channel_dependency_edge_kind\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find("\"address_dependency_graph\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find("\"address_dependency_edges\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find("\"caller_channel\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find("\"caller_generation\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find("\"dependency_generation\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find("\"edge_kind\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find("\"direct_caller_count\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find(
               "\"effective_direct_caller_count\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find(
               "\"blocked_direct_caller_count\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find("\"own_call_matches\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find("\"downstream_call_matches\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find("\"parent_channel\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find("\"dependency_channel\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find("\"channel_generation\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find("\"dependency_edge_kind\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find("\"effective_enabled\"") !=
           std::string_view::npos);
    assert(std::string_view(machine_summary).find("\"dependency_available\"") !=
           std::string_view::npos);
    sbm_string_free(machine_summary);
    sbm_machine_destroy(left);
    sbm_machine_destroy(source);
    sbm_machine_destroy(resumed);
    sbm_config_destroy(machine_config);
    sbm_dataset_destroy(token_dataset);

    const uint32_t ids[]{1, 2, 3, 1, 2, 4};
    const uint64_t offsets[]{0, 3, 6};
    sbm_dataset_handle* external = sbm_token_dataset_from_ids(ids, 6, 8, offsets, 3);
    assert(external != nullptr);
    assert(sbm_dataset_sequence_count(external) == 2U);
    assert(sbm_dataset_example_count(external) == 4U);
    const char* shard_path = "sbm_c_api_shard.sbt";
    std::remove(shard_path);
    assert(sbm_token_shard_write(external, shard_path) == 0);
    sbm_token_shard_handle* shard = sbm_token_shard_open(shard_path, 3U, 1);
    assert(shard != nullptr);
    assert(sbm_token_shard_vocab_size(shard) == 8U);
    assert(sbm_token_shard_token_count(shard) == 6U);
    assert(sbm_token_shard_sequence_count(shard) == 2U);
    sbm_token_shard_cursor cursor{3U, 0U, 0U};
    uint32_t input = 0U;
    uint32_t target = 0U;
    assert(sbm_token_shard_next(shard, &cursor, &input, &target) == 1);
    assert(input == 1U && target == 2U);
    assert(sbm_token_shard_next(shard, &cursor, &input, &target) == 1);
    assert(input == 2U && target == 3U);
    assert(sbm_token_shard_next(shard, &cursor, &input, &target) == 1);
    assert(input == 1U && target == 2U);
    assert(sbm_token_shard_next(shard, &cursor, &input, &target) == 1);
    assert(input == 2U && target == 4U);
    assert(sbm_token_shard_next(shard, &cursor, &input, &target) == 0);
    sbm_token_shard_cursor malformed_cursor{3U, 0U, UINT64_MAX};
    assert(sbm_token_shard_next(
               shard, &malformed_cursor, &input, &target) == -1);
    assert(std::string_view(sbm_last_error()).find("token_offset") !=
           std::string_view::npos);
    assert(malformed_cursor.sequence_index == 0U);
    assert(malformed_cursor.token_offset == UINT64_MAX);
    sbm_token_shard_cursor explicit_end{3U, 2U, 0U};
    assert(sbm_token_shard_next(shard, &explicit_end, &input, &target) == 0);
    char* shard_result = sbm_run_token_shard_experiment_json(
        shard, 2U, config, 1, 0U, 0U, 0U);
    assert(shard_result != nullptr);
    assert(std::string_view(shard_result).find(
        "\"task\": \"real_corpus_next_token_cross_entropy\"") !=
        std::string_view::npos);
    sbm_string_free(shard_result);
    sbm_token_shard_destroy(shard);

    sbm_token_corpus_handle* corpus = sbm_token_corpus_create();
    assert(corpus != nullptr);
    assert(sbm_token_corpus_add_shard(corpus, shard_path, 0U, 0U, 1) == 0);
    assert(sbm_token_corpus_add_shard(corpus, shard_path, 1U, 1U, 1) == 0);
    char* corpus_result = sbm_run_token_corpus_experiment_json(
        corpus, config, 1, 0U, 0U, 0U);
    assert(corpus_result != nullptr);
    assert(std::string_view(corpus_result).find("\"train_examples\": 4") !=
           std::string_view::npos);
    assert(std::string_view(corpus_result).find("\"eval_examples\": 4") !=
           std::string_view::npos);
    sbm_string_free(corpus_result);
    sbm_token_corpus_destroy(corpus);
    std::remove(shard_path);
    sbm_dataset_destroy(external);

    sbm_config_destroy(config);
    std::cout << "C API tests passed\n";
}
