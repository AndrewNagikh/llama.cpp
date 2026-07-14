if (EMSCRIPTEN)
    return()
endif()

set(DIST_TRANSPORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/transport)
set(DIST_WORKERS_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/workers)

set(DIST_TRANSPORT_SOURCES
    ${DIST_TRANSPORT_DIR}/split_tcp_wire.cpp
    ${DIST_TRANSPORT_DIR}/split_wave_wire.cpp
    ${DIST_TRANSPORT_DIR}/runtime_protocol.cpp
)

set(DIST_PLATFORM_SOURCES
    dist_process.cpp
)

set(DIST_PERF_TRACE_SOURCES
    runtime_debug/perf_trace.cpp
    runtime_debug/perf_trace_export.cpp
    runtime_debug/perf_gpu_sampler.cpp
    runtime_debug/perf_ggml.cpp
    runtime_debug/node_log_export.cpp
)

set(DIST_RUNTIME_DEBUG_SOURCES
    runtime_debug/runtime_debug.cpp
    runtime_debug/tensor_stats.cpp
    runtime_debug/trace_recorder.cpp
    ${DIST_PERF_TRACE_SOURCES}
    runtime_debug/debug_hooks.cpp
    runtime_debug/layer_trace.cpp
    runtime_debug/runtime_state.cpp
    runtime_debug/hidden_transport.cpp
    runtime_debug/hidden_transport_breakdown.cpp
    ${DIST_TRANSPORT_SOURCES}
    verification/verification_common.cpp
    verification/deps/sha256/sha256.c
)

set(DIST_STATE_VERIFY_SOURCES
    verification/runtime_api_verify.cpp
    verification/decode_state_parity.cpp
)

set(DIST_FINAL_VERIFY_SOURCES
    verification/final_runtime_verify.cpp
    verification/context_params_verify.cpp
    verification/decode_graph_verify.cpp
    verification/runtime_cache_verify.cpp
)

set(DIST_TRANSPORT_VERIFY_SOURCES
    verification/hidden_transport_verify.cpp
    ${DIST_TRANSPORT_SOURCES}
)

set(DIST_LAYER_VERIFY_SOURCES
    verification/layer_runner.cpp
    verification/layer_equivalence.cpp
    verification/graph_equivalence.cpp
)

set(DIST_RUNTIME_VERIFY_SOURCES
    verification/monolithic_trace.cpp
    verification/decode_loop_parity.cpp
    verification/hidden_pipeline_parity.cpp
    verification/logits_compare.cpp
    ${DIST_LAYER_VERIFY_SOURCES}
    ${DIST_STATE_VERIFY_SOURCES}
    ${DIST_FINAL_VERIFY_SOURCES}
    ${DIST_RUNTIME_DEBUG_SOURCES}
)

set(DIST_ARCHITECTURE_SOURCES
    architecture/semantic_blob.cpp
    architecture/semantic_roles.cpp
    architecture/worker_requirement.cpp
    architecture/architecture_descriptor.cpp
    architecture/architecture_plugin.cpp
    architecture/blob_builder.cpp
    architecture/descriptor_service.cpp
    architecture/install_planning.cpp
    architecture/tensor_plan.cpp
    architecture/tensor_graph.cpp
    architecture/semantic_runtime_descriptor.cpp
    runtime/architecture_descriptor/blob_requirements.cpp
    runtime/architecture_descriptor/distributed_runtime_descriptor.cpp
    architecture/plugins/llama_plugin.cpp
    architecture/plugins/qwen_plugin.cpp
    architecture/plugins/gemma_plugin.cpp
    architecture/plugins/phi_plugin.cpp
    architecture/plugins/smol_plugin.cpp
    architecture/plugins/deepseek_plugin.cpp
)

set(DIST_CONSISTENCY_SOURCES
    orchestrator/consistency/state_snapshot.cpp
    orchestrator/consistency/install_journal.cpp
    orchestrator/consistency/store_verifier.cpp
    orchestrator/consistency/cluster_consistency.cpp
)

set(DIST_COMMON_SOURCES
    ${DIST_TRANSPORT_SOURCES}
    dist_common.cpp
    model_catalog.cpp
    ${DIST_ARCHITECTURE_SOURCES}
    orchestrator/model_registry.cpp
    orchestrator/model_provider/model_provider.cpp
    orchestrator/model_provider/huggingface_provider.cpp
    orchestrator/manifest_builder/manifest_builder.cpp
    orchestrator/layout_planner/layout_planner.cpp
    orchestrator/layout_estimator/layout_estimator.cpp
    orchestrator/optimizer/cluster_optimizer.cpp
    orchestrator/coverage/coverage.cpp
    orchestrator/coverage/runtime_coverage.cpp
    orchestrator/install_planner/install_planner.cpp
    orchestrator/registry_persistence.cpp
    node_core.cpp
)

set(DIST_BENCHMARK_SOURCES
    node_benchmark.cpp
)

set(DIST_PLANNER_SOURCES
    layer_planner.cpp
    memory_estimator.cpp
)

set(DIST_RUNTIME_SOURCES
    runtime/runtime_role.cpp
    runtime/runtime_descriptor.cpp
    runtime/runtime_role_descriptor.cpp
    runtime/runtime_graph.cpp
    runtime/runtime_cost_model.cpp
    runtime/runtime_role_planner.cpp
    runtime/layer_store_tensor_provider.cpp
    runtime/runtime_worker_bind.cpp
    runtime/runtime_config.cpp
    runtime/runtime_scheduler.cpp
    runtime/llama_layer_store_model_load.cpp
    runtime/runtime_install_planning.cpp
)

set(DIST_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/orchestrator
    ${CMAKE_CURRENT_SOURCE_DIR}/verification
    ${CMAKE_CURRENT_SOURCE_DIR}/verification/deps/sha256
    ${CMAKE_SOURCE_DIR}/examples/gguf-hash/deps
    ${CMAKE_CURRENT_SOURCE_DIR}/node_agent
    ${CMAKE_CURRENT_SOURCE_DIR}/node_agent/synchronization
    ${DIST_TRANSPORT_DIR}
    ${DIST_WORKERS_DIR}
    ${CMAKE_SOURCE_DIR}/vendor
    ${CMAKE_SOURCE_DIR}/vendor/cpp-httplib
)

set(NODE_AGENT_SOURCES
    ${DIST_PLATFORM_SOURCES}
    dist_http_fetch.cpp
    node_agent/layer_store/layer_store.cpp
    node_agent/layer_store/layer_checksum.cpp
    node_agent/layer_store/descriptor_materialize.cpp
    node_agent/layer_store/layer_gguf_assembler.cpp
    node_agent/layer_store/worker_builder.cpp
    node_agent/synchronization/sync_types.cpp
    node_agent/synchronization/synchronization_engine.cpp
    node_agent/synchronization/executors/http_range/http_range_executor.cpp
    orchestrator/install_planner/install_planner.cpp
    orchestrator/manifest_builder/manifest_builder.cpp
    orchestrator/coverage/coverage.cpp
    orchestrator/coverage/runtime_coverage.cpp
    ${DIST_ARCHITECTURE_SOURCES}
)

set(VERIFICATION_SOURCES
    verification/deps/sha256/sha256.c
    verification/verification_common.cpp
    verification/gguf_inspector.cpp
    verification/gguf_diff.cpp
    verification/tensor_verify.cpp
    verification/layer_verify.cpp
    verification/metadata_verify.cpp
    verification/alignment_verify.cpp
    verification/parity_verify.cpp
    verification/materialization_verify.cpp
    verification/manifest_verify.cpp
    verification/worker_tensor_plan.cpp
    verification/load_verify.cpp
    verification/hidden_state_verify.cpp
    verification/logits_compare.cpp
    verification/verification_stages.cpp
    verification/verification_pipeline.cpp
    verification/layer_store_populate.cpp
    verification/worker_verify.cpp
    verification/verification_suite.cpp
)

set(DIST_VERIFY_RUNTIME_SOURCES
    runtime/runtime_role.cpp
    runtime/runtime_descriptor.cpp
    runtime/runtime_graph.cpp
    runtime/runtime_install_planning.cpp
)

function(dist_link_platform_libs target)
    if (WIN32)
        target_link_libraries(${target} PRIVATE ws2_32)
    elseif (CMAKE_SYSTEM_NAME MATCHES "Linux|Darwin")
        target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
    endif()
endfunction()

function(dist_add_verify_exe name main_source)
    add_executable(${name} ${main_source} ${VERIFICATION_SOURCES} ${NODE_AGENT_SOURCES} ${DIST_PERF_TRACE_SOURCES} ${DIST_VERIFY_RUNTIME_SOURCES} dist_common.cpp memory_estimator.cpp)
    target_include_directories(${name} PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(${name} PRIVATE cxx_std_17)
    target_link_libraries(${name} PRIVATE llama llama-common cpp-httplib Threads::Threads)
    dist_link_platform_libs(${name})
endfunction()

function(dist_add_exe name main_source)
    add_executable(${name} ${main_source} ${DIST_COMMON_SOURCES} ${DIST_PLANNER_SOURCES})
    target_include_directories(${name} PRIVATE ${DIST_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE llama llama-common cpp-httplib Threads::Threads)
    dist_link_platform_libs(${name})
endfunction()

function(dist_add_exe_light name main_source)
    add_executable(${name} ${main_source} dist_common.cpp model_catalog.cpp ${DIST_PLANNER_SOURCES} ${DIST_BENCHMARK_SOURCES} ${NODE_AGENT_SOURCES})
    target_include_directories(${name} PRIVATE ${DIST_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE llama llama-common cpp-httplib Threads::Threads)
    dist_link_platform_libs(${name})
endfunction()

function(dist_add_test_exe name main_source)
    add_executable(${name} ${main_source} dist_common.cpp model_catalog.cpp ${DIST_PLANNER_SOURCES} ${DIST_TRANSPORT_SOURCES})
    target_include_directories(${name} PRIVATE ${DIST_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE llama llama-common cpp-httplib Threads::Threads)
    dist_link_platform_libs(${name})
endfunction()

set(RUNTIME_WORKER_LOAD_SOURCES
    ${DIST_PLATFORM_SOURCES}
    dist_common.cpp
    dist_http_fetch.cpp
    node_agent/layer_store/layer_store.cpp
    node_agent/layer_store/layer_checksum.cpp
    node_agent/layer_store/descriptor_materialize.cpp
    orchestrator/manifest_builder/manifest_builder.cpp
    ${DIST_ARCHITECTURE_SOURCES}
    runtime/layer_store_tensor_provider.cpp
    runtime/runtime_config.cpp
    runtime/runtime_descriptor.cpp
    runtime/llama_layer_store_model_load.cpp
    runtime/runtime_install_planning.cpp
    runtime/runtime_graph.cpp
    runtime/runtime_role.cpp
    workers/split_gen_model_load.cpp
)

function(dist_add_worker name main_source)
    add_executable(${name} ${main_source}
        ${DIST_TRANSPORT_SOURCES}
        ${DIST_RUNTIME_DEBUG_SOURCES}
        ${RUNTIME_WORKER_LOAD_SOURCES}
        ${DIST_WORKERS_DIR}/wave_inbound_queue.cpp)
    target_include_directories(${name} PRIVATE ${DIST_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE llama llama-common Threads::Threads)
    dist_link_platform_libs(${name})
endfunction()

dist_add_exe_light(node_agent node_agent.cpp)
target_sources(node_agent PRIVATE ${DIST_TRANSPORT_SOURCES} ${DIST_RUNTIME_DEBUG_SOURCES}
    ${DIST_RUNTIME_SOURCES}
    ${DIST_WORKERS_DIR}/wave_inbound_queue.cpp
    ${DIST_TRANSPORT_DIR}/runtime_entry_queue.cpp
    runtime/runtime_compat_materialize.cpp
    runtime/runtime_worker_gguf_resolver.cpp
    verification/worker_verify.cpp)
dist_add_exe(orchestrator orchestrator.cpp)
target_sources(orchestrator PRIVATE
    ${DIST_PLATFORM_SOURCES}
    ${DIST_RUNTIME_SOURCES}
    ${DIST_PERF_TRACE_SOURCES}
    dist_http_fetch.cpp
    node_agent/layer_store/layer_store.cpp
    node_agent/layer_store/layer_checksum.cpp
    node_agent/layer_store/descriptor_materialize.cpp
    dist_rss_probe.cpp
    ${DIST_CONSISTENCY_SOURCES}
)

dist_add_verify_exe(gguf_inspect gguf_inspect.cpp)
dist_add_verify_exe(gguf_diff gguf_diff.cpp)
dist_add_verify_exe(tensor_verify tensor_verify.cpp)
dist_add_verify_exe(layer_verify layer_verify.cpp)
dist_add_verify_exe(materialization_verify materialization_verify_tool.cpp)
target_sources(materialization_verify PRIVATE memory_estimator_gguf.cpp)
dist_add_verify_exe(logits_compare logits_compare_tool.cpp)
dist_add_verify_exe(hidden_state_compare hidden_state_compare_tool.cpp)

add_executable(architecture-report architecture_report_tool.cpp
    ${DIST_ARCHITECTURE_SOURCES}
    orchestrator/manifest_builder/manifest_builder.cpp
    verification/verification_suite.cpp
    dist_common.cpp)
target_include_directories(architecture-report PRIVATE ${DIST_INCLUDE_DIRS})
target_compile_features(architecture-report PRIVATE cxx_std_17)
target_link_libraries(architecture-report PRIVATE llama llama-common Threads::Threads)

add_executable(mono_reference mono_reference_tool.cpp ${DIST_RUNTIME_VERIFY_SOURCES})
target_include_directories(mono_reference PRIVATE ${DIST_INCLUDE_DIRS})
target_compile_features(mono_reference PRIVATE cxx_std_17)
target_link_libraries(mono_reference PRIVATE llama llama-common Threads::Threads)

add_executable(verify_decode_loop_parity verify_decode_loop_parity.cpp ${DIST_RUNTIME_VERIFY_SOURCES})
target_include_directories(verify_decode_loop_parity PRIVATE ${DIST_INCLUDE_DIRS})
target_compile_features(verify_decode_loop_parity PRIVATE cxx_std_17)
target_link_libraries(verify_decode_loop_parity PRIVATE llama llama-common Threads::Threads)

add_executable(verify_hidden_pipeline verify_hidden_pipeline_tool.cpp ${DIST_RUNTIME_VERIFY_SOURCES})
target_include_directories(verify_hidden_pipeline PRIVATE ${DIST_INCLUDE_DIRS})
target_compile_features(verify_hidden_pipeline PRIVATE cxx_std_17)
target_link_libraries(verify_hidden_pipeline PRIVATE llama llama-common Threads::Threads)

add_executable(verify_logits_pipeline verify_logits_pipeline_tool.cpp ${DIST_RUNTIME_VERIFY_SOURCES})
target_include_directories(verify_logits_pipeline PRIVATE ${DIST_INCLUDE_DIRS})
target_compile_features(verify_logits_pipeline PRIVATE cxx_std_17)
target_link_libraries(verify_logits_pipeline PRIVATE llama llama-common Threads::Threads)

add_executable(verify_layer_equivalence verify_layer_equivalence_tool.cpp ${DIST_RUNTIME_VERIFY_SOURCES})
target_include_directories(verify_layer_equivalence PRIVATE ${DIST_INCLUDE_DIRS})
target_compile_features(verify_layer_equivalence PRIVATE cxx_std_17)
target_link_libraries(verify_layer_equivalence PRIVATE llama llama-common Threads::Threads)

add_executable(diagnose_hidden_boundary diagnose_hidden_boundary.cpp ${DIST_RUNTIME_VERIFY_SOURCES})
target_include_directories(diagnose_hidden_boundary PRIVATE ${DIST_INCLUDE_DIRS})
target_compile_features(diagnose_hidden_boundary PRIVATE cxx_std_17)
target_link_libraries(diagnose_hidden_boundary PRIVATE llama llama-common Threads::Threads)

add_executable(verify_hidden_transport verify_hidden_transport_tool.cpp ${DIST_RUNTIME_VERIFY_SOURCES} ${DIST_TRANSPORT_VERIFY_SOURCES})
target_include_directories(verify_hidden_transport PRIVATE ${DIST_INCLUDE_DIRS})
target_compile_features(verify_hidden_transport PRIVATE cxx_std_17)
target_link_libraries(verify_hidden_transport PRIVATE llama llama-common Threads::Threads)

add_executable(verify_runtime_api verify_runtime_api_tool.cpp ${DIST_RUNTIME_VERIFY_SOURCES})
target_include_directories(verify_runtime_api PRIVATE ${DIST_INCLUDE_DIRS})
target_compile_features(verify_runtime_api PRIVATE cxx_std_17)
target_link_libraries(verify_runtime_api PRIVATE llama llama-common Threads::Threads)

add_executable(verify_final_runtime verify_final_runtime_tool.cpp ${DIST_RUNTIME_VERIFY_SOURCES})
target_include_directories(verify_final_runtime PRIVATE ${DIST_INCLUDE_DIRS})
target_compile_features(verify_final_runtime PRIVATE cxx_std_17)
target_link_libraries(verify_final_runtime PRIVATE llama llama-common Threads::Threads)

add_executable(verify_final_logits verify_final_logits_tool.cpp ${DIST_RUNTIME_VERIFY_SOURCES})
target_include_directories(verify_final_logits PRIVATE ${DIST_INCLUDE_DIRS})
target_compile_features(verify_final_logits PRIVATE cxx_std_17)
target_link_libraries(verify_final_logits PRIVATE llama llama-common Threads::Threads)

add_executable(verify_decode_graph verify_decode_graph_tool.cpp ${DIST_RUNTIME_VERIFY_SOURCES})
target_include_directories(verify_decode_graph PRIVATE ${DIST_INCLUDE_DIRS})
target_compile_features(verify_decode_graph PRIVATE cxx_std_17)
target_link_libraries(verify_decode_graph PRIVATE llama llama-common Threads::Threads)

add_executable(verify_context_params verify_context_params_tool.cpp ${DIST_RUNTIME_VERIFY_SOURCES})
target_include_directories(verify_context_params PRIVATE ${DIST_INCLUDE_DIRS})
target_compile_features(verify_context_params PRIVATE cxx_std_17)
target_link_libraries(verify_context_params PRIVATE llama llama-common Threads::Threads)

dist_add_worker(split_gen3_a ${DIST_WORKERS_DIR}/split_gen3_a.cpp)
dist_add_worker(split_gen3_b ${DIST_WORKERS_DIR}/split_gen3_b.cpp)
dist_add_worker(split_gen3_c ${DIST_WORKERS_DIR}/split_gen3_c.cpp)
target_link_libraries(split_gen3_c PRIVATE cpp-httplib)

# End-to-end cluster validation suite (Task 7.5). Defined unconditionally so a
# single `build.sh e2e` builds them even when LLAMA_BUILD_TESTS is OFF.
dist_add_test_exe(test-cluster-e2e test-cluster-e2e.cpp)
dist_add_test_exe(test-cluster-e2e-install-reuse test-cluster-e2e-install-reuse.cpp)
dist_add_test_exe(test-cluster-e2e-node-loss test-cluster-e2e-node-loss.cpp)
dist_add_test_exe(test-cluster-e2e-orchestrator-restart test-cluster-e2e-orchestrator-restart.cpp)
dist_add_test_exe(test-cluster-e2e-optimizer-api test-cluster-e2e-optimizer-api.cpp)
dist_add_test_exe(test-cluster-e2e-optimizer-join test-cluster-e2e-optimizer-join.cpp)
dist_add_test_exe(test-cluster-e2e-verification test-cluster-e2e-verification.cpp)

if (LLAMA_BUILD_TESTS)
    # v0.2 release-critical distributed runtime gates. Legacy Task 8/9
    # planner, registry, optimizer, sync, and verification experiments remain as
    # source files, but are intentionally excluded from the active release test
    # surface unless promoted back into this curated gate list.
    add_executable(test-runtime-role-planner test-runtime-role-planner.cpp
        runtime/runtime_role.cpp
        runtime/runtime_descriptor.cpp
        runtime/runtime_role_descriptor.cpp
        runtime/runtime_graph.cpp
        runtime/runtime_cost_model.cpp
        runtime/runtime_role_planner.cpp
        architecture/worker_requirement.cpp
        layer_planner.cpp
        dist_common.cpp)
    target_include_directories(test-runtime-role-planner PRIVATE ${DIST_INCLUDE_DIRS})
    target_link_libraries(test-runtime-role-planner PRIVATE llama llama-common Threads::Threads)

    add_executable(test-embedding-service test-embedding-service.cpp
        runtime/runtime_role.cpp
        runtime/runtime_descriptor.cpp
        runtime/runtime_role_descriptor.cpp
        runtime/runtime_graph.cpp
        runtime/runtime_cost_model.cpp
        runtime/runtime_role_planner.cpp
        architecture/worker_requirement.cpp
        dist_common.cpp)
    target_include_directories(test-embedding-service PRIVATE ${DIST_INCLUDE_DIRS})
    target_link_libraries(test-embedding-service PRIVATE llama llama-common Threads::Threads)

    add_executable(test-runtime-cost test-runtime-cost.cpp
        runtime/runtime_role.cpp
        runtime/runtime_role_descriptor.cpp
        runtime/runtime_cost_model.cpp
        architecture/worker_requirement.cpp
        dist_common.cpp)
    target_include_directories(test-runtime-cost PRIVATE ${DIST_INCLUDE_DIRS})
    target_link_libraries(test-runtime-cost PRIVATE llama llama-common Threads::Threads)

    set(RUNTIME_UNIT_TEST_SOURCES
        ${DIST_PLATFORM_SOURCES}
        dist_common.cpp
        dist_http_fetch.cpp
        node_agent/layer_store/layer_store.cpp
        node_agent/layer_store/layer_checksum.cpp
        node_agent/layer_store/descriptor_materialize.cpp
        node_agent/layer_store/worker_builder.cpp
        orchestrator/manifest_builder/manifest_builder.cpp
        ${DIST_ARCHITECTURE_SOURCES}
        ${DIST_RUNTIME_SOURCES}
    )

    add_executable(test-runtime-worker-bind test-runtime-worker-bind.cpp ${RUNTIME_UNIT_TEST_SOURCES})
    target_include_directories(test-runtime-worker-bind PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-runtime-worker-bind PRIVATE cxx_std_17)
    target_link_libraries(test-runtime-worker-bind PRIVATE llama cpp-httplib Threads::Threads)

    add_executable(test-layer-store-tensor-provider test-layer-store-tensor-provider.cpp ${RUNTIME_UNIT_TEST_SOURCES})
    target_include_directories(test-layer-store-tensor-provider PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-layer-store-tensor-provider PRIVATE cxx_std_17)
    target_link_libraries(test-layer-store-tensor-provider PRIVATE llama cpp-httplib Threads::Threads)

    add_executable(test-runtime-config test-runtime-config.cpp runtime/runtime_config.cpp)
    target_include_directories(test-runtime-config PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-runtime-config PRIVATE cxx_std_17)
    target_link_libraries(test-runtime-config PRIVATE Threads::Threads)

    add_executable(test-perf-trace test-perf-trace.cpp runtime_debug/perf_trace.cpp runtime_debug/perf_gpu_sampler.cpp)
    target_include_directories(test-perf-trace PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-perf-trace PRIVATE cxx_std_17)
    target_link_libraries(test-perf-trace PRIVATE Threads::Threads)
    if (CMAKE_SYSTEM_NAME MATCHES "Linux|Darwin")
        target_link_libraries(test-perf-trace PRIVATE ${CMAKE_DL_LIBS})
    endif()

    add_executable(test-wave-wire test-wave-wire.cpp ${DIST_TRANSPORT_SOURCES})
    target_include_directories(test-wave-wire PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-wave-wire PRIVATE cxx_std_17)
    target_link_libraries(test-wave-wire PRIVATE Threads::Threads)

    add_executable(test-runtime-protocol test-runtime-protocol.cpp ${DIST_TRANSPORT_SOURCES})
    target_include_directories(test-runtime-protocol PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-runtime-protocol PRIVATE cxx_std_17)
    target_link_libraries(test-runtime-protocol PRIVATE Threads::Threads)

    add_executable(test-wave-inbound-queue test-wave-inbound-queue.cpp workers/wave_inbound_queue.cpp)
    target_include_directories(test-wave-inbound-queue PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-wave-inbound-queue PRIVATE cxx_std_17)
    target_link_libraries(test-wave-inbound-queue PRIVATE Threads::Threads)

    add_executable(test-hidden-inbound-queue test-hidden-inbound-queue.cpp workers/wave_inbound_queue.cpp)
    target_include_directories(test-hidden-inbound-queue PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-hidden-inbound-queue PRIVATE cxx_std_17)
    target_link_libraries(test-hidden-inbound-queue PRIVATE Threads::Threads)

    set(RUNTIME_DESCRIPTOR_TEST_SOURCES
        runtime/runtime_descriptor.cpp
        runtime/runtime_role.cpp
    )

    add_executable(test-runtime-descriptor test-runtime-descriptor.cpp ${RUNTIME_DESCRIPTOR_TEST_SOURCES})
    target_include_directories(test-runtime-descriptor PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-runtime-descriptor PRIVATE cxx_std_17)
    target_link_libraries(test-runtime-descriptor PRIVATE Threads::Threads)

    add_executable(test-runtime-service test-runtime-service.cpp ${RUNTIME_DESCRIPTOR_TEST_SOURCES})
    target_include_directories(test-runtime-service PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-runtime-service PRIVATE cxx_std_17)
    target_link_libraries(test-runtime-service PRIVATE Threads::Threads)

    add_executable(test-runtime-schema test-runtime-schema.cpp ${RUNTIME_DESCRIPTOR_TEST_SOURCES})
    target_include_directories(test-runtime-schema PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-runtime-schema PRIVATE cxx_std_17)
    target_link_libraries(test-runtime-schema PRIVATE Threads::Threads)

    add_executable(test-runtime-install-planning test-runtime-install-planning.cpp
        runtime/runtime_install_planning.cpp
        runtime/runtime_descriptor.cpp
        runtime/runtime_graph.cpp
        runtime/runtime_role.cpp)
    target_include_directories(test-runtime-install-planning PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-runtime-install-planning PRIVATE cxx_std_17)
    target_link_libraries(test-runtime-install-planning PRIVATE Threads::Threads)

    add_executable(test-runtime-acceptance test-runtime-acceptance.cpp
        runtime/runtime_install_planning.cpp
        runtime/runtime_scheduler.cpp
        runtime/runtime_descriptor.cpp
        runtime/runtime_graph.cpp
        runtime/runtime_role.cpp)
    target_include_directories(test-runtime-acceptance PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-runtime-acceptance PRIVATE cxx_std_17)
    target_link_libraries(test-runtime-acceptance PRIVATE Threads::Threads)

    add_executable(test-runtime-presmoke test-runtime-presmoke.cpp
        ${RUNTIME_UNIT_TEST_SOURCES}
        runtime/runtime_compat_materialize.cpp)
    target_include_directories(test-runtime-presmoke PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-runtime-presmoke PRIVATE cxx_std_17)
    target_link_libraries(test-runtime-presmoke PRIVATE llama cpp-httplib Threads::Threads)

    add_executable(test-runtime-layer-store-model-load test-runtime-layer-store-model-load.cpp
        ${VERIFICATION_SOURCES}
        ${NODE_AGENT_SOURCES}
        ${DIST_PERF_TRACE_SOURCES}
        ${DIST_RUNTIME_SOURCES}
        dist_common.cpp
        memory_estimator.cpp)
    target_include_directories(test-runtime-layer-store-model-load PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-runtime-layer-store-model-load PRIVATE cxx_std_17)
    target_link_libraries(test-runtime-layer-store-model-load PRIVATE llama llama-common cpp-httplib Threads::Threads)

    add_custom_target(runtime-presmoke-gates
        DEPENDS
            test-runtime-presmoke
            test-runtime-acceptance
            verify_hidden_pipeline
            verify_logits_pipeline
            verify_final_runtime
            verify_final_logits
            verify_decode_graph
            verify_decode_loop_parity)
endif()
