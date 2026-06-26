if (EMSCRIPTEN)
    return()
endif()

set(DIST_TRANSPORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/transport)
set(DIST_WORKERS_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/workers)

set(DIST_TRANSPORT_SOURCES
    ${DIST_TRANSPORT_DIR}/split_tcp_wire.cpp
)

set(DIST_ARCHITECTURE_DESCRIPTOR_SOURCES
    architecture_descriptor/descriptor_builder.cpp
)

set(DIST_COMMON_SOURCES
    ${DIST_TRANSPORT_SOURCES}
    dist_common.cpp
    model_catalog.cpp
    ${DIST_ARCHITECTURE_DESCRIPTOR_SOURCES}
    orchestrator/model_registry.cpp
    orchestrator/model_provider/model_provider.cpp
    orchestrator/model_provider/huggingface_provider.cpp
    orchestrator/manifest_builder/manifest_builder.cpp
    orchestrator/layout_planner/layout_planner.cpp
    orchestrator/layout_estimator/layout_estimator.cpp
    orchestrator/optimizer/cluster_optimizer.cpp
    orchestrator/coverage/coverage.cpp
    orchestrator/install_planner/install_planner.cpp
    node_core.cpp
)

set(DIST_BENCHMARK_SOURCES
    node_benchmark.cpp
)

set(DIST_PLANNER_SOURCES
    layer_planner.cpp
    memory_estimator.cpp
)

set(DIST_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
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
    node_agent/layer_store/layer_store.cpp
    node_agent/layer_store/layer_checksum.cpp
    node_agent/layer_store/layer_gguf_assembler.cpp
    node_agent/synchronization/sync_types.cpp
    node_agent/synchronization/synchronization_engine.cpp
    node_agent/synchronization/executors/http_range/http_range_executor.cpp
    orchestrator/install_planner/install_planner.cpp
    orchestrator/manifest_builder/manifest_builder.cpp
    ${DIST_ARCHITECTURE_DESCRIPTOR_SOURCES}
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
)

function(dist_add_verify_exe name main_source)
    add_executable(${name} ${main_source} ${VERIFICATION_SOURCES} ${NODE_AGENT_SOURCES} dist_common.cpp memory_estimator.cpp)
    target_include_directories(${name} PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(${name} PRIVATE cxx_std_17)
    target_link_libraries(${name} PRIVATE llama llama-common cpp-httplib Threads::Threads)
endfunction()

function(dist_add_exe name main_source)
    add_executable(${name} ${main_source} ${DIST_COMMON_SOURCES} ${DIST_PLANNER_SOURCES})
    target_include_directories(${name} PRIVATE ${DIST_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE llama llama-common cpp-httplib Threads::Threads)
endfunction()

function(dist_add_exe_light name main_source)
    add_executable(${name} ${main_source} dist_common.cpp model_catalog.cpp ${DIST_PLANNER_SOURCES} ${DIST_BENCHMARK_SOURCES} ${NODE_AGENT_SOURCES})
    target_include_directories(${name} PRIVATE ${DIST_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE llama llama-common cpp-httplib Threads::Threads)
endfunction()

function(dist_add_test_exe name main_source)
    add_executable(${name} ${main_source} dist_common.cpp model_catalog.cpp ${DIST_PLANNER_SOURCES} ${DIST_TRANSPORT_SOURCES})
    target_include_directories(${name} PRIVATE ${DIST_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE llama llama-common cpp-httplib Threads::Threads)
endfunction()

function(dist_add_worker name main_source)
    add_executable(${name} ${main_source} ${DIST_TRANSPORT_SOURCES})
    target_include_directories(${name} PRIVATE ${DIST_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE llama llama-common Threads::Threads)
endfunction()

dist_add_exe_light(node_agent node_agent.cpp)
target_sources(node_agent PRIVATE ${DIST_TRANSPORT_SOURCES} verification/worker_verify.cpp verification/verification_common.cpp verification/deps/sha256/sha256.c)
dist_add_exe(orchestrator orchestrator.cpp)
target_sources(orchestrator PRIVATE
    node_agent/layer_store/layer_store.cpp
    node_agent/layer_store/layer_checksum.cpp
    node_agent/layer_store/layer_gguf_assembler.cpp
    ${VERIFICATION_SOURCES}
)

dist_add_verify_exe(gguf_inspect gguf_inspect.cpp)
dist_add_verify_exe(gguf_diff gguf_diff.cpp)
dist_add_verify_exe(tensor_verify tensor_verify.cpp)
dist_add_verify_exe(layer_verify layer_verify.cpp)
dist_add_verify_exe(materialization_verify materialization_verify_tool.cpp)
dist_add_verify_exe(logits_compare logits_compare_tool.cpp)
dist_add_verify_exe(hidden_state_compare hidden_state_compare_tool.cpp)

dist_add_worker(split_gen3_a ${DIST_WORKERS_DIR}/split_gen3_a.cpp)
dist_add_worker(split_gen3_b ${DIST_WORKERS_DIR}/split_gen3_b.cpp)
dist_add_worker(split_gen3_c ${DIST_WORKERS_DIR}/split_gen3_c.cpp)

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
    add_executable(test-layer-planner test-layer-planner.cpp ${DIST_PLANNER_SOURCES} dist_common.cpp)
    target_include_directories(test-layer-planner PRIVATE ${DIST_INCLUDE_DIRS})
    target_link_libraries(test-layer-planner PRIVATE llama llama-common Threads::Threads)

    add_executable(test-model-catalog test-model-catalog.cpp model_catalog.cpp)
    target_include_directories(test-model-catalog PRIVATE ${DIST_INCLUDE_DIRS})
    target_link_libraries(test-model-catalog PRIVATE Threads::Threads)

    dist_add_test_exe(test-model-install test-model-install.cpp)

    dist_add_test_exe(test-orchestrator-3node test-orchestrator-3node.cpp)
    dist_add_test_exe(test-orchestrator-dynamic-layout test-orchestrator-dynamic-layout.cpp)
    dist_add_test_exe(test-split-materialized-2stage test-split-materialized-2stage.cpp)
    target_sources(test-split-materialized-2stage PRIVATE ${VERIFICATION_SOURCES} ${NODE_AGENT_SOURCES})

    add_executable(test-node-benchmark test-node-benchmark.cpp ${DIST_BENCHMARK_SOURCES})
    target_include_directories(test-node-benchmark PRIVATE ${DIST_INCLUDE_DIRS})
    target_link_libraries(test-node-benchmark PRIVATE llama llama-common Threads::Threads)

    dist_add_test_exe(test-registration-benchmark test-registration-benchmark.cpp)

    # Task 8 unit tests
    add_executable(test-memory-estimator test-memory-estimator.cpp memory_estimator.cpp dist_common.cpp)
    target_include_directories(test-memory-estimator PRIVATE ${DIST_INCLUDE_DIRS})
    target_link_libraries(test-memory-estimator PRIVATE llama llama-common Threads::Threads)

    add_executable(test-capacity-check test-capacity-check.cpp memory_estimator.cpp dist_common.cpp)
    target_include_directories(test-capacity-check PRIVATE ${DIST_INCLUDE_DIRS})
    target_link_libraries(test-capacity-check PRIVATE llama llama-common Threads::Threads)

    add_executable(test-memory-aware-planner test-memory-aware-planner.cpp layer_planner.cpp memory_estimator.cpp dist_common.cpp)
    target_include_directories(test-memory-aware-planner PRIVATE ${DIST_INCLUDE_DIRS})
    target_link_libraries(test-memory-aware-planner PRIVATE llama llama-common Threads::Threads)

    add_executable(test-planner-no-overlap test-planner-no-overlap.cpp layer_planner.cpp memory_estimator.cpp dist_common.cpp)
    target_include_directories(test-planner-no-overlap PRIVATE ${DIST_INCLUDE_DIRS})
    target_link_libraries(test-planner-no-overlap PRIVATE llama llama-common Threads::Threads)

    add_executable(test-planner-low-memory test-planner-low-memory.cpp layer_planner.cpp memory_estimator.cpp dist_common.cpp)
    target_include_directories(test-planner-low-memory PRIVATE ${DIST_INCLUDE_DIRS})
    target_link_libraries(test-planner-low-memory PRIVATE llama llama-common Threads::Threads)

    add_executable(test-capacity-api test-capacity-api.cpp memory_estimator.cpp dist_common.cpp)
    target_include_directories(test-capacity-api PRIVATE ${DIST_INCLUDE_DIRS})
    target_link_libraries(test-capacity-api PRIVATE llama llama-common cpp-httplib Threads::Threads)

    add_executable(test-probe-memory test-probe-memory.cpp dist_common.cpp)
    target_include_directories(test-probe-memory PRIVATE ${DIST_INCLUDE_DIRS})
    target_link_libraries(test-probe-memory PRIVATE llama Threads::Threads)

    set(MODEL_REGISTRY_SOURCES
        orchestrator/model_registry.cpp
        orchestrator/model_provider/model_provider.cpp
        orchestrator/model_provider/huggingface_provider.cpp
        orchestrator/manifest_builder/manifest_builder.cpp
        ${DIST_ARCHITECTURE_DESCRIPTOR_SOURCES}
        orchestrator/layout_planner/layout_planner.cpp
        orchestrator/layout_estimator/layout_estimator.cpp
        orchestrator/optimizer/cluster_optimizer.cpp
        orchestrator/coverage/coverage.cpp
        orchestrator/install_planner/install_planner.cpp
        memory_estimator.cpp
        dist_common.cpp
    )
    set(MODEL_REGISTRY_LINK_LIBS cpp-httplib llama Threads::Threads)

    # Task 9.1 model registry tests
    add_executable(test-model-registry test-model-registry.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-model-registry PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-model-registry PRIVATE cxx_std_17)
    target_link_libraries(test-model-registry PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-model-status test-model-status.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-model-status PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-model-status PRIVATE cxx_std_17)
    target_link_libraries(test-model-status PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-model-api test-model-api.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-model-api PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-model-api PRIVATE cxx_std_17)
    target_link_libraries(test-model-api PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    # Task 9.2 remote model discovery tests
    add_executable(test-provider-interface test-provider-interface.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-provider-interface PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-provider-interface PRIVATE cxx_std_17)
    target_link_libraries(test-provider-interface PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-discovery-result test-discovery-result.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-discovery-result PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-discovery-result PRIVATE cxx_std_17)
    target_link_libraries(test-discovery-result PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-registry-discovery test-registry-discovery.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-registry-discovery PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-registry-discovery PRIVATE cxx_std_17)
    target_link_libraries(test-registry-discovery PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-huggingface-provider test-huggingface-provider.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-huggingface-provider PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-huggingface-provider PRIVATE cxx_std_17)
    target_link_libraries(test-huggingface-provider PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-model-discovery test-model-discovery.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-model-discovery PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-model-discovery PRIVATE cxx_std_17)
    target_link_libraries(test-model-discovery PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-discovery-errors test-discovery-errors.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-discovery-errors PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-discovery-errors PRIVATE cxx_std_17)
    target_link_libraries(test-discovery-errors PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-discovery-update test-discovery-update.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-discovery-update PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-discovery-update PRIVATE cxx_std_17)
    target_link_libraries(test-discovery-update PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    # Task 9.3 GGUF manifest builder tests
    add_executable(test-gguf-header test-gguf-header.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-gguf-header PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-gguf-header PRIVATE cxx_std_17)
    target_link_libraries(test-gguf-header PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-gguf-metadata test-gguf-metadata.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-gguf-metadata PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-gguf-metadata PRIVATE cxx_std_17)
    target_link_libraries(test-gguf-metadata PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-tensor-directory test-tensor-directory.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-tensor-directory PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-tensor-directory PRIVATE cxx_std_17)
    target_link_libraries(test-tensor-directory PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-layer-descriptor test-layer-descriptor.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-layer-descriptor PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-layer-descriptor PRIVATE cxx_std_17)
    target_link_libraries(test-layer-descriptor PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-manifest-builder test-manifest-builder.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-manifest-builder PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-manifest-builder PRIVATE cxx_std_17)
    target_link_libraries(test-manifest-builder PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-build-manifest test-build-manifest.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-build-manifest PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-build-manifest PRIVATE cxx_std_17)
    target_link_libraries(test-build-manifest PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-manifest-refresh test-manifest-refresh.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-manifest-refresh PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-manifest-refresh PRIVATE cxx_std_17)
    target_link_libraries(test-manifest-refresh PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-corrupted-header test-corrupted-header.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-corrupted-header PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-corrupted-header PRIVATE cxx_std_17)
    target_link_libraries(test-corrupted-header PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    # Task 9.4 desired cluster layout planner tests
    add_executable(test-layout-planner test-layout-planner.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-layout-planner PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-layout-planner PRIVATE cxx_std_17)
    target_link_libraries(test-layout-planner PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-layer-placement test-layer-placement.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-layer-placement PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-layer-placement PRIVATE cxx_std_17)
    target_link_libraries(test-layer-placement PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-memory-fit test-memory-fit.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-memory-fit PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-memory-fit PRIVATE cxx_std_17)
    target_link_libraries(test-memory-fit PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-no-overlap test-no-overlap.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-no-overlap PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-no-overlap PRIVATE cxx_std_17)
    target_link_libraries(test-no-overlap PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-full-coverage test-full-coverage.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-full-coverage PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-full-coverage PRIVATE cxx_std_17)
    target_link_libraries(test-full-coverage PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-gpu-priority test-gpu-priority.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-gpu-priority PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-gpu-priority PRIVATE cxx_std_17)
    target_link_libraries(test-gpu-priority PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-build-layout test-build-layout.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-build-layout PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-build-layout PRIVATE cxx_std_17)
    target_link_libraries(test-build-layout PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-layout-update test-layout-update.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-layout-update PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-layout-update PRIVATE cxx_std_17)
    target_link_libraries(test-layout-update PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-layout-node-loss test-layout-node-loss.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-layout-node-loss PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-layout-node-loss PRIVATE cxx_std_17)
    target_link_libraries(test-layout-node-loss PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    # Task 9.5 cluster coverage tests
    add_executable(test-coverage test-coverage.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-coverage PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-coverage PRIVATE cxx_std_17)
    target_link_libraries(test-coverage PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-reconcile test-reconcile.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-reconcile PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-reconcile PRIVATE cxx_std_17)
    target_link_libraries(test-reconcile PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-checksum test-checksum.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-checksum PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-checksum PRIVATE cxx_std_17)
    target_link_libraries(test-checksum PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-empty-cluster test-empty-cluster.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-empty-cluster PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-empty-cluster PRIVATE cxx_std_17)
    target_link_libraries(test-empty-cluster PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-partial-coverage test-partial-coverage.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-partial-coverage PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-partial-coverage PRIVATE cxx_std_17)
    target_link_libraries(test-partial-coverage PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-ready-coverage test-ready-coverage.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-ready-coverage PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-ready-coverage PRIVATE cxx_std_17)
    target_link_libraries(test-ready-coverage PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-degraded-coverage test-degraded-coverage.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-degraded-coverage PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-degraded-coverage PRIVATE cxx_std_17)
    target_link_libraries(test-degraded-coverage PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-node-report test-node-report.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-node-report PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-node-report PRIVATE cxx_std_17)
    target_link_libraries(test-node-report PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-coverage-refresh test-coverage-refresh.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-coverage-refresh PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-coverage-refresh PRIVATE cxx_std_17)
    target_link_libraries(test-coverage-refresh PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-reconcile-after-node-loss test-reconcile-after-node-loss.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-reconcile-after-node-loss PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-reconcile-after-node-loss PRIVATE cxx_std_17)
    target_link_libraries(test-reconcile-after-node-loss PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    # Task 9.6 install planner tests
    add_executable(test-install-plan test-install-plan.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-install-plan PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-install-plan PRIVATE cxx_std_17)
    target_link_libraries(test-install-plan PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-download-operation test-download-operation.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-download-operation PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-download-operation PRIVATE cxx_std_17)
    target_link_libraries(test-download-operation PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-repair-operation test-repair-operation.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-repair-operation PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-repair-operation PRIVATE cxx_std_17)
    target_link_libraries(test-repair-operation PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-delete-operation test-delete-operation.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-delete-operation PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-delete-operation PRIVATE cxx_std_17)
    target_link_libraries(test-delete-operation PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-operation-grouping test-operation-grouping.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-operation-grouping PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-operation-grouping PRIVATE cxx_std_17)
    target_link_libraries(test-operation-grouping PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-install-validation test-install-validation.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-install-validation PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-install-validation PRIVATE cxx_std_17)
    target_link_libraries(test-install-validation PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-build-install-plan test-build-install-plan.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-build-install-plan PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-build-install-plan PRIVATE cxx_std_17)
    target_link_libraries(test-build-install-plan PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-install-plan-after-node-loss test-install-plan-after-node-loss.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-install-plan-after-node-loss PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-install-plan-after-node-loss PRIVATE cxx_std_17)
    target_link_libraries(test-install-plan-after-node-loss PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    add_executable(test-install-plan-update test-install-plan-update.cpp ${MODEL_REGISTRY_SOURCES})
    target_include_directories(test-install-plan-update PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-install-plan-update PRIVATE cxx_std_17)
    target_link_libraries(test-install-plan-update PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    set(SYNC_AGENT_SOURCES
        node_agent/layer_store/layer_store.cpp
        node_agent/layer_store/layer_checksum.cpp
        node_agent/layer_store/layer_gguf_assembler.cpp
        node_agent/synchronization/sync_types.cpp
        node_agent/synchronization/synchronization_engine.cpp
        node_agent/synchronization/executors/http_range/http_range_executor.cpp
        orchestrator/install_planner/install_planner.cpp
        orchestrator/manifest_builder/manifest_builder.cpp
        dist_common.cpp
    )
    set(SYNC_AGENT_LINK_LIBS cpp-httplib llama Threads::Threads)

    # Task 9.7 synchronization engine tests
    add_executable(test-layer-store test-layer-store.cpp ${SYNC_AGENT_SOURCES})
    target_include_directories(test-layer-store PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-layer-store PRIVATE cxx_std_17)
    target_link_libraries(test-layer-store PRIVATE ${SYNC_AGENT_LINK_LIBS})

    add_executable(test-layer-verify test-layer-verify.cpp ${SYNC_AGENT_SOURCES})
    target_include_directories(test-layer-verify PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-layer-verify PRIVATE cxx_std_17)
    target_link_libraries(test-layer-verify PRIVATE ${SYNC_AGENT_LINK_LIBS})

    add_executable(test-http-range-executor test-http-range-executor.cpp ${SYNC_AGENT_SOURCES})
    target_include_directories(test-http-range-executor PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-http-range-executor PRIVATE cxx_std_17)
    target_link_libraries(test-http-range-executor PRIVATE ${SYNC_AGENT_LINK_LIBS})

    add_executable(test-install-job test-install-job.cpp ${SYNC_AGENT_SOURCES})
    target_include_directories(test-install-job PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-install-job PRIVATE cxx_std_17)
    target_link_libraries(test-install-job PRIVATE ${SYNC_AGENT_LINK_LIBS})

    add_executable(test-job-progress test-job-progress.cpp ${SYNC_AGENT_SOURCES})
    target_include_directories(test-job-progress PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-job-progress PRIVATE cxx_std_17)
    target_link_libraries(test-job-progress PRIVATE ${SYNC_AGENT_LINK_LIBS})

    add_executable(test-layer-delete test-layer-delete.cpp ${SYNC_AGENT_SOURCES})
    target_include_directories(test-layer-delete PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-layer-delete PRIVATE cxx_std_17)
    target_link_libraries(test-layer-delete PRIVATE ${SYNC_AGENT_LINK_LIBS})

    add_executable(test-sync-engine test-sync-engine.cpp ${SYNC_AGENT_SOURCES})
    target_include_directories(test-sync-engine PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-sync-engine PRIVATE cxx_std_17)
    target_link_libraries(test-sync-engine PRIVATE ${SYNC_AGENT_LINK_LIBS})

    add_executable(test-partial-install test-partial-install.cpp ${SYNC_AGENT_SOURCES})
    target_include_directories(test-partial-install PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-partial-install PRIVATE cxx_std_17)
    target_link_libraries(test-partial-install PRIVATE ${SYNC_AGENT_LINK_LIBS})

    add_executable(test-corrupted-layer test-corrupted-layer.cpp ${SYNC_AGENT_SOURCES})
    target_include_directories(test-corrupted-layer PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-corrupted-layer PRIVATE cxx_std_17)
    target_link_libraries(test-corrupted-layer PRIVATE ${SYNC_AGENT_LINK_LIBS})

    add_executable(test-coverage-after-sync test-coverage-after-sync.cpp
        ${MODEL_REGISTRY_SOURCES}
        node_agent/layer_store/layer_store.cpp
        node_agent/layer_store/layer_checksum.cpp
        node_agent/layer_store/layer_gguf_assembler.cpp
        node_agent/synchronization/sync_types.cpp
        node_agent/synchronization/synchronization_engine.cpp
        node_agent/synchronization/executors/http_range/http_range_executor.cpp)
    target_include_directories(test-coverage-after-sync PRIVATE ${DIST_INCLUDE_DIRS})
    target_compile_features(test-coverage-after-sync PRIVATE cxx_std_17)
    target_link_libraries(test-coverage-after-sync PRIVATE ${MODEL_REGISTRY_LINK_LIBS})

    # Task 9.8 cluster optimizer tests
    macro(dist_add_optimizer_test name)
        add_executable(${name} ${name}.cpp ${MODEL_REGISTRY_SOURCES})
        target_include_directories(${name} PRIVATE ${DIST_INCLUDE_DIRS})
        target_compile_features(${name} PRIVATE cxx_std_17)
        target_link_libraries(${name} PRIVATE ${MODEL_REGISTRY_LINK_LIBS})
    endmacro()

    dist_add_optimizer_test(test-layout-estimator)
    dist_add_optimizer_test(test-storage-only-node)
    dist_add_optimizer_test(test-performance-simulation)
    dist_add_optimizer_test(test-optimizer-policy)
    dist_add_optimizer_test(test-no-benefit)
    dist_add_optimizer_test(test-improvement-threshold)
    dist_add_optimizer_test(test-node-join)
    dist_add_optimizer_test(test-slow-node)
    dist_add_optimizer_test(test-node-loss)
    dist_add_optimizer_test(test-memory-change)

    macro(dist_add_verify_test name)
        add_executable(${name} ${name}.cpp ${VERIFICATION_SOURCES} ${NODE_AGENT_SOURCES} dist_common.cpp memory_estimator.cpp)
        target_include_directories(${name} PRIVATE ${DIST_INCLUDE_DIRS})
        target_compile_features(${name} PRIVATE cxx_std_17)
        target_link_libraries(${name} PRIVATE llama llama-common cpp-httplib Threads::Threads)
    endmacro()

    dist_add_verify_test(test-gguf-inspector)
    dist_add_verify_test(test-gguf-diff)
    dist_add_verify_test(test-tensor-checksum)
    dist_add_verify_test(test-layer-checksum)
    dist_add_verify_test(test-alignment)
    dist_add_verify_test(test-materialization-repeatability)
    dist_add_verify_test(test-materialized-load)
    dist_add_verify_test(test-logit-parity)
    dist_add_verify_test(test-sampling-parity)
    dist_add_verify_test(test-materialization-verification)
    dist_add_verify_test(test-layer-store-verification)
    dist_add_verify_test(test-verification-multi-model)

    set(ARCHITECTURE_TEST_SOURCES
        ${DIST_ARCHITECTURE_DESCRIPTOR_SOURCES}
        orchestrator/manifest_builder/manifest_builder.cpp
        dist_common.cpp
    )
    macro(dist_add_architecture_test name)
        add_executable(${name} ${name}.cpp ${ARCHITECTURE_TEST_SOURCES})
        target_include_directories(${name} PRIVATE ${DIST_INCLUDE_DIRS})
        target_compile_features(${name} PRIVATE cxx_std_17)
        target_link_libraries(${name} PRIVATE llama llama-common Threads::Threads)
    endmacro()

    dist_add_architecture_test(test-architecture-descriptor)
    dist_add_architecture_test(test-semantic-roles)
    dist_add_architecture_test(test-worker-requirements)
    dist_add_architecture_test(test-tied-embeddings)
    dist_add_architecture_test(test-output-head)
    dist_add_architecture_test(test-qwen-descriptor)
    dist_add_architecture_test(test-gemma-descriptor)
    dist_add_architecture_test(test-deepseek-descriptor)
endif()
