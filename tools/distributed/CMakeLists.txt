if (EMSCRIPTEN)
    return()
endif()

set(DIST_TRANSPORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/transport)
set(DIST_WORKERS_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/workers)

set(DIST_TRANSPORT_SOURCES
    ${DIST_TRANSPORT_DIR}/split_tcp_wire.cpp
)

set(DIST_COMMON_SOURCES
    ${DIST_TRANSPORT_SOURCES}
    dist_common.cpp
    model_catalog.cpp
    orchestrator/model_registry.cpp
    orchestrator/model_provider/model_provider.cpp
    orchestrator/model_provider/huggingface_provider.cpp
    orchestrator/manifest_builder/manifest_builder.cpp
    orchestrator/layout_planner/layout_planner.cpp
    orchestrator/coverage/coverage.cpp
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
    ${DIST_TRANSPORT_DIR}
    ${DIST_WORKERS_DIR}
    ${CMAKE_SOURCE_DIR}/vendor
    ${CMAKE_SOURCE_DIR}/vendor/cpp-httplib
)

function(dist_add_exe name main_source)
    add_executable(${name} ${main_source} ${DIST_COMMON_SOURCES} ${DIST_PLANNER_SOURCES})
    target_include_directories(${name} PRIVATE ${DIST_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE llama llama-common cpp-httplib Threads::Threads)
endfunction()

function(dist_add_exe_light name main_source)
    add_executable(${name} ${main_source} dist_common.cpp model_catalog.cpp ${DIST_PLANNER_SOURCES} ${DIST_BENCHMARK_SOURCES})
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
dist_add_exe(orchestrator orchestrator.cpp)

dist_add_worker(split_gen3_a ${DIST_WORKERS_DIR}/split_gen3_a.cpp)
dist_add_worker(split_gen3_b ${DIST_WORKERS_DIR}/split_gen3_b.cpp)
dist_add_worker(split_gen3_c ${DIST_WORKERS_DIR}/split_gen3_c.cpp)

# End-to-end cluster validation suite (Task 7.5). Defined unconditionally so a
# single `build.sh e2e` builds them even when LLAMA_BUILD_TESTS is OFF.
dist_add_test_exe(test-cluster-e2e test-cluster-e2e.cpp)
dist_add_test_exe(test-cluster-e2e-install-reuse test-cluster-e2e-install-reuse.cpp)
dist_add_test_exe(test-cluster-e2e-node-loss test-cluster-e2e-node-loss.cpp)
dist_add_test_exe(test-cluster-e2e-orchestrator-restart test-cluster-e2e-orchestrator-restart.cpp)

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
        orchestrator/layout_planner/layout_planner.cpp
        orchestrator/coverage/coverage.cpp
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
endif()
