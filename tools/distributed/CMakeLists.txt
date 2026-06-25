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
endif()
