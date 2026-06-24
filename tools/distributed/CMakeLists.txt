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

if (LLAMA_BUILD_TESTS)
    add_executable(test-layer-planner test-layer-planner.cpp ${DIST_PLANNER_SOURCES})
    target_include_directories(test-layer-planner PRIVATE ${DIST_INCLUDE_DIRS})
    target_link_libraries(test-layer-planner PRIVATE Threads::Threads)

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
endif()
