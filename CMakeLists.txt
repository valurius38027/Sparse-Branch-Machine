cmake_minimum_required(VERSION 3.20)
project(sparse_branch_machine VERSION 4.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

option(SBM_ENABLE_NATIVE "Enable native CPU tuning" ON)
option(SBM_BUILD_TESTS "Build tests" ON)
option(SBM_BUILD_CLI "Build the thin CLI executable" ON)
option(SBM_BUILD_PYTHON_SMOKE_TEST "Run the Python ctypes smoke test with CTest" ON)

function(sbm_configure_target target)
  target_include_directories(${target} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  if(SBM_ENABLE_NATIVE AND NOT MSVC)
    target_compile_options(${target} PRIVATE -O3 -march=native)
  endif()
  set_target_properties(${target} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})
endfunction()

add_library(sbm_core SHARED
  src/modules/vector_ops.cpp
  src/modules/signature.cpp
)
sbm_configure_target(sbm_core)

add_library(sbm_machine SHARED
  src/modules/address_interpreter.cpp
  src/modules/implicit_output.cpp
  src/modules/sparse_output_policy.cpp
  src/modules/machine_storage.cpp
  src/modules/machine_checkpoint.cpp
  src/modules/machine_topology.cpp
  src/modules/machine_routing.cpp
  src/modules/machine_learning.cpp
  src/modules/token_objective.cpp
  src/modules/token_sparse_output.cpp
  src/modules/machine_maintenance.cpp
)
sbm_configure_target(sbm_machine)
target_link_libraries(sbm_machine PUBLIC sbm_core)

add_library(sbm_dataset SHARED
  src/modules/dataset.cpp
  src/modules/token_shard.cpp
)
sbm_configure_target(sbm_dataset)
target_link_libraries(sbm_dataset PUBLIC sbm_core)

add_library(sbm_experiment SHARED
  src/modules/experiment.cpp
  src/modules/token_experiment.cpp
)
sbm_configure_target(sbm_experiment)
target_link_libraries(sbm_experiment PUBLIC sbm_machine sbm_dataset sbm_core)

# Stable C ABI loaded by the CLI and Python ctypes wrapper.
add_library(sbm_api SHARED
  src/api/c_api.cpp
)
sbm_configure_target(sbm_api)
target_compile_definitions(sbm_api PRIVATE SBM_API_BUILD)
set_target_properties(sbm_api PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)
target_link_libraries(sbm_api PRIVATE sbm_experiment sbm_machine sbm_dataset sbm_core)

# Compatibility umbrella target for direct C++ embedding.
add_library(sbm INTERFACE)
target_include_directories(sbm INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(sbm INTERFACE sbm_experiment sbm_machine sbm_dataset sbm_core)

if(SBM_BUILD_CLI)
  add_executable(sbm_cli src/cli/main.cpp)
  set_target_properties(sbm_cli PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
  target_include_directories(sbm_cli PRIVATE include)
  target_link_libraries(sbm_cli PRIVATE sbm_api)
  if(NOT MSVC)
    # The CLI is a thin argument parser.  Optimizing it at -O3 materially
    # increases build latency without affecting model throughput.
    target_compile_options(sbm_cli PRIVATE -O1)
  endif()
  # Preserve the historical build target name without producing another exe.
  add_executable(sbm_benchmark ALIAS sbm_cli)
endif()

if(SBM_BUILD_TESTS)
  enable_testing()

  function(sbm_enable_test_assertions target)
    if(MSVC)
      target_compile_options(${target} PRIVATE /UNDEBUG)
    else()
      target_compile_options(${target} PRIVATE -UNDEBUG)
    endif()
  endfunction()

  add_executable(sbm_tests tests/test_sbm.cpp)
  target_link_libraries(sbm_tests PRIVATE sbm)
  sbm_enable_test_assertions(sbm_tests)
  add_test(NAME sbm_cpp_api COMMAND sbm_tests)

  add_executable(sbm_c_api_tests tests/test_c_api.cpp)
  target_include_directories(sbm_c_api_tests PRIVATE include)
  target_link_libraries(sbm_c_api_tests PRIVATE sbm_api)
  sbm_enable_test_assertions(sbm_c_api_tests)
  add_test(NAME sbm_c_api COMMAND sbm_c_api_tests)

  add_executable(sbm_scaling_tests tests/test_scaling.cpp)
  target_link_libraries(sbm_scaling_tests PRIVATE sbm)
  sbm_enable_test_assertions(sbm_scaling_tests)
  add_test(NAME sbm_scaling_report COMMAND sbm_scaling_tests report)
  add_test(NAME sbm_scaling_address COMMAND sbm_scaling_tests address)
  add_test(NAME sbm_scaling_implicit_output COMMAND sbm_scaling_tests implicit)
  add_test(NAME sbm_scaling_output_memory COMMAND sbm_scaling_tests output)
  add_test(NAME sbm_scaling_sparse_capacity COMMAND sbm_scaling_tests capacity)
  add_test(NAME sbm_scaling_prior COMMAND sbm_scaling_tests prior)
  add_test(NAME sbm_scaling_channels COMMAND sbm_scaling_tests channels)
  add_test(NAME sbm_scaling_address_frames COMMAND sbm_scaling_tests address_frames)
  add_test(NAME sbm_scaling_dependency_attribution
           COMMAND sbm_scaling_tests dependency_attribution)
  add_test(NAME sbm_scaling_gpaf_shadow
           COMMAND sbm_scaling_tests gpaf_shadow)
  add_test(NAME sbm_scaling_gpaf_retrieval
           COMMAND sbm_scaling_tests gpaf_retrieval)
  add_test(NAME sbm_scaling_gpaf_checkpoint
           COMMAND sbm_scaling_tests gpaf_checkpoint)
  add_test(NAME sbm_scaling_gpaf_frozen
           COMMAND sbm_scaling_tests gpaf_frozen)
  add_test(NAME sbm_scaling_gpaf_lifecycle
           COMMAND sbm_scaling_tests gpaf_lifecycle)
  add_test(NAME sbm_scaling_gpaf_structural_call
           COMMAND sbm_scaling_tests gpaf_structural_call)

  if(SBM_BUILD_PYTHON_SMOKE_TEST)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
      add_test(
        NAME sbm_python_api
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python_api.py
                --library $<TARGET_FILE:sbm_api>)
      add_test(
        NAME sbm_shard_tool
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_shard_tool.py)
      add_test(
        NAME sbm_tune_scheduler
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tune_scheduler.py
                --library $<TARGET_FILE:sbm_api>)
      add_test(
        NAME sbm_resumable_runner
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_resumable_runner.py
                --library $<TARGET_FILE:sbm_api>)
    endif()
  endif()
endif()

install(TARGETS sbm_core sbm_machine sbm_dataset sbm_experiment sbm_api
  EXPORT sbmTargets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
if(SBM_BUILD_CLI)
  install(TARGETS sbm_cli RUNTIME DESTINATION bin)
endif()
install(DIRECTORY include/ DESTINATION include)
install(FILES python/sbm_runtime.py DESTINATION python)
install(PROGRAMS scripts/tune.py scripts/run_python_api.py scripts/theory_ablation.py
  scripts/convert_naime_dataset.py DESTINATION bin)
