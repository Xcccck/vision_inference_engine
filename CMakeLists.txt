cmake_minimum_required(VERSION 3.16)

project(vision_inference_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Heavy backends stay off until that phase exists. Turning one on without
# the SDK warns and leaves the core library unchanged.
option(VIE_ENABLE_CUDA "Enable CUDA" OFF)
option(VIE_ENABLE_TENSORRT "Enable native TensorRT" OFF)
option(VIE_ENABLE_ONNXRUNTIME "Enable ONNX Runtime" OFF)
option(VIE_ENABLE_OPENCV "Enable OpenCV" OFF)
option(VIE_BUILD_TESTS "Build GoogleTest targets" ON)

if(VIE_ENABLE_CUDA)
  find_package(CUDAToolkit QUIET)
  if(NOT CUDAToolkit_FOUND)
    message(WARNING "VIE_ENABLE_CUDA is ON but CUDAToolkit was not found. CUDA stays disabled.")
    set(VIE_ENABLE_CUDA OFF CACHE BOOL "Enable CUDA" FORCE)
  else()
    message(STATUS "CUDA toolkit: ${CUDAToolkit_VERSION}")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 89)
    endif()
    enable_language(CUDA)
  endif()
endif()

if(VIE_ENABLE_TENSORRT)
  message(WARNING "VIE_ENABLE_TENSORRT is ON, but this version of VIE has no TensorRT sources yet. TensorRT stays disabled.")
  set(VIE_ENABLE_TENSORRT OFF CACHE BOOL "Enable native TensorRT" FORCE)
endif()

set(VIE_ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime CPU prefix (include/ and lib/)")
set(VIE_ONNXRUNTIME_GPU_ROOT "" CACHE PATH "ONNX Runtime GPU prefix; used only when CUDA is enabled")
if(VIE_ENABLE_ONNXRUNTIME)
  if(VIE_ONNXRUNTIME_ROOT STREQUAL "" AND DEFINED ENV{ONNXRUNTIME_ROOT})
    set(VIE_ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
  endif()
  find_path(VIE_ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS
      ${VIE_ONNXRUNTIME_ROOT}
      /opt/onnxruntime-linux-x64-1.20.1
    PATH_SUFFIXES include)
  find_library(VIE_ONNXRUNTIME_LIBRARY onnxruntime
    HINTS
      ${VIE_ONNXRUNTIME_ROOT}
      /opt/onnxruntime-linux-x64-1.20.1
    PATH_SUFFIXES lib lib64)
  if(NOT VIE_ONNXRUNTIME_INCLUDE_DIR OR NOT VIE_ONNXRUNTIME_LIBRARY)
    message(WARNING
      "VIE_ENABLE_ONNXRUNTIME is ON but ONNX Runtime was not found. "
      "Set VIE_ONNXRUNTIME_ROOT to the CPU SDK prefix. ONNX Runtime stays disabled.")
    set(VIE_ENABLE_ONNXRUNTIME OFF CACHE BOOL "Enable ONNX Runtime" FORCE)
  else()
    message(STATUS "ONNX Runtime include: ${VIE_ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "ONNX Runtime library: ${VIE_ONNXRUNTIME_LIBRARY}")
    get_filename_component(VIE_ONNXRUNTIME_LIB_DIR "${VIE_ONNXRUNTIME_LIBRARY}" DIRECTORY)
  endif()
endif()

set(VIE_ENABLE_ONNXRUNTIME_CUDA OFF)
if(VIE_ENABLE_CUDA AND VIE_ENABLE_ONNXRUNTIME AND NOT VIE_ONNXRUNTIME_GPU_ROOT STREQUAL "")
  find_path(VIE_ONNXRUNTIME_GPU_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS ${VIE_ONNXRUNTIME_GPU_ROOT}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH)
  find_library(VIE_ONNXRUNTIME_GPU_LIBRARY onnxruntime
    HINTS ${VIE_ONNXRUNTIME_GPU_ROOT}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH)
  if(NOT VIE_ONNXRUNTIME_GPU_INCLUDE_DIR OR NOT VIE_ONNXRUNTIME_GPU_LIBRARY)
    message(WARNING
      "VIE_ONNXRUNTIME_GPU_ROOT is set but the GPU package was not found. "
      "The CUDA ONNX Runtime backend stays disabled.")
  else()
    set(VIE_ENABLE_ONNXRUNTIME_CUDA ON)
    get_filename_component(VIE_ONNXRUNTIME_GPU_LIB_DIR "${VIE_ONNXRUNTIME_GPU_LIBRARY}" DIRECTORY)
    message(STATUS "ONNX Runtime GPU include: ${VIE_ONNXRUNTIME_GPU_INCLUDE_DIR}")
    message(STATUS "ONNX Runtime GPU library: ${VIE_ONNXRUNTIME_GPU_LIBRARY}")
  endif()
endif()

find_package(nlohmann_json 3.2.0 REQUIRED)

add_library(vie
  src/core/types.cpp
  src/core/device_context.cpp
  src/core/tensor_storage.cpp
  src/core/tensor.cpp
  src/core/cuda_memory.cpp
  src/model/manifest.cpp
  src/model/manifest_validator.cpp
  src/model/model_inspection.cpp
  src/backend/backend_registry.cpp
  src/backend/mock_backend.cpp
  src/decoder/decoder_registry.cpp
  src/decoder/classification_decoder.cpp
  src/decoder/classification_probabilities_decoder.cpp
  src/decoder/mock_decoder.cpp
  src/decoder/segmentation.cpp
  src/decoder/semantic_segmentation.cpp
  src/decoder/detection_ops.cpp
  src/decoder/yolov8_cpu_decode.cpp
  src/decoder/yolov8_detect_decoder.cpp
  src/decoder/obb_geometry.cpp
  src/decoder/yolov8_obb_decoder.cpp
  src/decoder/yolov8_pose_decoder.cpp
  src/decoder/rtdetr_decoder.cpp
  src/decoder/rtdetr_cuda_decoder.cpp
  src/decoder/yolov8_seg_decoder.cpp
  src/engine/inference_engine.cpp
  src/analyzer/onnx_analyzer.cpp
  src/analyzer/analysis_format.cpp
  src/recognizer/recognition.cpp
  src/recognizer/recognizer_registry.cpp
  src/recognizer/yolov8_detect_recognizer.cpp
  src/recognizer/yolov8_obb_recognizer.cpp
  src/recognizer/yolov8_pose_recognizer.cpp
  src/recognizer/yolov8_seg_recognizer.cpp
  src/recognizer/rtdetr_recognizer.cpp
  src/recognizer/classification_recognizer.cpp
  src/recognizer/deeplabv3_semantic_segmentation_recognizer.cpp
  src/recognizer/manifest_generator.cpp
  src/preprocess/preprocess.cpp
  src/preprocess/preprocess_cpu.cpp
  src/runtime/thread_pool.cpp
  src/runtime/runtime_metrics.cpp
  src/runtime/model_instance.cpp
  src/runtime/request_queue.cpp
  src/runtime/dynamic_batch_aggregator.cpp
  src/runtime/batch_execution_queue.cpp
  src/runtime/execution_resource_manager.cpp
  src/runtime/cuda_input_staging.cpp
  src/runtime/runtime.cpp
  src/diagnostics/ort_profile_parser.cpp
)

add_library(vie::vie ALIAS vie)

target_include_directories(vie
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(vie PRIVATE nlohmann_json::nlohmann_json)

if(VIE_ENABLE_CUDA)
  target_sources(vie PRIVATE src/preprocess/preprocess_cuda.cu src/decoder/yolov8_detect_decoder.cu
                 src/decoder/yolov8_seg_decoder.cu src/decoder/rtdetr_cuda_decoder.cu)
  target_link_libraries(vie PUBLIC CUDA::cudart)
  target_compile_definitions(vie PUBLIC VIE_HAS_CUDA=1)
  # nvcc emits GNU line markers. -Wpedantic treats those as errors, so this
  # translation unit keeps -Wall -Wextra -Werror and turns that one warning off.
  set_source_files_properties(src/preprocess/preprocess_cuda.cu src/decoder/yolov8_detect_decoder.cu PROPERTIES
    COMPILE_OPTIONS "-Xcompiler=-Wall,-Wextra,-Werror,-Wno-pedantic")
  set_source_files_properties(src/decoder/rtdetr_cuda_decoder.cu PROPERTIES
    COMPILE_OPTIONS "--fmad=false;-Xcompiler=-Wall,-Wextra,-Werror,-Wno-pedantic")
  set_source_files_properties(src/decoder/yolov8_seg_decoder.cu PROPERTIES
    COMPILE_OPTIONS "--fmad=false;-Xcompiler=-Wall,-Wextra,-Werror,-Wno-pedantic")
endif()

if(VIE_ENABLE_ONNXRUNTIME)
  target_sources(vie PRIVATE src/backend/onnxruntime_backend.cpp)
  target_compile_definitions(vie PUBLIC VIE_HAS_ONNXRUNTIME=1)
  if(VIE_ENABLE_ONNXRUNTIME_CUDA)
    # One process can load one libonnxruntime.so.1. The GPU package also
    # contains the CPU execution provider. The CPU backend still requests only
    # that provider; this binary does not link the CPU package as well.
    target_sources(vie PRIVATE src/backend/onnxruntime_cuda_backend.cpp)
    target_include_directories(vie PRIVATE "${VIE_ONNXRUNTIME_GPU_INCLUDE_DIR}")
    target_link_libraries(vie PUBLIC "${VIE_ONNXRUNTIME_GPU_LIBRARY}")
    target_compile_definitions(vie PUBLIC VIE_HAS_ONNXRUNTIME_CUDA=1)
  else()
    target_include_directories(vie PRIVATE "${VIE_ONNXRUNTIME_INCLUDE_DIR}")
    target_link_libraries(vie PUBLIC "${VIE_ONNXRUNTIME_LIBRARY}")
  endif()
endif()

if(VIE_ENABLE_OPENCV)
  # Optional cv::Mat -> ImageInput adapter. Only the core module is needed; VIE
  # does not decode, read files or process colors here. When this option is OFF
  # the core library has no OpenCV include path, no link dependency and no
  # VIE_HAS_OPENCV define, so include/vie/opencv/image_input.hpp compiles away.
  find_package(OpenCV QUIET COMPONENTS core)
  if(NOT OpenCV_FOUND)
    message(WARNING "VIE_ENABLE_OPENCV is ON but OpenCV was not found. The cv::Mat adapter stays disabled.")
    set(VIE_ENABLE_OPENCV OFF CACHE BOOL "Enable the optional cv::Mat input adapter" FORCE)
  else()
    message(STATUS "OpenCV ${OpenCV_VERSION}: cv::Mat input adapter enabled")
    target_sources(vie PRIVATE src/opencv/image_input.cpp)
    target_include_directories(vie PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(vie PUBLIC ${OpenCV_LIBS})
    target_compile_definitions(vie PUBLIC VIE_HAS_OPENCV=1)
  endif()
endif()

target_compile_features(vie PUBLIC cxx_std_17)

target_compile_options(vie PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror>)

target_compile_definitions(vie
  PUBLIC
    VIE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    VIE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    VIE_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

if(VIE_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  add_executable(vie_tests
    tests/test_tensor.cpp
    tests/test_batch_inference.cpp
    tests/test_classification.cpp
    tests/test_manifest.cpp
    tests/test_engine.cpp
    tests/test_engine_segmentation.cpp
    tests/test_integration_regression.cpp
    tests/test_onnx_analyzer.cpp
    tests/test_recognizer.cpp
    tests/test_classification_recognizer.cpp
    tests/test_classification_real_models.cpp
    tests/test_yolov8_seg_recognizer.cpp
    tests/test_obb.cpp
    tests/test_pose.cpp
    tests/test_rtdetr_recognizer.cpp
    tests/test_rtdetr_decoder.cpp
    tests/test_preprocess.cpp
    tests/test_preprocess_resize_crop.cpp
    tests/test_decoder.cpp
    tests/test_segmentation.cpp
    tests/test_semantic_segmentation.cpp
    tests/test_yolov8_seg.cpp
    tests/test_model_inspection.cpp
    tests/test_runtime.cpp
    tests/test_runtime_audit.cpp
    tests/test_async_runtime.cpp
    tests/test_dynamic_batch_aggregator.cpp
    tests/test_runtime_concurrency.cpp
    tests/test_async_lifecycle_audit.cpp
    tests/test_execution_scheduling.cpp
    tests/test_ort_profile_parser.cpp
  )
  target_link_libraries(vie_tests PRIVATE vie::vie GTest::gtest GTest::gtest_main)
  target_compile_options(vie_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_compile_definitions(vie_tests PRIVATE VIE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  if(VIE_ENABLE_OPENCV)
    target_sources(vie_tests PRIVATE tests/test_opencv_input.cpp)
  endif()
  if(VIE_ENABLE_ONNXRUNTIME)
    target_sources(vie_tests PRIVATE tests/test_onnxruntime_backend.cpp)
    if(VIE_ENABLE_ONNXRUNTIME_CUDA)
      target_sources(vie_tests PRIVATE tests/test_onnxruntime_cuda_backend.cpp)
      set_property(TARGET vie_tests APPEND PROPERTY BUILD_RPATH "${VIE_ONNXRUNTIME_GPU_LIB_DIR}")
    else()
      set_property(TARGET vie_tests APPEND PROPERTY BUILD_RPATH "${VIE_ONNXRUNTIME_LIB_DIR}")
    endif()
  endif()
  if(VIE_ENABLE_CUDA)
    target_sources(vie_tests PRIVATE tests/test_cuda_tensor.cpp tests/test_preprocess_cuda.cpp
                                   tests/test_decoder_cuda.cpp tests/test_yolov8_seg_cuda.cpp
                                   tests/test_rtdetr_cuda_decoder.cpp tests/test_batch_cuda.cpp)
    set_property(TARGET vie_tests APPEND PROPERTY BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}")
  endif()
  include(GoogleTest)
  gtest_discover_tests(vie_tests WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

add_executable(vie_analyze_onnx examples/analyze_onnx.cpp)
target_link_libraries(vie_analyze_onnx PRIVATE vie::vie)
target_compile_options(vie_analyze_onnx PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(vie_recognize_onnx examples/recognize_onnx.cpp)
target_link_libraries(vie_recognize_onnx PRIVATE vie::vie)
target_compile_options(vie_recognize_onnx PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(vie_model_inspect tools/vie_model_inspect.cpp)
target_link_libraries(vie_model_inspect PRIVATE vie::vie)
target_compile_options(vie_model_inspect PRIVATE -Wall -Wextra -Wpedantic -Werror)

# Standalone ORT profile / provider-placement auditor (CPU-only, no CUDA/Session).
add_executable(vie_ort_profile_inspect tools/vie_ort_profile_inspect.cpp)
target_link_libraries(vie_ort_profile_inspect PRIVATE vie::vie)
target_compile_options(vie_ort_profile_inspect PRIVATE -Wall -Wextra -Wpedantic -Werror)

# Model-deployment verifier: runs a real image through the VIE engine and emits
# session/detection/latency/ORT-profile records for validate_model_deployment.py.
add_executable(vie_deploy_verify tools/diagnostics/model_deployment/vie_deploy_verify.cpp)
target_link_libraries(vie_deploy_verify PRIVATE vie::vie)
target_compile_options(vie_deploy_verify PRIVATE -Wall -Wextra -Wpedantic -Werror)

# Phase 9.4 real-model semantic-segmentation verifier: recognize -> manifest ->
# preprocess/backend/decoder dump for the Python reference comparison.
add_executable(vie_semantic_verify tools/diagnostics/semantic_segmentation/vie_semantic_verify.cpp)
target_link_libraries(vie_semantic_verify PRIVATE vie::vie)
target_compile_options(vie_semantic_verify PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(multi_model_runtime examples/multi_model_runtime.cpp)
target_link_libraries(multi_model_runtime PRIVATE vie::vie)
target_compile_options(multi_model_runtime PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(runtime_benchmark tools/runtime_benchmark.cpp)
target_link_libraries(runtime_benchmark PRIVATE vie::vie)
target_compile_options(runtime_benchmark PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(real_model_benchmark tools/real_model_benchmark.cpp)
target_link_libraries(real_model_benchmark PRIVATE vie::vie)
target_compile_options(real_model_benchmark PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(batch_benchmark tools/batch_benchmark.cpp)
target_link_libraries(batch_benchmark PRIVATE vie::vie)
target_compile_options(batch_benchmark PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(batch_aggregator_benchmark tools/batch_aggregator_benchmark.cpp)
target_link_libraries(batch_aggregator_benchmark PRIVATE vie::vie)
target_compile_options(batch_aggregator_benchmark PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_compile_definitions(batch_aggregator_benchmark PRIVATE VIE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(multi_model_concurrency_benchmark tools/multi_model_concurrency_benchmark.cpp)
target_link_libraries(multi_model_concurrency_benchmark PRIVATE vie::vie)
target_compile_options(multi_model_concurrency_benchmark PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_compile_definitions(multi_model_concurrency_benchmark PRIVATE VIE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

if(VIE_ENABLE_ONNXRUNTIME)
  if(VIE_ENABLE_ONNXRUNTIME_CUDA)
    set_property(TARGET vie_analyze_onnx vie_recognize_onnx vie_model_inspect vie_ort_profile_inspect vie_deploy_verify vie_semantic_verify multi_model_runtime runtime_benchmark real_model_benchmark batch_benchmark batch_aggregator_benchmark multi_model_concurrency_benchmark
      APPEND PROPERTY BUILD_RPATH "${VIE_ONNXRUNTIME_GPU_LIB_DIR}")
  else()
    set_property(TARGET vie_analyze_onnx vie_recognize_onnx vie_model_inspect vie_ort_profile_inspect vie_deploy_verify vie_semantic_verify multi_model_runtime runtime_benchmark real_model_benchmark batch_benchmark batch_aggregator_benchmark multi_model_concurrency_benchmark
      APPEND PROPERTY BUILD_RPATH "${VIE_ONNXRUNTIME_LIB_DIR}")
  endif()
endif()
if(VIE_ENABLE_CUDA)
  set_property(TARGET vie_analyze_onnx vie_recognize_onnx vie_model_inspect vie_ort_profile_inspect vie_deploy_verify vie_semantic_verify multi_model_runtime runtime_benchmark real_model_benchmark batch_benchmark batch_aggregator_benchmark multi_model_concurrency_benchmark
    APPEND PROPERTY BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}")
endif()

# The Ultralytics-derived fixtures are not distributed with this repository
# (AGPL-3.0 weights), so a fresh clone starts without them. Say which ones are
# missing here, because the tests that read them fail until they are rebuilt.
set(VIE_MISSING_TEST_MODELS "")
foreach(vie_model IN ITEMS
    testdata/onnx/yolov8n.onnx
    testdata/onnx/yolov8n-seg.onnx
    testdata/onnx/rtdetr-l.onnx
    testdata/onnx_dynamic/yolov8n_batch.onnx)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${vie_model}")
    list(APPEND VIE_MISSING_TEST_MODELS "${vie_model}")
  endif()
endforeach()
if(VIE_MISSING_TEST_MODELS)
  message(WARNING
    "Test models are missing: ${VIE_MISSING_TEST_MODELS}. "
    "Run 'python3 tools/export_test_models.py' before ctest; the real-model "
    "tests skip and the sample tools fail without these files.")
endif()

message(STATUS "VIE ${PROJECT_VERSION}")
message(STATUS "  CUDA=${VIE_ENABLE_CUDA} TENSORRT=${VIE_ENABLE_TENSORRT} ONNXRUNTIME=${VIE_ENABLE_ONNXRUNTIME} ONNXRUNTIME_CUDA=${VIE_ENABLE_ONNXRUNTIME_CUDA} OPENCV=${VIE_ENABLE_OPENCV}")
message(STATUS "  TESTS=${VIE_BUILD_TESTS}")
