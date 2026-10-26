cmake_minimum_required(VERSION 4.0)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")

project(vkpp_examples LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(ProjectContracts)
include(FetchContent)
if(NOT DEFINED FETCHCONTENT_SOURCE_DIR_VKPP AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../Vulkan-pp/CMakeLists.txt")
  set(FETCHCONTENT_SOURCE_DIR_VKPP "${CMAKE_CURRENT_SOURCE_DIR}/../Vulkan-pp" CACHE PATH "Local Vulkan-pp checkout (W1 workspace)")
endif()
FetchContent_Declare(vkpp
  GIT_REPOSITORY git@github.com:F1STjd/Vulkan-pp.git
  GIT_TAG        d77f5653fa8e747304b534e9b95682d601d96a7c
  FIND_PACKAGE_ARGS CONFIG)
  set(VKPP_WITH_GPU ON CACHE BOOL "Build vkpp::gpu for the transform example")
FetchContent_MakeAvailable(vkpp)

if(NOT COMMAND vkpp_enable_warnings)
  include(ProjectWarnings)
endif()

find_package(Vulkan REQUIRED)
find_package(SFML 3 REQUIRED COMPONENTS Window Graphics System)
include(ImGui)
include(SlangShaders)

set(SHADER_SLANG_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.slang)
add_slang_shader_target(slang_shaders SOURCES ${SHADER_SLANG_SOURCES})

set(TEXTURE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/textures")
set(MODEL_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/models")

add_subdirectory(examples/f1st)
add_subdirectory(examples/transform)
