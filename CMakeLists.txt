cmake_minimum_required(VERSION 4.0)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")

project(vkpp_examples LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(FetchContent)
FetchContent_Declare(vkpp
  GIT_REPOSITORY git@github.com:F1STjd/Vulkan-pp.git
  GIT_TAG        <sha from 72b.6>
  FIND_PACKAGE_ARGS CONFIG)
FetchContent_MakeAvailable(vkpp)

if(NOT COMMAND vkpp_enable_warnings)
  include(ProjectWarnings)
endif()
include(ProjectContracts)

find_package(Vulkan REQUIRED)
find_package(SFML 3 REQUIRED COMPONENTS Window Graphics System)
include(ImGui)
include(SlangShaders)

set(SHADER_SLANG_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.slang)
add_slang_shader_target(slang_shaders SOURCES ${SHADER_SLANG_SOURCES})

set(TEXTURE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/textures")
set(MODEL_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/models")

add_subdirectory(examples/f1st)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_custom_target(copy_compile_commands ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_BINARY_DIR}/compile_commands.json"
      "${CMAKE_SOURCE_DIR}/compile_commands.json")
endif()