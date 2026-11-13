add_executable(histogram main.cpp)
target_compile_features(histogram PRIVATE cxx_std_26)
set_target_properties(histogram
  PROPERTIES
    CXX_MODULE_STD ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_libraries(histogram PRIVATE vkpp::gpu vkpp::vkpp Vulkan::cppm)
target_link_libraries(histogram PRIVATE $<$<PLATFORM_ID:Windows>:stdc++exp>)
vkpp_enable_warnings(histogram)