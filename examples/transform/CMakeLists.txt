add_executable(transform main.cpp)
target_compile_features(transform PRIVATE cxx_std_26)
set_target_properties(transform
  PROPERTIES
    CXX_MODULE_STD ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_libraries(transform PRIVATE vkpp::gpu vkpp::vkpp Vulkan::cppm)
target_link_libraries(transform PRIVATE $<$<PLATFORM_ID:Windows>:stdc++exp>)
vkpp_enable_warnings(transform)