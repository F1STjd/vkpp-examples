add_executable(scan main.cpp)
target_compile_features(scan PRIVATE cxx_std_26)
set_target_properties(scan
  PROPERTIES
    CXX_MODULE_STD ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_libraries(scan PRIVATE vkpp::gpu vkpp::vkpp Vulkan::cppm)
target_link_libraries(scan PRIVATE $<$<PLATFORM_ID:Windows>:stdc++exp>)
vkpp_enable_warnings(scan)