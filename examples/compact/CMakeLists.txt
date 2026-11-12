add_executable(compact main.cpp)
target_compile_features(compact PRIVATE cxx_std_26)
set_target_properties(compact
  PROPERTIES
    CXX_MODULE_STD ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_libraries(compact PRIVATE vkpp::gpu vkpp::vkpp Vulkan::cppm)
target_link_libraries(compact PRIVATE $<$<PLATFORM_ID:Windows>:stdc++exp>)
vkpp_enable_warnings(compact)