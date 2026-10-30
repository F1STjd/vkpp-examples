add_executable(normalise main.cpp)
target_compile_features(normalise PRIVATE cxx_std_26)
set_target_properties(normalise
  PROPERTIES
    CXX_MODULE_STD ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_libraries(normalise PRIVATE vkpp::gpu vkpp::vkpp Vulkan::cppm)
target_link_libraries(normalise PRIVATE $<$<PLATFORM_ID:Windows>:stdc++exp>)
vkpp_enable_warnings(normalise)