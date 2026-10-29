add_executable(reduce main.cpp)
target_compile_features(reduce PRIVATE cxx_std_26)
set_target_properties(reduce
  PROPERTIES
    CXX_MODULE_STD ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_libraries(reduce PRIVATE vkpp::gpu vkpp::vkpp Vulkan::cppm)
target_link_libraries(reduce PRIVATE $<$<PLATFORM_ID:Windows>:stdc++exp>)
vkpp_enable_warnings(reduce)