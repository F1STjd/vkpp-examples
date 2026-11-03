add_executable(square_total main.cpp)
target_compile_features(square_total PRIVATE cxx_std_26)
set_target_properties(square_total
  PROPERTIES
    CXX_MODULE_STD ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_libraries(square_total PRIVATE vkpp::gpu vkpp::vkpp Vulkan::cppm)
target_link_libraries(square_total PRIVATE $<$<PLATFORM_ID:Windows>:stdc++exp>)
vkpp_enable_warnings(square_total)