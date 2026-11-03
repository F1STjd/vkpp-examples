add_executable(particles main.cpp)
target_compile_features(particles PRIVATE cxx_std_26)
set_target_properties(particles
  PROPERTIES
    CXX_MODULE_STD ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_libraries(particles PRIVATE vkpp::gpu vkpp::vkpp Vulkan::cppm)
target_link_libraries(particles PRIVATE $<$<PLATFORM_ID:Windows>:stdc++exp>)
vkpp_enable_warnings(particles)
target_compile_options(particles PRIVATE -freflection)