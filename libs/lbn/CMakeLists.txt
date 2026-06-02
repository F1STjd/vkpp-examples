add_library(lbn STATIC)
target_sources(lbn
  PUBLIC FILE_SET CXX_MODULES
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    FILES lbn.cppm
  PRIVATE lbn.cpp)
target_compile_features(lbn PUBLIC cxx_std_26)
set_target_properties(lbn PROPERTIES CXX_MODULE_STD ON)
target_link_libraries(lbn PUBLIC Vulkan::cppm utils SFML::Window PRIVATE project_warnings)
