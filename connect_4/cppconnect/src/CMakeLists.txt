include(CMakePrintHelpers)
cmake_print_variables(CMAKE_MODULE_PATH)
find_package(tensorflow REQUIRED)

add_library(lib STATIC Tree_search.cpp Game_controller.cpp)
target_include_directories(lib PUBLIC ${CMAKE_SOURCE_DIR}/include ${tensorflow_INCLUDE_DIRS})
target_link_libraries(lib PUBLIC ${tensorflow_LIBRARIES})

#nanobind_add_module(player ../include/Player.h Player.cpp ../bin/main.cpp)
nanobind_add_module(cppconnect cppconnect.cpp)
target_link_libraries(cppconnect PRIVATE lib)