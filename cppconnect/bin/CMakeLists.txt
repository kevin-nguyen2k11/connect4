add_executable(C___Connect_4 main.cpp)
#target_link_libraries(C___Connect_4 PRIVATE lib)
target_link_options(C___Connect_4 PRIVATE "LINKER:-ld_classic")