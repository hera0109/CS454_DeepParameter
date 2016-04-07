cmake_minimum_required(VERSION 2.8)

project(edge)
find_package( OpenCV REQUIRED )
add_executable(edge main.cpp)
target_link_libraries(edge ${OpenCV_LIBS} )

install(TARGETS pc_camera RUNTIME DESTINATION bin)
