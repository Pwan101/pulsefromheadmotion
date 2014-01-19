cmake_minimum_required (VERSION 2.8)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

project (pulsefromheadmotion)

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set (LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package (OpenCV REQUIRED)

include_directories ("${PROJECT_SOURCE_DIR}/lib")
add_library (pfhmlib SHARED
	include/sys_lib.h
	include/pfhm_lib.h)
set_target_properties(pfhmlib PROPERTIES LINKER_LANGUAGE CXX)

include_directories ("${PROJECT_SOURCE_DIR}/include")

add_executable (pfhmain src/pfhmain.cpp)
target_link_libraries(pfhmain ${OpenCV_LIBS})
target_link_libraries(pfhmain pfhmlib)