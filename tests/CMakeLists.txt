cmake_minimum_required(VERSION 3.3.0 FATAL_ERROR)
project(tests LANGUAGES CXX)

include(../cmake/utils.cmake)
set(tests_INCLUDE_DIRS "${PROJECT_INCLUDE_DIR}")
set(tests_LIBS "${PROJECT_LIBRARY}")
set(tests_DLLS "")
set(tests_DEPS "${PROJECT_TARGET}")

if(NOT TORCH_FOUND)
	find_package(Torch REQUIRED)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
list(APPEND tests_INCLUDE_DIRS "${TORCH_INCLUDE_DIR}")
list(APPEND tests_LIBS "${TORCH_LIBRARIES}")
list(APPEND tests_DLLS "${TORCH_DLLS}")
list(APPEND tests_DEPS "torch")

# if(NOT OPENCV_FOUND)
# 	find_package(OpenCV REQUIRED)
# endif()
# set(OPENCV_INCLUDE_DIRS "${OPENCV_INCLUDE_DIR}")  # compat MSVC
# list(APPEND tests_INCLUDE_DIRS  "${OPENCV_INCLUDE_DIRS}")
# list(APPEND tests_LIBS "${OPENCV_LIBRARIES}")
# list(APPEND tests_DLLS "${OPENCV_DLLS}")

file(GLOB tests_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.h")
file(GLOB tests_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")

message_pkg_refs("tests" "${tests_DEFINES}" "${tests_HEADERS}" "${tests_SOURCES}"
                 "${tests_DEPS}" "${tests_LIBS}" "${tests_DLLS}")

# each cpp is its own/standalone test executable
foreach(test_src ${tests_SOURCES})
    string(REPLACE ".cpp" "" test_name ${test_src} )
    add_executable(${test_name} "${tests_HEADERS}" "${test_src}")
    add_dependencies(${test_name} ${tests_DEPS})
    set_property(TARGET ${test_name} PROPERTY CXX_STANDARD 14)
    target_include_directories(${test_name} PUBLIC ${tests_INCLUDE_DIRS})
    target_link_libraries(${test_name} ${tests_LIBS})
endforeach()
