cmake_minimum_required(VERSION 3.3.0 FATAL_ERROR)
project(tests LANGUAGES CXX)

include(../cmake/utils.cmake)

if(NOT TORCH_FOUND)
	find_package(Torch REQUIRED)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(tests_INCLUDE_DIRS "${TORCH_INCLUDE_DIR};${PROJECT_INCLUDE_DIR}")
set(tests_LIBS "${TORCH_LIBRARIES};${PROJECT_LIBRARY}")
set(tests_DLLS "${TORCH_DLLS}")
set(tests_DEPS "torch;${PROJECT_TARGET}")

file(GLOB tests_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.h")
file(GLOB tests_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")

message_pkg_refs("tests" "${tests_DEFINES}" "${tests_HEADERS}" "${tests_SOURCES}"
                 "${tests_DEPS}" "${tests_LIBS}" "${tests_DLLS}")

add_executable(tests "${tests_HEADERS}" "${tests_SOURCES}")
add_dependencies(tests ${tests_DEPS})
set_property(TARGET tests PROPERTY CXX_STANDARD 14)
target_include_directories(tests PUBLIC ${tests_INCLUDE_DIRS})
target_link_libraries(tests ${tests_LIBS})
