cmake_minimum_required(VERSION 3.3.0 FATAL_ERROR)
project(TestBench)

include(../cmake/configs.cmake)
include(../cmake/utils.cmake)
find_arch()

file(GLOB TestBench_INCLUDE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.h")
file(GLOB TestBench_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
if(WIN32)
    file(GLOB TestBench_INCLUDE_WINDOWS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "windows/*.h")
    file(GLOB TestBench_SOURCES_WINDOWS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "windows/*.cpp")
    list(APPEND TestBench_INCLUDE ${TestBench_INCLUDE_WINDOWS})
    list(APPEND TestBench_SOURCES ${TestBench_SOURCES_WINDOWS})
endif()

set(TestBench_DEFINES "")
set(TestBench_PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(TestBench_INCLUDE_DIRS "${PROJECT_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
set(TestBench_EXTERNAL_LIBS "")
set(TestBench_EXTERNAL_DLLS "")
set(TestBench_EXTERNAL_DEPS "")

# find torch to avoid conflicts across sub-projects
if(NOT TORCH_FOUND)
	find_package(Torch REQUIRED)
endif()
list(APPEND TestBench_INCLUDE_DIRS  "${TORCH_INCLUDE_DIR}")
list(APPEND TestBench_EXTERNAL_LIBS "${TORCH_LIBRARIES}")
list(APPEND TestBench_EXTERNAL_DLLS "${TORCH_DLLS}")
list(APPEND TestBench_EXTERNAL_DEPS "torch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
if(NOT TORCHVISION_FOUND)
    # https://github.com/pytorch/vision
    find_package(TorchVision REQUIRED)
endif()
list(APPEND TestBench_INCLUDE_DIRS  "${TORCHVISION_INCLUDE_DIRS}")
list(APPEND TestBench_EXTERNAL_LIBS "${TORCHVISION_TARGETS}")
list(APPEND TestBench_EXTERNAL_DLLS "${TORCHVISION_DLLS}")
if(NOT OPENCV_FOUND)
    # https://github.com/opencv/opencv
    # (better if Torch/TorchVision also compiled using it for optimizations)
	find_package(OpenCV REQUIRED)
endif()
set(OPENCV_INCLUDE_DIRS "${opencv_INCLUDE_DIR}")  # compat MSVC
message(DEBUG "Detected OpenCV includes: ${OPENCV_INCLUDE_DIRS}")
message(DEBUG "OPENCV VERSION MAJOR: ${OpenCV_VERSION_MAJOR}")
list(APPEND TestBench_INCLUDE_DIRS  "${OPENCV_INCLUDE_DIRS}")
list(APPEND TestBench_EXTERNAL_LIBS "${OPENCV_LIBRARIES}")
list(APPEND TestBench_EXTERNAL_DLLS "${OPENCV_DLLS}")
if(NOT CLI11_FOUND)
    # https://github.com/CLIUtils/CLI11
    find_package(CLI11 REQUIRED)
endif()
list(APPEND TestBench_INCLUDE_DIRS   "${CLI11_INCLUDE_DIR}")
list(APPEND TestBench_EXTERNAL_DEPS  "${CLI11_TARGETS}")

# add main project with lib extensions
list(APPEND TestBench_EXTERNAL_DEPS "${PROJECT_TARGET}")
list(APPEND TestBench_EXTERNAL_LIBS "${PROJECT_TARGET}")

list(APPEND TestBench_DEFINES -DUNICODE)
if(NOT ${ARCH_POSTFIX} STREQUAL "")
	list(APPEND TestBench_DEFINES -DAMD64)	# x64
endif()
set(TestBench_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TestBench_SOURCES_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

message(DEBUG "===========================================")
message_pkg_refs("TestBench" "${TestBench_DEFINES}" "${TestBench_INCLUDE}" "${TestBench_SOURCES}" "" "" "")
message(DEBUG "-------------------------------------------")
message_pkg_refs("External" "" "${TestBench_INCLUDE_DIRS}" ""
                 "${TestBench_EXTERNAL_DEPS}" "${TestBench_EXTERNAL_LIBS}" "${TestBench_EXTERNAL_DLLS}")
message(DEBUG "===========================================")

add_executable(TestBench ${TestBench_INCLUDE} ${TestBench_SOURCES})
target_include_directories(TestBench PRIVATE "${TestBench_INCLUDE_DIRS}")
set_target_properties(TestBench PROPERTIES PUBLIC_HEADER "${TestBench_INCLUDE}")
target_link_libraries(TestBench PRIVATE "${TestBench_EXTERNAL_LIBS}")
add_dependencies(TestBench ${TestBench_EXTERNAL_DEPS})
set_property(TARGET TestBench PROPERTY CXX_STANDARD 14)
foreach(def ${TestBench_DEFINES})
	target_compile_definitions(TestBench PUBLIC ${def})
endforeach(def)

# precompiled headers
option(WITH_PRECOMPILED_HEADER "Use precompiled stdafx headers" OFF)
option(MSVC_Z7_OVERRIDE "Enforce MSVC /Z7 CXX flag when using precompiled headers to fix sccache bug (recommended)" ON)
if(${WITH_PRECOMPILED_HEADER})
	message(STATUS "TestBench package using precompiled headers (see: WITH_PRECOMPILED_HEADER)")
	target_precompile_headers(TestBench PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../stdafx.h")
    if(NOT ${TORCH_FOUND})
        find_package(Torch REQUIRED)
    endif()
    target_compile_definitions(TestBench PUBLIC PRECOMPILE_TORCH)
    target_include_directories(TestBench PRIVATE ${TORCH_INCLUDE_DIRECTORIES})
	if(MSVC AND MSVC_Z7_OVERRIDE)
		# enforce the flag to avoid incompatibility with other targets that will use Core
		# by default cmake will specify /Z7 to debug targets not adding (just using) precompiled headers
		string(REGEX REPLACE "/Z[iI7]" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
		set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Z7")
	endif()
endif()

# installation
install(TARGETS TestBench
		EXPORT TestBench-targets
		PUBLIC_HEADER DESTINATION include/TestBench
		INCLUDES DESTINATION include/TestBench
		LIBRARY DESTINATION lib
)
#install(DIRECTORY ${TestBench_INCLUDE_DIR} DESTINATION TestBench)
#file(GLOB_RECURSE INSTALL_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/TestBench/*.h")
#install(FILES "${INSTALL_INCLUDES}" DESTINATION TestBench)
export(PACKAGE TestBench)

# provide variables to other projects
set(TestBench_DEFINES ${TestBench_DEFINES} PARENT_SCOPE)
set(TestBench_INCLUDE_DIR ${TestBench_INCLUDE_DIR} PARENT_SCOPE)
set(TestBench_SOURCES_DIR ${TestBench_SOURCES_DIR} PARENT_SCOPE)
