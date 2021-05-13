cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)

# solution name
set(PROJECT_SOLUTION_NAME "CRIM-PyTorch-Extensions" CACHE STRING "Name of the generated solution" FORCE)
if("${PROJECT_SOLUTION_NAME}" STREQUAL "")
	message(FATAL_ERROR "Solution name cannot be empty")
endif()
# generate the solution with given name (MSVC)
# this will set PROJECT_NAME
project(${PROJECT_SOLUTION_NAME} LANGUAGES CXX)

set(PROJECT_TARGET crim_libtorch_extensions)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake" FORCE)
include(cmake/configs.cmake)
include(cmake/utils.cmake)

set(INCLUDE_DIRS "")
set(EXTERNAL_LIBS "")
set(EXTERNAL_DEPS "")
set(EXTERNAL_DLLS "")

# build configuration type
#   https://stackoverflow.com/questions/31546278/where-to-set-cmake-configuration-types-in-a-project-with-subprojects
if(NOT __SET_UP_CONFIGURATIONS_DONE)
    set(__SET_UP_CONFIGURATIONS_DONE 1 CACHE STRING "")
    mark_as_advanced(__SET_UP_CONFIGURATIONS_DONE)

    # No reason to set CMAKE_CONFIGURATION_TYPES if it's not a multiconfig generator
    # Also no reason mess with CMAKE_BUILD_TYPE if it's a multiconfig generator.
    if(CMAKE_CONFIGURATION_TYPES) # multiconfig generator?
        set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
    else()
        if(NOT CMAKE_BUILD_TYPE)
            message("Defaulting to release build.")
            set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
        endif()
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY HELPSTRING "Choose the type of build")
        # set the valid options for cmake-gui drop-down list
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release")
    endif()
    # now set up the Profile configuration
    #set(CMAKE_C_FLAGS_PROFILE "...")
    #set(CMAKE_CXX_FLAGS_PROFILE "...")
    #set(CMAKE_EXE_LINKER_FLAGS_PROFILE "...")
endif()

# external modules
option(WITH_EXTERNAL_MODULE "Look for external module sources (must define its own cmake files)" OFF)
set(PROJECT_EXTERNAL_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external" CACHE STRING "External module location")
#mark_as_advanced(WITH_EXTERNAL_MODULE)
if(${WITH_EXTERNAL_MODULE})
	if(EXISTS "${PROJECT_EXTERNAL_MODULE_DIR}")
		add_subdirectory("${PROJECT_EXTERNAL_MODULE_DIR}")
	endif()
endif()

# tests
option(WITH_TESTS "Build tests modules" OFF)
if(WITH_TESTS)
    # FIXME: any tests to add?
endif()

# sources
set(PROJECT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(PROJECT_SOURCES_DIR "${PROJECT_SOURCE_DIR}/source")

option(WITH_DATA_AUGMENTATION "Include data augmentation operations (Requires OpenCV)" ON)
option(WITH_TEST_BENCH "Build TestBench CLI utility." ON)

file(GLOB_RECURSE PROJECT_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${PROJECT_INCLUDE_DIR}/*.h")
file(GLOB_RECURSE PROJECT_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${PROJECT_SOURCES_DIR}/*.cpp")

# data augmentation required by test bench
if(NOT WITH_DATA_AUGMENTATION AND WITH_TEST_BENCH)
    message(WARNING "Forcing WITH_DATA_AUGMENTATION=ON because WITH_TEST_BENCH was requested")
    set(WITH_DATA_AUGMENTATION CACHE BOOL ON FORCE)
endif()
if(NOT WITH_DATA_AUGMENTATION)
    # patch over globbing
    message(INFO "Disabled WITH_DATA_AUGMENTATION - Will remove all items in 'data' directories.")
    file(GLOB_RECURSE EXCLUDE_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${PROJECT_INCLUDE_DIR}/data/*.h")
    file(GLOB_RECURSE EXCLUDE_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${PROJECT_SOURCES_DIR}/data/*.cpp")
    foreach(file ${EXCLUDE_HEADERS})
        list(REMOVE_ITEM PROJECT_HEADERS ${file})
    endforeach()
    foreach(file ${EXCLUDE_SOURCES})
        list(REMOVE_ITEM PROJECT_SOURCES ${file})
    endforeach()
else()
    list()
endif()

message(DEBUG "Headers files for project ${PROJECT_TARGET}")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
    foreach(file ${PROJECT_HEADERS})
        message(DEBUG "Found ${file}")
    endforeach()
list(POP_BACK CMAKE_MESSAGE_INDENT)
message(DEBUG "Source files for project ${PROJECT_TARGET}")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
    foreach(file ${PROJECT_SOURCES})
        message(DEBUG "Found ${file}")
    endforeach()
list(POP_BACK CMAKE_MESSAGE_INDENT)
set(PROJECT_LIBRARY_TYPE "SHARED" CACHE STRING "Library type to build.")
set_property(CACHE PROJECT_LIBRARY_TYPE PROPERTY STRINGS SHARED STATIC)
if(MSVC)
    source_group(TREE "${PROJECT_INCLUDE_DIR}" PREFIX "Headers" FILES ${PROJECT_HEADERS})
    source_group(TREE "${PROJECT_SOURCES_DIR}" PREFIX "Sources" FILES ${PROJECT_SOURCES})
endif()

# torch libs
option(PROJECT_FORCE_THREADS "Force finding and adding Threads library.")
if(PROJECT_FORCE_THREAD)
    find_library(Threads REQUIRED)
    list(APPEND EXTERNAL_LIBS Threads)
endif()
if(NOT TORCH_FOUND)
	find_package(Torch REQUIRED)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
list(APPEND INCLUDE_DIRS  "${TORCH_INCLUDE_DIR}")
list(APPEND EXTERNAL_LIBS "${TORCH_LIBRARIES}")
list(APPEND EXTERNAL_DLLS "${TORCH_DLLS}")
list(APPEND EXTERNAL_DEPS "torch")

# depending on Python bindinds or not, create library or module
option(WITH_PYTHON "Enable Python support. Must match compilation of Torch library." ON)
message(DEBUG "WITH Python? ${WITH_PYTHON}")
if(${WITH_PYTHON})
    # find Python and pybind required to generate bindings
    find_package(Python COMPONENTS Development Interpreter HINTS ${PYTHON_EXECUTABLE})
    list(APPEND INCLUDE_DIRS ${Python_INCLUDE_DIRS})
    list(APPEND EXTERNAL_LIBS ${Python_LIBRARIES})

    if(NOT EXISTS ${PYBIND11_DIR})
        message(FATAL "Must set PYBIND11_DIR")
    endif()
    # add_subdirectory(${PYBIND11_DIR} ${CMAKE_BINARY_DIR}/pybind11 EXCLUDE_FROM_ALL)
    find_package(pybind11 COMPONENTS HINTS ${PYBIND11_DIR})

    message(DEBUG "Python Headers:" ${Python_INCLUDE_DIRS})
    message(DEBUG "PyBind Headers:" ${pybind11_INCLUDE_DIRS})

    # let pybind define all the required targets and libs references
    # it setups additional items that will no correctly install the python package if otherwise using 'add_library'
    pybind11_add_module(${PROJECT_TARGET} ${TORCH_INCLUDE_DIR} ${pybind11_INCLUDE_DIRS} ${PROJECT_HEADERS} ${PROJECT_SOURCES})
    # target_include_directories(${PROJECT_TARGET} ${Python_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS})

    # patch missing definitions between pybind and pytorch
    # (https://github.com/pytorch/pytorch/issues/38122)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
    list(APPEND EXTERNAL_LIBS ${TORCH_PYTHON_LIBRARY})

    remove_definitions(-DNO_PYTHON)
    # no bindings added if compiled directly for use in C++ only
    add_library(${PROJECT_TARGET} ${PROJECT_LIBRARY_TYPE} ${PROJECT_HEADERS} ${PROJECT_SOURCES})
endif()

# library
add_dependencies(${PROJECT_TARGET} ${EXTERNAL_DEPS})
set_property(TARGET ${PROJECT_TARGET} PROPERTY CXX_STANDARD 14)
set_property(TARGET ${PROJECT_TARGET} PROPERTY OUTPUT_NAME "${PROJECT_TARGET}")
set_target_properties(${PROJECT_TARGET} PROPERTIES PUBLIC_HEADER "${PROJECT_HEADERS}")
message(DEBUG "Include dirs:  " "${INCLUDE_DIRS}")
message(DEBUG "Include libs:  " "${LIBRARIES}")
message(DEBUG "External libs: " "${EXTERNAL_LIBS}")
target_include_directories(${PROJECT_TARGET} PUBLIC "${INCLUDE_DIRS}" "${PROJECT_INCLUDE_DIR}")
target_link_libraries(${PROJECT_TARGET} PUBLIC "${EXTERNAL_LIBS}")

# utilities
message(DEBUG "WITH TestBench? ${WITH_TEST_BENCH}")
if(WITH_TEST_BENCH)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/TestBench)
endif(WITH_TEST_BENCH)

# can be skipped if already processed by TestBench, but must process if added by itself
if(WITH_DATA_AUGMENTATION AND NOT WITH_TEST_BENCH)
    if(NOT OPENCV_FOUND)
        # https://github.com/opencv/opencv
        # (better if Torch/TorchVision also compiled using it for optimizations)
        find_package(OpenCV REQUIRED)
        set(OPENCV_INCLUDE_DIRS "${opencv_INCLUDE_DIR}")  # compat MSVC
        message(DEBUG "Detected OpenCV includes: ${OPENCV_INCLUDE_DIRS}")
        message(DEBUG "OPENCV VERSION MAJOR: ${OpenCV_VERSION_MAJOR}")
        list(APPEND INCLUDE_DIRS  "${OPENCV_INCLUDE_DIRS}")
        list(APPEND EXTERNAL_LIBS "${OPENCV_LIBRARIES}")
        list(APPEND EXTERNAL_DLLS "${OPENCV_DLLS}")
    endif()
endif()

# installation
include(GNUInstallDirs)
install(TARGETS ${PROJECT_TARGET}
		EXPORT ${PROJECT_TARGET}Config
		PUBLIC_HEADER DESTINATION include/${PROJECT_TARGET}
		INCLUDES DESTINATION include/${PROJECT_TARGET}
		ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
		LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
#install(DIRECTORY "${PROJECT_INCLUDE_DIR}" DESTINATION "include/${PROJECT_TARGET}")

if(${PROJECT_LIBRARY_TYPE} STREQUAL "SHARED")
    set(${PROJECT_TARGET}_SHARED_LIBRARIES "${PROJECT_TARGET}")
else()
    set(${PROJECT_TARGET}_STATIC_LIBRARIES "${PROJECT_TARGET}")
endif()
# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if(MSVC)
	add_custom_command(TARGET ${PROJECT_TARGET}
					   POST_BUILD
					   COMMAND ${CMAKE_COMMAND} -E copy_if_different
					   ${EXTERNAL_DLLS}
					   $<TARGET_FILE_DIR:${PROJECT_TARGET}>)
	install(FILES ${EXTERNAL_DLLS} DESTINATION bin)
    set(${PROJECT_TARGET}_SHARED_LIBRARIES ${EXTERNAL_DLLS})
endif(MSVC)
message(DEBUG "${PROJECT_TARGET}_STATIC_LIBRARIES: ${${PROJECT_TARGET}_STATIC_LIBRARIES}")
message(DEBUG "${PROJECT_TARGET}_SHARED_LIBRARIES: ${${PROJECT_TARGET}_SHARED_LIBRARIES}")

export(PACKAGE ${PROJECT_TARGET})
