cmake_minimum_required(VERSION 3.3.0 FATAL_ERROR)

# solution name
set(PROJECT_SOLUTION_NAME "CRIM-Torch-Extensions" CACHE STRING "Name of the generated solution" FORCE)
if("${PROJECT_SOLUTION_NAME}" STREQUAL "")
	message(FATAL_ERROR "Solution name cannot be empty")
endif()
# generate the solution with given name (MSVC)
# this will set PROJECT_NAME
project(${PROJECT_SOLUTION_NAME} LANGUAGES CXX)

set(PROJECT_TARGET crim_torch_extensions)
set(PROJECT_LIBRARY ${PROJECT_TARGET})

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

find_arch()
list(APPEND PROJECT_DEFINES -DUNICODE)
if(NOT ${ARCH_POSTFIX} STREQUAL "")
	list(APPEND PROJECT_DEFINES -DAMD64)	# x64
endif()

# options
option(WITH_EXTERNAL_MODULE "Look for external module sources (must define its own cmake files)" OFF)
set(PROJECT_EXTERNAL_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external" CACHE STRING "External module location")
if(${WITH_EXTERNAL_MODULE})
	if(EXISTS "${PROJECT_EXTERNAL_MODULE_DIR}")
		add_subdirectory("${PROJECT_EXTERNAL_MODULE_DIR}")
	endif()
endif()
option(WITH_DATA_AUGMENTATION "Include data augmentation operations (Requires OpenCV)" ON)
option(WITH_TEST_BENCH "Build TestBench CLI utility." ON)
option(WITH_TESTS "Build tests modules" OFF)
option(WITH_PYTHON "Enable Python support. Must match compilation of Torch library." OFF)
message(DEBUG "Options:")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
message(DEBUG "WITH External Module ... ${WITH_EXTERNAL_MODULE}")
message(DEBUG "WITH DataAugmentation .. ${WITH_DATA_AUGMENTATION}")
message(DEBUG "WITH Python ............ ${WITH_PYTHON}")
message(DEBUG "WITH tests ............. ${WITH_TESTS}")
message(DEBUG "WITH TestBench ......... ${WITH_TEST_BENCH}")
list(POP_BACK CMAKE_MESSAGE_INDENT)

# sources
set(PROJECT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(PROJECT_SOURCES_DIR "${PROJECT_SOURCE_DIR}/source")

file(GLOB_RECURSE PROJECT_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${PROJECT_INCLUDE_DIR}/*.h")
file(GLOB_RECURSE PROJECT_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${PROJECT_SOURCES_DIR}/*.cpp")

if(NOT WIN32)
    file(GLOB_RECURSE EXCLUDE_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${PROJECT_INCLUDE_DIR}/windows/*.h")
    file(GLOB_RECURSE EXCLUDE_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${PROJECT_SOURCES_DIR}/windows/*.cpp")
    foreach(file ${EXCLUDE_HEADERS})
        list(REMOVE_ITEM PROJECT_HEADERS ${file})
    endforeach()
    foreach(file ${EXCLUDE_SOURCES})
        list(REMOVE_ITEM PROJECT_SOURCES ${file})
    endforeach()
endif()

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
endif()

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
if(NOT TORCHVISION_FOUND)
    # https://github.com/pytorch/vision
    find_package(TorchVision REQUIRED)
endif()
list(APPEND INCLUDE_DIRS  "${TORCHVISION_INCLUDE_DIRS}")
list(APPEND EXTERNAL_LIBS "${TORCHVISION_TARGETS}")
list(APPEND EXTERNAL_DLLS "${TORCHVISION_DLLS}")

# depending on Python bindinds or not, create library or module
if(${WITH_PYTHON})
    if(${WITH_TESTS} OR ${WITH_TEST_BENCH})
        message(FATAL "Cannot enable simultaneous Tests/TestBench and Python modules!")
    endif()

    # find Python and pybind required to generate bindings
    find_package(Python3 COMPONENTS Development Interpreter HINTS ${PYTHON_EXECUTABLE} ${Python3_ROOT_DIR})
    list(APPEND INCLUDE_DIRS ${Python_INCLUDE_DIRS})
    list(APPEND EXTERNAL_LIBS ${Python_LIBRARIES})

    if(NOT "${pybind_DIR}" STREQUAL "")
        if("${PYBIND11_DIR}" STREQUAL "")
            set(PYBIND11_DIR "${pybind_DIR}")
        endif()
        mark_as_advanced(FORCE pybind11_DIR)
        mark_as_advanced(CLEAR PYBIND11_DIR)
    endif()
    if(NOT EXISTS ${PYBIND11_DIR})
        message(FATAL "Must set PYBIND11_DIR")
    endif()
    # add_subdirectory(${PYBIND11_DIR} ${CMAKE_BINARY_DIR}/pybind11 EXCLUDE_FROM_ALL)
    find_package(pybind11 COMPONENTS HINTS ${PYBIND11_DIR})

    message(DEBUG "Python Headers: ${Python_INCLUDE_DIRS}")
    message(DEBUG "PyBind Headers: ${pybind11_INCLUDE_DIRS}")

    # let pybind define all the required targets and libs references
    # it setups additional items that will no correctly install the python package if otherwise using 'add_library'
    pybind11_add_module(${PROJECT_TARGET}
        ${TORCH_INCLUDE_DIR}
        ${pybind11_INCLUDE_DIRS}
        ${PROJECT_HEADERS}
        ${PROJECT_SOURCES}
    )
    # target_include_directories(${PROJECT_TARGET} ${Python_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS})

    # patch missing definitions between pybind and pytorch
    # (https://github.com/pytorch/pytorch/issues/38122)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
    list(APPEND EXTERNAL_LIBS ${TORCH_PYTHON_LIBRARY})

    remove_definitions(-DNO_PYTHON)
    # no bindings added if compiled directly for use in C++ only
else()
    # patch over globbing of python bindings
    message(DEBUG "Disabled WITH_PYTHON - Will remove all items in 'python' directories.")
    file(GLOB_RECURSE EXCLUDE_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${PROJECT_INCLUDE_DIR}/python/*.h")
    file(GLOB_RECURSE EXCLUDE_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${PROJECT_SOURCES_DIR}/python/*.cpp")
    foreach(file ${EXCLUDE_HEADERS})
        list(REMOVE_ITEM PROJECT_HEADERS ${file})
    endforeach()
    foreach(file ${EXCLUDE_SOURCES})
        list(REMOVE_ITEM PROJECT_SOURCES ${file})
    endforeach()
    add_library(${PROJECT_LIBRARY} ${PROJECT_LIBRARY_TYPE} ${PROJECT_HEADERS} ${PROJECT_SOURCES})
endif()

# library
add_dependencies(${PROJECT_TARGET} ${EXTERNAL_DEPS})
set_property(TARGET ${PROJECT_TARGET} PROPERTY CXX_STANDARD 14)
set_property(TARGET ${PROJECT_TARGET} PROPERTY OUTPUT_NAME "${PROJECT_TARGET}")
set_target_properties(${PROJECT_TARGET} PROPERTIES PUBLIC_HEADER "${PROJECT_HEADERS}")
target_include_directories(${PROJECT_TARGET} PUBLIC ${INCLUDE_DIRS} ${PROJECT_INCLUDE_DIR})
target_link_libraries(${PROJECT_TARGET} PUBLIC ${EXTERNAL_LIBS})

# must enforce if data-aug added by itself, regardless of if TestBench is added or not
if(WITH_DATA_AUGMENTATION)
    if(NOT OPENCV_FOUND)
        # https://github.com/opencv/opencv
        # (better if Torch/TorchVision also compiled using it for optimizations)
        find_package(OpenCV REQUIRED)
    endif()
    set(OPENCV_INCLUDE_DIRS "${OPENCV_INCLUDE_DIR}")  # compat MSVC
    message(DEBUG "Detected OpenCV includes: ${OPENCV_INCLUDE_DIRS}")
    message(DEBUG "OPENCV VERSION MAJOR: ${OpenCV_VERSION_MAJOR}")
    list(APPEND INCLUDE_DIRS  "${OPENCV_INCLUDE_DIRS}")
    list(APPEND EXTERNAL_LIBS "${OPENCV_LIBRARIES}")
    list(APPEND EXTERNAL_DLLS "${OPENCV_DLLS}")
    target_include_directories(${PROJECT_TARGET} PUBLIC ${INCLUDE_DIRS} ${PROJECT_INCLUDE_DIR})
    target_link_libraries(${PROJECT_TARGET} PUBLIC ${EXTERNAL_LIBS})
endif()

option(WITH_PRECOMPILED_HEADER "Use precompiled stdafx headers" OFF)
option(MSVC_Z7_OVERRIDE "Enforce MSVC /Z7 CXX flag when using precompiled headers to fix sccache bug (recommended)" ON)
if(${WITH_PRECOMPILED_HEADER})
	message(STATUS "${PROJECT_TARGET} package using precompiled headers (see: WITH_PRECOMPILED_HEADER)")
	target_precompile_headers(${PROJECT_TARGET} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../stdafx.h")
    target_compile_definitions(${PROJECT_TARGET} PUBLIC PRECOMPILE_TORCH)
    target_include_directories(${PROJECT_TARGET} PRIVATE ${TORCH_INCLUDE_DIRECTORIES})
	if(MSVC AND MSVC_Z7_OVERRIDE)
		# enforce the flag to avoid incompatibility with other targets that will use Core
		# by default cmake will specify /Z7 to debug targets not adding (just using) precompiled headers
		string(REGEX REPLACE "/Z[iI7]" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
		set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Z7")
	endif()
endif()

message(DEBUG "===========================================")
message_pkg_refs("${PROJECT_TARGET}" "${PROJECT_DEFINES}" "${PROJECT_HEADERS}" "${PROJECT_SOURCES}"
                 "${EXTERNAL_DEPS}" "${EXTERNAL_LIBS}" "${EXTERNAL_DLLS}")
message(DEBUG "===========================================")

# tests
if(WITH_TESTS)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")
    message(DEBUG "===========================================")
endif()

# utilities
if(WITH_TEST_BENCH)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/TestBench)
endif(WITH_TEST_BENCH)

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
    set(PROJECT_SHARED_LIBRARIES "${PROJECT_TARGET}")
else()
    set(PROJECT_STATIC_LIBRARIES "${PROJECT_TARGET}")
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
    set(PROJECT_SHARED_LIBRARIES ${EXTERNAL_DLLS})
    add_definitions(-Dcte_EXPORTS)
endif(MSVC)
message(DEBUG "PROJECT_STATIC_LIBRARIES: ${PROJECT_STATIC_LIBRARIES}")
message(DEBUG "PROJECT_SHARED_LIBRARIES: ${PROJECT_SHARED_LIBRARIES}")

export(PACKAGE ${PROJECT_TARGET})
