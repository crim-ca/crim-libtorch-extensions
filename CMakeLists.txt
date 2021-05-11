cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)

# solution name
set(PROJECT_SOLUTION_NAME "CRIM-PyTorch-Extensions" CACHE STRING "Name of the generated solution" FORCE)
if("${PROJECT_SOLUTION_NAME}" STREQUAL "")
	message(FATAL_ERROR "Solution name cannot be empty")
endif()
#mark_as_advanced(PROJECT_SOLUTION_NAME)
project(crim_libtorch_extensions LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake" FORCE)
include(cmake/configs.cmake)
include(cmake/utils.cmake)

# build configuration type
#   https://stackoverflow.com/questions/31546278/where-to-set-cmake-configuration-types-in-a-project-with-subprojects
if(NOT SET_UP_CONFIGURATIONS_DONE)
    set(SET_UP_CONFIGURATIONS_DONE 1 CACHE STRING "")
    mark_as_advanced(SET_UP_CONFIGURATIONS_DONE)

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
option(WITH_TESTS "Build tests module(s)" OFF)
#mark_as_advanced(WITH_TESTS)

# sources
set(${PROJECT_NAME}_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(${PROJECT_NAME}_SOURCES_DIR "${PROJECT_SOURCE_DIR}/source")
file(GLOB_RECURSE PROJECT_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${${PROJECT_NAME}_INCLUDE_DIR}/*.h")
file(GLOB_RECURSE PROJECT_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${${PROJECT_NAME}_SOURCES_DIR}/*.cpp")
message(DEBUG "Headers files for project ${PROJECT_NAME}")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
    foreach(file ${PROJECT_HEADERS})
        message(DEBUG "Found ${file}")
    endforeach()
list(POP_BACK CMAKE_MESSAGE_INDENT)
message(DEBUG "Source files for project ${PROJECT_NAME}")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
    foreach(file ${PROJECT_SOURCES})
        message(DEBUG "Found ${file}")
    endforeach()
list(POP_BACK CMAKE_MESSAGE_INDENT)
set(PROJECT_LIBRARY_TYPE "SHARED" CACHE STRING "Library type to build.")
set_property(CACHE PROJECT_LIBRARY_TYPE PROPERTY STRINGS SHARED STATIC)
if(MSVC)
    source_group(TREE "${${PROJECT_NAME}_INCLUDE_DIR}" PREFIX "Headers" FILES ${PROJECT_HEADERS})
    source_group(TREE "${${PROJECT_NAME}_SOURCES_DIR}" PREFIX "Sources" FILES ${PROJECT_SOURCES})
endif()

# torch libs
set(EXTERNAL_LIBS "")
set(EXTERNAL_DEPS "")
set(EXTERNAL_DLLS "")
option(PROJECT_FORCE_THREADS "Force finding and adding Threads library.")
if(PROJECT_FORCE_THREAD)
    find_library(Threads REQUIRED)
    list(APPEND EXTERNAL_LIBS Threads)
endif()
if(NOT TORCH_FOUND)
	find_package(Torch REQUIRED)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
list(APPEND INCLUDE_DIRS "${TORCH_INCLUDE_DIR}")
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
    pybind11_add_module(${PROJECT_NAME} ${TORCH_INCLUDE_DIR} ${pybind11_INCLUDE_DIRS} ${PROJECT_HEADERS} ${PROJECT_SOURCES})
    # target_include_directories(${PROJECT_NAME} ${Python_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS})

    # patch missing definitions between pybind and pytorch
    # (https://github.com/pytorch/pytorch/issues/38122)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
    list(APPEND EXTERNAL_LIBS ${TORCH_PYTHON_LIBRARY})

    remove_definitions(-DNO_PYTHON)

else()
    # no bindings added if compiled directly for use in C++ only
    add_library(${PROJECT_NAME} ${PROJECT_LIBRARY_TYPE} ${PROJECT_HEADERS} ${PROJECT_SOURCES})
endif()

# library
add_dependencies(${PROJECT_NAME} ${EXTERNAL_DEPS})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
set_property(TARGET ${PROJECT_NAME} PROPERTY OUTPUT_NAME "${PROJECT_NAME}")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${PROJECT_HEADERS}")
message(DEBUG "Include dirs:  " "${INCLUDE_DIRS}")
message(DEBUG "Include libs:  " "${LIBRARIES}")
message(DEBUG "External libs: " "${EXTERNAL_LIBS}")
target_include_directories(${PROJECT_NAME} PUBLIC "${INCLUDE_DIRS}" "${${PROJECT_NAME}_INCLUDE_DIR}")
target_link_libraries(${PROJECT_NAME} PUBLIC "${EXTERNAL_LIBS}")


option(WITH_TEST_BENCH "Enable TestBench creation." ON)
message(DEBUG "WITH TestBench? ${WITH_TEST_BENCH}")
if(WITH_TEST_BENCH)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/TestBench)
endif(WITH_TEST_BENCH)

# installation
include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME}
		EXPORT ${PROJECT_NAME}Config
		PUBLIC_HEADER DESTINATION include/${PROJECT_NAME}
		INCLUDES DESTINATION include/${PROJECT_NAME}
		ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
		LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
#install(DIRECTORY "${${PROJECT_NAME}_INCLUDE_DIR}" DESTINATION "include/${PROJECT_NAME}")

if(${PROJECT_LIBRARY_TYPE} STREQUAL "SHARED")
    set(${PROJECT_NAME}_SHARED_LIBRARIES "${PROJECT_NAME}")
else()
    set(${PROJECT_NAME}_STATIC_LIBRARIES "${PROJECT_NAME}")
endif()
# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if(MSVC)
	add_custom_command(TARGET ${PROJECT_NAME}
					   POST_BUILD
					   COMMAND ${CMAKE_COMMAND} -E copy_if_different
					   ${EXTERNAL_DLLS}
					   $<TARGET_FILE_DIR:${PROJECT_NAME}>)
	install(FILES ${EXTERNAL_DLLS} DESTINATION bin)
    set(${PROJECT_NAME}_SHARED_LIBRARIES ${EXTERNAL_DLLS})
endif(MSVC)
message(DEBUG "${PROJECT_NAME}_STATIC_LIBRARIES: ${${PROJECT_NAME}_STATIC_LIBRARIES}")
message(DEBUG "${PROJECT_NAME}_SHARED_LIBRARIES: ${${PROJECT_NAME}_SHARED_LIBRARIES}")

export(PACKAGE ${PROJECT_NAME})
