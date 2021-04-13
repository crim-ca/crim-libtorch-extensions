cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)

# solution name
set(CONFIG_SOLUTION_NAME "EfficientNet" CACHE STRING "Name of the generated solution" FORCE)
if("${CONFIG_SOLUTION_NAME}" STREQUAL "")
	message(FATAL_ERROR "Solution name cannot be empty")
endif()
#mark_as_advanced(CONFIG_SOLUTION_NAME)
project("${CONFIG_SOLUTION_NAME}" LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
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
option(CONFIG_WITH_EXTERNAL_MODULE "Look for external module sources (must define its own cmake files)" OFF)
set(CONFIG_EXTERNAL_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external" CACHE STRING "External module location")
#mark_as_advanced(CONFIG_WITH_EXTERNAL_MODULE)
if(${CONFIG_WITH_EXTERNAL_MODULE})
	if(EXISTS "${CONFIG_EXTERNAL_MODULE_DIR}")
		add_subdirectory("${CONFIG_EXTERNAL_MODULE_DIR}")
	endif()
endif()

# tests
option(CONFIG_WITH_TESTS "Build tests module(s)" OFF)
#mark_as_advanced(CONFIG_WITH_TESTS)

# source
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/source")

