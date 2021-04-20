cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)

# solution name
set(CONFIG_SOLUTION_NAME "EfficientNet" CACHE STRING "Name of the generated solution" FORCE)
if("${CONFIG_SOLUTION_NAME}" STREQUAL "")
	message(FATAL_ERROR "Solution name cannot be empty")
endif()
#mark_as_advanced(CONFIG_SOLUTION_NAME)
project("${CONFIG_SOLUTION_NAME}" LANGUAGES CXX)

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

# sources
set(${PROJECT_NAME}_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(${PROJECT_NAME}_SOURCES_DIR "${PROJECT_SOURCE_DIR}/source")
file(GLOB_RECURSE PROJECT_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${${PROJECT_NAME}_INCLUDE_DIR}/*.h")
file(GLOB_RECURSE PROJECT_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${${PROJECT_NAME}_SOURCES_DIR}/*.cpp")
add_library(${PROJECT_NAME} SHARED ${PROJECT_HEADERS} ${PROJECT_SOURCES})
if(MSVC)
    source_group(TREE "${${PROJECT_NAME}_INCLUDE_DIR}" PREFIX "Headers" FILES ${PROJECT_HEADERS})
    source_group(TREE "${${PROJECT_NAME}_SOURCES_DIR}" PREFIX "Sources" FILES ${PROJECT_SOURCES})
endif()

# torch libs
set(EXTERNAL_LIBS "")
set(EXTERNAL_DEPS "")
set(EXTERNAL_DLLS "")
if(NOT TORCH_FOUND)
	find_package(Torch REQUIRED)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
list(APPEND INCLUDE_DIRS "${TORCH_INCLUDE_DIR}")
list(APPEND EXTERNAL_LIBS "${TORCH_LIBRARIES}")
list(APPEND EXTERNAL_DLLS "${TORCH_DLLS}")
list(APPEND EXTERNAL_DEPS "torch")

# library
add_dependencies(${PROJECT_NAME} ${EXTERNAL_DEPS})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
set_property(TARGET ${PROJECT_NAME} PROPERTY OUTPUT_NAME "${PROJECT_NAME}")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${PROJECT_HEADERS}")
message(DEBUG "Include dirs:" ${INCLUDE_DIRS})
message(DEBUG "Include libs:" ${LIBRARIES})
message(DEBUG "External libs:" ${EXTERNAL_LIBS})
target_include_directories(${PROJECT_NAME} PUBLIC "${INCLUDE_DIRS}" "${${PROJECT_NAME}_INCLUDE_DIR}")
target_link_libraries(${PROJECT_NAME} PUBLIC "${EXTERNAL_LIBS}")
if(${CONFIG_WITH_CORE_PRECOMPILED_HEADER_COMMON})
    # FIXME: this should work according to doc, but causes header to be deleted by the target just before it starts compiling
    #   results into header being 'not-found' and failing whole compile
    #   still seem to do the right thing automatically becasue of added dependency to 'Core' target...
    #   doc:
    #   https://cmake.org/cmake/help/latest/command/target_precompile_headers.html?highlight=precompiled%20header#reusing-precompile-headers
	#target_precompile_headers(${PROJECT_NAME} REUSE_FROM Core)
endif()

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

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
set(${PROJECT_NAME}_SHARED_LIBRARIES "")
set(${PROJECT_NAME}_STATIC_LIBRARIES "${PROJECT_NAME}")
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
