cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)
set(PROJECT_NAME "EfficientNetCpp")
project(${PROJECT_NAME})

include(../cmake/configs.cmake)

# sources
set(${PROJECT_NAME}_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(${PROJECT_NAME}_SOURCES_DIR "${PROJECT_SOURCE_DIR}/source")
file(GLOB_RECURSE HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${${PROJECT_NAME}_INCLUDE_DIR}/*.h")
file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${${PROJECT_NAME}_SOURCES_DIR}/*.cpp")
add_library(${PROJECT_NAME} STATIC ${HEADERS} ${SOURCES})
if(MSVC)
    source_group(TREE "${${PROJECT_NAME}_INCLUDE_DIR}" PREFIX "Headers" FILES ${HEADERS})
    source_group(TREE "${${PROJECT_NAME}_SOURCES_DIR}" PREFIX "Sources" FILES ${SOURCES})
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
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")
target_include_directories(${PROJECT_NAME} PUBLIC "${INCLUDE_DIRS}" "${${PROJECT_NAME}_INCLUDE_DIR}")
target_link_libraries(${PROJECT_NAME} PUBLIC Core "${EXTERNAL_LIBS}")
if(${CONFIG_WITH_CORE_PRECOMPILED_HEADER_COMMON})
    # FIXME: this should work according to doc, but causes header to be deleted by the target just before it starts compiling
    #   results into header being 'not-found' and failing whole compile
    #   still seem to do the right thing automatically becasue of added dependency to 'Core' target...
    #   doc:
    #   https://cmake.org/cmake/help/latest/command/target_precompile_headers.html?highlight=precompiled%20header#reusing-precompile-headers
	#target_precompile_headers(${PROJECT_NAME} REUSE_FROM Core)
endif()

# installation
install(TARGETS ${PROJECT_NAME}
		EXPORT ${PROJECT_NAME}-targets
		PUBLIC_HEADER DESTINATION include/${PROJECT_NAME}
		INCLUDES DESTINATION include/${PROJECT_NAME}
		LIBRARY DESTINATION lib
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
