# FindTorchVision
# -------
#
# Finds the TorchVision library (requires Torch)
#
# This will define the following variables:
#
#  	TORCHVISION_FOUND  			- True if the system has the Torch library
#  	TORCHVISION_INCLUDE_DIRS	- The include directories for torch
#  	TORCHVISION_LIBRARIES 		- Static libraries to link against
#	TORCHVISION_DLLS			- Dynamic libraries for runtime execution
# 	TORCHVISION_TARGETS			- TorchVision::TorchVision defined by TorchVision's cmake

include(${CMAKE_CURRENT_LIST_DIR}/utils.cmake)
find_arch()

# handle Torch_FOUND/TORCH_FOUND variants
if(NOT Torch_FOUND AND NOT TORCH_FOUND)
	find_package(Torch REQUIRED)
endif()

# handle TorchVision_DIR/TORCHVISION_DIR variants
if(TorchVision_DIR AND NOT TORCHVISION_DIR)
	set(TORCHVISION_DIR "${TorchVision_DIR}")
endif()
mark_as_advanced(TORCHVISION_DIR)
set(TorchVision_DIR "${TorchVision_DIR}" CACHE PATH "TorchVision location")
message(STATUS "TorchVision: using location: ${TORCHVISION_DIR}")

set(TORCHVISION_TARGETS "TorchVision::TorchVision")
mark_as_advanced(TORCHVISION_TARGETS)

# attempt to find some reference directory
#   using cmake config 
find_file(TORCHVISION_CMAKE_CFG TorchVisionConfig.cmake
	PATHS 
		"${TORCHVISION_DIR}"
		"${TORCHVISION_DIR}/share/cmake/TorchVision"
		"${TORCHVISION_DIR}/cmake/TorchVision"
		"${TORCHVISION_DIR}/TorchVision"
		"${TORCHVISION_DIR}/vision"
		"${TORCHVISION_ROOT}"
		"${TORCHVISION_ROOT}/share/cmake/TorchVision"
		"${TORCHVISION_ROOT}/cmake/TorchVision"
		"${TORCHVISION_ROOT}/TorchVision"
		"${TORCHVISION_ROOT}/vision"
		"${CONFIG_THIRD_PARTY_DIR}/TorchVision/share/cmake/TorchVision"
		"${CONFIG_THIRD_PARTY_DIR}/TorchVision/cmake/TorchVision"
        "${CONFIG_THIRD_PARTY_DIR}/TorchVision"
        "${CONFIG_THIRD_PARTY_DIR}/vision"
)
if(EXISTS "${TORCHVISION_CMAKE_CFG}")
	include("${TORCHVISION_CMAKE_CFG}")
else()
	find_path(TORCHVISION_DIR 
		NAMES
			vision.h
		PATH_SUFFIXES
			# installed
			torchvision
			include/torchvision
			# source
			torchvision/csrc
			include/torchvision/csrc
		PATHS
			"${TORCHVISION_ROOT}"
            "${CONFIG_THIRD_PARTY_DIR}/TorchVision"
			NO_DEFAULT_PATH
	)
endif()
if(EXISTS "${TORCHVISION_DIR}" AND NOT EXISTS "${TORCHVISION_CMAKE_CFG}")
	find_file(TORCHVISION_CMAKE_CFG TorchVisionConfig.cmake
		PATHS "${TORCHVISION_DIR}/share/cmake/TorchVision"
	)
endif()

# parse version from config if available
# if(EXISTS "${TORCHVISION_CMAKE_CFG}")
	# get_filename_component(TORCHVISION_CMAKE_DIR "${TORCHVISION_CMAKE_CFG}" DIRECTORY)
	# set(TORCHVISION_CMAKE_VERSION "${TORCHVISION_CMAKE_DIR}/TorchConfigVersion.cmake")
# endif()
# if(EXISTS "${TORCHVISION_CMAKE_VERSION}" AND NOT TORCHVISION_VERSION)
	# file(READ "${TORCHVISION_CMAKE_VERSION}" _cfg_content)
	# string(REGEX MATCH ".*PACKAGE_VERSION \"+([0-9]+.[0-9]+.[0-9]+).*" TORCHVISION_VERSION ${_cfg_content})
	# string(REGEX REPLACE ".*PACKAGE_VERSION \"+([0-9]+.[0-9]+.[0-9]+).*" "\\1" TORCHVISION_VERSION ${_cfg_content})
	# set(TORCHVISION_VERSION ${TORCHVISION_VERSION})
# endif()
# find include/library directories
find_path(TORCHVISION_INCLUDE_DIR
	NAMES 
		torchvision/vision.h
		torchvision/csrc/vision.h 
	PATHS
		"${TORCHVISION_DIR}/include"
		"${TORCHVISION_ROOT}/include"
		"${TORCHVISION_DIR}/torchvision/include"
		"${TORCHVISION_ROOT}/torchvision/include"
		NO_DEFAULT_PATH
)
find_path(TORCHVISION_LIBDIR torchvision.lib
	PATHS				
		"${TORCHVISION_DIR}/lib"
		"${TORCHVISION_DIR}/torchvision/lib"
)

find_library(TORCHVISION_LIBRARY torchvision
	PATHS 
		"${TORCHVISION_LIBDIR}"
		NO_DEFAULT_PATH
)

# HACK: 
#	Since the include dir could be defined as the source, some downloaded zip or cmake installed path, 
#	plug all existing variants such that including a relative file will work regardless (eg: #include "models/resnet.h")
# 	TorchVision_INCLUDE_DIR could be any of the following variants
set(TORCHVISION_INCLUDE_DIRS "${TorchVision_INCLUDE_DIR}")
foreach(sub "torchvision" "torchvision/csrc")
	if (EXISTS "${TorchVision_INCLUDE_DIR}/${sub}")
		list(APPEND TORCHVISION_INCLUDE_DIRS "${TorchVision_INCLUDE_DIR}/${sub}")
	endif()
endforeach()
mark_as_advanced(TORCHVISION_INCLUDE_DIRS)

# runtime libraries
include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
if(MSVC)
	find_file(TORCHVISION_DLLS
		NAMES "torchvision.dll"
		PATHS 
			"${TORCHVISION_DIR}/bin"
			"${TORCHVISION_DIR}/lib"
			"${TORCHVISION_DIR}/torchvision/bin"
			"${TORCHVISION_DIR}/torchvision/lib"
		PATH_SUFFIXES
			${ARCH_DIR}
			${ARCH_DIR}/$<CONFIG>
			$<CONFIG>
	)
else()
    set(TORCHVISION_DLLS "")
endif()
mark_as_advanced(TORCHVISION_DLLS)

set(TORCHVISION_LIBRARIES ${TORCHVISION_LIBRARY})
set(TORCHVISION_STATIC_LIBRARIES ${TORCHVISION_LIBRARY})
set(TORCHVISION_SHARED_LIBRARIES ${TORCHVISION_DLLS})
mark_as_advanced(TORCHVISION_LIBRARIES)
mark_as_advanced(TORCHVISION_STATIC_LIBRARIES)
mark_as_advanced(TORCHVISION_SHARED_LIBRARIES)

find_package_handle_standard_args(TorchVision
    REQUIRED_VARS 
        TORCHVISION_LIBRARY 
        TorchVision_INCLUDE_DIR
        TORCHVISION_TARGETS
)
