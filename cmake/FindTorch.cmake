# FindTorch
# -------
#
# Finds the Torch library
#
# This will define the following variables:
#
#   TORCH_FOUND					- True if the system has the Torch library
#	TORCH_ROOT					- Root directory of found library
#   TORCH_INCLUDE_DIRECTORIES   - The include directories for torch
#   TORCH_LIBRARIES 			- Static libraries to link against
#	TORCH_DLLS					- Dynamic libraries for runtime execution
#
# and the following imported targets::
#
#     Torch
#
# To locate installation path of Torch, following variables are employed
#
#	TORCH_INSTALL_PREFIX
#	TORCH_DIR
#	Torch_DIR
#
# First match (in that order) is used.
#

cmake_minimum_required(VERSION 3.6.0 FATAL_ERROR)

# handle Torch_DIR/TORCH_DIR variants
if(EXISTS Torch_DIR AND NOT TORCH_DIR)
	set(TORCH_DIR "${Torch_DIR}")
endif()
if (NOT EXISTS "${TORCH_DIR}" AND EXISTS "${PROJECT_THIRD_PARTY_DIR}")
    set(TORCH_DIR "${PROJECT_THIRD_PARTY_DIR}")
endif()

# attempt to find some reference directory
#   using cmake config
find_file(TORCH_CMAKE_CFG TorchConfig.cmake
	PATHS
		"${TORCH_INSTALL_PREFIX}"
		"${TORCH_DIR}"
		"${TORCH_DIR}/share/cmake/Torch"
		"${TORCH_DIR}/cmake/Torch"
		"${TORCH_DIR}/Torch"
		"${TORCH_ROOT}"
		"${TORCH_ROOT}/share/cmake/Torch"
		"${TORCH_ROOT}/cmake/Torch"
		"${TORCH_ROOT}/Torch"
		"${PROJECT_THIRD_PARTY_DIR}/share/cmake/Torch"
		"${PROJECT_THIRD_PARTY_DIR}/cmake/Torch"
		"${PROJECT_THIRD_PARTY_DIR}/Torch"
)
if(EXISTS "${TORCH_CMAKE_CFG}")
    if(NOT TARGET Torch)
        include("${TORCH_CMAKE_CFG}")
    endif()
    message(DEBUG "TORCH LIB?? ${TORCH_LIBRARY}")
    list(APPEND TORCH_LIBRARIES ${TORCH_LIBRARY})
#   using source directory
elseif(NOT EXISTS TORCH_DIR)
	find_path(TORCH_DIR
		NAMES
			torch/csrc/api/include/torch/torch.h
			include/torch/csrc/api/include/torch/torch.h
		PATHS
			"${PROJECT_SOURCE_DIR}/3rdparty/pytorch"
			"${TORCH_ROOT}"
			NO_DEFAULT_PATH
	)
	if(EXISTS "${TORCH_DIR}" AND NOT EXISTS "${TORCH_CMAKE_CFG}")
		find_file(TORCH_CMAKE_CFG TorchConfig.cmake
			PATHS "${TORCH_DIR}/share/cmake/Torch"
		)
	endif()
endif()

# parse version from config if available
if(EXISTS "${TORCH_CMAKE_CFG}")
	get_filename_component(TORCH_CMAKE_DIR "${TORCH_CMAKE_CFG}" DIRECTORY)
	set(TORCH_CMAKE_VERSION "${TORCH_CMAKE_DIR}/TorchConfigVersion.cmake")
endif()
if(EXISTS "${TORCH_CMAKE_VERSION}" AND NOT TORCH_VERSION)
	file(READ "${TORCH_CMAKE_VERSION}" _cfg_content)
	string(REGEX MATCH ".*PACKAGE_VERSION \"+([0-9]+.[0-9]+.[0-9]+).*" TORCH_VERSION ${_cfg_content})
	string(REGEX REPLACE ".*PACKAGE_VERSION \"+([0-9]+.[0-9]+.[0-9]+).*" "\\1" TORCH_VERSION ${_cfg_content})
	set(Torch_VERSION ${TORCH_VERSION})
endif()

# find include/library directories
find_path(TORCH_INCLUDE_DIR torch/csrc/api/include/torch/torch.h
	PATHS
		"${TORCH_DIR}/include"
		"${TORCH_ROOT}/include"
		"${TORCH_DIR}/torch/include"
		"${TORCH_ROOT}/torch/include"
		NO_DEFAULT_PATH
)
find_path(TORCH_LIBDIR torch.lib
	PATHS
		"${TORCH_DIR}/lib"
		"${TORCH_DIR}/torch/lib"
		"${TORCH_DIR}/lib/tmp_install"
		"${TORCH_DIR}/torch/lib/tmp_install"
)
find_library(TORCH_LIBRARY torch
	PATHS
		"${TORCH_LIBDIR}"
		NO_DEFAULT_PATH
)
find_library(CAFFE2_LIB caffe2
	PATHS
		"${TORCH_LIBDIR}"
		NO_DEFAULT_PATH
)
find_library(C10_LIB c10
	PATHS
		"${TORCH_LIBDIR}"
		NO_DEFAULT_PATH
)
set(TORCH_INCLUDE_DIRECTORIES
	${TORCH_INCLUDE_DIR}
	"${TORCH_INCLUDE_DIR}/torch/csrc/api/include"
	"${TORCH_INCLUDE_DIR}/ATen"
	"${TORCH_INCLUDE_DIR}/c10"
)
mark_as_advanced(TORCH_INCLUDE_DIRECTORIES)

### Optionally, link CUDA
option(USE_CUDA "Use CUDA enabled Torch." ON)
if(USE_CUDA)
	if(NOT CUDA_FOUND)
		find_package(CUDA)
	endif()
endif()
if(CUDA_FOUND)
	add_definitions(-DCUDA_FOUND)
	find_library(CAFFE2_CUDA_LIB caffe2_gpu
		PATHS "${TORCH_LIBDIR}"
		NO_DEFAULT_PATH
	)
	find_library(C10_CUDA_LIB c10_cuda
		PATHS "${TORCH_LIBDIR}"
		NO_DEFAULT_PATH
	)
	list(APPEND TORCH_INCLUDE_DIRECTORIES ${CUDA_TOOLKIT_INCLUDE})
	if(MSVC)
	list(APPEND CAFFE2_LIB
		${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib
		${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart.lib
		${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nvrtc.lib)
	else(MSVC)
	list(APPEND CAFFE2_LIB -L"${CUDA_TOOLKIT_ROOT_DIR}/lib64" cuda cudart nvrtc nvToolsExt)
	endif(MSVC)
	list(APPEND TORCH_INCLUDE_DIRECTORIES "${TORCH_DIR}/aten/src/THC")
	list(APPEND TORCH_INCLUDE_DIRECTORIES "${TORCH_DIR}/torch/lib/tmp_install/include/THC")
endif(CUDA_FOUND)

add_definitions(-DNO_PYTHON)
if(NOT TARGET Torch)
    set(TORCH_LIBRARY_TYPE "SHARED" CACHE STRING "Torch library type to link against.")
    set_property(CACHE TORCH_LIBRARY_TYPE PROPERTY STRINGS SHARED STATIC)
    add_library(Torch ${TORCH_LIBRARY_TYPE} IMPORTED)
endif()
if(MSVC)
	set_target_properties(Torch PROPERTIES IMPORTED_IMPLIB "${TORCH_LIBRARY}")
else(MSVC)
	set_target_properties(Torch PROPERTIES IMPORTED_LOCATION "${TORCH_LIBRARY}")
endif(MSVC)
set_target_properties(Torch PROPERTIES
	INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRECTORIES}"
	INTERFACE_LINK_LIBRARIES "${CAFFE2_LIB};${CAFFE2_CUDA_LIB};${C10_LIB};${C10_CUDA_LIB}"
)

set(TORCH_DLLS "")
mark_as_advanced(TORCH_DLLS)
if (MSVC)
	file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
endif()

list(FILTER TORCH_LIBRARIES EXCLUDE REGEX ".*NOTFOUND")
list(FILTER TORCH_DLLS EXCLUDE REGEX ".*NOTFOUND")

if(EXISTS ${TORCH_INCLUDE_DIR})
	get_filename_component(TORCH_ROOT ${TORCH_INCLUDE_DIR} DIRECTORY)
endif()

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
if (NOT EXISTS ${TORCH_LIBRARY})
    message(WARNING "Could not find Torch library. Define TORCH_DIR or TORCH_INSTALL_PREFIX with install location.")
endif()
find_package_handle_standard_args(Torch
    REQUIRED_VARS
		TORCH_ROOT
        TORCH_LIBRARY
        TORCH_INCLUDE_DIR
    VERSION_VAR TORCH_VERSION
)
message(DEBUG "Torch root: ${TORCH_ROOT}")
message(DEBUG "Torch libs: ${TORCH_LIBRARIES}")
message(DEBUG "Torch dlls: ${TORCH_DLLS}")
