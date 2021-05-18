# Source: https://github.com/orfeotoolbox/OTB/blob/develop/CMake/FindOpenCV.cmake
# Modified by: Francis Charette-Migneault
#
# Copyright (C) 2005-2019 Centre National d'Etudes Spatiales (CNES)
#
# This file is part of Orfeo Toolbox
#
#     https://www.orfeo-toolbox.org/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

include("${CMAKE_CURRENT_LIST_DIR}/utils.cmake")
find_arch()

message(DEBUG "Find OpenCV")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
# handle OpenCV_DIR/OPENCV_DIR variants
if(OPENCV_DIR AND NOT OpenCV_DIR)
	set(OpenCV_DIR "${OpenCV_DIR}")
endif()
set(OpenCV_DIR "${OpenCV_DIR}" CACHE PATH "OpenCV install location")

set(OPENCV_SEARCH_PATH)

if(OpenCV_DIR)
    get_filename_component(OPENCV_SEARCH_PATH "${OpenCV_DIR}" REALPATH)
    if(OPENCV_SEARCH_PATH)
        get_filename_component(OPENCV_SEARCH_PATH "${OPENCV_SEARCH_PATH}" REALPATH)
    endif()
else()
    set(OpenCV_DIR "OpenCV installation directory" CACHE PATH)
endif()

# our project requires MSVC_VERSION >= 1900 (ie: VS2015)
# they all use the same toolset group (v14x) (ie: VS2015:v140, VS2017:v141, VS2019:v142)
# suppose only sub-directory 'vc15' as a possible match
# handle other cases here as required if you need it (and it works somehow...)
if(MSVC)
	set(OPENCV_INSTALL_SUFFIX_DIR "${ARCH_DIR}/vc15")
else()
	set(OPENCV_INSTALL_SUFFIX_DIR "")
endif(MSVC)

if(NOT EXISTS ${OPENCV_SEARCH_PATH})
    find_path(
        OPENCV_SEARCH_PATH
        include/opencv2/opencv.hpp
        PATHS
            ${OpenCV_DIR}
            "${CONFIG_THIRD_PARTY_DIR}/opencv"
            "${CONFIG_THIRD_PARTY_DIR}/opencv2"
            "${CONFIG_THIRD_PARTY_DIR}/opencv3"
            "${CONFIG_THIRD_PARTY_DIR}/opencv4"
            "${CONFIG_THIRD_PARTY_DIR}/cv"
            "${CONFIG_THIRD_PARTY_DIR}/cv2"
        #no additional paths are added to the search if OpenCV_DIR
        NO_DEFAULT_PATH
        PATH_SUFFIXES
            "install"
        DOC "The directory where opencv is installed"
    )
endif()
message(DEBUG "OpenCV: tmp location: ${OPENCV_SEARCH_PATH}")
message(DEBUG "OpenCV: sub dir: ${OPENCV_INSTALL_SUFFIX_DIR}")

# validate path with version header
find_path(
    opencv_INCLUDE_DIR
    opencv2/core/version.hpp
    PATHS
        ${OpenCV_DIR}/include
        ${OPENCV_SEARCH_PATH}/include
        ${OpenCV_DIR}/${OPENCV_INSTALL_SUFFIX_DIR}/include
        ${OPENCV_SEARCH_PATH}/${OPENCV_INSTALL_SUFFIX_DIR}/include
    PATH_SUFFIXES
        "opencv4"
    DOC "The directory where opencv headers are installed"
)
if(NOT EXISTS ${OPENCV_SEARCH_PATH})
  get_filename_component(OPENCV_SEARCH_PATH "${opencv_INCLUDE_DIR}" PATH)
  # include dir is include/opencv4 in v4 UNIX
  if(UNIX AND OpenCV_VERSION_MAJOR EQUAL 4)
    get_filename_component(OPENCV_SEARCH_PATH "${OPENCV_SEARCH_PATH}" PATH)
  endif()
endif()
message(STATUS "OpenCV: dir: ${OpenCV_DIR}")
message(STATUS "OpenCV: using location: ${OPENCV_SEARCH_PATH}")
mark_as_advanced(OPENCV_SEARCH_PATH)
mark_as_advanced(opencv_INCLUDE_DIR)

if(NOT EXISTS ${opencv_INCLUDE_DIR})
    message(ERROR "Could not find OpenCV includes")
else()
  set(OPENCV_INCLUDE_DIRS "${opencv_INCLUDE_DIR}")

  if(NOT OpenCV_VERSION)
	message(DEBUG "OpenCV include dir: ${opencv_INCLUDE_DIR}")
    file(READ "${opencv_INCLUDE_DIR}/opencv2/core/version.hpp" _header_content)

    # detect the type of version file (2.3.x , 2.4.x, 3.x or 4.x)
    string(REGEX MATCH  ".*# *define +CV_VERSION_EPOCH +([0-9]+).*" has_epoch ${_header_content})
    string(REGEX MATCH  ".*# *define +CV_MAJOR_VERSION +([0-9]+).*" has_old_major ${_header_content})
    string(REGEX MATCH  ".*# *define +CV_MINOR_VERSION +([0-9]+).*" has_old_minor ${_header_content})
    string(REGEX MATCH  ".*# *define +CV_SUBMINOR_VERSION +([0-9]+).*" has_old_subminor ${_header_content})

    if(has_old_major AND has_old_minor AND has_old_subminor)
      #for opencv 2.3.x
      string(REGEX REPLACE ".*# *define +CV_MAJOR_VERSION +([0-9]+).*" "\\1"
        OpenCV_VERSION_MAJOR ${_header_content})
      string(REGEX REPLACE ".*# *define +CV_MINOR_VERSION +([0-9]+).*" "\\1"
        OpenCV_VERSION_MINOR ${_header_content})
      string(REGEX REPLACE ".*# *define +CV_SUBMINOR_VERSION +([0-9]+).*" "\\1"
        OpenCV_VERSION_PATCH ${_header_content})
      set(OpenCV_VERSION_TWEAK)
    elseif(has_epoch)
      # for opencv 2.4.x
      string(REGEX REPLACE ".*# *define +CV_VERSION_EPOCH +([0-9]+).*" "\\1"
        OpenCV_VERSION_MAJOR ${_header_content})
      string(REGEX REPLACE ".*# *define +CV_VERSION_MAJOR +([0-9]+).*" "\\1"
        OpenCV_VERSION_MINOR ${_header_content})
      string(REGEX REPLACE ".*# *define +CV_VERSION_MINOR +([0-9]+).*" "\\1"
        OpenCV_VERSION_PATCH ${_header_content})
      string(REGEX REPLACE ".*# *define +CV_VERSION_REVISION +([0-9]+).*" "\\1"
        OpenCV_VERSION_TWEAK ${_header_content})
    else()
      string(REGEX REPLACE ".*# *define +CV_VERSION_MAJOR +([0-9]+).*" "\\1"
        OpenCV_VERSION_MAJOR ${_header_content})
      string(REGEX REPLACE ".*# *define +CV_VERSION_MINOR +([0-9]+).*" "\\1"
        OpenCV_VERSION_MINOR ${_header_content})
      string(REGEX REPLACE ".*# *define +CV_VERSION_REVISION +([0-9]+).*" "\\1"
        OpenCV_VERSION_PATCH ${_header_content})
      set(OpenCV_VERSION_TWEAK)
    endif()
    set(OpenCV_VERSION
      "${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.${OpenCV_VERSION_PATCH}")
  endif()

  if(WIN32)
    set(opencv_lib_name_suffix "${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}")
  endif()
endif()

# helper macro to find various opencv lib by name
set(OPENCV_base_lib_parts "core;calib3d;dnn;features2d;flann;highgui;imgcodecs;imgproc;ml;objdetect;photo;shape;superres;stitching;videoio;videostab")
set(OPENCV_cuda_lib_parts "cudev;cudaoptflow;cudaobjdetect;cudalegacy;cudaimgproc;cudafeatures2d;cudastereo;cudacodec;cudafilters;cudawarping;cudabgsegm;cudaarithm")
set(OPENCV_static_suffixes
	"${CMAKE_STATIC_LIBRARY_PREFIX}"
	"${CMAKE_STATIC_LIBRARY_PREFIX}${ARCH_POSTFIX}"
	"${CMAKE_STATIC_LIBRARY_PREFIX}/${ARCH_DIR}"
	"${ARCH_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}"
	"${CMAKE_STATIC_LIBRARY_PREFIX}${OPENCV_INSTALL_SUFFIX_DIR}/lib"
	"${OPENCV_INSTALL_SUFFIX_DIR}/lib"
)
list(APPEND CMAKE_MESSAGE_INDENT "  ")
foreach(suffix ${OPENCV_static_suffixes})
	message(DEBUG "Search in: ${OPENCV_SEARCH_PATH}/${suffix}")
endforeach()
list(POP_BACK CMAKE_MESSAGE_INDENT)
set(OPENCV_LIBRARIES "")
set(OPENCV_DLLS "")
foreach(lib_part_name ${OPENCV_base_lib_parts} ${OPENCV_cuda_lib_parts})
	message(DEBUG "OpenCV - Search library: opencv_${lib_part_name}")
	find_library(
	  OPENCV_${lib_part_name}_LIBRARY
	  NAMES "opencv_${lib_part_name}" "opencv_${lib_part_name}${opencv_lib_name_suffix}"
			"opencv_${lib_part_name}d" "opencv_${lib_part_name}${opencv_lib_name_suffix}d"
	  PATHS ${OPENCV_SEARCH_PATH}
	  PATH_SUFFIXES ${OPENCV_static_suffixes}
	  NO_DEFAULT_PATH
	  DOC "Path to opencv_${lib_part_name} library")
    # append lib if found
	if(EXISTS "${OPENCV_${lib_part_name}_LIBRARY}")
		list(APPEND OPENCV_LIBRARIES "${OPENCV_${lib_part_name}_LIBRARY}")
		# retrieve dlls
		if(MSVC AND OPENCV_DLLS STREQUAL "")
			message(DEBUG "LOOKING INTO: ${OPENCV_${lib_part_name}_LIBRARY}")
			get_filename_component(OPENCV_DLL_DIR "${OPENCV_${lib_part_name}_LIBRARY}" DIRECTORY)
			get_filename_component(OPENCV_DLL_DIR "${OPENCV_DLL_DIR}" DIRECTORY)
			set(OPENCV_DLL_DIR "${OPENCV_DLL_DIR}/bin")
			message(DEBUG "LOOKING WITH: ${OPENCV_DLL_DIR}")
			file(GLOB OPENCV_DLLS "${OPENCV_DLL_DIR}/*.dll")
		endif()
	endif()
endforeach(lib_part_name)

# results
message(DEBUG "OpenCV LIBS:")
message_items("${OPENCV_LIBRARIES}")
message(DEBUG "OpenCV DLLS:")
message_items("${OPENCV_DLLS}")

set(OpenCV_FOUND FALSE)
if( OPENCV_INCLUDE_DIRS AND NOT OPENCV_LIBRARIES STREQUAL "")
  set(OpenCV_FOUND TRUE)
  set(OPENCV_VERSION ${OpenCV_VERSION}) #for compatility
endif()

# provide more standard variable names
set(OPENCV_STATIC_LIBRARIES ${OPENCV_LIBRARIES})
set(OPENCV_SHARED_LIBRARIES ${OPENCV_DLLS})
mark_as_advanced(OPENCV_STATIC_LIBRARIES)
mark_as_advanced(OPENCV_SHARED_LIBRARIES)

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
if(MSVC)
	find_package_handle_standard_args(OpenCV
	  REQUIRED_VARS
	  OPENCV_LIBRARIES
	  OPENCV_DLLS
	  OPENCV_INCLUDE_DIRS
	  VERSION_VAR OpenCV_VERSION)
else()
	find_package_handle_standard_args(OpenCV
	  REQUIRED_VARS
	  OPENCV_LIBRARIES
	  opencv_INCLUDE_DIR
	  VERSION_VAR OpenCV_VERSION)
endif()

list(POP_BACK CMAKE_MESSAGE_INDENT)
