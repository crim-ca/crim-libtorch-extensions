
if(POLICY CMP0074)
    set(CMAKE_POLICY_DEFAULT_CMP0074 NEW)
    cmake_policy(SET CMP0074 NEW)
endif()

# make utility cmake directory available
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_LIST_DIR}")

# generic target properties
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS On)

# add cuda definition WITH_CUDA which is common across CMake packages
# (note: corresponding libraries must also have been compiled with CUDA support)
option(WITH_CUDA "Enable CUDA support" ON)
if(${WITH_CUDA})
	add_definitions(-DWITH_CUDA)
endif()

# OpenCV uses the WITH_ notation, while Torch/TorchVision employs the USE_ notation
# mask the USE_ that are matched by a corresponding WITH, and override the value as needed
get_cmake_property(VARIABLE_NAMES VARIABLES)
foreach(var ${VARIABLE_NAMES})
    if("${var}" MATCHES "^WITH_.+")
        string(REPLACE "WITH_" "USE_" use_var ${var})
        if(${use_var} IN_LIST VARIABLE_NAMES)
            message(DEBUG "${use_var} -> ${var} = ${${var}}")
            set(${use_var} ${${var}} CACHE FORCE)
            mark_as_advanced(${use_var})
        endif()
    endif()
endforeach()
