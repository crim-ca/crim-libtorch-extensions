
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
option(PROJECT_WITH_CUDA "Enable CUDA support" ON)
if(${PROJECT_WITH_CUDA})
	add_definitions(-DWITH_CUDA)
endif()
