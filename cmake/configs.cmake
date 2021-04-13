
# avoid CUDA_ROOT, CUDNN_ROOT, etc. warning messages
set(CMAKE_POLICY_DEFAULT_CMP0074 OLD)
cmake_policy(SET CMP0074 OLD)

# make utility cmake directory available
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_LIST_DIR}")

# generic target properties
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS On)

# add cuda definition WITH_CUDA which is common across CMake packages
# (note: corresponding libraries must also have been compiled with CUDA support)
option(CONFIG_WITH_CUDA "Enable cuda support" ON)
if(${CONFIG_WITH_CUDA})
	add_definitions(-DWITH_CUDA)
endif()
