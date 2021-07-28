# FindPLog
# ---------
#
# Finds the plog installation
# (https://github.com/SergiusTheBest/plog)
#
# This will define the following variables:
#
#   PLOG_FOUND         - True if the package was found
#   PLOG_DIR           - Location of PLOG top directory
#   PLOG_INCLUDE_DIR   - Location of headers for PLOG
#
# To help find preinstalled location, define:
#
#   PLOG_ROOT = <path/to/install/location>
#

message(DEBUG "Find PLOG")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
message(DEBUG "PLOG_ROOT: ${PLOG_ROOT}")
message(DEBUG "PLOG_DIR:  ${PLOG_DIR}")
find_path(PLOG_ROOT
  NAMES include/plog/Log.h
  HINTS "${PLOG_DIR}" "${PLOG_ROOT}"
)

find_path(PLOG_INCLUDE_DIR
  NAMES plog/Log.h
  HINTS ${PLOG_ROOT}/include
)
message(DEBUG "Detected PLOG include: ${PLOG_INCLUDE_DIR}")

if(EXISTS "${PLOG_INCLUDE_DIR}/plog/Log.h")
  set(PLOG_DIR "${PLOG_INCLUDE_DIR}")
endif()

# make variable visible if not defined
mark_as_advanced(
  CLEAR
    PLOG_ROOT
)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  PLOG
  FOUND_VAR
    PLOG_FOUND
  REQUIRED_VARS
    PLOG_DIR
    PLOG_INCLUDE_DIR
)
mark_as_advanced(
  PLOG_DIR
  PLOG_INCLUDE_DIR
)
list(POP_BACK CMAKE_MESSAGE_INDENT)
