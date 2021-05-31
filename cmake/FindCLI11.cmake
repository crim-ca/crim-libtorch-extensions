# FindCLI11
# -------
#
# Finds the CLI11 installation
# (https://github.com/CLIUtils/CLI11)
#
# This will define the following variables:
#
#   CLI11_FOUND         - True if the package was found
#   CLI11_DIR           - Location of CLI11 top directory
#   CLI11_INCLUDE_DIR   - Location of headers for CLI11
#   CLI11_TARGETS       - Targets to be included by other targets using CLI11
#
# To help find preinstalled location, define:
#
#   CLI11_ROOT = <path/to/install/location>
#

message(DEBUG "Find CLI11")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
message(DEBUG "CLI11_ROOT: ${CLI11_ROOT}")
message(DEBUG "CLI11_DIR:  ${CLI11_DIR}")
find_path(CLI11_ROOT
  NAMES include/CLI/CLI.hpp
  HINTS "${CLI11_DIR}" "${CLI11_ROOT}"
)

find_path(CLI11_INCLUDE_DIR
  NAMES CLI/CLI.hpp
  HINTS ${CLI11_ROOT}/include
)
message(DEBUG "Detected CLI11 include: ${CLI11_INCLUDE_DIR}")

if(EXISTS "${CLI11_INCLUDE_DIR}/CLI/CLI.hpp")
  set(CLI11_TARGETS CLI11::CLI11)
  add_library(CLI11::CLI11 INTERFACE IMPORTED)
  set_target_properties(CLI11::CLI11 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CLI11_INCLUDE_DIR}")
  set(CLI11_DIR "${CLI11_INCLUDE_DIR}")
endif()

# make variable visible if not defined
mark_as_advanced(
  CLEAR
    CLI11_ROOT
)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  CLI11
  FOUND_VAR
    CLI11_FOUND
  REQUIRED_VARS
    CLI11_DIR
    CLI11_INCLUDE_DIR
    CLI11_TARGETS
)
mark_as_advanced(
  CLI11_DIR
  CLI11_INCLUDE_DIR
  CLI11_TARGETS
)
list(POP_BACK CMAKE_MESSAGE_INDENT)
