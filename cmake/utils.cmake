
# Finds the current architecture of the system
# Defines following variables for the calling scope: 
# 	- ARCH_DIR
#	- ARCH_POSTFIX
function(find_arch)
	if (CMAKE_SIZEOF_VOID_P EQUAL 4)
		message(DEBUG "Architecture: x86")
		set(ARCH_POSTFIX "" PARENT_SCOPE)
		set(ARCH_DIR "x86" PARENT_SCOPE)
	else()
		message(DEBUG "Architecture: x64")
		set(ARCH_POSTFIX 64 PARENT_SCOPE)
		set(ARCH_DIR "x64" PARENT_SCOPE)
		add_definitions(-DWIN64)
	endif()
endfunction(find_arch)
