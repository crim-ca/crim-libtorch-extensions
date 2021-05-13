
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


function(message_items file_list  # prefix
)
    set(extra_macro_args ${ARGN})
    list(LENGTH extra_macro_args num_extra_args)
    if (${num_extra_args} GREATER 0)
        list(GET extra_macro_args 0 _prefix)
    else()
        set(_prefix "Found")
    endif()

    list(APPEND CMAKE_MESSAGE_INDENT "  ")
        foreach(file ${file_list})
            message(DEBUG "${_prefix} ${file}")
        endforeach()
    list(POP_BACK CMAKE_MESSAGE_INDENT)
endfunction(message_items)
