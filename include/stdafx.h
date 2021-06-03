#ifndef __STDAFX_H
#define __STDAFX_H
#pragma once

#ifdef _WIN32

    #ifndef NOMINMAX
    #define NOMINMAX // inhibit min and max defines
    #endif/*NOMINMAX*/

    #ifndef
    #define TYPENAME typename // used in some cases where compilers disagree on 'typename' usage
    #endif

    #if defined(UNICODE) && !defined(_UNICODE)
    #define _UNICODE
    #endif
    #if defined(_UNICODE) && !defined(UNICODE)
    #define UNICODE
    #endif

    #ifndef _CRT_SECURE_FORCE_DEPRECATE
    #ifndef _CRT_SECURE_NO_DEPRECATE
    #define _CRT_SECURE_NO_DEPRECATE // eliminate deprecation warnings
    #endif
    #ifndef _CRT_SECURE_NO_WARNINGS
    #define _CRT_SECURE_NO_WARNINGS // eliminate deprecation warnings
    #endif
    #endif

    #define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS

    #define WIN32_LEAN_AND_MEAN     // Exclude rarely-used stuff from Windows headers
    #include <windows.h>

#endif /* _WIN32 */

// torch headers
#ifdef PRECOMPILE_TORCH
#include <torch/torch.h>
#endif

#ifndef NO_PYTHON
#include <torch/extension.h>
#include <torch/csrc/utils/pybind.h>
#endif  // NO_PYTHON

#endif  // __STDAFX_H
