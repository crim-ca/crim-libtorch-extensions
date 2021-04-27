#pragma once

#ifdef _WIN32

#define NOMINMAX // inhibit min and max defines

#ifndef _CRT_SECURE_FORCE_DEPRECATE
#ifndef _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_DEPRECATE // eliminate deprecation warnings
#endif
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS // eliminate deprecation warnings
#endif
#endif

#define WIN32_LEAN_AND_MEAN     // Exclude rarely-used stuff from Windows headers
#include <windows.h>

#endif /* _WIN32 */

// torch headers
#include <torch/torch.h>

#ifndef NO_PYTHON
#include <torch/extension.h>
#include <torch/csrc/utils/pybind.h>
#endif
