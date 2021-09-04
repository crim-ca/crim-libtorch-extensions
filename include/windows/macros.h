#pragma once

#ifdef _WIN32
#if defined(cte_EXPORTS)
#define CTE_API __declspec(dllexport)
#else
#define CTE_API __declspec(dllimport)
#endif
#else
#define CTE_API
#endif
