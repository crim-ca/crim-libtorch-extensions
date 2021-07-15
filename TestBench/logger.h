#pragma once

#ifdef USE_LOG_COUT

    #define LOGGER(level) std::cout

#else

    #define PLOG_OMIT_LOG_DEFINES   // avoid conflict with torch/caffe loggers
    #include <plog/Log.h>
    #include "plog/Initializers/RollingFileInitializer.h"
    #include <plog/Formatters/TxtFormatter.h>
    #include <plog/Appenders/ConsoleAppender.h>
    #include <plog/Appenders/ColorConsoleAppender.h>
    #include <plog/Appenders/RollingFileAppender.h>

    #include "formatter.h"

    #define VERBOSE plog::verbose
    #define DEBUG plog::debug
    #define INFO plog::info
    #define WARN plog::warning
    #define ERROR plog::error
    #define FATAL plog::fatal
    #define LOGGER(level) PLOG(level)

#endif
