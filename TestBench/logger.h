#pragma once

#ifdef USE_LOG_COUT

    #include <iostream>
    #include <mutex>

    // https://stackoverflow.com/a/45046349/5936364
    // Async Console Output
    struct acout
    {
        std::unique_lock<std::mutex> lk;
        acout() : lk(std::unique_lock<std::mutex>(mtx_cout)) {}

        template<typename T>
        acout& operator<<(const T& t)
        {
            std::cout << t;
            return *this;
        }

        acout& operator<<(std::ostream& (*fp)(std::ostream&))
        {
            std::cout << fp;
            return *this;
        }
    };

    #define LOGGER(level) acout

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
