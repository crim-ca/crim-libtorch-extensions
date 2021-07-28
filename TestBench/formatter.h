#pragma once

#ifndef USE_LOG_COUT

#include <plog/Record.h>
#include <plog/Util.h>
#include <iomanip>

namespace plog
{
    template <bool showFunction, bool showLineNumber>
    class VerboseFormatterImpl
    {
    public:
        static util::nstring header()
        {
            return util::nstring();
        }

        static util::nstring format(const Record& record)
        {
            tm t;
            util::localtime_s(&t, &record.getTime().time);

            util::nostringstream ss;
            ss << t.tm_year + 1900 << "-" << std::setfill(PLOG_NSTR('0')) << std::setw(2)
               << t.tm_mon + 1 << PLOG_NSTR("-") << std::setfill(PLOG_NSTR('0'))
               << std::setw(2) << t.tm_mday << PLOG_NSTR(" ");
            ss << std::setfill(PLOG_NSTR('0')) << std::setw(2) << t.tm_hour << PLOG_NSTR(":")
               << std::setfill(PLOG_NSTR('0')) << std::setw(2) << t.tm_min << PLOG_NSTR(":")
               << std::setfill(PLOG_NSTR('0')) << std::setw(2) << t.tm_sec << PLOG_NSTR(".")
               << std::setfill(PLOG_NSTR('0')) << std::setw(3) << static_cast<int> (record.getTime().millitm)
               << PLOG_NSTR(" ");
            ss << std::setfill(PLOG_NSTR(' ')) << std::setw(5) << std::left
               << severityToString(record.getSeverity());
            if (showFunction)
                ss << PLOG_NSTR(" [") << record.getTid() << PLOG_NSTR("]");
            if (showLineNumber)
                ss << PLOG_NSTR(" [") << record.getFunc() << PLOG_NSTR("@") << record.getLine() << PLOG_NSTR("]");
            ss << PLOG_NSTR(" ") << record.getMessage();// << PLOG_NSTR("\n");

            return ss.str();
        }
    };

    class VerboseFormatter : public VerboseFormatterImpl<true, true> {};
    class MinimalFormatter : public VerboseFormatterImpl<false, false> {};
}

#endif // ! USE_LOG_COUT
