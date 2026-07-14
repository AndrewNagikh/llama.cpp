#include "node_log_export.h"

#include <fstream>
#include <sstream>

std::string node_log_resolve_path(const std::string & models_dir, const std::string & filename) {
    if (models_dir.empty() || filename.empty()) {
        return {};
    }
    return models_dir + "/logs/" + filename;
}

std::string node_log_tail(const std::string & path, const size_t max_lines, const size_t max_bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }

    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    const std::streamoff start = (max_bytes > 0 && (std::streamoff) max_bytes < size)
            ? size - (std::streamoff) max_bytes
            : 0;
    in.seekg(start, std::ios::beg);

    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string buf = ss.str();

    if (max_lines == 0 || buf.empty()) {
        return buf;
    }

    // A lone trailing '\n' terminates the last line rather than separating
    // it from a (nonexistent) following one, so it must not be counted as
    // a line boundary -- otherwise every tail is off by one line short.
    size_t end = buf.size();
    if (buf[end - 1] == '\n') {
        --end;
    }

    size_t pos = end;
    size_t lines = 0;
    while (pos > 0 && lines < max_lines) {
        --pos;
        if (buf[pos] == '\n') {
            ++lines;
        }
    }
    const size_t tail_start = (lines == max_lines) ? pos + 1 : 0;
    return buf.substr(tail_start);
}
