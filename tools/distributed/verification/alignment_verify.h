#pragma once

#include "verification_types.h"

#include <string>

struct alignment_verify_result {
    verify_check_result summary;
    std::vector<std::string> issues;
};

alignment_verify_result alignment_verify_file(const std::string & path);
