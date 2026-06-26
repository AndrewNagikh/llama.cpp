#pragma once

#include "nlohmann/json.hpp"

#include <string>

nlohmann::json gguf_inspect_file(const std::string & path);
std::string    gguf_inspect_file_json(const std::string & path);
