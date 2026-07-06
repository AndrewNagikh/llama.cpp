#pragma once

#include "llama.h"

#include <string>

// Task 11.7 — load worker model from GGUF file or Layer Store (DIST_RUNTIME_LAYER_FIRST).

llama_model * split_gen_load_model(const char * model_path, std::string & err);

bool split_gen_model_uses_layer_store();
