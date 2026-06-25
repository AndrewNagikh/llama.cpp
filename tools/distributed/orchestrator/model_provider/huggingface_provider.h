#pragma once

#include <memory>
#include "model_provider.h"

std::unique_ptr<model_provider> create_huggingface_provider();
