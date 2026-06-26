#pragma once

#include "model_registry.h"

#include <string>

// Persist cluster registry (layouts, manifests, coverage) under models_dir/.orchestrator/
bool registry_persistence_load(const std::string & models_dir, cluster_model_registry & registry);
bool registry_persistence_save(const std::string & models_dir, const cluster_model_registry & registry);
