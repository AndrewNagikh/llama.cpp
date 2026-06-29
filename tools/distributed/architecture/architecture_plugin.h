#pragma once

#include "architecture/architecture_descriptor.h"
#include "architecture/semantic_runtime_descriptor.h"
#include "runtime/architecture_descriptor/distributed_runtime_descriptor.h"

#include "orchestrator/manifest_builder/manifest_builder.h"

#include <string>

// Task 9.9 — Architecture Plugin API (registry + default implementations).

class architecture_plugin {
public:
    virtual ~architecture_plugin() = default;

    virtual std::string family() const = 0;

    virtual bool matches(const model_manifest & manifest) const = 0;

    virtual architecture_descriptor build_descriptor(const model_manifest & manifest) const = 0;

    virtual semantic_runtime_descriptor build_runtime_descriptor(
            const model_manifest & manifest) const;

    virtual distributed_runtime_descriptor build_distributed_descriptor(
            const model_manifest & manifest) const;

    virtual bool verify_runtime(
            const distributed_runtime_descriptor & desc,
            std::string & error) const;
};

const architecture_plugin & select_architecture_plugin(const model_manifest & manifest);
