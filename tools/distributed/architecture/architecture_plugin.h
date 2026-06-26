#pragma once

#include "architecture_descriptor.h"

class architecture_plugin {
public:
    virtual ~architecture_plugin() = default;

    virtual std::string family() const = 0;

    virtual bool matches(const model_manifest & manifest) const = 0;

    virtual architecture_descriptor build_descriptor(const model_manifest & manifest) const = 0;
};

const architecture_plugin & select_architecture_plugin(const model_manifest & manifest);
