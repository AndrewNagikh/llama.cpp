#pragma once

#include "architecture/architecture_plugin.h"

class gemma_architecture_plugin : public architecture_plugin {
public:
    std::string family() const override;
    bool matches(const model_manifest & manifest) const override;
    architecture_descriptor build_descriptor(const model_manifest & manifest) const override;
};
