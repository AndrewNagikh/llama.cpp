#include "architecture_plugin.h"

#include "architecture/blob_builder.h"
#include "architecture/plugins/llama_plugin.h"
#include "architecture/plugins/qwen_plugin.h"
#include "architecture/plugins/gemma_plugin.h"

#include <vector>

namespace {

const std::vector<const architecture_plugin *> all_plugins() {
    static const llama_architecture_plugin  llama_plugin;
    static const qwen_architecture_plugin   qwen_plugin;
    static const gemma_architecture_plugin  gemma_plugin;
    static const std::vector<const architecture_plugin *> plugins = {
        &qwen_plugin,
        &gemma_plugin,
        &llama_plugin,
    };
    return plugins;
}

} // namespace

const architecture_plugin & select_architecture_plugin(const model_manifest & manifest) {
    for (const architecture_plugin * plugin : all_plugins()) {
        if (plugin->matches(manifest)) {
            return *plugin;
        }
    }
    return *all_plugins().back();
}