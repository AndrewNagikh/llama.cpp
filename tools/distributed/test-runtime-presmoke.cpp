#include "architecture/semantic_runtime_descriptor.h"
#include "node_agent/layer_store/layer_store.h"
#include "runtime/runtime_compat_materialize.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

struct presmoke_gate {
    const char * id;
    const char * binary;
    const char * evidence;
    bool requires_model_artifacts;
};

struct presmoke_artifacts {
    std::string reference_runtime;
    std::string reference_source;
    std::string reference_materialization_status;
    std::string reference_materialization_reason;
    std::string worker_final_gguf;
    std::string entry_trace;
    std::string middle_trace;
    std::string final_trace;
    std::string hidden_bin;
    std::string runtime_graph_path;
    std::string prompt;
};

struct gate_command {
    std::string id;
    std::string command;
    std::vector<std::pair<std::string, std::string>> inputs;
    std::string evidence_path;
};

struct gate_result {
    std::string id;
    std::string status;
    std::string reason;
};

struct pipeline_stage_info {
    int stage_index = 0;
    int layer_start = 0;
    int layer_end = 0;
    std::string node_id;
    std::string role;
};

struct pipeline_boundary {
    int index = 0;
    int boundary = 0;
    pipeline_stage_info producer;
    bool has_consumer = false;
    pipeline_stage_info consumer;
};

static const presmoke_gate REQUIRED_GATES[] = {
    { "descriptor_acceptance", "test-runtime-acceptance", "descriptor, graph, scheduler, install readiness", false },
    { "reference_runtime", "test-runtime-presmoke", "Reference Runtime selection and materialization validation", false },
    { "pipeline_boundaries", "test-runtime-presmoke", "descriptor/runtime graph derived hidden boundary checks", true },
    { "logits_parity", "verify_logits_pipeline", "prefill logits parity against monolithic runtime", true },
    { "sampler_runtime", "verify_final_runtime", "sampler/final runtime decode-step equivalence", true },
    { "final_logits", "verify_final_logits", "OutputHead logits from descriptor-defined hidden input", true },
    { "decode_graph", "verify_decode_graph", "prefill/decode graph equivalence", true },
    { "decode_loop_trace", "verify_decode_loop_parity", "trace-level decode loop parity", true },
};

static std::string env_value(const char * name, const std::string & fallback = {}) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return value;
}

static bool path_usable(const std::string & path) {
    if (path.empty()) {
        return false;
    }
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec) &&
            std::filesystem::file_size(path, ec) > 0;
}

static std::string shell_quote(const std::string & value) {
    std::string out = "'";
    for (const char ch : value) {
        if (ch == '\'') {
            out += "'\\''";
        } else {
            out += ch;
        }
    }
    out += "'";
    return out;
}

static std::string json_escape(const std::string & value) {
    std::string out;
    for (const char ch : value) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += ch;      break;
        }
    }
    return out;
}

static std::string binary_path(const char * argv0, const std::string & name) {
    const std::filesystem::path self = std::filesystem::absolute(argv0);
    return (self.parent_path() / name).string();
}

static bool contains_token(const std::string & name, const std::string & token) {
    return token.empty() || name.find(token) != std::string::npos;
}

static bool usable_artifact(
        const std::filesystem::path & path,
        const std::string & extension) {
    std::error_code ec;
    const uintmax_t size = std::filesystem::file_size(path, ec);
    if (ec || size == 0) {
        return false;
    }

    const std::string filename = path.filename().string();
    if (extension == ".gguf") {
        if (filename.find("vocab") != std::string::npos ||
                filename.find("tokenizer") != std::string::npos) {
            return false;
        }
        return size > 1024U * 1024U;
    }
    return true;
}

static std::vector<std::filesystem::path> search_roots() {
    std::vector<std::filesystem::path> roots;
    roots.push_back(std::filesystem::current_path());

    const std::string home = env_value("HOME");
    if (!home.empty()) {
        roots.push_back(std::filesystem::path(home) / ".distributed-llm");
        roots.push_back(std::filesystem::path(home) / "Documents" / "node-agent" / "logs");
        roots.push_back(std::filesystem::path(home) / "Documents" / "node-agent" / "llama.cpp" / "logs");
    }

    const std::string out_dir = env_value("DIST_PRESMOKE_OUT_DIR");
    if (!out_dir.empty()) {
        roots.push_back(out_dir);
    }

    return roots;
}

static std::string discover_file(
        const char * env_name,
        const std::string & extension,
        const std::string & preferred_token,
        const bool allow_any_match) {
    const std::string explicit_path = env_value(env_name);
    if (!explicit_path.empty()) {
        return explicit_path;
    }

    std::string fallback;
    for (const std::filesystem::path & root : search_roots()) {
        std::error_code ec;
        if (!std::filesystem::exists(root, ec)) {
            continue;
        }
        for (std::filesystem::recursive_directory_iterator it(root, ec), end; it != end && !ec; it.increment(ec)) {
            if (!it->is_regular_file(ec)) {
                continue;
            }
            const std::filesystem::path path = it->path();
            if (path.extension() != extension) {
                continue;
            }
            if (!usable_artifact(path, extension)) {
                continue;
            }
            const std::string filename = path.filename().string();
            if (contains_token(filename, preferred_token)) {
                return path.string();
            }
            if (allow_any_match && fallback.empty()) {
                fallback = path.string();
            }
        }
    }
    return fallback;
}

static std::string discover_full_reference_gguf() {
    return {};
}

static std::string infer_model_hint_from_path(const std::string & path) {
    if (path.empty()) {
        return {};
    }
    for (const std::filesystem::path & part : std::filesystem::path(path)) {
        const std::string value = part.string();
        if (value.find("llama") != std::string::npos ||
                value.find("qwen") != std::string::npos ||
                value.find("gemma") != std::string::npos ||
                value.find("phi") != std::string::npos ||
                value.find("smol") != std::string::npos ||
                value.find("deepseek") != std::string::npos) {
            return value;
        }
    }
    return {};
}

static std::string infer_model_hint(const presmoke_artifacts & artifacts) {
    std::string hint = infer_model_hint_from_path(artifacts.worker_final_gguf);
    if (!hint.empty()) {
        return hint;
    }
    hint = infer_model_hint_from_path(artifacts.entry_trace);
    if (!hint.empty()) {
        return hint;
    }
    return infer_model_hint_from_path(artifacts.final_trace);
}

static std::string discover_full_reference_gguf(const std::string & model_hint) {
    const std::string explicit_path = env_value("DIST_PRESMOKE_REFERENCE_RUNTIME");
    if (!explicit_path.empty()) {
        return explicit_path;
    }

    std::string fallback;
    for (const std::filesystem::path & root : search_roots()) {
        std::error_code ec;
        if (!std::filesystem::exists(root, ec)) {
            continue;
        }
        for (std::filesystem::recursive_directory_iterator it(root, ec), end; it != end && !ec; it.increment(ec)) {
            if (!it->is_regular_file(ec) || it->path().extension() != ".gguf") {
                continue;
            }
            if (!usable_artifact(it->path(), ".gguf")) {
                continue;
            }
            const std::string filename = it->path().filename().string();
            if (filename.find("worker") != std::string::npos ||
                    filename.find("entry") != std::string::npos ||
                    filename.find("middle") != std::string::npos ||
                    filename.find("final") != std::string::npos ||
                    filename.find("reference_runtime") != std::string::npos) {
                continue;
            }
            if (!model_hint.empty() && it->path().string().find(model_hint) == std::string::npos) {
                continue;
            }
            if (filename == "model.gguf") {
                return it->path().string();
            }
            if (fallback.empty()) {
                fallback = it->path().string();
            }
        }
    }
    return fallback;
}

static std::string discover_cached_reference_runtime() {
    const std::string explicit_path = env_value("DIST_PRESMOKE_CACHED_REFERENCE_RUNTIME");
    if (!explicit_path.empty()) {
        return explicit_path;
    }
    return discover_file("DIST_PRESMOKE_CACHED_REFERENCE_RUNTIME", ".gguf", "reference_runtime", false);
}

static std::vector<std::pair<std::filesystem::path, std::string>> discover_layer_store_candidates() {
    std::vector<std::pair<std::filesystem::path, std::string>> out;
    for (const std::filesystem::path & root : search_roots()) {
        std::error_code ec;
        if (!std::filesystem::exists(root, ec)) {
            continue;
        }
        for (std::filesystem::recursive_directory_iterator it(root, ec), end; it != end && !ec; it.increment(ec)) {
            if (!it->is_regular_file(ec) || it->path().filename() != "manifest.json") {
                continue;
            }
            const std::filesystem::path model_root = it->path().parent_path();
            const std::string model_id = model_root.filename().string();
            const std::filesystem::path store_root = model_root.parent_path();
            if (model_id.empty() || store_root.empty()) {
                continue;
            }
            out.push_back({ store_root, model_id });
        }
    }
    return out;
}

static void prioritize_model_hint(
        std::vector<std::pair<std::filesystem::path, std::string>> & candidates,
        const std::string & model_hint) {
    if (model_hint.empty()) {
        return;
    }
    std::stable_sort(
            candidates.begin(),
            candidates.end(),
            [&](const auto & lhs, const auto & rhs) {
                const bool l = lhs.second.find(model_hint) != std::string::npos;
                const bool r = rhs.second.find(model_hint) != std::string::npos;
                return l && !r;
            });
}

static bool materialize_reference_from_layer_store(
        const std::string & out_dir,
        const std::string & model_hint,
        std::string & path,
        std::string & reason) {
    const std::string explicit_root = env_value("DIST_PRESMOKE_LAYER_STORE_ROOT");
    const std::string explicit_model = env_value("DIST_PRESMOKE_MODEL_ID");
    std::vector<std::pair<std::filesystem::path, std::string>> candidates;
    if (!explicit_root.empty() && !explicit_model.empty()) {
        candidates.push_back({ explicit_root, explicit_model });
    }
    const std::vector<std::pair<std::filesystem::path, std::string>> discovered =
            discover_layer_store_candidates();
    candidates.insert(candidates.end(), discovered.begin(), discovered.end());
    prioritize_model_hint(candidates, model_hint);

    if (candidates.empty()) {
        reason = "no layer store manifest found";
        return false;
    }

    for (const auto & candidate : candidates) {
        layer_store store(candidate.first, candidate.second);
        const auto manifest = store.load_manifest();
        if (!manifest.has_value() || manifest->n_layer == 0) {
            reason = "manifest unavailable or empty for " + candidate.second;
            continue;
        }

        const semantic_runtime_descriptor rt = build_semantic_runtime_descriptor(*manifest);
        const std::string out_path =
                (std::filesystem::path(out_dir) / ("reference_runtime_" + candidate.second + ".gguf")).string();
        std::string err;
        if (!runtime_compat_materialize_worker_gguf(
                    store,
                    *manifest,
                    rt,
                    worker_role::full,
                    0,
                    static_cast<int32_t>(manifest->n_layer),
                    out_path,
                    err)) {
            reason = "materializer failed for " + candidate.second + ": " + err;
            continue;
        }
        if (!path_usable(out_path)) {
            reason = "materializer produced unusable reference for " + candidate.second;
            continue;
        }
        path = out_path;
        reason = "materialized from layer store model_id=" + candidate.second;
        return true;
    }

    if (reason.empty()) {
        reason = "no materializable layer store candidate";
    }
    return false;
}

static std::string downloaded_reference_runtime_reason() {
    const std::string url = env_value("DIST_PRESMOKE_REFERENCE_GGUF_URL");
    if (url.empty()) {
        return "download skipped: DIST_PRESMOKE_REFERENCE_GGUF_URL not set";
    }
    return "download skipped: automatic downloader not configured in pre-smoke runner";
}

static void resolve_reference_runtime(
        presmoke_artifacts & artifacts,
        const std::string & out_dir) {
    const std::string model_hint = infer_model_hint(artifacts);
    artifacts.reference_runtime = discover_full_reference_gguf(model_hint);
    if (!artifacts.reference_runtime.empty()) {
        artifacts.reference_source = "existing_full_gguf";
        artifacts.reference_materialization_status = "SKIPPED";
        artifacts.reference_materialization_reason = "existing full GGUF selected";
        return;
    }

    std::string materialized;
    std::string materialize_reason;
    if (materialize_reference_from_layer_store(out_dir, model_hint, materialized, materialize_reason)) {
        artifacts.reference_runtime = materialized;
        artifacts.reference_source = "materialized_from_layer_store";
        artifacts.reference_materialization_status = "PASS";
        artifacts.reference_materialization_reason = materialize_reason;
        return;
    }

    const std::string cached = discover_cached_reference_runtime();
    if (!cached.empty()) {
        artifacts.reference_runtime = cached;
        artifacts.reference_source = "cached_reference_runtime";
        artifacts.reference_materialization_status = "SKIPPED";
        artifacts.reference_materialization_reason =
                "materialization unavailable (" + materialize_reason + "), cached reference selected";
        return;
    }

    artifacts.reference_source = "unavailable";
    artifacts.reference_materialization_status = "SKIPPED";
    artifacts.reference_materialization_reason =
            materialize_reason + "; " + downloaded_reference_runtime_reason();
}

static bool find_runtime_graph_json(const nlohmann::json & input, nlohmann::json & graph) {
    if (input.is_object()) {
        if (input.contains("runtime_graph") && input["runtime_graph"].is_object()) {
            graph = input["runtime_graph"];
            return true;
        }
        if (input.contains("roles") && input["roles"].is_array() &&
                input.contains("model_id") && input.contains("n_layers")) {
            graph = input;
            return true;
        }
        for (auto it = input.begin(); it != input.end(); ++it) {
            if (find_runtime_graph_json(it.value(), graph)) {
                return true;
            }
        }
    } else if (input.is_array()) {
        for (const nlohmann::json & item : input) {
            if (find_runtime_graph_json(item, graph)) {
                return true;
            }
        }
    }
    return false;
}

static bool load_runtime_graph_json(const std::string & path, nlohmann::json & graph) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    nlohmann::json root;
    try {
        in >> root;
    } catch (...) {
        return false;
    }
    return find_runtime_graph_json(root, graph);
}

static std::string discover_runtime_graph_path(const std::string & model_hint) {
    const std::string explicit_path = env_value("DIST_PRESMOKE_RUNTIME_GRAPH_JSON");
    if (!explicit_path.empty()) {
        return explicit_path;
    }

    for (int pass = 0; pass < 2; ++pass) {
    for (const std::filesystem::path & root : search_roots()) {
        std::error_code ec;
        if (!std::filesystem::exists(root, ec)) {
            continue;
        }
        for (std::filesystem::recursive_directory_iterator it(root, ec), end; it != end && !ec; it.increment(ec)) {
            if (!it->is_regular_file(ec) || it->path().extension() != ".json") {
                continue;
            }
            if (pass == 0 && !model_hint.empty() &&
                    it->path().string().find(model_hint) == std::string::npos) {
                continue;
            }
            nlohmann::json graph;
            if (load_runtime_graph_json(it->path().string(), graph)) {
                return it->path().string();
            }
        }
    }
    }
    return {};
}

static std::vector<pipeline_boundary> runtime_graph_boundaries(const std::string & graph_path) {
    std::vector<pipeline_boundary> boundaries;
    nlohmann::json graph;
    if (!load_runtime_graph_json(graph_path, graph) || !graph.contains("roles") || !graph["roles"].is_array()) {
        return boundaries;
    }

    std::vector<pipeline_stage_info> stages;
    for (const nlohmann::json & role : graph["roles"]) {
        if (role.value("role", "") != "pipeline_stage") {
            continue;
        }
        pipeline_stage_info stage{};
        stage.stage_index = role.value("stage_index", 0);
        stage.layer_start = role.value("layer_start", 0);
        stage.layer_end = role.value("layer_end", 0);
        stage.node_id = role.value("node_id", "");
        stage.role = role.value("role", "");
        if (stage.layer_end > stage.layer_start) {
            stages.push_back(stage);
        }
    }
    std::sort(stages.begin(), stages.end(), [](const pipeline_stage_info & a, const pipeline_stage_info & b) {
        if (a.layer_start != b.layer_start) {
            return a.layer_start < b.layer_start;
        }
        return a.stage_index < b.stage_index;
    });

    for (size_t i = 0; i < stages.size(); ++i) {
        pipeline_boundary boundary{};
        boundary.index = static_cast<int>(i);
        boundary.boundary = stages[i].layer_end;
        boundary.producer = stages[i];
        if (i + 1 < stages.size()) {
            boundary.has_consumer = true;
            boundary.consumer = stages[i + 1];
        }
        boundaries.push_back(boundary);
    }
    return boundaries;
}

static std::string boundary_trace_for_producer(
        const presmoke_artifacts & artifacts,
        const pipeline_boundary & boundary) {
    if (boundary.producer.stage_index == 0) {
        return artifacts.entry_trace;
    }
    if (boundary.producer.stage_index == 1) {
        return artifacts.middle_trace;
    }
    return artifacts.final_trace;
}

static presmoke_artifacts discover_artifacts(const std::string & out_dir = "logs/presmoke") {
    presmoke_artifacts artifacts{};
    artifacts.worker_final_gguf =
            discover_file("DIST_PRESMOKE_WORKER_FINAL_GGUF", ".gguf", "final", false);
    if (artifacts.worker_final_gguf.empty()) {
        artifacts.worker_final_gguf =
                discover_file("DIST_PRESMOKE_WORKER_FINAL_GGUF", ".gguf", "worker", false);
    }
    artifacts.entry_trace = discover_file("DIST_PRESMOKE_ENTRY_TRACE", ".jsonl", "entry", false);
    if (artifacts.entry_trace.empty()) {
        artifacts.entry_trace = discover_file("DIST_PRESMOKE_ENTRY_TRACE", ".jsonl", "stage", false);
    }
    artifacts.middle_trace = discover_file("DIST_PRESMOKE_MIDDLE_TRACE", ".jsonl", "middle", false);
    artifacts.final_trace = discover_file("DIST_PRESMOKE_FINAL_TRACE", ".jsonl", "final", false);
    if (artifacts.final_trace.empty()) {
        artifacts.final_trace = discover_file("DIST_PRESMOKE_FINAL_TRACE", ".jsonl", "output", false);
    }
    artifacts.hidden_bin = discover_file("DIST_PRESMOKE_HIDDEN_BIN", ".bin", "hidden", false);
    artifacts.prompt = env_value("DIST_PRESMOKE_PROMPT", "The capital of France is");
    artifacts.runtime_graph_path = discover_runtime_graph_path(infer_model_hint(artifacts));
    resolve_reference_runtime(artifacts, out_dir);
    return artifacts;
}

static int print_manifest() {
    printf("test-runtime-presmoke: required gates before first smoke\n");
    for (const presmoke_gate & gate : REQUIRED_GATES) {
        printf("- %s: %s (%s)%s\n",
                gate.id,
                gate.binary,
                gate.evidence,
                gate.requires_model_artifacts ? " [artifact-backed; may SKIP]" : "");
    }
    printf("\n");
    printf("artifact run mode:\n");
    printf("  test-runtime-presmoke --run-boundary-gates\n");
    printf("  test-runtime-presmoke --run-artifact-gates\n");
    printf("\n");
    printf("manual env vars are optional overrides, not required launch inputs:\n");
    printf("  DIST_PRESMOKE_REFERENCE_RUNTIME\n");
    printf("  DIST_PRESMOKE_CACHED_REFERENCE_RUNTIME\n");
    printf("  DIST_PRESMOKE_LAYER_STORE_ROOT / DIST_PRESMOKE_MODEL_ID\n");
    printf("  DIST_PRESMOKE_RUNTIME_GRAPH_JSON\n");
    printf("  DIST_PRESMOKE_WORKER_FINAL_GGUF\n");
    printf("  DIST_PRESMOKE_ENTRY_TRACE\n");
    printf("  DIST_PRESMOKE_MIDDLE_TRACE\n");
    printf("  DIST_PRESMOKE_FINAL_TRACE\n");
    printf("  DIST_PRESMOKE_HIDDEN_BIN\n");
    printf("  DIST_PRESMOKE_PROMPT\n");
    printf("  DIST_PRESMOKE_OUT_DIR (default: logs/presmoke)\n");
    return 0;
}

static int print_artifacts() {
    const presmoke_artifacts artifacts = discover_artifacts();
    printf("test-runtime-presmoke: discovered artifacts\n");
    printf("REFERENCE_RUNTIME=%s\n",
            artifacts.reference_runtime.empty() ? "SKIPPED:not-found" : artifacts.reference_runtime.c_str());
    printf("REFERENCE_SOURCE=%s\n", artifacts.reference_source.c_str());
    printf("REFERENCE_MATERIALIZATION=%s:%s\n",
            artifacts.reference_materialization_status.c_str(),
            artifacts.reference_materialization_reason.c_str());
    printf("WORKER_FINAL_GGUF=%s\n",
            artifacts.worker_final_gguf.empty() ? "SKIPPED:not-found" : artifacts.worker_final_gguf.c_str());
    printf("ENTRY_TRACE=%s\n", artifacts.entry_trace.empty() ? "SKIPPED:not-found" : artifacts.entry_trace.c_str());
    printf("MIDDLE_TRACE=%s\n", artifacts.middle_trace.empty() ? "SKIPPED:not-found" : artifacts.middle_trace.c_str());
    printf("FINAL_TRACE=%s\n", artifacts.final_trace.empty() ? "SKIPPED:not-found" : artifacts.final_trace.c_str());
    printf("HIDDEN_BIN=%s\n", artifacts.hidden_bin.empty() ? "SKIPPED:not-found" : artifacts.hidden_bin.c_str());
    printf("RUNTIME_GRAPH=%s\n",
            artifacts.runtime_graph_path.empty() ? "SKIPPED:not-found" : artifacts.runtime_graph_path.c_str());
    printf("PROMPT=%s\n", artifacts.prompt.c_str());
    const std::vector<pipeline_boundary> boundaries =
            runtime_graph_boundaries(artifacts.runtime_graph_path);
    for (const pipeline_boundary & boundary : boundaries) {
        printf("BOUNDARY #%d=%d producer=%s[%d,%d) consumer=%s[%d,%d)\n",
                boundary.index,
                boundary.boundary,
                boundary.producer.node_id.c_str(),
                boundary.producer.layer_start,
                boundary.producer.layer_end,
                boundary.has_consumer ? boundary.consumer.node_id.c_str() : "none",
                boundary.has_consumer ? boundary.consumer.layer_start : -1,
                boundary.has_consumer ? boundary.consumer.layer_end : -1);
    }
    return 0;
}

static int write_env_template() {
    printf("# Optional overrides. The runner auto-discovers artifacts when these are unset.\n");
    printf("export DIST_PRESMOKE_REFERENCE_RUNTIME=/path/to/existing-reference-runtime.gguf\n");
    printf("export DIST_PRESMOKE_CACHED_REFERENCE_RUNTIME=/path/to/cached-reference-runtime.gguf\n");
    printf("export DIST_PRESMOKE_LAYER_STORE_ROOT=/path/to/layer-store-root\n");
    printf("export DIST_PRESMOKE_MODEL_ID=model-id\n");
    printf("export DIST_PRESMOKE_RUNTIME_GRAPH_JSON=/path/to/session-or-runtime-graph.json\n");
    printf("export DIST_PRESMOKE_WORKER_FINAL_GGUF=/path/to/materialized-final-worker.gguf\n");
    printf("export DIST_PRESMOKE_ENTRY_TRACE=/path/to/entry-or-first-stage.trace.jsonl\n");
    printf("export DIST_PRESMOKE_MIDDLE_TRACE=/path/to/middle-stage.trace.jsonl\n");
    printf("export DIST_PRESMOKE_FINAL_TRACE=/path/to/final-or-output.trace.jsonl\n");
    printf("export DIST_PRESMOKE_HIDDEN_BIN=/path/to/hidden-boundary.bin\n");
    printf("export DIST_PRESMOKE_PROMPT='The capital of France is'\n");
    printf("export DIST_PRESMOKE_OUT_DIR=logs/presmoke\n");
    return 0;
}

static std::vector<gate_command> artifact_gate_commands(
        const char * argv0,
        const std::string & out_dir,
        const presmoke_artifacts & artifacts,
        const bool include_heavy_gates) {
    std::vector<gate_command> commands = {
        {
            "descriptor_acceptance",
            shell_quote(binary_path(argv0, "test-runtime-acceptance")) +
                    " > " + shell_quote(out_dir + "/descriptor_acceptance.txt"),
            {},
        },
    };

    const std::vector<pipeline_boundary> boundaries =
            runtime_graph_boundaries(artifacts.runtime_graph_path);
    if (boundaries.empty()) {
        commands.push_back({
            "pipeline_boundaries",
            "",
            {
                { "RUNTIME_GRAPH", artifacts.runtime_graph_path },
                { "PIPELINE_BOUNDARIES", "" },
            },
        });
    }
    for (const pipeline_boundary & boundary : boundaries) {
        if (!boundary.has_consumer) {
            const std::string id = "boundary_" + std::to_string(boundary.index) + "_logits";
            const std::string evidence_path =
                    out_dir + "/boundary_" + std::to_string(boundary.index) + "_logits.json";
            commands.push_back({
                id,
                shell_quote(binary_path(argv0, "verify_logits_pipeline")) +
                        " " + shell_quote(artifacts.reference_runtime) +
                        " " + shell_quote(artifacts.final_trace) +
                        " " + shell_quote(artifacts.prompt) +
                        " > " + shell_quote(evidence_path),
                {
                    { "REFERENCE_RUNTIME", artifacts.reference_runtime },
                    { "RUNTIME_GRAPH", artifacts.runtime_graph_path },
                    { "FINAL_TRACE", artifacts.final_trace },
                    { "BOUNDARY_" + std::to_string(boundary.index), std::to_string(boundary.boundary) },
                },
                evidence_path,
            });
            continue;
        }
        const std::string trace_path = boundary_trace_for_producer(artifacts, boundary);
        const std::string id = "boundary_" + std::to_string(boundary.index) + "_hidden";
        const std::string evidence_path =
                out_dir + "/boundary_" + std::to_string(boundary.index) + "_hidden.json";
        commands.push_back({
            id,
            shell_quote(binary_path(argv0, "verify_hidden_pipeline")) +
                    " " + shell_quote(artifacts.reference_runtime) +
                    " " + shell_quote(trace_path) +
                    " " + shell_quote(artifacts.prompt) +
                    " " + shell_quote(std::to_string(boundary.boundary)) +
                    " --boundary-index " + shell_quote(std::to_string(boundary.index)) +
                    " > " + shell_quote(evidence_path),
            {
                { "REFERENCE_RUNTIME", artifacts.reference_runtime },
                { "RUNTIME_GRAPH", artifacts.runtime_graph_path },
                { "PRODUCER_TRACE", trace_path },
                { "BOUNDARY_" + std::to_string(boundary.index), std::to_string(boundary.boundary) },
            },
            evidence_path,
        });
    }

    if (!include_heavy_gates) {
        return commands;
    }

    const std::string final_boundary =
            boundaries.empty() ? std::string{} : std::to_string(boundaries.back().boundary);
    commands.insert(commands.end(), {
        {
            "logits_parity",
            shell_quote(binary_path(argv0, "verify_logits_pipeline")) +
                    " " + shell_quote(artifacts.reference_runtime) +
                    " " + shell_quote(artifacts.final_trace) +
                    " " + shell_quote(artifacts.prompt) +
                    " > " + shell_quote(out_dir + "/logits_parity.json"),
            {
                { "REFERENCE_RUNTIME", artifacts.reference_runtime },
                { "FINAL_TRACE", artifacts.final_trace },
            },
        },
        {
            "sampler_runtime",
            shell_quote(binary_path(argv0, "verify_final_runtime")) +
                    " " + shell_quote(artifacts.reference_runtime) +
                    " " + shell_quote(artifacts.worker_final_gguf) +
                    " " + shell_quote(artifacts.prompt) +
                    " " + shell_quote(final_boundary) +
                    " --out " + shell_quote(out_dir) +
                    " > " + shell_quote(out_dir + "/sampler_runtime.json"),
            {
                { "REFERENCE_RUNTIME", artifacts.reference_runtime },
                { "WORKER_FINAL_GGUF", artifacts.worker_final_gguf },
                { "FINAL_BOUNDARY", final_boundary },
            },
        },
        {
            "final_logits",
            shell_quote(binary_path(argv0, "verify_final_logits")) +
                    " " + shell_quote(artifacts.reference_runtime) +
                    " " + shell_quote(artifacts.worker_final_gguf) +
                    " " + shell_quote(artifacts.hidden_bin) +
                    " " + shell_quote(artifacts.prompt) +
                    " " + shell_quote(final_boundary) +
                    " > " + shell_quote(out_dir + "/final_logits.json"),
            {
                { "REFERENCE_RUNTIME", artifacts.reference_runtime },
                { "WORKER_FINAL_GGUF", artifacts.worker_final_gguf },
                { "HIDDEN_BIN", artifacts.hidden_bin },
                { "FINAL_BOUNDARY", final_boundary },
            },
        },
        {
            "decode_graph",
            shell_quote(binary_path(argv0, "verify_decode_graph")) +
                    " " + shell_quote(artifacts.reference_runtime) +
                    " " + shell_quote(artifacts.worker_final_gguf) +
                    " " + shell_quote(artifacts.prompt) +
                    " " + shell_quote(final_boundary) +
                    " --out " + shell_quote(out_dir) +
                    " --prefill" +
                    " > " + shell_quote(out_dir + "/decode_graph.json"),
            {
                { "REFERENCE_RUNTIME", artifacts.reference_runtime },
                { "WORKER_FINAL_GGUF", artifacts.worker_final_gguf },
                { "FINAL_BOUNDARY", final_boundary },
            },
        },
        {
            "decode_loop_trace",
            shell_quote(binary_path(argv0, "verify_decode_loop_parity")) +
                    " --model " + shell_quote(artifacts.reference_runtime) +
                    " --prompt " + shell_quote(artifacts.prompt) +
                    " --dist-entry " + shell_quote(artifacts.entry_trace) +
                    " --dist-final " + shell_quote(artifacts.final_trace) +
                    " > " + shell_quote(out_dir + "/decode_loop_trace.json"),
            {
                { "REFERENCE_RUNTIME", artifacts.reference_runtime },
                { "ENTRY_TRACE", artifacts.entry_trace },
                { "FINAL_TRACE", artifacts.final_trace },
            },
        },
    });
    return commands;
}

static std::string missing_inputs(const gate_command & gate) {
    std::string missing;
    for (const auto & input : gate.inputs) {
        if (input.second.empty()) {
            if (!missing.empty()) {
                missing += ", ";
            }
            missing += input.first;
        }
    }
    return missing;
}

static int run_command(const gate_command & gate) {
    printf("test-runtime-presmoke: running %s\n", gate.id.c_str());
    printf("%s\n", gate.command.c_str());
    const int rc = std::system(gate.command.c_str());
    if (rc != 0) {
        fprintf(stderr, "test-runtime-presmoke: gate failed: %s rc=%d\n", gate.id.c_str(), rc);
        return 1;
    }
    return 0;
}

static std::string gate_result_reason(const gate_command & gate) {
    if (gate.evidence_path.empty()) {
        return "command failed";
    }
    std::ifstream in(gate.evidence_path);
    if (!in) {
        return "command failed";
    }
    try {
        nlohmann::json evidence;
        in >> evidence;
        const std::string message = evidence.value("message", "");
        if (!message.empty()) {
            return message;
        }
    } catch (...) {
    }
    return "command failed";
}

static void write_summary(
        const std::string & out_dir,
        const presmoke_artifacts & artifacts,
        const std::vector<gate_result> & results) {
    std::ofstream out(out_dir + "/presmoke_summary.json", std::ios::trunc);
    if (!out) {
        return;
    }

    bool smoke_ready = true;
    for (const gate_result & result : results) {
        if (result.status != "PASS") {
            smoke_ready = false;
        }
    }

    out << "{\n";
    out << "  \"smoke_ready\": " << (smoke_ready ? "true" : "false") << ",\n";
    out << "  \"artifacts\": {\n";
    out << "    \"reference_runtime\": \"" << json_escape(artifacts.reference_runtime) << "\",\n";
    out << "    \"reference_source\": \"" << json_escape(artifacts.reference_source) << "\",\n";
    out << "    \"reference_materialization_status\": \""
        << json_escape(artifacts.reference_materialization_status) << "\",\n";
    out << "    \"reference_materialization_reason\": \""
        << json_escape(artifacts.reference_materialization_reason) << "\",\n";
    out << "    \"worker_final_gguf\": \"" << json_escape(artifacts.worker_final_gguf) << "\",\n";
    out << "    \"entry_trace\": \"" << json_escape(artifacts.entry_trace) << "\",\n";
    out << "    \"middle_trace\": \"" << json_escape(artifacts.middle_trace) << "\",\n";
    out << "    \"final_trace\": \"" << json_escape(artifacts.final_trace) << "\",\n";
    out << "    \"hidden_bin\": \"" << json_escape(artifacts.hidden_bin) << "\",\n";
    out << "    \"runtime_graph\": \"" << json_escape(artifacts.runtime_graph_path) << "\",\n";
    out << "    \"prompt\": \"" << json_escape(artifacts.prompt) << "\"\n";
    out << "  },\n";
    const std::vector<pipeline_boundary> boundaries =
            runtime_graph_boundaries(artifacts.runtime_graph_path);
    out << "  \"boundaries\": [\n";
    for (size_t i = 0; i < boundaries.size(); ++i) {
        const pipeline_boundary & boundary = boundaries[i];
        out << "    { \"index\": " << boundary.index
            << ", \"boundary\": " << boundary.boundary
            << ", \"producer\": \"" << json_escape(boundary.producer.node_id)
            << "\", \"producer_range\": [" << boundary.producer.layer_start
            << ", " << boundary.producer.layer_end << "]"
            << ", \"consumer\": \"" << json_escape(boundary.has_consumer ? boundary.consumer.node_id : "")
            << "\", \"consumer_range\": [" << (boundary.has_consumer ? boundary.consumer.layer_start : -1)
            << ", " << (boundary.has_consumer ? boundary.consumer.layer_end : -1) << "] }";
        out << (i + 1 == boundaries.size() ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"gates\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const gate_result & result = results[i];
        out << "    { \"id\": \"" << json_escape(result.id) << "\", \"status\": \""
            << json_escape(result.status) << "\", \"reason\": \""
            << json_escape(result.reason) << "\" }";
        out << (i + 1 == results.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
}

static int print_artifact_commands(const char * argv0, const bool include_heavy_gates) {
    const std::string out_dir = env_value("DIST_PRESMOKE_OUT_DIR", "logs/presmoke");
    const presmoke_artifacts artifacts = discover_artifacts();
    printf("test-runtime-presmoke: artifact-backed gate commands\n");
    printf("mkdir -p %s\n", shell_quote(out_dir).c_str());
    for (const gate_command & gate : artifact_gate_commands(argv0, out_dir, artifacts, include_heavy_gates)) {
        const std::string missing = missing_inputs(gate);
        if (!missing.empty()) {
            printf("# %s SKIPPED missing: %s\n", gate.id.c_str(), missing.c_str());
        } else {
            printf("# %s\n%s\n", gate.id.c_str(), gate.command.c_str());
        }
    }
    return 0;
}

static int run_artifact_gates(const char * argv0, const bool include_heavy_gates) {
    const std::string out_dir = env_value("DIST_PRESMOKE_OUT_DIR", "logs/presmoke");
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    if (ec) {
        fprintf(stderr,
                "test-runtime-presmoke: failed to create output dir %s: %s\n",
                out_dir.c_str(),
                ec.message().c_str());
        return 1;
    }

    const presmoke_artifacts artifacts = discover_artifacts();
    std::vector<gate_result> results;
    bool failed = false;

    if (artifacts.reference_runtime.empty()) {
        results.push_back({
            "reference_runtime",
            "SKIPPED",
            artifacts.reference_materialization_reason.empty() ?
                    "reference runtime unavailable" :
                    artifacts.reference_materialization_reason,
        });
    } else {
        results.push_back({
            "reference_runtime",
            artifacts.reference_materialization_status == "FAIL" ? "FAIL" : "PASS",
            artifacts.reference_source + ": " + artifacts.reference_materialization_reason,
        });
        if (artifacts.reference_materialization_status == "FAIL") {
            failed = true;
        }
    }

    for (const gate_command & gate : artifact_gate_commands(argv0, out_dir, artifacts, include_heavy_gates)) {
        const std::string missing = missing_inputs(gate);
        if (!missing.empty()) {
            printf("test-runtime-presmoke: SKIPPED %s missing: %s\n",
                    gate.id.c_str(),
                    missing.c_str());
            results.push_back({ gate.id, "SKIPPED", "missing: " + missing });
            continue;
        }
        if (run_command(gate) != 0) {
            failed = true;
            results.push_back({ gate.id, "FAIL", gate_result_reason(gate) });
            continue;
        }
        results.push_back({ gate.id, "PASS", "" });
    }

    write_summary(out_dir, artifacts, results);
    printf("test-runtime-presmoke: evidence_dir=%s\n", out_dir.c_str());
    printf("test-runtime-presmoke: summary=%s\n", (out_dir + "/presmoke_summary.json").c_str());
    return failed ? 1 : 0;
}

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "--manifest";
    if (mode == "--manifest") {
        return print_manifest();
    }
    if (mode == "--require-artifacts" || mode == "--discover-artifacts") {
        return print_artifacts();
    }
    if (mode == "--run-artifact-gates") {
        return run_artifact_gates(argv[0], true);
    }
    if (mode == "--run-boundary-gates") {
        return run_artifact_gates(argv[0], false);
    }
    if (mode == "--print-commands") {
        return print_artifact_commands(argv[0], true);
    }
    if (mode == "--print-boundary-commands") {
        return print_artifact_commands(argv[0], false);
    }
    if (mode == "--write-env-template") {
        return write_env_template();
    }
    fprintf(stderr,
            "usage: %s [--manifest|--discover-artifacts|--require-artifacts|--run-boundary-gates|--run-artifact-gates|--print-boundary-commands|--print-commands|--write-env-template]\n",
            argv[0]);
    return 2;
}
