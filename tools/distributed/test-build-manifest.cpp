#include "orchestrator/manifest_builder/manifest_builder.h"
#include "orchestrator/model_provider/model_provider.h"
#include "orchestrator/model_registry.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

using json = nlohmann::json;

static cluster_model_registry g_reg;
static httplib::Server g_svr;
static std::string g_models_dir;
static std::string g_model_path;

static void start_server(int & port) {
    port = g_svr.bind_to_any_port("0.0.0.0");
    assert(port > 0);

    g_svr.Post("/models/register", [](const httplib::Request & req, httplib::Response & res) {
        json body = json::parse(req.body);
        dist_model_record record = dist_model_record_from_json(body);
        record.status = dist_model_status::discovered;
        g_reg.add_or_update(record);
        res.set_content(record.to_json().dump(), "application/json");
    });

    g_svr.Post(R"(/models/([^/]+)/discover)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];
        dist_model_record * record = g_reg.find(model_id);
        if (!record) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }

        auto provider = create_model_provider(record->source.empty() ? "huggingface" : record->source);
        if (!provider) {
            res.status = 400;
            res.set_content(json({ { "error", "unknown provider" } }).dump(), "application/json");
            return;
        }

        const auto result = provider->discover(*record);
        if (!result.success) {
            res.status = 502;
            res.set_content(json({ { "error", result.error } }).dump(), "application/json");
            return;
        }

        g_reg.apply_discovery(model_id, result, record);
        res.set_content(json({ { "status", "ok" } }).dump(), "application/json");
    });

    g_svr.Post(R"(/models/([^/]+)/manifest)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];
        dist_model_record * record = g_reg.find(model_id);
        if (!record) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }

        const auto build = build_manifest_for_record(*record, g_models_dir, g_model_path);
        if (!build.success) {
            res.status = 502;
            res.set_content(json({ { "error", build.error } }).dump(), "application/json");
            return;
        }

        g_reg.apply_manifest(model_id, build.manifest, record);
        res.set_content(json({
            { "status", "ok" },
            { "n_layer", build.manifest.n_layer },
            { "layers", static_cast<int>(build.manifest.layers.size()) },
            { "metadata_bytes_read", build.bytes_read },
        }).dump(), "application/json");
    });

    g_svr.Get(R"(/models/([^/]+)/manifest)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];
        const auto * record = g_reg.find(model_id);
        if (!record || record->status != dist_model_status::manifest_ready || !record->manifest) {
            res.status = 404;
            res.set_content(json({ { "error", "manifest not ready" } }).dump(), "application/json");
            return;
        }
        res.set_content(record->manifest->to_json().dump(), "application/json");
    });

    g_svr.Get(R"(/models/([^/]+))", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];
        const auto * record = g_reg.find(model_id);
        if (!record) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }
        res.set_content(record->to_json().dump(), "application/json");
    });

    std::thread([]() { g_svr.listen_after_bind(); }).detach();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

int main() {
    const char * model_env = std::getenv("MODEL");
    if (!model_env || !std::filesystem::exists(model_env)) {
        std::cerr << "test-build-manifest: set MODEL to a local GGUF file\n";
        return 77;
    }

    g_model_path = model_env;
    g_models_dir = std::filesystem::path(g_model_path).parent_path().string();

    int port = 0;
    start_server(port);

    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(120, 0);

    const std::string filename = std::filesystem::path(g_model_path).filename().string();

    json reg_body = {
        { "model_id", "llama-3.2-1b" },
        { "source", "huggingface" },
        { "repository", "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF" },
        { "filename", filename },
        { "revision", "main" },
    };

    auto reg_res = cli.Post("/models/register", reg_body.dump(), "application/json");
    assert(reg_res && reg_res->status == 200);

    auto disc_res = cli.Post("/models/llama-3.2-1b/discover", "", "application/json");
    assert(disc_res && disc_res->status == 200);

    auto get_pending = cli.Get("/models/llama-3.2-1b");
    assert(get_pending && get_pending->status == 200);
    assert(json::parse(get_pending->body).value("status", "") == "MANIFEST_PENDING");

    auto build_res = cli.Post("/models/llama-3.2-1b/manifest", "", "application/json");
    if (!build_res || build_res->status != 200) {
        std::cerr << "test-build-manifest: build failed: "
                  << (build_res ? build_res->body : "no response") << "\n";
        return 1;
    }

    const json build_json = json::parse(build_res->body);
    assert(build_json.value("status", "") == "ok");
    assert(build_json.value("n_layer", 0) > 0);
    assert(build_json.value("layers", 0) > 0);
    assert(build_json.value("metadata_bytes_read", static_cast<uint64_t>(0)) > 0);

    auto get_ready = cli.Get("/models/llama-3.2-1b");
    assert(get_ready && get_ready->status == 200);
    assert(json::parse(get_ready->body).value("status", "") == "MANIFEST_READY");

    auto manifest_res = cli.Get("/models/llama-3.2-1b/manifest");
    assert(manifest_res && manifest_res->status == 200);
    const json manifest_json = json::parse(manifest_res->body);
    assert(manifest_json.value("n_layer", 0) > 0);
    assert(manifest_json["layers"].is_array());
    assert(!manifest_json["layers"].empty());
    assert(manifest_json["tensors"].is_array());
    assert(!manifest_json["tensors"].empty());

    const uint64_t file_size = (uint64_t) std::filesystem::file_size(g_model_path);
    const uint64_t meta_read = manifest_json.value("metadata_bytes_read", static_cast<uint64_t>(0));
    assert(meta_read > 0);
    assert(meta_read < file_size);

    g_svr.stop();
    std::cout << "test-build-manifest: OK meta_bytes=" << meta_read
              << " file_size=" << file_size << "\n";
    return 0;
}
