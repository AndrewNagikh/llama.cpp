#include "orchestrator/model_provider/model_provider.h"
#include "orchestrator/model_registry.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

using json = nlohmann::json;

static cluster_model_registry g_reg;
static httplib::Server g_svr;

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

        const std::string provider_name = record->source.empty() ? "huggingface" : record->source;
        auto provider = create_model_provider(provider_name);
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
        res.set_content(json({
            { "status", "ok" },
            { "provider", result.provider },
            { "files", static_cast<int>(result.files.size()) },
            { "revision", result.revision }
        }).dump(), "application/json");
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
    int port = 0;
    start_server(port);

    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(60, 0);

    json reg_body = {
        { "model_id", "llama-3.2-1b" },
        { "source", "huggingface" },
        { "repository", "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF" },
        { "revision", "main" }
    };

    auto reg_res = cli.Post("/models/register", reg_body.dump(), "application/json");
    assert(reg_res && reg_res->status == 200);

    auto disc_res = cli.Post("/models/llama-3.2-1b/discover", "", "application/json");
    if (!disc_res || disc_res->status != 200) {
        std::cerr << "test-model-discovery: discovery request failed: "
                  << (disc_res ? disc_res->body : "no response") << "\n";
        return 1;
    }

    const json disc_json = json::parse(disc_res->body);
    assert(disc_json.value("status", "") == "ok");
    assert(disc_json.value("files", 0) > 0);

    auto get_res = cli.Get("/models/llama-3.2-1b");
    assert(get_res && get_res->status == 200);

    const json model_json = json::parse(get_res->body);
    assert(model_json.value("status", "") == "MANIFEST_PENDING");
    assert(model_json.value("provider", "") == "huggingface");
    assert(model_json.value("provider_revision", "") == "main");
    assert(model_json["files"].is_array());
    assert(!model_json["files"].empty());

    [[maybe_unused]] bool found_gguf = false;
    for (const auto & f : model_json["files"]) {
        const uint64_t size = f.value("size_bytes", static_cast<uint64_t>(0));
        assert(size > 0);
        if (f.value("filename", "") == "llama-3.2-1b-instruct-q4_k_m.gguf") {
            found_gguf = true;
            std::cout << "test-model-discovery: GGUF size=" << size << "\n";
        }
    }
    assert(found_gguf);

    g_svr.stop();
    std::cout << "test-model-discovery: OK\n";
    return 0;
}
