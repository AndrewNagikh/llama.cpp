// Unit tests for the Cluster Model Registry REST API.

#include "orchestrator/model_registry.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

using json = nlohmann::json;

class registry_server {
public:
    registry_server() {
        svr.Post("/models/register", [this](const httplib::Request & req, httplib::Response & res) {
            json body;
            try {
                body = json::parse(req.body);
            } catch (...) {
                res.status = 400;
                res.set_content(R"({"error":"invalid json"})", "application/json");
                return;
            }

            if (!body.contains("model_id") || body.value("model_id", "").empty()) {
                res.status = 400;
                res.set_content(json({ { "error", "model_id is required" } }).dump(), "application/json");
                return;
            }

            dist_model_record record = dist_model_record_from_json(body);
            record.status = dist_model_status::discovered;
            registry.add_or_update(record);
            res.set_content(record.to_json().dump(), "application/json");
        });

        svr.Get("/models", [this](const httplib::Request &, httplib::Response & res) {
            const auto records = registry.list();
            json out = json::array();
            for (const auto & r : records) {
                out.push_back(r.to_json());
            }
            res.set_content(out.dump(), "application/json");
        });

        svr.Get(R"(/models/(.+))", [this](const httplib::Request & req, httplib::Response & res) {
            const std::string model_id = req.matches[1];
            const auto * record = registry.find(model_id);
            if (!record) {
                res.status = 404;
                res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
                return;
            }
            res.set_content(record->to_json().dump(), "application/json");
        });

        svr.Delete(R"(/models/(.+))", [this](const httplib::Request & req, httplib::Response & res) {
            const std::string model_id = req.matches[1];
            if (registry.remove(model_id)) {
                res.set_content(json({ { "ok", true } }).dump(), "application/json");
                return;
            }
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
        });
    }

    void start(int port) {
        server_thread_ = std::thread([this, port]() {
            svr.listen("127.0.0.1", port);
        });

        for (int i = 0; i < 50; ++i) {
            httplib::Client probe("127.0.0.1", port);
            probe.set_connection_timeout(0, 100);
            if (probe.Get("/models")) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    void stop() {
        svr.stop();
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
    }

private:
    httplib::Server svr;
    std::thread server_thread_;
    cluster_model_registry registry;
};

static json register_payload() {
    return {
        { "model_id", "llama-3.2-1b" },
        { "display_name", "Llama 3.2 1B Q4_K_M" },
        { "source", "huggingface" },
        { "repository", "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF" },
        { "filename", "llama-3.2-1b-instruct-q4_k_m.gguf" },
        { "revision", "main" },
        { "architecture", "llama" }
    };
}

int main() {
    const int port = 29876;
    registry_server srv;
    srv.start(port);

    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(1, 0);

    // POST /models/register
    {
        const auto res = cli.Post("/models/register", register_payload().dump(), "application/json");
        if (!res || res->status != 200) {
            fprintf(stderr, "test-model-api: POST register failed\n");
            srv.stop();
            return 1;
        }
        json body = json::parse(res->body);
        if (body.value("model_id", "") != "llama-3.2-1b" ||
            body.value("status", "") != "DISCOVERED") {
            fprintf(stderr, "test-model-api: unexpected register response\n");
            srv.stop();
            return 1;
        }
    }

    // POST /models/register missing model_id
    {
        const json bad = { { "display_name", "no id" } };
        const auto res = cli.Post("/models/register", bad.dump(), "application/json");
        if (!res || res->status != 400) {
            fprintf(stderr, "test-model-api: expected 400 for missing model_id\n");
            srv.stop();
            return 1;
        }
    }

    // GET /models/{model_id}
    {
        const auto res = cli.Get("/models/llama-3.2-1b");
        if (!res || res->status != 200) {
            fprintf(stderr, "test-model-api: GET model failed\n");
            srv.stop();
            return 1;
        }
        json body = json::parse(res->body);
        if (body.value("filename", "") != "llama-3.2-1b-instruct-q4_k_m.gguf" ||
            body.value("status", "") != "DISCOVERED") {
            fprintf(stderr, "test-model-api: unexpected GET model response\n");
            srv.stop();
            return 1;
        }
    }

    // GET /models (list)
    {
        const auto res = cli.Get("/models");
        if (!res || res->status != 200) {
            fprintf(stderr, "test-model-api: GET models list failed\n");
            srv.stop();
            return 1;
        }
        json body = json::parse(res->body);
        if (!body.is_array() || body.size() != 1) {
            fprintf(stderr, "test-model-api: expected list of size 1\n");
            srv.stop();
            return 1;
        }
    }

    // DELETE /models/{model_id}
    {
        const auto res = cli.Delete("/models/llama-3.2-1b");
        if (!res || res->status != 200) {
            fprintf(stderr, "test-model-api: DELETE failed\n");
            srv.stop();
            return 1;
        }
        json body = json::parse(res->body);
        if (body.value("ok", false) != true) {
            fprintf(stderr, "test-model-api: unexpected DELETE response\n");
            srv.stop();
            return 1;
        }
    }

    // GET after delete -> 404
    {
        const auto res = cli.Get("/models/llama-3.2-1b");
        if (!res || res->status != 404) {
            fprintf(stderr, "test-model-api: expected 404 after delete\n");
            srv.stop();
            return 1;
        }
    }

    srv.stop();
    printf("test-model-api: all tests passed\n");
    return 0;
}
