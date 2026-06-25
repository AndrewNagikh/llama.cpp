// Unit tests for registered model status handling.

#include "orchestrator/model_registry.h"

#include <cassert>
#include <cstdio>

static void test_default_status_is_discovered() {
    cluster_model_registry reg;

    dist_model_record r;
    r.model_id = "llama-3.2-1b";
    reg.add_or_update(r);

    [[maybe_unused]] const auto * found = reg.find("llama-3.2-1b");
    assert(found != nullptr);
    assert(found->status == dist_model_status::discovered);

    printf("test_default_status_is_discovered passed\n");
}

static void test_status_to_string() {
    assert(dist_model_status_to_string(dist_model_status::discovered)           == "DISCOVERED");
    assert(dist_model_status_to_string(dist_model_status::manifest_pending)    == "MANIFEST_PENDING");
    assert(dist_model_status_to_string(dist_model_status::manifest_ready)      == "MANIFEST_READY");
    assert(dist_model_status_to_string(dist_model_status::installing)          == "INSTALLING");
    assert(dist_model_status_to_string(dist_model_status::partially_available) == "PARTIALLY_AVAILABLE");
    assert(dist_model_status_to_string(dist_model_status::available)           == "AVAILABLE");
    assert(dist_model_status_to_string(dist_model_status::degraded)            == "DEGRADED");
    assert(dist_model_status_to_string(dist_model_status::unavailable)         == "UNAVAILABLE");

    printf("test_status_to_string passed\n");
}

static void test_status_from_string_case_insensitive() {
    assert(dist_model_status_from_string("DISCOVERED") == dist_model_status::discovered);
    assert(dist_model_status_from_string("available")  == dist_model_status::available);
    assert(dist_model_status_from_string("Installing") == dist_model_status::installing);
    assert(dist_model_status_from_string("UNKNOWN")    == dist_model_status::discovered);

    printf("test_status_from_string_case_insensitive passed\n");
}

static void test_registration_forces_discovered() {
    nlohmann::json body = {
        { "model_id", "m" },
        { "status", "AVAILABLE" }
    };

    dist_model_record r = dist_model_record_from_json(body);
    assert(r.status == dist_model_status::discovered);

    printf("test_registration_forces_discovered passed\n");
}

int main() {
    test_default_status_is_discovered();
    test_status_to_string();
    test_status_from_string_case_insensitive();
    test_registration_forces_discovered();

    printf("test-model-status: all tests passed\n");
    return 0;
}
