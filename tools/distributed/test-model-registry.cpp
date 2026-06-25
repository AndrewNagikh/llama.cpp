// Unit tests for the cluster model registry.

#include "orchestrator/model_registry.h"

#include <cassert>
#include <cstdio>
#include <string>

static void test_create_and_find() {
    cluster_model_registry reg;

    dist_model_record r;
    r.model_id     = "llama-3.2-1b";
    r.display_name = "Llama 3.2 1B";
    r.source       = "huggingface";
    r.repository   = "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF";
    r.filename     = "llama-3.2-1b-instruct-q4_k_m.gguf";
    r.revision     = "main";
    r.architecture = "llama";
    reg.add_or_update(r);

    [[maybe_unused]] const auto * found = reg.find("llama-3.2-1b");
    assert(found != nullptr);
    assert(found->model_id == "llama-3.2-1b");
    assert(found->filename == "llama-3.2-1b-instruct-q4_k_m.gguf");
    assert(found->status == dist_model_status::discovered);

    [[maybe_unused]] const auto * missing = reg.find("does-not-exist");
    assert(missing == nullptr);

    printf("test_create_and_find passed\n");
}

static void test_re_register_updates() {
    cluster_model_registry reg;

    dist_model_record r1;
    r1.model_id     = "m1";
    r1.display_name = "First";
    r1.filename     = "first.gguf";
    reg.add_or_update(r1);

    dist_model_record r2;
    r2.model_id     = "m1";
    r2.display_name = "Second";
    r2.filename     = "second.gguf";
    r2.architecture = "llama";
    reg.add_or_update(r2);

    [[maybe_unused]] const auto * found = reg.find("m1");
    assert(found != nullptr);
    assert(found->display_name == "Second");
    assert(found->filename == "second.gguf");
    assert(found->architecture == "llama");

    printf("test_re_register_updates passed\n");
}

static void test_remove() {
    cluster_model_registry reg;

    dist_model_record r;
    r.model_id = "remove-me";
    reg.add_or_update(r);

    assert(reg.remove("remove-me") == true);
    assert(reg.find("remove-me") == nullptr);
    assert(reg.remove("remove-me") == false);

    printf("test_remove passed\n");
}

static void test_list() {
    cluster_model_registry reg;

    for (int i = 0; i < 3; ++i) {
        dist_model_record r;
        r.model_id = "model-" + std::to_string(i);
        reg.add_or_update(r);
    }

    const auto models = reg.list();
    assert(models.size() == 3);

    printf("test_list passed\n");
}

static void test_unique_model_id() {
    cluster_model_registry reg;

    dist_model_record a;
    a.model_id = "same-id";
    a.filename = "a.gguf";
    reg.add_or_update(a);

    dist_model_record b;
    b.model_id = "same-id";
    b.filename = "b.gguf";
    reg.add_or_update(b);

    const auto models = reg.list();
    assert(models.size() == 1);
    assert(models.front().filename == "b.gguf");

    printf("test_unique_model_id passed\n");
}

int main() {
    test_create_and_find();
    test_re_register_updates();
    test_remove();
    test_list();
    test_unique_model_id();

    printf("test-model-registry: all tests passed\n");
    return 0;
}
