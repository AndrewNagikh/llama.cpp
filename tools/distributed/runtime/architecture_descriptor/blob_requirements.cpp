#include "blob_requirements.h"

#include "architecture/semantic_blob.h"
#include "architecture/semantic_roles.h"

namespace {

void mark_blob(
        std::map<std::string, blob_stage_requirements> & out,
        const std::string & blob_id,
        worker_role role) {
    blob_stage_requirements & req = out[blob_id];
    switch (role) {
        case worker_role::entry:
        case worker_role::full:
            req.required_for_entry = true;
            break;
        case worker_role::middle:
            req.required_for_middle = true;
            break;
        case worker_role::final:
            req.required_for_final = true;
            break;
    }
}

} // namespace

std::map<std::string, blob_stage_requirements> compute_blob_stage_requirements(
        const semantic_runtime_descriptor & rt) {
    std::map<std::string, blob_stage_requirements> out;

    for (const worker_descriptor & worker : rt.workers) {
        for (const std::string & blob_id : worker.required_blob_ids) {
            mark_blob(out, blob_id, worker.role);
        }
    }

    for (const semantic_blob & blob : rt.blobs) {
        if (blob.deploy == blob_deploy_target::all_nodes) {
            blob_stage_requirements & req = out[blob.id];
            req.required_for_entry  = true;
            req.required_for_middle = true;
            req.required_for_final  = true;
        } else if (blob.deploy == blob_deploy_target::entry_node) {
            out[blob.id].required_for_entry = true;
        } else if (blob.deploy == blob_deploy_target::final_node) {
            out[blob.id].required_for_final = true;
        }
    }

    for (const semantic_blob & blob : rt.blobs) {
        if (blob.role != tensor_semantic_role::transformer_layer) {
            continue;
        }
        blob_stage_requirements & req = out[blob.id];
        req.required_for_entry  = true;
        req.required_for_middle = true;
        req.required_for_final  = true;
    }

    return out;
}

bool blob_required_for_role(
        const blob_stage_requirements & req,
        const worker_role role) {
    switch (role) {
        case worker_role::entry:
        case worker_role::full:
            return req.required_for_entry;
        case worker_role::middle:
            return req.required_for_middle;
        case worker_role::final:
            return req.required_for_final;
    }
    return false;
}
