#pragma once

#include "layer.h"
#include <nvtx3/nvToolsExt.h>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

// Only the requested rank constructs this scheduler. All state notifications run
// under scheduler_mutex, so marks from the engine and receive threads are ordered.
class GapTracedDefragScheduler final : public UnifiedDefraggingLayerScheduler {
public:
    using UnifiedDefraggingLayerScheduler::UnifiedDefraggingLayerScheduler;

    static const char* worker() {
        static const char* value = std::getenv("DMOE_GAP_WORKER");
        return value && *value ? value : nullptr;
    }

    void begin_trace() {
        depths.assign(num_layers, 0);
        nvtxMarkA("WG.ready=0");
    }

    static void queue_updated(UnifiedLayerSchedulerBase* scheduler, int layer, const char* source) {
        if (worker()) {
            if (auto* trace = dynamic_cast<GapTracedDefragScheduler*>(scheduler)) {
                trace->update_depth(layer, source);
            }
        }
    }

    static void selected(UnifiedLayerSchedulerBase* scheduler, int candidate_rows) {
        if (worker()) {
            if (auto* trace = dynamic_cast<GapTracedDefragScheduler*>(scheduler)) {
                if (trace->total_rows != candidate_rows) {
                    throw std::logic_error("Gap trace disagrees with scheduler ready rows");
                }
                nvtxMarkA(("WG.claim=" + std::to_string(++trace->next_claim)).c_str());
            }
        }
    }

private:
    std::vector<int> depths;
    std::int64_t total_rows{0};
    std::uint64_t next_claim{0};

    void update_depth(int layer, const char* source) {
        int depth = layers[layer]->get_num_tokens();
        bool was_empty = total_rows == 0;
        total_rows += depth - depths[layer];
        depths[layer] = depth;
        if (was_empty && total_rows > 0) {
            nvtxMarkA("WG.ready=1");
            int physical_layer = is_attn_layer(layer) ? layer : layer - num_attn_layers;
            nvtxMarkA(("WG.wake=" + std::string(source) + ":" + std::to_string(physical_layer)).c_str());
        } else if (!was_empty && total_rows == 0) {
            nvtxMarkA("WG.ready=0");
        }
    }
};
