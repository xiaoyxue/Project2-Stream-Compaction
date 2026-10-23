#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace EfficientHardwareRecursive {
        StreamCompaction::Common::PerformanceTimer& timer();

        // Set devicePointers when both input and output are device buffers.
        void scan(int n, int *odata, const int *idata, bool devicePointers = false);
        void conflictFreeScan(int n, int *odata, const int *idata, bool devicePointers = false);

        int compact(int n, int *odata, const int *idata);
        int conflictFreeCompact(int n, int *odata, const int *idata);
    }
}
