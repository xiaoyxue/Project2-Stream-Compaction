#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace EfficientHardwareOpt {
        StreamCompaction::Common::PerformanceTimer& timer();

        // Set devicePointers when both input and output are device buffers.
        void scan(int n, int *odata, const int *idata, bool devicePointers = false);

        // Stable removal of zeros from host input; returns the valid output count.
        int compact(int n, int *odata, const int *idata);
    }
}
