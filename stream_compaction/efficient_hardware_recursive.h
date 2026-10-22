#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace EfficientHardwareRecursive {
        StreamCompaction::Common::PerformanceTimer& timer();

        // Unimplemented placeholders; both functions throw std::logic_error.
        void scan(int n, int *odata, const int *idata, bool devicePointers = false);
        void conflictFreeScan(int n, int *odata, const int *idata, bool devicePointers = false);

        int compact(int n, int *odata, const int *idata);
    }
}
