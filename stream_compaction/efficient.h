#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        __global__ void kernUpSweep(int n, int offset, int *data);
        __global__ void kernDownSweep(int n, int offset, int *data, const int *idata);

        // Set devicePointers when both input and output are device buffers.
        void scan(int n, int *odata, const int *idata, bool devicePointers = false);

        int compact(int n, int *odata, const int *idata);
    }
}
