#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace RadixSort {
        StreamCompaction::Common::PerformanceTimer& timer();

        // Sort host input into host output in ascending order.
        void sort(int n, int *odata, const int *idata);
    }
}
