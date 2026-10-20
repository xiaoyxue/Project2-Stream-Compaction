#include <stdexcept>
#include "radix_sort.h"

#ifndef STREAM_COMPACTION_RADIX_SORT_ENABLE_TIMING
#define STREAM_COMPACTION_RADIX_SORT_ENABLE_TIMING 1
#endif

namespace StreamCompaction {
    namespace RadixSort {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        void sort(int n, int *odata, const int *idata) {
#if STREAM_COMPACTION_RADIX_SORT_ENABLE_TIMING
            const bool ownsTimer = !timer().isGpuTimerStarted();
            if (ownsTimer) {
                timer().startGpuTimer();
            }
#endif

            // TODO: Implement radix sort.

#if STREAM_COMPACTION_RADIX_SORT_ENABLE_TIMING
            if (ownsTimer) {
                timer().endGpuTimer();
            }
#endif
            throw std::logic_error("RadixSort::sort is not implemented");
        }
    }
}
