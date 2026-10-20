#include <stdexcept>
#include "efficient_hardware_opt.h"

#ifndef STREAM_COMPACTION_EFFICIENT_HARDWARE_OPT_ENABLE_TIMING
#define STREAM_COMPACTION_EFFICIENT_HARDWARE_OPT_ENABLE_TIMING 1
#endif

namespace StreamCompaction {
    namespace EfficientHardwareOpt {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        void scan(int n, int *odata, const int *idata, bool devicePointers) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
        }

        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
        }
    }
}
