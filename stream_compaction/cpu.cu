#include <cstdio>
#include "cpu.h"

#include "common.h"

#ifndef STREAM_COMPACTION_CPU_ENABLE_TIMING
#define STREAM_COMPACTION_CPU_ENABLE_TIMING 1
#endif

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        void MapToBoolean(int n, int *odata, const int *idata) {
            for (int i = 0; i < n; ++i) {
                odata[i] = (idata[i] != 0) ? 1 : 0;
            }
        }

        int Scatter(int n, int *odata, const int *idata, const int *bools, const int *scan) {
            int count = 0;
            for (int i = 0; i < n; ++i) {
                if (bools[i]) {
                    odata[scan[i]] = idata[i];
                    count++;
                }
            }
            return count;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n < 1) {
                return;
            }
#if STREAM_COMPACTION_CPU_ENABLE_TIMING
            const bool ownsTimer = !timer().isCpuTimerStarted();
            if (ownsTimer) {
                timer().startCpuTimer();
            }
#endif
			odata[0] = 0;
			for (int i = 1; i < n; i++) {
				odata[i] = odata[i - 1] + idata[i - 1];
			}
#if STREAM_COMPACTION_CPU_ENABLE_TIMING
            if (ownsTimer) {
                timer().endCpuTimer();
            }
#endif
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
#if STREAM_COMPACTION_CPU_ENABLE_TIMING
            const bool ownsTimer = !timer().isCpuTimerStarted();
            if (ownsTimer) {
                timer().startCpuTimer();
            }
#endif
            // TODO
            int j = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[j++] = idata[i];
                }
            }
#if STREAM_COMPACTION_CPU_ENABLE_TIMING
            if (ownsTimer) {
                timer().endCpuTimer();
            }
#endif
            return j;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {

            int *bools = new int[n];
            int *exclusive_scan = new int[n];
#if STREAM_COMPACTION_CPU_ENABLE_TIMING
            const bool ownsTimer = !timer().isCpuTimerStarted();
            if (ownsTimer) {
                timer().startCpuTimer();
            }
#endif
            // TODO
            MapToBoolean(n, bools, idata);
            scan(n, exclusive_scan, bools);
            int count = Scatter(n, odata, idata, bools, exclusive_scan);

#if STREAM_COMPACTION_CPU_ENABLE_TIMING
            if (ownsTimer) {
                timer().endCpuTimer();
            }
#endif
            delete[] bools;
            delete[] exclusive_scan;
            return count;
        }
    }
}
