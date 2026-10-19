#include <cuda.h>
#include <cuda_runtime.h>
#include <utility>
#include "common.h"
#include "efficient.h"

#ifndef STREAM_COMPACTION_EFFICIENT_ENABLE_TIMING
#define STREAM_COMPACTION_EFFICIENT_ENABLE_TIMING 1
#endif

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // offset = 2^(d+1)
        __global__ void kernUpSweep(int n, int offset, int *data) {
            // TODO: Implement the efficient scan kernel
            int k = threadIdx.x + blockIdx.x * blockDim.x;
            if ( (k < n) && (k % offset == 0) ) {
                data[k + offset - 1] += data[k + offset / 2 - 1];
            }
        }

        // offset = 2^(d+1)
        __global__ void kernDownSweep(int n, int offset, int *data, const int *idata) {
            // TODO: Implement the efficient scan kernel for down-sweep
            int k = threadIdx.x + blockIdx.x * blockDim.x;
            if ( (k < n) && (k % offset == 0) ) {
                int t = idata[k + (offset >> 1) - 1];
                data[k + (offset >> 1) - 1] = idata[k + offset - 1];
                data[k + offset - 1] += t;
            }
        }

        void upSweep(int n, int *data) {
            // TODO: Implement the up-sweep phase of the efficient scan
            dim3 threads(256);
            dim3 blocks(1 + (n - 1) / 256);
            int level = ilog2ceil(n);
            for (int d = 0; d <= level - 1; ++d) {
                int offset = (1 << (d + 1));
                kernUpSweep<<<blocks, threads>>>(n, offset, data);
                cudaDeviceSynchronize();
            }
            cudaDeviceSynchronize();
            checkCUDAError("upSweep kernel execution failed!");
        }

        void downSweep(int n, int *data) {
            // TODO: Implement the down-sweep phase of the efficient scan
            dim3 threads(256);
            dim3 blocks(1 + (n - 1) / 256);
            int level = ilog2ceil(n);

            // x[n - 1] = 0;
            cudaMemset(data + n - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset data + n - 1 failed!");
            
            for (int d = level - 1; d >= 0; --d) {
                int offset = (1 << (d + 1));
                kernDownSweep<<<blocks, threads>>>(n, offset, data, data);
                cudaDeviceSynchronize();
            }
            cudaDeviceSynchronize();
            checkCUDAError("downSweep kernel execution failed!");
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, bool devicePointers) {
            // TODO
            int power_of_two = (1 << ilog2ceil(n));

            const size_t bytes = static_cast<size_t>(power_of_two) * sizeof(int);
            int *dev_data;
            cudaMalloc((void**)&dev_data, bytes);
            checkCUDAError("cudaMalloc dev_data failed!");

            cudaMemset(dev_data, 0, bytes);
            checkCUDAError("cudaMemset dev_data failed!");

            cudaMemcpy(dev_data, idata, n * sizeof(int),
                       devicePointers ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_data failed!");

#if STREAM_COMPACTION_EFFICIENT_ENABLE_TIMING
            const bool ownsTimer = !timer().isGpuTimerStarted();
            if (ownsTimer) {
                timer().startGpuTimer();
            }
#endif
            upSweep(power_of_two, dev_data);

            downSweep(power_of_two, dev_data);

#if STREAM_COMPACTION_EFFICIENT_ENABLE_TIMING
            if (ownsTimer) {
                timer().endGpuTimer();
            }
#endif

            cudaMemcpy(odata, dev_data, n * sizeof(int),
                       devicePointers ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy scan output failed!");
            cudaFree(dev_data);
            checkCUDAError("cudaFree dev_data failed!");




        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            // TODO
            int *dev_bools;
            int *dev_idata;
            int *dev_odata;
            int *indices = new int[n];
            
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_idata failed!");

            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");

            dim3 threads(256);
            dim3 blocks(1 + (n - 1) / 256);
            StreamCompaction::Common::kernMapToBoolean<<<blocks, threads>>>(n, dev_bools, dev_idata);
            cudaDeviceSynchronize();
            checkCUDAError("kernMapToBoolean kernel execution failed!");

            int *dev_indices;
            cudaMalloc((void**)&dev_indices, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");

            cudaMemset(dev_indices, 0, n * sizeof(int));
            checkCUDAError("cudaMemset dev_indices failed!");

#if STREAM_COMPACTION_EFFICIENT_ENABLE_TIMING
            const bool ownsTimer = !timer().isGpuTimerStarted();
            if (ownsTimer) {
                timer().startGpuTimer();
            }
#endif
            scan(n, dev_indices, dev_bools, true);
            checkCUDAError("scan dev_indices failed!");

            StreamCompaction::Common::kernScatter<<<blocks, threads>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
            cudaDeviceSynchronize();
            checkCUDAError("kernScatter kernel execution failed!");

#if STREAM_COMPACTION_EFFICIENT_ENABLE_TIMING
            if (ownsTimer) {
                timer().endGpuTimer();
            }
#endif
            int count;
            cudaMemcpy(&count, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy count failed!");
            count += (idata[n - 1] != 0);

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_odata failed!");

            cudaFree(dev_indices);
            checkCUDAError("cudaFree dev_indices failed!");

            cudaFree(dev_bools);
            checkCUDAError("cudaFree dev_bools failed!");

            cudaFree(dev_idata);
            checkCUDAError("cudaFree dev_idata failed!");

            cudaFree(dev_odata);
            checkCUDAError("cudaFree dev_odata failed!");

            return count;
        }
    }
}
