#include <cuda.h>
#include <cuda_runtime.h>
#include <utility>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernNaiveScan(int n, int offset, int *odata, const int *idata) {
            int k = threadIdx.x + blockIdx.x * blockDim.x;
            if (k >= n) {
                return;
            }
            odata[k] = idata[k];
            if (k >= offset) {
                odata[k] += idata[k - offset];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            const size_t bytes = static_cast<size_t>(n) * sizeof(int);
            int *gpuInput;
            int *gpuOutput;
            cudaMalloc(&gpuInput, bytes);
            checkCUDAError("Naive::scan input allocation");
            cudaMalloc(&gpuOutput, bytes);
            checkCUDAError("Naive::scan output allocation");

            // Shift right and insert zero so the scan produces exclusive sums.
            cudaMemset(gpuInput, 0, sizeof(int));
            checkCUDAError("Naive::scan first element initialization");

            if (n > 1) {
                cudaMemcpy(gpuInput + 1, idata, bytes - sizeof(int), cudaMemcpyHostToDevice);
                checkCUDAError("Naive::scan input copy");
            }
            timer().startGpuTimer();

            const dim3 threads(256);
            const dim3 blocks(1 + (n - 1) / 256);
            const int levels = ilog2ceil(n);
            checkCUDAError("Naive::scan timer start");
            for (int d = 1; d <= levels; ++d) {
                kernNaiveScan<<<blocks, threads>>>(n, 1 << (d - 1), gpuOutput, gpuInput);
                checkCUDAError("Naive::scan kernel launch");
                std::swap(gpuInput, gpuOutput);
            }
            timer().endGpuTimer();


            cudaMemcpy(odata, gpuInput, bytes, cudaMemcpyDeviceToHost);
            checkCUDAError("Naive::scan output copy");
            cudaFree(gpuInput);
            checkCUDAError("Naive::scan input release");
            cudaFree(gpuOutput);
            checkCUDAError("Naive::scan output release");
            
        }
    }
}
