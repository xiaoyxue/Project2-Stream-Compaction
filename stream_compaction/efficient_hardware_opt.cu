#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include "common.h"
#include "efficient.h"
#include "efficient_hardware_opt.h"

#ifndef STREAM_COMPACTION_EFFICIENT_HARDWARE_OPT_ENABLE_TIMING
#define STREAM_COMPACTION_EFFICIENT_HARDWARE_OPT_ENABLE_TIMING 1
#endif

#define TILE_WIDTH 512

namespace StreamCompaction {
    namespace EfficientHardwareOpt {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernTileScan(int n, int *odata, int* auxiliary_array, const int *idata) {
            __shared__ int temp[TILE_WIDTH * 2];
            int blockId = blockIdx.x;
            int tid = threadIdx.x;
            int index = blockId * blockDim.x + tid;
            temp[tid * 2] = (index * 2 < n) ? idata[index * 2] : 0;
            temp[tid * 2 + 1] = (index * 2 + 1 < n) ? idata[index * 2 + 1] : 0;
            __syncthreads();

            // Up-sweep
            for (int offset = 2; offset <= 2 * TILE_WIDTH; offset <<= 1) {
                // int offset = (1 << (d + 1));
                if (tid * 2 % offset == 0 && tid < TILE_WIDTH) {
                    temp[tid * 2 + offset - 1] += temp[tid * 2 + (offset >> 1) - 1];
                }
                __syncthreads();
            }
            // Save the total sum of this tile to the auxiliary array for later reduction
            if (tid == 0 && auxiliary_array) {
                auxiliary_array[blockId] = temp[2 * TILE_WIDTH - 1];
            }
            // Down-sweep
            if (tid == 0) {
                temp[2 * TILE_WIDTH - 1] = 0;
            }
            __syncthreads();

            for (int offset = 2 * TILE_WIDTH; offset >= 2; (offset >>= 1)) {
                // int offset = (1 << (d + 1));
                if (tid * 2 % offset == 0 && tid < TILE_WIDTH) {
                    int t = temp[tid * 2 + (offset >> 1) - 1];
                    temp[tid * 2 + (offset >> 1) - 1] = temp[tid * 2 + offset - 1];
                    temp[tid * 2 + offset - 1] += t;
                }
                __syncthreads();
            }

            if (index * 2 < n) {
                odata[index * 2] = temp[tid * 2];
            }
            if (index * 2 + 1 < n) {
                odata[index * 2 + 1] = temp[tid * 2 + 1];
            }
        }

        __global__ void kernReduce(int n, int *odata, const int* auxiliary_array) {
            int blockId = blockIdx.x;
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            
            int pre_scan = auxiliary_array[blockId];

            if (index < n) {
                odata[index] += pre_scan;
            }
        }

        // Device pointers; n is a positive power of two. No separate timer.
        void efficientScan(int n, int *odata, const int *idata) {
            if (odata != idata) {
                cudaMemcpy(odata, idata, static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToDevice);
                checkCUDAError("cudaMemcpy efficientScan");
            }

            dim3 threads(TILE_WIDTH);
            int level = ilog2ceil(n);
            for (int d = 0; d <= level - 1; ++d) {
                int offset = (1 << (d + 1));
                int activeThreads = n / offset;
                dim3 blocks((activeThreads + threads.x - 1) / threads.x);
                Efficient::kernUpSweep<<<blocks, threads>>>(n, offset, odata);
                checkCUDAError("upSweep kernel launch failed!");
            }
            cudaDeviceSynchronize();
            checkCUDAError("upSweep kernel execution failed!");

            cudaMemset(odata + n - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset odata + n - 1 failed!");

            for (int d = level - 1; d >= 0; --d) {
                int offset = (1 << (d + 1));
                int activeThreads = n / offset;
                dim3 blocks((activeThreads + threads.x - 1) / threads.x);
                Efficient::kernDownSweep<<<blocks, threads>>>(n, offset, odata, odata);
                checkCUDAError("downSweep kernel launch failed!");
            }
            cudaDeviceSynchronize();
            checkCUDAError("downSweep kernel execution failed!");
        }

        void scan(int n, int *odata, const int *idata, bool devicePointers) {
            if (n < 0) {
                throw std::invalid_argument("EfficientHardwareOpt::scan requires n >= 0");
            }
            if (n == 0) {
                return;
            }

            int* dev_idata;
            int* dev_odata;
            int* dev_auxiliary_array;

            const int n_power_of_two = (n < 2 * TILE_WIDTH) ? 2 * TILE_WIDTH : (1 << ilog2ceil(n));

            const int auxiliary_array_size = (n_power_of_two + TILE_WIDTH * 2 - 1) / (TILE_WIDTH * 2);

            const int auxiliary_array_size_power_of_two = (auxiliary_array_size < 2 * TILE_WIDTH) ? 2 * TILE_WIDTH : (1 << ilog2ceil(auxiliary_array_size));

            cudaMalloc((void**)&dev_idata, n_power_of_two * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata");
            cudaMalloc((void**)&dev_odata, n_power_of_two * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata");
            cudaMalloc((void**)&dev_auxiliary_array, auxiliary_array_size_power_of_two * sizeof(int));
            checkCUDAError("cudaMalloc dev_auxiliary_array");

            cudaMemset(dev_idata, 0, n_power_of_two * sizeof(int));
            checkCUDAError("cudaMemset dev_idata");
            cudaMemcpy(dev_idata, idata, n * sizeof(int),
                       devicePointers ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_idata");
            cudaMemset(dev_odata, 0, n_power_of_two * sizeof(int));
            checkCUDAError("cudaMemset dev_odata");
            cudaMemset(dev_auxiliary_array, 0, auxiliary_array_size_power_of_two * sizeof(int));
            checkCUDAError("cudaMemset dev_auxiliary_array");


#if STREAM_COMPACTION_EFFICIENT_HARDWARE_OPT_ENABLE_TIMING
            const bool ownsTimer = !timer().isGpuTimerStarted();
            if (ownsTimer) {
                timer().startGpuTimer();
            }
#endif
            dim3 blockSize(TILE_WIDTH);
            dim3 gridSize((n_power_of_two + TILE_WIDTH * 2 - 1) / (TILE_WIDTH * 2));
            kernTileScan<<<gridSize, blockSize>>>(n_power_of_two, dev_odata, dev_auxiliary_array, dev_idata);
            checkCUDAError("kernTileScan");
            cudaDeviceSynchronize();
            checkCUDAError("cudaDeviceSynchronize");

            if (gridSize.x > 1) {
                dim3 auxiliaryBlockSize(TILE_WIDTH);
                dim3 auxiliaryGridSize(1);
                if (gridSize.x > 2 * TILE_WIDTH) {
                    efficientScan(auxiliary_array_size_power_of_two, dev_auxiliary_array, dev_auxiliary_array);
                    checkCUDAError("efficientScan auxiliary");
                } 
                else {
                    kernTileScan<<<auxiliaryGridSize, auxiliaryBlockSize>>>(auxiliary_array_size_power_of_two, dev_auxiliary_array, nullptr, dev_auxiliary_array);
                    checkCUDAError("kernTileScan auxiliary");
                }
               dim3 reduceBlockSize(2 * TILE_WIDTH);
               dim3 reduceGridSize((n_power_of_two + 2 * TILE_WIDTH - 1) / (2 * TILE_WIDTH));
               kernReduce<<<reduceGridSize, reduceBlockSize>>>(n_power_of_two, dev_odata, dev_auxiliary_array);
               checkCUDAError("kernReduce");
               cudaDeviceSynchronize();
               checkCUDAError("cudaDeviceSynchronize");
            }
#if STREAM_COMPACTION_EFFICIENT_HARDWARE_OPT_ENABLE_TIMING
            if (ownsTimer) {
                timer().endGpuTimer();
            }
#endif

            cudaMemcpy(odata, dev_odata, n * sizeof(int),
                       devicePointers ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata");

            cudaFree(dev_idata);
            checkCUDAError("cudaFree dev_idata");
            cudaFree(dev_odata);
            checkCUDAError("cudaFree dev_odata");
            cudaFree(dev_auxiliary_array);
            checkCUDAError("cudaFree dev_auxiliary_array");
            
        }

        int compact(int n, int *odata, const int *idata) {
            if (n < 0) {
                throw std::invalid_argument("EfficientHardwareOpt::compact requires n >= 0");
            }
            if (n == 0) {
                return 0;
            }

            int *dev_bools;
            int *dev_idata;
            int *dev_odata;
            int *dev_indices;
            const size_t bytes = static_cast<size_t>(n) * sizeof(int);

            cudaMalloc((void**)&dev_odata, bytes);
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_idata, bytes);
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMemcpy(dev_idata, idata, bytes, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_idata failed!");

            cudaMalloc((void**)&dev_bools, bytes);
            checkCUDAError("cudaMalloc dev_bools failed!");

            dim3 threads(TILE_WIDTH);
            dim3 blocks(1 + (n - 1) / TILE_WIDTH);
            StreamCompaction::Common::kernMapToBoolean<<<blocks, threads>>>(n, dev_bools, dev_idata);
            cudaDeviceSynchronize();
            checkCUDAError("kernMapToBoolean kernel execution failed!");

            cudaMalloc((void**)&dev_indices, bytes);
            checkCUDAError("cudaMalloc dev_indices failed!");
            cudaMemset(dev_indices, 0, bytes);
            checkCUDAError("cudaMemset dev_indices failed!");

#if STREAM_COMPACTION_EFFICIENT_HARDWARE_OPT_ENABLE_TIMING
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

#if STREAM_COMPACTION_EFFICIENT_HARDWARE_OPT_ENABLE_TIMING
            if (ownsTimer) {
                timer().endGpuTimer();
            }
#endif

            int count;
            cudaMemcpy(&count, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy count failed!");
            count += (idata[n - 1] != 0);

            if (count > 0) {
                cudaMemcpy(odata, dev_odata, static_cast<size_t>(count) * sizeof(int), cudaMemcpyDeviceToHost);
                checkCUDAError("cudaMemcpy dev_odata failed!");
            }

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
