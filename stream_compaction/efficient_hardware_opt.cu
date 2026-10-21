#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient_hardware_opt.h"

#ifndef STREAM_COMPACTION_EFFICIENT_HARDWARE_OPT_ENABLE_TIMING
#define STREAM_COMPACTION_EFFICIENT_HARDWARE_OPT_ENABLE_TIMING 1
#endif

#define TILE_WIDTH 256

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
            temp[tid * 2] = (index < n) ? idata[index * 2] : 0;
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

            odata[index * 2] = temp[tid * 2];
            if (index * 2 + 1 < n) {
                odata[index * 2 + 1] = temp[tid * 2 + 1];
            }
        }

        __global__ void kernReduce(int n, int *odata, const int* auxiliary_array) {
            int blockId = blockIdx.x;
            int tid = threadIdx.x;
            int index = blockId * (2 * TILE_WIDTH) + 2 * tid;
            
            int pre_scan = auxiliary_array[blockId];
            
            if (index < n) {
                odata[index] += pre_scan;
            }
            if (index + 1 < n) {
                odata[index + 1] += pre_scan;
            }
        }

        void scan(int n, int *odata, const int *idata, bool devicePointers) {
            
            int* dev_idata;
            int* dev_odata;
            int* dev_auxiliary_array;

            int power_of_two = (n < 2 * TILE_WIDTH) ? 2 * TILE_WIDTH : (1 << ilog2ceil(n));

            cudaMalloc((void**)&dev_idata, power_of_two * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata");
            cudaMalloc((void**)&dev_odata, power_of_two * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata");
            cudaMalloc((void**)&dev_auxiliary_array, ((power_of_two + TILE_WIDTH * 2 - 1) / (TILE_WIDTH * 2)) * sizeof(int));
            checkCUDAError("cudaMalloc dev_auxiliary_array");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_idata");
            cudaMemset(dev_odata, 0, power_of_two * sizeof(int));
            checkCUDAError("cudaMemset dev_odata");
            cudaMemset(dev_auxiliary_array, 0, ((power_of_two + TILE_WIDTH * 2 - 1) / (TILE_WIDTH * 2)) * sizeof(int));
            checkCUDAError("cudaMemset dev_auxiliary_array");


            timer().startGpuTimer();
            dim3 blockSize(TILE_WIDTH);
            dim3 gridSize((power_of_two + TILE_WIDTH * 2 - 1) / (TILE_WIDTH * 2));
            kernTileScan<<<gridSize, blockSize>>>(power_of_two, dev_odata, dev_auxiliary_array, dev_idata);
            checkCUDAError("kernTileScan");
            cudaDeviceSynchronize();
            checkCUDAError("cudaDeviceSynchronize");

            if (gridSize.x > 1) {
                dim3 auxiliaryBlockSize(gridSize.x);
                dim3 auxiliaryGridSize(1);
               if (gridSize.x > TILE_WIDTH) {
                auxiliaryBlockSize = dim3(TILE_WIDTH);
                auxiliaryGridSize = dim3((gridSize.x + TILE_WIDTH - 1) / TILE_WIDTH);
               }
               kernTileScan<<<auxiliaryGridSize, auxiliaryBlockSize>>>(gridSize.x, dev_auxiliary_array, nullptr, dev_auxiliary_array);
               checkCUDAError("kernTileScan auxiliary");
               kernReduce<<<gridSize, blockSize>>>(power_of_two, dev_odata, dev_auxiliary_array);
               checkCUDAError("kernReduce");
               cudaDeviceSynchronize();
               checkCUDAError("cudaDeviceSynchronize");
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata");

            cudaFree(dev_idata);
            checkCUDAError("cudaFree dev_idata");
            cudaFree(dev_odata);
            checkCUDAError("cudaFree dev_odata");
            cudaFree(dev_auxiliary_array);
            checkCUDAError("cudaFree dev_auxiliary_array");
            
        }

        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
