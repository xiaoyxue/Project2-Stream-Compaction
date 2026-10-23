#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include "common.h"
#include "efficient_hardware_recursive.h"

#define TILE_WIDTH 512

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#define CONFLICT_FREE(n) ((n) + CONFLICT_FREE_OFFSET(n))

#ifndef STREAM_COMPACTION_EFFICIENT_HARDWARE_RECURSIVE_ENABLE_TIMING
#define STREAM_COMPACTION_EFFICIENT_HARDWARE_RECURSIVE_ENABLE_TIMING 1
#endif // STREAM_COMPACTION_EFFICIENT_HARDWARE_RECURSIVE_ENABLE_TIMING
namespace StreamCompaction {
    namespace EfficientHardwareRecursive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        std::vector<int*> arrays_0;
        std::vector<int*> arrays_1;
        std::vector<int> arrays_sizes;

        __global__ void kernTileScan(int n, int *odata, int* small_array, const int *large_array) {
            __shared__ int temp[TILE_WIDTH * 2];
            int blockId = blockIdx.x;
            int tid = threadIdx.x;
            int index = blockId * blockDim.x + tid;
            temp[tid * 2] = (index * 2 < n) ? large_array[index * 2] : 0;
            temp[tid * 2 + 1] = (index * 2 + 1 < n) ? large_array[index * 2 + 1] : 0;
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
            if (tid == 0 && small_array) {
                small_array[blockId] = temp[2 * TILE_WIDTH - 1];
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

        __global__ void kernConflictFreeTileScan(int n, int *odata, int* small_array, const int *large_array) {
            __shared__ int temp[CONFLICT_FREE(TILE_WIDTH * 2)];
            int blockId = blockIdx.x;
            int tid = threadIdx.x;
            int index = blockId * blockDim.x + tid;
            temp[CONFLICT_FREE(tid * 2)] = (index * 2 < n) ? large_array[index * 2] : 0;
            temp[CONFLICT_FREE(tid * 2 + 1)] = (index * 2 + 1 < n) ? large_array[index * 2 + 1] : 0;
            __syncthreads();

            // Up-sweep
            for (int offset = 2; offset <= 2 * TILE_WIDTH; offset <<= 1) {
                // int offset = (1 << (d + 1));
                if (tid * 2 % offset == 0 && tid < TILE_WIDTH) {
                    temp[CONFLICT_FREE(tid * 2 + offset - 1)] += temp[CONFLICT_FREE(tid * 2 + (offset >> 1) - 1)];
                }
                __syncthreads();
            }
            // Save the total sum of this tile to the auxiliary array for later reduction
            if (tid == 0 && small_array) {
                small_array[blockId] = temp[CONFLICT_FREE(2 * TILE_WIDTH - 1)];
            }
            // Down-sweep
            if (tid == 0) {
                temp[CONFLICT_FREE(2 * TILE_WIDTH - 1)] = 0;
            }
            __syncthreads();

            for (int offset = 2 * TILE_WIDTH; offset >= 2; (offset >>= 1)) {
                // int offset = (1 << (d + 1));
                if (tid * 2 % offset == 0 && tid < TILE_WIDTH) {
                    int t = temp[CONFLICT_FREE(tid * 2 + (offset >> 1) - 1)];
                    temp[CONFLICT_FREE(tid * 2 + (offset >> 1) - 1)] = temp[CONFLICT_FREE(tid * 2 + offset - 1)];
                    temp[CONFLICT_FREE(tid * 2 + offset - 1)] += t;
                }
                __syncthreads();
            }

            if (index * 2 < n) {
                odata[index * 2] = temp[CONFLICT_FREE(tid * 2)];
            }
            if (index * 2 + 1 < n) {
                odata[index * 2 + 1] = temp[CONFLICT_FREE(tid * 2 + 1)];
            }
        }

        __global__ void kernReduce(int n, int *odata, const int* small_array, const int* large_array) {
            int blockId = blockIdx.x;
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            
            int pre_scan = small_array[blockId];

            if (index < n) {
                odata[index] = large_array[index] + pre_scan;
            }
        }

        void prepareArrays(int n, const int *idata, bool devicePointers) {
            // generate n_power_of_two
            const int two_tile_width = 2 * TILE_WIDTH;
            const int n_power_of_two = (n < two_tile_width) ? two_tile_width : (1 << ilog2ceil(n));
            int power_of_two = n_power_of_two;
            while (power_of_two >= two_tile_width) {
                arrays_sizes.push_back(power_of_two);
                int* d_array;
                cudaMalloc(&d_array, power_of_two * sizeof(int));
                checkCUDAError("cudaMalloc failed for array");
                cudaMemset(d_array, 0, power_of_two * sizeof(int));
                checkCUDAError("cudaMemset failed for array");
                arrays_0.push_back(d_array);
                cudaMalloc(&d_array, power_of_two * sizeof(int));
                checkCUDAError("cudaMalloc failed for array");
                cudaMemset(d_array, 0, power_of_two * sizeof(int));
                checkCUDAError("cudaMemset failed for array");
                arrays_1.push_back(d_array);
                power_of_two = (power_of_two + two_tile_width - 1) / two_tile_width;
				if (power_of_two == 1) {
					break;
				}
                power_of_two = power_of_two < two_tile_width ? two_tile_width : (1 << ilog2ceil(power_of_two));
            }
            if (idata && arrays_0.size() > 0 && arrays_1.size() > 0) {
                cudaMemcpy(arrays_0[0], idata, n * sizeof(int),
                           devicePointers ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice);
                checkCUDAError("cudaMemcpy arrays_0 input failed!");
                cudaMemcpy(arrays_1[0], idata, n * sizeof(int),
                           devicePointers ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice);
                checkCUDAError("cudaMemcpy arrays_1 input failed!");
            }
        }

        void freeArrays() {
            for (int* d_array : arrays_0) {
                cudaFree(d_array);
            }
            for (int* d_array : arrays_1) {
                cudaFree(d_array);
            }
            arrays_0.clear();
            arrays_1.clear();
            arrays_sizes.clear();
        }

        void copyResult(int n, int *odata, bool devicePointers) {
            if (arrays_0.size() > 0) {
                cudaMemcpy(odata, arrays_0[0], n * sizeof(int),
                           devicePointers ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost);
                checkCUDAError("cudaMemcpy scan result failed!");
            }
        }

        void scan(int n, int *odata, const int *idata, bool devicePointers) {
            prepareArrays(n, idata, devicePointers);
#if STREAM_COMPACTION_EFFICIENT_HARDWARE_RECURSIVE_ENABLE_TIMING
            const bool ownsTimer = !timer().isGpuTimerStarted();
            if (ownsTimer) {
                timer().startGpuTimer();
            }
#endif
            for (int i = 0; i < arrays_0.size(); i++) {
                // Call the scan kernel for each array level
                if (i < arrays_0.size() - 1) {
                    // Call the scan kernel for each array level except the last one
                    dim3 blockDim(TILE_WIDTH);
                    dim3 gridDim((arrays_sizes[i] + 2 * TILE_WIDTH - 1) / (2 * TILE_WIDTH));
                    kernTileScan<<<gridDim, blockDim>>>(arrays_sizes[i], arrays_1[i], arrays_0[i + 1], arrays_0[i]);
                    checkCUDAError("kernTileScan failed");
                    std::swap(arrays_0[i], arrays_1[i]);
                }
                else {
                    dim3 blockDim(TILE_WIDTH);
                    dim3 gridDim(1);
                    kernTileScan<<<gridDim, blockDim>>>(arrays_sizes[i], arrays_1[i], nullptr, arrays_0[i]);
                    checkCUDAError("kernTileScan failed");
                    std::swap(arrays_0[i], arrays_1[i]);
                }
            }

            // Call the reduce kernel for each array level except the last one
            for (int i = arrays_0.size() - 2; i >= 0; i--) {
                dim3 blockDim(2 * TILE_WIDTH);
                dim3 gridDim((arrays_sizes[i] + 2 * TILE_WIDTH - 1) / (2 * TILE_WIDTH));
                kernReduce<<<gridDim, blockDim>>>(arrays_sizes[i], arrays_1[i], arrays_0[i + 1], arrays_0[i]);
                checkCUDAError("kernReduce failed");
                std::swap(arrays_0[i], arrays_1[i]);
            }
#if STREAM_COMPACTION_EFFICIENT_HARDWARE_RECURSIVE_ENABLE_TIMING
            if (ownsTimer) {
                timer().endGpuTimer();
            }
#endif
            copyResult(n, odata, devicePointers);
            freeArrays();
        }

        void conflictFreeScan(int n, int *odata, const int *idata, bool devicePointers) {
            prepareArrays(n, idata, devicePointers);
#if STREAM_COMPACTION_EFFICIENT_HARDWARE_RECURSIVE_ENABLE_TIMING
            const bool ownsTimer = !timer().isGpuTimerStarted();
            if (ownsTimer) {
                timer().startGpuTimer();
            }
#endif
            for (int i = 0; i < arrays_0.size(); i++) {
                // Call the scan kernel for each array level
                if (i < arrays_0.size() - 1) {
                    // Call the scan kernel for each array level except the last one
                    dim3 blockDim(TILE_WIDTH);
                    dim3 gridDim((arrays_sizes[i] + 2 * TILE_WIDTH - 1) / (2 * TILE_WIDTH));
                    kernConflictFreeTileScan<<<gridDim, blockDim>>>(arrays_sizes[i], arrays_1[i], arrays_0[i + 1], arrays_0[i]);
                    checkCUDAError("kernConflictFreeTileScan failed");
                    std::swap(arrays_0[i], arrays_1[i]);
                }
                else {
                    dim3 blockDim(TILE_WIDTH);
                    dim3 gridDim(1);
                    kernConflictFreeTileScan<<<gridDim, blockDim>>>(arrays_sizes[i], arrays_1[i], nullptr, arrays_0[i]);
                    checkCUDAError("kernConflictFreeTileScan failed");
                    std::swap(arrays_0[i], arrays_1[i]);
                }
            }

            // Call the reduce kernel for each array level except the last one
            for (int i = arrays_0.size() - 2; i >= 0; i--) {
                dim3 blockDim(2 * TILE_WIDTH);
                dim3 gridDim((arrays_sizes[i] + 2 * TILE_WIDTH - 1) / (2 * TILE_WIDTH));
                kernReduce<<<gridDim, blockDim>>>(arrays_sizes[i], arrays_1[i], arrays_0[i + 1], arrays_0[i]);
                checkCUDAError("kernReduce failed");
                std::swap(arrays_0[i], arrays_1[i]);
            }
#if STREAM_COMPACTION_EFFICIENT_HARDWARE_RECURSIVE_ENABLE_TIMING
            if (ownsTimer) {
                timer().endGpuTimer();
            }
#endif
            copyResult(n, odata, devicePointers);
            freeArrays();
        }

        static int compact(int n, int *odata, const int *idata, bool conflictFree) {
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

#if STREAM_COMPACTION_EFFICIENT_HARDWARE_RECURSIVE_ENABLE_TIMING
            const bool ownsTimer = !timer().isGpuTimerStarted();
            if (ownsTimer) {
                timer().startGpuTimer();
            }
#endif
            if (conflictFree) {
                conflictFreeScan(n, dev_indices, dev_bools, true);
            } else {
                scan(n, dev_indices, dev_bools, true);
            }
            checkCUDAError("scan dev_indices failed!");

            StreamCompaction::Common::kernScatter<<<blocks, threads>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
            cudaDeviceSynchronize();
            checkCUDAError("kernScatter kernel execution failed!");

#if STREAM_COMPACTION_EFFICIENT_HARDWARE_RECURSIVE_ENABLE_TIMING
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

        int compact(int n, int *odata, const int *idata) {
            return compact(n, odata, idata, false);
        }

        int conflictFreeCompact(int n, int *odata, const int *idata) {
            return compact(n, odata, idata, true);
        }
    }
}
