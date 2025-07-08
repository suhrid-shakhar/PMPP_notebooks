#include <stdio.h>
#include <cuda_runtime.h>

#define cudaCheck(call)                                                            \
    {                                                                              \
        cudaError_t err = call;                                                    \
        if (err != cudaSuccess)                                                    \
        {                                                                          \
            printf("%s at %s: %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        }                                                                          \
    }

#define LENGTH 1024
#define BLOCK_SIZE 512

__global__ void parallelReductionKernel(float *input, float *output)
{
    unsigned int i = threadIdx.x * 2;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        if(threadIdx.x % stride == 0)
        {
            input[i] += input [i+stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0)
        *output = input[0];
}
__global__ void initData(float *data)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    while (idx < LENGTH) 
    {
        data[idx] = 2.5f;
        idx += stride;
    }
}

void cpuReduction(float *input, float *output)
{
    float sum = 0.0f;
    for(int i=0; i<LENGTH; i++)
        sum += input[i];
    *output = sum;
}

int main(int argc, char* argv[])
{
    float *data_h, *data_d, *result_h, *result_d, result_cpu;
    int size = sizeof(float) * LENGTH;
    cudaStream_t memcpy;
    cudaCheck(cudaStreamCreateWithFlags(&memcpy, cudaStreamNonBlocking));

    cudaCheck(cudaMalloc(&data_d, size));
    cudaCheck(cudaMallocHost(&data_h, size));

    initData<<<(LENGTH + BLOCK_SIZE -1)/BLOCK_SIZE, BLOCK_SIZE>>>(data_d);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpyAsync(data_h, data_d, size, cudaMemcpyDeviceToHost, memcpy));

    cudaCheck(cudaMalloc(&result_d, sizeof(float)));
    cudaCheck(cudaMallocHost(&result_h, sizeof(float)));

    dim3 block(BLOCK_SIZE);
    // int gridSize = (LENGTH + block.x - 1)/ block.x;
    dim3 grid(1);
    parallelReductionKernel<<<grid, block>>>(data_d, result_d);
    cudaCheck(cudaDeviceSynchronize());
    cpuReduction(data_h, &result_cpu);
    cudaCheck(cudaMemcpy(result_h, result_d, sizeof(float), cudaMemcpyDeviceToHost));

    if(result_cpu == *result_h) 
        printf("GPU value matches CPU value\n");
    else 
        printf("CPU and GPU value do not match. CPU = %f, GPU = %f.\n", result_cpu, *result_h);
    
    cudaCheck(cudaFree(data_d));
    cudaCheck(cudaFree(result_d));
    cudaCheck(cudaFreeHost(data_h));
    cudaCheck(cudaFreeHost(result_h));

    return 0;
}