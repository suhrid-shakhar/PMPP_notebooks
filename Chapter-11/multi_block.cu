#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define cudaCheck(call)                                                            \
    {                                                                              \
        cudaError_t err = call;                                                    \
        if (err != cudaSuccess)                                                    \
        {                                                                          \
            printf("%s at %s: %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    }

#define LENGTH 256
#define SECTION_SIZE 128

// CPU Exclusive Scan
void cpuSequentialScan(float *input, float *output, unsigned int N)
{
    output[0] = 0.0f;
    for (int i = 1; i <= N; i++)
    {
        output[i] = output[i - 1] + input[i - 1];
    }
}

// Initialize input array
__global__ void initFloat(float *arr, unsigned int size)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    while (idx < size)
    {
        arr[idx] = 2.0f;
        idx += stride;
    }
}

// Kogge-Stone per-block inclusive scan
__global__ void Kogge_Stone_Scan_Kernel(float *in, float *out, float *blockSum, unsigned int N)
{
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N)
        XY[threadIdx.x] = in[i];
    else
        XY[threadIdx.x] = 0.0f;

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();
        float temp = 0.0f;
        if (threadIdx.x >= stride)
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        __syncthreads();
        if (threadIdx.x >= stride)
            XY[threadIdx.x] = temp;
    }

    __syncthreads();

    if (i < N)
        out[i] = XY[threadIdx.x];

    if (threadIdx.x == blockDim.x - 1)
        blockSum[blockIdx.x] = XY[threadIdx.x];
}

// Kogge-Stone scan of block sums (exclusive)
__global__ void Kogge_Stone_Second_Scan_Kernel(float *in, float *out, unsigned int N)
{
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = threadIdx.x;

    if (i == 0)
        XY[i] = 0.0f;
    else if (i < N)
        XY[i] = in[i - 1];
    else
        XY[i] = 0.0f;

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();
        float temp = 0.0f;
        if (i >= stride)
            temp = XY[i] + XY[i - stride];
        __syncthreads();
        if (i >= stride)
            XY[i] = temp;
    }

    __syncthreads();

    if (i < N)
        out[i] = XY[i];
}

// Add scanned block sums to scanned data
__global__ void secondary_block_sum_kernel(float *out, float *blockSum, unsigned int N)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        out[i] += blockSum[blockIdx.x];
    }
}

int main()
{
    float *in_h, *out_h, *out_cpu;
    float *in_d, *out_d, *blockSum_d, *blockExclusiveSum_d;
    float *blockSum_h;

    unsigned int memSize = LENGTH * sizeof(float);
    unsigned int numBlocks = (LENGTH + SECTION_SIZE - 1) / SECTION_SIZE;

    // Host allocations
    cudaCheck(cudaMallocHost(&in_h, memSize));
    cudaCheck(cudaMallocHost(&out_h, memSize));
    cudaCheck(cudaMallocHost(&out_cpu, (LENGTH + 1) * sizeof(float)));
    blockSum_h = (float *)malloc(sizeof(float) * numBlocks);

    // Device allocations
    cudaCheck(cudaMalloc(&in_d, memSize));
    cudaCheck(cudaMalloc(&out_d, memSize));
    cudaCheck(cudaMalloc(&blockSum_d, sizeof(float) * numBlocks));
    cudaCheck(cudaMalloc(&blockExclusiveSum_d, sizeof(float) * numBlocks));

    dim3 block(SECTION_SIZE);
    dim3 grid(numBlocks);

    // Initialize device input
    initFloat<<<grid, block>>>(in_d, LENGTH);
    cudaCheck(cudaDeviceSynchronize());

    // Step 1: Scan each block
    Kogge_Stone_Scan_Kernel<<<grid, block>>>(in_d, out_d, blockSum_d, LENGTH);
    cudaCheck(cudaDeviceSynchronize());

    // Step 2: Scan block sums
    Kogge_Stone_Second_Scan_Kernel<<<1, SECTION_SIZE>>>(blockSum_d, blockExclusiveSum_d, numBlocks);
    cudaCheck(cudaDeviceSynchronize());

    // Step 3: Apply scanned block sums to each section
    secondary_block_sum_kernel<<<grid, block>>>(out_d, blockExclusiveSum_d, LENGTH);
    cudaCheck(cudaDeviceSynchronize());

    // Copy results back
    cudaCheck(cudaMemcpy(in_h, in_d, memSize, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(out_h, out_d, memSize, cudaMemcpyDeviceToHost));

    // CPU scan for comparison
    cpuSequentialScan(in_h, out_cpu, LENGTH);

    // Compare results
    bool flag = true;
    for (int i = 0; i < LENGTH; i++)
    {
        if (fabs(out_cpu[i + 1] - out_h[i]) > 1e-3f)
        {
            printf("Mismatch at %d: CPU = %f, GPU = %f\n", i, out_cpu[i + 1], out_h[i]);
            flag = false;
        }
    }

    if (flag)
        printf("GPU result matches CPU scan!\n");
    else
        printf("GPU result does not match.\n");

    // Cleanup
    cudaFree(in_d);
    cudaFree(out_d);
    cudaFree(blockSum_d);
    cudaFree(blockExclusiveSum_d);
    cudaFreeHost(in_h);
    cudaFreeHost(out_h);
    cudaFreeHost(out_cpu);
    free(blockSum_h);

    return 0;
}