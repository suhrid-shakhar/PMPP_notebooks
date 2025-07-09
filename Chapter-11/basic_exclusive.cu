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

#define LENGTH 512
#define SECTION_SIZE LENGTH //uses single grid to do scan, i.e., SECTION_SIZE = LENGTH

void cpuSequentialScan(float *input, float *output, unsigned int N)
{
    output[0] = 0;//input[0];
    for(int i=0; i< N; i++)
    {
        output[i+1] = output[i] + input[i];
    }
}

__global__ void Kogge_Stone_Scan_Kernel(float *in, float *out, unsigned int N)
{
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N && threadIdx.x != 0)
    {
        XY[threadIdx.x] = in[i-1];
    }
    else {
        XY[threadIdx.x] = 0.0f;
    }
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();
        float temp;
        if (threadIdx.x >= stride)
        {
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        }
        __syncthreads();
        if(threadIdx.x >= stride)
            XY[threadIdx.x] = temp;
    }
    if(i < N)
    {
        out[i] = XY[threadIdx.x];
    }
}
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
int main(int argc, char* argv[])
{
    float *in_h, *out_h, *in_d, *out_d, *out_cpu;

    float memSize = (LENGTH + 1) * sizeof(float);

    cudaCheck(cudaMalloc(&in_d, memSize));
    cudaCheck(cudaMallocHost(&in_h, memSize));

    cudaCheck(cudaMalloc(&out_d, memSize));
    cudaCheck(cudaMallocHost(&out_h, memSize));

    cudaCheck(cudaMallocHost(&out_cpu, memSize));

    dim3 block(SECTION_SIZE);
    dim3 grid((LENGTH + SECTION_SIZE - 1)/SECTION_SIZE);
    initFloat<<<grid, block>>>(in_d, LENGTH);
    cudaCheck(cudaDeviceSynchronize());
    
    Kogge_Stone_Scan_Kernel<<<grid, block>>>(in_d, out_d, LENGTH);
    cudaCheck(cudaMemcpy(in_h, in_d, memSize, cudaMemcpyDeviceToHost));
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(out_h, out_d, memSize, cudaMemcpyDeviceToHost));
    cpuSequentialScan(in_h, out_cpu, LENGTH);

    bool flag = true;
    for(int i=0; i< LENGTH; i++)
    {
        if(out_cpu[i] != out_h[i])
        {
            printf("Value mismatch at %d. CPU: %f, GPU: %f\n", i, out_cpu[i], out_h[i]);
            flag = false;
            // break;
        }
    }
    if(flag)
        printf("GPU value matched CPU values.\n");

    cudaCheck(cudaFree(in_d));
    cudaCheck(cudaFree(out_d));
    cudaCheck(cudaFreeHost(in_h));
    cudaCheck(cudaFreeHost(out_h));
    cudaCheck(cudaFreeHost(out_cpu));
    
    return 0;
}