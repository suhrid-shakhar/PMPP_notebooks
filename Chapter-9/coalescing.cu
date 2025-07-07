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

#define NUM_BINS 7
#define BLOCK_SIZE 256
#define CFACTOR 4

#define LENGTH 20480

__global__ void private_histo_kernel(char *data, unsigned int length, unsigned int *histogram)
{
    __shared__ unsigned int histro_s[NUM_BINS];
    for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
    {
        histro_s[bin] = 0;
    }
    __syncthreads();
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for ( unsigned int i = tid; i < length; i+= blockDim.x * gridDim.x)
    {
        int alphabet_position = data[i] - 'a';
        if(alphabet_position >=0 && alphabet_position < 26)
        {
            atomicAdd(&histro_s[alphabet_position/4], 1);
        }
    }
    __syncthreads();
    for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
    {
        unsigned int binValue = histro_s[bin];
        if(binValue > 0) 
        {
            atomicAdd(&histogram[bin], binValue);
        }
    }
}

void cpu_histogram(const char *data, int length, unsigned int *histogram)
{
    for (int i = 0; i < length; i++)
    {
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26)
        {
            histogram[pos / 4]++;
        }
    }
}

void generateRandomString(char *data, int length)
{
    for(int i=0; i<length; i++)
        data[i] = rand()%26 + 'a';
}

int main(int argc, char* argv[])
{
    unsigned int length = LENGTH;
    char *data_h;
    data_h = (char *)malloc(sizeof(char)*length + 1);
    generateRandomString(data_h, length);

    int numBlocks = (length + (BLOCK_SIZE*CFACTOR) - 1) / (BLOCK_SIZE * CFACTOR);
    size_t histoSize = NUM_BINS * numBlocks * sizeof(unsigned int);
    char *data_d;
    unsigned int *histogram_d, *gpu_histogram_h, *cpu_histogram_h;

    cudaCheck(cudaMallocHost(&gpu_histogram_h, histoSize));
    cudaCheck(cudaMallocHost(&cpu_histogram_h, histoSize));

    cudaCheck(cudaMalloc((void **)&data_d, length * sizeof(char)));
    cudaCheck(cudaMalloc((void **)&histogram_d, histoSize));
    cudaCheck(cudaMemset(histogram_d, 0, histoSize));
    cudaCheck(cudaMemcpy(data_d, data_h, length * sizeof(char), cudaMemcpyHostToDevice));

    private_histo_kernel<<<numBlocks, BLOCK_SIZE>>>(data_d, length, histogram_d);
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(gpu_histogram_h, histogram_d, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    cpu_histogram(data_h, length, cpu_histogram_h);

    bool flag = true;
    for(int i=0; i< NUM_BINS; i++)
    {
        if(cpu_histogram_h[i] != gpu_histogram_h[i]) 
        {
            printf("Value mismatched at %d, cpu: %u, gpu:%u\n", i, cpu_histogram_h[i], gpu_histogram_h[i]);
            flag = false;
            break;
        }
    }
    if(flag)
        printf("GPU value matches CPU values.\n");
    cudaCheck(cudaFree(data_d));
    cudaCheck(cudaFree(histogram_d));
    cudaCheck(cudaFreeHost(gpu_histogram_h));
    cudaCheck(cudaFreeHost(cpu_histogram_h));

    return 0;
}