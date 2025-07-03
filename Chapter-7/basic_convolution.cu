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

#define MATRIX_HEIGHT 2048
#define MATRIX_WIDTH 2048

#define FILTER_RADIUS 2

// Initializes array with simple test values
__global__ void initData(float *data, int size, float value = 1.0f)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_y = blockDim.x * gridDim.x;
    int stride_x = blockDim.y * gridDim.y;
    while (col < size && row < size)
    {
        data[row * size + col] = value;
        row += stride_x;
        col += stride_y;
    }
}
void print(float *data, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            printf("(%d, %d) = %f\t", i, j, data[i * size + j]);
        }
    }
}

__global__ void convolution_2D_basic_kernel(float *N, float *F, float *P, int r, int width, int height)
{
    int outCol = threadIdx.x + blockIdx.x * blockDim.x;
    int outRow = threadIdx.y + blockIdx.y * blockDim.y;
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * r + 1; fRow++)
    {
        for (int fCol = 0; fCol < 2 * r + 1; fCol++)
        {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
            {
                Pvalue += F[fRow * (2 * r + 1) + fCol] * N[inRow * width + inCol];
            }
        }
    }
    P[outRow * width + outCol] = Pvalue;
}

void verify_convolution(float *input, float *filter, float *gpu_output, int radius, int width, int height)
{
    bool correct = true;
    int filterSize = 2 * radius + 1;
    float *cpu_output = (float *)malloc(width * height * sizeof(float));

    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            float val = 0.0f;
            for (int fRow = 0; fRow < filterSize; fRow++)
            {
                for (int fCol = 0; fCol < filterSize; fCol++)
                {
                    int inRow = row - radius + fRow;
                    int inCol = col - radius + fCol;
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                    {
                        val += filter[fRow * filterSize + fCol] * input[inRow * width + inCol];
                    }
                }
            }
            cpu_output[row * width + col] = val;

            float gpu_val = gpu_output[row * width + col];
            if (fabs(cpu_output[row * width + col] - gpu_val) > 1e-4)
            {
                printf("Mismatch at (%d, %d): CPU = %f, GPU = %f\n", row, col, cpu_output[row * width + col], gpu_val);
                correct = false;
            }
        }
    }

    if (correct)
        printf("GPU output matches CPU output!\n");
    else
        printf("Mismatch found between GPU and CPU results.\n");

    free(cpu_output);
}

int main(int argc, char *argv[])
{
    int matrixSize = MATRIX_HEIGHT * MATRIX_WIDTH * sizeof(float);
    int filterSize = (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * sizeof(float);

    float *matrix_h, *filter_h, *output_h;
    float *matrix_d, *filter_d, *output_d;

    cudaCheck(cudaMallocHost(&matrix_h, matrixSize));
    cudaCheck(cudaMalloc(&matrix_d, matrixSize));
    cudaCheck(cudaMallocHost(&filter_h, filterSize));
    cudaCheck(cudaMalloc(&filter_d, filterSize));
    cudaCheck(cudaMallocHost(&output_h, matrixSize));
    cudaCheck(cudaMalloc(&output_d, matrixSize));

    dim3 block(16, 16);
    dim3 grid(ceil((float)MATRIX_HEIGHT / block.x), ceil((float)MATRIX_WIDTH / block.y));
    initData<<<grid, block>>>(matrix_d, MATRIX_HEIGHT, 2.0);
    cudaCheck(cudaDeviceSynchronize());

    block = dim3(5, 5);
    grid = dim3(ceil((float)(2 * FILTER_RADIUS + 1) / block.x), ceil((float)(2 * FILTER_RADIUS + 1) / block.y));
    initData<<<grid, block>>>(filter_d, (2 * FILTER_RADIUS + 1), 3.0);
    cudaCheck(cudaDeviceSynchronize());

    block = dim3(16, 16);
    grid = dim3(ceil((float)MATRIX_HEIGHT / block.x), ceil((float)MATRIX_WIDTH / block.y));
    convolution_2D_basic_kernel<<<grid, block>>>(matrix_d, filter_d, output_d, FILTER_RADIUS, MATRIX_HEIGHT, MATRIX_WIDTH);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(output_h, output_d, matrixSize, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(matrix_h, matrix_d, matrixSize, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(filter_h, filter_d, filterSize, cudaMemcpyDeviceToHost));
    cudaCheck(cudaDeviceSynchronize());
    verify_convolution(matrix_h, filter_h, output_h, FILTER_RADIUS, MATRIX_WIDTH, MATRIX_HEIGHT);

    cudaFree(matrix_d);
    cudaFree(matrix_h);
    cudaFree(filter_d);
    cudaFree(filter_h);
    cudaFree(output_d);
    cudaFree(output_h);
    return 0;
}
