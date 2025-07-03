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

#define MATRIX_HEIGHT (1<<14)
#define MATRIX_WIDTH (1<<14)

#define FILTER_RADIUS 2

#define IN_TILE_DIM 16
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * (FILTER_RADIUS))

__constant__ float F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

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

void initFilterHost(float *data, int size, float value = 1.0f)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            data[i * size + j] = value;
        }
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

__global__ void convolution_2D_basic_kernel(float *N, float *P, int r, int width, int height)
{

    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];

    if (row >= 0 && row < height && col >= 0 && col < width)
    {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    }
    else
    {
        N_s[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();

    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;

    if (col >= 0 && col < width && row >= 0 && row < height)
    {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM)
        {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2 * r + 1; fRow++)
            {
                for (int fCol = 0; fCol < 2 * r + 1; fCol++)
                {
                    Pvalue += F[fRow][fCol] * N_s[tileRow + fRow][tileCol + fCol];
                }
            }
            P[row * width + col] = Pvalue;
        }
    }
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
    float *matrix_d, *output_d;

    cudaCheck(cudaMallocHost(&matrix_h, matrixSize));
    cudaCheck(cudaMalloc(&matrix_d, matrixSize));
    cudaCheck(cudaMallocHost(&filter_h, filterSize));
    cudaCheck(cudaMallocHost(&output_h, matrixSize));
    cudaCheck(cudaMalloc(&output_d, matrixSize));

    dim3 block(16, 16);
    dim3 grid(ceil((float)MATRIX_HEIGHT / block.x), ceil((float)MATRIX_WIDTH / block.y));
    initData<<<grid, block>>>(matrix_d, MATRIX_HEIGHT, 2.0);
    cudaCheck(cudaDeviceSynchronize());

    initFilterHost(filter_h, 2 * FILTER_RADIUS + 1, 3.0f);
    cudaCheck(cudaMemcpyToSymbol(F, filter_h, filterSize));

    block = dim3(IN_TILE_DIM, IN_TILE_DIM);
    grid = dim3((MATRIX_WIDTH + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (MATRIX_HEIGHT + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    convolution_2D_basic_kernel<<<grid, block>>>(matrix_d, output_d, FILTER_RADIUS, MATRIX_HEIGHT, MATRIX_WIDTH);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(output_h, output_d, matrixSize, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(matrix_h, matrix_d, matrixSize, cudaMemcpyDeviceToHost));
    cudaCheck(cudaDeviceSynchronize());
    verify_convolution(matrix_h, filter_h, output_h, FILTER_RADIUS, MATRIX_WIDTH, MATRIX_HEIGHT);

    cudaFree(matrix_d);
    cudaFree(matrix_h);
    cudaFree(filter_h);
    cudaFree(output_d);
    cudaFree(output_h);
    return 0;
}
