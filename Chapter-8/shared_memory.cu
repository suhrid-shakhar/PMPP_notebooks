#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

#define cudaCheck(call)                                                            \
    {                                                                              \
        cudaError_t err = call;                                                    \
        if (err != cudaSuccess)                                                    \
        {                                                                          \
            printf("%s at %s: %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        }                                                                          \
    }


#define MATRIX_WIDTH 1024
#define BLOCK_DIM 8
#define IN_TILE_DIM BLOCK_DIM
#define OUT_TILE_DIM (IN_TILE_DIM - 2)

#define C0 0.1f
#define C1 0.1f
#define C2 0.1f
#define C3 0.1f
#define C4 0.1f
#define C5 0.1f
#define C6 0.1f

__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
    int i = blockIdx.z*OUT_TILE_DIM + threadIdx.z - 1;
    int j = blockIdx.y*OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x*OUT_TILE_DIM + threadIdx.x - 1;

    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    if(i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i*N*N + j*N + k];
    }
    __syncthreads();
    if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
         if(threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM - 1 
            && threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 
            && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) 
        {
            out[i*N*N + j*N + k] = C0*in_s[threadIdx.z][threadIdx.y][threadIdx.x]
                                 + C1*in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1]
                                 + C2*in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1]
                                 + C3*in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x]
                                 + C4*in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x]
                                 + C5*in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x]
                                 + C6*in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x];
        }
    }
}
void stencil_cpu(const std::vector<float>& in, std::vector<float>& out, unsigned int N) {
    for (unsigned int i = 1; i < N - 1; ++i) {
        for (unsigned int j = 1; j < N - 1; ++j) {
            for (unsigned int k = 1; k < N - 1; ++k) {
                out[i * N * N + j * N + k] =
                      C0 * in[i * N * N + j * N + k]
                    + C1 * in[i * N * N + j * N + (k - 1)]
                    + C2 * in[i * N * N + j * N + (k + 1)]
                    + C3 * in[i * N * N + (j - 1) * N + k]
                    + C4 * in[i * N * N + (j + 1) * N + k]
                    + C5 * in[(i - 1) * N * N + j * N + k]
                    + C6 * in[(i + 1) * N * N + j * N + k];
            }
        }
    }
}
int main() {
    size_t size = MATRIX_WIDTH * MATRIX_WIDTH * MATRIX_WIDTH * sizeof(float);

    std::vector<float> in_h(MATRIX_WIDTH * MATRIX_WIDTH * MATRIX_WIDTH);
    std::vector<float> out_h(MATRIX_WIDTH * MATRIX_WIDTH * MATRIX_WIDTH, 0);

    for (int i = 0; i < MATRIX_WIDTH * MATRIX_WIDTH * MATRIX_WIDTH; ++i) {
        in_h[i] = 2.0f;
    }

    float *in_d, *out_d;
    cudaCheck(cudaMalloc(&in_d, size));
    cudaCheck(cudaMalloc(&out_d, size));

    cudaCheck(cudaMemcpy(in_d, in_h.data(), size, cudaMemcpyHostToDevice));

    dim3 blockDim(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((MATRIX_WIDTH - 2 + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
             (MATRIX_WIDTH - 2 + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
             (MATRIX_WIDTH - 2 + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    stencil_kernel<<<gridDim, blockDim>>>(in_d, out_d, MATRIX_WIDTH);
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(out_h.data(), out_d, size, cudaMemcpyDeviceToHost));
    cudaCheck(cudaDeviceSynchronize());

    std::vector<float> ref_h(MATRIX_WIDTH * MATRIX_WIDTH * MATRIX_WIDTH, 0);
    stencil_cpu(in_h, ref_h, MATRIX_WIDTH);

    // Compare results
    int mismatch = 0;
    for (int i = 1; i < MATRIX_WIDTH - 1 && !mismatch; ++i) {
        for (int j = 1; j < MATRIX_WIDTH - 1 && !mismatch; ++j) {
            for (int k = 1; k < MATRIX_WIDTH - 1 && !mismatch; ++k) {
                int idx = i * MATRIX_WIDTH * MATRIX_WIDTH + j * MATRIX_WIDTH + k;
                float diff = fabs(ref_h[idx] - out_h[idx]);
                if (diff > 1e-5f) {
                    std::cout << "Mismatch at (" << i << "," << j << "," << k << ") "
                            << "CPU: " << ref_h[idx] << ",GPU: "<< out_h[idx] << std::endl;               
                    mismatch++;
                }
            }
        }
    }
    if(!mismatch) 
        std::cout<<"CPU Matches GPU result."<<std::endl;
    else 
        std::cout<<"Result incorrect."<<std::endl;
    cudaFree(in_d);
    cudaFree(out_d);

    return 0;
}