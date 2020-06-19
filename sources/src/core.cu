#include <cuda_runtime.h>
#include "core.h"
#define VERSION 7

#if VERSION == 7

__global__ void cudaCallbackKernel(
    const int width,
    const int height,
    const float *__restrict__ input,
    float *__restrict__ output)
{
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    signed char cnt[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (signed char offsety = -2; offsety <= 2; ++offsety)
    {
        const int py = idy + offsety;
        if (0 <= py && py < height)
            for (signed char offsetx = -2; offsetx <= 2; ++offsetx)
            {
                const int px = idx + offsetx;
                if (0 <= px && px < width)
                    ++cnt[(signed char)input[py * width + px]];
            }
    }
    double ans = 0.0;
    for (signed char i = 0; i < 16; ++i)
        if (cnt[i])
            ans += -0.04F * cnt[i] * log(0.04F * cnt[i]);
    if (idy < height && idx < width)
        output[idy * width + idx] = ans;
}

void cudaCallback(
    int width,
    int height,
    float *sample,
    float **result)
{
    float *input_d, *output_d;

    CHECK(cudaMalloc((void **)&input_d, sizeof(float) * width * height));
    CHECK(cudaMalloc((void **)&output_d, sizeof(float) * width * height));
    CHECK(cudaMemcpy(input_d, sample, sizeof(float) * width * height, cudaMemcpyHostToDevice));

    const int
        BLOCK_DIM_X = 32,
        BLOCK_DIM_Y = 32;

    const dim3
        blockDim(BLOCK_DIM_X, BLOCK_DIM_Y),
        gridDim(divup(width, BLOCK_DIM_X), divup(height, BLOCK_DIM_Y));

    cudaCallbackKernel<<<
        gridDim,
        blockDim>>>(
        width,
        height,
        input_d,
        output_d);

    *result = (float *)malloc(sizeof(float) * width * height);
    CHECK(cudaMemcpy(*result, output_d, sizeof(float) * width * height, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(input_d));
    CHECK(cudaFree(output_d));
}

#elif VERSION == 6

texture<float> plogp_tex;
__device__ float plogp[26];
struct InitPlogp
{
    InitPlogp()
    {
        float plogp_h[26] = {0.0}, *plogp_d;
        for (char i = 1; i < 26; ++i)
            plogp_h[i] = -0.04 * i * log(0.04 * i);
        cudaMemcpyToSymbol(plogp, plogp_h, sizeof(float) * 26);
        cudaGetSymbolAddress((void **)&plogp_d, plogp);
        cudaBindTexture(0, plogp_tex, plogp_d);
    }
} tmpInit;

__global__ void cudaCallbackKernel(
    const int width,
    const int height,
    const float *__restrict__ input,
    float *__restrict__ output)
{
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    signed char cnt[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (signed char offsety = -2; offsety <= 2; ++offsety)
    {
        const int py = idy + offsety;
        if (0 <= py && py < height)
            for (signed char offsetx = -2; offsetx <= 2; ++offsetx)
            {
                const int px = idx + offsetx;
                if (0 <= px && px < width)
                    ++cnt[(signed char)input[py * width + px]];
            }
    }
    double ans = 0.0;
    for (signed char i = 0; i < 16; ++i)
        ans += tex1Dfetch(plogp_tex, cnt[i]);
    if (idy < height && idx < width)
        output[idy * width + idx] = ans;
}

void cudaCallback(
    int width,
    int height,
    float *sample,
    float **result)
{
    float *input_d, *output_d;

    CHECK(cudaMalloc((void **)&input_d, sizeof(float) * width * height));
    CHECK(cudaMalloc((void **)&output_d, sizeof(float) * width * height));
    CHECK(cudaMemcpy(input_d, sample, sizeof(float) * width * height, cudaMemcpyHostToDevice));

    const int
        BLOCK_DIM_X = 32,
        BLOCK_DIM_Y = 32;

    const dim3
        blockDim(BLOCK_DIM_X, BLOCK_DIM_Y),
        gridDim(divup(width, BLOCK_DIM_X), divup(height, BLOCK_DIM_Y));

    cudaCallbackKernel<<<
        gridDim,
        blockDim>>>(
        width,
        height,
        input_d,
        output_d);

    *result = (float *)malloc(sizeof(float) * width * height);
    CHECK(cudaMemcpy(*result, output_d, sizeof(float) * width * height, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(input_d));
    CHECK(cudaFree(output_d));
}

#elif VERSION == 5

texture<float> plogp_tex;
__device__ float plogp[26];
struct InitPlogp
{
    InitPlogp()
    {
        float plogp_h[26] = {0.0}, *plogp_d;
        for (char i = 1; i < 26; ++i)
            plogp_h[i] = -0.04 * i * log(0.04 * i);
        cudaMemcpyToSymbol(plogp, plogp_h, sizeof(float) * 26);
        cudaGetSymbolAddress((void **)&plogp_d, plogp);
        cudaBindTexture(0, plogp_tex, plogp_d);
    }
} tmpInit;

__global__ void cudaCallbackKernel(
    const int width,
    const int height,
    const float *__restrict__ input,
    float *__restrict__ output)
{
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int cnt[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int offsety = -2; offsety <= 2; ++offsety)
    {
        const int py = idy + offsety;
        if (0 <= py && py < height)
            for (int offsetx = -2; offsetx <= 2; ++offsetx)
            {
                const int px = idx + offsetx;
                if (0 <= px && px < width)
                    ++cnt[(int)input[py * width + px]];
            }
    }
    double ans = 0.0;
    for (int i = 0; i < 16; ++i)
        ans += tex1Dfetch(plogp_tex, cnt[i]);
    if (idy < height && idx < width)
        output[idy * width + idx] = ans;
}

void cudaCallback(
    int width,
    int height,
    float *sample,
    float **result)
{
    float *input_d, *output_d;

    CHECK(cudaMalloc((void **)&input_d, sizeof(float) * width * height));
    CHECK(cudaMalloc((void **)&output_d, sizeof(float) * width * height));
    CHECK(cudaMemcpy(input_d, sample, sizeof(float) * width * height, cudaMemcpyHostToDevice));

    const int
        BLOCK_DIM_X = 32,
        BLOCK_DIM_Y = 32;

    const dim3
        blockDim(BLOCK_DIM_X, BLOCK_DIM_Y),
        gridDim(divup(width, BLOCK_DIM_X), divup(height, BLOCK_DIM_Y));

    cudaCallbackKernel<<<
        gridDim,
        blockDim>>>(
        width,
        height,
        input_d,
        output_d);

    *result = (float *)malloc(sizeof(float) * width * height);
    CHECK(cudaMemcpy(*result, output_d, sizeof(float) * width * height, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(input_d));
    CHECK(cudaFree(output_d));
}

#elif VERSION == 4

__device__ double plogp[26];
struct InitPlogp
{
    InitPlogp()
    {
        double plogp_h[26] = {0.0};
        for (char i = 1; i < 26; ++i)
            plogp_h[i] = -0.04 * i * log(0.04 * i);
        cudaMemcpyToSymbol(plogp, plogp_h, sizeof(float) * 26);
    }
} tmpInit;

__global__ void cudaCallbackKernel(
    const int width,
    const int height,
    const float *__restrict__ input,
    float *__restrict__ output)
{
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int cnt[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int offsety = -2; offsety <= 2; ++offsety)
    {
        const int py = idy + offsety;
        if (0 <= py && py < height)
            for (int offsetx = -2; offsetx <= 2; ++offsetx)
            {
                const int px = idx + offsetx;
                if (0 <= px && px < width)
                    ++cnt[(int)input[py * width + px]];
            }
    }
    double ans = 0.0;
    for (int i = 0; i < 16; ++i)
        ans += plogp[cnt[i]];
    if (idy < height && idx < width)
        output[idy * width + idx] = ans;
}

void cudaCallback(
    int width,
    int height,
    float *sample,
    float **result)
{
    float *input_d, *output_d;

    CHECK(cudaMalloc((void **)&input_d, sizeof(float) * width * height));
    CHECK(cudaMalloc((void **)&output_d, sizeof(float) * width * height));
    CHECK(cudaMemcpy(input_d, sample, sizeof(float) * width * height, cudaMemcpyHostToDevice));

    const int
        BLOCK_DIM_X = 32,
        BLOCK_DIM_Y = 32;

    const dim3
        blockDim(BLOCK_DIM_X, BLOCK_DIM_Y),
        gridDim(divup(width, BLOCK_DIM_X), divup(height, BLOCK_DIM_Y));

    cudaCallbackKernel<<<
        gridDim,
        blockDim>>>(
        width,
        height,
        input_d,
        output_d);

    *result = (float *)malloc(sizeof(float) * width * height);
    CHECK(cudaMemcpy(*result, output_d, sizeof(float) * width * height, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(input_d));
    CHECK(cudaFree(output_d));
}

#elif VERSION == 3

__constant__ double plogp[26];
struct InitPlogp
{
    InitPlogp()
    {
        double plogp_h[26] = {0.0};
        for (char i = 1; i < 26; ++i)
            plogp_h[i] = -0.04 * i * log(0.04 * i);
        cudaMemcpyToSymbol(plogp, plogp_h, sizeof(float) * 26);
    }
} tmpInit;

__global__ void cudaCallbackKernel(
    const int width,
    const int height,
    const float *__restrict__ input,
    float *__restrict__ output)
{
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int cnt[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int offsety = -2; offsety <= 2; ++offsety)
    {
        const int py = idy + offsety;
        if (0 <= py && py < height)
            for (int offsetx = -2; offsetx <= 2; ++offsetx)
            {
                const int px = idx + offsetx;
                if (0 <= px && px < width)
                    ++cnt[(int)input[py * width + px]];
            }
    }
    double ans = 0.0;
    for (int i = 0; i < 16; ++i)
        ans += plogp[cnt[i]];
    if (idy < height && idx < width)
        output[idy * width + idx] = ans;
}

void cudaCallback(
    int width,
    int height,
    float *sample,
    float **result)
{
    float *input_d, *output_d;

    CHECK(cudaMalloc((void **)&input_d, sizeof(float) * width * height));
    CHECK(cudaMalloc((void **)&output_d, sizeof(float) * width * height));
    CHECK(cudaMemcpy(input_d, sample, sizeof(float) * width * height, cudaMemcpyHostToDevice));

    const int
        BLOCK_DIM_X = 32,
        BLOCK_DIM_Y = 32;

    const dim3
        blockDim(BLOCK_DIM_X, BLOCK_DIM_Y),
        gridDim(divup(width, BLOCK_DIM_X), divup(height, BLOCK_DIM_Y));

    cudaCallbackKernel<<<
        gridDim,
        blockDim>>>(
        width,
        height,
        input_d,
        output_d);

    *result = (float *)malloc(sizeof(float) * width * height);
    CHECK(cudaMemcpy(*result, output_d, sizeof(float) * width * height, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(input_d));
    CHECK(cudaFree(output_d));
}

#elif VERSION == 2

__global__ void cudaCallbackKernel(
    const int width,
    const int height,
    const float *__restrict__ input,
    float *__restrict__ output)
{
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int cnt[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int offsety = -2; offsety <= 2; ++offsety)
    {
        const int py = idy + offsety;
        if (0 <= py && py < height)
            for (int offsetx = -2; offsetx <= 2; ++offsetx)
            {
                const int px = idx + offsetx;
                if (0 <= px && px < width)
                    ++cnt[(int)input[py * width + px]];
            }
    }
    double ans = 0.0;
    __shared__ double plogp[26];
    if (threadIdx.y == 0 && threadIdx.x < 26)
        plogp[threadIdx.x] = threadIdx.x == 0 ? 0.0 : -0.04 * threadIdx.x * log(0.04 * threadIdx.x);
    __syncthreads();
    for (int i = 0; i < 16; ++i)
        ans += plogp[cnt[i]];
    if (idy < height && idx < width)
        output[idy * width + idx] = ans;
}

void cudaCallback(
    int width,
    int height,
    float *sample,
    float **result)
{
    float *input_d, *output_d;

    CHECK(cudaMalloc((void **)&input_d, sizeof(float) * width * height));
    CHECK(cudaMalloc((void **)&output_d, sizeof(float) * width * height));
    CHECK(cudaMemcpy(input_d, sample, sizeof(float) * width * height, cudaMemcpyHostToDevice));

    const int
        BLOCK_DIM_X = 32,
        BLOCK_DIM_Y = 32;

    const dim3
        blockDim(BLOCK_DIM_X, BLOCK_DIM_Y),
        gridDim(divup(width, BLOCK_DIM_X), divup(height, BLOCK_DIM_Y));

    cudaCallbackKernel<<<
        gridDim,
        blockDim>>>(
        width,
        height,
        input_d,
        output_d);

    *result = (float *)malloc(sizeof(float) * width * height);
    CHECK(cudaMemcpy(*result, output_d, sizeof(float) * width * height, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(input_d));
    CHECK(cudaFree(output_d));
}

#elif VERSION == 1

__global__ void cudaCallbackKernel(
    const int width,
    const int height,
    const float *__restrict__ input,
    float *__restrict__ output)
{
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int cnt[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int offsety = -2; offsety <= 2; ++offsety)
    {
        const int py = idy + offsety;
        if (0 <= py && py < height)
            for (int offsetx = -2; offsetx <= 2; ++offsetx)
            {
                const int px = idx + offsetx;
                if (0 <= px && px < width)
                    ++cnt[(int)input[py * width + px]];
            }
    }
    double ans = 0.0;
    const double plogp[26] = {
        -0.0,
        -0.04 * log(0.04),
        -0.08 * log(0.08),
        -0.12 * log(0.12),
        -0.16 * log(0.16),
        -0.20 * log(0.20),
        -0.24 * log(0.24),
        -0.28 * log(0.28),
        -0.32 * log(0.32),
        -0.36 * log(0.36),
        -0.40 * log(0.40),
        -0.44 * log(0.44),
        -0.48 * log(0.48),
        -0.52 * log(0.52),
        -0.56 * log(0.56),
        -0.60 * log(0.60),
        -0.64 * log(0.64),
        -0.68 * log(0.68),
        -0.72 * log(0.72),
        -0.76 * log(0.76),
        -0.80 * log(0.80),
        -0.84 * log(0.84),
        -0.88 * log(0.88),
        -0.92 * log(0.92),
        -0.96 * log(0.96),
        -0.0};
    for (int i = 0; i < 16; ++i)
        ans += plogp[cnt[i]];
    if (idy < height && idx < width)
        output[idy * width + idx] = ans;
}

void cudaCallback(
    int width,
    int height,
    float *sample,
    float **result)
{
    float *input_d, *output_d;

    CHECK(cudaMalloc((void **)&input_d, sizeof(float) * width * height));
    CHECK(cudaMalloc((void **)&output_d, sizeof(float) * width * height));
    CHECK(cudaMemcpy(input_d, sample, sizeof(float) * width * height, cudaMemcpyHostToDevice));

    const int
        BLOCK_DIM_X = 32,
        BLOCK_DIM_Y = 32;

    const dim3
        blockDim(BLOCK_DIM_X, BLOCK_DIM_Y),
        gridDim(divup(width, BLOCK_DIM_X), divup(height, BLOCK_DIM_Y));

    cudaCallbackKernel<<<
        gridDim,
        blockDim>>>(
        width,
        height,
        input_d,
        output_d);

    *result = (float *)malloc(sizeof(float) * width * height);
    CHECK(cudaMemcpy(*result, output_d, sizeof(float) * width * height, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(input_d));
    CHECK(cudaFree(output_d));
}
#elif VERSION == 0

__global__ void cudaCallbackKernel(
    const int width,
    const int height,
    const float *__restrict__ input,
    float *__restrict__ output)
{
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int cnt[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int offsety = -2; offsety <= 2; ++offsety)
    {
        const int py = idy + offsety;
        if (0 <= py && py < height)
            for (int offsetx = -2; offsetx <= 2; ++offsetx)
            {
                const int px = idx + offsetx;
                if (0 <= px && px < width)
                    ++cnt[(int)input[py * width + px]];
            }
    }
    double ans = 0.0;
    for (int i = 0; i < 16; ++i)
        if (cnt[i])
            ans += -0.04 * cnt[i] * log(0.04 * cnt[i]);
    if (idy < height && idx < width)
        output[idy * width + idx] = ans;
}

void cudaCallback(
    int width,
    int height,
    float *sample,
    float **result)
{
    float *input_d, *output_d;

    CHECK(cudaMalloc((void **)&input_d, sizeof(float) * width * height));
    CHECK(cudaMalloc((void **)&output_d, sizeof(float) * width * height));
    CHECK(cudaMemcpy(input_d, sample, sizeof(float) * width * height, cudaMemcpyHostToDevice));

    const int
        BLOCK_DIM_X = 32,
        BLOCK_DIM_Y = 32;

    const dim3
        blockDim(BLOCK_DIM_X, BLOCK_DIM_Y),
        gridDim(divup(width, BLOCK_DIM_X), divup(height, BLOCK_DIM_Y));

    cudaCallbackKernel<<<
        gridDim,
        blockDim>>>(
        width,
        height,
        input_d,
        output_d);

    *result = (float *)malloc(sizeof(float) * width * height);
    CHECK(cudaMemcpy(*result, output_d, sizeof(float) * width * height, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(input_d));
    CHECK(cudaFree(output_d));
}

#endif