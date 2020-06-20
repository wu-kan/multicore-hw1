#include "core.h"

namespace wk0
{
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
} // namespace wk0
namespace wk1
{
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
} // namespace wk1
namespace wk2
{
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
} // namespace wk2
namespace wk3
{
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
} // namespace wk3
namespace wk4
{
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
} // namespace wk4
namespace wk5
{
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
} // namespace wk5
namespace wk6
{
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
} // namespace wk6
namespace wk7
{
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
} // namespace wk7
namespace wk8
{
    texture<float, 2> input_tex;

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
                        ++cnt[(signed char)tex2D(input_tex, px, py)];
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

        CHECK(cudaBindTexture2D(
            0,
            &input_tex,
            input_d,
            &input_tex.channelDesc,
            width,
            height,
            width * sizeof(float)));

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

        CHECK(cudaUnbindTexture(input_tex));

        *result = (float *)malloc(sizeof(float) * width * height);
        CHECK(cudaMemcpy(*result, output_d, sizeof(float) * width * height, cudaMemcpyDeviceToHost));
        CHECK(cudaFree(input_d));
        CHECK(cudaFree(output_d));
    }
} // namespace wk8
namespace wk9
{
    template <
        int BLOCK_DIM_X,
        int BLOCK_DIM_Y>
    __global__ void cudaCallbackKernel(
        const int width,
        const int height,
        const float *__restrict__ input,
        float *__restrict__ output)
    {
        const int idy = blockIdx.y * (BLOCK_DIM_Y - 4) + threadIdx.y - 2;
        const int idx = blockIdx.x * (BLOCK_DIM_X - 4) + threadIdx.x - 2;
        __shared__ int input_s[BLOCK_DIM_Y][BLOCK_DIM_X | 1];

        input_s[threadIdx.y][threadIdx.x] = 0 <= idy && idy < height && 0 <= idx && idx < width ? input[idy * width + idx] : 16;

        __syncthreads();

        if (1 < threadIdx.y && threadIdx.y < BLOCK_DIM_Y - 2 &&
            1 < threadIdx.x && threadIdx.x < BLOCK_DIM_X - 2 &&
            idy < height && idx < width)
        {
            signed char cnt[17] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            for (signed char offsety = -2; offsety <= 2; ++offsety)
                for (signed char offsetx = -2; offsetx <= 2; ++offsetx)
                    ++cnt[input_s[threadIdx.y + offsety][threadIdx.x + offsetx]];
            double ans = 0.0;
            for (signed char i = 0; i < 16; ++i)
                if (cnt[i])
                    ans += -0.04F * cnt[i] * log(0.04F * cnt[i]);
            output[idy * width + idx] = ans;
        }
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
            gridDim(divup(width, BLOCK_DIM_X - 4), divup(height, BLOCK_DIM_Y - 4));

        cudaCallbackKernel<
            BLOCK_DIM_X,
            BLOCK_DIM_Y><<<
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
} // namespace wk9
namespace wk10
{
    void cudaCallback(
        int width,
        int height,
        float *sample,
        float **result)
    {
        *result = (float *)malloc(sizeof(float) * width * height);
#pragma omp parallel for
        for (int pos = 0; pos < width * height; ++pos)
        {
            const int
                idy = pos / width,
                idx = pos - idy * width;
            int cnt[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            for (int offsety = -2; offsety <= 2; ++offsety)
                for (int offsetx = -2; offsetx <= 2; ++offsetx)
                {
                    const int py = idy + offsety, px = idx + offsetx;
                    if (0 <= py && py < height && 0 <= px && px < width)
                        ++cnt[(int)sample[py * width + px]];
                }
            double ans = 0.0;
            for (int i = 0; i < 16; ++i)
                if (cnt[i])
                    ans += -0.04 * cnt[i] * log(0.04 * cnt[i]);
            (*result)[pos] = ans;
        }
    }
} // namespace wk10
namespace wk11
{
    void cudaCallback(
        int width,
        int height,
        float *sample,
        float **result)
    {
        *result = (float *)malloc(sizeof(float) * width * height);
#pragma omp parallel for
        for (int pos = 0; pos < width * height; ++pos)
        {
            const int
                idy = pos / width,
                idx = pos - idy * width;
            int cnt[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            for (int offsety = -2; offsety <= 2; ++offsety)
                for (int offsetx = -2; offsetx <= 2; ++offsetx)
                {
                    const int py = idy + offsety, px = idx + offsetx;
                    if (0 <= py && py < height && 0 <= px && px < width)
                        ++cnt[(int)sample[py * width + px]];
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
            (*result)[pos] = ans;
        }
    }
} // namespace wk11
namespace wk12
{
    void cudaCallback(
        int width,
        int height,
        float *sample,
        float **result)
    {
        *result = (float *)malloc(sizeof(float) * width * height);
        int *sum[16];
#pragma omp parallel for
        for (int i = 0; i < 16; ++i)
        {
            int *p = (int *)malloc(sizeof(int) * (width + 5) * (height + 5));
            for (int pos = 0; pos < (width + 5) * (height + 5); ++pos)
            {
                const int idy = pos / (width + 5), idx = pos - idy * (width + 5);
                if (idy && idx)
                {
                    p[pos] = p[(idy - 1) * (width + 5) + idx] + p[idy * (width + 5) + idx - 1] - p[(idy - 1) * (width + 5) + (idx - 1)];
                    const int py = idy - 3, px = idx - 3;
                    if (0 <= py && py < height && 0 <= px && px < width && i == sample[py * width + px])
                        ++p[pos];
                }
                else
                    p[pos] = 0;
            }
            sum[i] = p;
        }
#pragma omp parallel for
        for (int pos = 0; pos < width * height; ++pos)
        {
            const int
                idy = pos / width,
                idx = pos - idy * width;
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
                ans += plogp[sum[i][(idy + 5) * (width + 5) + idx + 5] - sum[i][(idy + 5) * (width + 5) + idx] - sum[i][idy * (width + 5) + idx + 5] + sum[i][idy * (width + 5) + idx]];
            (*result)[pos] = ans;
        }
    }
} // namespace wk12

void cudaCallback(
    int width,
    int height,
    float *sample,
    float **result)
{
    return wk12::cudaCallback(width, height, sample, result);
}
