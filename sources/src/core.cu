#include "core.h"

namespace v0 //cuda baseline
{
    static __global__ void cudaCallbackKernel(
        const int width,
        const int height,
        const float *__restrict__ input,
        float *__restrict__ output)
    {
        const int idy = blockIdx.y * blockDim.y + threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idy < height && idx < width)
        {
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
            double
                n = (min(idx, 2) + 1 + min(width - idx, 2)) * (min(idy, 2) + 1 + min(height - idy, 2)),
                n_inv = 1.0 / n,
                ans = log(n);
            for (int i = 0; i < 16; ++i)
                if (cnt[i])
                    ans -= log((double)cnt[i]) * cnt[i] * n_inv;
            output[idy * width + idx] = ans;
        }
    }

    static void cudaCallback(
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
} // namespace v0
namespace v1 //cuda 预处理log到寄存器
{
    static __global__ void cudaCallbackKernel(
        const int width,
        const int height,
        const float *__restrict__ input,
        float *__restrict__ output)
    {
        const int idy = blockIdx.y * blockDim.y + threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idy < height && idx < width)
        {
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
            const double mylog[26] = {
                0.0,
                log(1.0),
                log(2.0),
                log(3.0),
                log(4.0),
                log(5.0),
                log(6.0),
                log(7.0),
                log(8.0),
                log(9.0),
                log(10.0),
                log(11.0),
                log(12.0),
                log(13.0),
                log(14.0),
                log(15.0),
                log(16.0),
                log(17.0),
                log(18.0),
                log(19.0),
                log(20.0),
                log(21.0),
                log(22.0),
                log(23.0),
                log(24.0),
                log(25.0)};

            const int n = (min(idx, 2) + 1 + min(width - idx, 2)) * (min(idy, 2) + 1 + min(height - idy, 2));
            double ans = mylog[n], n_inv = 1.0 / n;
            for (int i = 0; i < 16; ++i)
                ans -= mylog[cnt[i]] * cnt[i] * n_inv;
            output[idy * width + idx] = ans;
        }
    }

    static void cudaCallback(
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
} // namespace v1
namespace v2 //cuda 预处理log到shared memory
{
    static __global__ void cudaCallbackKernel(
        const int width,
        const int height,
        const float *__restrict__ input,
        float *__restrict__ output)
    {
        const int idy = blockIdx.y * blockDim.y + threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;

        __shared__ double mylog[26];
        if (threadIdx.y == 0 && threadIdx.x < 26)
            mylog[threadIdx.x] = threadIdx.x == 0 ? 0.0 : log((double)threadIdx.x);
        __syncthreads();
        if (idy < height && idx < width)
        {
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
            const int n = (min(idx, 2) + 1 + min(width - idx, 2)) * (min(idy, 2) + 1 + min(height - idy, 2));
            double ans = mylog[n], n_inv = 1.0 / n;
            for (int i = 0; i < 16; ++i)
                ans -= mylog[cnt[i]] * cnt[i] * n_inv;
            output[idy * width + idx] = ans;
        }
    }

    static void cudaCallback(
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
} // namespace v2
namespace v3 //cuda 预处理log到constant memory
{
    static __constant__ double mylog[26];
    static struct InitPlogp
    {
        InitPlogp()
        {
            const double mylog_h[26] = {
                0.0,
                log(1.0),
                log(2.0),
                log(3.0),
                log(4.0),
                log(5.0),
                log(6.0),
                log(7.0),
                log(8.0),
                log(9.0),
                log(10.0),
                log(11.0),
                log(12.0),
                log(13.0),
                log(14.0),
                log(15.0),
                log(16.0),
                log(17.0),
                log(18.0),
                log(19.0),
                log(20.0),
                log(21.0),
                log(22.0),
                log(23.0),
                log(24.0),
                log(25.0)};
            cudaMemcpyToSymbol(mylog, mylog_h, sizeof(double) * 26);
        }
    } tmpInit;

    static __global__ void cudaCallbackKernel(
        const int width,
        const int height,
        const float *__restrict__ input,
        float *__restrict__ output)
    {
        const int idy = blockIdx.y * blockDim.y + threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idy < height && idx < width)
        {
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
            const int n = (min(idx, 2) + 1 + min(width - idx, 2)) * (min(idy, 2) + 1 + min(height - idy, 2));
            double ans = mylog[n], n_inv = 1.0 / n;
            for (int i = 0; i < 16; ++i)
                ans -= mylog[cnt[i]] * cnt[i] * n_inv;
            output[idy * width + idx] = ans;
        }
    }

    static void cudaCallback(
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
} // namespace v3
namespace v4 //cuda 预处理log到device memory
{
    static __device__ double mylog[26];
    static struct InitPlogp
    {
        InitPlogp()
        {
            const double mylog_h[26] = {
                0.0,
                log(1.0),
                log(2.0),
                log(3.0),
                log(4.0),
                log(5.0),
                log(6.0),
                log(7.0),
                log(8.0),
                log(9.0),
                log(10.0),
                log(11.0),
                log(12.0),
                log(13.0),
                log(14.0),
                log(15.0),
                log(16.0),
                log(17.0),
                log(18.0),
                log(19.0),
                log(20.0),
                log(21.0),
                log(22.0),
                log(23.0),
                log(24.0),
                log(25.0)};
            cudaMemcpyToSymbol(mylog, mylog_h, sizeof(double) * 26);
        }
    } tmpInit;

    static __global__ void cudaCallbackKernel(
        const int width,
        const int height,
        const float *__restrict__ input,
        float *__restrict__ output)
    {
        const int idy = blockIdx.y * blockDim.y + threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idy < height && idx < width)
        {
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
            const int n = (min(idx, 2) + 1 + min(width - idx, 2)) * (min(idy, 2) + 1 + min(height - idy, 2));
            double ans = mylog[n], n_inv = 1.0 / n;
            for (int i = 0; i < 16; ++i)
                ans -= mylog[cnt[i]] * cnt[i] * n_inv;
            output[idy * width + idx] = ans;
        }
    }

    static void cudaCallback(
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
} // namespace v4
namespace v5 //cuda 预处理log到texure memory
{
    static texture<float> mylog_tex;
    static __device__ float mylog[26];
    static struct InitPlogp
    {
        InitPlogp()
        {
            float mylog_h[26] = {
                0.0,
                log(1.0),
                log(2.0),
                log(3.0),
                log(4.0),
                log(5.0),
                log(6.0),
                log(7.0),
                log(8.0),
                log(9.0),
                log(10.0),
                log(11.0),
                log(12.0),
                log(13.0),
                log(14.0),
                log(15.0),
                log(16.0),
                log(17.0),
                log(18.0),
                log(19.0),
                log(20.0),
                log(21.0),
                log(22.0),
                log(23.0),
                log(24.0),
                log(25.0)},
                  *mylog_d;
            cudaMemcpyToSymbol(mylog, mylog_h, sizeof(float) * 26);
            cudaGetSymbolAddress((void **)&mylog_d, mylog);
            cudaBindTexture(0, mylog_tex, mylog_d);
        }
    } tmpInit;

    static __global__ void cudaCallbackKernel(
        const int width,
        const int height,
        const float *__restrict__ input,
        float *__restrict__ output)
    {
        const int idy = blockIdx.y * blockDim.y + threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idy < height && idx < width)
        {
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
            const int n = (min(idx, 2) + 1 + min(width - idx, 2)) * (min(idy, 2) + 1 + min(height - idy, 2));
            double ans = tex1Dfetch(mylog_tex, n), n_inv = 1.0 / n;
            for (int i = 0; i < 16; ++i)
                ans -= tex1Dfetch(mylog_tex, cnt[i]) * cnt[i] * n_inv;
            output[idy * width + idx] = ans;
        }
    }

    static void cudaCallback(
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
} // namespace v5
namespace v6 //cuda 预处理log到寄存器+使用更小的整型类型
{
    static __global__ void cudaCallbackKernel(
        const int width,
        const int height,
        const float *__restrict__ input,
        float *__restrict__ output)
    {
        const int idy = blockIdx.y * blockDim.y + threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idy < height && idx < width)
        {
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
            const double mylog[26] = {
                0.0,
                log(1.0),
                log(2.0),
                log(3.0),
                log(4.0),
                log(5.0),
                log(6.0),
                log(7.0),
                log(8.0),
                log(9.0),
                log(10.0),
                log(11.0),
                log(12.0),
                log(13.0),
                log(14.0),
                log(15.0),
                log(16.0),
                log(17.0),
                log(18.0),
                log(19.0),
                log(20.0),
                log(21.0),
                log(22.0),
                log(23.0),
                log(24.0),
                log(25.0)};

            const signed char n = (min(idx, 2) + 1 + min(width - idx, 2)) * (min(idy, 2) + 1 + min(height - idy, 2));
            double ans = mylog[n], n_inv = 1.0 / n;
            for (signed char i = 0; i < 16; ++i)
                ans -= mylog[cnt[i]] * cnt[i] * n_inv;
            output[idy * width + idx] = ans;
        }
    }

    static void cudaCallback(
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
} // namespace v6
namespace v7 //cuda 预处理log到寄存器+使用更小的整型类型+使用更小的浮点类型
{
    static __global__ void cudaCallbackKernel(
        const int width,
        const int height,
        const float *__restrict__ input,
        float *__restrict__ output)
    {
        const int idy = blockIdx.y * blockDim.y + threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idy < height && idx < width)
        {
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
            const float mylog[26] = {
                0.0,
                log(1.0),
                log(2.0),
                log(3.0),
                log(4.0),
                log(5.0),
                log(6.0),
                log(7.0),
                log(8.0),
                log(9.0),
                log(10.0),
                log(11.0),
                log(12.0),
                log(13.0),
                log(14.0),
                log(15.0),
                log(16.0),
                log(17.0),
                log(18.0),
                log(19.0),
                log(20.0),
                log(21.0),
                log(22.0),
                log(23.0),
                log(24.0),
                log(25.0)};

            const signed char n = (min(idx, 2) + 1 + min(width - idx, 2)) * (min(idy, 2) + 1 + min(height - idy, 2));
            double ans = mylog[n], n_inv = 1.0 / n;
            for (signed char i = 0; i < 16; ++i)
                ans -= mylog[cnt[i]] * n_inv * cnt[i];
            output[idy * width + idx] = ans;
        }
    }

    static void cudaCallback(
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
} // namespace v7
namespace v8 //cuda 预处理log到寄存器+使用更小的整型类型+使用更小的浮点类型+使用texure memory优化读入
{
    static texture<float, 2> input_tex;

    static __global__ void cudaCallbackKernel(
        const int width,
        const int height,
        float *__restrict__ output)
    {
        const int idy = blockIdx.y * blockDim.y + threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idy < height && idx < width)
        {
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
            const float mylog[26] = {
                0.0,
                log(1.0),
                log(2.0),
                log(3.0),
                log(4.0),
                log(5.0),
                log(6.0),
                log(7.0),
                log(8.0),
                log(9.0),
                log(10.0),
                log(11.0),
                log(12.0),
                log(13.0),
                log(14.0),
                log(15.0),
                log(16.0),
                log(17.0),
                log(18.0),
                log(19.0),
                log(20.0),
                log(21.0),
                log(22.0),
                log(23.0),
                log(24.0),
                log(25.0)};

            const signed char n = (min(idx, 2) + 1 + min(width - idx, 2)) * (min(idy, 2) + 1 + min(height - idy, 2));
            double ans = mylog[n], n_inv = 1.0 / n;
            for (signed char i = 0; i < 16; ++i)
                ans -= mylog[cnt[i]] * n_inv * cnt[i];
            output[idy * width + idx] = ans;
        }
    }

    static void cudaCallback(
        int width,
        int height,
        float *sample,
        float **result)
    {
        float *output_d;

        CHECK(cudaMalloc((void **)&output_d, sizeof(float) * width * height));
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaArray *cuArray;
        CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));
        CHECK(cudaMemcpyToArray(cuArray, 0, 0, sample, sizeof(float) * width * height, cudaMemcpyHostToDevice));
        CHECK(cudaBindTextureToArray(input_tex, cuArray));

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
            output_d);

        *result = (float *)malloc(sizeof(float) * width * height);
        CHECK(cudaMemcpy(*result, output_d, sizeof(float) * width * height, cudaMemcpyDeviceToHost));
        CHECK(cudaFreeArray(cuArray));
        CHECK(cudaFree(output_d));
    }
} // namespace v8
namespace v9 //cuda 预处理log到寄存器+使用更小的整型类型+使用更小的浮点类型+使用shared memory优化读入
{
    template <
        int BLOCK_DIM_X,
        int BLOCK_DIM_Y>
    static __global__ void cudaCallbackKernel(
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

            const float mylog[26] = {
                0.0,
                log(1.0),
                log(2.0),
                log(3.0),
                log(4.0),
                log(5.0),
                log(6.0),
                log(7.0),
                log(8.0),
                log(9.0),
                log(10.0),
                log(11.0),
                log(12.0),
                log(13.0),
                log(14.0),
                log(15.0),
                log(16.0),
                log(17.0),
                log(18.0),
                log(19.0),
                log(20.0),
                log(21.0),
                log(22.0),
                log(23.0),
                log(24.0),
                log(25.0)};

            const signed char n = (min(idx, 2) + 1 + min(width - idx, 2)) * (min(idy, 2) + 1 + min(height - idy, 2));
            double ans = mylog[n], n_inv = 1.0 / n;
            for (signed char i = 0; i < 16; ++i)
                ans -= mylog[cnt[i]] * n_inv * cnt[i];
            output[idy * width + idx] = ans;
        }
    }

    static void cudaCallback(
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
} // namespace v9
namespace v10 //openmp baseline
{
    static void cudaCallback(
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
            double n = (min(idx, 2) + 1 + min(width - idx, 2)) * (min(idy, 2) + 1 + min(height - idy, 2)), ans = log(n), n_inv = 1.0 / n;
            for (int i = 0; i < 16; ++i)
                if (cnt[i])
                    ans -= log((double)cnt[i]) * cnt[i] * n_inv;
            (*result)[pos] = ans;
        }
    }
} // namespace v10
namespace v11 //openmp 预处理log到寄存器
{
    static void cudaCallback(
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
            const double mylog[26] = {
                0.0,
                log(1.0),
                log(2.0),
                log(3.0),
                log(4.0),
                log(5.0),
                log(6.0),
                log(7.0),
                log(8.0),
                log(9.0),
                log(10.0),
                log(11.0),
                log(12.0),
                log(13.0),
                log(14.0),
                log(15.0),
                log(16.0),
                log(17.0),
                log(18.0),
                log(19.0),
                log(20.0),
                log(21.0),
                log(22.0),
                log(23.0),
                log(24.0),
                log(25.0)};
            const int n = (min(idx, 2) + 1 + min(width - idx, 2)) * (min(idy, 2) + 1 + min(height - idy, 2));
            double ans = mylog[n], n_inv = 1.0 / n;
            for (int i = 0; i < 16; ++i)
                ans -= mylog[cnt[i]] * n_inv * cnt[i];
            (*result)[pos] = ans;
        }
    }
} // namespace v11
namespace v12 //openmp 预处理log到寄存器+使用更小的类型
{
    static void cudaCallback(
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
            signed char cnt[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            for (signed char offsety = -2; offsety <= 2; ++offsety)
                for (signed char offsetx = -2; offsetx <= 2; ++offsetx)
                {
                    const int py = idy + offsety, px = idx + offsetx;
                    if (0 <= py && py < height && 0 <= px && px < width)
                        ++cnt[(signed char)sample[py * width + px]];
                }
            const float mylog[26] = {
                0.0,
                log(1.0),
                log(2.0),
                log(3.0),
                log(4.0),
                log(5.0),
                log(6.0),
                log(7.0),
                log(8.0),
                log(9.0),
                log(10.0),
                log(11.0),
                log(12.0),
                log(13.0),
                log(14.0),
                log(15.0),
                log(16.0),
                log(17.0),
                log(18.0),
                log(19.0),
                log(20.0),
                log(21.0),
                log(22.0),
                log(23.0),
                log(24.0),
                log(25.0)};
            const signed char n = (min(idx, 2) + 1 + min(width - idx, 2)) * (min(idy, 2) + 1 + min(height - idy, 2));
            double ans = mylog[n], n_inv = 1.0 / n;
            for (signed char i = 0; i < 16; ++i)
                ans -= mylog[cnt[i]] * n_inv * cnt[i];
            (*result)[pos] = ans;
        }
    }
} // namespace v12
namespace v13 //openmp 预处理log到寄存器+使用更小的类型+预处理前缀和
{
    static void cudaCallback(
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
            const float mylog[26] = {
                0.0,
                log(1.0),
                log(2.0),
                log(3.0),
                log(4.0),
                log(5.0),
                log(6.0),
                log(7.0),
                log(8.0),
                log(9.0),
                log(10.0),
                log(11.0),
                log(12.0),
                log(13.0),
                log(14.0),
                log(15.0),
                log(16.0),
                log(17.0),
                log(18.0),
                log(19.0),
                log(20.0),
                log(21.0),
                log(22.0),
                log(23.0),
                log(24.0),
                log(25.0)};
            const signed char n = (min(idx, 2) + 1 + min(width - idx, 2)) * (min(idy, 2) + 1 + min(height - idy, 2));
            double ans = mylog[n], n_inv = 1.0 / n;
            for (signed char i = 0; i < 16; ++i)
            {
                const signed char cnti = sum[i][(idy + 5) * (width + 5) + idx + 5] - sum[i][(idy + 5) * (width + 5) + idx] - sum[i][idy * (width + 5) + idx + 5] + sum[i][idy * (width + 5) + idx];
                ans -= mylog[cnti] * n_inv * cnti;
            }
            (*result)[pos] = ans;
        }
        for (int i = 0; i < 16; ++i)
            free(sum[i]);
    }
} // namespace v13
namespace v14 //cuda+openmp 多卡，基于v7、v9、v11
{
    static void cudaCallback(
        int width,
        int height,
        float *sample,
        float **result)
    {
        int num_threads = 0;
        CHECK(cudaGetDeviceCount(&num_threads));
        if (num_threads > height - 4)
            num_threads = height - 4;
        if (num_threads < 1)
            return v11::cudaCallback(width, height, sample, result);
        if (num_threads < 2 || width * height < 1e5)
            return v7::cudaCallback(width, height, sample, result);
        *result = (float *)malloc(sizeof(float) * width * height);
#pragma omp parallel num_threads(num_threads)
        {
            int thread_num = omp_get_thread_num(),
                thread_hgt = (height - 4) / num_threads,
                thread_beg = thread_hgt * thread_num + 2;
            if (thread_num == num_threads - 1)
                thread_hgt = height - 2 - thread_beg;
            float *thread_result;
            CHECK(cudaSetDevice(thread_num));
            v9::cudaCallback(
                width,
                thread_hgt + 4,
                sample + width * (thread_beg - 2),
                &thread_result);
            float
                *dst = (*result) + width * thread_beg,
                *src = thread_result + width * 2;
            if (thread_num == 0)
                dst -= width * 2, src -= width * 2, thread_hgt += 2;
            if (thread_num == num_threads - 1)
                thread_hgt += 2;
            memcpy(
                dst,
                src,
                sizeof(float) * width * thread_hgt);
            free(thread_result);
        }
    }
} // namespace v14
void cudaCallback(
    int width,
    int height,
    float *sample,
    float **result)
{
    v14::cudaCallback(width, height, sample, result);
}
