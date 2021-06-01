#include "core.h"
#include <cuda_fp16.h>

namespace v0 //cuda baseline
{
    static __global__ void cudaCallbackKernel( //调用的核函数
        const int width,                       // 输入矩阵宽，下同
        const int height,                      //输入矩阵高，下同
        const float *__restrict__ input,       //输入矩阵
        float *__restrict__ output)            //输出矩阵
    {
        const int idy = blockIdx.y * blockDim.y + threadIdx.y; //该线程对应元素的行坐标
        const int idx = blockIdx.x * blockDim.x + threadIdx.x; //该线程对应元素的列坐标
        if (idy < height && idx < width)
        {
            int cnt[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; //循环中统计每个元素的出现次数
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
                n = (min(idx, 2) + 1 + min(width - idx, 2)) * (min(idy, 2) + 1 + min(height - idy, 2)), //当前位置邻域的大小
                n_inv = 1.0 / n,
                ans = log(n); //ans = logn - n_i/n*log(n_i)
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
        //接下来在显卡上分配内存空间，并将输入拷贝到显卡上
        CHECK(cudaMalloc((void **)&output_d, sizeof(float) * width * height));
        CHECK(cudaMalloc((void **)&input_d, sizeof(float) * width * height));
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
        //将结果写回，并释放显存空间
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
                0.0, //log 0设置为0
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
                log(25.0)}; //预处理对数表到寄存器。此处计算在编译时就已经完成

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
        //shared memory不允许直接初始化，要在运行的时候由每个线程计算
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
            log(25.0)}; //计算并将值发送到constant memory
        CHECK(cudaMemcpyToSymbol(mylog, (const double *)mylog_h, sizeof(mylog_h)));
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
        CHECK(cudaMemcpyToSymbol(mylog, (const double *)mylog_h, sizeof(mylog_h)));
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
    static __device__ float mylog[26]; //texture只允许4字节的float

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
            signed char cnt[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; //寄存器类型从int改成char，减少了4倍寄存器压力
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
    template <int BLOCK_DIM_X>
    static __global__ __launch_bounds__(BLOCK_DIM_X) void cudaCallbackKernel(
        const int width,
        const int height,
        const float *__restrict__ input,
        float *__restrict__ output)
    {
        const int idy = blockIdx.y;
        const int idx = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
        if (idy < height && idx < width)
        {
            __shared__ signed char cnts[16][BLOCK_DIM_X];
            signed char *cnt = cnts[0] + threadIdx.x;
            for (signed char i = 0; i < 16; ++i)
                cnt[i * BLOCK_DIM_X] = 0;
            for (signed char offsety = -2; offsety <= 2; ++offsety)
            {
                const int py = idy + offsety;
                if (0 <= py && py < height)
                    for (signed char offsetx = -2; offsetx <= 2; ++offsetx)
                    {
                        const int px = idx + offsetx;
                        if (0 <= px && px < width)
                            ++cnt[(int)input[py * width + px] * BLOCK_DIM_X];
                    }
            }

            const float mylog[24] = {
                2 * log(2.0),
                3 * log(3.0),
                4 * log(4.0),
                5 * log(5.0),
                6 * log(6.0),
                7 * log(7.0),
                8 * log(8.0),
                9 * log(9.0),
                10 * log(10.0),
                11 * log(11.0),
                12 * log(12.0),
                13 * log(13.0),
                14 * log(14.0),
                15 * log(15.0),
                16 * log(16.0),
                17 * log(17.0),
                18 * log(18.0),
                19 * log(19.0),
                20 * log(20.0),
                21 * log(21.0),
                22 * log(22.0),
                23 * log(23.0),
                24 * log(24.0),
                25 * log(25.0)};

            const signed char n = (min(idx, 2) + 1 + min(width - idx, 2)) * (min(idy, 2) + 1 + min(height - idy, 2));
            double ans = mylog[n - 2];
            for (signed char i = 0; i < 16; ++i)
            {
                signed char c = cnt[i * BLOCK_DIM_X] - (signed char)2;
                if (c >= 0)
                    ans -= mylog[c];
            }
            output[idy * width + idx] = ans / n;
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
            BLOCK_DIM_X = 512;

        const dim3
            blockDim(BLOCK_DIM_X),
            gridDim(divup(width, BLOCK_DIM_X), height);

        cudaCallbackKernel<BLOCK_DIM_X><<<
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
    static __global__ void cudaCallbackKernel(
        cudaTextureObject_t texObj, //使用纹理对象
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
                            ++cnt[(signed char)tex2D<float>(texObj, px, py)];
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
        cudaArray *cuArray;
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));
        CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, sample, sizeof(float) * width, sizeof(float) * width, height, cudaMemcpyHostToDevice));

        // 绑定纹理到cudaArray上
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;

        // 设置纹理为只读
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;

        // 创建纹理对象
        cudaTextureObject_t texObj = 0;
        CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

        const int
            BLOCK_DIM_X = 32,
            BLOCK_DIM_Y = 32;

        const dim3
            blockDim(BLOCK_DIM_X, BLOCK_DIM_Y),
            gridDim(divup(width, BLOCK_DIM_X), divup(height, BLOCK_DIM_Y));

        cudaCallbackKernel<<<
            gridDim,
            blockDim>>>(
            texObj,
            width,
            height,
            output_d);

        *result = (float *)malloc(sizeof(float) * width * height);
        CHECK(cudaMemcpy(*result, output_d, sizeof(float) * width * height, cudaMemcpyDeviceToHost))
        CHECK(cudaDestroyTextureObject(texObj));
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

        //读入shared memory
        __shared__ char input_s[BLOCK_DIM_Y][BLOCK_DIM_X | 1];
        //溢出的值用16代替
        input_s[threadIdx.y][threadIdx.x] = 0 <= idy && idy < height && 0 <= idx && idx < width ? input[idy * width + idx] : 16;

        __syncthreads();

        if (1 < threadIdx.y && threadIdx.y < BLOCK_DIM_Y - 2 &&
            1 < threadIdx.x && threadIdx.x < BLOCK_DIM_X - 2 &&
            idy < height && idx < width)
        {
            signed char cnt[17] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; //此处计数器多开一位用于非法值
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
#pragma omp parallel for //每个位置没有循环依赖，可以直接并行
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
                log(25.0)}; //预处理对数表，其值在编译时已求得
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
#pragma omp parallel for //此处预处理 X =x_i 对答案的贡献的前缀和
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
                        ++p[pos]; //当前位置上的元素是i的话可更新
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
                const signed char cnti = sum[i][(idy + 5) * (width + 5) + idx + 5] - sum[i][(idy + 5) * (width + 5) + idx] - sum[i][idy * (width + 5) + idx + 5] + sum[i][idy * (width + 5) + idx]; //用前缀和公式计算
                ans -= mylog[cnti] * n_inv * cnti;
            }
            (*result)[pos] = ans;
        }
        for (int i = 0; i < 16; ++i)
            free(sum[i]);
    }
} // namespace v13
namespace v14 //cuda+openmp 多卡，基于v7、v12
{
    static void cudaCallback(
        int width,
        int height,
        float *sample,
        float **result)
    {
        int num_gpus = 0;
        CHECK(cudaGetDeviceCount(&num_gpus));
        if (num_gpus > height - 4) //显卡远多于可划分数据时适当减少使用的显卡
            num_gpus = height - 4;
        if (num_gpus < 1 || width * height < (80 * 2048)) //单张V100有80个SM，每个SM最多2048个常驻线程，不能满载时直接使用
            return v12::cudaCallback(width, height, sample, result);
        if (num_gpus < 2) //只有一张显卡时直接调用单卡版本减少开销
            return v7::cudaCallback(width, height, sample, result);
        *result = (float *)malloc(sizeof(float) * width * height);
#pragma omp parallel num_threads(num_gpus)
        {
            int thread_num = omp_get_thread_num(),
                thread_hgt = (height - 4) / num_gpus, //每个线程实际有效的height长度
                thread_beg = thread_hgt * thread_num + 2;
            if (thread_num == num_gpus - 1) //最后一个线程特判，因为不一定整除
                thread_hgt = height - 2 - thread_beg;
            float *thread_result;
            CHECK(cudaSetDevice(thread_num)); //不同线程指定不同显卡
            v7::cudaCallback(                 //划分为子问题，分别交给单卡版本
                width,
                thread_hgt + 4,
                sample + width * (thread_beg - 2),
                &thread_result);
            float
                *dst = (*result) + width * thread_beg,
                *src = thread_result + width * 2;
            if (thread_num == 0) //0号线程输出的上边界也是有效的
                dst -= width * 2, src -= width * 2, thread_hgt += 2;
            if (thread_num == num_gpus - 1) //最后一个线程的下边界也是有效的
                thread_hgt += 2;
            memcpy( //将子问题的答案拷贝回原问题
                dst,
                src,
                sizeof(float) * width * thread_hgt);
            free(thread_result); //释放子问题的内存空间
        }
    }
} // namespace v14
struct WarmUP
{
    WarmUP(int W, int H)
    {
        void (*cudaCallback[])(int, int, float *, float **) = {
            v0::cudaCallback,
            v1::cudaCallback,
            v2::cudaCallback,
            v3::cudaCallback,
            v4::cudaCallback,
            v5::cudaCallback,
            v6::cudaCallback,
            v7::cudaCallback,
            v8::cudaCallback,
            v9::cudaCallback,
            v10::cudaCallback,
            v11::cudaCallback,
            v12::cudaCallback,
            v13::cudaCallback}; //由于多卡版本是调用单卡版本实现的，因此无需热身
        float *sample = (float *)malloc(sizeof(float) * W * H);
#pragma omp parallel
        {
            unsigned seed = omp_get_thread_num(); //每个线程使用不同的随机数种子
#pragma omp for
            for (int i = 0; i < W * H; ++i)
                sample[i] = rand_r(&seed) & 15; //使用线程安全的随机数函数
        }
        for (int i = 0; i < sizeof(cudaCallback) / sizeof(cudaCallback[0]); ++i)
        {
            int num_gpus = 0;
            CHECK(cudaGetDeviceCount(&num_gpus));
#pragma omp parallel num_threads(num_gpus) //对于每张显卡都要优化
            {
                float *result;
                int thread_num = omp_get_thread_num();
                CHECK(cudaSetDevice(thread_num));
                cudaCallback[i](W, H, sample, &result);
                free(result);
            }
        }
        free(sample);
    }
};
struct Benchmark
{
    Benchmark(int W, int H)
    {
        void (*cudaCallback[])(int, int, float *, float **) = {
            v0::cudaCallback,
            v1::cudaCallback,
            v2::cudaCallback,
            v3::cudaCallback,
            v4::cudaCallback,
            v5::cudaCallback,
            v6::cudaCallback,
            v7::cudaCallback,
            v8::cudaCallback,
            v9::cudaCallback,
            v10::cudaCallback,
            v11::cudaCallback,
            v12::cudaCallback,
            v13::cudaCallback,
            v14::cudaCallback};
        float *sample = (float *)malloc(sizeof(float) * W * H);
#pragma omp parallel
        {
            unsigned seed = omp_get_thread_num();
#pragma omp for
            for (int i = 0; i < W * H; ++i)
                sample[i] = rand_r(&seed) & 15;
        }
        printf("\n\nStart benchmark with matrix size %d * %d:\n\n", W, H); //开始benchnmark
        for (int i = 0; i < sizeof(cudaCallback) / sizeof(cudaCallback[0]); ++i)
        {
            float *result;
            cudaEvent_t beg, end;
            cudaEventCreate(&beg);
            cudaEventCreate(&end);
            cudaEventRecord(beg);
            cudaCallback[i](W, H, sample, &result);
            cudaEventRecord(end);
            cudaEventSynchronize(beg);
            cudaEventSynchronize(end);
            float elapsed_time;
            cudaEventElapsedTime(
                &elapsed_time,
                beg,
                end);
            printf("Version %d: %fms\n", i, elapsed_time);
            free(result);
        }
        printf("\n\nFinish benchmark with matrix size %d * %d.\n\n", W, H);
        free(sample);
    }
};
static WarmUP warm_up(1, 1);
static Benchmark
    benchmark400(400, 400),
    benchmark2560(2560, 2560),
    benchmark10240(10240, 10240);
void cudaCallback(
    int width,
    int height,
    float *sample,
    float **result)
{
    v14::cudaCallback(width, height, sample, result);
}