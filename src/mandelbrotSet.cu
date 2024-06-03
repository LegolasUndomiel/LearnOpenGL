#include "mandelbrotSet.h"
#include "matplotlibcpp.h"
#include <chrono>
#include <omp.h>
#include <vector>

using std::cout;
using std::endl;
using std::vector;

using namespace std::chrono;
namespace plt = matplotlibcpp;

__host__ __device__ inline int mandelbrot(float real, float imag) {
    float r = real;
    float i = imag;
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        float r2 = r * r;
        float i2 = i * i;
        if (r2 + i2 > 4.0f) {
            return iter;
        }
        i = 2.0f * r * i + imag;
        r = r2 - i2 + real;
    }
    return MAX_ITERATIONS;
}

__global__ void mandelbrotKernel(float *data) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < WIDTH && y < HEIGHT) {
        float real = (x - WIDTH / 2.0) * 3.84 / WIDTH;
        float imag = (y - HEIGHT / 2.0) * 2.16 / HEIGHT;
        int value = mandelbrot(real, imag);
        data[y * WIDTH + x] = (float)value;
    }
}

void test02() {
    vector<float> data(WIDTH * HEIGHT, 0.0f);

    // 启动计时器
    auto start = high_resolution_clock::now();

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            float real = (x - WIDTH / 2.0) * 3.84 / WIDTH;
            float imag = (y - HEIGHT / 2.0) * 2.16 / HEIGHT;

            int value = mandelbrot(real, imag);

            data[y * WIDTH + x] = (float)value;
        }
    }

    // 停止计时器
    auto end = high_resolution_clock::now();

    // 计算持续时间
    duration<double, std::milli> duration = end - start;
    cout << "单线程版: " << duration.count() << "ms" << endl;

    // 保存图像 用时 5s 左右
    // 读写文件 用时 55s 左右
    const float *zptr = &(data[0]);
    const int colors = 1;
    plt::figure_size(1920, 1080);
    plt::imshow(zptr, HEIGHT, WIDTH, colors);
    plt::save("mandelbrot.png");
}

void test03() {
    vector<float> data(WIDTH * HEIGHT, 0.0f);

    // 启动计时器
    auto start = high_resolution_clock::now();

    int threads = omp_get_max_threads();
    omp_set_num_threads(threads);
#pragma omp parallel for
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        int x = i % WIDTH;
        int y = i / WIDTH;
        float real = (x - WIDTH / 2.0) * 3.84 / WIDTH;
        float imag = (y - HEIGHT / 2.0) * 2.16 / HEIGHT;
        int value = mandelbrot(real, imag);
        data[i] = (float)value;
    }

    // 停止计时器
    auto end = high_resolution_clock::now();

    // 计算持续时间
    duration<double, std::milli> duration = end - start;
    cout << "OpenMP第1版: " << duration.count() << "ms" << endl;

    const float *zptr = &(data[0]);
    const int colors = 1;
    plt::figure_size(1920, 1080);
    plt::imshow(zptr, HEIGHT, WIDTH, colors);
    plt::save("mandelbrot_OpenMP1.png");
}

void test04() {
    vector<float> data(WIDTH * HEIGHT, 0.0f);

    // 启动计时器
    auto start = high_resolution_clock::now();

    int threads = omp_get_max_threads();
    omp_set_num_threads(threads);
#pragma omp parallel for
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            float real = (x - WIDTH / 2.0) * 3.84 / WIDTH;
            float imag = (y - HEIGHT / 2.0) * 2.16 / HEIGHT;

            int value = mandelbrot(real, imag);

            data[y * WIDTH + x] = (float)value;
        }
    }

    // 停止计时器
    auto end = high_resolution_clock::now();

    // 计算持续时间
    duration<double, std::milli> duration = end - start;
    cout << "OpenMP第2版: " << duration.count() << "ms" << endl;

    const float *zptr = &(data[0]);
    const int colors = 1;
    plt::figure_size(1920, 1080);
    plt::imshow(zptr, HEIGHT, WIDTH, colors);
    plt::save("mandelbrot_OpenMP2.png");
}

void test05() {
    vector<float> data(WIDTH * HEIGHT, 0.0f);

    float *h_data = new float[WIDTH * HEIGHT];
    float *d_data;
    cudaMalloc((void **)&d_data, WIDTH * HEIGHT * sizeof(float));
    dim3 blockDim(32, 32);
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x,
                 (HEIGHT + blockDim.y - 1) / blockDim.y);

    // 启动计时器
    auto start = high_resolution_clock::now();
    mandelbrotKernel<<<gridDim, blockDim>>>(d_data);
    cudaMemcpy(h_data, d_data, WIDTH * HEIGHT * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    cudaDeviceSynchronize();

    // 停止计时器
    auto end = high_resolution_clock::now();

    // 计算持续时间
    duration<double, std::milli> duration = end - start;
    cout << "CUDA版: " << duration.count() << "ms" << endl;

    for (int i = 0; i < WIDTH * HEIGHT; i++)
        data[i] = h_data[i];

    delete[] h_data;

    const float *zptr = &(data[0]);
    const int colors = 1;
    plt::figure_size(1920, 1080);
    plt::imshow(zptr, HEIGHT, WIDTH, colors);
    plt::save("mandelbrot_CUDA.png");
}