#include "Julia/Julia.h"
#include "matplotlibcpp.h"
#include <chrono>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

using std::cout;
using std::endl;
using std::string;
using std::vector;

using namespace std::chrono;
namespace plt = matplotlibcpp;

HOST_DEVICE inline unsigned short int
mandelbrot(float real, float imag, unsigned short int maxIterations) {
    float r = real;
    float i = imag;
    for (int iter = 0; iter < maxIterations; ++iter) {
        float r2 = r * r;
        float i2 = i * i;
        if (r2 + i2 > 4.0f) {
            return iter;
        }
        i = 2.0f * r * i + imag;
        r = r2 - i2 + real;
    }
    return maxIterations;
}

#ifdef USE_CUDA
__global__ void Kernel(unsigned short int *data, unsigned int short WIDTH,
                       unsigned short int HEIGHT,
                       unsigned short int maxIterations) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < WIDTH && y < HEIGHT) {
        float real = (x - WIDTH / 2.0) * 3.84 / WIDTH;
        float imag = (y - HEIGHT / 2.0) * 2.16 / HEIGHT;
        unsigned short int value = mandelbrot(real, imag, maxIterations);
        data[y * WIDTH + x] = value;
    }
}
#endif

Mandelbrot::Mandelbrot(unsigned short int width, unsigned short int height,
                       unsigned short int maxIterations) {
    this->width_ = width;
    this->height_ = height;
    this->maxIterations_ = maxIterations;
    this->data_ = new unsigned short int[width * height];
    for (int i = 0; i < width * height; i++)
        this->data_[i] = 0;

#ifdef USE_CUDA
    cudaMalloc((void **)&this->d_data_,
               width * height * sizeof(unsigned short int));
#endif
}

Mandelbrot::~Mandelbrot() {
    delete[] this->data_;
    cout << "释放CPU内存" << endl;
#ifdef USE_CUDA
    cudaFree(this->d_data_);
    cout << "释放GPU内存" << endl;
#endif
}

void Mandelbrot::pixelCalculation() {
#ifdef USE_CUDA
    dim3 blockDim(32, 32);
    dim3 gridDim((this->width_ + blockDim.x - 1) / blockDim.x,
                 (this->height_ + blockDim.y - 1) / blockDim.y);

    // 启动计时器
    auto start = high_resolution_clock::now();

    Kernel<<<gridDim, blockDim>>>(this->d_data_, this->width_, this->height_,
                                  this->maxIterations_);
    cudaDeviceSynchronize();

    // 停止计时器
    auto end = high_resolution_clock::now();
    duration<double, std::milli> duration = end - start;

    cout << "CUDA版: " << duration.count() << "ms" << endl;
#else
    // 启动计时器
    auto start = high_resolution_clock::now();

    int threads = omp_get_max_threads();
    omp_set_num_threads(threads);
#pragma omp parallel for
    for (int y = 0; y < this->height_; ++y) {
        for (int x = 0; x < this->width_; ++x) {
            float real = (x - this->width_ / 2.0) * 3.84 / this->width_;
            float imag = (y - this->height_ / 2.0) * 2.16 / this->height_;

            unsigned short int value =
                mandelbrot(real, imag, this->maxIterations_);

            this->data_[y * this->width_ + x] = value;
        }
    }

    // 停止计时器
    auto end = high_resolution_clock::now();

    // 计算持续时间
    duration<double, std::milli> duration = end - start;
    cout << "OpenMP版: " << duration.count() << "ms" << endl;
#endif
}

void Mandelbrot::copyBack() {
#ifdef USE_CUDA
    cudaMemcpy(this->data_, this->d_data_,
               this->width_ * this->height_ * sizeof(unsigned short int),
               cudaMemcpyDeviceToHost);
#endif
}

void Mandelbrot::save() {
    vector<float> data(this->width_ * this->height_, 0.0f);
    for (int i = 0; i < this->width_ * this->height_; i++)
        data[i] = (float)this->data_[i];

    const float *zptr = &(data[0]);
    const int colors = 1;
    plt::figure_size(1920, 1080);
    plt::imshow(zptr, this->height_, this->width_, colors);
#ifdef USE_CUDA
    plt::save("mandelbrot_CUDA.png");
#else
    plt::save("mandelbrot_OpenMP.png");
#endif
}

void test01() {
    Mandelbrot m(1920, 1080, 8000);
    m.pixelCalculation();
    m.copyBack();
    m.save();
}

int main(int argc, char const *argv[]) {
    test01();
    return 0;
}
