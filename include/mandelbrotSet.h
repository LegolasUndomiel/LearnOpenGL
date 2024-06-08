#ifndef __MANDELBROT_SET_H__
#define __MANDELBROT_SET_H__

#define WIDTH (1920 * 1)
#define HEIGHT (1080 * 1)
#define MAX_ITERATIONS 8000
#include <cuda_runtime.h>

__host__ __device__ inline int mandelbrot(float real, float imag);
__global__ void mandelbrotKernel(float *data);
void test02();
void test03();
void test04();
void test05();

#endif