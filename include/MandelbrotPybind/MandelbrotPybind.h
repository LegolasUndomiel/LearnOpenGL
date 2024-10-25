#ifndef __MANDELBROTPYBIND_H__
#define __MANDELBROTPYBIND_H__

#ifdef USE_CUDA
#include <cuda_runtime.h>
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

#include <vector>

class Mandelbrot {
  private:
    /* data */
    unsigned short int width_, height_, maxIterations_;
    unsigned short int *data_;
#ifdef USE_CUDA
    unsigned short int *d_data_;
#endif

  public:
    Mandelbrot(unsigned short int, unsigned short int, unsigned short int);
    ~Mandelbrot();
    void pixelCalculation();
    void copyBack();
    std::vector<unsigned short int> getData();
};

#endif