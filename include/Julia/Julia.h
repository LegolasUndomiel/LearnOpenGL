#ifndef __JULIA_H__
#define __JULIA_H__

#ifdef USE_CUDA
#include <cuda_runtime.h>
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

class Julia {
  private:
    /* data */
  public:
    Julia(/* args */);
    ~Julia();
};

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
    void save();
};

#endif