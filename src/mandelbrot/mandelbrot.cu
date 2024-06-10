#include "mandelbrot/mandelbrotSet.h"

int main(int argc, char const *argv[]) {
    test02(); // Mandelbrot Set Single Thread
    test03(); // Mandelbrot Set OpenMP1
    test04(); // Mandelbrot Set OpenMP2
    test05(); // Mandelbrot Set CUDA

    return 0;
}
