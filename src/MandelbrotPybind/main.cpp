#include "MandelbrotPybind/MandelbrotPybind.h"

void test01() {
    Mandelbrot m(1920, 1080, 8000);
    m.pixelCalculation();
    m.copyBack();
}

int main(int argc, char const *argv[]) {
    test01();
    return 0;
}