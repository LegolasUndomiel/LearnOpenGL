#include "Julia/Julia.h"

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