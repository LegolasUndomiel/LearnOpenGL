#include "MandelbrotPybind/MandelbrotPybind.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(MandelbrotPybind, m) {
    m.doc() = "Mandelbrot";
    py::class_<Mandelbrot>(m, "Mandelbrot")
        .def(py::init<int, int, int>()) // (width, height, maxIterations)
        .def("pixelCalculation", &Mandelbrot::pixelCalculation)
        .def("copyBack", &Mandelbrot::copyBack)
        .def("getData", &Mandelbrot::getData);
}
