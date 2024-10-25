target("MandelbrotStatic_CUDA")
    set_kind("static")
    set_policy("build.cuda.devlink", true)

    add_includedirs("$(projectdir)/include/","$(env CUDA_PATH)/include")
    add_files("*.cu")

    add_defines("USE_CUDA")

    -- CUDA
    add_cugencodes("native")
target_end()

target("MandelbrotStatic_OpenMP")
    set_kind("static")
    set_policy("build.cuda.devlink", true)

    add_includedirs("$(projectdir)/include/")
    add_files("*.cu")

    -- OpenMP
    add_cuflags("-Xcompiler /openmp")
    add_culdflags("-Xcompiler /openmp")
target_end()

target("Mandelbrot_CUDA")
    set_kind("binary")
    add_includedirs("$(projectdir)/include/","$(env CUDA_PATH)/include")
    add_files("main.cpp")
    add_deps("MandelbrotStatic_CUDA")
target_end()

target("Mandelbrot_OpenMP")
    set_kind("binary")
    add_includedirs("$(projectdir)/include/")
    add_files("main.cpp")
    add_deps("MandelbrotStatic_OpenMP")
target_end()

target("MandelbrotPybind")
    set_kind("shared")
    set_extension(".pyd")

    add_includedirs("$(projectdir)/include/")
    add_includedirs("$(env CONDA_PATH)/include")
    add_includedirs("$(env NUMPY_CORE)/include")
    add_includedirs("$(projectdir)/dependencies/pybind11/include")

    add_files("MandelbrotPybind.cpp")
    add_deps("MandelbrotStatic_CUDA")

    add_links("python3")
    add_linkdirs("$(env CONDA_PATH)/libs")
target_end()