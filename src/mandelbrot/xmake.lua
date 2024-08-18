target("Mandelbrot")
    add_includedirs("$(projectdir)/dependencies/matplotlib-cpp")
    add_includedirs("$(env CONDA_PATH)/include")
    add_includedirs("$(env NUMPY_CORE)/include")
    add_includedirs("$(env CUDA_PATH)/include")

    add_files("*.cu")

    -- OpenMP
    add_cuflags("-Xcompiler /openmp")
    add_culdflags("-Xcompiler /openmp")

    -- CUDA
    add_cugencodes("native")

    -- Anaconda
    add_links("python3")
    add_linkdirs("$(env CONDA_PATH)/libs")
target_end()
