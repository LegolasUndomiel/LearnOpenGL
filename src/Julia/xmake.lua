target("Julia_CUDA")
    add_includedirs("$(projectdir)/dependencies/matplotlib-cpp")
    add_includedirs("$(env CONDA_INCLUDE)")
    add_includedirs("$(env NUMPY_CORE)/include")
    add_includedirs("$(env CUDA_PATH)/include")

    add_files("*.cu")
    add_files("*.cpp")

    add_defines("USE_CUDA")

    -- CUDA
    add_cugencodes("native")

    -- Anaconda
    add_links("python3")
    add_linkdirs("$(env CONDA_LIBS)")
target_end()

target("Julia_OpenMP")
    add_includedirs("$(projectdir)/dependencies/matplotlib-cpp")
    add_includedirs("$(env CONDA_INCLUDE)")
    add_includedirs("$(env NUMPY_CORE)/include")

    add_files("*.cu")
    add_files("*.cpp")

    -- OpenMP
    add_cuflags("-Xcompiler /openmp")
    add_culdflags("-Xcompiler /openmp")

    -- Anaconda
    add_links("python3")
    add_linkdirs("$(env CONDA_LIBS)")
target_end()
