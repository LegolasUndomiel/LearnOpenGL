-- 模式添加
add_rules("mode.debug", "mode.release")

-- 设置编码格式
-- set_encodings("utf-8")

-- 设置编译标准
set_languages("c17", "cxx17")

set_kind("binary")
set_runtimes("MD")
set_targetdir("bin/$(mode)")
set_suffixname("-$(plat)-$(arch)-$(mode)")

-- 自动更新Visual Studio解决方案
add_rules("plugin.vsxmake.autoupdate")

add_includedirs("include")

-- 设置自定义清理脚本
on_clean(function (target)
    print("All Files Deleted")
    -- 删除所有文件
    os.rm("$(buildir)")
    os.rm(target:targetdir())
end)

target("LearnOpenGL")
    add_includedirs("dependencies/GLAD/include")
    add_includedirs("dependencies/GLFW/include")

    add_files("dependencies/GLAD/src/*.c")
    add_files("src/LearnOpenGL/*.cc")

    -- 系统(Visual Studio)默认
    add_links("OpenGL32", "user32", "gdi32", "shell32")
    -- 项目自带第三方库
    add_links("glfw3")
    add_linkdirs("dependencies/GLFW/lib")
target_end()

target("Mandelbrot")
    add_includedirs("dependencies/matplotlib-cpp")
    add_includedirs("$(env CONDA_PATH)/include")
    add_includedirs("$(env NUMPY_CORE)/include")
    add_includedirs("$(env CUDA_PATH)/include")

    add_files("src/mandelbrot/*.cu")

    -- OpenMP
    add_cxflags("/openmp")
    add_ldflags("/openmp")
    add_cuflags("-Xcompiler /openmp")
    add_culdflags("-Xcompiler /openmp")

    -- CUDA
    add_cugencodes("native")

    -- Anaconda
    add_links("python3")
    add_linkdirs("$(env CONDA_PATH)/libs")
target_end()

target("Imgui")
    add_includedirs("dependencies/GLFW/include")
    add_includedirs("dependencies/imgui")
    add_includedirs("dependencies/imgui/backends")

    add_files("src/imgui/*.cpp")
    add_files("dependencies/imgui/*.cpp")
    add_files("dependencies/imgui/backends/*.cpp")

    -- 系统(Visual Studio)默认
    add_links("OpenGL32", "user32", "gdi32", "shell32")
    -- 项目自带第三方库
    add_links("glfw3")
    add_linkdirs("dependencies/GLFW/lib")
target_end()

target("ImguiWasm")
    set_plat("wasm")

    add_includedirs("dependencies/GLFW/include")
    add_includedirs("dependencies/imgui")
    add_includedirs("dependencies/imgui/backends")
    add_includedirs("dependencies/emscripten")

    add_files("src/imgui/*.cpp")
    add_files("dependencies/imgui/*.cpp")
    add_files("dependencies/imgui/backends/*.cpp")

    -- 项目自带第三方库
    add_links("glfw3")
    add_linkdirs("dependencies/GLFW/lib")
target_end()
