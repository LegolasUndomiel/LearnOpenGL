target("Imgui")

    add_includedirs("$(projectdir)/dependencies/GLFW/include")
    add_includedirs("$(projectdir)/dependencies/imgui")
    add_includedirs("$(projectdir)/dependencies/imgui/backends")

    add_files("*.cpp")
    add_files("$(projectdir)/dependencies/imgui/*.cpp")
    add_files("$(projectdir)/dependencies/imgui/backends/*.cpp")

    -- 系统(Visual Studio)默认
    add_links("OpenGL32", "user32", "gdi32", "shell32")
    -- 项目自带第三方库
    add_links("glfw3")
    add_linkdirs("$(projectdir)/dependencies/GLFW/lib")
target_end()

target("ImguiWasm")
    set_plat("wasm")

    add_includedirs("$(projectdir)/dependencies/GLFW/include")
    add_includedirs("$(projectdir)/dependencies/imgui")
    add_includedirs("$(projectdir)/dependencies/imgui/backends")
    add_includedirs("$(projectdir)/dependencies/emscripten")

    add_files("*.cpp")
    add_files("$(projectdir)/dependencies/imgui/*.cpp")
    add_files("$(projectdir)/dependencies/imgui/backends/*.cpp")

    -- 项目自带第三方库
    add_links("glfw3")
    add_linkdirs("$(projectdir)/dependencies/GLFW/lib")
target_end()