target("OpenGL-Window")
    add_includedirs("$(projectdir)/dependencies/GLAD/include")
    add_includedirs("$(projectdir)/dependencies/GLFW/include")

    add_files("$(projectdir)/dependencies/GLAD/src/*.c")
    add_files("*.cc")

    -- 系统(Visual Studio)默认
    add_links("OpenGL32", "user32", "gdi32", "shell32")
    -- 项目自带第三方库
    add_links("glfw3")
    add_linkdirs("$(projectdir)/dependencies/GLFW/lib")
target_end()