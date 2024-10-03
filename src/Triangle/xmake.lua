target("Triangle")
    -- add_includedirs("$(projectdir)/dependencies/GLAD/include")
    add_includedirs("$(projectdir)/dependencies/glew/include")
    add_includedirs("$(projectdir)/dependencies/GLFW/include")

    -- add_files("$(projectdir)/dependencies/GLAD/src/*.c")
    add_files("*.cc")
    add_defines("GLEW_STATIC")

    -- 系统(Visual Studio)默认
    add_links("OpenGL32", "user32", "gdi32", "shell32")
    -- 项目自带第三方库
    add_links("glew32s", "glfw3")
    add_linkdirs("$(projectdir)/dependencies/GLFW/lib")
    add_linkdirs("$(projectdir)/dependencies/glew/lib/Release/x64")
target_end()