-- 模式添加
add_rules("mode.debug", "mode.release")

-- 设置编码格式
set_encodings("utf-8")

-- 设置编译标准
set_languages("c++17")

-- 自动更新Visual Studio解决方案
add_rules("plugin.vsxmake.autoupdate")

-- 设置自定义清理脚本
on_clean(function (target)
    print("All Files Deleted")
    -- 删除所有文件
    os.rm("$(buildir)")
    os.rm(target:targetdir())
end)

target("LearnOpenGL")
    set_kind("binary")
    set_runtimes("MD")
    set_targetdir("bin/$(mode)")
    set_suffixname("-$(plat)-$(arch)-$(mode)")

    add_includedirs("include")
    add_includedirs("C:/Program Files/Matplot++ 1.2.0/include")

    add_files("src/*.c")
    add_files("src/*.cc")

    add_links("OpenGL32", "glfw3", "user32", "gdi32", "shell32", "matplot", "nodesoup")
    add_linkdirs("lib", "$(env MATPLOT_PATH)/lib", "$(env MATPLOT_PATH)/lib/Matplot++")
target_end()