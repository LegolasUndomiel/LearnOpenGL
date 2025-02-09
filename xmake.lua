-- 模式添加
add_rules("mode.debug", "mode.release")

-- 设置编码格式
-- set_encodings("utf-8")

-- 设置编译标准
set_languages("c17", "cxx17")

set_kind("binary")
set_runtimes("MD")
set_targetdir("bin/$(plat)/$(arch)/$(mode)")

-- 自动更新Visual Studio解决方案
add_rules("plugin.vsxmake.autoupdate")
-- 自动更新compile_commands.json
add_rules("plugin.compile_commands.autoupdate", {outputdir = ".vscode"})

add_includedirs("include")

-- 设置自定义清理脚本
on_clean(function (target)
    print("All Files Deleted")
    -- 删除所有文件
    os.rm("$(buildir)")
    os.rm(target:targetdir())
end)

includes("**/xmake.lua")
