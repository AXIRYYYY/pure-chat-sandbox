@echo off
echo正在从 GitHub 更新所有工具模块...
git submodule update --remote --merge
echo.
echo 更新完成！
echo 请记得查看各个工具目录下的 requirements.txt 是否有变化。
