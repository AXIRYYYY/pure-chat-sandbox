@echo off
:: 切换代码页至 UTF-8
chcp 65001 >nul
echo 正在唤醒纯净沙盒环境...
:: 检查是否存在 streamlit 命令
where streamlit >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ 错误：未发现 Streamlit，请先运行 pip install streamlit
    pause
    exit
)
:: 核心指令：拉起服务
streamlit run app.py
pause