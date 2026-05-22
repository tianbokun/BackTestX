@echo off
chcp 65001 >nul
title WSL 端口转发设置 (Streamlit 8501)

echo =============================================
echo  WSL 端口转发设置
echo  需要管理员权限运行
echo =============================================
echo.
echo 请右键本文件 -> "以管理员身份运行"
echo.
pause

:: 获取 WSL IP
for /f %%i in ('wsl ip addr show eth0 ^| findstr "inet "') do set wsl_ip=%%i
set wsl_ip=%wsl_ip:*.inet =%
set wsl_ip=%wsl_ip:/20=%
echo WSL IP: %wsl_ip%

:: 删除旧规则
netsh interface portproxy delete v4tov4 listenport=8501 listenaddress=0.0.0.0 >nul 2>&1

:: 添加端口转发
netsh interface portproxy add v4tov4 listenport=8501 listenaddress=0.0.0.0 connectport=8501 connectaddress=%wsl_ip%

:: 添加防火墙规则
netsh advfirewall firewall add rule name="WSL Streamlit 8501" dir=in action=allow protocol=TCP localport=8501 >nul 2>&1

echo.
echo =============================================
echo  当前端口转发规则:
netsh interface portproxy show v4tov4
echo.
echo ✅ 设置完成!
echo.
echo 现在可以通过以下地址访问:
echo   http://localhost:8501
echo =============================================
pause
