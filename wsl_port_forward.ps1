# WSL 端口转发设置脚本
# 以管理员身份运行: 右键 -> "以 PowerShell 管理员身份运行"

$wslPort = 8501
$windowsPort = 8501

# 获取 WSL IP
$wslIp = wsl ip addr show eth0 | Select-String "inet " | ForEach-Object { $_ -replace '.*inet (\d+\.\d+\.\d+\.\d+).*', '$1' }
Write-Host "WSL IP: $wslIp"

# 删除旧规则 (如果有)
netsh interface portproxy delete v4tov4 listenport=$windowsPort listenaddress=0.0.0.0 2>$null

# 添加新规则
netsh interface portproxy add v4tov4 listenport=$windowsPort listenaddress=0.0.0.0 connectport=$wslPort connectaddress=$wslIp

Write-Host "Port forwarding: Windows :$windowsPort -> WSL :$wslPort"
netsh interface portproxy show v4tov4

# 添加防火墙规则 (允许入站)
New-NetFirewallRule -DisplayName "WSL Streamlit $windowsPort" -Direction Inbound -LocalPort $windowsPort -Protocol TCP -Action Allow 2>$null

Write-Host ""
Write-Host "✅ 设置完成!"
Write-Host "现在可以通过以下地址访问:"
Write-Host "  - http://localhost:$windowsPort"
Write-Host "  - http://本机IP:$windowsPort (局域网其他设备)"
