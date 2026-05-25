<div align="center">
  <h1>📈 A股定投回测系统</h1>
  <p><strong>A-Share DCA Backtest System</strong></p>
  <p>多策略对比 · Walk-Forward 网格搜索 · 智能定投 · Streamlit Web App</p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.10-blue" alt="Python">
    <img src="https://img.shields.io/badge/Streamlit-1.57-red" alt="Streamlit">
    <img src="https://img.shields.io/badge/AKShare-1.18-green" alt="AKShare">
    <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
  </p>
</div>

---

## 🌟 简介 | Introduction

A股定投回测系统是一个功能全面的 **定投策略回测 Web 应用**，支持：

- **6 种定投频率**：每日 / 每周 / 每两周 / 每月 / 每季度 / 每年
- **5 种智能定投策略**：均线偏离 / 成本定投 / 价值平均 / 趋势定投 / 支付宝慧定投
- **事件驱动策略**：下跌加仓法（跌幅触发买入）
- **一次性投入对比**：作为基准对照
- **Walk-Forward 网格搜索**：自动寻找最优参数组合
- **多策略同图叠加对比**：收益率、持仓市值、累计投入一览无余

> 数据来源：东方财富 (East Money) → AKShare 接口，支持个股 / ETF / LOF / 开放式基金 / 指数，自动缓存、自动降级、自动跨类型回退。

---

## ✨ 核心特性 | Features

### 📊 定投回测
| 特性 | 说明 |
|------|------|
| 多频率对比 | 最多 6 种频率同图对比，高亮最优收益率 |
| 日均投入语义 | 输入"日均投入"，各频率自动按交易日乘数换算，投入可比 |
| XIRR 年化 | 牛顿法求解内部收益率，精确计算不规则现金流年化收益 |
| 总投资上限 | 达到上限自动停止，模拟真实资金约束 |
| 市值 / 投入 / 收益率三图 | Plotly 交互式叠加图，hover 统一显示 |

### 🧠 智能策略
| 策略 | 核心逻辑 | 参考来源 |
|------|----------|----------|
| **均线偏离法** | 价格低于均线多投(≤2x)，高于均线少投(≥0.5x) | 支付宝/天天基金慧定投 |
| **成本定投法** | 净值低于持仓成本多投，高于成本少投 | 各平台成本模式 |
| **价值平均法** | 设定目标市值路径，不足则补足 | Michael Edelson 经典 |
| **趋势定投** | MA20/MA120 金叉多投(1.5x)，死叉少投(0.5x) | 华安基金/各银行 |
| **支付宝慧定投** | 均线偏离 + 10日振幅双因子调节(0.6~2.1x) | 支付宝官方算法 |
| **下跌加仓** | 日跌幅超过 X% 时买入 Y 元，事件驱动 | 常见量化策略 |

### 🔬 网格搜索
- **超参数遍历**：对 X（跌幅阈值）、Y（买入金额）进行全面网格扫描
- **Walk-Forward 交叉验证**：将时间序列切为 N 折，每折依次作为验证集，防止过拟合
- **全局最优 + 每折最优**：展示不同市场环境下的最优参数变化
- **参数热力图**：RdYlGn 色阶直观展示参数敏感度
- **结果持久化**：每次搜索自动保存（Parquet + JSON），支持历史结果加载回看

### 🛡️ 数据层
- **5 种资产类型**：个股、ETF、LOF、开放式基金、指数
- **自动跨类型回退**：选错类型自动尝试其他类型
- **QDII 自动降级**：ETF/LOF 的 K 线失败时自动切换净值 API
- **30 天本地缓存**：Parquet 格式，第二次查询同代码只需 0.2 秒
- **请求重试**：指数退避 + 浏览器级 Header，最大限度提升成功率

---

## 🚀 快速开始 | Quick Start

### 安装

```bash
git clone https://github.com/your-username/stock-dca-backtest.git
cd stock-dca-backtest
pip install -r requirements.txt
```

### 运行

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

打开浏览器访问 `http://localhost:8501`

### 生产部署 (PM2)

```bash
# 安装 PM2
npm install -g pm2

# 启动
pm2 start ecosystem.config.js

# 查看状态
pm2 status

# 保存进程列表（开机自启）
pm2 save
pm2 startup
```

---

## 🖥️ 界面预览 | UI Preview

> 建议截图并放置于此，例如：
>
> - `docs/screenshots/dca_mode.png`  — 定投回测模式（多策略对比图）
> - `docs/screenshots/grid_search.png` — 网格搜索模式（热力图 + Walk-Forward 结果）
> - `docs/screenshots/comparison.png`  — 全能对比（13+ 策略同图）

---

## 🧩 项目结构 | Project Structure

```
stock_history_analysis/
├── app.py                    # Streamlit 主应用（双模式 UI）
├── data_fetcher.py           # 数据获取层（缓存、重试、降级、回退）
├── backtest/
│   ├── dca.py                # 定投回测引擎（6 频率 + XIRR + 一次性投入）
│   ├── strategies.py         # 智能策略实现（5 种 + 下跌加仓）
│   ├── grid_search.py        # 网格搜索引擎（Walk-Forward CV + 持久化）
│   └── strategy_results/     # 网格搜索结果存储目录
├── cache/                    # Parquet 数据缓存（30 天过期）
├── logs/                     # PM2 运行日志
├── requirements.txt          # Python 依赖
├── ecosystem.config.js       # PM2 进程管理配置
├── start_streamlit.sh        # 自动端口检测启动脚本
└── wsl_port_forward.*        # WSL2 端口转发脚本（Windows 访问）
```

---

## 📐 金额与频率换算 | Amount Semantics

系统采用 **"日均投入"** 作为统一的金额输入语义，各频率自动换算：

| 频率 | 交易日乘数 | 每期实际投入 |
|------|-----------|-------------|
| 每日 | ×1 | 日均 × 1 |
| 每周 | ×5 | 日均 × 5 |
| 每两周 | ×10 | 日均 × 10 |
| 每月 | ×22 | 日均 × 22 |
| 每季度 | ×66 | 日均 × 66 |
| 每年 | ×252 | 日均 × 252 |

例如：日均投入 100 元 → 每月实际投 2,200 元，每年实际投 25,200 元。

这样设置后，**各频率的总投入在相同时间段内基本一致**，收益率可以直接公平对比。

---

## 🔧 技术栈 | Tech Stack

| 组件 | 技术 |
|------|------|
| Web UI | [Streamlit](https://streamlit.io/) |
| 数据源 | [AKShare](https://github.com/akfamily/akshare) + 东方财富直连 |
| 图表 | [Plotly](https://plotly.com/python/) (go.Figure + Heatmap) |
| 数据处理 | Pandas / NumPy |
| 部署 | PM2 / systemd / WSL2 |
| 缓存格式 | Apache Parquet |

---

## ⚠️ 免责声明 | Disclaimer

> **本工具仅供学习研究使用，回测历史收益不代表未来表现，不构成任何投资建议。**
> 数据来源于东方财富（AKShare），可能存在延迟或不准确的情况。
> 在使用本工具做出的任何投资决策，风险由使用者自行承担。

---

## 📄 License

MIT License — 详见 [LICENSE](LICENSE) 文件。

---

## ⭐ Star 历史

如果你觉得这个项目对你有帮助，欢迎点亮右上角的 Star！你的支持是持续改进的动力。

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/stock-dca-backtest&type=Date)](https://star-history.com/#your-username/stock-dca-backtest&Date)
