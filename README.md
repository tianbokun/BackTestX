<div align="center">
  <h1>📈 A股量化分析 · 定投回测 · 市场情绪系统</h1>
  <p><strong>Quantitative Analysis · DCA Backtest · Sentiment Dashboard for China A-Share Market</strong></p>
  <p>多策略对比 · Walk-Forward 网格搜索 · 智能定投 · 强化学习交易 · 板块情绪分析 · Streamlit Web App</p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.10-blue" alt="Python">
    <img src="https://img.shields.io/badge/Streamlit-1.57-red" alt="Streamlit">
    <img src="https://img.shields.io/badge/AKShare-1.18-green" alt="AKShare">
    <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
    <img src="https://img.shields.io/github/last-commit/your-username/stock_history_analysis" alt="Last Commit">
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
- **🤖 DQN 强化学习交易系统**：RL 交易智能体，支持实时信号、模型持久化、超参搜索
- **🧠 分层强化学习 (PPO+DQN)**：上层 PPO 择时 + 下层 DQN 选股，多 ETF 组合管理，交易记录明细，多基准对比（等权持有/月定投/均线偏离定投/单ETF全仓）
- **📋 代码管理中心**：统一代码注册表，缓存优先数据获取，RL 勾选使用
- **📊 情绪数据看板**：东方财富股吧/新闻情感分析 + 板块综合情绪排行（5 因子加权），一键导出 PNG 报告
- **🔄 后台任务管理**：RL 训练/HP 搜索异步队列，实时进度曲线，结果持久化，重训练

> 数据来源：东方财富 (East Money) → AKShare 接口，支持个股 / ETF / LOF / 开放式基金 / 指数，自动缓存、自动降级、自动跨类型回退。

>> **强化学习**：基于 DQN 的交易智能体，支持 3 种系统版本（basic/1.0/2.0），Walk-Forward 超参搜索，模型持久化，实时交易信号。

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

### 🤖 强化学习交易系统

| 特性 | 说明 |
|------|------|
| **算法** | DQN (Deep Q-Network) 双网络架构，经验回放 |
| **状态空间** | 18 维特征向量（价格、均线、布林带、ATR、ADX、CCI、KDJ、RSI、成交量、溢价率） |
| **动作空间** | 离散 {-1, 0, 1} = {卖出, 持有, 买入}，全仓交易 |
| **奖励函数** | 滚动窗口组合收益率（10 步），最后一笔使用总收益率 |
| **交易成本** | 佣金万2.35（最低5元）+ 卖出印花税千1 |
| **系统版本** | basic（仅价格）、1.0（+技术指标）、2.0（+SVM+XGBoost 涨跌信号） |
| **超参搜索** | Walk-Forward 3 折 × 5 参数网格（324 组合），以夏普比率选优 |
| **模型持久化** | PyTorch 模型保存/加载，含元信息（持仓代码、版本、训练日期） |
| **实时信号** | 页面底部信号面板，显示买入/持有/卖出决策 + 溢价率 |

**训练流程**：历史数据 → 特征工程 → DQN 训练（n 轮 episode）→ 测试集回测 → 与 Buy & Hold 基准对比 → 保存模型

**数据划分**：训练集 / 验证集 / 测试集三段式（默认 33/33/33），验证集用于超参搜索，测试集仅做最终评估。

#### 🛠️ 后台任务管理器

| 特性 | 说明 |
|------|------|
| **异步队列** | `threading.Semaphore(3)` 限 3 任务并发，`threading.Event` 可取消 |
| **持久化** | JSON 文件原子写入（`tmp + replace`），页面刷新后任务状态不丢失 |
| **实时进度** | `@st.fragment(run_every=1.0)` 仅刷新图表区域，不卡整个页面 |
| **重训练** | 任务详情页 inline 修改参数重新提交，无需切换标签页 |
| **超参搜索** | Walk-Forward 网格搜索迁移为后台任务，可查看中间结果 |
| **自动存模型** | 训练完成后自动保存 `.pt` 文件到 `saved_models/rl/`，支持模型浏览器加载 |
| **优雅退出** | `atexit` 注册清理函数，退出时等待运行中任务自然结束 |

### 💬 市场情绪分析

系统提供**个股**和**板块**双模式情绪看板，数据均来自东方财富 (EastMoney)，通过 AKShare 接口获取。

| 特性 | 说明 |
|------|------|
| **个股情绪** | 东方财富股吧帖子 + 新闻原文，中文金融词典打分（可选用 DeepSeek LLM 增强） |
| **板块情绪排行** | 概念板块（486 个）+ 行业板块，5 因子加权综合得分 |
| **算法** | 宽度 (30%) · 主力资金 (25%) · 异动 (20%) · 涨跌幅 (15%) · 热度 (10%)，归一化至 [-1, 1] |
| **散点图** | 情绪 vs 涨跌幅，全量低透明度背景 + 正负各 Top 5 极端值高亮标注 |
| **板块热帖穿透** | 展开排行行 → 并发爬取该板块成分股的股吧帖子 |
| **历史快照** | Parquet zstd 自动保存每日板块快照，支持按日期回溯 |
| **一键导出 PNG** | matplotlib 组合图（散点+直方+排行+免责声明），自动下载缓存 CJK 字体 |
| **情绪特征供 RL** | `sentiment` feature group（5 列）可选集成到 DQN 训练特征向量 |

### 🛡️ 数据层
- **5 种资产类型**：个股、ETF、LOF、开放式基金、指数
- **自动跨类型回退**：选错类型自动尝试其他类型
- **QDII 自动降级**：ETF/LOF 的 K 线失败时自动切换净值 API
- **30 天本地缓存**：Parquet 格式，第二次查询同代码只需 0.2 秒
- **请求重试**：指数退避 + 浏览器级 Header + `requests.Session` 复用，最大限度提升成功率
- **美股支持**：EastMoney REST API（NASDAQ 105/NYSE 106/AMEX 107），列名与 A 股一致
- **情绪数据源**：东方财富股吧帖子 + 新闻原文 + 板块行情（概念/行业）+ 板块异动数据
- **历史持久化**：板块情绪快照 Parquet zstd 格式，增量写入 + 按列名去重

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

---

## 🧩 项目结构 | Project Structure

```
stock_history_analysis/
├── app.py                    # Streamlit 主应用（轻量路由器 + 侧边栏 + 数据获取）
│
├── core/                     # 共享核心模块（Phase 1）
│   ├── config.py             #   中心化配置（费用、频率映射）
│   └── xirr.py               #   唯一 XIRR 实现（牛顿法）
│
├── data/                     # 数据层模块（Phase 2）
│   ├── asset_config.py       #   资产类型配置（个股/ETF/LOF/基金/指数）
│   ├── cache.py              #   本地缓存（Parquet, 30天过期, LRU）
│   ├── fetcher.py            #   HTTP/API 数据获取（重试/降级/回退）+ 情绪数据入口
│   ├── symbol_registry.py    #   代码注册表管理（CRUD + 缓存优先获取）
│   └── sentiment/            #   情绪分析模块
│       ├── base.py           #     抽象基类
│       ├── cache.py          #     情绪结果缓存
│       ├── deepseek_client.py  #   LLM 增强分析（DeepSeek API）
│       ├── guba.py           #     东方财富股吧爬虫
│       ├── history.py        #     情绪数据持久化（Parquet）
│       ├── lexicon.py        #     中文金融情感词典
│       ├── news.py           #     新闻数据源
│       ├── poller.py         #     定时轮询脚本
│       └── sector_sentiment.py  #  板块情绪引擎（5 因子加权 + 板块行情/异动）
│
├── data_fetcher.py           # 向后兼容 shim（→ data.fetcher）
│
├── ui/                       # 前端展示模块（Phase 3）
│   ├── _helpers.py           #   共享工具（cached_fetch, COLORS）
│   ├── dca_backtest.py       #   定投回测页面
│   ├── grid_search.py        #   网格搜索页面
│   ├── hierarchical_rl.py    #   分层强化学习页面（PPO+DQN 择时选股）
│   ├── rl_signal.py          #   实时信号面板
│   ├── rl_training.py        #   强化学习训练页面（含超参搜索）
│   ├── sentiment.py          #   情绪数据看板（个股+板块双模式，一键导出）
│   ├── task_manager.py       #   后台训练任务管理（列表+详情+持久化）
│   └── symbol_manager.py     #   代码管理中心 UI（CRUD + 批量删除 + 同步）
│
├── backtest/
│   ├── dca.py                # 定投回测引擎（6 频率 + XIRR + 一次性投入）
│   ├── strategies.py         # 智能策略实现（5 种 + 下跌加仓）
│   ├── grid_search.py        # 网格搜索引擎（Walk-Forward CV + 持久化）
│   ├── strategy_results/     # 网格搜索结果存储目录
│   └── rl/
│       ├── dqn_agent.py      #   DQN 智能体（Q 网络、经验回放、保存/加载）
│       ├── environment.py    #   强化学习交易环境（状态/动作/奖励/交易成本）
│       ├── feature_engineer.py  # 特征工程（18+ 技术指标 + SVM/XGBoost）
│       ├── hierarchical_trainer.py  # 分层强化学习训练器（PPO 择时 + DQN 选股）
│       ├── metrics.py        #   评估指标（夏普比率、最大回撤）
│       ├── multi_asset_env.py    # 多资产交易环境（分层 RL 用）
│       ├── ppo_agent.py      #   PPO 智能体（Actor-Critic，择时决策）
│       └── trainer.py        #   训练引擎 + 超参搜索 + 评估 + 信号预测
├── data/symbol_registry.json # 代码注册表持久化文件（自动管理）
├── saved_models/rl/          # RL 模型存储目录（.pt 格式）
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
| 图表 | [Plotly](https://plotly.com/python/) (go.Figure + Heatmap) + [Matplotlib](https://matplotlib.org/) |
| 数据处理 | Pandas / NumPy / Apache Parquet |
| 强化学习 | [PyTorch](https://pytorch.org/) (DQN, PPO, 经验回放, 目标网络) |
| 信号增强 | scikit-learn (LinearSVC) + XGBoost |
| 技术指标 | 手写实现 (MA/EMA/BB/ATR/ADX/CCI/KDJ/RSI) |
| 文本分析 | jieba 分词 + 中文金融情感词典 + DeepSeek API |
| 部署 | PM2 / systemd / WSL2 |
| 后台任务 | threading + JSON 持久化 + `@st.fragment` 实时进度 |

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
