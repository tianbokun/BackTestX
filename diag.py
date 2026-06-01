import numpy as np
import sys
sys.path.insert(0, "/home/tianbo/home/各类文件/投资/stock_history_analysis")

from data_fetcher import fetch_history
from backtest.rl.trainer import train_dqn, evaluate
from backtest.rl.environment import StockTradingEnv
from backtest.rl.feature_engineer import compute_technical_indicators, normalize_indicators, get_state_vector

print("=" * 60)
print("STEP 1: Fetching data for 510300 ETF")
df = fetch_history("etf", "510300", "20230101", "20260526", adjust="qfq")
print(f"Data shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Columns: {list(df.columns)}")
print()

print("=" * 60)
print("STEP 2: Training small DQN model")
agent, states, _ = train_dqn(df, system_version="1.0", n_episodes=8, batch_size=64, lr=1e-4, hidden=64,
    target_update=100, buffer_capacity=2000, epsilon_decay=200,
    commission_rate=0.000235, min_commission=5.0, stamp_duty=0.001,
    initial_capital=100000.0)
print("Training done.")
print()

close = df["收盘"].values.astype(float)
print(f"Close price: len={len(close)}, min={close.min():.4f}, max={close.max():.4f}")
print(f"close[:10]: {close[:10]}")
print(f"close[-10:]: {close[-10:]}")
print()

print("=" * 60)
print("STEP 3: Running evaluate()")
result = evaluate(agent, df, system_version="1.0", initial_capital=100000.0,
    commission_rate=0.000235, min_commission=5.0, stamp_duty=0.001)

actions = result["actions"]
print(f"Action distribution: 0={np.sum(actions==0)}, 1={np.sum(actions==1)}, 2={np.sum(actions==2)}")
print(f"Action bincount: {np.bincount(actions.astype(int), minlength=3)}")

pv = result["equity_curve"]
print(f"Equity curve: len={len(pv)}, min={pv.min():.2f}, max={pv.max():.2f}, mean={pv.mean():.2f}")
print(f"First 10 pv values: {pv[:10]}")
print(f"Last 10 pv values: {pv[-10:]}")
print(f"All values same? {len(np.unique(pv)) == 1}")
print(f"num_trades: {result['num_trades']}")
print(f"total_return_pct: {result['total_return_pct']}")
print()

print("=" * 60)
print("STEP 4: Manual env replay with SAME actions")
indicators = compute_technical_indicators(df)
indicators = normalize_indicators(indicators)
state_vectors = np.array([get_state_vector(indicators, t, "1.0", None, None) for t in range(len(close))])

env2 = StockTradingEnv(state_vectors, close, df.index.tolist(),
    initial_capital=100000.0, commission_rate=0.000235,
    min_commission=5.0, stamp_duty=0.001)
env2.reset()

for i in range(len(actions)):
    _, r, _ = env2.step(int(actions[i]))
    if i < 10 or i >= len(actions) - 10 or i % 100 == 0:
        print(f"Step {i:5d}: action={int(actions[i])}, cash={env2.cash:10.2f}, shares={env2.shares:10.6f}, close={close[i]:8.4f}, pv={env2.portfolio_values[-1]:12.2f}")

print()
print("=" * 60)
print("DIAGNOSIS:")
if len(np.unique(pv)) == 1:
    if np.all(actions == 0):
        print("PROBLEM: Agent always predicts HOLD (action=0). Never trades. Portfolio stays at initial_capital.")
        print("  -> DQN hasn't learned to trade after only 8 episodes with 3 actions.")
        print("  -> Q-values likely all similar; argmax returns action=0 (first output neuron).")
        print("  -> The epsilon-greedy with epsilon_end might still be high enough to cause action=0.")
        print()
        print("Checking Q-values for a sample state:")
        import torch
        s = torch.FloatTensor(state_vectors[0]).unsqueeze(0).to(agent.device)
        for j in range(min(5, len(state_vectors))):
            s = torch.FloatTensor(state_vectors[j]).unsqueeze(0).to(agent.device)
            q = agent.q_net(s)
            print(f"  state[{j}]: Q-values = {q.cpu().detach().numpy().flatten().round(4)}, argmax={q.argmax().item()}")
    else:
        print(f"PROBLEM: Agent trades ({np.sum(actions != 0)} non-hold actions) but portfolio stays flat.")
        print("  -> Check if trades are being executed correctly in env. Buy/sell logic may have issues.")
        print(f"  -> num_trades from evaluate(): {result['num_trades']}")
        print(f"  -> Manual env2 after loop: cash={env2.cash:.2f}, shares={env2.shares:.6f}")
else:
    print("OK: Portfolio value changes. Agent is successfully trading.")
    print(f"Total return: {result['total_return_pct']:.2f}%")
