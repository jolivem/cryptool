import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from ta.momentum import RSIIndicator

# === 1. Charger les données PEPE/USDC (fichier requis au format Binance historique) ===
# Exemple de fichier attendu : 'PEPEUSDC.csv' avec colonnes: timestamp, open, high, low, close, volume
root_dir = "C:/Users/joliv/Documents/binance-data/"
crypto = "PEPEUSDC-1s-2025-06.csv"

columns = [
    "timestamp",	
    "open",	
    "high",	
    "low",	
    "close",	
    "Volume",	
    "Close time",
    "Quote asset volume",	
    "Number of trades",	
    "Taker buy base asset volume",	
    "Taker buy quote asset volume",	
    "Ignore"
]

path = root_dir + crypto
data = pd.read_csv(path, header=None, names=columns)
#data = pd.read_csv("PEPEUSDC-1h-2025-06.csv")
data['rsi'] = RSIIndicator(data['close'], window=14).rsi().fillna(0)
data = data.reset_index(drop=True)

# === 2. Définir l'environnement de trading ===
class ScalpingEnv(gym.Env):
    def __init__(self, df):
        super(ScalpingEnv, self).__init__()
        self.df = df.copy()
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 14
        self.balance = 1000
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.cumulative_fees = 0
        return self._next_observation(), {}

    def _next_observation(self):
        row = self.df.iloc[self.current_step]
        obs = np.array([
            row['close'] / self.df['close'].max(),
            row['rsi'] / 100,
            int(self.position > 0)
        ])
        return obs

    def step(self, action):
        row = self.df.iloc[self.current_step]
        price = row['close']
        fee_rate = 0.001

        reward = 0

        if action == 1 and self.position == 0:  # BUY
            amount_usdc = self.balance
            amount_crypto = (amount_usdc / price) * (1 - fee_rate)
            fee = amount_usdc * fee_rate
            self.cumulative_fees += fee
            self.trades.append({'step': self.current_step, 'type': 'buy', 'price': price, 'fee': fee})
            self.position = amount_crypto
            self.entry_price = price
            self.balance = 0

        elif action == 2 and self.position > 0:  # SELL
            gross_usdc = self.position * price
            fee = gross_usdc * fee_rate
            net_usdc = gross_usdc - fee
            pnl = net_usdc - (self.entry_price * self.position)
            self.cumulative_fees += fee
            self.trades.append({'step': self.current_step, 'type': 'sell', 'price': price, 'fee': fee, 'pnl': pnl})
            self.balance = net_usdc
            self.position = 0
            reward = pnl

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        obs = self._next_observation()
        return obs, reward, done, False, {}

# === 3. Entraîner l'agent PPO ===
env = ScalpingEnv(data)
model = PPO("MlpPolicy", env, verbose=1,
    tensorboard_log="./ppo_scalping_tensorboard/")
model.learn(total_timesteps=10000)

# === 4. Simulation de l'agent ===
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

# === 5. Créer un log des trades ===
trades_df = pd.DataFrame(env.trades)
trades_df['time'] = trades_df['step'].apply(lambda x: data.iloc[x]['timestamp'])
trades_df['cumulative_fees'] = trades_df['fee'].cumsum()

# === 6. Graphe avec solde et points d'achat/vente ===
balance_over_time = []
balance = 1000
position = 0
entry_price = 0

for i in range(14, len(data)):
    price = data.iloc[i]['close']
    trade = next((t for t in env.trades if t['step'] == i), None)

    if trade and trade['type'] == 'buy':
        entry_price = trade['price']
        position = balance / price * (1 - 0.001)
        balance = 0

    elif trade and trade['type'] == 'sell':
        balance = position * price * (1 - 0.001)
        position = 0

    value = balance + (position * price)
    balance_over_time.append(value)

# Tracer le graphe
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(data['close'].iloc[14:].values, label='Prix (PEPE)', color='gray')
ax1.set_ylabel("Prix", color="gray")

ax2 = ax1.twinx()
ax2.plot(balance_over_time, label='Solde (USDC)', color='blue')
ax2.set_ylabel("Solde", color="blue")

buy_points = trades_df[trades_df['type'] == 'buy']
sell_points = trades_df[trades_df['type'] == 'sell']

ax1.scatter(buy_points['step'] - 14, buy_points['price'], marker='^', color='green', label='Achat')
ax1.scatter(sell_points['step'] - 14, sell_points['price'], marker='v', color='red', label='Vente')

fig.legend(loc='upper left')
plt.title("Scalping RL - Prix, Solde et Transactions")
plt.show()

# === 7. Affichage résumé ===
final_balance = balance_over_time[-1]
print(f"💰 Solde final : {final_balance:.6f} USDC")
print(f"📉 Frais totaux : {env.cumulative_fees:.6f} USDC")
print(f"📈 Nombre de trades : {len(trades_df)}")
print(trades_df.head())

