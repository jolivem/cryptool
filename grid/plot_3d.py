import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # nécessaire pour activer le 3D

# 1) Charger le CSV
df = pd.read_csv("SOLUSDC-2025-06.csv", sep="\t")


# 3) Extraire les colonnes
x = df["buy_pullback_pct"]
y = df["sell_pullback_pct"]
z = df["profit"]

# 4) Tracé 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(x, y, z)  # nuage 3D

# Plot 3D line
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot(x, y, z, color='blue', marker='o')  # lines with markers

ax.set_xlabel("buy_pullback_pct")
ax.set_ylabel("sell_pullback_pct")
ax.set_zlabel("profit")
ax.set_title("Profit en fonction de buy/sell pullback (%)")

plt.show()
