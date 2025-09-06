import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Create parser
parser = argparse.ArgumentParser(description="plot 2D file")

# Add arguments
parser.add_argument("--crypto", required=True, help="PEPEUSDC-2025-05")

# Parse arguments
args = parser.parse_args()

# Access them
print("Crypto:", args.crypto)
# 1) Charger le CSV
df = pd.read_csv(args.crypto + ".csv", sep="\t")


# Extract data
x = df["sell_gain_pct"]
y = df["profit"]

# Plot 2D line chart
plt.plot(x, y, marker='o', color='blue')  # with markers
# plt.plot(x, y)  # if you just want the line

plt.xlabel("sell_gain_pct")
plt.ylabel("profit")
plt.title("Profit vs Sell Pullback (%)")
plt.grid(True)

plt.show()


# def plot_2d(filename):
#     # 1) Charger le CSV
#     df = pd.read_csv(filename, sep="\t")


#     # Extract data
#     x = df["sell_pullback_pct"]
#     y = df["profit"]

#     # Plot 2D line chart
#     plt.plot(x, y, marker='o', color='blue')  # with markers
#     # plt.plot(x, y)  # if you just want the line

#     plt.xlabel("sell_pullback_pct")
#     plt.ylabel("profit")
#     plt.title("Profit vs Sell Pullback (%)")
#     plt.grid(True)

#     plt.show()
