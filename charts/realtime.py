import pandas as pd

# ==== CONFIGURATION ====
root_dir = "C:\\Users\\joliv\\Documents\\binance-data\\"
crypto = "FLOKIUSDC-1s-2025-07"

# ==== 1. Load and Prepare Data ====
# Load the CSV
columns = [
    "timespan",	
    "Open",	
    "High",	
    "Low",	
    "close",	
    "Volume",	
    "Close time",
    "Quote asset volume",	
    "Number of trades",	
    "Taker buy base asset volume",	
    "Taker buy quote asset volume",	
    "Ignore"
]

file_path = root_dir + crypto + ".csv"
df = pd.read_csv(file_path, header=None, names=columns)
df = df[['timespan', 'close', 'Volume']]
df = df[df["Volume"] != 0]
df['timespan'] = pd.to_datetime(df['timespan'], unit='us')  # timestamp en =us
output_path = root_dir + crypto + "_formatted.csv"
df.to_csv(output_path, index=False, date_format="%Y-%m-%d %H:%M:%S")