import os
import bestbot

root_dir = r"C:\\Users\\joliv\\Documents\\binance-data"

for filename in os.listdir(root_dir):
    if filename.endswith(".csv") and not filename.endswith("_results.csv"):
        try:
            #run_simulation(filename, root_dir)
            bestbot.find_best_bot_random(1000, filename)
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")
