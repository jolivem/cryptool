import requests
import pandas as pd
import os
import zipfile
import shutil

datas_folder = "/mnt/d/DEVS/datas/"
result_folder = "/mnt/d/DEVS/results/"


def download_binance_file(filename):

    # filename = "SOLUSDC-1s-2025-06.csv"

    # remove extension
    name = filename.removesuffix(".csv")   # Python 3.9+
    # or: name = filename[:-4]

    parts = name.split("-")  # ['SOLUSDC', '1s', '2025', '06']

    crypto = parts[0]
    interval = parts[1]
    year_month = f"{parts[2]}-{parts[3]}"

    print("crypto =", crypto)
    print("interval =", interval)
    print("year_month =", year_month)

    download_binance_crypto(crypto, interval, year_month)

def download_and_join_binance_file(allmonths):

    # allmonths = "SOLUSDC-1s-2025-06_05-04"

    # remove extension
    #name = filename.removesuffix(".csv")   # Python 3.9+
    # or: name = filename[:-4]
    output_file = datas_folder + allmonths + ".csv"

    if os.path.exists(output_file):
        print("Le fichier " +  allmonths + ".csv existe.")
        return

    # first download
    months = split_into_month_files(allmonths)
    for name in months:
        parts = name.split("-")  # ['SOLUSDC', '1s', '2025', '06']

        crypto = parts[0]
        interval = parts[1]
        year_month = f"{parts[2]}-{parts[3]}"

        print("crypto =", crypto)
        print("interval =", interval)
        print("year_month =", year_month)

        download_binance_crypto(crypto, interval, year_month)

    # then join 
    if len(months) > 1:
        with open(output_file, "w", encoding="utf-8") as outfile:
            for fname in months:
                print(f"Append {fname}...")
                with open(datas_folder + fname + ".csv", "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
                    outfile.write("\n")  # optional: add newline between files

    print(f"Files merged into {output_file}")    


def split_into_month_files(s: str):
    parts = s.split("-")
    
    # Find where the year starts (assuming it's always 4-digit)
    year_idx = next(i for i, p in enumerate(parts) if len(p) == 4 and p.isdigit())
    
    prefix = "-".join(parts[:year_idx+1])  # e.g. SOLUSDC-1s
    lst = parts[year_idx+1:]        # e.g. ['06','05',...]
    date_parts = sorted(lst, key=int)

    print(f"date_parts={date_parts}") 
    results = []
    for i in range(0, len(date_parts)):
        print(f"i={i} date_parts[i]={date_parts[i]}")    
        results.append(prefix + "-" + date_parts[i])
    
    return results

def download_binance_crypto(symbol, interval, year_month):
    try:
        # Construct URL and local path
        filename = f"{symbol}-{interval}-{year_month}.zip"
        url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{filename}"
        local_path = os.path.join(datas_folder, filename)

        if os.path.exists(local_path):
            print("Le fichier " +  filename + " existe.")
            return

        # Make sure destination directory exists
        os.makedirs(datas_folder, exist_ok=True)

        print(f"Downloading {url} ...")
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Downloaded and saved to: {local_path}")
            
            unzip_csv(local_path, datas_folder)

        else:
            print(f"❌ Failed to download. Status code: {response.status_code}")
            print(f"URL: {url}")
    except ValueError as e:
        # What to do if a ValueError happens
        print("download_binance_crypto: Oops! Could not find best bot:", e)


def unzip_csv(zip_path, extract_to_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # List files in the ZIP
        for file in zip_ref.namelist():
            if file.endswith(".csv"):
                print(f"🔓 Extracting {file} to {extract_to_dir}")
                zip_ref.extract(file, path=extract_to_dir)
                return os.path.join(extract_to_dir, file)
    print("❌ No CSV file found in the ZIP.")
    return None

    
#crypto = 
# filename without path and with csv at the end
# return data when volume != 0
#        with columes timestamp=datetime, close price and volume
def load_crypto_month(crypto_filename):
    # Chargement du CSV sans en-tête
    columns = [
        "timespan",	
        "open",	
        "high",	
        "low",	
        "close",	
        "volume",	
        "close time",
        "quote asset volume",	
        "number of trades",	
        "taker buy base asset volume",	
        "taker buy quote asset volume",	
        "ignore"
    ]

    #print(first_part)  # Output: ARBUSDC
    #crypto = "ARBUSDC-1s-2025-06.csv"
    kline_file = os.path.join(datas_folder,crypto_filename)
    data = pd.read_csv(kline_file, header=None, names=columns)
    #data = data[data["volume"] != 0]
    data['timestamp'] = pd.to_datetime(data['timespan'], unit='us')
    #data['price'] = data['close']

    return data

def load_crypto_all_column(crypto_filename):
    # Chargement du CSV sans en-tête
    columns = [
        "timespan",	
        "open",	
        "high",	
        "low",	
        "close",	
        "volume",	
        "close time",
        "quote asset volume",	
        "number of trades",	
        "taker buy base asset volume",	
        "taker buy quote asset volume",	
        "ignore"
    ]

    kline_file = os.path.join(datas_folder,crypto_filename)
    data = pd.read_csv(kline_file, header=None, names=columns)

    return data

