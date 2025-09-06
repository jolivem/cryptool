get binance-public-data from https://github.com/binance/binance-public-data/

cd binance-public-data-master/python
python3 -m venv venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir ~/binance-data
export STORE_DIRECTORY=~/binance-data

Exemple: 
python3 download-kline.py -t spot -s ETHUSDT BTCUSDT BNBBUSD -i 1w -y 2020 -m 02 12 -c 1

Exemple pour SOLUSDC 1sec mois de juin
python3 download-kline.py -t spot -s SOLUSDC -i 1s -y 2025 -m 06
-> résultat dans /home/michel/binance-data/data/spot/monthly/klines/SOLUSDC/1s/SOLUSDC-1s-2025-06.zip

Description du contenu du fichier dans https://github.com/binance/binance-public-data/tree/master
contenu:
    .Open time,	
    .Open,	
    .High,	
    .Low,	
    .Close,	
    .Volume,	
    .Close time,
    .Quote asset volume,	
    .Number of trades,	
    .Taker buy base asset volume,	
    .Taker buy quote asset volume,	
    .Ignore