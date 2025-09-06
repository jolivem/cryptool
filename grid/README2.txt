get the historical data, example for ETHUSDC 1sec 06/2025: 
https://data.binance.vision/?prefix=data/spot/monthly/klines/ETHUSDC/1s/
unzip and paste in C:\Users\joliv\Documents\binance-data

puis dans cryptool:
python3 -m venv venv
venv\Scripts\activate
pip install pandas


puis modifie optimize.py: le nom du fichier ETHUSDC-1s-2025-06.csv