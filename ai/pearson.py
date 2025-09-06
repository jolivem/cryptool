import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données fusionnées
df = pd.read_csv("merged_crypto_data.csv", index_col="timespan", parse_dates=True)

# Afficher la corrélation entre les colonnes
corr = df.corr(method='pearson')
print(corr)
