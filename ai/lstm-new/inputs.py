

# -------------------------
# Paramètres principaux
# -------------------------
#CSV_PATH = "D:\DEVS\cryptool\datas\ETHUSDC-1s-2025-07.csv"       # votre fichier
TARGET_COL = "close"        # variable à prédire
USE_RETURNS = True          # True = prédire le rendement (log-return), False = prix
WINDOW = 60                 # longueur de la fenêtre (pas de temps)
HORIZON = 1                 # prédire t+1 (même granularité)
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
VAL_SPLIT = 0.15            # portion de la fin du jeu d'entraînement pour la validation
TEST_SPLIT = 0.15           # portion de la fin totale pour test (walk-forward)

CRYPTO = "ETHUSDC-1s-2025-07"