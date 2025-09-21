import bestbot


crytpos = [
    "SOLUSDC-1s-2025-07"
]


# buy_drop_pct, buy_pullback_pct, sell_gain_pct, sell_pullback_pct
params_list = [
    [0.015, 0.005, 0.025, 0.01],
    [0.015, 0.005, 0.026, 0.009],
    [0.015, 0.005, 0.027, 0.008],
    [0.015, 0.005, 0.028, 0.007],
    [0.015, 0.005, 0.029, 0.006],
    [0.015, 0.005, 0.030, 0.005],
    [0.015, 0.005, 0.031, 0.004],
    [0.015, 0.005, 0.032, 0.003],
    [0.015, 0.005, 0.033, 0.002],
    [0.015, 0.005, 0.034, 0.001],
    [0.015, 0.005, 0.035, 0.0],
]

# Boucle sur les fichiers
for crytpo in crytpos:
    bestbot.find_best_bot_with_params(-111, crytpo, params_list, delete=True)
