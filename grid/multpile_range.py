import bestbot

# Liste prédéfinie
months = ["2025-03", "2025-04", "2025-05", "2025-06", "2025-07"]

# Boucle sur chaque mois
for ym in months:
    filename = "PEPEUSDC-1s-" + ym + ".csv"
    bestbot.find_best_bot_with_loop(-111, filename)

