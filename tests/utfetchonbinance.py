import sys
sys.path.append("../utils")
import fetchonbinance

#fetchonbinance.generate_month_files("INJUSDC-1s-2025-07-06-05-04-03")

#print(fetchonbinance.split_into_month_files("SOLUSDC-1s-2025-06-05"))
# ['SOLUSDC-1s-2025', 'SOLUSDC-1s-2025-06', 'SOLUSDC-1s-2025-06-05']

print(fetchonbinance.download_and_join_binance_file("INJUSDC-1s-2025-07-06-05-04-03"))

#fetchonbinance.download_and_join_binance_file("INJUSDC-1s-2025-07-06-05-04-03")