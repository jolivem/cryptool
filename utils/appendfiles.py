root_dir=r"C:/Users/joliv/Documents/binance-data/"
# List of files in the order you want to append them
files = [
    "SOLUSDC-1s-2025-03.csv",
    "SOLUSDC-1s-2025-04.csv",
    "SOLUSDC-1s-2025-05.csv",
    "SOLUSDC-1s-2025-06.csv",
    "SOLUSDC-1s-2025-07.csv"
]

# Output file
output_file = "SOLUSDC-1s-2025-03-04-05-06-07.csv"

with open(root_dir + output_file, "w", encoding="utf-8") as outfile:
    for fname in files:
        print(f"Append {fname}...")
        with open(root_dir + fname, "r", encoding="utf-8") as infile:
            outfile.write(infile.read())
            outfile.write("\n")  # optional: add newline between files

print(f"Files merged into {output_file}")
