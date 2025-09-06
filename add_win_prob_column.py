# add_winprob_column.py
import pandas as pd
import glob
import os

# List of CSVs you want to process
INPUT_FILES = [
    "ready_singles_all.csv"
]

for file in INPUT_FILES:
    if not os.path.exists(file):
        print(f"⚠️ File not found, skipping: {file}")
        continue

    df = pd.read_csv(file)

    if "Consensus P" not in df.columns:
        print(f"⚠️ No 'Consensus P' column in {file}, skipping.")
        continue

    # Add Win Prob column (percentage)
    df["Win Prob (Consensus)"] = (df["Consensus P"] * 100).round(1)

    # Save back with a suffix so you don’t overwrite unless you want to
    out_file = file#.replace(".csv", "_with_winprob.csv")
    df.to_csv(out_file, index=False)

    print(f"✅ Processed {file} → {out_file}")