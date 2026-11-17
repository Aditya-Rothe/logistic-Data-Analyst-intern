
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

csv_files = sorted(DATA_DIR.glob("*.csv"))

for file_path in csv_files:

    df = pd.read_csv(file_path)

    print("\n" + "=" * 65)
    print(f"DATASET: {file_path.name}")
    print("=" * 65)

    print("Dataset Shape:", df.shape)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:", df.duplicated().sum())

    print("\nData Types:")
    print(df.dtypes)

    print("\nSample Records:")
    print(df.head(3))
