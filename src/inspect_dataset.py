import pandas as pd
import os

data_path = "data/raw"

for file in os.listdir(data_path):

    if file.endswith(".csv"):

        df = pd.read_csv(os.path.join(data_path, file))

        print("\n" + "=" * 60)
        print(f"Dataset: {file}")
        print("=" * 60)

        print(f"Rows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")

        print("\nColumn Names:")
        for col in df.columns:
            print("-", col)

        print("\nData Types:")
        print(df.dtypes)