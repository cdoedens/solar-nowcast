import pandas as pd
from pathlib import Path

# READ PARQUET TABULAR DATA
data_path = Path('/scratch/er8/cd3022/xgb_datasets/')

for month in range(1, 13):
    df = pd.concat(
        pd.read_parquet(f)
        for f in data_path.glob(f'all_training_2025-{month:02d}*')
    )
    df.to_parquet(data_path / f"all_training_month_{month:02d}.parquet")