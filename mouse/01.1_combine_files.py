import pandas as pd
from pathlib import Path

# READ PARQUET TABULAR DATA
data_path = Path('/scratch/er8/cd3022/xgb_datasets/')
df = pd.concat(
    pd.read_parquet(f)
    for f in data_path.glob('*all_training*')
)

df[df['month'].isin([2, 6, 10])].to_parquet(data_path / "test_months.parquet")
df[~df['month'].isin([2, 6, 10])].to_parquet(data_path / "train_months.parquet")