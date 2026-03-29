from src.models.train import load_and_label, train_val_split, train_xgboost

df = load_and_label('data/btc_1m.parquet')
train_df, val_df = train_val_split(df)
train_xgboost(train_df, val_df, 'models/artifacts/', n_trials=20)
