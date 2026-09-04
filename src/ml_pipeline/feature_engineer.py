import os
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.ml_pipeline.config import ML_FEATURES, TARGET_COL, TRAIN_END, TEST_START


def engineer_features(df):
    df = df.copy()

    # fix dtypes - some columns come out as object from csv
    df['geo_distance_km'] = df['geo_distance_km'].astype(float)
    df['is_disposable_email'] = df['is_disposable_email'].astype(int)
    df['is_cod'] = df['is_cod'].astype(int)
    df['transaction_velocity_24h'] = df['transaction_velocity_24h'].astype(int)
    df['transaction_velocity_7d'] = df['transaction_velocity_7d'].astype(int)
    df['avg_order_value_ratio'] = df['avg_order_value_ratio'].astype(float)
    df['account_age_days'] = df['account_age_days'].astype(int)
    df['is_high_risk_category'] = df['is_high_risk_category'].astype(int)
    df['hour_of_day'] = df['hour_of_day'].astype(int)
    df['is_first_order'] = df['is_first_order'].astype(int)
    df['return_history_rate'] = df['return_history_rate'].astype(float)
    df['is_return'] = df['is_return'].astype(int)

    # clip outliers - some values were going crazy
    df['avg_order_value_ratio'] = df['avg_order_value_ratio'].clip(0, 20)
    df['geo_distance_km'] = df['geo_distance_km'].clip(0, 4000)
    df['transaction_velocity_24h'] = df['transaction_velocity_24h'].clip(0, 50)
    df['transaction_velocity_7d'] = df['transaction_velocity_7d'].clip(0, 100)
    df['return_history_rate'] = df['return_history_rate'].clip(0, 1)

    # fill nulls if any
    for col in ML_FEATURES:
        if df[col].isna().any():
            if df[col].dtype in ['float64', 'float32']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)

    return df


def chronological_split(df):
    # using time-based split instead of random split to avoid leakage
    # train = jan-oct 2025, test = nov-dec 2025
    df = df.copy()
    df['order_timestamp'] = pd.to_datetime(df['order_timestamp'])

    train_df = df[df['order_timestamp'] <= pd.Timestamp(TRAIN_END)].copy()
    test_df = df[df['order_timestamp'] >= pd.Timestamp(TEST_START)].copy()

    return train_df, test_df


def prepare_ml_data(df):
    # just splits features and label
    X = df[ML_FEATURES].copy()
    y = df[TARGET_COL].copy()
    return X, y


def main():
    raw_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'transactions_synthetic.csv')

    if not os.path.exists(raw_path):
        print("raw data not found, run generate_data.py first")
        sys.exit(1)

    print(f"Loading data from {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df)} rows")

    df = engineer_features(df)

    # check for nulls after engineering
    for col in ML_FEATURES:
        nulls = df[col].isna().sum()
        if nulls > 0:
            print(f"WARNING: {col} has {nulls} nulls")

    train_df, test_df = chronological_split(df)
    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    print(f"Train return rate: {train_df['is_return'].mean():.2%}")
    print(f"Test return rate: {test_df['is_return'].mean():.2%}")

    processed_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    train_df.to_csv(os.path.join(processed_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(processed_dir, 'test.csv'), index=False)
    print("Saved train.csv and test.csv")

    # quick correlation check with target
    X_train, y_train = prepare_ml_data(train_df)
    corrs = X_train.corrwith(y_train).sort_values(ascending=False)
    print("\nCorrelation with is_return:")
    print(corrs)


if __name__ == "__main__":
    main()
