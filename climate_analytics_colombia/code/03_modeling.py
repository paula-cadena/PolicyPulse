import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os
import joblib

# Add utils to path
sys.path.append(os.path.dirname(__file__))
from utils import logger


def merge_datasets(financial_data, precipitation_data, climate_features):
    """Merge financial, precipitation, and climate data"""
    logger.info("Merging financial and climate datasets")

    # Ensure datetime
    financial_data['date'] = pd.to_datetime(financial_data['date'])
    precipitation_data['date'] = pd.to_datetime(precipitation_data['date'])

    merged = pd.merge(financial_data, precipitation_data,
                      on=['municipality_code', 'date'], how='inner')
    merged = pd.merge(merged, climate_features,
                      on='municipality_code', how='left')

    logger.info(f"Merged dataset: {len(merged)} records")
    return merged


def create_features(df):
    """Create advanced + simple features"""
    logger.info("Creating features")

    # Time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter

    # Climate anomalies
    df['precip_anomaly'] = df['precipitation'] - df['precip_mean']
    df['precip_std_anomaly'] = df['precip_anomaly'] / df['precip_std']
    df['extreme_precip'] = (df['precip_std_anomaly'].abs() > 2).astype(int)

    # Lags & rolling stats
    df = df.sort_values(['municipality_code', 'date'])
    df['precip_lag1'] = df.groupby('municipality_code')['precipitation'].shift(1)
    df['precip_lag2'] = df.groupby('municipality_code')['precipitation'].shift(2)
    df['precip_rolling_3m'] = df.groupby('municipality_code')['precipitation'].rolling(3).mean().reset_index(0, drop=True)
    df['precip_rolling_6m'] = df.groupby('municipality_code')['precipitation'].rolling(6).mean().reset_index(0, drop=True)

    # Local climate norms
    precip_stats = df.groupby('municipality_code')['precipitation'].agg(['mean', 'std']).reset_index()
    precip_stats.columns = ['municipality_code', 'precip_mean_local', 'precip_std_local']
    df = pd.merge(df, precip_stats, on='municipality_code', how='left')
    df['extreme_dry'] = (df['precipitation'] < df['precip_mean_local'] - df['precip_std_local']).astype(int)
    df['extreme_wet'] = (df['precipitation'] > df['precip_mean_local'] + df['precip_std_local']).astype(int)

    df = df.dropna()
    logger.info(f"Final dataset with features: {len(df)} records")
    return df


def run_regression(model_data, outcome_var):
    """Run OLS regression"""
    logger.info(f"Running regression for {outcome_var}")

    features = ['precipitation', 'precip_anomaly', 'precip_std_anomaly',
                'extreme_precip', 'precip_lag1', 'precip_lag2',
                'precip_rolling_3m', 'precip_rolling_6m',
                'month', 'quarter']

    available = [f for f in features if f in model_data.columns]
    X = model_data[available].copy()
    y = model_data[outcome_var]

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    logger.info(f"R²={model.rsquared:.3f}, N={len(X)}")
    return model


def run_random_forest(model_data, outcome_var):
    """Run Random Forest regression"""
    logger.info(f"Running Random Forest for {outcome_var}")

    features = ['precipitation', 'precip_mean', 'precip_std',
                'precip_anomaly', 'precip_std_anomaly',
                'precip_lag1', 'precip_lag2',
                'precip_rolling_3m', 'precip_rolling_6m',
                'month', 'quarter', 'year',
                'precip_mean_local', 'precip_std_local',
                'extreme_dry', 'extreme_wet']

    available = [f for f in features if f in model_data.columns]
    X = model_data[available]
    y = model_data[outcome_var]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    mse, r2 = mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)

    feat_imp = pd.DataFrame({'feature': available,
                             'importance': rf.feature_importances_}).sort_values('importance', ascending=False)

    logger.info(f"RF Results: MSE={mse:.2f}, R²={r2:.3f}")
    return rf, feat_imp, {'mse': mse, 'r2': r2}


def main():
    logger.info("Starting unified modeling pipeline")

    # Load inputs
    fin = pd.read_csv('data_processed/financial_processed.csv')
    prec = pd.read_csv('data_processed/precipitation_processed.csv')
    clim = pd.read_csv('data_processed/climate_features.csv')

    # Merge & features
    merged = merge_datasets(fin, prec, clim)
    model_data = create_features(merged)

    # Save cleaned dataset
    model_data.to_csv('data_processed/modeling_dataset_cleaned.csv', index=False)

    outcomes = ['credit_portfolio', 'savings_deposits']
    results = {}

    for outcome in outcomes:
        outcome_data = model_data[model_data[outcome].notna()].copy()
        if len(outcome_data) == 0:
            logger.warning(f"No data for {outcome}")
            continue

        reg = run_regression(outcome_data, outcome)
        rf, fi, metrics = run_random_forest(outcome_data, outcome)

        results[outcome] = {'regression': reg, 'rf_model': rf,
                            'feature_importance': fi, 'metrics': metrics}

        # Save outputs
        os.makedirs('outputs/tables', exist_ok=True)
        os.makedirs('outputs/model_artifacts', exist_ok=True)

        with open(f'outputs/tables/regression_summary_{outcome}.txt', 'w') as f:
            f.write(reg.summary().as_text())
        fi.to_csv(f'outputs/tables/feature_importance_{outcome}.csv', index=False)
        joblib.dump(rf, f'outputs/model_artifacts/rf_model_{outcome}.pkl')

    logger.info("Unified pipeline finished successfully")
    return results


if __name__ == "__main__":
    results = main()
