import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import sys
import os
import joblib

sys.path.append(os.path.dirname(__file__))
from utils import logger

def load_and_clean_data():
    """Load and clean data for modeling"""
    logger.info("Loading and cleaning data for modeling")
    
    try:
        # Trying to load the cleaned dataset first
        model_data = pd.read_csv('data_processed/modeling_dataset_cleaned.csv')
        logger.info(f"Loaded cleaned dataset: {len(model_data)} records")
    except FileNotFoundError:
        # Fall back to original dataset, if fails
        model_data = pd.read_csv('data_processed/modeling_dataset.csv')
        logger.info(f"Loaded original dataset: {len(model_data)} records")
        
        # Basic cleaning
        numeric_cols = ['credit_portfolio', 'savings_deposits', 'precipitation']
        for col in numeric_cols:
            if col in model_data.columns:
                model_data[col] = pd.to_numeric(model_data[col], errors='coerce')
        
        # Removing rows with missing key variables
        key_cols = ['credit_portfolio', 'savings_deposits', 'precipitation', 'municipality_code']
        model_data = model_data.dropna(subset=key_cols)
        logger.info(f"After basic cleaning: {len(model_data)} records")
    
    return model_data

def create_simple_features(model_data):
    """Create simple features for modeling"""
    logger.info("Creating simple features")
    
    df = model_data.copy()
    
    # Time features
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
    
    # Climate features
    if 'precipitation' in df.columns:
        # Grouping by municipality to calculate norms
        precip_stats = df.groupby('municipality_code')['precipitation'].agg(['mean', 'std']).reset_index()
        precip_stats.columns = ['municipality_code', 'precip_mean_local', 'precip_std_local']
        
        df = pd.merge(df, precip_stats, on='municipality_code', how='left')
        
        # Calculating simple anomaly
        df['precip_anomaly'] = df['precipitation'] - df['precip_mean_local']
        
        # Calculating extreme precipitation (simple definition)
        df['extreme_dry'] = (df['precipitation'] < df['precip_mean_local'] - df['precip_std_local']).astype(int)
        df['extreme_wet'] = (df['precipitation'] > df['precip_mean_local'] + df['precip_std_local']).astype(int)
    
    return df

def run_robust_regression(model_data, outcome_var='credit_portfolio'):
    """Run robust regression that handles data issues gracefully"""
    logger.info(f"Running robust regression for {outcome_var}")
    
    # Selecting feature set
    simple_features = ['precipitation', 'precip_anomaly', 'month', 'quarter']
    available_features = [f for f in simple_features if f in model_data.columns]
    
    if not available_features:
        logger.error("No features available for regression")
        return None
    
    X = model_data[available_features].copy()
    y = model_data[outcome_var]
    
    # Ensuring numeric data types
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    
    # Removing missing values
    non_missing = X.notna().all(axis=1) & y.notna()
    X_clean = X[non_missing]
    y_clean = y[non_missing]
    
    if len(X_clean) < 100:
        logger.warning(f"Insufficient data for regression: {len(X_clean)} samples")
        return None
    
    # Adding a constant
    X_clean = sm.add_constant(X_clean)
    
    try:
        # Using simple OLS without clustering first
        model = sm.OLS(y_clean, X_clean).fit()
        
        logger.info(f"Regression completed for {outcome_var}")
        logger.info(f"R-squared: {model.rsquared:.3f}")
        logger.info(f"Observations: {len(X_clean)}")
        logger.info(f"Significant variables (p < 0.05):")
        
        for feature in available_features:
            if feature in model.pvalues.index:
                pval = model.pvalues[feature]
                coef = model.params[feature]
                if pval < 0.05:
                    logger.info(f"  {feature}: coef={coef:.4f}, p={pval:.4f}***")
                else:
                    logger.info(f"  {feature}: coef={coef:.4f}, p={pval:.4f}")
        
        return model
        
    except Exception as e:
        logger.error(f"Regression failed: {e}")
        return None

def run_simple_machine_learning(model_data, outcome_var='credit_portfolio'):
    """Run simple machine learning model"""
    logger.info(f"Running simple Random Forest for {outcome_var}")
    
    # Selecting simple features
    feature_columns = ['precipitation', 'month', 'quarter', 'year']
    
    # Adding local climate features when available
    local_features = ['precip_mean_local', 'precip_std_local', 'precip_anomaly', 'extreme_dry', 'extreme_wet']
    for f in local_features:
        if f in model_data.columns:
            feature_columns.append(f)
    
    available_features = [col for col in feature_columns if col in model_data.columns]
    
    if not available_features:
        logger.error("No features available for ML")
        return None, None, None
    
    X = model_data[available_features]
    y = model_data[outcome_var]
    
    # Cleaning data
    X_clean = X.apply(pd.to_numeric, errors='coerce')
    y_clean = pd.to_numeric(y, errors='coerce')
    
    # Removing missing
    non_missing = X_clean.notna().all(axis=1) & y_clean.notna()
    X_final = X_clean[non_missing]
    y_final = y_clean[non_missing]
    
    if len(X_final) < 100:
        logger.warning(f"Insufficient data for ML: {len(X_final)} samples")
        return None, None, None
    
    try:
        # Faster model
        rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        rf_model.fit(X_final, y_final)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top features for {outcome_var}:")
        for _, row in feature_importance.head().iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.3f}")
        
        return rf_model, feature_importance, {'n_samples': len(X_final)}
        
    except Exception as e:
        logger.error(f"ML failed: {e}")
        return None, None, None

def main():
    """Main modeling function with robust error handling"""
    logger.info("Starting robust modeling process")
    
    # Loading and cleaning data
    model_data = load_and_clean_data()
    
    if len(model_data) == 0:
        logger.error("No data available for modeling")
        return {}
    
    # Creating simple features
    model_data = create_simple_features(model_data)
    
    # Running analyses for both outcome variables
    outcomes = ['credit_portfolio', 'savings_deposits']
    results = {}
    
    for outcome in outcomes:
        logger.info(f"\n=== ANALYZING IMPACT ON {outcome.upper()} ===")
        
        # Filtering to relevant data
        outcome_data = model_data[model_data[outcome].notna()].copy()
        
        if len(outcome_data) > 100:
            # Running robust regression
            regression_model = run_robust_regression(outcome_data, outcome)
            
            # Running simple ML
            ml_model, feature_importance, ml_metrics = run_simple_machine_learning(outcome_data, outcome)
            
            # Storing results
            results[outcome] = {
                'regression_model': regression_model,
                'ml_model': ml_model,
                'feature_importance': feature_importance,
                'ml_metrics': ml_metrics,
                'n_observations': len(outcome_data)
            }
            
            # Saving outputs
            if regression_model is not None:
                with open(f'outputs/tables/regression_summary_{outcome}.txt', 'w') as f:
                    f.write(regression_model.summary().as_text())
            
            if feature_importance is not None:
                feature_importance.to_csv(f'outputs/tables/feature_importance_{outcome}.csv', index=False)
                
        else:
            logger.warning(f"Insufficient data for {outcome}: {len(outcome_data)} records")
    
    logger.info("Robust modeling completed")
    return results

if __name__ == "__main__":
    results = main()