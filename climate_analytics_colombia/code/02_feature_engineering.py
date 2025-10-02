import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os
import joblib

sys.path.append(os.path.dirname(__file__))
from utils import logger

def merge_datasets(financial_data, precipitation_data, climate_features):
    """
    Merges financial data with precipitation data and climate features
    """
    logger.info("Merging financial and climate datasets")
    
    # Ensuring dates are datetime
    financial_data['date'] = pd.to_datetime(financial_data['date'])
    precipitation_data['date'] = pd.to_datetime(precipitation_data['date'])
    
    # Merging financial data with precipitation data on municipality code and date
    merged_data = pd.merge(financial_data, precipitation_data, 
                          on=['municipality_code', 'date'], 
                          how='inner')
    
    # Merging with climate features
    merged_data = pd.merge(merged_data, climate_features,
                          on='municipality_code',
                          how='left')
    
    logger.info(f"Merged dataset: {len(merged_data)} records")
    return merged_data

def create_additional_features(merged_data):
    """Create additional features for modeling"""
    logger.info("Creating additional features for modeling")
    
    df = merged_data.copy()
    
    # Creating time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    
    # Creating climate anomaly features
    if 'precip_mean' in df.columns and 'precipitation' in df.columns:
        df['precip_anomaly'] = df['precipitation'] - df['precip_mean']
        if 'precip_std' in df.columns:
            df['precip_std_anomaly'] = df['precip_anomaly'] / df['precip_std']
            df['extreme_precip'] = (df['precip_std_anomaly'].abs() > 2).astype(int)
    
    # Creating lagged features
    df = df.sort_values(['municipality_code', 'date'])
    df['precip_lag1'] = df.groupby('municipality_code')['precipitation'].shift(1)
    df['precip_lag2'] = df.groupby('municipality_code')['precipitation'].shift(2)
    
    # Creating rolling statistics
    df['precip_rolling_3m'] = df.groupby('municipality_code')['precipitation'].rolling(3).mean().reset_index(0, drop=True)
    df['precip_rolling_6m'] = df.groupby('municipality_code')['precipitation'].rolling(6).mean().reset_index(0, drop=True)
    
    # Dropping rows with NaN values from lagged features
    initial_count = len(df)
    df = df.dropna()
    final_count = len(df)
    
    logger.info(f"Final modeling dataset: {final_count} records (dropped {initial_count - final_count} due to missing values)")
    
    return df

def clean_data_for_regression(X, y):
    """Clean data for regression by ensuring all values are numeric and handling missing values"""
    logger.info("Cleaning data for regression")
    
    # Converting all columns to numeric, coercing errors to NaN
    X_clean = X.apply(pd.to_numeric, errors='coerce')
    y_clean = pd.to_numeric(y, errors='coerce')
    
    # Removing rows with any NaN values, to avoid errors in regression
    non_missing_mask = X_clean.notna().all(axis=1) & y_clean.notna()
    X_clean = X_clean[non_missing_mask]
    y_clean = y_clean[non_missing_mask]
    
    logger.info(f"After cleaning: {len(X_clean)} records remaining")
    
    return X_clean, y_clean

def run_simple_regression(model_data, outcome_var='credit_portfolio'):
    """Run a simplified OLS regression without fixed effects first"""
    logger.info(f"Running simple OLS regression for {outcome_var}")
    
    # Selecting a simpler set of variables to avoid dimensionality issues
    simple_vars = ['precipitation', 'precip_anomaly', 'month', 'quarter']
    
    # Defining available variables
    available_vars = [var for var in simple_vars if var in model_data.columns]
    
    X = model_data[available_vars].copy()
    y = model_data[outcome_var]
    
    # Cleaning the data
    X_clean, y_clean = clean_data_for_regression(X, y)
    
    if len(X_clean) == 0:
        logger.warning(f"No valid data for regression after cleaning for {outcome_var}")
        return None
    
    # Adding a constant
    X_clean = sm.add_constant(X_clean)
    
    # Running OLS regression
    try:
        model = sm.OLS(y_clean, X_clean).fit()
        
        logger.info(f"Simple regression completed for {outcome_var}")
        logger.info(f"R-squared: {model.rsquared:.3f}")
        logger.info(f"Number of observations: {len(X_clean)}")
        
        return model
    except Exception as e:
        logger.error(f"Error in simple regression: {e}")
        return None

def run_panel_regression(model_data, outcome_var='credit_portfolio'):
    """Run panel data regression to quantify climate impact"""
    logger.info(f"Running panel regression for {outcome_var}")
    
    # Preparing variables for fixed effects regression
    X_vars = ['precipitation', 'precip_anomaly', 'precip_std_anomaly', 
              'extreme_precip', 'precip_lag1', 'precip_lag2',
              'precip_rolling_3m', 'precip_rolling_6m',
              'month', 'quarter']
    
    # Selecting available variables
    available_vars = [var for var in X_vars if var in model_data.columns]
    
    X = model_data[available_vars].copy()
    y = model_data[outcome_var]
    
    # Cleaning the data
    X_clean, y_clean = clean_data_for_regression(X, y)
    
    if len(X_clean) == 0:
        logger.warning(f"No valid data for panel regression after cleaning for {outcome_var}")
        return None
    
    # Adding a constant
    X_clean = sm.add_constant(X_clean)
    
    # Adding municipality fixed effects using dummy variables
    # Getting the corresponding municipality codes for the cleaned data
    model_data_clean = model_data.loc[X_clean.index]
    municipality_dummies = pd.get_dummies(model_data_clean['municipality_code'], prefix='mun', drop_first=True)
    
    # Ensuring dummies are numeric
    municipality_dummies = municipality_dummies.apply(pd.to_numeric, errors='coerce')
    
    # Combining features
    X_final = pd.concat([X_clean, municipality_dummies], axis=1)
    
    # Removing any remaining NaN values
    non_missing_final = X_final.notna().all(axis=1)
    X_final = X_final[non_missing_final]
    y_final = y_clean[non_missing_final]
    
    if len(X_final) == 0:
        logger.warning(f"No valid data after final cleaning for {outcome_var}")
        return None
    
    # Running OLS regression with clustered standard errors
    try:
        model = sm.OLS(y_final, X_final).fit(cov_type='cluster', 
                                           cov_kwds={'groups': model_data_clean.loc[X_final.index, 'municipality_code']})
        
        logger.info(f"Panel regression completed for {outcome_var}")
        logger.info(f"R-squared: {model.rsquared:.3f}")
        logger.info(f"Number of observations: {len(X_final)}")
        logger.info(f"Number of municipalities: {model_data_clean.loc[X_final.index, 'municipality_code'].nunique()}")
        
        return model
    except Exception as e:
        logger.error(f"Error in panel regression: {e}")
        logger.info("Falling back to simple OLS without clustering")
        try:
            model = sm.OLS(y_final, X_final).fit()
            logger.info(f"Simple OLS completed as fallback. R-squared: {model.rsquared:.3f}")
            return model
        except Exception as e2:
            logger.error(f"Error in fallback regression: {e2}")
            return None

def run_machine_learning(model_data, outcome_var='credit_portfolio'):
    """Run machine learning model to predict outcomes"""
    logger.info(f"Running Random Forest model for {outcome_var}")
    
    # Selecting simpler set for stability
    feature_columns = ['precipitation', 'precip_anomaly', 'month', 'quarter', 'year']
    
    # Adding climate features if available
    climate_features = ['precip_mean', 'precip_std', 'precip_min', 'precip_max']
    for cf in climate_features:
        if cf in model_data.columns:
            feature_columns.append(cf)
    
    available_features = [col for col in feature_columns if col in model_data.columns]
    
    X = model_data[available_features]
    y = model_data[outcome_var]
    
    # Cleaning the data
    X_clean = X.apply(pd.to_numeric, errors='coerce')
    y_clean = pd.to_numeric(y, errors='coerce')
    
    # Removing missing values
    non_missing = X_clean.notna().all(axis=1) & y_clean.notna()
    X_clean = X_clean[non_missing]
    y_clean = y_clean[non_missing]
    
    if len(X_clean) < 100:  # Need minimum samples
        logger.warning(f"Insufficient data for ML model: {len(X_clean)} samples")
        return None, None, None
    
    # Spliting data
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
    
    # Training model with simpler parameters for stability
    try:
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
        rf_model.fit(X_train, y_train)
        
        # Predicting and evaluating
        y_pred = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Random Forest results for {outcome_var}:")
        logger.info(f"MSE: {mse:.2f}")
        logger.info(f"R²: {r2:.3f}")
        
        # Featuring importance
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 5 features by importance:")
        for _, row in feature_importance.head().iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.3f}")
        
        return rf_model, feature_importance, {'mse': mse, 'r2': r2}
    
    except Exception as e:
        logger.error(f"Error in Random Forest: {e}")
        return None, None, None

def main():
    """Main modeling function"""
    logger.info("Starting modeling process")
    
    # Loading processed data
    try:
        financial_processed = pd.read_csv('data_processed/financial_processed.csv')
        precipitation_processed = pd.read_csv('data_processed/precipitation_processed.csv') 
        climate_features = pd.read_csv('data_processed/climate_features.csv')
        
        logger.info(f"Loaded financial data: {len(financial_processed)} records")
        logger.info(f"Loaded precipitation data: {len(precipitation_processed)} records")
        logger.info(f"Loaded climate features: {len(climate_features)} municipalities")
        
    except FileNotFoundError as e:
        logger.error(f"Required data files not found: {e}")
        logger.error("Please run steps 01 and 02 first")
        return {}
    
    # Checking if the data is available
    if len(financial_processed) == 0 or len(precipitation_processed) == 0:
        logger.error("No data available for modeling")
        return {}
    
    # Merging datasets
    merged_data = merge_datasets(financial_processed, precipitation_processed, climate_features)
    
    # Creating additional features
    model_data = create_additional_features(merged_data)
    
    # Saving modeling dataset
    model_data.to_csv('data_processed/modeling_dataset.csv', index=False)
    
    # Running analyses for both outcome variables
    outcomes = ['credit_portfolio', 'savings_deposits']
    results = {}
    
    for outcome in outcomes:
        logger.info(f"\n--- Analyzing impact on {outcome} ---")
        
        # Filtering to relevant data for this outcome
        outcome_data = model_data[model_data[outcome].notna()].copy()
        
        if len(outcome_data) > 100:
            # Trying simple regression
            regression_model = run_simple_regression(outcome_data, outcome)
            
            # Trying panel regression if simple works
            if regression_model is not None:
                panel_model = run_panel_regression(outcome_data, outcome)
            else:
                panel_model = None
                logger.warning(f"Skipping panel regression for {outcome} due to simple regression failure")
            
            # Running machine learning
            ml_model, feature_importance, ml_metrics = run_machine_learning(outcome_data, outcome)
            
            # Storing results
            results[outcome] = {
                'simple_regression': regression_model,
                'panel_regression': panel_model,
                'ml_model': ml_model,
                'feature_importance': feature_importance,
                'ml_metrics': ml_metrics,
                'n_observations': len(outcome_data)
            }
            
            # Saving model artifacts if available
            if ml_model is not None:
                joblib.dump(ml_model, f'outputs/model_artifacts/rf_model_{outcome}.pkl')
            if feature_importance is not None:
                feature_importance.to_csv(f'outputs/tables/feature_importance_{outcome}.csv', index=False)
            
            # Saving regression summary
            if regression_model is not None:
                with open(f'outputs/tables/regression_summary_{outcome}.txt', 'w') as f:
                    f.write(regression_model.summary().as_text())
        else:
            logger.warning(f"Insufficient data for {outcome}: only {len(outcome_data)} records")
    
    logger.info("Modeling completed successfully")
    return results

if __name__ == "__main__":
    results = main()