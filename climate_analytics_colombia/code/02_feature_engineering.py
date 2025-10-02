import pandas as pd
import numpy as np
import sys
import os
from fuzzywuzzy import fuzz

sys.path.append(os.path.dirname(__file__))
from utils import normalize_names, logger

def substring_match_cities(unmatched_df, api_df, dept_threshold=80):
    """Use substring matching for cities and fuzzy matching for departments"""
    
    # Getting unique city-department combinations
    unique_combinations = unmatched_df[['city_norm', 'department_norm', 'city', 'department']].drop_duplicates()
    
    logger.info(f"Applying substring matching for {len(unique_combinations)} unique city-department combinations...")
    
    matches = []
    
    for _, row in unique_combinations.iterrows():
        target_city = row['city_norm']
        target_dept = row['department_norm']
        
        # Finding API cities where the target city is a substring of the API city name or vice versa
        city_matches = []
        for _, api_row in api_df.iterrows():
            api_city = api_row['city_norm']
            api_dept = api_row['department_norm']
            
            # Check if either city contains the other
            if (target_city in api_city) or (api_city in target_city):
                city_matches.append(api_row)
        
        if len(city_matches) > 0:
            # For the city matches, finding the best department match using fuzzy matching
            best_match = None
            best_score = 0
            
            for api_row in city_matches:
                dept_score = fuzz.token_sort_ratio(target_dept, api_row['department_norm'])
                
                if dept_score >= dept_threshold and dept_score > best_score:
                    best_score = dept_score
                    best_match = api_row
            
            if best_match is not None:
                matches.append({
                    'original_city': row['city'],
                    'original_department': row['department'],
                    'matched_city': best_match['city_api'],
                    'matched_department': best_match['department_api'],
                    'municipality_code': best_match['municipality_code'],
                    'similarity_score': best_score
                })
    
    return pd.DataFrame(matches)

def match_precipitation_to_municipalities(precipitation_data, municipality_data):
    """Match precipitation data with municipality codes"""
    logger.info("Matching precipitation data with municipality codes")
    
    # Normalizing names in both datasets
    precipitation_data['city_norm'] = normalize_names(precipitation_data['city'])
    precipitation_data['department_norm'] = normalize_names(precipitation_data['department'])
    municipality_data['city_norm'] = normalize_names(municipality_data['city_api'])
    municipality_data['department_norm'] = normalize_names(municipality_data['department_api'])
    
    # First merge - exact match
    merged_df = pd.merge(
        precipitation_data,
        municipality_data,
        left_on=['city_norm', 'department_norm'],
        right_on=['city_norm', 'department_norm'],
        how='left'
    )
    
    # Check unmatched cities
    unmatched_mask = merged_df['municipality_code'].isna()
    unmatched_count = unmatched_mask.sum()
    
    logger.info(f"Exact matches: {len(merged_df) - unmatched_count} cities")
    logger.info(f"Unmatched after exact merge: {unmatched_count} cities")
    
    if unmatched_count > 0:
        # Get unmatched rows for substring matching
        unmatched_precipitation = precipitation_data[unmatched_mask].copy()
        
        # Applying substring matching
        substring_matches = substring_match_cities(unmatched_precipitation, municipality_data)
        
        if len(substring_matches) > 0:
            logger.info(f"Substring matches found: {len(substring_matches)} cities")
            
            # Applying substring matches to the main dataframe
            for _, match in substring_matches.iterrows():
                mask = (merged_df['city'] == match['original_city']) & (merged_df['department'] == match['original_department'])
                merged_df.loc[mask, 'municipality_code'] = match['municipality_code']
                
            logger.info("Applied substring matches")
            for _, match in substring_matches.iterrows():
                logger.info(f"'{match['original_city']}' ({match['original_department']}) -> '{match['matched_city']}' ({match['matched_department']})")
    
    # Manual assignment for specific cities that need special handling
    manual_codes = {
        'BOGOTA': '11001',
        'SAMPUES': '70670', 
        'SAN JOSE GUAVIARE': '95001'
    }
    
    for city_name, code in manual_codes.items():
        mask = (merged_df['city_norm'] == city_name) & (merged_df['municipality_code'].isna())
        if mask.any():
            merged_df.loc[mask, 'municipality_code'] = code
            logger.info(f"Manually assigned {city_name} with code: {code}")
    
    # Final unmatched count
    final_unmatched = merged_df['municipality_code'].isna().sum()
    if final_unmatched > 0:
        unmatched_cities = merged_df[merged_df['municipality_code'].isna()][['city', 'department']].drop_duplicates()
        logger.info(f"Still unmatched: {final_unmatched} records from {len(unmatched_cities)} cities")
        for _, city in unmatched_cities.iterrows():
            logger.info(f"- {city['city']} ({city['department']})")
    
    # Keeping only needed columns
    final_df = merged_df[['date', 'year', 'month', 'city', 'department', 'precipitation', 'municipality_code']]
    
    # Calculating matching statistics
    matched_count = final_df['municipality_code'].notna().sum()
    total_count = len(final_df)
    matched_cities = final_df[final_df['municipality_code'].notna()]['city'].nunique()
    total_cities = final_df['city'].nunique()
    
    logger.info(f"Precipitation matching results:")
    logger.info(f"Total records: {total_count}")
    logger.info(f"Matched records: {matched_count} ({matched_count/total_count*100:.1f}%)")
    logger.info(f"Cities matched: {matched_cities}/{total_cities} ({matched_cities/total_cities*100:.1f}%)")
    
    return final_df

def match_financial_to_municipalities(financial_data, municipality_data):
    """Match financial data with municipality codes"""
    logger.info("Matching financial data with municipality codes")
    
    # Normalizing names in both datasets
    financial_data['city_norm'] = normalize_names(financial_data['city'])
    financial_data['department_norm'] = normalize_names(financial_data['department'])
    municipality_data['city_norm'] = normalize_names(municipality_data['city_api'])
    municipality_data['department_norm'] = normalize_names(municipality_data['department_api'])
    
    # First merge - exact match on normalized city and department name
    merged_df = pd.merge(
        financial_data,
        municipality_data,
        left_on=['city_norm', 'department_norm'],
        right_on=['city_norm', 'department_norm'],
        how='left'
    )
    
    # Check unmatched records
    unmatched_mask = merged_df['municipality_code'].isna()
    unmatched_count = unmatched_mask.sum()
    
    logger.info(f"Exact city+department matches: {len(merged_df) - unmatched_count} records")
    logger.info(f"Unmatched after exact merge: {unmatched_count} records")
    
    if unmatched_count > 0:
        # Applying substring matching for remaining cities
        unmatched_data = financial_data[unmatched_mask].copy()
        substring_matches = substring_match_financial_cities(unmatched_data, municipality_data)
        
        if len(substring_matches) > 0:
            logger.info(f"Substring matches found: {len(substring_matches)} cities")
            
            # Applying substring matches to the main dataframe
            for _, match in substring_matches.iterrows():
                mask = (merged_df['city'] == match['original_city']) & \
                       (merged_df['department'] == match['original_department'])
                merged_df.loc[mask, 'municipality_code'] = match['municipality_code']
    
    # Manual assignment for specific cities
    manual_codes = {
        'BOGOTA': '11001',
        'SAMPUES': '70670', 
        'SAN JOSE GUAVIARE': '95001'
    }
    
    for city_name, code in manual_codes.items():
        mask = (merged_df['city_norm'] == city_name) & (merged_df['municipality_code'].isna())
        if mask.any():
            merged_df.loc[mask, 'municipality_code'] = code
            logger.info(f"Manually assigned {city_name} with code: {code}")
    
    # Selecting only the columns we want to keep
    columns_to_keep = ['date', 'credit_portfolio', 'savings_deposits', 'municipality_code']
    final_df = merged_df[columns_to_keep]
    
    # Final stats
    matched_count = final_df['municipality_code'].notna().sum()
    logger.info(f"Financial data matching results:")
    logger.info(f"Total records: {len(final_df)}")
    logger.info(f"Records with municipality codes: {matched_count} ({matched_count/len(final_df)*100:.1f}%)")
    
    return final_df

def substring_match_financial_cities(unmatched_financial, api_df, dept_threshold=80):
    """Use substring matching for cities and fuzzy matching for departments - for financial data"""
    
    # Getting unique city-department combinations
    unique_combinations = unmatched_financial[['city_norm', 'department_norm', 'city', 'department']].drop_duplicates()
    
    logger.info(f"Applying substring matching for {len(unique_combinations)} unique city-department combinations...")
    
    matches = []
    
    for _, row in unique_combinations.iterrows():
        financial_city = row['city_norm']
        financial_dept = row['department_norm']
        
        # Finding API cities where either city name contains the other
        for _, api_row in api_df.iterrows():
            api_city = api_row['city_norm']
            api_dept = api_row['department_norm']
            
            # Checking if either city contains the other
            if (financial_city in api_city) or (api_city in financial_city):
                # Checking if departments are similar
                dept_score = fuzz.token_sort_ratio(financial_dept, api_dept)
                
                if dept_score >= dept_threshold:
                    matches.append({
                        'original_city': row['city'],
                        'original_department': row['department'],
                        'matched_city': api_row['city_api'],
                        'matched_department': api_row['department_api'],
                        'municipality_code': api_row['municipality_code'],
                        'similarity_score': dept_score
                    })
                    break  # Take the first good match
    
    return pd.DataFrame(matches)

def create_climate_features(precipitation_data):
    """Create climate variability features from precipitation data"""
    logger.info("Creating climate variability features")
    
    # Sorting by municipality and date
    precipitation_data = precipitation_data.sort_values(['municipality_code', 'date'])
    
    # Calculating rolling statistics for climate variability
    climate_features = precipitation_data.groupby('municipality_code').agg({
        'precipitation': [
            'mean',  # Average precipitation
            'std',   # Precipitation variability
            'min',   # Minimum precipitation
            'max'    # Maximum precipitation
        ]
    }).round(2)
    
    # Flatten column names
    climate_features.columns = [f'precip_{col[1]}' for col in climate_features.columns]
    climate_features = climate_features.reset_index()
    
    logger.info(f"Created climate features for {len(climate_features)} municipalities")
    return climate_features

def main():
    """Main function for feature engineering"""
    logger.info("Starting feature engineering process")
    
    # Loading raw data
    precipitation_data = pd.read_csv('data_raw/precipitation_raw.csv')
    municipality_data = pd.read_csv('data_raw/municipality_reference.csv')
    financial_data = pd.read_csv('data_raw/financial_raw.csv')
    
    # Converting date columns
    precipitation_data['date'] = pd.to_datetime(precipitation_data['date'])
    financial_data['date'] = pd.to_datetime(financial_data['date'])
    
    # Matching datasets
    precipitation_matched = match_precipitation_to_municipalities(precipitation_data, municipality_data)
    financial_matched = match_financial_to_municipalities(financial_data, municipality_data)
    
    # Creating climate features
    climate_features = create_climate_features(precipitation_matched)
    
    # Saving processed data
    precipitation_matched.to_csv('data_processed/precipitation_processed.csv', index=False)
    financial_matched.to_csv('data_processed/financial_processed.csv', index=False)
    climate_features.to_csv('data_processed/climate_features.csv', index=False)
    
    logger.info("Feature engineering completed successfully")
    return precipitation_matched, financial_matched, climate_features

if __name__ == "__main__":
    precipitation_matched, financial_matched, climate_features = main()