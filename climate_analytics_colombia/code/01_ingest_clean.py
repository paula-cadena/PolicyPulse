import pandas as pd
import numpy as np
import requests
from io import BytesIO
from sodapy import Socrata
from fuzzywuzzy import fuzz
import sys
import os

sys.path.append(os.path.dirname(__file__))
from utils import normalize_names, get_superfinanciera_department_mapping, logger

def get_city_department_mapping():
    """Extract city to department mapping from the 'Índice' sheet"""
    logger.info("Extracting city-department mapping from IDEAM data")
    
    url = "https://bart.ideam.gov.co/indiecosistemas/ind/clima/datos/D_indice_precipitacion.xlsx"
    
    response = requests.get(url, verify=False, timeout=60)
    excel_file = BytesIO(response.content)
    
    # Reading Índice sheet starting from row 5
    index_df = pd.read_excel(excel_file, sheet_name='Índice', header=4)
    
    # Processing both sections (design in two halves)
    section1 = index_df.iloc[:, 1:5].copy()
    section1.columns = ['station', 'item', 'department', 'city']
    section2 = index_df.iloc[:, 5:9].copy()
    section2.columns = ['station', 'item', 'department', 'city']
    
    # Combining and clean
    combined_df = pd.concat([section1, section2], ignore_index=True)
    combined_df = combined_df.dropna(subset=['city'])
    
    # Creating mapping dictionary
    city_dept_mapping = {}
    for _, row in combined_df.iterrows():
        city = str(row['city']).strip().upper()
        department = str(row['department']).strip().upper()
        city_dept_mapping[city] = department
    
    logger.info(f"Extracted mapping for {len(city_dept_mapping)} cities")
    return city_dept_mapping

def extract_precipitation_data():
    """Extract and clean precipitation data with department info"""
    logger.info("Extracting precipitation data from IDEAM")
    
    url = "https://bart.ideam.gov.co/indiecosistemas/ind/clima/datos/D_indice_precipitacion.xlsx"
    
    response = requests.get(url, verify=False, timeout=60)
    excel_file = BytesIO(response.content)
    xl = pd.ExcelFile(excel_file)
    
    # Getting city-department mapping
    city_dept_mapping = get_city_department_mapping()
    
    all_data = []
    
    for sheet in xl.sheet_names:
        if sheet != 'Índice':
            df = pd.read_excel(excel_file, sheet_name=sheet, header=3)
            df = df.iloc[:, 1:]  # Removing empty column
            
            # Setting column names
            cols = ['year', 'annual', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                   'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
            df.columns = cols[:len(df.columns)]
            
            df['city'] = sheet
            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')

            # Cleanning precipitation data
            precip_cols = [col for col in df.columns if col not in ['year', 'city']]
            for col in precip_cols:
                df[col] = df[col].replace('(-)', np.nan)
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df[df['year'].between(2000, 2024)]
            
            # Adding department information
            normalized_city = sheet.upper().strip()
            df['department'] = city_dept_mapping.get(normalized_city, 'UNKNOWN')
            
            all_data.append(df)
    
    # Combining all data
    result = pd.concat(all_data, ignore_index=True)
    
    # Converting to long format (one row per city-year-month)
    result = result.drop('annual', axis=1)
    month_columns = [col for col in result.columns if col not in ['year', 'city', 'department']]
    
    long_df = result.melt(
        id_vars=['year', 'city', 'department'],
        value_vars=month_columns,
        var_name='month',
        value_name='precipitation'
    )
    
    # Creating date column (to match financial data)
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    long_df['month_num'] = long_df['month'].map(month_map)
    long_df['temp_date'] = pd.to_datetime(long_df['year'].astype(str) + '-' + long_df['month_num'].astype(str) + '-01')
    long_df['date'] = long_df['temp_date'] + pd.offsets.MonthEnd(0)
    long_df = long_df.drop(['month_num', 'temp_date'], axis=1)
    
    logger.info(f"Processed precipitation data: {len(long_df)} records from {long_df['city'].nunique()} cities")
    return long_df[['date', 'year', 'month', 'city', 'department', 'precipitation']]

def get_municipality_reference_data():
    """Get municipality reference data from API (DANE codes)"""
    logger.info("Fetching municipality reference data from API")
    
    client = Socrata("www.datos.gov.co", None)
    results = client.get("gdxc-w37w", limit=2000)
    api_df = pd.DataFrame.from_records(results)
    
    api_df = api_df[['nom_mpio', 'dpto', 'cod_mpio']].copy()
    api_df = api_df.rename(columns={
        'nom_mpio': 'city_api', 
        'dpto': 'department_api',
        'cod_mpio': 'municipality_code'
    })
    
    logger.info(f"Fetched {len(api_df)} municipality records")
    return api_df.drop_duplicates()

def extract_financial_data(limit=290000):
    """Extract and clean financial data from Superfinanciera"""
    logger.info("Extracting financial data from Superfinanciera API")
    
    client = Socrata("www.datos.gov.co", None)
    results = client.get("u2wk-tfe3", limit=limit)
    results_df = pd.DataFrame.from_records(results)
    
    df_clean = results_df.copy()
    
    # Renaming columns (to English and more intuitive names)
    column_mapping = {
        "fecha_de_corte": "date",
        "cartera_de_cr_ditos": "credit_portfolio", 
        "dep_sitos_de_ahorro": "savings_deposits",
        "c_digo_del_departamento": "superfinanciera_dept_code",
        "nombre_del_municipio": "city"
    }
    
    df_clean = df_clean.rename(columns=column_mapping)
    
    # Converting date format to datetime for proper merging
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    
    # Mapping Superfinanciera codes to department names (different from DANE)
    sf_mapping = get_superfinanciera_department_mapping()
    df_clean['department'] = df_clean['superfinanciera_dept_code'].astype(str).map(sf_mapping)
    
    # Converting numeric columns
    df_clean['credit_portfolio'] = pd.to_numeric(df_clean['credit_portfolio'], errors='coerce')
    df_clean['savings_deposits'] = pd.to_numeric(df_clean['savings_deposits'], errors='coerce')
    
    # Sum financial indicators by date, city, and department 
    # (original data has entries by financial institution)
    logger.info("Aggregating financial data by date, city, and department...")
    
    # Saving aggregation stats for logging
    original_records = len(df_clean)
    original_city_date_combinations = df_clean[['date', 'city', 'department']].drop_duplicates().shape[0]
    
    # Groupping by date, city, and department, then sum the financial indicators
    aggregated_df = df_clean.groupby(['date', 'city', 'department']).agg({
        'credit_portfolio': 'sum',
        'savings_deposits': 'sum'
    }).reset_index()
    
    # Saving after aggregation stats for logging
    aggregated_records = len(aggregated_df)
    aggregated_city_date_combinations = aggregated_df[['date', 'city', 'department']].drop_duplicates().shape[0]
    
    # Logging summary statistics
    logger.info(f"Aggregation summary:")
    logger.info(f"  Original records: {original_records}")
    logger.info(f"  After aggregation: {aggregated_records}")
    logger.info(f"  Original unique city-date combinations: {original_city_date_combinations}")
    logger.info(f"  After aggregation: {aggregated_city_date_combinations}")
    logger.info(f"  Average records aggregated per group: {original_records / aggregated_records:.1f}")
    
    logger.info(f"Financial data summary after aggregation:")
    logger.info(f"  Total credit portfolio: {aggregated_df['credit_portfolio'].sum():,.0f}")
    logger.info(f"  Total savings deposits: {aggregated_df['savings_deposits'].sum():,.0f}")
    logger.info(f"  Date range: {aggregated_df['date'].min()} to {aggregated_df['date'].max()}")
    logger.info(f"  Number of unique cities: {aggregated_df['city'].nunique()}")
    logger.info(f"  Number of unique departments: {aggregated_df['department'].nunique()}")
    
    return aggregated_df

def main():
    """Main function for data ingestion and cleaning"""
    logger.info("Starting data ingestion and cleaning process")
    
    # Extracting all data sources
    precipitation_data = extract_precipitation_data()
    municipality_data = get_municipality_reference_data() 
    financial_data = extract_financial_data()
    
    # Saving raw data (for reference)
    precipitation_data.to_csv('data_raw/precipitation_raw.csv', index=False)
    municipality_data.to_csv('data_raw/municipality_reference.csv', index=False)
    financial_data.to_csv('data_raw/financial_raw.csv', index=False)
    
    logger.info("Data ingestion completed successfully")
    return precipitation_data, municipality_data, financial_data

if __name__ == "__main__":
    precipitation_data, municipality_data, financial_data = main()