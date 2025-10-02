import pandas as pd
import numpy as np
import requests
from io import BytesIO
import urllib3
from sodapy import Socrata
from fuzzywuzzy import fuzz, process
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def normalize_names(series):
    """Normalize names to uppercase without special characters"""
    return (
        series
        .str.upper()
        .str.normalize('NFKD')
        .str.encode('ascii', errors='ignore')
        .str.decode('utf-8')
        .str.replace(r'[^A-Z\s]', '', regex=True)
        .str.strip()
    )

def get_superfinanciera_department_mapping():
    """Create mapping between Superfinanciera codes and department names"""
    mapping = {
        '3': 'BOGOTÁ D.C.',
        '1': 'ANTIOQUIA', 
        '2': 'ATLÁNTICO',
        '4': 'BOLÍVAR',
        '5': 'BOYACÁ',
        '6': 'CALDAS',
        '7': 'CAQUETÁ',
        '8': 'CAUCA',
        '9': 'CESAR',
        '10': 'CÓRDOBA',
        '11': 'CUNDINAMARCA',
        '12': 'CHOCÓ',
        '13': 'HUILA',
        '14': 'LA GUAJIRA',
        '15': 'MAGDALENA',
        '16': 'META',
        '17': 'NARIÑO',
        '18': 'NORTE DE SANTANDER',
        '19': 'QUINDÍO',
        '20': 'RISARALDA',
        '21': 'SANTANDER',
        '22': 'SUCRE',
        '23': 'TOLIMA',
        '24': 'VALLE DEL CAUCA',
        '25': 'ARAUCA',
        '26': 'CASANARE',
        '27': 'PUTUMAYO',
        '28': 'ARCHIPIÉLAGO DE SAN ANDRÉS, PROVIDENCIA Y SANTA CATALINA',
        '29': 'AMAZONAS',
        '30': 'GUAINÍA',
        '31': 'GUAVIARE',
        '32': 'VAUPÉS',
        '33': 'VICHADA'
    }
    return mapping