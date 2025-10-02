# Climate Impact on Financial Outcomes in Colombian Municipalities

## Project Overview

This project analyzes the relationship between climate variability (specifically precipitation patterns) and financial outcomes at the municipal level in Colombia. The analysis integrates climate data from IDEAM with financial data from Superintendencia Financiera to quantify how precipitation variability affects credit portfolios and savings deposits across Colombian municipalities.

### Research Question
**"How does climate variability, specifically precipitation patterns, affect financial outcomes at the municipal level in Colombia?"**

---

## Data Sources

### Climate Data
- **Source**: Instituto de Hidrología, Meteorología y Estudios Ambientales (IDEAM)
- **Dataset**: Monthly Precipitation Indices (2000-2024)
- **Variables**: Monthly precipitation, climate norms, anomalies
- **Coverage**: 200+ municipalities across Colombia
- **Access**: [Public Download](https://bart.ideam.gov.co/indiecosistemas/ind/clima/datos/D_indice_precipitacion.xlsx)
- **Documentation**: [Methodology PDF](https://bart.ideam.gov.co/indiecosistemas/ind/clima/hm/HM_indice_de_precipitacion.pdf)

### Financial Data
- **Source**: Superintendencia Financiera de Colombia
- **Dataset**: Credit Portfolios and Savings Deposits by Municipality
- **Variables**: Credit portfolio balances, savings deposits, municipality codes
- **Coverage**: National coverage of financial institutions
- **Access**: [Public API](https://www.datos.gov.co/resource/u2wk-tfe3.json)
- **Documentation**: [Data Description](https://www.datos.gov.co/Econom-a-y-Finanzas/Saldo-de-las-captaciones-y-colocaciones-por-munici/u2wk-tfe3/about_data)

### Geographic Reference Data
- **Source**: Colombian Government Open Data Portal
- **Dataset**: Municipalities of Colombia
- **Variables**: Municipality names, department names, DANE codes
- **Access**: [Public API](https://www.datos.gov.co/resource/gdxc-w37w.json)

---

## Methodology

### Data Processing Pipeline

| Step | Description | Output |
|------|-------------|--------|
| **1. Data Ingestion & Cleaning** | Download and clean raw data from multiple sources | Cleaned CSV files |
| **2. Municipality Matching** | Fuzzy string matching (85% accuracy) with manual validation | Matched municipality codes |
| **3. Feature Engineering** | Climate anomalies, rolling statistics, extreme events | Modeling-ready dataset |
| **4. Statistical Analysis** | Panel regression + Machine Learning validation | Statistical results & insights |

### Analytical Approach
- **Panel Data Regression**: Fixed effects models with clustered standard errors
- **Machine Learning**: Random Forest for feature importance and non-linear relationships
- **Climate Anomalies**: Localized precipitation deviations from municipal norms
- **Temporal Analysis**: Monthly aggregation and seasonal pattern detection

---

## Project Structure

```
climate_analytics_colombia/
├── 📁 code/
│   ├── 01_ingest_clean.py      # Data extraction and cleaning
│   ├── 02_feature_engineering.py # Feature creation and merging
│   ├── 03_modeling.py          # Statistical and ML modeling
│   ├── 04_eval_visuals.py      # Visualization and presentation
│   └── utils.py                # Helper functions
├── 📁 data_processed/          # Processed data (not in repo)
├── 📁 outputs/
│   ├── 📁 figures/             # Generated visualizations
│   ├── 📁 tables/              # Statistical results
│   └── 📁 model_artifacts/     # Trained models
├── 📁 docs/
│   └── slides.pptx            # PowerPoint presentation
├── 📄 data_links.txt          # Data source documentation
├── 📄 requirements.txt        # Python dependencies
├── 📄 run.sh                  # One-click execution script
└── 📄 README.md               # This file
```

---

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Git
- Internet connection (for data download)

### Setup & Execution
```bash
# Clone and run complete pipeline
git clone https://github.com/paula-cadena/PolicyPulse.git
cd climate_analytics_colombia
chmod +x run.sh
./run.sh
```

---

## Outputs Generated

### Visualizations (`outputs/figures/`)
- `climate_distributions.png` - Precipitation patterns and anomalies
- `financial_distributions.png` - Credit and savings distributions  
- `climate_financial_relationships.png` - Relationship analysis

### Statistical Results (`outputs/tables/`)
- `summary_statistics.csv` - Dataset descriptive statistics
- `feature_importance_*.csv` - Machine learning feature importance
- `policy_implications.csv` - Policy recommendations
- `regression_summary_*.txt` - Statistical model results

### Models (`outputs/model_artifacts/`)
- `rf_model_credit_portfolio.pkl` - Trained Random Forest for credit
- `rf_model_savings_deposits.pkl` - Trained Random Forest for savings

### 🎤 Presentation (`docs/`)
- `slides.pptx` - Comprehensive PowerPoint presentation

---

## 🔄 Reproducibility

This project follows reproducible research standards:

- **Environment Management**: Virtual environment with pinned dependencies
- **One-Click Execution**: Complete pipeline via `./run.sh`
- **Modular Code**: Separated by data processing stages
- **Comprehensive Logging**: Detailed progress and error reporting
- **Data Provenance**: All data sources documented with links

---

## Technical Specifications

### Python Packages
Core dependencies (see `requirements.txt` for complete list):
- `pandas`, `numpy` - Data manipulation
- `statsmodels`, `scikit-learn` - Statistical modeling  
- `matplotlib`, `seaborn` - Visualization
- `python-pptx` - Presentation generation
- `sodapy` - API data access

### System Requirements
- Python 3.8+
- 4GB RAM minimum
- 1GB disk space for data and outputs
- Internet connection for data download

---

## 📄 License & Attribution

### Data Licenses
- Climate Data: Public data from IDEAM (Colombian government)
- Financial Data: Public data from Superintendencia Financiera  
- Geographic Data: Creative Commons Attribution

