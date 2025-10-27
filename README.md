# Pharmaceutical Sales Prediction - Setup Guide

### 1. Install Dependencies

```bash
# Navigate to your project folder
cd your-project-folder

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install pandas numpy scikit-learn xgboost lightgbm torch requests beautifulsoup4 selenium matplotlib seaborn scipy
```

## Running the System

### Option 1: Quick Demo (Sample Data)

Run with built-in sample data - no data file needed:

```bash
python main_pipeline.py --sample
```

This will:
- Generate 1000 synthetic claims records
- Train all 6 models
- Generate predictions
- Create analysis reports and visualizations
- Save outputs to `pipeline_outputs/` folder

**Expected runtime: ~1 minute**

### Option 2: Use Your Own Data

```bash
python main_pipeline.py --data your_claims_data.csv
```

#### Required CSV Columns:

Your `your_claims_data.csv` must have these columns:

| Column | Type | Example |
|--------|------|---------|
| `patient_id` | string | `PAT_001` |
| `patient_age` | integer | `65` |
| `patient_gender` | string | `M` or `F` |
| `prescriber_npi_nm` | string | `DR_SMITH` |
| `prescriber_npi_state_cd` | string | `CA` |
| `drug_generic_name` | string | `Pembrolizumab` |
| `drug_brand_name` | string | `Keytruda` |
| `service_date_dd` | date | `2024-01-15` |
| `transaction_status` | string | `Dispensed` |
| `days_supply_val` | integer | `30` |
| `total_paid_amt` | float | `12500.00` |
| `patient_to_pay_amt` | float | `500.00` |
| `payer_channel_name` | string | `Medicare` |
| `payer_name` | string | `UnitedHealthcare` |
| `clinical_service_line` | string | `Oncology` |
| `reject_reason_1_cd` | string | `MR` (can be empty) |
| `pharmacy_npi_nbr` | string | `1234567890` |

## Output Files

All outputs are saved in `pipeline_outputs/` folder:

| File | Description |
|------|-------------|
| `predictions_YYYYMMDD_HHMMSS.csv` | **Main output** - HCP-Drug predictions |
| `executive_summary_*.txt` | Summary report with key metrics |
| `cleaned_data_*.csv` | Preprocessed data |
| `model_metrics_*.csv` | Performance metrics (R², MAPE, etc.) |
| `model_viz_*/` | Model performance charts |
| `feature_viz_*/` | Feature analysis charts |

### Key Output: predictions_*.csv

The main prediction file contains:

```csv
hcp_id,Drug_name,Sales_from_db,Predicted_sales,Combined_multiplier
DR_001,Keytruda,45000.00,52500.00,1.167
DR_002,Opdivo,32000.00,35200.00,1.100
...
```

## Understanding the Output

### Prediction Columns

- **Sales_from_db**: Historical sales from your data
- **Predicted_sales**: Model's prediction (use this for forecasting)
- **Combined_multiplier**: Adjustment factor applied (>1 means growth, <1 means decline)

### Multiplier Interpretation

```
1.2 = 20% predicted increase
1.0 = No change expected
0.8 = 20% predicted decrease
```

### Model Performance

Check `executive_summary_*.txt` for:
- **R² Score**: >0.7 is good, >0.8 is excellent
- **MAPE**: <15% is good, <10% is excellent
- **Direction Accuracy**: >75% means model predicts trends well

## Folder Structure

After first run:

```
your-project-folder/
├── main_pipeline.py
├── pharma_sales_prediction.py
├── web_scraping_module.py
├── model_analysis.py
├── feature_analysis.py
│
├── pipeline_outputs/          
│   ├── predictions_*.csv
│   ├── executive_summary_*.txt
│   ├── model_viz_*/
│   └── feature_viz_*/
│
└── web_cache/                 
    └── *.pkl
```
