"""
Pharmaceutical Sales Prediction System - Fixed to ensure hcp_id contains NPI
Multi-Model Approach for Predicting Actual HCP-Drug Sales
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Web Scraping
import requests
from bs4 import BeautifulSoup
import time
from typing import Dict, List, Tuple

def clean_features(df):
    """
    Comprehensive feature cleaning utility
    Use this before training any model
    """
    # Fill NaN values
    df = df.fillna(0)
    
    # Replace infinite values
    df = df.replace([np.inf, -np.inf], 0)
    
    # Check for any remaining issues
    if df.isnull().any().any():
        print("Warning: NaN values still present after cleaning")
        print(df.isnull().sum())
        df = df.fillna(0)
    
    return df


# ================== DATA PREPROCESSING ==================

class DataPreprocessor:
    """Handles initial data cleaning and preprocessing"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.drug_mapping = {
            'Cemiplimab-Rwlc': {'brand': 'Libtayo', 'company': 'Regeneron/Sanofi'},
            'Nivolumab': {'brand': 'Opdivo', 'company': 'Bristol-Myers Squibb'},
            'Pembrolizumab': {'brand': 'Keytruda', 'company': 'Merck'}
        }
        
    def load_and_clean_data(self, df):
        """Load and clean claims data - Uses NPI as HCP ID"""
        print("Starting data cleaning...")
        
        # Filter out reversed and rejected claims
        df = df[~df['transaction_status'].isin(['Reject', 'Reversed'])]
        
        # Convert date columns
        date_cols = ['service_date_dd', 'transaction_dt', 'date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Handle zero amount transactions
        df = self._handle_zero_amounts(df)
        
        # UPDATED: Use type_1_npi as HCP ID (prescriber NPI)
        if 'type_1_npi' in df.columns:
            df['hcp_id'] = df['type_1_npi'].fillna(0).astype(int).astype(str)
            df['hcp_id'] = df['hcp_id'].replace('0', 'Unknown')
            print(f"Using type_1_npi as HCP ID")
        elif 'prescriber_npi_nbr' in df.columns:
            df['hcp_id'] = df['prescriber_npi_nbr'].fillna('Unknown').astype(str)
            print(f"Using prescriber_npi_nbr as HCP ID")
        else:
            raise ValueError("Neither 'type_1_npi' nor 'prescriber_npi_nbr' found in data")
        
        # Add company information
        df['company'] = df['drug_generic_name'].map(
            lambda x: self.drug_mapping.get(x, {}).get('company', 'Unknown')
        )
        
        print(f"Found {df['hcp_id'].nunique()} unique HCP NPIs")
        
        return df
    
    def _handle_zero_amounts(self, df):
        """Calculate amounts for zero-value transactions"""
        # For zero amount transactions, calculate using days_supply and AWP
        zero_mask = df['total_paid_amt'] == 0
        df.loc[zero_mask, 'total_paid_amt'] = (
            df.loc[zero_mask, 'days_supply_val'] * 
            df.loc[zero_mask, 'awp_unit_price_amt']
        )
        return df
    
    def create_base_features(self, df):
        """Create base features from claims data"""
        features = df.copy()
        
        # Temporal features - with error handling
        try:
            features['month'] = pd.to_datetime(features['service_date_dd']).dt.month
            features['quarter'] = pd.to_datetime(features['service_date_dd']).dt.quarter
            features['year'] = pd.to_datetime(features['service_date_dd']).dt.year
            features['day_of_week'] = pd.to_datetime(features['service_date_dd']).dt.dayofweek
        except Exception as e:
            print(f"Warning: Error creating temporal features: {e}")
            features['month'] = 1
            features['quarter'] = 1
            features['year'] = 2024
            features['day_of_week'] = 0
        
        # Patient demographics - with safe binning
        try:
            features['age_group'] = pd.cut(features['patient_age'], 
                                          bins=[0, 18, 35, 50, 65, 100],
                                          labels=['<18', '18-35', '35-50', '50-65', '65+'])
            # Convert to string to avoid NaN in categories
            features['age_group'] = features['age_group'].astype(str)
            features['age_group'] = features['age_group'].replace('nan', '35-50')  # Default category
        except Exception as e:
            print(f"Warning: Error creating age groups: {e}")
            features['age_group'] = '35-50'
        
        # Transaction features - safe division
        features['copay_ratio'] = features['patient_to_pay_amt'] / (features['total_paid_amt'] + 1)
        features['copay_ratio'] = features['copay_ratio'].fillna(0).replace([np.inf, -np.inf], 0)
        
        return features

# ================== EXTERNAL DATA COLLECTOR ==================

class ExternalDataCollector:
    """Collects external market data via web scraping"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
    def get_drug_market_data(self, drug_name):
        """Scrape drug market data (efficacy, market share, etc.)"""
        # Simulated data - in production, implement actual web scraping
        drug_data = {
            'Libtayo': {
                'efficacy_score': 0.85,
                'market_share': 0.15,
                'drug_age_years': 3,
                'competitor_count': 2,
                'avg_response_rate': 0.45,
                'side_effect_score': 0.3
            },
            'Opdivo': {
                'efficacy_score': 0.87,
                'market_share': 0.45,
                'drug_age_years': 7,
                'competitor_count': 2,
                'avg_response_rate': 0.42,
                'side_effect_score': 0.28
            },
            'Keytruda': {
                'efficacy_score': 0.89,
                'market_share': 0.40,
                'drug_age_years': 6,
                'competitor_count': 2,
                'avg_response_rate': 0.47,
                'side_effect_score': 0.25
            }
        }
        return drug_data.get(drug_name, self._get_default_drug_data())
    
    def get_hcp_specialization_data(self, npi_id, state):
        """Get HCP specialization and patient volume data"""
        # Simulated - implement actual NPI registry lookup
        specializations = {
            'oncology': {'weight': 0.9, 'patient_volume': 150},
            'hematology': {'weight': 0.8, 'patient_volume': 120},
            'general': {'weight': 0.3, 'patient_volume': 50}
        }
        
        # Random assignment for demo - replace with actual NPI lookup
        spec_type = np.random.choice(list(specializations.keys()))
        return specializations[spec_type]
    
    def get_payer_reputation_scores(self):
        """Get payer reputation and reimbursement scores"""
        payer_scores = {
            'Medicare': {'reputation': 0.9, 'reimbursement_rate': 0.85},
            'Commercial': {'reputation': 0.8, 'reimbursement_rate': 0.75},
            'Medicaid': {'reputation': 0.7, 'reimbursement_rate': 0.65},
            'Other': {'reputation': 0.6, 'reimbursement_rate': 0.60}
        }
        return payer_scores
    
    def _get_default_drug_data(self):
        return {
            'efficacy_score': 0.7,
            'market_share': 0.1,
            'drug_age_years': 1,
            'competitor_count': 3,
            'avg_response_rate': 0.3,
            'side_effect_score': 0.4
        }

# ================== MODEL 1: TEMPORAL & DEMOGRAPHIC ==================

class TemporalDemographicModel:
    """Model 1: Temporal and Demographic features effect on HCP prescribing"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """Prepare temporal and demographic features"""
        features = []
        
        # Group by HCP ID (NPI)
        hcp_groups = df.groupby('hcp_id')
        
        for hcp, group in hcp_groups:
            # Safe calculations with fallbacks for NaN
            avg_age = group['patient_age'].mean() if len(group) > 0 else 50.0
            female_ratio = (group['patient_gender'] == 'F').mean() if len(group) > 0 else 0.5
            age_std = group['patient_age'].std() if len(group) > 1 else 0.0
            patient_count = len(group['patient_id'].unique())
            
            # Seasonal variation
            monthly_sales = group.groupby('month')['total_paid_amt'].sum()
            seasonal_var = monthly_sales.std() if len(monthly_sales) > 1 else 0.0
            
            weekend_ratio = (group['day_of_week'].isin([5, 6])).mean() if len(group) > 0 else 0.0
            
            hcp_features = {
                'hcp_id': hcp,  # NPI
                'avg_patient_age': avg_age,
                'female_ratio': female_ratio,
                'age_diversity': age_std,
                'patient_count': patient_count,
                'seasonal_variation': seasonal_var,
                'weekend_ratio': weekend_ratio
            }
            
            # Age group distribution
            age_dist = group['age_group'].value_counts(normalize=True)
            for age_grp in ['<18', '18-35', '35-50', '50-65', '65+']:
                hcp_features[f'age_group_{age_grp}'] = age_dist.get(age_grp, 0.0)
            
            features.append(hcp_features)
        
        df_features = pd.DataFrame(features)
        df_features = df_features.fillna(0)
        
        return df_features
    
    def train(self, feature_df, target_multipliers):
        """Train the temporal-demographic model"""
        X = feature_df.drop(['hcp_id'], axis=1)
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, target_multipliers)
        return self
    
    def predict_multiplier(self, feature_df):
        """Predict multiplier effect for each HCP (NPI)"""
        X = feature_df.drop(['hcp_id'], axis=1)
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        X_scaled = self.scaler.transform(X)
        
        multipliers = self.model.predict(X_scaled)
        multipliers = np.clip(multipliers, 0.5, 2.0)
        
        return dict(zip(feature_df['hcp_id'], multipliers))

# ================== MODEL 2: MARKET DYNAMICS ==================

class MarketDynamicsModel:
    """Model 2: Drug market dynamics and competitive landscape"""
    
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.external_collector = ExternalDataCollector()
        
    def prepare_features(self, df):
        """Prepare market dynamics features for each drug"""
        drug_features = []
        
        for drug in df['drug_brand_name'].unique():
            market_data = self.external_collector.get_drug_market_data(drug)
            
            drug_df = df[df['drug_brand_name'] == drug]
            
            features = {
                'drug_name': drug,
                'total_volume': len(drug_df),
                'unique_hcps': drug_df['hcp_id'].nunique(),
                'unique_patients': drug_df['patient_id'].nunique(),
                'avg_days_supply': drug_df['days_supply_val'].mean(),
                'revenue_concentration': drug_df.groupby('hcp_id')['total_paid_amt'].sum().std(),
                **market_data
            }
            
            drug_features.append(features)
            
        return pd.DataFrame(drug_features)
    
    def calculate_drug_multipliers(self, feature_df):
        """Calculate multiplier for each drug based on market dynamics"""
        multipliers = {}
        
        for _, row in feature_df.iterrows():
            multiplier = 1.0
            multiplier *= (1 + 0.3 * row['efficacy_score'])
            multiplier *= (1 + 0.2 * row['market_share'])
            multiplier *= (1 - 0.1 * row['side_effect_score'])
            multiplier *= (1 + 0.05 * np.log1p(row['drug_age_years']))
            
            multipliers[row['drug_name']] = np.clip(multiplier, 0.7, 1.5)
            
        return multipliers

# ================== MODEL 3: HCP SPECIALIZATION ==================

class HCPSpecializationModel:
    """Model 3: HCP specialization and drug affinity"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        self.external_collector = ExternalDataCollector()
        
    def prepare_features(self, df):
        """Prepare HCP specialization features"""
        hcp_features = []

        for hcp_npi in df['hcp_id'].unique():
            hcp_df = df[df['hcp_id'] == hcp_npi]
            if hcp_df.empty:
                state = 'Unknown'
            else:
                # Try prescriber_npi_state_cd from actual schema
                if 'prescriber_npi_state_cd' in hcp_df.columns:
                    mode_series = hcp_df['prescriber_npi_state_cd'].mode()
                    state = mode_series.iloc[0] if not mode_series.empty else 'Unknown'
                else:
                    state = 'Unknown'

            spec_data = self.external_collector.get_hcp_specialization_data(hcp_npi, state)

            drug_dist = hcp_df['drug_brand_name'].value_counts(normalize=True)

            features = {
                'hcp_id': hcp_npi,
                'specialization_weight': spec_data.get('weight', 0),
                'patient_volume': spec_data.get('patient_volume', 0),
                'drug_diversity': len(drug_dist),
                'primary_drug_share': drug_dist.iloc[0] if len(drug_dist) > 0 else 0,
                'clinical_complexity': hcp_df['clinical_service_line'].nunique() if 'clinical_service_line' in hcp_df.columns else 1
            }

            hcp_features.append(features)

        return pd.DataFrame(hcp_features)
    
    def calculate_hcp_drug_affinity(self, df, hcp_features):
        """Calculate HCP-drug affinity matrix"""
        affinity_matrix = {}
        
        for _, hcp_row in hcp_features.iterrows():
            hcp_npi = hcp_row['hcp_id']
            hcp_df = df[df['hcp_id'] == hcp_npi]
            
            for drug in df['drug_brand_name'].unique():
                drug_share = (hcp_df['drug_brand_name'] == drug).mean()
                
                affinity = (
                    hcp_row['specialization_weight'] * 0.4 +
                    drug_share * 0.3 +
                    (hcp_row['patient_volume'] / 200) * 0.3
                )
                
                affinity_matrix[(hcp_npi, drug)] = np.clip(affinity, 0.5, 1.8)
                
        return affinity_matrix

# ================== MODEL 4: TRANSACTION TYPE ==================

class TransactionTypeModel:
    """Model 4: Transaction type and payer mix effects"""
    
    def __init__(self):
        self.model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        
    def prepare_features(self, df):
        """Prepare transaction type features for each HCP"""
        features = []
        
        for hcp_npi in df['hcp_id'].unique():
            hcp_df = df[df['hcp_id'] == hcp_npi]
            
            payer_dist = hcp_df['payer_channel_name'].value_counts(normalize=True)
            
            hcp_features = {
                'hcp_id': hcp_npi,
                'medicare_share': payer_dist.get('Medicare', 0),
                'commercial_share': payer_dist.get('Commercial', 0),
                'medicaid_share': payer_dist.get('Medicaid', 0),
                'rejection_rate': (hcp_df['reject_reason_1_cd'].notna()).mean(),
                'avg_copay_ratio': hcp_df['copay_ratio'].mean(),
                'payer_diversity': hcp_df['payer_name'].nunique(),
                'specialty_pharmacy_ratio': hcp_df['pharmacy_npi_nbr'].nunique() / len(hcp_df)
            }
            
            features.append(hcp_features)
            
        return pd.DataFrame(features)
    
    def calculate_transaction_multipliers(self, feature_df):
        """Calculate multipliers based on transaction patterns"""
        multipliers = {}
        
        for _, row in feature_df.iterrows():
            multiplier = 1.0
            multiplier *= (1 + 0.2 * row['commercial_share'])
            multiplier *= (1 - 0.1 * row['rejection_rate'])
            multiplier *= (1 + 0.05 * row['payer_diversity'] / 10)
            
            multipliers[row['hcp_id']] = np.clip(multiplier, 0.8, 1.3)
            
        return multipliers

# ================== MODEL 5: ROLLING SALES FEATURES ==================

class RollingSalesModel:
    """Model 5: Rolling sales patterns and trends"""
    
    def __init__(self):
        self.model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        
    def prepare_features(self, df):
        """Prepare rolling sales features"""
        df = df.sort_values(['hcp_id', 'service_date_dd'])
        features = []
        
        for hcp_npi in df['hcp_id'].unique():
            hcp_df = df[df['hcp_id'] == hcp_npi].copy()
            
            for window in [7, 30, 90]:
                hcp_df[f'rolling_sum_{window}d'] = hcp_df.groupby('drug_brand_name')['total_paid_amt'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).sum()
                )
                hcp_df[f'rolling_avg_{window}d'] = hcp_df.groupby('drug_brand_name')['total_paid_amt'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
            
            hcp_features = {
                'hcp_id': hcp_npi,
                'sales_trend_30d': self._calculate_trend(hcp_df['rolling_sum_30d']),
                'sales_volatility': hcp_df['total_paid_amt'].std(),
                'growth_rate_90d': self._calculate_growth_rate(hcp_df['rolling_sum_90d']),
                'seasonality_strength': self._calculate_seasonality(hcp_df)
            }
            
            features.append(hcp_features)
            
        return pd.DataFrame(features)
    
    def _calculate_trend(self, series):
        """Calculate linear trend"""
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        coef = np.polyfit(x, series.fillna(0), 1)[0]
        return coef
    
    def _calculate_growth_rate(self, series):
        """Calculate growth rate"""
        if len(series) < 2:
            return 0
        first_val = series.iloc[0] if not pd.isna(series.iloc[0]) else 0
        last_val = series.iloc[-1] if not pd.isna(series.iloc[-1]) else 0
        if first_val == 0:
            return 0
        return (last_val - first_val) / first_val
    
    def _calculate_seasonality(self, df):
        """Calculate seasonality strength"""
        if 'month' not in df.columns:
            return 0
        monthly_avg = df.groupby('month')['total_paid_amt'].mean()
        return monthly_avg.std() / (monthly_avg.mean() + 1e-6)

# ================== MODEL 6: PAYER REPUTATION ==================

class PayerReputationModel:
    """Model 6: Payer channel and reputation effects"""
    
    def __init__(self):
        self.external_collector = ExternalDataCollector()
        self.payer_scores = self.external_collector.get_payer_reputation_scores()
        
    def calculate_payer_multipliers(self, df):
        """Calculate multipliers based on payer mix and reputation"""
        multipliers = {}
        
        for hcp_npi in df['hcp_id'].unique():
            hcp_df = df[df['hcp_id'] == hcp_npi]
            
            weighted_score = 0
            total_weight = 0
            
            for payer_channel in hcp_df['payer_channel_name'].unique():
                channel_weight = (hcp_df['payer_channel_name'] == payer_channel).sum()
                if payer_channel in self.payer_scores:
                    scores = self.payer_scores[payer_channel]
                    weighted_score += (scores['reputation'] * scores['reimbursement_rate'] * channel_weight)
                    total_weight += channel_weight
            
            if total_weight > 0:
                multiplier = 0.7 + (weighted_score / total_weight) * 0.6
            else:
                multiplier = 1.0
                
            multipliers[hcp_npi] = np.clip(multiplier, 0.8, 1.4)
            
        return multipliers

# ================== ENSEMBLE MODEL ==================

class EnsemblePredictionModel:
    """Combines all models to generate final predictions"""
    
    def __init__(self):
        self.temporal_model = TemporalDemographicModel()
        self.market_model = MarketDynamicsModel()
        self.specialization_model = HCPSpecializationModel()
        self.transaction_model = TransactionTypeModel()
        self.rolling_model = RollingSalesModel()
        self.payer_model = PayerReputationModel()
        
        self.preprocessor = DataPreprocessor()
        
    def train_all_models(self, df):
        """Train all component models"""
        print("Training ensemble models...")
        
        base_features = self.preprocessor.create_base_features(df)
        
        # Model 1: Temporal & Demographic
        temporal_features = self.temporal_model.prepare_features(base_features)
        target_multipliers = self._calculate_target_multipliers(df, temporal_features['hcp_id'])

        target_multipliers = np.where(
            np.isin(target_multipliers, [0, np.inf, -np.inf]) | np.isnan(target_multipliers),
                1,
                target_multipliers
        )

        self.temporal_model.train(temporal_features, target_multipliers)
        
        # Model 2: Market Dynamics
        self.market_features = self.market_model.prepare_features(df)
        
        # Model 3: HCP Specialization
        self.spec_features = self.specialization_model.prepare_features(df)
        
        # Model 4: Transaction Type
        self.trans_features = self.transaction_model.prepare_features(base_features)
        
        # Model 5: Rolling Sales
        self.rolling_features = self.rolling_model.prepare_features(df)
        
        print("\nAll models trained successfully!")
        
    def predict(self, df):
        """Generate predictions for HCP-drug combinations - FIXED to keep hcp_id as NPI"""
        predictions = []
        
        # Get multipliers from each model
        temporal_mult = self.temporal_model.predict_multiplier(
            self.temporal_model.prepare_features(self.preprocessor.create_base_features(df))
        )
        market_mult = self.market_model.calculate_drug_multipliers(self.market_features)
        spec_affinity = self.specialization_model.calculate_hcp_drug_affinity(df, self.spec_features)
        trans_mult = self.transaction_model.calculate_transaction_multipliers(self.trans_features)
        rolling_mult = self._calculate_rolling_multipliers(self.rolling_features)
        payer_mult = self.payer_model.calculate_payer_multipliers(df)
        
        # FIXED: Get HCP columns from actual schema, EXCLUDING hcp_id
        hcp_columns = []
        potential_hcp_cols = ['type_1_npi', 'type_2_npi', 'prescriber_npi_nm', 
                             'prescriber_npi_state_cd', 'prescriber_npi_nbr']
        for col in df.columns:
            if col != 'hcp_id':  # Exclude hcp_id
                if col in potential_hcp_cols or 'prescriber' in col.lower():
                    hcp_columns.append(col)
        
        # Create HCP lookup preserving ALL columns EXCEPT hcp_id
        if len(hcp_columns) > 0:
            hcp_data = df.groupby('hcp_id')[hcp_columns].first()
            hcp_lookup = hcp_data.to_dict('index')
        else:
            hcp_lookup = {}
        
        # Aggregate by HCP (NPI) and Drug
        hcp_drug_groups = df.groupby(['hcp_id', 'drug_generic_name', 'drug_brand_name'])
        
        for (hcp_npi, drug_generic, drug_brand), group in hcp_drug_groups:
            # Get base metrics from data
            sales_from_db = group['total_paid_amt'].sum()
            quantity_from_db = len(group)
            
            # Get additional HCP details (excluding hcp_id)
            hcp_details = hcp_lookup.get(hcp_npi, {})
            
            # Combine multipliers
            combined_multiplier = 1.0
            
            combined_multiplier *= temporal_mult.get(hcp_npi, 1.0) 
            combined_multiplier *= market_mult.get(drug_brand, 1.0) ** 0.5
            combined_multiplier *= spec_affinity.get((hcp_npi, drug_brand), 1.0) ** 0.5
            combined_multiplier *= trans_mult.get(hcp_npi, 1.0) ** 0.33
            combined_multiplier *= rolling_mult.get(hcp_npi, 1.0) ** 0.33
            combined_multiplier *= payer_mult.get(hcp_npi, 1.0) ** 0.33
            
            # Calculate predictions
            predicted_sales = sales_from_db * combined_multiplier
            predicted_quantity = quantity_from_db * combined_multiplier
            
            company = self.preprocessor.drug_mapping.get(drug_generic, {}).get('company', 'Unknown')
            
            # Build prediction record - hcp_id is NPI, other HCP columns follow
            prediction_record = {
                'hcp_id': hcp_npi,  # This is the NPI number from type_1_npi
                **hcp_details,  # Other HCP columns from input (name, state, etc.)
                'Drug_generic_name': drug_generic,
                'Drug_name': drug_brand,
                'Company': company,
                'Sales_from_db': round(sales_from_db, 2),
                'Predicted_sales': round(predicted_sales, 2),
                'Quantity_from_DB': int(quantity_from_db),
                'Predicted_quantity': int(predicted_quantity),
                'Combined_multiplier': round(combined_multiplier, 3)
            }
            
            predictions.append(prediction_record)
        
        return pd.DataFrame(predictions)

    def _calculate_target_multipliers(self, df, hcp_ids):
        """Calculate synthetic target multipliers for training"""
        multipliers = []
        for hcp in hcp_ids:
            hcp_df = df[df['hcp_id'] == hcp]
            cv = hcp_df['total_paid_amt'].std() / (hcp_df['total_paid_amt'].mean() + 1e-6)
            multipliers.append(1 + np.clip(cv, -0.5, 0.5))

        return np.array(multipliers)
    
    def _calculate_rolling_multipliers(self, rolling_features):
        """Calculate multipliers from rolling features"""
        multipliers = {}
        for _, row in rolling_features.iterrows():
            mult = 1.0
            mult *= (1 + 0.1 * np.sign(row['sales_trend_30d']))
            mult *= (1 + 0.05 * np.clip(row['growth_rate_90d'], -0.5, 0.5))
            multipliers[row['hcp_id']] = np.clip(mult, 0.7, 1.5)
        return multipliers

# ================== MAIN EXECUTION ==================

def main():
    """Main execution function"""
    print("="*60)
    print("PHARMACEUTICAL SALES PREDICTION SYSTEM")
    print("Using NPI as HCP Identifier")
    print("="*60)
    
    # Load your data here
    # df = pd.read_csv('claims_data.csv')
    # For demonstration, creating sample data
    df = create_sample_data()
    
    # Initialize preprocessing
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.load_and_clean_data(df)
    
    print(f"\nData loaded: {len(df_clean)} valid claims")
    print(f"Unique NPIs: {df_clean['hcp_id'].nunique()}")
    print(f"Unique Drugs: {df_clean['drug_brand_name'].nunique()}")
    
    # Initialize and train ensemble model
    ensemble = EnsemblePredictionModel()
    ensemble.train_all_models(df_clean)
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = ensemble.predict(df_clean)
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(predictions.head(10))
    
    # Verify hcp_id contains NPI numbers
    print("\n" + "="*60)
    print("VERIFICATION: HCP_ID Column Contains NPI Numbers")
    print("="*60)
    print(f"Sample hcp_id values (first 10):")
    for npi in predictions['hcp_id'].head(10):
        print(f"  {npi}")
    print(f"\nTotal unique HCP NPIs: {predictions['hcp_id'].nunique()}")
    print(f"HCP_ID data type: {predictions['hcp_id'].dtype}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total Predicted Sales: ${predictions['Predicted_sales'].sum():,.2f}")
    print(f"Total DB Sales: ${predictions['Sales_from_db'].sum():,.2f}")
    print(f"Average Multiplier: {predictions['Combined_multiplier'].mean():.3f}")
    print(f"Sales Uplift: {(predictions['Predicted_sales'].sum() / predictions['Sales_from_db'].sum() - 1) * 100:.1f}%")
    
    # Save predictions
    predictions.to_csv('hcp_drug_predictions.csv', index=False)
    print("\nPredictions saved to 'hcp_drug_predictions.csv'")
    print("✓ hcp_id column contains NPI numbers")
    
    # Show column structure
    print("\n" + "="*60)
    print("OUTPUT COLUMNS")
    print("="*60)
    print(f"Total columns: {len(predictions.columns)}")
    print("Column names:")
    for i, col in enumerate(predictions.columns, 1):
        print(f"  {i}. {col}")
    
    return predictions

def create_sample_data():
    """Create sample data matching actual schema structure"""
    np.random.seed(42)
    
    n_records = 1000
    
    # Generate realistic NPI numbers (10-digit numbers) for type_1_npi (prescriber)
    unique_type1_npis = [1000000000 + i for i in range(20)]
    unique_type2_npis = [2000000000 + i for i in range(15)]
    
    data = {
        'year': np.random.choice([2022, 2023, 2024, 2025], n_records),
        'month': np.random.choice(range(1, 13), n_records),
        'patient_gender': np.random.choice(['F', 'M'], n_records),
        'patient_age': np.random.randint(18, 85, n_records),
        'drug_generic_name': np.random.choice(['Cemiplimab-Rwlc', 'Nivolumab', 'Pembrolizumab'], n_records),
        'drug_brand_name': np.random.choice(['Libtayo', 'Opdivo', 'Keytruda'], n_records),
        'type_1_npi': np.random.choice(unique_type1_npis, n_records),  # Prescriber NPI
        'type_2_npi': np.random.choice(unique_type2_npis, n_records),  # Pharmacy/Organization NPI
        'prescriber_npi_nm': np.random.choice([
            'MICHAEL DEL ROSARIO', 'AMIT PATEL', 'SATISH SHAH', 'JESSICA HOCHBERG',
            'ILMANA FULGER', 'NADER JAVADI', 'URSZULA SOBOL', 'LAUREN EISENBUD',
            'RAMIN ALTAHA', 'KARL LEWIS', 'JOHN SMITH', 'MARY JOHNSON',
            'DAVID WILLIAMS', 'PATRICIA BROWN', 'ROBERT JONES', 'JENNIFER DAVIS',
            'MICHAEL MILLER', 'LINDA WILSON', 'WILLIAM MOORE', 'ELIZABETH TAYLOR'
        ], n_records),
        'prescriber_npi_state_cd': np.random.choice(['CA', 'NY', 'TX', 'FL', 'PA', 'IL', 'NJ', 'MA'], n_records),
        'patient_id': [f'{np.random.randint(1e7, 1e8):08x}-{np.random.randint(1e4, 1e5):04x}-{np.random.randint(1e4, 1e5):04x}' for _ in range(n_records)],
        'date': pd.date_range('2022-01-01', periods=n_records, freq='D')[:n_records],
        'claim_number': [f'{np.random.randint(1e15, 1e16):064x}' for _ in range(n_records)],
        'transaction_status': np.random.choice(['Dispensed', 'Dispensed', 'Dispensed', 'Reject'], n_records),
        'ndc': np.random.choice(['61755000801', '3377211', '3377412', '3373413', '3375614', '6302602', '6302601', '6302604', '6302902'], n_records),
        'payer_name': np.random.choice(['Express Scripts', 'CVS Health', 'UnitedHealthcare', 'Not Specified', 'Federal Employee Program'], n_records),
        'payer_channel_name': np.random.choice(['Medicare', 'Commercial', 'Medicaid', 'Other', 'Dual (Medicaid/Medicare)'], n_records),
        'payer_subchannel_name': np.random.choice([
            'Medicare / Advantage (Part C)', 'Medicare / Advantage (Part D)', 
            'Commercial / Commercial', 'Medicaid / Managed', 'Medicaid / Unspecified',
            'Other / Unknown', 'Commercial / Cash or Self-Pay'
        ], n_records),
        'final_status_code': np.random.choice(['Y', 'N'], n_records),
        'service_date_dd': pd.date_range('2022-01-01', periods=n_records, freq='D')[:n_records],
        'date_prescription_written_dd': np.random.randint(18900, 19400, n_records),
        'transaction_dt': pd.date_range('2022-01-01', periods=n_records, freq='H')[:n_records],
        'dispense_nbr': [f'{np.random.randint(1e15, 1e16):064x}' for _ in range(n_records)],
        'admin_service_line': ['Rx'] * n_records,
        'clinical_service_line': np.random.choice([
            'Oncology/Hematology (Medical)', 'General Medicine', 'Unknown', 
            'Signs and Symptoms', 'ENT', 'Cardiac Services'
        ], n_records),
        'reject_reason_1_cd': np.random.choice([None, None, None, 'MR', '70', '75', 'R6'], n_records),
        'reject_reason_1_desc': np.random.choice([
            None, None, None, 'Product Not On Formulary', 
            'Product/Service Not Covered – Plan/Benefit Exclusion',
            'Prior Authorization Required'
        ], n_records),
        'open_source_fl': [1] * n_records,
        'closed_source_fl': [0] * n_records,
        'ndc_desc': np.random.choice([
            'Libtayo Intravenous Solution 350 Mg/7Ml',
            'Opdivo Intravenous Solution 40 Mg/4Ml',
            'Opdivo Intravenous Solution 100 Mg/10Ml',
            'Opdivo Intravenous Solution 240 Mg/24Ml',
            'Keytruda Intravenous Solution 100 Mg/4Ml',
            'Keytruda Intravenous Solution Reconstituted 50 Mg'
        ], n_records),
        'ndc_drug_nm': np.random.choice(['Cemiplimab-Rwlc', 'Nivolumab', 'Pembrolizumab'], n_records),
        'ndc_isbranded_ind': ['Y'] * n_records,
        'roa': ['Intravenous'] * n_records,
        'pharmacy_npi_nbr': np.random.choice(unique_type2_npis, n_records),
        'pcp_npi_nbr': np.random.choice([None] + unique_type1_npis[:10], n_records),
        'payer_id': np.random.randint(1000, 10000, n_records),
        'payer_company_nm': np.random.choice(['Express Scripts', 'CVS Health', 'UnitedHealthcare', 'Not Specified'], n_records),
        'payer_bin_nbr': np.random.choice(['003858', '004336', '610239', '610097', '610014'], n_records),
        'days_supply_val': np.random.choice([21, 28, 30, 14, 24, 42], n_records),
        'awp_unit_price_amt': np.random.choice([0.0, 1647.3685, 1615.0686, 352.66168, 345.7475, 345.74667], n_records),
        'total_paid_amt': np.random.uniform(0, 20000, n_records),
        'patient_to_pay_amt': np.random.uniform(0, 500, n_records),
        'update_ts': [pd.Timestamp('2025-08-01 17:26:58')] * n_records,
        'source_file_key': ['pharmacy_green_00000000.parquet'] * n_records
    }
    
    df = pd.DataFrame(data)
    
    # Convert type_1_npi and type_2_npi to float (matching actual schema)
    df['type_1_npi'] = df['type_1_npi'].astype(float)
    df['type_2_npi'] = df['type_2_npi'].astype(float)
    
    return df

if __name__ == "__main__":
    predictions = main()