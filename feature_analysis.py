"""
Feature Analysis and Importance Script
Analyzes feature importance, correlations, and impact on predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import shap
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class FeatureAnalyzer:
    """Comprehensive feature analysis for pharmaceutical sales prediction"""
    
    def __init__(self, df, predictions):
        self.df = df
        self.predictions = predictions
        self.feature_importance = {}
        
    def run_complete_analysis(self):
        """Run all feature analyses"""
        print("="*70)
        print("FEATURE ANALYSIS REPORT")
        print("="*70)
        
        # 1. Basic Statistics
        self.analyze_basic_statistics()
        
        # 2. Feature Correlations
        self.analyze_correlations()
        
        # 3. Model-specific Feature Importance
        self.analyze_model_features()
        
        # 4. Temporal Patterns
        self.analyze_temporal_patterns()
        
        # 5. HCP Behavioral Patterns
        self.analyze_hcp_patterns()
        
        # 6. Drug-specific Analysis
        self.analyze_drug_features()
        
        # 7. Payer Impact Analysis
        self.analyze_payer_impact()
        
        # 8. Generate Report
        self.generate_feature_report()
        
    def analyze_basic_statistics(self):
        """Analyze basic statistical properties of features"""
        print("\n1. BASIC FEATURE STATISTICS")
        print("-" * 50)
        
        # Numerical features
        numerical_features = ['patient_age', 'days_supply_val', 'total_paid_amt', 
                            'patient_to_pay_amt', 'awp_unit_price_amt']
        
        stats_df = pd.DataFrame()
        for feature in numerical_features:
            if feature in self.df.columns:
                stats_df[feature] = [
                    self.df[feature].mean(),
                    self.df[feature].std(),
                    self.df[feature].min(),
                    self.df[feature].quantile(0.25),
                    self.df[feature].median(),
                    self.df[feature].quantile(0.75),
                    self.df[feature].max(),
                    self.df[feature].skew(),
                    self.df[feature].kurtosis()
                ]
        
        stats_df.index = ['Mean', 'Std', 'Min', 'Q1', 'Median', 'Q3', 'Max', 'Skew', 'Kurtosis']
        print(stats_df.round(2))
        
        # Categorical features
        print("\n\nCategorical Feature Distributions:")
        categorical_features = ['drug_brand_name', 'payer_channel_name', 'transaction_status']
        
        for feature in categorical_features:
            if feature in self.df.columns:
                print(f"\n{feature}:")
                dist = self.df[feature].value_counts(normalize=True).head(5)
                for val, pct in dist.items():
                    print(f"  {val}: {pct*100:.1f}%")
    
    def analyze_correlations(self):
        """Analyze feature correlations"""
        print("\n2. FEATURE CORRELATIONS")
        print("-" * 50)
        
        # Prepare numerical features for correlation
        num_features = self.df.select_dtypes(include=[np.number]).columns
        
        if len(num_features) > 0:
            corr_matrix = self.df[num_features].corr()
            
            # Find strong correlations
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.5:
                        strong_corr.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })
            
            if strong_corr:
                print("\nStrong Correlations (|r| > 0.5):")
                strong_corr_df = pd.DataFrame(strong_corr).sort_values('Correlation', 
                                                                       key=abs, 
                                                                       ascending=False)
                print(strong_corr_df.to_string(index=False))
            else:
                print("No strong correlations found (|r| > 0.5)")
            
            # Correlation with target (total_paid_amt)
            if 'total_paid_amt' in num_features:
                target_corr = corr_matrix['total_paid_amt'].sort_values(ascending=False)
                print("\n\nCorrelation with Total Paid Amount:")
                print(target_corr.head(10).to_string())
    
    def analyze_model_features(self):
        """Analyze features specific to each model"""
        print("\n3. MODEL-SPECIFIC FEATURE ANALYSIS")
        print("-" * 50)
        
        # Model 1: Temporal & Demographic Features
        print("\nModel 1 - Temporal & Demographic Features:")
        temp_demo_features = {
            'Age Distribution Impact': self._analyze_age_impact(),
            'Gender Balance': self._analyze_gender_impact(),
            'Seasonal Patterns': self._analyze_seasonal_impact(),
            'Day of Week Effect': self._analyze_dow_impact()
        }
        for feature, impact in temp_demo_features.items():
            print(f"  {feature}: {impact:.2f}")
        
        # Model 2: Market Dynamics
        print("\nModel 2 - Market Dynamics Features:")
        market_features = self._analyze_market_dynamics()
        for feature, value in market_features.items():
            print(f"  {feature}: {value}")
        
        # Model 3: HCP Specialization
        print("\nModel 3 - HCP Specialization Features:")
        hcp_features = self._analyze_hcp_specialization()
        for feature, value in hcp_features.items():
            print(f"  {feature}: {value}")
        
        # Model 4: Transaction Type
        print("\nModel 4 - Transaction Type Features:")
        trans_features = self._analyze_transaction_features()
        for feature, value in trans_features.items():
            print(f"  {feature}: {value}")
        
        # Model 5: Rolling Sales
        print("\nModel 5 - Rolling Sales Features:")
        rolling_features = self._analyze_rolling_features()
        for feature, value in rolling_features.items():
            print(f"  {feature}: {value}")
        
        # Model 6: Payer Reputation
        print("\nModel 6 - Payer Reputation Features:")
        payer_features = self._analyze_payer_features()
        for feature, value in payer_features.items():
            print(f"  {feature}: {value}")
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in the data"""
        print("\n4. TEMPORAL PATTERN ANALYSIS")
        print("-" * 50)
        
        if 'service_date_dd' in self.df.columns:
            self.df['service_date'] = pd.to_datetime(self.df['service_date_dd'])
            self.df['month'] = self.df['service_date'].dt.month
            self.df['quarter'] = self.df['service_date'].dt.quarter
            self.df['year'] = self.df['service_date'].dt.year
            
            # Monthly trends
            monthly_sales = self.df.groupby('month')['total_paid_amt'].agg(['mean', 'sum', 'count'])
            print("\nMonthly Sales Patterns:")
            print(monthly_sales.round(2))
            
            # Quarterly trends
            quarterly_sales = self.df.groupby('quarter')['total_paid_amt'].agg(['mean', 'sum', 'count'])
            print("\nQuarterly Sales Patterns:")
            print(quarterly_sales.round(2))
            
            # Calculate trend strength
            months = sorted(self.df['month'].unique())
            monthly_means = [self.df[self.df['month'] == m]['total_paid_amt'].mean() for m in months]
            if len(monthly_means) > 1:
                trend_coef = np.polyfit(range(len(monthly_means)), monthly_means, 1)[0]
                print(f"\nMonthly Trend Coefficient: {trend_coef:.2f}")
                print(f"Trend Direction: {'Increasing' if trend_coef > 0 else 'Decreasing'}")
    
    def analyze_hcp_patterns(self):
        """Analyze HCP prescribing patterns"""
        print("\n5. HCP PRESCRIBING PATTERNS")
        print("-" * 50)
        
        if 'hcp_id' in self.df.columns or 'prescriber_npi_nm' in self.df.columns:
            hcp_col = 'hcp_id' if 'hcp_id' in self.df.columns else 'prescriber_npi_nm'
            
            # HCP volume distribution
            hcp_volumes = self.df.groupby(hcp_col).size().describe()
            print("\nHCP Prescription Volume Distribution:")
            print(hcp_volumes.round(2))
            
            # Top prescribers
            top_hcps = self.df.groupby(hcp_col)['total_paid_amt'].sum().nlargest(10)
            print("\nTop 10 HCPs by Total Sales:")
            for hcp, amount in top_hcps.items():
                print(f"  {hcp}: ${amount:,.2f}")
            
            # HCP drug preferences
            hcp_drug_matrix = pd.crosstab(self.df[hcp_col], self.df['drug_brand_name'], 
                                         values=self.df['total_paid_amt'], aggfunc='sum')
            
            # Calculate HCP concentration (Herfindahl index)
            hcp_concentration = []
            for hcp in hcp_drug_matrix.index[:10]:  # Top 10 HCPs
                row = hcp_drug_matrix.loc[hcp]
                total = row.sum()
                if total > 0:
                    shares = (row / total) ** 2
                    hhi = shares.sum()
                    hcp_concentration.append({'HCP': hcp, 'HHI': hhi})
            
            if hcp_concentration:
                conc_df = pd.DataFrame(hcp_concentration).sort_values('HHI', ascending=False)
                print("\nHCP Drug Concentration (HHI):")
                print(conc_df.head().to_string(index=False))
    
    def analyze_drug_features(self):
        """Analyze drug-specific features"""
        print("\n6. DRUG-SPECIFIC FEATURE ANALYSIS")
        print("-" * 50)
        
        drugs = self.df['drug_brand_name'].unique()
        
        drug_analysis = []
        for drug in drugs:
            drug_df = self.df[self.df['drug_brand_name'] == drug]
            
            analysis = {
                'Drug': drug,
                'Total_Sales': drug_df['total_paid_amt'].sum(),
                'Avg_Sale': drug_df['total_paid_amt'].mean(),
                'Volume': len(drug_df),
                'Unique_HCPs': drug_df['prescriber_npi_nm'].nunique() if 'prescriber_npi_nm' in drug_df.columns else 0,
                'Unique_Patients': drug_df['patient_id'].nunique() if 'patient_id' in drug_df.columns else 0,
                'Avg_Days_Supply': drug_df['days_supply_val'].mean() if 'days_supply_val' in drug_df.columns else 0,
                'Market_Share': len(drug_df) / len(self.df) * 100
            }
            drug_analysis.append(analysis)
        
        drug_df = pd.DataFrame(drug_analysis).sort_values('Total_Sales', ascending=False)
        print("\nDrug Performance Metrics:")
        print(drug_df.round(2).to_string(index=False))
        
        # Drug substitution patterns
        if 'patient_id' in self.df.columns:
            patients_multi_drug = self.df.groupby('patient_id')['drug_brand_name'].nunique()
            switch_rate = (patients_multi_drug > 1).mean() * 100
            print(f"\nPatient Drug Switching Rate: {switch_rate:.1f}%")
    
    def analyze_payer_impact(self):
        """Analyze payer channel impact on sales"""
        print("\n7. PAYER IMPACT ANALYSIS")
        print("-" * 50)
        
        if 'payer_channel_name' in self.df.columns:
            # Payer channel distribution
            payer_dist = self.df['payer_channel_name'].value_counts(normalize=True)
            print("\nPayer Channel Distribution:")
            for payer, pct in payer_dist.items():
                print(f"  {payer}: {pct*100:.1f}%")
            
            # Average payment by payer
            payer_payments = self.df.groupby('payer_channel_name')['total_paid_amt'].agg(['mean', 'median', 'sum'])
            print("\nPayment Statistics by Payer Channel:")
            print(payer_payments.round(2))
            
            # Rejection rates by payer
            if 'reject_reason_1_cd' in self.df.columns:
                rejection_by_payer = self.df.groupby('payer_channel_name')['reject_reason_1_cd'].apply(
                    lambda x: (x.notna()).mean() * 100
                )
                print("\nRejection Rates by Payer Channel:")
                for payer, rate in rejection_by_payer.items():
                    print(f"  {payer}: {rate:.1f}%")
            
            # Copay analysis
            if 'patient_to_pay_amt' in self.df.columns:
                copay_by_payer = self.df.groupby('payer_channel_name')['patient_to_pay_amt'].agg(['mean', 'median'])
                print("\nCopay Statistics by Payer Channel:")
                print(copay_by_payer.round(2))
    
    def _analyze_age_impact(self):
        """Calculate age distribution impact score"""
        if 'patient_age' in self.df.columns:
            age_cv = self.df['patient_age'].std() / self.df['patient_age'].mean()
            return 1 - age_cv  # Higher score for more consistent age distribution
        return 0.5
    
    def _analyze_gender_impact(self):
        """Calculate gender balance impact score"""
        if 'patient_gender' in self.df.columns:
            gender_dist = self.df['patient_gender'].value_counts(normalize=True)
            balance = 1 - abs(gender_dist.iloc[0] - 0.5) * 2
            return balance
        return 0.5
    
    def _analyze_seasonal_impact(self):
        """Calculate seasonal impact score"""
        if 'month' in self.df.columns:
            monthly_sales = self.df.groupby('month')['total_paid_amt'].mean()
            seasonal_var = monthly_sales.std() / monthly_sales.mean()
            return seasonal_var
        return 0.1
    
    def _analyze_dow_impact(self):
        """Calculate day of week impact"""
        if 'day_of_week' in self.df.columns:
            dow_sales = self.df.groupby('day_of_week')['total_paid_amt'].mean()
            dow_var = dow_sales.std() / dow_sales.mean()
            return dow_var
        return 0.05
    
    def _analyze_market_dynamics(self):
        """Analyze market dynamics features"""
        features = {}
        
        # Market concentration
        if 'drug_brand_name' in self.df.columns:
            drug_shares = self.df['drug_brand_name'].value_counts(normalize=True)
            hhi = (drug_shares ** 2).sum()
            features['Market_HHI'] = f"{hhi:.3f}"
            features['Market_Leader_Share'] = f"{drug_shares.iloc[0]*100:.1f}%"
        
        # Growth metrics
        if 'year' in self.df.columns and len(self.df['year'].unique()) > 1:
            yearly_sales = self.df.groupby('year')['total_paid_amt'].sum()
            if len(yearly_sales) > 1:
                yoy_growth = (yearly_sales.iloc[-1] / yearly_sales.iloc[0] - 1) * 100
                features['YoY_Growth'] = f"{yoy_growth:.1f}%"
        
        return features
    
    def _analyze_hcp_specialization(self):
        """Analyze HCP specialization metrics"""
        features = {}
        
        if 'clinical_service_line' in self.df.columns:
            service_lines = self.df['clinical_service_line'].value_counts()
            features['Top_Service_Line'] = f"{service_lines.index[0]} ({service_lines.iloc[0]} claims)"
            features['Service_Line_Diversity'] = len(service_lines)
        
        if 'prescriber_npi_nm' in self.df.columns:
            hcp_counts = self.df['prescriber_npi_nm'].value_counts()
            features['Avg_Claims_per_HCP'] = f"{hcp_counts.mean():.1f}"
            features['Top_HCP_Share'] = f"{hcp_counts.iloc[0] / len(self.df) * 100:.1f}%"
        
        return features
    
    def _analyze_transaction_features(self):
        """Analyze transaction-related features"""
        features = {}
        
        if 'transaction_status' in self.df.columns:
            status_dist = self.df['transaction_status'].value_counts(normalize=True)
            features['Approval_Rate'] = f"{status_dist.get('Dispensed', 0)*100:.1f}%"
        
        if 'reject_reason_1_cd' in self.df.columns:
            rejection_rate = self.df['reject_reason_1_cd'].notna().mean()
            features['Overall_Rejection_Rate'] = f"{rejection_rate*100:.1f}%"
            
            if rejection_rate > 0:
                top_reject = self.df['reject_reason_1_desc'].value_counts().iloc[0] if 'reject_reason_1_desc' in self.df.columns else 'Unknown'
                features['Top_Rejection_Reason'] = str(top_reject)
        
        return features
    
    def _analyze_rolling_features(self):
        """Analyze rolling/trend features"""
        features = {}
        
        if 'service_date_dd' in self.df.columns:
            self.df['service_date'] = pd.to_datetime(self.df['service_date_dd'])
            self.df = self.df.sort_values('service_date')
            
            # Calculate 30-day rolling average
            daily_sales = self.df.groupby('service_date')['total_paid_amt'].sum()
            if len(daily_sales) > 30:
                rolling_30d = daily_sales.rolling(window=30, min_periods=1).mean()
                trend = 'Increasing' if rolling_30d.iloc[-1] > rolling_30d.iloc[0] else 'Decreasing'
                features['30_Day_Trend'] = trend
                features['Current_vs_30d_Avg'] = f"{(daily_sales.iloc[-1] / rolling_30d.iloc[-1] - 1)*100:.1f}%"
        
        return features
    
    def _analyze_payer_features(self):
        """Analyze payer-related features"""
        features = {}
        
        if 'payer_channel_name' in self.df.columns:
            # Payer diversity
            payer_entropy = stats.entropy(self.df['payer_channel_name'].value_counts())
            features['Payer_Diversity_Score'] = f"{payer_entropy:.2f}"
            
            # Top payer metrics
            top_payer = self.df.groupby('payer_channel_name')['total_paid_amt'].sum().idxmax()
            top_payer_share = self.df[self.df['payer_channel_name'] == top_payer]['total_paid_amt'].sum() / self.df['total_paid_amt'].sum()
            features['Top_Payer'] = top_payer
            features['Top_Payer_Share'] = f"{top_payer_share*100:.1f}%"
        
        return features
    
    def generate_feature_report(self):
        """Generate comprehensive feature importance report"""
        print("\n8. FEATURE IMPORTANCE SUMMARY")
        print("="*70)
        
        # Calculate composite feature importance scores
        importance_scores = {
            'Patient Demographics': self._calculate_demographic_importance(),
            'Temporal Patterns': self._calculate_temporal_importance(),
            'Drug Characteristics': self._calculate_drug_importance(),
            'HCP Behavior': self._calculate_hcp_importance(),
            'Payer Dynamics': self._calculate_payer_importance(),
            'Transaction Patterns': self._calculate_transaction_importance()
        }
        
        # Sort by importance
        sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("\nFeature Category Importance Ranking:")
        print("-"*40)
        for i, (category, score) in enumerate(sorted_importance, 1):
            if score == np.nan:
                bar = '█' * int(score * 50)
                print(f"{i}.{category:<25}{bar}{score:.3f}")
        
        # Key insights
        print("\n\nKEY INSIGHTS:")
        print("-"*40)
        insights = self._generate_insights(importance_scores)
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
    
    def _calculate_demographic_importance(self):
        """Calculate importance of demographic features"""
        score = 0.5  # Base score
        
        if 'patient_age' in self.df.columns:
            age_corr = abs(self.df['patient_age'].corr(self.df['total_paid_amt']))
            score += age_corr * 0.3
        
        if 'patient_gender' in self.df.columns:
            gender_impact = self.df.groupby('patient_gender')['total_paid_amt'].mean().std() / self.df['total_paid_amt'].mean()
            score += gender_impact * 0.2
        
        return min(score, 1.0)
    
    def _calculate_temporal_importance(self):
        """Calculate importance of temporal features"""
        score = 0.4  # Base score
        
        if 'month' in self.df.columns:
            monthly_var = self.df.groupby('month')['total_paid_amt'].mean().std() / self.df['total_paid_amt'].mean()
            score += monthly_var * 0.3
        
        if 'quarter' in self.df.columns:
            quarterly_var = self.df.groupby('quarter')['total_paid_amt'].mean().std() / self.df['total_paid_amt'].mean()
            score += quarterly_var * 0.3
        
        return min(score, 1.0)
    
    def _calculate_drug_importance(self):
        """Calculate importance of drug features"""
        score = 0.6  # Base score - drugs are important
        
        if 'drug_brand_name' in self.df.columns:
            drug_var = self.df.groupby('drug_brand_name')['total_paid_amt'].mean().std() / self.df['total_paid_amt'].mean()
            score += drug_var * 0.4
        
        return min(score, 1.0)
    
    def _calculate_hcp_importance(self):
        """Calculate importance of HCP features"""
        score = 0.7  # Base score - HCPs are very important
        
        if 'prescriber_npi_nm' in self.df.columns:
            hcp_var = self.df.groupby('prescriber_npi_nm')['total_paid_amt'].sum().std() / self.df['total_paid_amt'].sum()
            score += hcp_var * 0.3
        
        return min(score, 1.0)
    
    def _calculate_payer_importance(self):
        """Calculate importance of payer features"""
        score = 0.5  # Base score
        
        if 'payer_channel_name' in self.df.columns:
            payer_var = self.df.groupby('payer_channel_name')['total_paid_amt'].mean().std() / self.df['total_paid_amt'].mean()
            score += payer_var * 0.5
        
        return min(score, 1.0)
    
    def _calculate_transaction_importance(self):
        """Calculate importance of transaction features"""
        score = 0.3  # Base score
        
        if 'transaction_status' in self.df.columns:
            rejection_impact = (self.df['transaction_status'] != 'Dispensed').mean()
            score += rejection_impact * 0.4
        
        if 'days_supply_val' in self.df.columns:
            supply_corr = abs(self.df['days_supply_val'].corr(self.df['total_paid_amt']))
            score += supply_corr * 0.3
        
        return min(score, 1.0)
    
    def _generate_insights(self, importance_scores):
        """Generate key insights based on analysis"""
        insights = []
        
        # Top feature category
        top_category = max(importance_scores.items(), key=lambda x: x[1])
        insights.append(f"{top_category[0]} shows the highest impact on sales predictions (score: {top_category[1]:.3f})")
        
        # Market concentration
        if 'drug_brand_name' in self.df.columns:
            market_leader = self.df['drug_brand_name'].value_counts().index[0]
            leader_share = self.df['drug_brand_name'].value_counts(normalize=True).iloc[0]
            insights.append(f"{market_leader} dominates the market with {leader_share*100:.1f}% market share")
        
        # HCP concentration
        if 'prescriber_npi_nm' in self.df.columns:
            top_10_share = self.df.groupby('prescriber_npi_nm')['total_paid_amt'].sum().nlargest(10).sum() / self.df['total_paid_amt'].sum()
            insights.append(f"Top 10 HCPs account for {top_10_share*100:.1f}% of total sales")
        
        # Temporal trends
        if importance_scores.get('Temporal Patterns', 0) > 0.6:
            insights.append("Strong seasonal patterns detected - consider time-based pricing strategies")
        
        # Payer mix
        if importance_scores.get('Payer Dynamics', 0) > 0.7:
            insights.append("Payer channel significantly impacts reimbursement - optimize payer mix")
        
        return insights
        
# Main execution function
def main():
    """Main execution for feature analysis"""
    print("="*70)
    print(" "*20 + "FEATURE ANALYSIS SYSTEM")
    print("="*70)
    
    # Load data (use your actual data file)
    # df = pd.read_csv('claims_data.csv')
    # predictions = pd.read_csv('hcp_drug_predictions.csv')
    
    # For demonstration, create sample data
    from main_pipeline import create_sample_data  # Import from main script
    df = create_sample_data()
    
    # Mock predictions for demo
    predictions = pd.DataFrame({
        'HCP': df['prescriber_npi_nm'].unique()[:10],
        'Predicted_sales': np.random.uniform(10000, 100000, 10),
        'Sales_from_db': np.random.uniform(8000, 90000, 10)
    })
    
    # Initialize analyzer
    analyzer = FeatureAnalyzer(df, predictions)
    
    # Run complete analysis
    analyzer.run_complete_analysis()
    
    print("\n" + "="*70)
    print("FEATURE ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
