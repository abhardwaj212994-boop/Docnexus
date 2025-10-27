"""
Main Pipeline Orchestrator
Runs the complete pharmaceutical sales prediction system in correct sequence
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all modules
try:
    from pharma_sales_prediction import (
        EnsemblePredictionModel, 
        DataPreprocessor,
        create_sample_data
    )
    from web_scraping_module import (
        IntelligentWebScraper,
        MarketIntelligenceAggregator
    )
    from model_analysis import (
        ModelAnalyzer,
        perform_statistical_tests
    )
    from feature_analysis import (
        FeatureAnalyzer,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all required modules are in the same directory:")
    print("- pharma_sales_prediction.py")
    print("- web_scraping_module.py")
    print("- model_analysis.py")
    print("- feature_analysis.py")
    sys.exit(1)


class PipelineOrchestrator:
    """Orchestrates the complete prediction pipeline"""
    
    def __init__(self, data_path=None, use_sample_data=False):
        """
        Initialize pipeline
        
        Parameters:
        -----------
        data_path : str, optional
            Path to claims data CSV file
        use_sample_data : bool, default=False
            If True, uses synthetic sample data for demonstration
        """
        self.data_path = data_path
        self.use_sample_data = use_sample_data
        self.output_dir = 'pipeline_outputs'
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.preprocessor = None
        self.web_scraper = None
        self.ensemble_model = None
        self.df_clean = None
        self.predictions = None
        
        print("="*80)
        print(" "*20 + "PHARMACEUTICAL SALES PREDICTION PIPELINE")
        print("="*80)
        print(f"Initialized at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {self.output_dir}")
        print("="*80)
    
    def run_complete_pipeline(self):
        """Execute the complete prediction pipeline"""
        
        try:
            # Step 1: Load and preprocess data
            self.step1_load_data()
            
            # Step 2: Web scraping for external data
            self.step2_web_scraping()
            
            # Step 3: Train prediction models
            self.step3_train_models()
            
            # Step 4: Generate predictions
            self.step4_generate_predictions()
            
            # Step 5: Model performance analysis
            self.step5_model_analysis()
            
            # Step 6: Feature importance analysis
            self.step6_feature_analysis()
            
            # Step 7: Generate final report
            self.step7_generate_report()
            
            print("\n" + "="*80)
            print(" "*25 + "PIPELINE COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"\nAll outputs saved to: {self.output_dir}/")
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(" "*30 + "PIPELINE ERROR")
            print("="*80)
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            print("\nPipeline terminated.")
    
    def step1_load_data(self):
        """Step 1: Load and preprocess data"""
        print("\n" + "="*80)
        print("STEP 1: DATA LOADING AND PREPROCESSING")
        print("="*80)
        
        if self.use_sample_data or self.data_path is None:
            print("\nUsing sample data for demonstration...")
            df = create_sample_data()
            print(f"Sample data created: {len(df)} records")
        else:
            print(f"\nLoading data from: {self.data_path}")
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            df = pd.read_csv(self.data_path)
            print(f"Data loaded: {len(df)} records")
        
        # Initialize preprocessor and clean data
        self.preprocessor = DataPreprocessor()
        self.df_clean = self.preprocessor.load_and_clean_data(df)
        
        print(f"\nData cleaning complete:")
        print(f"  Valid claims: {len(self.df_clean)}")
        print(f"  Unique HCPs: {self.df_clean['hcp_id'].nunique()}")
        print(f"  Unique Drugs: {self.df_clean['drug_brand_name'].nunique()}")
        print(f"  Unique Patients: {self.df_clean['patient_id'].nunique()}")
        print(f"  Total Sales: ${self.df_clean['total_paid_amt'].sum():,.2f}")
        
        # Save cleaned data
        cleaned_path = os.path.join(self.output_dir, f'cleaned_data_{self.timestamp}.csv')
        self.df_clean.to_csv(cleaned_path, index=False)
        print(f"\nCleaned data saved to: {cleaned_path}")
    
    def step2_web_scraping(self):
        """Step 2: Collect external market intelligence"""
        print("\n" + "="*80)
        print("STEP 2: WEB SCRAPING & EXTERNAL DATA COLLECTION")
        print("="*80)
        
        # Initialize web scraper
        self.web_scraper = IntelligentWebScraper(use_cache=True)
        aggregator = MarketIntelligenceAggregator(self.web_scraper)
        
        # Collect market data
        print("\n2.1 Collecting market share data...")
        market_data = self.web_scraper.scrape_market_share_data()
        print(f"    Market data collected for {len(market_data['drug_shares'])} drugs")
        
        # Collect drug efficacy data
        print("\n2.2 Collecting drug efficacy data...")
        drugs = self.df_clean['drug_brand_name'].unique()
        drug_efficacy = {}
        for drug in drugs:
            efficacy = self.web_scraper.scrape_drug_efficacy_data(drug)
            drug_efficacy[drug] = efficacy
            print(f"    {drug}: Response Rate = {efficacy['efficacy_metrics'].get('overall_response_rate', 0):.1%}")
        
        # Collect HCP profiles
        print("\n2.3 Collecting HCP profiles...")
        sample_hcps = self.df_clean['hcp_id'].unique()[:10]
        hcp_profiles = []
        for hcp in sample_hcps:
            profile = self.web_scraper.scrape_hcp_profile(hcp)
            hcp_profiles.append(profile)
        print(f"    Collected profiles for {len(hcp_profiles)} HCPs")
        
        # Collect payer data
        print("\n2.4 Collecting payer policies...")
        payer_landscape = aggregator.get_payer_landscape()
        print(f"    Collected data for {len(payer_landscape)} payer channels")
        
        # Collect pricing data
        print("\n2.5 Collecting pricing data...")
        pricing_df = self.web_scraper.scrape_pricing_data()
        print(f"    Pricing data collected for {len(pricing_df)} drugs")
        
        # Save external data
        ext_data_path = os.path.join(self.output_dir, f'external_data_{self.timestamp}.csv')
        pricing_df.to_csv(ext_data_path, index=False)
        
        # Save market intelligence summary
        market_summary = {
            'market_data': market_data,
            'drug_efficacy': drug_efficacy,
            'payer_landscape': payer_landscape
        }
        
        import json
        summary_path = os.path.join(self.output_dir, f'market_intelligence_{self.timestamp}.json')
        with open(summary_path, 'w') as f:
            json.dump(market_summary, f, indent=2, default=str)
        
        print(f"\nExternal data saved to: {self.output_dir}/")
    
    def step3_train_models(self):
        """Step 3: Train all prediction models"""
        print("\n" + "="*80)
        print("STEP 3: MODEL TRAINING")
        print("="*80)
        
        # Initialize ensemble model
        self.ensemble_model = EnsemblePredictionModel()
        
        print("\n3.1 Training Model 1: Temporal & Demographic...")
        print("3.2 Training Model 2: Market Dynamics...")
        print("3.3 Training Model 3: HCP Specialization...")
        print("3.4 Training Model 4: Transaction Type...")
        print("3.5 Training Model 5: Rolling Sales...")
        print("3.6 Training Model 6: Payer Reputation...")
        
        # Train all models
        self.ensemble_model.train_all_models(self.df_clean)
        
        print("\n All models trained successfully!")
    
    def step4_generate_predictions(self):
        """Step 4: Generate predictions"""
        print("\n" + "="*80)
        print("STEP 4: GENERATING PREDICTIONS")
        print("="*80)
        
        # Generate predictions (now includes all HCP data from input)
        print("\nGenerating HCP-Drug level predictions...")
        self.predictions = self.ensemble_model.predict(self.df_clean)
        
        # Display summary
        print(f"\nPredictions generated for:")
        print(f"  Total HCP-Drug combinations: {len(self.predictions)}")
        print(f"  Unique HCPs: {self.predictions['hcp_id'].nunique()}")
        print(f"  Unique Drugs: {self.predictions['Drug_name'].nunique()}")
        
        print(f"\nPrediction Summary:")
        print(f"  Total Predicted Sales: ${self.predictions['Predicted_sales'].sum():,.2f}")
        print(f"  Total DB Sales:        ${self.predictions['Sales_from_db'].sum():,.2f}")
        print(f"  Average Multiplier:    {self.predictions['Combined_multiplier'].mean():.3f}")
        print(f"  Sales Uplift:          {(self.predictions['Predicted_sales'].sum() / self.predictions['Sales_from_db'].sum() - 1) * 100:.1f}%")
        
        # Top predictions - show first few HCP columns
        print("\nTop 5 Predicted Sales:")
        hcp_cols = [col for col in self.predictions.columns if 'prescriber' in col.lower() or col == 'hcp_id'][:3]
        display_cols = hcp_cols + ['Drug_name', 'Predicted_sales', 'Sales_from_db', 'Combined_multiplier']
        top_5 = self.predictions.nlargest(5, 'Predicted_sales')[display_cols]
        print(top_5.to_string(index=False))
        
        # Save predictions
        pred_path = os.path.join(self.output_dir, f'predictions_{self.timestamp}.csv')
        self.predictions.to_csv(pred_path, index=False)
        print(f"\nPredictions saved to: {pred_path}")
        print(f"Output contains {len(self.predictions.columns)} columns including all HCP data from input")
    
    def step5_model_analysis(self):
        """Step 5: Analyze model performance"""
        print("\n" + "="*80)
        print("STEP 5: MODEL PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Initialize analyzer
        analyzer = ModelAnalyzer(self.predictions)
        
        # Run complete analysis
        analyzer.run_complete_analysis()
        
        # Perform statistical tests
        print("\n" + "-"*80)
        perform_statistical_tests(self.predictions)
        
        # Save metrics
        if 'sales' in analyzer.metrics:
            metrics_df = pd.DataFrame({
                'Metric': ['MAE', 'RMSE', 'MAPE', 'R2', 'Direction_Accuracy'],
                'Value': [
                    analyzer.metrics['sales']['MAE'],
                    analyzer.metrics['sales']['RMSE'],
                    analyzer.metrics['sales']['MAPE'],
                    analyzer.metrics['sales']['R2'],
                    analyzer.metrics['sales']['Direction_Accuracy']
                ]
            })
            
            metrics_path = os.path.join(self.output_dir, f'model_metrics_{self.timestamp}.csv')
            metrics_df.to_csv(metrics_path, index=False)
            print(f"\nModel metrics saved to: {metrics_path}")
    
    def step6_feature_analysis(self):
        """Step 6: Analyze feature importance"""
        print("\n" + "="*80)
        print("STEP 6: FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        # Initialize feature analyzer
        feature_analyzer = FeatureAnalyzer(self.df_clean, self.predictions)
        
        # Run complete analysis
        feature_analyzer.run_complete_analysis()
        
    def step7_generate_report(self):
        """Step 7: Generate executive summary report"""
        print("\n" + "="*80)
        print("STEP 7: GENERATING EXECUTIVE SUMMARY")
        print("="*80)
        
        report = self._create_executive_summary()
        
        # Save report
        report_path = os.path.join(self.output_dir, f'executive_summary_{self.timestamp}.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"\nExecutive summary saved to: {report_path}")
    
    def _create_executive_summary(self):
        """Create executive summary report"""
        report = []
        report.append("="*80)
        report.append(" "*20 + "EXECUTIVE SUMMARY REPORT")
        report.append("="*80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Pipeline Run ID: {self.timestamp}")
        
        # Data Overview
        report.append("\n" + "-"*80)
        report.append("DATA OVERVIEW")
        report.append("-"*80)
        report.append(f"Total Claims Analyzed:        {len(self.df_clean):,}")
        report.append(f"Unique Healthcare Providers:  {self.df_clean['hcp_id'].nunique():,}")
        report.append(f"Unique Drugs:                 {self.df_clean['drug_brand_name'].nunique()}")
        report.append(f"Unique Patients:              {self.df_clean['patient_id'].nunique():,}")
        report.append(f"Total Historical Sales:       ${self.df_clean['total_paid_amt'].sum():,.2f}")
        
        # Prediction Summary
        report.append("\n" + "-"*80)
        report.append("PREDICTION SUMMARY")
        report.append("-"*80)
        report.append(f"Total Predicted Sales:        ${self.predictions['Predicted_sales'].sum():,.2f}")
        report.append(f"Prediction Count:             {len(self.predictions):,}")
        report.append(f"Average Sales per HCP-Drug:   ${self.predictions['Predicted_sales'].mean():,.2f}")
        report.append(f"Median Sales per HCP-Drug:    ${self.predictions['Predicted_sales'].median():,.2f}")
        
        # Calculate accuracy metrics
        from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
        
        mae = mean_absolute_error(
            self.predictions['Sales_from_db'], 
            self.predictions['Predicted_sales']
        )
        mape = mean_absolute_percentage_error(
            self.predictions['Sales_from_db'], 
            self.predictions['Predicted_sales']
        ) * 100
        r2 = r2_score(
            self.predictions['Sales_from_db'], 
            self.predictions['Predicted_sales']
        )
        
        report.append("\n" + "-"*80)
        report.append("MODEL PERFORMANCE")
        report.append("-"*80)
        report.append(f"Mean Absolute Error (MAE):    ${mae:,.2f}")
        report.append(f"Mean Absolute % Error (MAPE): {mape:.2f}%")
        report.append(f"R-squared (R²):               {r2:.3f}")
        
        # Performance grade
        if r2 > 0.8 and mape < 10:
            grade = "Excellent (A)"
        elif r2 > 0.6 and mape < 20:
            grade = "Good (B)"
        elif r2 > 0.4 and mape < 30:
            grade = "Satisfactory (C)"
        else:
            grade = "Needs Improvement"
        
        report.append(f"Overall Model Grade:          {grade}")
        
        # Drug-level insights
        report.append("\n" + "-"*80)
        report.append("DRUG-LEVEL INSIGHTS")
        report.append("-"*80)
        
        drug_summary = self.predictions.groupby('Drug_name').agg({
            'Predicted_sales': 'sum',
            'Sales_from_db': 'sum',
            'hcp_id': 'count'
        }).round(2)
        
        for drug in drug_summary.index:
            pred = drug_summary.loc[drug, 'Predicted_sales']
            actual = drug_summary.loc[drug, 'Sales_from_db']
            count = drug_summary.loc[drug, 'hcp_id']
            uplift = (pred / actual - 1) * 100 if actual > 0 else 0
            
            report.append(f"\n{drug}:")
            report.append(f"  Predicted Sales: ${pred:,.2f}")
            report.append(f"  Historical Sales: ${actual:,.2f}")
            report.append(f"  Sales Uplift: {uplift:+.1f}%")
            report.append(f"  HCP-Drug Combinations: {int(count)}")
        
        # Top HCPs
        report.append("\n" + "-"*80)
        report.append("TOP 10 HCPs BY PREDICTED SALES")
        report.append("-"*80)
        
        # Use hcp_id column
        top_hcps = self.predictions.groupby('hcp_id')['Predicted_sales'].sum().nlargest(10)
        for i, (hcp, sales) in enumerate(top_hcps.items(), 1):
            report.append(f"{i:2d}. {hcp}: ${sales:,.2f}")
        
        report.append("\n" + "="*80)
        report.append("END OF EXECUTIVE SUMMARY")
        report.append("="*80)
        
        return "\n".join(report)


def main():
    """Main entry point for the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Pharmaceutical Sales Prediction Pipeline'
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Path to claims data CSV file'
    )
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Use sample data for demonstration'
    )
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = PipelineOrchestrator(
        data_path=args.data,
        use_sample_data=args.sample or args.data is None
    )
    
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()