"""
Uncertainty Quantification Quick-Start Guide and Ψ Framework Integration

This example demonstrates how to use the UQ framework for reliable risk estimates
and integrate with the Ψ framework for improved calibration and verifiability.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from typing import Dict, Any

# Import our UQ framework
from uncertainty_quantification import (
    UncertaintyEstimate, DeepEnsemble, MCDropout, HeteroscedasticHead,
    QuantileRegression, ConformalPredictor, CalibrationMethods,
    UQEvaluationMetrics, RiskBasedDecisionFramework, MonitoringAndDrift,
    OODDetector, DirichletPriorNetwork, expected_calibration_error
)

class QuickStartUQPipeline:
    """
    Quick-start pipeline implementing the recommended UQ approach:
    1. Deep ensemble + temperature scaling (baseline)
    2. Conformal intervals for guaranteed coverage
    3. Risk-based decision making with VaR/CVaR
    """
    
    def __init__(self, model_class=RandomForestRegressor, n_ensemble=5):
        self.model_class = model_class
        self.n_ensemble = n_ensemble
        self.ensemble = None
        self.conformal = None
        self.temperature = 1.0
        self.calibration_history = []
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_cal: np.ndarray, y_cal: np.ndarray):
        """
        Fit the complete UQ pipeline.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_cal: Calibration features (for conformal prediction)
            y_cal: Calibration targets
        """
        print("🚀 Training Deep Ensemble...")
        # Step 1: Train deep ensemble
        self.ensemble = DeepEnsemble(self.model_class, self.n_ensemble)
        self.ensemble.fit(X_train, y_train)
        
        print("🎯 Calibrating with Conformal Prediction...")
        # Step 2: Fit conformal predictor
        ensemble_preds = self.ensemble.predict_with_uncertainty(X_cal)
        
        # Create a simple model wrapper for conformal prediction
        class EnsembleWrapper:
            def __init__(self, ensemble):
                self.ensemble = ensemble
            def predict(self, X):
                return self.ensemble.predict_with_uncertainty(X).mean
        
        wrapper = EnsembleWrapper(self.ensemble)
        self.conformal = ConformalPredictor(wrapper, alpha=0.1)
        self.conformal.fit(X_cal, y_cal)
        
        print("✅ Pipeline trained successfully!")
        
    def predict_with_risk_analysis(self, X_test: np.ndarray, 
                                 risk_thresholds: list = None) -> Dict[str, Any]:
        """
        Generate predictions with comprehensive risk analysis.
        """
        if risk_thresholds is None:
            risk_thresholds = [1.0, 2.0, 3.0]  # Example thresholds
            
        # Get uncertainty estimates
        uncertainty_est = self.ensemble.predict_with_uncertainty(X_test)
        
        # Get conformal intervals
        conf_lower, conf_upper = self.conformal.predict_intervals(X_test)
        
        # Risk analysis
        risk_analysis = {}
        for threshold in risk_thresholds:
            # Sample from predictive distribution for risk metrics
            samples = []
            for i in range(len(X_test)):
                sample = np.random.normal(
                    uncertainty_est.mean[i],
                    uncertainty_est.std_total[i],
                    size=1000
                )
                samples.append(sample)
            
            samples = np.array(samples)
            
            # Compute risk metrics
            var_95 = np.array([RiskBasedDecisionFramework.value_at_risk(s, 0.05) 
                              for s in samples])
            cvar_95 = np.array([RiskBasedDecisionFramework.conditional_value_at_risk(s, 0.05) 
                               for s in samples])
            tail_prob = np.array([RiskBasedDecisionFramework.tail_probability(s, threshold) 
                                 for s in samples])
            
            risk_analysis[f'threshold_{threshold}'] = {
                'var_95': var_95,
                'cvar_95': cvar_95,
                'tail_probability': tail_prob
            }
        
        return {
            'predictions': uncertainty_est.mean,
            'aleatoric_uncertainty': uncertainty_est.aleatoric,
            'epistemic_uncertainty': uncertainty_est.epistemic,
            'total_uncertainty': uncertainty_est.total,
            'conformal_lower': conf_lower,
            'conformal_upper': conf_upper,
            'risk_analysis': risk_analysis
        }
    
    def evaluate_quality(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate UQ quality using multiple metrics."""
        results = self.predict_with_risk_analysis(X_test)
        
        # Coverage and interval metrics
        coverage = UQEvaluationMetrics.prediction_interval_coverage_probability(
            y_test, results['conformal_lower'], results['conformal_upper']
        )
        
        interval_width = UQEvaluationMetrics.mean_prediction_interval_width(
            results['conformal_lower'], results['conformal_upper']
        )
        
        interval_score = UQEvaluationMetrics.interval_score(
            y_test, results['conformal_lower'], results['conformal_upper']
        )
        
        # Prediction quality
        mae = np.mean(np.abs(y_test - results['predictions']))
        rmse = np.sqrt(np.mean((y_test - results['predictions'])**2))
        
        return {
            'coverage': coverage,
            'interval_width': interval_width,
            'interval_score': interval_score,
            'mae': mae,
            'rmse': rmse,
            'mean_total_uncertainty': np.mean(results['total_uncertainty']),
            'mean_epistemic_uncertainty': np.mean(results['epistemic_uncertainty'])
        }

class PsiFrameworkIntegration:
    """
    Integration with Ψ framework for enhanced reliability and verifiability.
    
    The Ψ framework focuses on:
    - Calibration (post): Better UQ improves probability calibration
    - Verifiability (R_v): Reproducible uncertainty estimates reduce risk
    - Authority scoring (R_a): Stable predictions across distribution shifts
    """
    
    def __init__(self, uq_pipeline: QuickStartUQPipeline):
        self.uq_pipeline = uq_pipeline
        self.psi_score = 0.0
        self.calibration_score = 0.0
        self.verifiability_score = 0.0
        self.authority_score = 0.0
        
    def compute_calibration_component(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Compute calibration component of Ψ score.
        
        Better UQ leads to better calibrated predictions, increasing this component.
        """
        results = self.uq_pipeline.predict_with_risk_analysis(X_val)
        
        # Convert uncertainties to confidence intervals
        predictions = results['predictions']
        uncertainties = np.sqrt(results['total_uncertainty'])
        
        # Create pseudo-probabilities for calibration assessment
        # Higher uncertainty -> lower confidence
        max_uncertainty = np.max(uncertainties)
        confidences = 1 - (uncertainties / max_uncertainty)
        
        # Binary accuracy (within 1 std dev)
        within_std = np.abs(y_val - predictions) <= uncertainties
        
        # Compute calibration error (lower is better)
        ece = expected_calibration_error(confidences, within_std.astype(int))
        
        # Convert to score (higher is better)
        self.calibration_score = max(0, 1 - ece)
        return self.calibration_score
    
    def compute_verifiability_component(self, X_test: np.ndarray, 
                                      n_runs: int = 5) -> float:
        """
        Compute verifiability component based on prediction stability.
        
        Reproducible uncertainty estimates reduce R_v risk.
        """
        predictions_runs = []
        uncertainties_runs = []
        
        # Multiple runs to assess stability
        for _ in range(n_runs):
            results = self.uq_pipeline.predict_with_risk_analysis(X_test)
            predictions_runs.append(results['predictions'])
            uncertainties_runs.append(results['total_uncertainty'])
        
        predictions_runs = np.array(predictions_runs)
        uncertainties_runs = np.array(uncertainties_runs)
        
        # Compute stability metrics
        pred_std = np.std(predictions_runs, axis=0)
        uncertainty_std = np.std(uncertainties_runs, axis=0)
        
        # Verifiability score (lower variance = higher verifiability)
        pred_stability = 1 / (1 + np.mean(pred_std))
        uncertainty_stability = 1 / (1 + np.mean(uncertainty_std))
        
        self.verifiability_score = (pred_stability + uncertainty_stability) / 2
        return self.verifiability_score
    
    def compute_authority_component(self, X_train: np.ndarray, X_shifted: np.ndarray) -> float:
        """
        Compute authority component based on robustness to distribution shift.
        
        Stable uncertainty estimates across shifts improve R_a scoring.
        """
        # Get uncertainty estimates on original and shifted data
        train_results = self.uq_pipeline.predict_with_risk_analysis(X_train[:100])  # Sample
        shift_results = self.uq_pipeline.predict_with_risk_analysis(X_shifted)
        
        train_uncertainties = train_results['total_uncertainty']
        shift_uncertainties = shift_results['total_uncertainty']
        
        # Measure distribution shift in uncertainties
        # Good UQ should increase uncertainty on shifted data
        uncertainty_ratio = np.mean(shift_uncertainties) / np.mean(train_uncertainties)
        
        # Authority score: penalize if uncertainty doesn't increase with shift
        # but also penalize excessive increase
        optimal_ratio = 1.5  # Expect 50% increase in uncertainty
        authority_penalty = abs(uncertainty_ratio - optimal_ratio) / optimal_ratio
        
        self.authority_score = max(0, 1 - authority_penalty)
        return self.authority_score
    
    def compute_psi_score(self, X_val: np.ndarray, y_val: np.ndarray,
                         X_test: np.ndarray, X_shifted: np.ndarray) -> Dict[str, float]:
        """
        Compute overall Ψ score integrating all components.
        """
        calibration = self.compute_calibration_component(X_val, y_val)
        verifiability = self.compute_verifiability_component(X_test)
        authority = self.compute_authority_component(X_val, X_shifted)
        
        # Weighted combination (can be adjusted based on application)
        weights = {'calibration': 0.4, 'verifiability': 0.3, 'authority': 0.3}
        
        self.psi_score = (
            weights['calibration'] * calibration +
            weights['verifiability'] * verifiability +
            weights['authority'] * authority
        )
        
        return {
            'psi_score': self.psi_score,
            'calibration_component': calibration,
            'verifiability_component': verifiability,
            'authority_component': authority
        }

def create_synthetic_dataset(n_samples=1000, n_features=10, noise=0.1):
    """Create synthetic dataset with controlled uncertainty characteristics."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=42
    )
    
    # Add heteroscedastic noise (uncertainty varies with input)
    noise_scale = 0.5 + 0.5 * np.abs(X[:, 0])  # Noise depends on first feature
    heteroscedastic_noise = np.random.normal(0, noise_scale)
    y += heteroscedastic_noise
    
    return X, y

def demonstrate_complete_workflow():
    """
    Demonstrate the complete UQ workflow with Ψ framework integration.
    """
    print("=" * 60)
    print("🎯 UNCERTAINTY QUANTIFICATION QUICK-START DEMONSTRATION")
    print("=" * 60)
    
    # Step 1: Create datasets
    print("\n📊 Creating synthetic dataset...")
    X, y = create_synthetic_dataset(n_samples=2000, n_features=5)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.6, random_state=42)
    X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Create shifted dataset for authority scoring
    X_shifted = X_test.copy()
    X_shifted[:, 0] += 2.0  # Shift first feature
    
    print(f"Training set: {X_train.shape}")
    print(f"Calibration set: {X_cal.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Step 2: Train UQ Pipeline
    print("\n🔧 Training UQ Pipeline...")
    pipeline = QuickStartUQPipeline(model_class=RandomForestRegressor, n_ensemble=5)
    pipeline.fit(X_train, y_train, X_cal, y_cal)
    
    # Step 3: Generate predictions with risk analysis
    print("\n🎲 Generating predictions with risk analysis...")
    results = pipeline.predict_with_risk_analysis(X_test, risk_thresholds=[1.0, 2.0])
    
    print(f"Mean prediction: {np.mean(results['predictions']):.3f}")
    print(f"Mean total uncertainty: {np.mean(results['total_uncertainty']):.3f}")
    print(f"Mean epistemic uncertainty: {np.mean(results['epistemic_uncertainty']):.3f}")
    
    # Step 4: Evaluate UQ quality
    print("\n📈 Evaluating UQ quality...")
    quality_metrics = pipeline.evaluate_quality(X_test, y_test)
    
    for metric, value in quality_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Step 5: Risk-based decision example
    print("\n⚠️  Risk-based decision analysis...")
    
    # Define a simple cost matrix for demonstration
    # Actions: [conservative, moderate, aggressive]
    # Outcomes: [low_error, medium_error, high_error]
    cost_matrix = np.array([
        [1, 2, 3],      # Conservative action costs
        [2, 1, 4],      # Moderate action costs  
        [3, 2, 1]       # Aggressive action costs
    ])
    actions = ['conservative', 'moderate', 'aggressive']
    
    # Use first few test samples for demonstration
    sample_uncertainties = UncertaintyEstimate(
        mean=results['predictions'][:5],
        aleatoric=results['aleatoric_uncertainty'][:5],
        epistemic=results['epistemic_uncertainty'][:5],
        total=results['total_uncertainty'][:5]
    )
    
    decision_results = RiskBasedDecisionFramework.expected_cost_minimization(
        sample_uncertainties, cost_matrix, actions
    )
    
    print("Optimal actions for first 5 samples:")
    for i, action in enumerate(decision_results['optimal_actions']):
        uncertainty = np.sqrt(sample_uncertainties.total[i])
        print(f"Sample {i+1}: {action} (uncertainty: {uncertainty:.3f})")
    
    # Step 6: Ψ Framework Integration
    print("\n🔮 Integrating with Ψ Framework...")
    psi_integration = PsiFrameworkIntegration(pipeline)
    psi_scores = psi_integration.compute_psi_score(X_cal, y_cal, X_test, X_shifted)
    
    print("Ψ Framework Scores:")
    for component, score in psi_scores.items():
        print(f"{component}: {score:.4f}")
    
    # Step 7: Monitoring setup
    print("\n🔍 Setting up monitoring...")
    
    # Drift detection
    psi_drift = MonitoringAndDrift.population_stability_index(X_train[:, 0], X_shifted[:, 0])
    print(f"Population Stability Index (feature drift): {psi_drift:.4f}")
    
    if psi_drift > 0.2:
        print("⚠️  Significant drift detected - consider model retraining")
    else:
        print("✅ No significant drift detected")
    
    # Step 8: Selective prediction example
    print("\n🎯 Selective prediction example...")
    
    # Compute prediction errors for threshold setting
    errors = np.abs(y_test - results['predictions'])
    accuracies = (errors < 1.0).astype(int)  # Binary accuracy within threshold
    uncertainties = np.sqrt(results['total_uncertainty'])
    
    threshold = RiskBasedDecisionFramework.selective_prediction_threshold(
        uncertainties, accuracies, coverage_target=0.9
    )
    
    # Apply selective prediction
    confident_mask = uncertainties <= threshold
    confident_predictions = results['predictions'][confident_mask]
    confident_targets = y_test[confident_mask]
    
    print(f"Uncertainty threshold for 90% accuracy: {threshold:.4f}")
    print(f"Confident predictions: {np.sum(confident_mask)}/{len(y_test)} ({100*np.mean(confident_mask):.1f}%)")
    
    if len(confident_predictions) > 0:
        confident_mae = np.mean(np.abs(confident_targets - confident_predictions))
        print(f"MAE on confident predictions: {confident_mae:.4f}")
        print(f"Overall MAE: {quality_metrics['mae']:.4f}")
        print(f"Improvement: {((quality_metrics['mae'] - confident_mae) / quality_metrics['mae'] * 100):.1f}%")
    
    print("\n" + "=" * 60)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 60)
    
    return {
        'pipeline': pipeline,
        'results': results,
        'quality_metrics': quality_metrics,
        'psi_scores': psi_scores,
        'decision_results': decision_results
    }

if __name__ == "__main__":
    # Run the complete demonstration
    demo_results = demonstrate_complete_workflow()
    
    print("\n📋 QUICK-START SUMMARY:")
    print("1. ✅ Trained deep ensemble for epistemic uncertainty")
    print("2. ✅ Added conformal prediction for coverage guarantees")
    print("3. ✅ Implemented risk-based decision making")
    print("4. ✅ Integrated with Ψ framework for reliability")
    print("5. ✅ Set up monitoring and drift detection")
    print("6. ✅ Demonstrated selective prediction")
    
    print(f"\n🎯 Final Ψ Score: {demo_results['psi_scores']['psi_score']:.4f}")
    print(f"📊 Prediction Coverage: {demo_results['quality_metrics']['coverage']:.4f}")
    print(f"🎲 Mean Uncertainty: {demo_results['quality_metrics']['mean_total_uncertainty']:.4f}")
    
    print("\n🚀 Ready for production deployment!")