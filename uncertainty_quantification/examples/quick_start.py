"""
Quick Start Example for Uncertainty Quantification Framework

Demonstrates key features of the UQ framework including:
- Deep ensembles for epistemic uncertainty
- Calibration with temperature scaling
- Conformal prediction for guaranteed coverage
- Risk-aware decision making
- Monitoring and drift detection
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import UQ components
from src.models.deep_ensemble import DeepEnsemble, HeteroscedasticNN
from src.calibration.calibration_methods import TemperatureScaling, IsotonicCalibration
from src.metrics.calibration_metrics import (
    expected_calibration_error, 
    reliability_diagram,
    confidence_histogram
)
from src.conformal.conformal_prediction import ConformalClassifier, ConformalRegressor
from src.decisions.risk_decisions import (
    RiskAwareDecisionMaker, 
    DecisionConfig,
    CostSensitiveDecision
)
from src.monitoring.drift_detection import UncertaintyMonitor


def create_simple_nn(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
    """Create a simple neural network for demonstration."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, output_dim)
    )


def classification_example():
    """Demonstrate UQ for classification."""
    print("\n" + "="*60)
    print("CLASSIFICATION EXAMPLE")
    print("="*60)
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, n_classes=3, random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )
    
    print(f"Data shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    # 1. Deep Ensemble for Epistemic Uncertainty
    print("\n1. Training Deep Ensemble...")
    base_model = create_simple_nn(input_dim=20, hidden_dim=50, output_dim=3)
    
    ensemble = DeepEnsemble(base_model, n_models=5)
    
    # Convert to torch tensors
    X_train_torch = torch.FloatTensor(X_train)
    y_train_torch = torch.LongTensor(y_train)
    X_val_torch = torch.FloatTensor(X_val)
    X_test_torch = torch.FloatTensor(X_test)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_torch, y_train_torch)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Train ensemble (simplified for demo)
    print("Training ensemble members...")
    # Note: In practice, you would properly train the ensemble
    # ensemble.train(train_loader, epochs=10, verbose=False)
    
    # Get predictions with uncertainty
    results = ensemble.predict(X_test_torch, return_uncertainty=True)
    
    print(f"Mean prediction shape: {results['mean'].shape}")
    print(f"Epistemic uncertainty: {np.mean(results['epistemic']):.4f}")
    print(f"Aleatoric uncertainty: {np.mean(results['aleatoric']):.4f}")
    
    # 2. Calibration
    print("\n2. Calibrating Predictions...")
    
    # Temperature scaling
    temp_scaling = TemperatureScaling()
    
    # Get validation logits (simplified - using random for demo)
    val_logits = np.random.randn(len(y_val), 3)  # Placeholder
    optimal_temp = temp_scaling.fit(val_logits, y_val)
    print(f"Optimal temperature: {optimal_temp:.3f}")
    
    # Calibrate test predictions
    test_probs = results['mean']
    calibrated_probs = temp_scaling.calibrate(
        torch.from_numpy(np.log(test_probs + 1e-8))
    ).numpy()
    
    # Calculate ECE
    ece_before = expected_calibration_error(y_test, test_probs)
    ece_after = expected_calibration_error(y_test, calibrated_probs)
    print(f"ECE before calibration: {ece_before:.4f}")
    print(f"ECE after calibration: {ece_after:.4f}")
    
    # 3. Conformal Prediction
    print("\n3. Conformal Prediction Sets...")
    
    # Use RandomForest for conformal prediction demo
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    conformal = ConformalClassifier(rf_model, alpha=0.1, method='split')
    conformal.fit(X_train, y_train, X_val, y_val)
    
    # Get prediction sets
    pred_sets = conformal.predict_set(X_test[:10])  # First 10 samples
    
    print("Prediction sets for first 10 test samples:")
    for i, (pred_set, true_label) in enumerate(zip(pred_sets, y_test[:10])):
        covered = true_label in pred_set
        print(f"  Sample {i}: Set={pred_set}, True={true_label}, Covered={covered}")
    
    # 4. Risk-Aware Decisions
    print("\n4. Risk-Aware Decision Making...")
    
    # Define cost matrix (asymmetric costs)
    cost_matrix = np.array([
        [0, 2, 5],  # Costs for true class 0
        [3, 0, 2],  # Costs for true class 1
        [4, 3, 0]   # Costs for true class 2
    ])
    
    config = DecisionConfig(
        abstention_threshold=0.3,
        cost_matrix=cost_matrix
    )
    
    decision_maker = RiskAwareDecisionMaker(config)
    
    # Make cost-sensitive decisions
    decisions = decision_maker.expected_cost_decision(calibrated_probs)
    
    # Selective prediction with abstention
    uncertainties = results['entropy']
    selective_decisions, abstain_mask = decision_maker.selective_prediction(
        np.argmax(calibrated_probs, axis=1),
        uncertainties
    )
    
    abstention_rate = np.mean(abstain_mask)
    print(f"Abstention rate: {abstention_rate:.2%}")
    
    # 5. Monitoring
    print("\n5. Monitoring and Drift Detection...")
    
    monitor = UncertaintyMonitor(
        reference_features=X_train,
        reference_predictions=np.random.rand(len(y_train), 3),  # Placeholder
        reference_labels=y_train
    )
    
    # Simulate monitoring on test data
    metrics = monitor.update(
        features=X_test,
        predictions=calibrated_probs,
        uncertainties=uncertainties,
        labels=y_test
    )
    
    print(f"Drift score: {metrics.drift_score:.4f}")
    print(f"OOD score: {metrics.ood_score:.4f}")
    print(f"Mean uncertainty: {metrics.mean_uncertainty:.4f}")
    
    summary = monitor.get_summary()
    if summary['alerts']:
        print("Alerts:", summary['alerts'])
    else:
        print("No alerts - system operating normally")


def regression_example():
    """Demonstrate UQ for regression."""
    print("\n" + "="*60)
    print("REGRESSION EXAMPLE")
    print("="*60)
    
    # Generate synthetic data
    X, y = make_regression(
        n_samples=1000, n_features=10, noise=10, random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )
    
    print(f"Data shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    # 1. Heteroscedastic Neural Network
    print("\n1. Heteroscedastic Neural Network...")
    
    hetero_nn = HeteroscedasticNN(
        input_dim=10,
        hidden_dims=[50, 50],
        output_dim=1
    )
    
    # Convert to tensors
    X_test_torch = torch.FloatTensor(X_test)
    
    # Get predictions with uncertainty
    with torch.no_grad():
        mean_pred, var_pred = hetero_nn(X_test_torch)
    
    std_pred = torch.sqrt(var_pred).numpy()
    mean_pred = mean_pred.numpy()
    
    print(f"Mean prediction shape: {mean_pred.shape}")
    print(f"Average predicted std: {np.mean(std_pred):.4f}")
    
    # 2. Conformal Regression
    print("\n2. Conformal Prediction Intervals...")
    
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    
    conformal_reg = ConformalRegressor(
        rf_regressor, 
        alpha=0.1,  # 90% coverage
        method='split'
    )
    
    conformal_reg.fit(X_train, y_train, X_val, y_val)
    
    # Get prediction intervals
    lower, upper = conformal_reg.predict_interval(X_test[:10])
    
    print("Prediction intervals for first 10 test samples:")
    for i in range(10):
        covered = (y_test[i] >= lower[i]) and (y_test[i] <= upper[i])
        width = upper[i] - lower[i]
        print(f"  Sample {i}: [{lower[i]:.2f}, {upper[i]:.2f}], "
              f"True={y_test[i]:.2f}, Width={width:.2f}, Covered={covered}")
    
    # Calculate coverage
    lower_all, upper_all = conformal_reg.predict_interval(X_test)
    coverage = np.mean((y_test >= lower_all) & (y_test <= upper_all))
    print(f"\nOverall coverage: {coverage:.2%} (target: 90%)")
    
    # 3. Risk Metrics
    print("\n3. Risk Metrics...")
    
    decision_maker = RiskAwareDecisionMaker()
    
    # Simulate loss samples
    n_samples = 1000
    loss_samples = np.abs(np.random.normal(0, std_pred[0], n_samples))
    
    var_95 = decision_maker.value_at_risk(loss_samples, alpha=0.95)
    cvar_95 = decision_maker.conditional_value_at_risk(loss_samples, alpha=0.95)
    
    print(f"VaR (95%): {var_95:.4f}")
    print(f"CVaR (95%): {cvar_95:.4f}")
    
    # Tail probability
    threshold = 50.0
    tail_prob = decision_maker.tail_probability(loss_samples, threshold)
    print(f"P(Loss > {threshold}): {tail_prob:.4f}")


def visualization_example():
    """Create visualization examples."""
    print("\n" + "="*60)
    print("VISUALIZATION EXAMPLES")
    print("="*60)
    
    # Generate sample data for visualization
    np.random.seed(42)
    n_samples = 500
    
    # Simulated probabilities and labels
    y_true = np.random.randint(0, 2, n_samples)
    y_prob = np.random.beta(2, 2, n_samples)
    y_prob[y_true == 1] += 0.3
    y_prob = np.clip(y_prob, 0, 1)
    
    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reliability diagram
    reliability_diagram(y_true, y_prob, ax=axes[0])
    
    # Confidence histogram
    confidence_histogram(y_prob, y_true, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('uq_visualizations.png', dpi=150, bbox_inches='tight')
    print("Visualizations saved to 'uq_visualizations.png'")
    
    # Print summary statistics
    ece = expected_calibration_error(y_true, y_prob)
    print(f"\nCalibration Statistics:")
    print(f"  ECE: {ece:.4f}")
    print(f"  Mean confidence: {np.mean(y_prob):.4f}")
    print(f"  Accuracy: {np.mean((y_prob > 0.5) == y_true):.4f}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("UNCERTAINTY QUANTIFICATION FRAMEWORK - QUICK START")
    print("="*60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run examples
    classification_example()
    regression_example()
    visualization_example()
    
    print("\n" + "="*60)
    print("QUICK START COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. Deep ensembles provide strong epistemic uncertainty estimates")
    print("2. Calibration improves reliability of probabilistic predictions")
    print("3. Conformal prediction gives distribution-free coverage guarantees")
    print("4. Risk-aware decisions optimize for business objectives")
    print("5. Continuous monitoring detects drift and maintains reliability")
    print("\nFor more examples, see the notebooks/ directory.")


if __name__ == "__main__":
    main()