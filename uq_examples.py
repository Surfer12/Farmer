"""
Practical Examples of Uncertainty Quantification
Demonstrates UQ methods for regression and classification with full pipelines
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class DeepEnsemble:
    """Deep ensemble for epistemic uncertainty estimation"""
    
    def __init__(self, models: List[nn.Module]):
        self.models = models
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns mean and epistemic uncertainty estimates"""
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        epistemic_var = predictions.var(dim=0)
        
        return mean, epistemic_var

class MCDropoutModel(nn.Module):
    """MC Dropout model for lightweight Bayesian inference"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        self.dropout_rate = dropout_rate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enable dropout during inference for uncertainty estimation"""
        self.train()  # Enable dropout
        predictions = []
        
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred)
        
        self.eval()  # Reset to eval mode
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0)
        
        return mean, uncertainty

class HeteroscedasticModel(nn.Module):
    """Heteroscedastic regression model predicting mean and variance"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.var_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive variance
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        mean = self.mean_head(features)
        var = self.var_head(features) + 1e-6  # Add small constant for numerical stability
        return mean, var
    
    def nll_loss(self, y_pred: Tuple[torch.Tensor, torch.Tensor], y_true: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood loss for heteroscedastic regression"""
        mean, var = y_pred
        return 0.5 * (torch.log(var) + (y_true - mean).pow(2) / var).mean()

class ConformalPredictor:
    """Conformal prediction for distribution-free coverage guarantees"""
    
    def __init__(self, base_model, alpha: float = 0.1):
        self.base_model = base_model
        self.alpha = alpha  # Miscoverage rate (1-alpha = coverage)
        self.quantile = None
    
    def calibrate(self, X_cal: torch.Tensor, y_cal: torch.Tensor):
        """Calibrate on held-out calibration set"""
        if hasattr(self.base_model, 'eval'):
            self.base_model.eval()
        
        with torch.no_grad():
            if hasattr(self.base_model, 'forward') and callable(getattr(self.base_model, 'forward')):
                predictions = self.base_model(X_cal)
            else:
                predictions = self.base_model.predict(X_cal)[0]  # For ensemble
            
            if isinstance(predictions, tuple):
                predictions = predictions[0]  # Take mean if tuple
                
        # Compute conformity scores (absolute residuals)
        scores = torch.abs(y_cal.squeeze() - predictions.squeeze())
        
        # Find quantile for desired coverage
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = torch.quantile(scores, q_level)
    
    def predict_interval(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return prediction intervals with coverage guarantee"""
        if hasattr(self.base_model, 'eval'):
            self.base_model.eval()
        
        with torch.no_grad():
            if hasattr(self.base_model, 'forward') and callable(getattr(self.base_model, 'forward')):
                predictions = self.base_model(X)
            else:
                predictions = self.base_model.predict(X)[0]  # For ensemble
                
            if isinstance(predictions, tuple):
                predictions = predictions[0]  # Take mean if tuple
        
        lower = predictions.squeeze() - self.quantile
        upper = predictions.squeeze() + self.quantile
        
        return lower, upper

class TemperatureScaling(nn.Module):
    """Temperature scaling for post-hoc calibration"""
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature
    
    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50):
        """Fit temperature on validation set"""
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = nn.CrossEntropyLoss()(scaled_logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction Interval Coverage Probability (PICP)"""
    coverage = ((y_true >= lower) & (y_true <= upper)).mean()
    return coverage

def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean Prediction Interval Width (MPIW)"""
    return (upper - lower).mean()

# REGRESSION EXAMPLE
def regression_uq_example():
    """Complete regression example with multiple UQ methods"""
    print("=== REGRESSION UNCERTAINTY QUANTIFICATION EXAMPLE ===\n")
    
    # Generate synthetic regression data with heteroscedastic noise
    X, y = make_regression(n_samples=1000, n_features=5, noise=10, random_state=42)
    
    # Add heteroscedastic noise (noise increases with X)
    noise_scale = 1 + 0.5 * np.abs(X[:, 0])
    y = y + np.random.normal(0, noise_scale, len(y))
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X = torch.FloatTensor(X_scaled)
    y = torch.FloatTensor(y).unsqueeze(1)
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_cal, y_train, y_cal = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Calibration set size: {len(X_cal)}")
    print(f"Test set size: {len(X_test)}\n")
    
    # Method 1: Deep Ensemble
    print("Training Deep Ensemble...")
    ensemble_models = []
    for i in range(5):
        model = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Train model
        for epoch in range(200):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        
        ensemble_models.append(model)
    
    ensemble = DeepEnsemble(ensemble_models)
    
    # Method 2: MC Dropout
    print("Training MC Dropout Model...")
    mc_model = MCDropoutModel(input_dim=5, hidden_dim=64, output_dim=1, dropout_rate=0.1)
    optimizer = torch.optim.Adam(mc_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = mc_model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    # Method 3: Heteroscedastic Model
    print("Training Heteroscedastic Model...")
    hetero_model = HeteroscedasticModel(input_dim=5, hidden_dim=64)
    optimizer = torch.optim.Adam(hetero_model.parameters(), lr=0.01)
    
    for epoch in range(200):
        optimizer.zero_grad()
        mean_pred, var_pred = hetero_model(X_train)
        loss = hetero_model.nll_loss((mean_pred, var_pred), y_train)
        loss.backward()
        optimizer.step()
    
    # Method 4: Conformal Prediction
    print("Setting up Conformal Prediction...")
    conformal_ensemble = ConformalPredictor(ensemble, alpha=0.1)  # 90% coverage
    conformal_ensemble.calibrate(X_cal, y_cal)
    
    conformal_mc = ConformalPredictor(mc_model, alpha=0.1)
    conformal_mc.calibrate(X_cal, y_cal)
    
    # Evaluate on test set
    print("\n=== EVALUATION RESULTS ===")
    
    # Deep Ensemble predictions
    ensemble_mean, ensemble_var = ensemble.predict(X_test)
    ensemble_std = torch.sqrt(ensemble_var)
    
    # MC Dropout predictions
    mc_mean, mc_var = mc_model.predict_with_uncertainty(X_test, n_samples=100)
    mc_std = torch.sqrt(mc_var)
    
    # Heteroscedastic predictions
    hetero_mean, hetero_var = hetero_model(X_test)
    hetero_std = torch.sqrt(hetero_var)
    
    # Conformal intervals
    conf_lower_ens, conf_upper_ens = conformal_ensemble.predict_interval(X_test)
    conf_lower_mc, conf_upper_mc = conformal_mc.predict_interval(X_test)
    
    # Calculate metrics
    methods = {
        'Deep Ensemble': {
            'predictions': ensemble_mean.squeeze(),
            'std': ensemble_std.squeeze(),
            'lower': ensemble_mean.squeeze() - 1.96 * ensemble_std.squeeze(),
            'upper': ensemble_mean.squeeze() + 1.96 * ensemble_std.squeeze(),
            'conf_lower': conf_lower_ens,
            'conf_upper': conf_upper_ens
        },
        'MC Dropout': {
            'predictions': mc_mean.squeeze(),
            'std': mc_std.squeeze(),
            'lower': mc_mean.squeeze() - 1.96 * mc_std.squeeze(),
            'upper': mc_mean.squeeze() + 1.96 * mc_std.squeeze(),
            'conf_lower': conf_lower_mc,
            'conf_upper': conf_upper_mc
        },
        'Heteroscedastic': {
            'predictions': hetero_mean.squeeze(),
            'std': hetero_std.squeeze(),
            'lower': hetero_mean.squeeze() - 1.96 * hetero_std.squeeze(),
            'upper': hetero_mean.squeeze() + 1.96 * hetero_std.squeeze(),
            'conf_lower': None,
            'conf_upper': None
        }
    }
    
    y_test_np = y_test.squeeze().numpy()
    
    for method_name, results in methods.items():
        pred_np = results['predictions'].detach().numpy()
        std_np = results['std'].detach().numpy()
        lower_np = results['lower'].detach().numpy()
        upper_np = results['upper'].detach().numpy()
        
        # Basic metrics
        mse = np.mean((y_test_np - pred_np) ** 2)
        mae = np.mean(np.abs(y_test_np - pred_np))
        
        # Uncertainty quality metrics
        coverage_gaussian = interval_coverage(y_test_np, lower_np, upper_np)
        width_gaussian = interval_width(lower_np, upper_np)
        
        print(f"\n{method_name}:")
        print(f"  MSE: {mse:.3f}")
        print(f"  MAE: {mae:.3f}")
        print(f"  Gaussian Coverage (95%): {coverage_gaussian:.3f}")
        print(f"  Gaussian Interval Width: {width_gaussian:.3f}")
        print(f"  Average Uncertainty: {std_np.mean():.3f}")
        
        # Conformal intervals
        if results['conf_lower'] is not None:
            conf_lower_np = results['conf_lower'].detach().numpy()
            conf_upper_np = results['conf_upper'].detach().numpy()
            coverage_conformal = interval_coverage(y_test_np, conf_lower_np, conf_upper_np)
            width_conformal = interval_width(conf_lower_np, conf_upper_np)
            print(f"  Conformal Coverage (90%): {coverage_conformal:.3f}")
            print(f"  Conformal Interval Width: {width_conformal:.3f}")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Sort by true values for better visualization
    sort_idx = np.argsort(y_test_np)
    x_plot = np.arange(len(sort_idx))
    y_true_sorted = y_test_np[sort_idx]
    
    for i, (method_name, results) in enumerate(methods.items()):
        plt.subplot(2, 2, i + 1)
        
        pred_sorted = results['predictions'].detach().numpy()[sort_idx]
        lower_sorted = results['lower'].detach().numpy()[sort_idx]
        upper_sorted = results['upper'].detach().numpy()[sort_idx]
        
        plt.scatter(x_plot, y_true_sorted, alpha=0.6, s=20, label='True', color='black')
        plt.plot(x_plot, pred_sorted, color='red', label='Prediction', linewidth=2)
        plt.fill_between(x_plot, lower_sorted, upper_sorted, alpha=0.3, color='red', label='95% CI')
        
        if results['conf_lower'] is not None:
            conf_lower_sorted = results['conf_lower'].detach().numpy()[sort_idx]
            conf_upper_sorted = results['conf_upper'].detach().numpy()[sort_idx]
            plt.fill_between(x_plot, conf_lower_sorted, conf_upper_sorted, alpha=0.2, color='blue', label='90% Conformal')
        
        plt.title(f'{method_name}')
        plt.xlabel('Sample (sorted by true value)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/regression_uq_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return methods

# CLASSIFICATION EXAMPLE
def classification_uq_example():
    """Complete classification example with multiple UQ methods"""
    print("\n\n=== CLASSIFICATION UNCERTAINTY QUANTIFICATION EXAMPLE ===\n")
    
    # Generate synthetic classification data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, 
                             n_informative=8, n_redundant=1, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X = torch.FloatTensor(X_scaled)
    y = torch.LongTensor(y)
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}\n")
    
    # Method 1: Deep Ensemble with Temperature Scaling
    print("Training Deep Ensemble for Classification...")
    ensemble_models = []
    for i in range(5):
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 3)
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Train model
        for epoch in range(150):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        
        ensemble_models.append(model)
    
    # Get ensemble logits for temperature scaling
    ensemble_logits = []
    for model in ensemble_models:
        model.eval()
        with torch.no_grad():
            logits = model(X_val)
            ensemble_logits.append(logits)
    
    ensemble_logits = torch.stack(ensemble_logits).mean(dim=0)
    
    # Temperature scaling
    temperature_scaler = TemperatureScaling()
    temperature_scaler.fit(ensemble_logits, y_val)
    
    # Method 2: MC Dropout for Classification
    print("Training MC Dropout for Classification...")
    mc_classifier = MCDropoutModel(input_dim=10, hidden_dim=64, output_dim=3, dropout_rate=0.1)
    optimizer = torch.optim.Adam(mc_classifier.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(150):
        optimizer.zero_grad()
        outputs = mc_classifier(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluation on test set
    print("\n=== CLASSIFICATION EVALUATION RESULTS ===")
    
    # Deep Ensemble with Temperature Scaling
    test_logits = []
    for model in ensemble_models:
        model.eval()
        with torch.no_grad():
            logits = model(X_test)
            test_logits.append(logits)
    
    test_logits = torch.stack(test_logits)
    ensemble_mean_logits = test_logits.mean(dim=0)
    ensemble_var_logits = test_logits.var(dim=0).mean(dim=-1)  # Average across classes
    
    # Apply temperature scaling
    calibrated_logits = temperature_scaler(ensemble_mean_logits)
    calibrated_probs = F.softmax(calibrated_logits, dim=-1)
    
    # Entropy (predictive uncertainty)
    entropy = -(calibrated_probs * torch.log(calibrated_probs + 1e-8)).sum(dim=-1)
    
    # MC Dropout predictions
    mc_logits, mc_var = mc_classifier.predict_with_uncertainty(X_test, n_samples=100)
    mc_probs = F.softmax(mc_logits, dim=-1)
    mc_entropy = -(mc_probs * torch.log(mc_probs + 1e-8)).sum(dim=-1)
    mc_epistemic = mc_var.mean(dim=-1)
    
    # Calculate metrics
    y_test_np = y_test.numpy()
    
    # Ensemble results
    ensemble_preds = calibrated_probs.argmax(dim=-1)
    ensemble_accuracy = (ensemble_preds == y_test).float().mean()
    ensemble_max_prob = calibrated_probs.max(dim=-1)[0]
    ensemble_ece = expected_calibration_error(y_test_np, ensemble_max_prob.detach().numpy())
    
    # MC Dropout results
    mc_preds = mc_probs.argmax(dim=-1)
    mc_accuracy = (mc_preds == y_test).float().mean()
    mc_max_prob = mc_probs.max(dim=-1)[0]
    mc_ece = expected_calibration_error(y_test_np, mc_max_prob.detach().numpy())
    
    print(f"Deep Ensemble (Temperature Scaled):")
    print(f"  Accuracy: {ensemble_accuracy:.3f}")
    print(f"  ECE: {ensemble_ece:.3f}")
    print(f"  Average Entropy: {entropy.mean():.3f}")
    print(f"  Average Epistemic Uncertainty: {ensemble_var_logits.mean():.3f}")
    print(f"  Temperature: {temperature_scaler.temperature.item():.3f}")
    
    print(f"\nMC Dropout:")
    print(f"  Accuracy: {mc_accuracy:.3f}")
    print(f"  ECE: {mc_ece:.3f}")
    print(f"  Average Entropy: {mc_entropy.mean():.3f}")
    print(f"  Average Epistemic Uncertainty: {mc_epistemic.mean():.3f}")
    
    # Reliability diagrams
    plt.figure(figsize=(12, 5))
    
    # Ensemble reliability diagram
    plt.subplot(1, 2, 1)
    plot_reliability_diagram(y_test_np, ensemble_max_prob.detach().numpy(), title="Deep Ensemble (Temperature Scaled)")
    
    # MC Dropout reliability diagram
    plt.subplot(1, 2, 2)
    plot_reliability_diagram(y_test_np, mc_max_prob.detach().numpy(), title="MC Dropout")
    
    plt.tight_layout()
    plt.savefig('/workspace/classification_uq_reliability.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Uncertainty vs Accuracy Analysis
    plt.figure(figsize=(12, 5))
    
    # Ensemble uncertainty vs accuracy
    plt.subplot(1, 2, 1)
    correct = (ensemble_preds == y_test).detach().numpy()
    plt.scatter(entropy.detach().numpy(), correct, alpha=0.6, s=30)
    plt.xlabel('Predictive Entropy')
    plt.ylabel('Correct Prediction')
    plt.title('Deep Ensemble: Uncertainty vs Correctness')
    plt.grid(True, alpha=0.3)
    
    # MC Dropout uncertainty vs accuracy
    plt.subplot(1, 2, 2)
    correct_mc = (mc_preds == y_test).detach().numpy()
    plt.scatter(mc_entropy.detach().numpy(), correct_mc, alpha=0.6, s=30)
    plt.xlabel('Predictive Entropy')
    plt.ylabel('Correct Prediction')
    plt.title('MC Dropout: Uncertainty vs Correctness')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/classification_uncertainty_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'ensemble': {
            'predictions': ensemble_preds,
            'probabilities': calibrated_probs,
            'entropy': entropy,
            'epistemic_uncertainty': ensemble_var_logits,
            'accuracy': ensemble_accuracy,
            'ece': ensemble_ece
        },
        'mc_dropout': {
            'predictions': mc_preds,
            'probabilities': mc_probs,
            'entropy': mc_entropy,
            'epistemic_uncertainty': mc_epistemic,
            'accuracy': mc_accuracy,
            'ece': mc_ece
        }
    }

def plot_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, title: str = "Reliability Diagram"):
    """Plot reliability diagram for calibration assessment"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences = []
    bin_sizes = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            accuracies.append(accuracy_in_bin)
            confidences.append(avg_confidence_in_bin)
            bin_sizes.append(in_bin.sum())
        else:
            accuracies.append(0)
            confidences.append((bin_lower + bin_upper) / 2)
            bin_sizes.append(0)
    
    # Plot
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    
    # Size-weighted scatter plot
    sizes = np.array(bin_sizes) * 100 / max(bin_sizes) if max(bin_sizes) > 0 else np.ones(len(bin_sizes))
    plt.scatter(confidences, accuracies, s=sizes, alpha=0.7, label='Model', color='red')
    
    # Add ECE to plot
    ece = expected_calibration_error(y_true, y_prob)
    plt.text(0.05, 0.95, f'ECE: {ece:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

def risk_aware_decision_example():
    """Demonstrate risk-aware decision making with uncertainty"""
    print("\n\n=== RISK-AWARE DECISION MAKING EXAMPLE ===\n")
    
    # Generate data for a simple binary classification problem
    X, y = make_classification(n_samples=500, n_features=5, n_classes=2, 
                             n_informative=3, random_state=42)
    
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a simple MC Dropout model
    model = MCDropoutModel(input_dim=5, hidden_dim=32, output_dim=2, dropout_rate=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    # Define cost matrix (cost of action i when true state is j)
    # Action 0: Classify as class 0, Action 1: Classify as class 1, Action 2: Abstain
    cost_matrix = np.array([
        [0, 5],    # Classify as 0: correct=0 cost, wrong=5 cost
        [3, 0],    # Classify as 1: wrong=3 cost, correct=0 cost
        [1, 1]     # Abstain: always 1 cost regardless of true class
    ])
    
    # Make predictions with uncertainty
    logits, uncertainty = model.predict_with_uncertainty(X_test, n_samples=50)
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    
    # Risk-aware decision making
    decisions = []
    expected_costs = []
    uncertainty_threshold = 0.5  # Threshold for abstention
    
    for i in range(len(X_test)):
        prob_i = probs[i].numpy()
        entropy_i = entropy[i].item()
        
        if entropy_i > uncertainty_threshold:
            # High uncertainty -> abstain
            action = 2
            rationale = f"High uncertainty ({entropy_i:.3f})"
        else:
            # Compute expected cost for each action
            expected_cost_0 = cost_matrix[0] @ prob_i
            expected_cost_1 = cost_matrix[1] @ prob_i
            expected_cost_abstain = cost_matrix[2] @ prob_i
            
            costs = [expected_cost_0, expected_cost_1, expected_cost_abstain]
            action = np.argmin(costs)
            rationale = f"Min expected cost: {min(costs):.3f}"
        
        decisions.append(action)
        expected_costs.append(costs if 'costs' in locals() else [0, 0, 1])
    
    # Evaluate decisions
    y_test_np = y_test.numpy()
    correct_decisions = 0
    total_cost = 0
    abstention_rate = 0
    
    for i, (decision, true_class) in enumerate(zip(decisions, y_test_np)):
        if decision == 2:  # Abstention
            abstention_rate += 1
            total_cost += 1  # Cost of abstention
        else:
            if decision == true_class:
                correct_decisions += 1
            # Add actual cost based on decision and true class
            total_cost += cost_matrix[decision, true_class]
    
    abstention_rate /= len(decisions)
    accuracy_on_predictions = correct_decisions / max(1, len(decisions) - sum(1 for d in decisions if d == 2))
    avg_cost = total_cost / len(decisions)
    
    print(f"Risk-Aware Decision Results:")
    print(f"  Abstention Rate: {abstention_rate:.3f}")
    print(f"  Accuracy on Non-Abstained: {accuracy_on_predictions:.3f}")
    print(f"  Average Cost per Decision: {avg_cost:.3f}")
    print(f"  Total Cost: {total_cost:.1f}")
    
    # Compare with naive approach (always predict most likely class)
    naive_decisions = probs.argmax(dim=-1).numpy()
    naive_cost = sum(cost_matrix[decision, true_class] 
                    for decision, true_class in zip(naive_decisions, y_test_np))
    naive_avg_cost = naive_cost / len(decisions)
    
    print(f"\nNaive Approach (no abstention):")
    print(f"  Accuracy: {(naive_decisions == y_test_np).mean():.3f}")
    print(f"  Average Cost per Decision: {naive_avg_cost:.3f}")
    print(f"  Total Cost: {naive_cost:.1f}")
    
    print(f"\nCost Reduction: {((naive_avg_cost - avg_cost) / naive_avg_cost * 100):.1f}%")
    
    # Visualization
    plt.figure(figsize=(12, 4))
    
    # Decision distribution
    plt.subplot(1, 3, 1)
    decision_counts = np.bincount(decisions, minlength=3)
    labels = ['Predict Class 0', 'Predict Class 1', 'Abstain']
    plt.bar(labels, decision_counts)
    plt.title('Decision Distribution')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Uncertainty distribution by decision
    plt.subplot(1, 3, 2)
    for decision_type in range(3):
        mask = np.array(decisions) == decision_type
        if mask.any():
            plt.hist(entropy.numpy()[mask], alpha=0.7, label=labels[decision_type], bins=15)
    plt.xlabel('Predictive Entropy')
    plt.ylabel('Count')
    plt.title('Uncertainty by Decision Type')
    plt.legend()
    
    # Cost comparison
    plt.subplot(1, 3, 3)
    methods = ['Risk-Aware', 'Naive']
    costs = [avg_cost, naive_avg_cost]
    colors = ['green', 'red']
    bars = plt.bar(methods, costs, color=colors, alpha=0.7)
    plt.ylabel('Average Cost per Decision')
    plt.title('Cost Comparison')
    
    # Add cost values on bars
    for bar, cost in zip(bars, costs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{cost:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/workspace/risk_aware_decisions.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Run all examples
    print("Running Uncertainty Quantification Examples...")
    
    # Regression example
    regression_results = regression_uq_example()
    
    # Classification example
    classification_results = classification_uq_example()
    
    # Risk-aware decision making example
    risk_aware_decision_example()
    
    print("\n" + "="*50)
    print("All examples completed successfully!")
    print("Check the generated plots for visual results.")
    print("="*50)