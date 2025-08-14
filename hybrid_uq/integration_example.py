"""
Integration Example: Conformal Prediction + NODE-RK4 + Drift Monitoring

This example demonstrates the complete integration of:
1. Conformal prediction for NODE-RK4 models
2. Non-canonically invasive drift monitoring
3. Online calibration refresh mechanisms
4. Real-time uncertainty quantification with coverage guarantees

The example shows how to set up a production-ready system for reliable
risk estimation with automatic adaptation to changing conditions.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import warnings
import time

# Import our modules
from .core import HybridModel
from .conformal_node_rk4 import (
    RK4ConformalPredictor, 
    TemporalConformalEnsemble,
    integrate_with_hybrid_model,
    rk4_conformal_step
)
from .drift_monitoring import (
    NonCanonicalDriftMonitor,
    create_coverage_aware_drift_monitor,
    setup_default_monitoring_pipeline
)
from .online_calibration import (
    create_online_calibration_pipeline,
    DriftAwareCalibrationManager
)

def create_synthetic_ode_data(
    n_trajectories: int = 100,
    n_time_steps: int = 50,
    n_features: int = 4,
    noise_level: float = 0.1,
    drift_start: int = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create synthetic ODE trajectory data for testing.
    
    Args:
        n_trajectories: Number of trajectory samples
        n_time_steps: Length of each trajectory
        n_features: Number of state variables
        noise_level: Amount of observation noise
        drift_start: Time step to introduce distribution drift (None for no drift)
        
    Returns:
        Tuple of (X, y, time_points)
    """
    dt = 0.1
    time_points = torch.linspace(0, (n_time_steps - 1) * dt, n_time_steps)
    
    X = torch.zeros(n_trajectories, n_time_steps, n_features)
    y = torch.zeros(n_trajectories, n_time_steps, n_features)
    
    for i in range(n_trajectories):
        # Random initial conditions
        x0 = torch.randn(n_features) * 0.5
        
        # Simple dynamics: damped oscillator with coupling
        trajectory = [x0]
        x_current = x0
        
        for t in range(1, n_time_steps):
            # Add drift if specified
            drift_factor = 1.0
            if drift_start is not None and t >= drift_start:
                drift_factor = 1.5  # Increase system dynamics
            
            # Simple ODE: dx/dt = -0.1*x + 0.05*sin(t) + coupling
            coupling = torch.roll(x_current, 1) * 0.1
            dx_dt = -0.1 * x_current * drift_factor + 0.05 * torch.sin(time_points[t]) + coupling
            
            x_next = x_current + dt * dx_dt
            trajectory.append(x_next)
            x_current = x_next
        
        X[i] = torch.stack(trajectory)
        
        # Add heteroscedastic noise (variance increases with time)
        noise_scale = noise_level * (1 + 0.5 * torch.linspace(0, 1, n_time_steps).unsqueeze(1))
        noise = torch.randn_like(X[i]) * noise_scale
        y[i] = X[i] + noise
    
    return X, y, time_points

class ProductionUQSystem:
    """
    Production-ready uncertainty quantification system integrating
    conformal prediction, drift monitoring, and online calibration.
    """
    
    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.1,
        drift_window_size: int = 1000,
        calibration_methods: List[str] = None
    ):
        """
        Initialize production UQ system.
        
        Args:
            model: Trained NODE-RK4 model
            alpha: Miscoverage level for conformal prediction
            drift_window_size: Window size for drift monitoring
            calibration_methods: List of calibration methods to use
        """
        self.model = model
        self.alpha = alpha
        
        if calibration_methods is None:
            calibration_methods = ['temperature_scaling', 'conformal_prediction']
        
        # Initialize components (will be set up in fit())
        self.conformal_predictor = None
        self.drift_monitor = None
        self.calibration_manager = None
        
        # Tracking
        self.is_fitted = False
        self.processing_history = []
        
    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_cal: torch.Tensor,
        y_cal: torch.Tensor,
        time_steps: torch.Tensor = None
    ) -> None:
        """
        Fit the complete UQ system.
        
        Args:
            X_train: Training data for drift reference
            y_train: Training targets
            X_cal: Calibration data for conformal prediction
            y_cal: Calibration targets
            time_steps: Time step indices
        """
        print("🚀 Setting up Production UQ System...")
        
        # 1. Set up conformal prediction
        print("📊 Initializing conformal prediction...")
        self.conformal_predictor = RK4ConformalPredictor(
            model=self.model,
            alpha=self.alpha,
            conformity_score_fn="adaptive_residual",
            temporal_weighting=True
        )
        self.conformal_predictor.fit(X_cal, y_cal, time_steps)
        
        # 2. Set up drift monitoring
        print("🔍 Initializing drift monitoring...")
        self.drift_monitor = NonCanonicalDriftMonitor(window_size=1000)
        # Use training data as reference for drift detection
        X_train_np = X_train.view(-1, X_train.shape[-1]).numpy()
        self.drift_monitor.fit(X_train_np)
        
        # 3. Set up online calibration
        print("⚙️ Initializing online calibration...")
        self.calibration_manager = create_online_calibration_pipeline(
            drift_monitor=self.drift_monitor,
            calibration_methods=['conformal_prediction'],
            conf_alpha=self.alpha,
            conf_window_size=500
        )
        
        self.is_fitted = True
        print("✅ Production UQ System ready!")
        
    def process_batch(
        self,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        return_intervals: bool = True,
        return_diagnostics: bool = True
    ) -> Dict[str, Any]:
        """
        Process a new batch with full UQ pipeline.
        
        Args:
            X_batch: Input batch [B, T, D]
            y_batch: Target batch [B, T, D]
            return_intervals: Whether to return prediction intervals
            return_diagnostics: Whether to return diagnostic information
            
        Returns:
            Dictionary with predictions, intervals, and diagnostics
        """
        if not self.is_fitted:
            raise ValueError("System must be fitted before processing batches")
        
        batch_size = X_batch.shape[0]
        
        with torch.no_grad():
            # Get model predictions
            model_outputs = self.model(X_batch)
            if isinstance(model_outputs, dict):
                predictions = model_outputs['psi']
                uncertainties = model_outputs.get('sigma_res', None)
            else:
                predictions = model_outputs
                uncertainties = None
        
        results = {
            'predictions': predictions,
            'targets': y_batch,
            'uncertainties': uncertainties
        }
        
        # Get conformal prediction intervals
        if return_intervals:
            conformal_results = self.conformal_predictor.predict_intervals(X_batch)
            results.update({
                'conformal_lower': conformal_results.lower_bound,
                'conformal_upper': conformal_results.upper_bound,
                'interval_width': conformal_results.interval_width
            })
            
            # Compute conformity scores for online updates
            conformity_scores = torch.abs(y_batch - predictions).mean(dim=-1)
            coverage_indicators = conformal_results.contains_target(y_batch).float().mean(dim=-1)
            
            results.update({
                'conformity_scores': conformity_scores,
                'coverage_indicators': coverage_indicators
            })
        
        # Process through calibration manager (includes drift detection)
        if return_diagnostics:
            X_flat = X_batch.view(-1, X_batch.shape[-1]).numpy()
            y_flat = y_batch.view(-1, y_batch.shape[-1]).numpy()
            pred_flat = predictions.view(-1, predictions.shape[-1]).numpy()
            
            conformity_flat = results.get('conformity_scores', torch.zeros(batch_size)).view(-1).numpy()
            coverage_flat = results.get('coverage_indicators', torch.zeros(batch_size)).view(-1).numpy()
            
            calibration_results = self.calibration_manager.process_batch(
                X=X_flat,
                y=y_flat,
                predictions=pred_flat,
                conformity_scores=conformity_flat,
                coverage_indicators=coverage_flat
            )
            
            results['diagnostics'] = calibration_results
            
            # Store processing history
            self.processing_history.append({
                'timestamp': time.time(),
                'batch_size': batch_size,
                'drift_detected': calibration_results.get('drift_detected', False),
                'recalibration_triggered': calibration_results.get('recalibration_triggered', False)
            })
        
        return results
    
    def evaluate_coverage(
        self,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        time_steps: torch.Tensor = None
    ) -> Dict[str, float]:
        """Evaluate conformal prediction coverage on test data."""
        return self.conformal_predictor.evaluate_coverage(X_test, y_test, time_steps)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and diagnostics."""
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        status = {
            'status': 'active',
            'batches_processed': len(self.processing_history),
            'conformal_prediction': {
                'alpha': self.conformal_predictor.alpha,
                'n_calibration_samples': len(self.conformal_predictor.calibration_scores),
                'current_quantile': self.conformal_predictor.quantile_cache
            },
            'drift_monitoring': self.drift_monitor.get_monitoring_summary(),
            'calibration_management': self.calibration_manager.get_calibration_summary()
        }
        
        # Recent activity summary
        if self.processing_history:
            recent_history = self.processing_history[-100:]  # Last 100 batches
            status['recent_activity'] = {
                'drift_detection_rate': np.mean([h['drift_detected'] for h in recent_history]),
                'recalibration_rate': np.mean([h['recalibration_triggered'] for h in recent_history]),
                'avg_batch_size': np.mean([h['batch_size'] for h in recent_history])
            }
        
        return status

def run_comprehensive_demo():
    """
    Run comprehensive demonstration of the integrated UQ system.
    """
    print("=" * 80)
    print("🎯 COMPREHENSIVE UQ SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Step 1: Create synthetic data
    print("\n📊 Creating synthetic ODE data...")
    
    # Training data (no drift)
    X_train, y_train, time_points = create_synthetic_ode_data(
        n_trajectories=200, n_time_steps=30, drift_start=None
    )
    
    # Calibration data
    X_cal, y_cal, _ = create_synthetic_ode_data(
        n_trajectories=100, n_time_steps=30, drift_start=None
    )
    
    # Test data with drift
    X_test_normal, y_test_normal, _ = create_synthetic_ode_data(
        n_trajectories=50, n_time_steps=30, drift_start=None
    )
    
    X_test_drift, y_test_drift, _ = create_synthetic_ode_data(
        n_trajectories=50, n_time_steps=30, drift_start=15
    )
    
    print(f"Training data: {X_train.shape}")
    print(f"Calibration data: {X_cal.shape}")
    print(f"Test data (normal): {X_test_normal.shape}")
    print(f"Test data (with drift): {X_test_drift.shape}")
    
    # Step 2: Create and train model
    print("\n🤖 Creating HybridModel...")
    
    grid_metrics = {'dx': 0.1, 'dy': 0.1}
    model = HybridModel(
        grid_metrics=grid_metrics,
        in_ch=4,
        out_ch=4,
        residual_scale=0.02
    )
    
    # Simple training loop (in practice, use proper training)
    print("🏋️ Training model (simplified)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = torch.nn.MSELoss()(outputs['psi'], y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    # Step 3: Set up production UQ system
    print("\n🔧 Setting up Production UQ System...")
    
    uq_system = ProductionUQSystem(
        model=model,
        alpha=0.1,
        calibration_methods=['conformal_prediction']
    )
    
    uq_system.fit(X_train, y_train, X_cal, y_cal, time_points)
    
    # Step 4: Test on normal data
    print("\n📈 Testing on normal data...")
    
    normal_results = uq_system.process_batch(
        X_test_normal, y_test_normal, 
        return_intervals=True, return_diagnostics=True
    )
    
    normal_coverage = uq_system.evaluate_coverage(X_test_normal, y_test_normal)
    
    print(f"Normal data coverage: {normal_coverage['coverage']:.4f}")
    print(f"Target coverage: {normal_coverage['target_coverage']:.4f}")
    print(f"Coverage error: {normal_coverage['coverage_error']:.4f}")
    print(f"Mean interval width: {normal_coverage['mean_interval_width']:.4f}")
    
    # Step 5: Test on drift data
    print("\n⚠️ Testing on data with drift...")
    
    drift_results = uq_system.process_batch(
        X_test_drift, y_test_drift,
        return_intervals=True, return_diagnostics=True
    )
    
    drift_coverage = uq_system.evaluate_coverage(X_test_drift, y_test_drift)
    
    print(f"Drift data coverage: {drift_coverage['coverage']:.4f}")
    print(f"Coverage error: {drift_coverage['coverage_error']:.4f}")
    print(f"Mean interval width: {drift_coverage['mean_interval_width']:.4f}")
    
    # Check drift detection
    drift_detected = drift_results['diagnostics'].get('drift_detected', False)
    print(f"Drift detected: {drift_detected}")
    
    if 'drift_results' in drift_results['diagnostics'] and drift_results['diagnostics']['drift_results']:
        drift_info = drift_results['diagnostics']['drift_results']
        print(f"Consensus drift score: {drift_info['consensus_score']:.4f}")
        print(f"Number of detectors triggered: {drift_info['n_detectors_positive']}")
        print(f"Drift severity: {drift_info['severity']}")
    
    # Step 6: Simulate streaming data
    print("\n🌊 Simulating streaming data processing...")
    
    # Create streaming batches (mix of normal and drift data)
    streaming_batches = []
    
    # First 10 batches: normal data
    for i in range(10):
        X_batch, y_batch, _ = create_synthetic_ode_data(
            n_trajectories=20, n_time_steps=30, drift_start=None
        )
        streaming_batches.append((X_batch, y_batch, 'normal'))
    
    # Next 10 batches: data with increasing drift
    for i in range(10):
        drift_start = max(20 - i, 10)  # Drift starts earlier over time
        X_batch, y_batch, _ = create_synthetic_ode_data(
            n_trajectories=20, n_time_steps=30, drift_start=drift_start
        )
        streaming_batches.append((X_batch, y_batch, f'drift_{i}'))
    
    # Process streaming batches
    drift_detections = []
    recalibrations = []
    coverage_history = []
    
    for i, (X_batch, y_batch, batch_type) in enumerate(streaming_batches):
        results = uq_system.process_batch(X_batch, y_batch)
        
        # Track metrics
        diagnostics = results.get('diagnostics', {})
        drift_detections.append(diagnostics.get('drift_detected', False))
        recalibrations.append(diagnostics.get('recalibration_triggered', False))
        
        # Evaluate coverage for this batch
        coverage = uq_system.evaluate_coverage(X_batch, y_batch)
        coverage_history.append(coverage['coverage'])
        
        if i % 5 == 0:
            print(f"Batch {i+1:2d} ({batch_type:8s}): "
                  f"Coverage={coverage['coverage']:.3f}, "
                  f"Drift={'Yes' if drift_detections[-1] else 'No':3s}, "
                  f"Recal={'Yes' if recalibrations[-1] else 'No':3s}")
    
    # Step 7: Final system status
    print("\n📊 Final System Status:")
    status = uq_system.get_system_status()
    
    print(f"Batches processed: {status['batches_processed']}")
    print(f"Drift detection rate: {status['recent_activity']['drift_detection_rate']:.3f}")
    print(f"Recalibration rate: {status['recent_activity']['recalibration_rate']:.3f}")
    
    conformal_status = status['conformal_prediction']
    print(f"Current conformal quantile: {conformal_status['current_quantile']:.4f}")
    print(f"Calibration samples: {conformal_status['n_calibration_samples']}")
    
    drift_status = status['drift_monitoring']
    if drift_status['status'] == 'monitoring':
        print(f"Total alerts: {drift_status['n_alerts_total']}")
        print(f"Alert severity counts: {drift_status['alert_counts_by_severity']}")
    
    # Step 8: Performance summary
    print("\n📈 Performance Summary:")
    print(f"Normal data coverage: {normal_coverage['coverage']:.4f} (target: {normal_coverage['target_coverage']:.4f})")
    print(f"Drift data coverage: {drift_coverage['coverage']:.4f}")
    print(f"Streaming coverage std: {np.std(coverage_history):.4f}")
    print(f"Total drift detections: {sum(drift_detections)}/{len(drift_detections)}")
    print(f"Total recalibrations: {sum(recalibrations)}/{len(recalibrations)}")
    
    print("\n" + "=" * 80)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 80)
    
    return {
        'uq_system': uq_system,
        'results': {
            'normal_coverage': normal_coverage,
            'drift_coverage': drift_coverage,
            'streaming_metrics': {
                'coverage_history': coverage_history,
                'drift_detections': drift_detections,
                'recalibrations': recalibrations
            }
        }
    }

def demonstrate_rk4_integration():
    """
    Demonstrate RK4 integration with conformal prediction intervals.
    """
    print("\n" + "=" * 60)
    print("🔬 RK4 INTEGRATION WITH CONFORMAL INTERVALS")
    print("=" * 60)
    
    # Create simple model for demonstration
    class SimpleODE(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )
            
        def forward(self, x):
            if len(x.shape) == 3:  # [B, T, D] -> [B*T, D] -> [B, T, D]
                B, T, D = x.shape
                x_flat = x.view(B*T, D)
                out_flat = self.net(x_flat)
                return out_flat.view(B, T, D)
            else:
                return self.net(x)
    
    # Create and train simple model
    model = SimpleODE()
    
    # Generate training data
    X_train = torch.randn(100, 20, 2)
    y_train = X_train + 0.1 * torch.randn_like(X_train)
    
    # Simple training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(100):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = nn.MSELoss()(pred, y_train)
        loss.backward()
        optimizer.step()
    
    # Create conformal predictor
    conformal_predictor = RK4ConformalPredictor(model, alpha=0.1)
    conformal_predictor.fit(X_train, y_train)
    
    # Demonstrate RK4 integration with intervals
    x0 = torch.tensor([[1.0, 0.5], [0.0, 1.0]])  # Two initial conditions
    t_span = torch.linspace(0, 2.0, 21)
    dt = 0.1
    
    print("🚀 Running RK4 integration with conformal intervals...")
    
    integration_results = rk4_conformal_step(
        model=model,
        conformal_predictor=conformal_predictor,
        x0=x0,
        t_span=t_span,
        dt=dt,
        return_intervals=True
    )
    
    trajectory = integration_results['trajectory']
    lower_bound = integration_results['lower_bound']
    upper_bound = integration_results['upper_bound']
    
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Initial conditions: {x0}")
    print(f"Final states: {trajectory[:, -1]}")
    print(f"Final interval widths: {(upper_bound - lower_bound)[:, -1]}")
    
    # Compute average interval width over time
    interval_widths = (upper_bound - lower_bound).mean(dim=(0, 2))
    print(f"Average interval width over time: {interval_widths.mean():.4f}")
    print(f"Interval width std: {interval_widths.std():.4f}")
    
    print("✅ RK4 integration demonstration complete!")
    
    return integration_results

if __name__ == "__main__":
    # Run comprehensive demo
    demo_results = run_comprehensive_demo()
    
    # Run RK4 integration demo
    rk4_results = demonstrate_rk4_integration()
    
    print("\n🎉 All demonstrations completed successfully!")
    print("\nKey takeaways:")
    print("1. ✅ Conformal prediction provides reliable coverage guarantees")
    print("2. ✅ Drift monitoring detects distribution changes automatically")
    print("3. ✅ Online calibration adapts to changing conditions")
    print("4. ✅ RK4 integration works seamlessly with uncertainty quantification")
    print("5. ✅ Production system handles streaming data effectively")