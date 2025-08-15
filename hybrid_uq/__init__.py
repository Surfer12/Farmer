# Hybrid Uncertainty Quantification Framework
"""
Hybrid UQ Framework with Conformal Prediction and Drift Monitoring

This framework provides comprehensive uncertainty quantification capabilities
for NODE-RK4 models with real-time drift monitoring and online calibration.

Key Components:
- HybridModel: Physics-informed neural network with uncertainty
- RK4ConformalPredictor: Conformal prediction for NODE-RK4
- NonCanonicalDriftMonitor: Advanced drift detection
- Online calibration mechanisms
- Production-ready integration examples
"""

# Core hybrid model
from .core import (
    HybridModel,
    PhysicsInterpolator,
    ResidualNet,
    AlphaScheduler,
    SplitConformal,
    loss_objective
)

# Conformal prediction for NODE-RK4
from .conformal_node_rk4 import (
    RK4ConformalPredictor,
    TemporalConformalEnsemble,
    ConformalPrediction,
    rk4_conformal_step,
    integrate_with_hybrid_model
)

# Drift monitoring system
from .drift_monitoring import (
    NonCanonicalDriftMonitor,
    DriftAlert,
    KolmogorovSmirnovDriftDetector,
    MaximumMeanDiscrepancyDetector,
    AdversarialDriftDetector,
    EnergyDistanceDetector,
    create_coverage_aware_drift_monitor,
    setup_default_monitoring_pipeline
)

# Online calibration
from .online_calibration import (
    OnlineTemperatureScaling,
    OnlineIsotonicRegression,
    OnlineConformalCalibration,
    DriftAwareCalibrationManager,
    CalibrationMetrics,
    create_online_calibration_pipeline
)

# Integration example
from .integration_example import (
    ProductionUQSystem,
    create_synthetic_ode_data,
    run_comprehensive_demo,
    demonstrate_rk4_integration
)

# Version information
__version__ = "1.0.0"

# Main exports for easy access
__all__ = [
    # Core components
    "HybridModel",
    "PhysicsInterpolator",
    "ResidualNet",
    "AlphaScheduler",
    "SplitConformal",
    "loss_objective",
    
    # Conformal prediction
    "RK4ConformalPredictor",
    "TemporalConformalEnsemble",
    "ConformalPrediction",
    "rk4_conformal_step",
    "integrate_with_hybrid_model",
    
    # Drift monitoring
    "NonCanonicalDriftMonitor",
    "DriftAlert",
    "create_coverage_aware_drift_monitor",
    "setup_default_monitoring_pipeline",
    
    # Online calibration
    "OnlineTemperatureScaling",
    "OnlineIsotonicRegression", 
    "OnlineConformalCalibration",
    "DriftAwareCalibrationManager",
    "CalibrationMetrics",
    "create_online_calibration_pipeline",
    
    # Production system
    "ProductionUQSystem",
    "create_synthetic_ode_data",
    "run_comprehensive_demo",
    "demonstrate_rk4_integration"
]

# Quick setup function for common use cases
def create_production_uq_system(
    model,
    X_train,
    y_train,
    X_cal,
    y_cal,
    alpha=0.1,
    **kwargs
):
    """
    Quick setup function for production UQ system.
    
    Args:
        model: Trained NODE-RK4 model
        X_train: Training data for drift reference
        y_train: Training targets
        X_cal: Calibration data
        y_cal: Calibration targets
        alpha: Miscoverage level
        **kwargs: Additional arguments
        
    Returns:
        Fitted ProductionUQSystem
    """
    uq_system = ProductionUQSystem(model=model, alpha=alpha, **kwargs)
    uq_system.fit(X_train, y_train, X_cal, y_cal)
    return uq_system