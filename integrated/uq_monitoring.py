"""
Uncertainty Quantification Monitoring System
Comprehensive monitoring for UQ models in production including drift detection,
calibration monitoring, and automated alerting systems.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import warnings
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass, asdict
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Data class for drift alerts"""

    timestamp: str
    alert_type: str
    metric_name: str
    current_value: float
    threshold: float
    severity: str
    message: str


@dataclass
class MonitoringMetrics:
    """Data class for monitoring metrics"""

    timestamp: str
    input_drift_score: float
    prediction_drift_score: float
    calibration_error: float
    coverage_rate: float
    avg_uncertainty: float
    prediction_count: int
    abstention_rate: float


class PopulationStabilityIndex:
    """Population Stability Index (PSI) for drift detection"""

    def __init__(self, n_bins: int = 10, min_bin_size: int = 5):
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size
        self.reference_bins = None
        self.reference_percentages = None

    def fit(self, reference_data: np.ndarray):
        """Fit PSI on reference data"""
        # Create bins based on quantiles to ensure roughly equal bin sizes
        self.reference_bins = np.percentile(
            reference_data, np.linspace(0, 100, self.n_bins + 1)
        )
        self.reference_bins[0] = -np.inf  # Handle edge cases
        self.reference_bins[-1] = np.inf

        # Calculate reference percentages
        ref_hist, _ = np.histogram(reference_data, bins=self.reference_bins)
        self.reference_percentages = (ref_hist + 1e-8) / (
            ref_hist.sum() + self.n_bins * 1e-8
        )

    def calculate_psi(self, current_data: np.ndarray) -> float:
        """Calculate PSI between reference and current data"""
        if self.reference_bins is None:
            raise ValueError("Must call fit() first")

        # Calculate current percentages
        cur_hist, _ = np.histogram(current_data, bins=self.reference_bins)
        current_percentages = (cur_hist + 1e-8) / (cur_hist.sum() + self.n_bins * 1e-8)

        # Calculate PSI
        psi = np.sum(
            (current_percentages - self.reference_percentages)
            * np.log(current_percentages / self.reference_percentages)
        )

        return psi


class KLDivergenceMonitor:
    """KL Divergence monitoring for distribution shift"""

    def __init__(self, n_bins: int = 50):
        self.n_bins = n_bins
        self.reference_dist = None
        self.bin_edges = None

    def fit(self, reference_data: np.ndarray):
        """Fit reference distribution"""
        self.reference_dist, self.bin_edges = np.histogram(
            reference_data, bins=self.n_bins, density=True
        )
        # Add small constant to avoid log(0)
        self.reference_dist = self.reference_dist + 1e-10
        self.reference_dist = self.reference_dist / self.reference_dist.sum()

    def calculate_kl_divergence(self, current_data: np.ndarray) -> float:
        """Calculate KL divergence from reference to current distribution"""
        if self.reference_dist is None:
            raise ValueError("Must call fit() first")

        current_dist, _ = np.histogram(current_data, bins=self.bin_edges, density=True)
        current_dist = current_dist + 1e-10
        current_dist = current_dist / current_dist.sum()

        # KL(P||Q) = sum(P * log(P/Q))
        kl_div = np.sum(
            self.reference_dist * np.log(self.reference_dist / current_dist)
        )
        return kl_div


class CalibrationMonitor:
    """Monitor calibration quality over time"""

    def __init__(self, n_bins: int = 10, window_size: int = 1000):
        self.n_bins = n_bins
        self.window_size = window_size
        self.predictions_buffer = deque(maxlen=window_size)
        self.labels_buffer = deque(maxlen=window_size)
        self.timestamps_buffer = deque(maxlen=window_size)

    def add_predictions(
        self, y_true: np.ndarray, y_prob: np.ndarray, timestamp: str = None
    ):
        """Add new predictions to monitoring buffer"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        for true_label, prob in zip(y_true, y_prob):
            self.predictions_buffer.append(prob)
            self.labels_buffer.append(true_label)
            self.timestamps_buffer.append(timestamp)

    def calculate_ece(self) -> float:
        """Calculate Expected Calibration Error on current buffer"""
        if len(self.predictions_buffer) < 10:
            return 0.0

        y_true = np.array(list(self.labels_buffer))
        y_prob = np.array(list(self.predictions_buffer))

        return expected_calibration_error(y_true, y_prob, self.n_bins)

    def calculate_brier_score(self) -> float:
        """Calculate Brier score on current buffer"""
        if len(self.predictions_buffer) < 10:
            return 0.0

        y_true = np.array(list(self.labels_buffer))
        y_prob = np.array(list(self.predictions_buffer))

        return np.mean((y_prob - y_true) ** 2)

    def get_calibration_trend(self, window_hours: int = 24) -> List[float]:
        """Get calibration trend over specified time window"""
        if len(self.timestamps_buffer) < 10:
            return []

        # Convert timestamps to datetime objects
        timestamps = [datetime.fromisoformat(ts) for ts in self.timestamps_buffer]
        current_time = max(timestamps)
        cutoff_time = current_time - timedelta(hours=window_hours)

        # Filter data within time window
        recent_indices = [i for i, ts in enumerate(timestamps) if ts >= cutoff_time]

        if len(recent_indices) < 10:
            return []

        # Calculate ECE in sliding windows
        window_size = max(100, len(recent_indices) // 10)
        ece_values = []

        for i in range(0, len(recent_indices) - window_size, window_size // 2):
            indices = recent_indices[i : i + window_size]
            y_true = np.array([self.labels_buffer[idx] for idx in indices])
            y_prob = np.array([self.predictions_buffer[idx] for idx in indices])
            ece = expected_calibration_error(y_true, y_prob, self.n_bins)
            ece_values.append(ece)

        return ece_values


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Calculate Expected Calibration Error"""
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


class UQProductionMonitor:
    """Comprehensive production monitoring system for UQ models"""

    def __init__(
        self,
        psi_threshold: float = 0.1,
        kl_threshold: float = 0.5,
        calibration_threshold: float = 0.05,
        coverage_threshold: float = 0.05,
        uncertainty_threshold: float = 2.0,
        window_size: int = 1000,
        alert_cooldown_hours: int = 1,
    ):

        # Thresholds
        self.psi_threshold = psi_threshold
        self.kl_threshold = kl_threshold
        self.calibration_threshold = calibration_threshold
        self.coverage_threshold = coverage_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.window_size = window_size
        self.alert_cooldown_hours = alert_cooldown_hours

        # Monitoring components
        self.psi_monitor = PopulationStabilityIndex()
        self.kl_monitor = KLDivergenceMonitor()
        self.calibration_monitor = CalibrationMonitor(window_size=window_size)

        # Reference data storage
        self.reference_features = None
        self.reference_predictions = None
        self.reference_uncertainties = None
        self.reference_coverage = None

        # Alert management
        self.alerts_history = []
        self.last_alert_times = {}

        # Metrics history
        self.metrics_history = []

        # Data buffers
        self.feature_buffer = deque(maxlen=window_size)
        self.prediction_buffer = deque(maxlen=window_size)
        self.uncertainty_buffer = deque(maxlen=window_size)
        self.coverage_buffer = deque(maxlen=window_size)
        self.timestamp_buffer = deque(maxlen=window_size)

    def set_reference_data(
        self,
        features: np.ndarray,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
        prediction_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """Set reference data for drift detection"""

        # Store reference data
        self.reference_features = features
        self.reference_predictions = predictions
        self.reference_uncertainties = uncertainties

        # Fit drift monitors on reference features (use PCA if high-dimensional)
        if features.shape[1] > 10:
            try:
                from sklearn.decomposition import PCA
            except ImportError:
                PCA = None
                return {'drift_detected': False, 'reason': 'PCA not available'}

            pca = PCA(n_components=min(10, features.shape[1]))
            features_reduced = pca.fit_transform(features)
            self.pca = pca
        else:
            features_reduced = features
            self.pca = None

        # Fit monitors on each feature
        self.feature_monitors = {}
        for i in range(features_reduced.shape[1]):
            psi_monitor = PopulationStabilityIndex()
            kl_monitor = KLDivergenceMonitor()

            psi_monitor.fit(features_reduced[:, i])
            kl_monitor.fit(features_reduced[:, i])

            self.feature_monitors[i] = {"psi": psi_monitor, "kl": kl_monitor}

        # Fit prediction drift monitors
        self.psi_monitor.fit(predictions.flatten())
        self.kl_monitor.fit(predictions.flatten())

        # Calculate reference coverage if intervals provided
        if true_labels is not None and prediction_intervals is not None:
            lower, upper = prediction_intervals
            self.reference_coverage = np.mean(
                (true_labels >= lower) & (true_labels <= upper)
            )

        logger.info(
            f"Reference data set with {len(features)} samples, {features.shape[1]} features"
        )

    def add_batch(
        self,
        features: np.ndarray,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
        prediction_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        timestamp: Optional[str] = None,
    ):
        """Add new batch of predictions for monitoring"""

        if timestamp is None:
            timestamp = datetime.now().isoformat()

        # Add to buffers
        for i in range(len(features)):
            self.feature_buffer.append(features[i])
            self.prediction_buffer.append(predictions[i])
            self.uncertainty_buffer.append(uncertainties[i])
            self.timestamp_buffer.append(timestamp)

            # Calculate coverage if intervals provided
            if true_labels is not None and prediction_intervals is not None:
                lower, upper = prediction_intervals
                in_interval = (true_labels[i] >= lower[i]) & (
                    true_labels[i] <= upper[i]
                )
                self.coverage_buffer.append(float(in_interval))

        # Add to calibration monitor if labels available
        if true_labels is not None and predictions.ndim > 1:
            # For classification, use max probability
            max_probs = np.max(predictions, axis=1)
            pred_classes = np.argmax(predictions, axis=1)
            correct = (pred_classes == true_labels).astype(float)
            self.calibration_monitor.add_predictions(correct, max_probs, timestamp)

        # Run monitoring checks
        self._check_drift_and_alert(timestamp)

    def _check_drift_and_alert(self, timestamp: str):
        """Check for drift and generate alerts if necessary"""

        if len(self.feature_buffer) < 100:  # Need minimum samples
            return

        current_time = datetime.fromisoformat(timestamp)
        alerts = []

        # Check feature drift
        current_features = np.array(list(self.feature_buffer))
        if self.pca is not None:
            current_features = self.pca.transform(current_features)

        max_psi = 0
        max_kl = 0

        for i, monitors in self.feature_monitors.items():
            if i < current_features.shape[1]:
                psi_score = monitors["psi"].calculate_psi(current_features[:, i])
                kl_score = monitors["kl"].calculate_kl_divergence(
                    current_features[:, i]
                )

                max_psi = max(max_psi, psi_score)
                max_kl = max(max_kl, kl_score)

        # Check prediction drift
        current_predictions = np.array(list(self.prediction_buffer))
        pred_psi = self.psi_monitor.calculate_psi(current_predictions.flatten())
        pred_kl = self.kl_monitor.calculate_kl_divergence(current_predictions.flatten())

        # Check calibration drift
        current_ece = self.calibration_monitor.calculate_ece()

        # Check coverage drift
        current_coverage = (
            np.mean(list(self.coverage_buffer)) if self.coverage_buffer else 0
        )
        coverage_drift = (
            abs(current_coverage - self.reference_coverage)
            if self.reference_coverage
            else 0
        )

        # Check uncertainty drift
        current_uncertainty = np.mean(list(self.uncertainty_buffer))
        ref_uncertainty = (
            np.mean(self.reference_uncertainties)
            if self.reference_uncertainties is not None
            else 0
        )
        uncertainty_drift = abs(current_uncertainty - ref_uncertainty) / (
            ref_uncertainty + 1e-8
        )

        # Generate alerts
        if max_psi > self.psi_threshold:
            alerts.append(
                DriftAlert(
                    timestamp=timestamp,
                    alert_type="feature_drift",
                    metric_name="PSI",
                    current_value=max_psi,
                    threshold=self.psi_threshold,
                    severity="high" if max_psi > 2 * self.psi_threshold else "medium",
                    message=f"Feature distribution drift detected (PSI: {max_psi:.4f})",
                )
            )

        if pred_psi > self.psi_threshold:
            alerts.append(
                DriftAlert(
                    timestamp=timestamp,
                    alert_type="prediction_drift",
                    metric_name="PSI",
                    current_value=pred_psi,
                    threshold=self.psi_threshold,
                    severity="high" if pred_psi > 2 * self.psi_threshold else "medium",
                    message=f"Prediction distribution drift detected (PSI: {pred_psi:.4f})",
                )
            )

        if current_ece > self.calibration_threshold:
            alerts.append(
                DriftAlert(
                    timestamp=timestamp,
                    alert_type="calibration_drift",
                    metric_name="ECE",
                    current_value=current_ece,
                    threshold=self.calibration_threshold,
                    severity=(
                        "high"
                        if current_ece > 2 * self.calibration_threshold
                        else "medium"
                    ),
                    message=f"Model calibration degraded (ECE: {current_ece:.4f})",
                )
            )

        if coverage_drift > self.coverage_threshold:
            alerts.append(
                DriftAlert(
                    timestamp=timestamp,
                    alert_type="coverage_drift",
                    metric_name="Coverage",
                    current_value=float(current_coverage),
                    threshold=float(self.reference_coverage) if self.reference_coverage is not None else 0.0,
                    severity="medium",
                    message=f"Prediction interval coverage drift (Current: {current_coverage:.3f}, Reference: {self.reference_coverage:.3f})",
                )
            )

        if uncertainty_drift > self.uncertainty_threshold:
            alerts.append(
                DriftAlert(
                    timestamp=timestamp,
                    alert_type="uncertainty_drift",
                    metric_name="Uncertainty",
                    current_value=current_uncertainty,
                    threshold=ref_uncertainty,
                    severity="medium",
                    message=f"Uncertainty distribution shift (Current: {current_uncertainty:.3f}, Reference: {ref_uncertainty:.3f})",
                )
            )

        # Filter alerts by cooldown period
        filtered_alerts = []
        for alert in alerts:
            alert_key = f"{alert.alert_type}_{alert.metric_name}"
            last_alert_time = self.last_alert_times.get(alert_key)

            if last_alert_time is None or current_time - datetime.fromisoformat(
                last_alert_time
            ) > timedelta(hours=self.alert_cooldown_hours):
                filtered_alerts.append(alert)
                self.last_alert_times[alert_key] = timestamp

        # Log alerts
        for alert in filtered_alerts:
            logger.warning(f"ALERT: {alert.message}")
            self.alerts_history.append(alert)

        # Store metrics
        metrics = MonitoringMetrics(
            timestamp=timestamp,
            input_drift_score=max_psi,
            prediction_drift_score=pred_psi,
            calibration_error=current_ece,
            coverage_rate=float(current_coverage) if current_coverage is not None else 0.0,
            avg_uncertainty=float(avg_uncertainty),
            prediction_count=len(self.prediction_buffer),
            abstention_rate=0.0,  # Would need to track abstentions separately
        )
        self.metrics_history.append(metrics)

    def get_monitoring_dashboard_data(self, hours_back: int = 24) -> Dict:
        """Get data for monitoring dashboard"""

        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=hours_back)

        # Filter recent metrics
        recent_metrics = [
            m
            for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) >= cutoff_time
        ]

        # Filter recent alerts
        recent_alerts = [
            a
            for a in self.alerts_history
            if datetime.fromisoformat(a.timestamp) >= cutoff_time
        ]

        # Calculate summary statistics
        if recent_metrics:
            metrics_df = pd.DataFrame([asdict(m) for m in recent_metrics])
            summary_stats = {
                "avg_input_drift": float(metrics_df["input_drift_score"].mean()),
                "max_input_drift": float(metrics_df["input_drift_score"].max()),
                "avg_calibration_error": float(metrics_df["calibration_error"].mean()),
                "max_calibration_error": float(metrics_df["calibration_error"].max()),
                "avg_coverage_rate": float(metrics_df["coverage_rate"].mean()),
                "total_predictions": int(metrics_df["prediction_count"].sum()),
                "alert_count": len(recent_alerts),
            }
        else:
            summary_stats = {
                "avg_input_drift": 0.0,
                "max_input_drift": 0.0,
                "avg_calibration_error": 0.0,
                "max_calibration_error": 0.0,
                "avg_coverage_rate": 0.0,
                "total_predictions": 0,
                "alert_count": 0,
            }

        return {
            "summary": summary_stats,
            "metrics_history": [asdict(m) for m in recent_metrics],
            "alerts_history": [asdict(a) for a in recent_alerts],
            "calibration_trend": self.calibration_monitor.get_calibration_trend(
                hours_back
            ),
        }

    def plot_monitoring_dashboard(self, hours_back: int = 24, save_path: str = None):
        """Create comprehensive monitoring dashboard"""

        dashboard_data = self.get_monitoring_dashboard_data(hours_back)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"UQ Model Monitoring Dashboard - Last {hours_back} Hours", fontsize=16
        )

        # Convert metrics to DataFrame for easier plotting
        if dashboard_data["metrics_history"]:
            metrics_df = pd.DataFrame(dashboard_data["metrics_history"])
            metrics_df["timestamp"] = pd.to_datetime(metrics_df["timestamp"])
        else:
            metrics_df = pd.DataFrame()

        # Plot 1: Input Drift Score
        ax1 = axes[0, 0]
        if not metrics_df.empty:
            ax1.plot(
                metrics_df["timestamp"],
                metrics_df["input_drift_score"],
                "b-",
                linewidth=2,
            )
            ax1.axhline(
                y=self.psi_threshold,
                color="r",
                linestyle="--",
                alpha=0.7,
                label="Threshold",
            )
        ax1.set_title("Input Drift Score (PSI)")
        ax1.set_ylabel("PSI Score")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Calibration Error
        ax2 = axes[0, 1]
        if not metrics_df.empty:
            ax2.plot(
                metrics_df["timestamp"],
                metrics_df["calibration_error"],
                "g-",
                linewidth=2,
            )
            ax2.axhline(
                y=self.calibration_threshold,
                color="r",
                linestyle="--",
                alpha=0.7,
                label="Threshold",
            )
        ax2.set_title("Calibration Error (ECE)")
        ax2.set_ylabel("ECE")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Coverage Rate
        ax3 = axes[0, 2]
        if not metrics_df.empty:
            ax3.plot(
                metrics_df["timestamp"],
                metrics_df["coverage_rate"],
                "purple",
                linewidth=2,
            )
            if self.reference_coverage:
                ax3.axhline(
                    y=self.reference_coverage,
                    color="orange",
                    linestyle="--",
                    alpha=0.7,
                    label="Reference",
                )
        ax3.set_title("Prediction Interval Coverage")
        ax3.set_ylabel("Coverage Rate")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Average Uncertainty
        ax4 = axes[1, 0]
        if not metrics_df.empty:
            ax4.plot(
                metrics_df["timestamp"],
                metrics_df["avg_uncertainty"],
                "orange",
                linewidth=2,
            )
        ax4.set_title("Average Uncertainty")
        ax4.set_ylabel("Uncertainty")
        ax4.grid(True, alpha=0.3)

        # Plot 5: Prediction Volume
        ax5 = axes[1, 1]
        if not metrics_df.empty:
            # Aggregate by hour for better visualization
            hourly_counts = (
                metrics_df.set_index("timestamp")
                .resample("1H")["prediction_count"]
                .sum()
            )
            ax5.bar(
                hourly_counts.index, hourly_counts.values, alpha=0.7, color="skyblue"
            )
        ax5.set_title("Prediction Volume (Hourly)")
        ax5.set_ylabel("Predictions/Hour")
        ax5.grid(True, alpha=0.3)

        # Plot 6: Alert Summary
        ax6 = axes[1, 2]
        if dashboard_data["alerts_history"]:
            alert_types = [a["alert_type"] for a in dashboard_data["alerts_history"]]
            alert_counts = pd.Series(alert_types).value_counts()
            ax6.bar(alert_counts.index, alert_counts.values, color="red", alpha=0.7)
            ax6.set_xticklabels(alert_counts.index, rotation=45, ha="right")
        ax6.set_title("Alert Summary")
        ax6.set_ylabel("Alert Count")
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    def export_monitoring_report(
        self, hours_back: int = 24, output_path: str = None
    ) -> Dict:
        """Export comprehensive monitoring report"""

        dashboard_data = self.get_monitoring_dashboard_data(hours_back)

        report = {
            "report_timestamp": datetime.now().isoformat(),
            "monitoring_period_hours": hours_back,
            "summary_statistics": dashboard_data["summary"],
            "thresholds": {
                "psi_threshold": self.psi_threshold,
                "calibration_threshold": self.calibration_threshold,
                "coverage_threshold": self.coverage_threshold,
                "uncertainty_threshold": self.uncertainty_threshold,
            },
            "recent_alerts": dashboard_data["alerts_history"],
            "metrics_trend": (
                dashboard_data["metrics_history"][-10:]
                if dashboard_data["metrics_history"]
                else []
            ),
            "recommendations": self._generate_recommendations(dashboard_data),
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

        return report

    def _generate_recommendations(self, dashboard_data: Dict) -> List[str]:
        """Generate actionable recommendations based on monitoring data"""

        recommendations = []
        summary = dashboard_data["summary"]

        # High drift recommendations
        if summary["max_input_drift"] > 2 * self.psi_threshold:
            recommendations.append(
                "CRITICAL: Severe input distribution drift detected. Consider retraining the model with recent data."
            )
        elif summary["avg_input_drift"] > self.psi_threshold:
            recommendations.append(
                "WARNING: Input drift detected. Monitor closely and consider data preprocessing adjustments."
            )

        # Calibration recommendations
        if summary["max_calibration_error"] > 2 * self.calibration_threshold:
            recommendations.append(
                "CRITICAL: Model calibration severely degraded. Immediate recalibration required."
            )
        elif summary["avg_calibration_error"] > self.calibration_threshold:
            recommendations.append(
                "WARNING: Calibration quality declining. Schedule temperature scaling recalibration."
            )

        # Coverage recommendations
        if (
            abs(summary["avg_coverage_rate"] - (self.reference_coverage or 0.9))
            > self.coverage_threshold
        ):
            recommendations.append(
                "WARNING: Prediction interval coverage deviating from target. Review conformal prediction quantiles."
            )

        # Alert frequency recommendations
        if summary["alert_count"] > 10:
            recommendations.append(
                "INFO: High alert frequency. Consider adjusting thresholds or alert cooldown periods."
            )
        elif summary["alert_count"] == 0 and summary["total_predictions"] > 1000:
            recommendations.append(
                "INFO: No recent alerts with high prediction volume. Monitoring system functioning normally."
            )

        # Volume recommendations
        if summary["total_predictions"] < 100:
            recommendations.append(
                "INFO: Low prediction volume. Ensure sufficient data for reliable monitoring statistics."
            )

        return recommendations


# Example usage and testing
def simulate_monitoring_scenario():
    """Simulate a monitoring scenario with drift"""

    print("=== UQ PRODUCTION MONITORING SIMULATION ===\n")

    # Create synthetic reference data
    np.random.seed(42)
    n_ref = 1000
    n_features = 5

    # Reference data (normal distribution)
    ref_features = np.random.normal(0, 1, (n_ref, n_features))
    ref_predictions = np.random.beta(2, 2, (n_ref, 1))  # Well-calibrated predictions
    ref_uncertainties = np.random.gamma(2, 0.1, n_ref)
    ref_labels = (ref_predictions.squeeze() > 0.5).astype(int)
    ref_intervals = (ref_predictions.squeeze() - 0.2, ref_predictions.squeeze() + 0.2)

    # Initialize monitor
    monitor = UQProductionMonitor(
        psi_threshold=0.1,
        calibration_threshold=0.05,
        coverage_threshold=0.05,
        window_size=500,
    )

    # Set reference data
    monitor.set_reference_data(
        features=ref_features,
        predictions=ref_predictions,
        uncertainties=ref_uncertainties,
        true_labels=ref_labels,
        prediction_intervals=ref_intervals,
    )

    print("Reference data set. Starting monitoring simulation...\n")

    # Simulate different scenarios over time
    scenarios = [
        {"name": "Normal Operation", "n_batches": 5, "drift_factor": 0.0},
        {"name": "Mild Drift", "n_batches": 3, "drift_factor": 0.5},
        {"name": "Severe Drift", "n_batches": 3, "drift_factor": 1.5},
        {"name": "Calibration Degradation", "n_batches": 4, "drift_factor": 0.2},
    ]

    for scenario in scenarios:
        print(f"--- {scenario['name']} ---")

        for batch in range(scenario["n_batches"]):
            # Generate batch data with drift
            batch_size = 100
            drift = scenario["drift_factor"]

            # Features with potential drift
            if scenario["name"] == "Severe Drift":
                batch_features = np.random.normal(
                    drift, 1 + drift * 0.5, (batch_size, n_features)
                )
            else:
                batch_features = np.random.normal(
                    drift * 0.3, 1, (batch_size, n_features)
                )

            # Predictions with potential calibration issues
            if scenario["name"] == "Calibration Degradation":
                # Overconfident predictions
                batch_predictions = np.random.beta(1, 1, (batch_size, 1))
                batch_predictions = np.clip(batch_predictions * 1.2 - 0.1, 0, 1)
            else:
                batch_predictions = np.random.beta(2, 2, (batch_size, 1))

            # Uncertainties
            batch_uncertainties = np.random.gamma(2 + drift * 0.5, 0.1, batch_size)

            # Labels (simulate some miscalibration)
            if scenario["name"] == "Calibration Degradation":
                batch_labels = (batch_predictions.squeeze() > 0.7).astype(
                    int
                )  # Miscalibrated
            else:
                batch_labels = (batch_predictions.squeeze() > 0.5).astype(int)

            # Intervals
            interval_width = 0.2 + drift * 0.1
            batch_intervals = (
                batch_predictions.squeeze() - interval_width,
                batch_predictions.squeeze() + interval_width,
            )

            # Add batch to monitor
            timestamp = (datetime.now() + timedelta(hours=batch * 2)).isoformat()
            monitor.add_batch(
                features=batch_features,
                predictions=batch_predictions,
                uncertainties=batch_uncertainties,
                true_labels=batch_labels,
                prediction_intervals=batch_intervals,
                timestamp=timestamp,
            )

        print(f"Completed {scenario['name']} scenario\n")

    # Generate monitoring report
    print("=== MONITORING REPORT ===")
    report = monitor.export_monitoring_report(hours_back=48)

    print(
        f"Total Predictions Monitored: {report['summary_statistics']['total_predictions']}"
    )
    print(f"Alerts Generated: {report['summary_statistics']['alert_count']}")
    print(
        f"Average Input Drift Score: {report['summary_statistics']['avg_input_drift']:.4f}"
    )
    print(
        f"Average Calibration Error: {report['summary_statistics']['avg_calibration_error']:.4f}"
    )
    print(
        f"Average Coverage Rate: {report['summary_statistics']['avg_coverage_rate']:.3f}"
    )

    print("\n=== RECOMMENDATIONS ===")
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"{i}. {rec}")

    print("\n=== RECENT ALERTS ===")
    for alert in report["recent_alerts"][-5:]:  # Show last 5 alerts
        print(f"[{alert['severity'].upper()}] {alert['timestamp']}: {alert['message']}")

    # Create dashboard
    print("\nGenerating monitoring dashboard...")
    monitor.plot_monitoring_dashboard(
        hours_back=48, save_path="/workspace/monitoring_dashboard.png"
    )

    return monitor, report


if __name__ == "__main__":
    # Run monitoring simulation
    monitor, report = simulate_monitoring_scenario()

    print("\n" + "=" * 60)
    print("Monitoring simulation completed successfully!")
    print("Check '/workspace/monitoring_dashboard.png' for visual dashboard")
    print("=" * 60)
